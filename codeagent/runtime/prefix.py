"""Prefix assembly — system prompt and tool registry for the agent."""

import json
import hashlib
import textwrap

from .. import tools as toolkit
from ..workspace import WorkspaceContext, now
from .constants import CHECKPOINT_NONE_STATUS
from .session_store import PromptPrefix


def build_tools(pico):
    """Build the tool registry from toolkit."""
    return toolkit.build_tool_registry(pico)


def tool_signature(pico):
    """SHA-256 hash of sorted tool definitions for cache invalidation."""
    payload = []
    for name in sorted(pico.tools):
        tool = pico.tools[name]
        payload.append(
            {
                "name": name,
                "schema": tool["schema"],
                "risky": tool["risky"],
                "description": tool["description"],
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_prefix(pico):
    """Assemble the agent's system prompt (tool list, rules, workspace context)."""
    tool_lines = []
    for name, tool in pico.tools.items():
        fields = ", ".join(f"{key}: {value}" for key, value in tool["schema"].items())
        risk = "approval required" if tool["risky"] else "safe"
        tool_lines.append(f"- {name}({fields}) [{risk}] {tool['description']}")
    tool_text = "\n".join(tool_lines)
    examples = "\n".join(
        [
            '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
            '<tool name="write_file" path="binary_search.py">'
            "<content>def binary_search(nums, target):\n    return -1\n</content></tool>",
            '<tool name="patch_file" path="binary_search.py">'
            "<old_text>return -1</old_text><new_text>return mid</new_text></tool>",
            '<tool>{"name":"run_shell","args":{"command":'
            '"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
            "<final>Done.</final>",
        ]
    )
    text = textwrap.dedent(
        f"""\
        You are pico, a small local coding agent working inside a local repository.

        Rules:
        - Use tools instead of guessing about the workspace.
        - Return exactly one <tool>...</tool> or one <final>...</final>.
        - Tool calls must look like:
          <tool>{{"name":"tool_name","args":{{...}}}}</tool>
        - For write_file and patch_file with multi-line text, prefer XML style:
          <tool name="write_file" path="file.py"><content>...</content></tool>
        - Final answers must look like:
          <final>your answer</final>
        - Never invent tool results.
        - Keep answers concise and concrete.
        - If the user asks you to create or update a specific file and the path
          is clear, use write_file or patch_file instead of repeatedly listing files.
        - Before writing tests for existing code, read the implementation first.
        - When writing tests, match the current implementation unless the user
          explicitly asked you to change the code.
        - New files should be complete and runnable, including obvious imports.
        - Do not repeat the same tool call with the same arguments if it did not
          help. Choose a different tool or return a final answer.
        - Required tool arguments must not be empty. Do not call read_file,
          write_file, patch_file, run_shell, or delegate with args={{}}.

        Tools:
        {tool_text}

        Valid response examples:
        {examples}

        {pico.workspace.text()}
        """
    ).strip()
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=pico.workspace.fingerprint(),
        tool_signature=tool_signature(pico),
        built_at=now(),
    )


def apply_prefix_state(pico, prefix_state):
    """Replace current prefix with a new state."""
    pico.prefix_state = prefix_state
    pico.prefix = prefix_state.text


def refresh_prefix(pico, force=False):
    """Check workspace changes and rebuild prefix if needed."""
    previous_hash = getattr(getattr(pico, "prefix_state", None), "hash", None)
    previous_workspace_fingerprint = getattr(
        getattr(pico, "prefix_state", None), "workspace_fingerprint", None
    )

    refreshed_workspace = WorkspaceContext.build(pico.root)
    refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
    workspace_changed = (
        force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
    )
    if workspace_changed:
        pico.workspace = refreshed_workspace

    prefix_state = (
        build_prefix(pico)
        if workspace_changed or force or previous_hash is None
        else pico.prefix_state
    )
    prefix_changed = force or previous_hash != prefix_state.hash
    if prefix_changed:
        apply_prefix_state(pico, prefix_state)

    pico._last_prefix_refresh = {
        "workspace_changed": workspace_changed,
        "prefix_changed": prefix_changed,
    }
    return dict(pico._last_prefix_refresh)


def build_prompt_and_metadata(pico, user_message):
    """Build the full prompt and collect metadata for this interaction turn."""
    refresh = refresh_prefix(pico)
    pico.resume_state = pico.evaluate_resume_state()
    prompt, metadata = pico.context_manager.build(user_message)
    metadata.update(
        {
            "prefix_chars": len(pico.prefix),
            "workspace_chars": len(pico.workspace.text()),
            "memory_chars": len(pico.memory_text()),
            "history_chars": len(pico.history_text()),
            "request_chars": len(user_message),
            "tool_count": len(pico.tools),
            "workspace_docs": len(pico.workspace.project_docs),
            "recent_commits": len(pico.workspace.recent_commits),
            "prefix_hash": pico.prefix_state.hash,
            "prompt_cache_key": pico.prefix_state.hash,
            "workspace_fingerprint": pico.prefix_state.workspace_fingerprint,
            "tool_signature": pico.prefix_state.tool_signature,
            "workspace_changed": refresh["workspace_changed"],
            "prefix_changed": refresh["prefix_changed"],
            "prompt_cache_supported": bool(
                getattr(pico.model_client, "supports_prompt_cache", False)
            ),
            "resume_status": pico.resume_state.get("status", CHECKPOINT_NONE_STATUS),
            "stale_summary_invalidations": int(
                pico.resume_state.get("stale_summary_invalidations", 0)
            ),
            "stale_paths": list(pico.resume_state.get("stale_paths", [])),
            "runtime_identity_mismatch_fields": list(
                pico.resume_state.get("runtime_identity_mismatch_fields", [])
            ),
        }
    )
    metadata.update(pico.detected_secret_env_summary())
    return prompt, metadata
