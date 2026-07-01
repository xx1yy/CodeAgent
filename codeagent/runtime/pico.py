"""Agent 运行时核心类 —— Pico。

Pico 是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。

本文件只保留 Pico 类的核心编排方法（__init__、ask 等），
具体子能力分散在 runtime/ 子包的其他模块中。
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from .. import memory as memorylib
from ..context_manager import ContextManager
from ..run_store import RunStore
from ..task_state import TaskState
from ..workspace import clip, now

from .constants import (
    DEFAULT_SHELL_ENV_ALLOWLIST,
    DEFAULT_FEATURE_FLAGS,
    CHECKPOINT_NONE_STATUS,
    CHECKPOINT_PARTIAL_STALE_STATUS,
    CHECKPOINT_WORKSPACE_MISMATCH_STATUS,
)


class Pico:
    """控制循环调度器 —— 驱动 agent 完成一次完整任务。"""

    def __init__(
        self,
        model_client,
        workspace,
        session_store,
        session=None,
        run_store=None,
        approval_policy="ask",
        max_steps=6,
        max_new_tokens=512,
        depth=0,
        max_depth=1,
        read_only=False,
        shell_env_allowlist=None,
        secret_env_names=None,
        feature_flags=None,
    ):
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".pico" / "runs")
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self.build_tools()
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state = None
        self.current_run_dir = None
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self.last_durable_promotions = []
        self.last_durable_rejections = []
        self.last_durable_superseded = []
        self._last_tool_result_metadata = {}
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(cls, model_client, workspace, session_store, session_id, **kwargs):
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    # ── thin wrappers ─────────────────────────────────────────────────

    def memory_text(self):
        return self.memory.render_memory_text()

    def feature_enabled(self, name):
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message):
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item):
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    # ── checkpoint helper (used by ask) ───────────────────────────────

    def _checkpoint_and_trace(self, task_state, user_message, trigger):
        checkpoint = self.create_checkpoint(task_state, user_message, trigger=trigger)
        self.run_store.write_task_state(task_state)
        self.emit_trace(
            task_state,
            "checkpoint_created",
            {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": trigger},
        )

    # ── main loop ─────────────────────────────────────────────────────

    def ask(self, user_message):
        """执行一次完整的 agent 回合，直到产出最终答案或命中停止条件。"""
        run_started_at = time.monotonic()
        self.memory.set_task_summary(user_message)
        self.record({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(
            run_id=self.new_run_id(),
            task_id=self.new_task_id(),
            user_request=user_message,
        )
        task_state.resume_status = self.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        self.current_task_state = task_state
        self.current_run_dir = self.run_store.start_run(task_state)
        self.emit_trace(
            task_state,
            "run_started",
            {"task_id": task_state.task_id, "user_request": clip(user_message, 300)},
        )

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            self.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = self._build_prompt_and_metadata(user_message)
            self.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                self._checkpoint_and_trace(task_state, user_message, "freshness_mismatch")
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                self.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(
                            prompt_metadata.get("runtime_identity_mismatch_fields", [])
                        ),
                    },
                )
                self._checkpoint_and_trace(task_state, user_message, "workspace_mismatch")
            if prompt_metadata.get("budget_reductions"):
                self._checkpoint_and_trace(task_state, user_message, "context_reduction")
            self.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            if getattr(self.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = self.model_client.complete(
                prompt,
                self.max_new_tokens,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(
                getattr(self.model_client, "last_completion_metadata", {}) or {}
            )
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            self.last_completion_metadata = completion_metadata
            self.last_prompt_metadata = prompt_metadata
            kind, payload = self.parse(raw)
            self.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                result = self.run_tool(name, args)
                self.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                self.run_store.write_task_state(task_state)
                self.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(self._last_tool_result_metadata or {}),
                    },
                )
                self._checkpoint_and_trace(task_state, user_message, "tool_executed")
                continue

            if kind == "retry":
                self.record(
                    {"role": "assistant", "content": payload, "created_at": now()}
                )
                self.run_store.write_task_state(task_state)
                continue

            final = (payload or raw).strip()
            self.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            self.promote_durable_memory(user_message, final)
            self._checkpoint_and_trace(task_state, user_message, "run_finished")
            self.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            self.run_store.write_report(
                task_state, self.redact_artifact(self.build_report(task_state))
            )
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = (
                "Stopped after too many malformed model responses "
                "without a valid tool call or final answer."
            )
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        self.record({"role": "assistant", "content": final, "created_at": now()})
        self.promote_durable_memory(user_message, final)
        self.run_store.write_task_state(task_state)
        self._checkpoint_and_trace(
            task_state, user_message, task_state.stop_reason or "run_stopped"
        )
        self.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        self.run_store.write_report(
            task_state, self.redact_artifact(self.build_report(task_state))
        )
        return final

    # ── lifecycle ─────────────────────────────────────────────────────

    def reset(self):
        self.session["history"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(
            self.session["memory"], workspace_root=self.root
        )
        self.session_store.save(self.session)


# ── attach methods from submodules ────────────────────────────────────

from .checkpoint import (  # noqa: E402
    evaluate_resume_state, checkpoint_state, current_checkpoint,
    invalidate_stale_memory, render_checkpoint_text, create_checkpoint,
    infer_next_step, current_runtime_identity,
)
from .durable import promote_durable_memory  # noqa: E402
from .executor import (  # noqa: E402
    run_tool, repeated_tool_call, approve, validate_tool,
    tool_list_files, tool_read_file, tool_search, tool_run_shell,
    tool_write_file, tool_patch_file, tool_delegate, tool_example,
    update_memory_after_tool, note_tool, record_process_note_for_tool,
    capture_workspace_snapshot, path as path_from_executor,
)
from .executor import remember, diff_workspace_snapshots  # noqa: E402
from .parser import (  # noqa: E402
    parse, retry_notice, parse_xml_tool, parse_attrs, extract, extract_raw,
)
from .prefix import (  # noqa: E402
    build_tools, tool_signature, build_prefix, apply_prefix_state,
    refresh_prefix, build_prompt_and_metadata,
)
from .render import render_history_text, build_report, emit_trace  # noqa: E402
from .secrets import (  # noqa: E402
    looks_sensitive_env_name, is_secret_env_name,
    configured_secret_env_items, detected_secret_env_items,
    secret_env_summary, detected_secret_env_summary,
    redact_text, redact_artifact, shell_env,
)
from .session import ensure_session_shape, new_task_id, new_run_id  # noqa: E402

Pico.evaluate_resume_state = evaluate_resume_state
Pico.checkpoint_state = checkpoint_state
Pico.current_checkpoint = current_checkpoint
Pico.invalidate_stale_memory = invalidate_stale_memory
Pico.render_checkpoint_text = render_checkpoint_text
Pico.create_checkpoint = create_checkpoint
Pico.infer_next_step = infer_next_step
Pico.current_runtime_identity = current_runtime_identity

Pico.promote_durable_memory = promote_durable_memory

Pico.run_tool = run_tool
Pico.repeated_tool_call = repeated_tool_call
Pico.approve = approve
Pico.validate_tool = validate_tool
Pico.tool_list_files = tool_list_files
Pico.tool_read_file = tool_read_file
Pico.tool_search = tool_search
Pico.tool_run_shell = tool_run_shell
Pico.tool_write_file = tool_write_file
Pico.tool_patch_file = tool_patch_file
Pico.tool_delegate = tool_delegate
Pico.tool_example = tool_example
Pico.update_memory_after_tool = update_memory_after_tool
Pico.note_tool = note_tool
Pico.record_process_note_for_tool = record_process_note_for_tool
Pico.capture_workspace_snapshot = capture_workspace_snapshot
Pico.path = path_from_executor

Pico.remember = staticmethod(remember)
Pico.diff_workspace_snapshots = staticmethod(diff_workspace_snapshots)

Pico.parse = staticmethod(parse)
Pico.retry_notice = staticmethod(retry_notice)
Pico.parse_xml_tool = staticmethod(parse_xml_tool)
Pico.parse_attrs = staticmethod(parse_attrs)
Pico.extract = staticmethod(extract)
Pico.extract_raw = staticmethod(extract_raw)

Pico.build_tools = build_tools
Pico.tool_signature = tool_signature
Pico.build_prefix = build_prefix
Pico._apply_prefix_state = apply_prefix_state
Pico.refresh_prefix = refresh_prefix
Pico._build_prompt_and_metadata = build_prompt_and_metadata

Pico.render_history_text = render_history_text
Pico.history_text = render_history_text
Pico.build_report = build_report
Pico.emit_trace = emit_trace

Pico.looks_sensitive_env_name = staticmethod(looks_sensitive_env_name)
Pico.is_secret_env_name = is_secret_env_name
Pico.configured_secret_env_items = configured_secret_env_items
Pico.detected_secret_env_items = detected_secret_env_items
Pico.secret_env_summary = secret_env_summary
Pico.detected_secret_env_summary = detected_secret_env_summary
Pico.redact_text = redact_text
Pico.redact_artifact = redact_artifact
Pico.shell_env = shell_env

Pico._ensure_session_shape = ensure_session_shape
Pico.new_task_id = staticmethod(new_task_id)
Pico.new_run_id = staticmethod(new_run_id)

MiniAgent = Pico
