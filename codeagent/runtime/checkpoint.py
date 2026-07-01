"""Checkpoint management — session state persistence and resume evaluation."""

import uuid
import hashlib

from .. import memory as memorylib
from ..workspace import clip, now
from .constants import (
    CHECKPOINT_NONE_STATUS,
    CHECKPOINT_FULL_VALID_STATUS,
    CHECKPOINT_PARTIAL_STALE_STATUS,
    CHECKPOINT_WORKSPACE_MISMATCH_STATUS,
    CHECKPOINT_SCHEMA_MISMATCH_STATUS,
    CHECKPOINT_SCHEMA_VERSION,
)


# ── identity ──────────────────────────────────────────────────────────

def current_runtime_identity(pico):
    """Capture current runtime configuration as a hashable identity dict."""
    return {
        "session_id": pico.session.get("id", ""),
        "cwd": str(pico.root),
        "model": str(getattr(pico.model_client, "model", "")),
        "model_client": pico.model_client.__class__.__name__,
        "approval_policy": pico.approval_policy,
        "read_only": bool(pico.read_only),
        "max_steps": int(pico.max_steps),
        "max_new_tokens": int(pico.max_new_tokens),
        "feature_flags": dict(pico.feature_flags),
        "shell_env_allowlist": list(pico.shell_env_allowlist),
        "workspace_fingerprint": getattr(
            getattr(pico, "prefix_state", None), "workspace_fingerprint",
            pico.workspace.fingerprint(),
        ),
        "tool_signature": pico.tool_signature(),
    }


# ── state accessors ───────────────────────────────────────────────────

def checkpoint_state(pico):
    """Return the checkpoint state dict, ensuring shape is valid."""
    pico._ensure_session_shape()
    return pico.session["checkpoints"]


def current_checkpoint(pico):
    """Return the current active checkpoint, or None."""
    state = checkpoint_state(pico)
    checkpoint_id = str(state.get("current_id", "")).strip()
    if not checkpoint_id:
        return None
    return state.get("items", {}).get(checkpoint_id)


# ── staleness ─────────────────────────────────────────────────────────

def invalidate_stale_memory(pico):
    """Invalidate file summaries whose freshness has changed."""
    invalidated = pico.memory.invalidate_stale_file_summaries()
    pico.session["memory"] = pico.memory.to_dict()
    return invalidated


# ── resume evaluation ─────────────────────────────────────────────────

def evaluate_resume_state(pico):
    """Evaluate whether the current checkpoint is still valid for resumption.

    Checks schema version, file freshness, and runtime identity.
    Returns a resume_state dict with status and diagnostics.
    """
    previous_resume_state = dict(pico.session.get("resume_state", {}) or {})
    invalidated = invalidate_stale_memory(pico)
    checkpoint = current_checkpoint(pico)
    status = CHECKPOINT_NONE_STATUS
    stale_paths = list(invalidated)
    mismatch_fields = []
    if checkpoint:
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
        else:
            for item in checkpoint.get("key_files", []):
                path = str(item.get("path", "")).strip()
                if not path:
                    continue
                expected = item.get("freshness")
                current = memorylib.file_freshness(path, pico.root)
                if expected != current and path not in stale_paths:
                    stale_paths.append(path)
            saved_identity = dict(
                checkpoint.get("runtime_identity", {})
                or pico.session.get("runtime_identity", {})
                or {}
            )
            current_identity = current_runtime_identity(pico)
            identity_keys = (
                "cwd", "model", "model_client", "approval_policy",
                "read_only", "max_steps", "max_new_tokens", "feature_flags",
                "shell_env_allowlist", "workspace_fingerprint", "tool_signature",
            )
            for key in identity_keys:
                if key not in saved_identity:
                    continue
                if saved_identity.get(key) != current_identity.get(key):
                    mismatch_fields.append(key)
            mismatch_fields.sort()
            if stale_paths:
                status = CHECKPOINT_PARTIAL_STALE_STATUS
            elif mismatch_fields:
                status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
            else:
                status = CHECKPOINT_FULL_VALID_STATUS

    resume_state = {
        "status": status,
        "stale_paths": stale_paths,
        "runtime_identity_mismatch_fields": mismatch_fields,
        "stale_summary_invalidations": max(
            len(invalidated),
            int(previous_resume_state.get("stale_summary_invalidations", 0))
            if status == CHECKPOINT_PARTIAL_STALE_STATUS
            else 0,
        ),
    }
    pico.session["resume_state"] = resume_state
    pico.session["runtime_identity"] = current_runtime_identity(pico)
    return resume_state


# ── text rendering ────────────────────────────────────────────────────

def render_checkpoint_text(pico):
    """Render the current checkpoint as human-readable text for prompts."""
    checkpoint = current_checkpoint(pico)
    if not checkpoint:
        return ""
    lines = [
        "Task checkpoint:",
        f"- Resume status: {pico.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
        f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
        f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
        f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
    ]
    key_files = [
        str(item.get("path", "")).strip()
        for item in checkpoint.get("key_files", [])
        if str(item.get("path", "")).strip()
    ]
    lines.append(f"- Key files: {', '.join(key_files) or '-'}")
    if checkpoint.get("completed"):
        lines.append(
            "- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", []))
        )
    if checkpoint.get("excluded"):
        lines.append(
            "- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", []))
        )
    if pico.resume_state.get("stale_paths"):
        lines.append("- Stale paths: " + ", ".join(pico.resume_state["stale_paths"]))
    summary = str(checkpoint.get("summary", "")).strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    return "\n".join(lines)


# ── checkpoint creation ───────────────────────────────────────────────

def create_checkpoint(pico, task_state, user_message, trigger):
    """Create a new checkpoint snapshot of the current task state."""
    state = checkpoint_state(pico)
    current = current_checkpoint(pico)
    checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
    key_files = []
    for path in pico.memory.to_dict()["working"]["recent_files"]:
        file_freshness = memorylib.file_freshness(path, pico.root)
        key_files.append({"path": path, "freshness": file_freshness})
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": now(),
        "current_goal": str(user_message),
        "completed": [task_state.final_answer] if task_state.final_answer else [],
        "excluded": [],
        "current_blocker": (
            ""
            if str(task_state.stop_reason or "") in ("", "final_answer_returned")
            else str(task_state.stop_reason)
        ),
        "next_step": infer_next_step(pico, task_state),
        "key_files": key_files,
        "freshness": {item["path"]: item["freshness"] for item in key_files},
        "summary": f"{trigger}: {clip(str(user_message), 120)}",
        "runtime_identity": current_runtime_identity(pico),
    }
    state["items"][checkpoint_id] = checkpoint
    state["current_id"] = checkpoint_id
    task_state.checkpoint_id = checkpoint_id
    pico.session["runtime_identity"] = checkpoint["runtime_identity"]
    pico.session_path = pico.session_store.save(pico.session)
    return checkpoint


def infer_next_step(pico, task_state):
    """Infer a human-readable next step description from task state."""
    if task_state.status == "completed":
        return "No next step recorded."
    if task_state.stop_reason == "step_limit_reached":
        return "Resume from the latest checkpoint and continue the task."
    if task_state.last_tool:
        return f"Decide the next action after {task_state.last_tool}."
    return "Continue the task from the latest checkpoint."
