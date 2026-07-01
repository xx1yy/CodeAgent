"""Tool execution with safety guardrails.

All functions take a ``pico`` (Pico instance) as their first argument
so the caller can decide which runtime to use.
"""

import json
import os
import re
import hashlib

from .. import memory as memorylib
from .. import tools as toolkit
from ..workspace import IGNORED_PATH_NAMES, clip
from .constants import REDACTED_VALUE


# ── workspace snapshots ───────────────────────────────────────────────

def capture_workspace_snapshot(pico):
    """SHA-256 hash every file in workspace for change detection."""
    snapshot = {}
    for path in pico.root.rglob("*"):
        try:
            relative_parts = path.relative_to(pico.root).parts
        except ValueError:
            continue
        if any(part in IGNORED_PATH_NAMES for part in relative_parts):
            continue
        if not path.is_file():
            continue
        try:
            snapshot[path.relative_to(pico.root).as_posix()] = (
                hashlib.sha256(path.read_bytes()).hexdigest()
            )
        except Exception:
            continue
    return snapshot


def diff_workspace_snapshots(before, after):
    """Compare two workspace snapshots and return changed paths + summaries."""
    changed_paths = []
    summaries = []
    all_paths = sorted(set(before) | set(after))
    for path in all_paths:
        if before.get(path) == after.get(path):
            continue
        changed_paths.append(path)
        if path not in before:
            summaries.append(f"created:{path}")
        elif path not in after:
            summaries.append(f"deleted:{path}")
        else:
            summaries.append(f"modified:{path}")
    return changed_paths, summaries


def remember(bucket, item, limit):
    """Add item to bucket, deduplicating and respecting a size limit."""
    if not item:
        return
    if item in bucket:
        bucket.remove(item)
    bucket.append(item)
    del bucket[:-limit]


# ── memory updates after tool execution ──────────────────────────────

def update_memory_after_tool(pico, name, args, result):
    """Selectively promote tool results into working memory.

    Only high-value facts are stored — full results already live in history.
    """
    if not pico.feature_enabled("memory"):
        return
    path = args.get("path")
    if not path:
        return

    canonical_path = pico.memory.canonical_path(path)
    if name in {"read_file", "write_file", "patch_file"}:
        pico.memory.remember_file(canonical_path)
    if name == "read_file":
        summary = memorylib.summarize_read_result(result)
        pico.memory.set_file_summary(canonical_path, summary)
        pico.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
    elif name in {"write_file", "patch_file"}:
        pico.memory.invalidate_file_summary(canonical_path)


def note_tool(pico, name, args, result):
    """Thin wrapper around update_memory_after_tool."""
    update_memory_after_tool(pico, name, args, result)


def record_process_note_for_tool(pico, name, metadata):
    """Record process notes for partial success, error, or rejection."""
    status = str(metadata.get("tool_status", "")).strip()
    if status not in {"partial_success", "error", "rejected"}:
        return
    affected_paths = [
        str(path).strip()
        for path in metadata.get("affected_paths", [])
        if str(path).strip()
    ]
    path_text = ", ".join(affected_paths) or "workspace"
    if status == "partial_success":
        text = f"{name} partial_success on {path_text}; inspect diff before retry"
    elif status == "error":
        text = f"{name} error on {path_text}; check the failure before retry"
    else:
        text = f"{name} rejected; choose a different action before retry"
    tags = ["process", status, *affected_paths]
    pico.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
    pico.session["memory"] = pico.memory.to_dict()


# ── tool execution (main entry point) ────────────────────────────────

def run_tool(pico, name, args):
    """Execute a tool call with full safety guardrails.

    Pipeline: existence check → argument validation → duplicate check →
    approval → execute → capture workspace diff → update memory.
    """
    tool = pico.tools.get(name)
    if tool is None:
        pico._last_tool_result_metadata = {
            "tool_status": "rejected",
            "tool_error_code": "unknown_tool",
            "security_event_type": "",
            "risk_level": "high",
            "read_only": False,
            "affected_paths": [],
            "workspace_changed": False,
            "diff_summary": [],
        }
        return f"error: unknown tool '{name}'"
    try:
        validate_tool(pico, name, args)
    except Exception as exc:
        example = tool_example(pico, name)
        message = f"error: invalid arguments for {name}: {exc}"
        if example:
            message += f"\nexample: {example}"
        security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
        pico._last_tool_result_metadata = {
            "tool_status": "rejected",
            "tool_error_code": "invalid_arguments",
            "security_event_type": security_event_type,
            "risk_level": "high" if tool["risky"] else "low",
            "read_only": not tool["risky"],
            "affected_paths": [],
            "workspace_changed": False,
            "diff_summary": [],
        }
        return message
    if repeated_tool_call(pico, name, args):
        pico._last_tool_result_metadata = {
            "tool_status": "rejected",
            "tool_error_code": "repeated_identical_call",
            "security_event_type": "",
            "risk_level": "high" if tool["risky"] else "low",
            "read_only": not tool["risky"],
            "affected_paths": [],
            "workspace_changed": False,
            "diff_summary": [],
        }
        return (
            f"error: repeated identical tool call for {name}; "
            "choose a different tool or return a final answer"
        )
    if tool["risky"] and not approve(pico, name, args):
        pico._last_tool_result_metadata = {
            "tool_status": "rejected",
            "tool_error_code": "approval_denied",
            "security_event_type": "read_only_block" if pico.read_only else "approval_denied",
            "risk_level": "high",
            "read_only": False,
            "affected_paths": [],
            "workspace_changed": False,
            "diff_summary": [],
        }
        return f"error: approval denied for {name}"
    before_snapshot = capture_workspace_snapshot(pico) if tool["risky"] else {}
    after_snapshot = before_snapshot
    try:
        result = clip(tool["run"](args))
        after_snapshot = capture_workspace_snapshot(pico) if tool["risky"] else before_snapshot
        affected_paths, diff_summary = diff_workspace_snapshots(before_snapshot, after_snapshot)
        workspace_changed = bool(affected_paths)
        tool_status = "ok"
        tool_error_code = ""
        if name == "run_shell":
            match = re.search(r"exit_code:\s*(-?\d+)", result)
            exit_code = int(match.group(1)) if match else 0
            if exit_code != 0 and workspace_changed:
                tool_status = "partial_success"
                tool_error_code = "tool_partial_success"
            elif exit_code != 0:
                tool_status = "error"
                tool_error_code = "tool_failed"
        update_memory_after_tool(pico, name, args, result)
        pico._last_tool_result_metadata = {
            "tool_status": tool_status,
            "tool_error_code": tool_error_code,
            "security_event_type": "",
            "risk_level": "high" if tool["risky"] else "low",
            "read_only": not tool["risky"],
            "affected_paths": affected_paths,
            "workspace_changed": workspace_changed,
            "workspace_fingerprint": pico.workspace.fingerprint(),
            "diff_summary": diff_summary,
        }
        record_process_note_for_tool(pico, name, pico._last_tool_result_metadata)
        return result
    except Exception as exc:
        after_snapshot = capture_workspace_snapshot(pico) if tool["risky"] else before_snapshot
        affected_paths, diff_summary = diff_workspace_snapshots(before_snapshot, after_snapshot)
        workspace_changed = bool(affected_paths)
        security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
        pico._last_tool_result_metadata = {
            "tool_status": "partial_success" if workspace_changed else "error",
            "tool_error_code": "tool_partial_success" if workspace_changed else "tool_failed",
            "security_event_type": security_event_type,
            "risk_level": "high" if tool["risky"] else "low",
            "read_only": not tool["risky"],
            "affected_paths": affected_paths,
            "workspace_changed": workspace_changed,
            "workspace_fingerprint": pico.workspace.fingerprint(),
            "diff_summary": diff_summary,
        }
        record_process_note_for_tool(pico, name, pico._last_tool_result_metadata)
        return f"error: tool {name} failed: {exc}"


# ── safety checks ─────────────────────────────────────────────────────

def repeated_tool_call(pico, name, args):
    """Detect identical tool calls in the last two history entries."""
    tool_events = [
        item for item in pico.session["history"]
        if item["role"] == "tool"
    ]
    if len(tool_events) < 2:
        return False
    recent = tool_events[-2:]
    return all(item["name"] == name and item["args"] == args for item in recent)


def approve(pico, name, args):
    """Check approval policy for risky tool calls."""
    if pico.read_only:
        return False
    if pico.approval_policy == "auto":
        return True
    if pico.approval_policy == "never":
        return False
    try:
        answer = input(
            f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] "
        )
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def validate_tool(pico, name, args):
    """Run generic validation plus runtime-specific constraints."""
    toolkit.validate_tool(pico, name, args)
    if name == "delegate":
        if pico.depth >= pico.max_depth:
            raise ValueError("delegate depth exceeded")


def tool_example(pico, name):
    """Return an example call string for the given tool, if available."""
    return toolkit.tool_example(name)


# ── tool implementations (thin wrappers around toolkit) ──────────────

def tool_list_files(pico, args):
    return toolkit.tool_list_files(pico, args)

def tool_read_file(pico, args):
    return toolkit.tool_read_file(pico, args)

def tool_search(pico, args):
    return toolkit.tool_search(pico, args)

def tool_run_shell(pico, args):
    return toolkit.tool_run_shell(pico, args)

def tool_write_file(pico, args):
    return toolkit.tool_write_file(pico, args)

def tool_patch_file(pico, args):
    return toolkit.tool_patch_file(pico, args)

def tool_delegate(pico, args):
    return toolkit.tool_delegate(pico, args)


# ── path resolution ──────────────────────────────────────────────────

def path(pico, raw_path):
    """Resolve and validate a path is within the workspace root."""
    from pathlib import Path

    p = Path(raw_path)
    p = p if p.is_absolute() else pico.root / p
    resolved = p.resolve()
    if os.path.commonpath([str(pico.root), str(resolved)]) != str(pico.root):
        raise ValueError(f"path escapes workspace: {raw_path}")
    return resolved
