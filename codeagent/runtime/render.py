"""Output rendering — history text, report, and trace emission."""

import json

from ..workspace import MAX_HISTORY, clip, now


def render_history_text(pico):
    """Format session history as condensed text for the model prompt.

    Recent entries get richer representation; older tool reads are deduplicated.
    """
    history = pico.session["history"]
    if not history:
        return "- empty"

    lines = []
    seen_reads = set()
    recent_start = max(0, len(history) - 6)
    for index, item in enumerate(history):
        recent = index >= recent_start
        if item["role"] == "tool" and item["name"] == "read_file" and not recent:
            path = str(item["args"].get("path", ""))
            if path in seen_reads:
                continue
            seen_reads.add(path)

        if item["role"] == "tool":
            limit = 900 if recent else 180
            lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
            lines.append(clip(item["content"], limit))
        else:
            limit = 900 if recent else 220
            lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

    return clip("\n".join(lines), MAX_HISTORY)


def build_report(pico, task_state):
    """Assemble the final run report dict — result + metadata + redacted secrets."""
    return {
        "run_id": task_state.run_id,
        "task_id": task_state.task_id,
        "status": task_state.status,
        "stop_reason": task_state.stop_reason,
        "final_answer": task_state.final_answer,
        "tool_steps": task_state.tool_steps,
        "attempts": task_state.attempts,
        "checkpoint_id": task_state.checkpoint_id,
        "resume_status": task_state.resume_status,
        "task_state": task_state.to_dict(),
        "prompt_metadata": pico.last_prompt_metadata,
        "durable_promotions": list(pico.last_durable_promotions),
        "durable_rejections": list(pico.last_durable_rejections),
        "durable_superseded": list(pico.last_durable_superseded),
        "redacted_env": pico.detected_secret_env_summary(),
    }


def emit_trace(pico, task_state, event, payload=None):
    """Write a trace event to the run store after redacting secrets."""
    payload = pico.redact_artifact(payload or {})
    payload["event"] = event
    payload["created_at"] = now()
    pico.run_store.append_trace(task_state, payload)
    return payload
