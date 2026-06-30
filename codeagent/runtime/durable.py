"""Durable memory extraction and promotion logic.

Extracts structured knowledge lines from model final answers
when the user intent indicates a "remember this" request.
"""

import re

from .constants import (
    REDACTED_VALUE,
    SECRET_SHAPED_TEXT_PATTERN,
    DURABLE_MEMORY_INTENT_PATTERN,
    DURABLE_MEMORY_INTENT_ZH_PATTERN,
    DURABLE_MEMORY_LINE_PATTERNS,
)


def reject_durable_reason(note_text):
    """Check if a note text should be rejected from durable memory.

    Returns a reason string if rejected, or empty string if accepted.
    """
    text = str(note_text or "").strip()
    lowered = text.lower()
    if not text:
        return "empty"
    if REDACTED_VALUE in text or SECRET_SHAPED_TEXT_PATTERN.search(text):
        return "secret_shaped"
    checkpoint_like_prefixes = (
        "current goal", "current blocker", "next step", "current phase",
        "key files", "freshness",
        "当前目标", "当前卡点", "下一步", "当前阶段", "关键文件",
        "已完成", "已排除",
    )
    if any(lowered.startswith(prefix) for prefix in checkpoint_like_prefixes):
        return "transient_task_state"
    if re.search(r"(?i)\b(stdout|stderr|traceback|exit_code)\b", text) or len(text) > 220:
        return "noisy_output"
    return ""


def extract_durable_promotions(user_message, final_answer):
    """Extract structured durable memory lines from the model's final answer.

    Only activates when user_message contains memory-related intent keywords.
    Returns (promotions, rejections) where promotions is a list of
    (topic, note_text) tuples and rejections is a list of reason strings.
    """
    user_text = str(user_message or "")
    if not (DURABLE_MEMORY_INTENT_PATTERN.search(user_text)
            or DURABLE_MEMORY_INTENT_ZH_PATTERN.search(user_text)):
        return [], []
    promotions = []
    rejections = []
    for line in str(final_answer or "").splitlines():
        text = line.strip()
        if not text or REDACTED_VALUE in text:
            continue
        for topic, pattern in DURABLE_MEMORY_LINE_PATTERNS:
            match = pattern.match(text)
            if not match:
                continue
            note_text = match.group(1).strip()
            if note_text:
                reason = reject_durable_reason(note_text)
                if reason:
                    rejections.append(f"{topic}:{reason}")
                    break
                promotions.append((topic, note_text))
            break
    return promotions, rejections


def promote_durable_memory(pico, user_message, final_answer):
    """Promote extracted durable memory lines into the agent's memory store.

    This is the main entry point called from Pico.ask() after a successful run.
    """
    promotions, rejections = extract_durable_promotions(user_message, final_answer)
    promoted, superseded = pico.memory.promote_durable(promotions)
    pico.session["memory"] = pico.memory.to_dict()
    pico.last_durable_promotions = promoted
    pico.last_durable_rejections = rejections
    pico.last_durable_superseded = superseded
    return promoted, rejections, superseded
