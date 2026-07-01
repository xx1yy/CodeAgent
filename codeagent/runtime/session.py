"""Session shape maintenance and task/run ID generation."""

import uuid
from datetime import datetime

from .. import memory as memorylib


def ensure_session_shape(pico):
    """Ensure session dict has all required keys with proper types."""
    pico.session.setdefault("history", [])
    pico.session.setdefault("memory", memorylib.default_memory_state())
    checkpoints = pico.session.setdefault("checkpoints", {})
    if not isinstance(checkpoints, dict):
        checkpoints = {}
        pico.session["checkpoints"] = checkpoints
    checkpoints.setdefault("current_id", "")
    checkpoints.setdefault("items", {})
    runtime_identity = pico.session.setdefault("runtime_identity", {})
    if not isinstance(runtime_identity, dict):
        pico.session["runtime_identity"] = {}
    resume_state = pico.session.setdefault("resume_state", {})
    if not isinstance(resume_state, dict):
        pico.session["resume_state"] = {}


def new_task_id():
    """Generate a unique task identifier."""
    return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def new_run_id():
    """Generate a unique run identifier."""
    return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
