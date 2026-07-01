"""Secret environment variable detection and redaction."""

import os

from .constants import SENSITIVE_ENV_NAME_MARKERS, REDACTED_VALUE


def looks_sensitive_env_name(name):
    """Check if an env var name looks like it holds a secret."""
    upper = str(name).upper()
    return any(
        upper == marker
        or upper.endswith(marker)
        or upper.endswith(f"_{marker}")
        for marker in SENSITIVE_ENV_NAME_MARKERS
    )


def is_secret_env_name(pico, name):
    """Check if an env var name is configured or detected as secret."""
    upper = str(name).upper()
    return upper in pico.secret_env_names or looks_sensitive_env_name(upper)


def configured_secret_env_items(pico):
    """Return sorted list of (name, value) for explicitly configured secrets."""
    items = [
        (name, value)
        for name, value in os.environ.items()
        if str(name).upper() in pico.secret_env_names and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def detected_secret_env_items(pico):
    """Return sorted list of all detected secret env vars."""
    items = [
        (name, value)
        for name, value in os.environ.items()
        if is_secret_env_name(pico, name) and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def secret_env_summary(pico):
    """Summary of configured secret env vars (names only)."""
    names = [name for name, _ in configured_secret_env_items(pico)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def detected_secret_env_summary(pico):
    """Summary of all detected secret env vars (names only)."""
    names = [name for name, _ in detected_secret_env_items(pico)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def redact_text(pico, text):
    """Replace all occurrences of secret values with <redacted>."""
    text = str(text)
    for _, value in sorted(
        detected_secret_env_items(pico),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        text = text.replace(value, REDACTED_VALUE)
    return text


def redact_artifact(pico, value, key=None):
    """Recursively redact secrets from a data structure."""
    if key and is_secret_env_name(pico, key):
        return REDACTED_VALUE
    if isinstance(value, dict):
        return {
            str(item_key): redact_artifact(pico, item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_artifact(pico, item, key=key) for item in value]
    if isinstance(value, tuple):
        return [redact_artifact(pico, item, key=key) for item in value]
    if isinstance(value, str):
        return redact_text(pico, value)
    return value


def shell_env(pico):
    """Build the environment dict for shell commands (allowlist + PWD)."""
    env = {
        name: os.environ[name]
        for name in pico.shell_env_allowlist
        if name in os.environ
    }
    env["PWD"] = str(pico.root)
    if "PATH" not in env and os.environ.get("PATH"):
        env["PATH"] = os.environ["PATH"]
    return env
