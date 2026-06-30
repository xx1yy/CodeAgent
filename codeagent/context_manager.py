"""Prompt 组装与上下文预算控制。

兼容转发层：所有实现已移至 pico/context/ 包。
"""

from .context import ContextManager

__all__ = ["ContextManager"]
