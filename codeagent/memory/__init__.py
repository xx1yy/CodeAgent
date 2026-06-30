"""memory — 重构后的记忆系统（Pydantic 版本）。

此模块提供：

- MemoryItem 统一基类，所有记忆条目共享同一数据结构
- 按职责拆分为独立模块（工作记忆 / 情景笔记 / 文件摘要 / 持久化记忆 / 检索 / 管理器）
- MemoryManager 作为单一入口的链式 API
- 完整兼容旧版 dict 序列化格式
"""

from .base import (
    EPISODIC_NOTE_LIMIT,
    FILE_SUMMARY_LIMIT,
    RELEVANT_MEMORY_LIMIT,
    WORKING_FILE_LIMIT,
    MemoryItem,
    canonicalize_path,
    clip,
    file_freshness,
    now,
    tokenize,
)
from .manager import MemoryManager, default_memory_state
from .retrieval import (
    is_effectively_empty,
    render_memory_text,
    retrieval_candidates,
    retrieval_view,
)
from .types import (
    DurableMemoryStore,
    DurableNote,
    EpisodicNote,
    EpisodicNoteStore,
    FileSummary,
    FileSummaryStore,
    WorkingContext,
)

__all__ = [
    # 基类
    "MemoryItem",
    # 常量
    "WORKING_FILE_LIMIT",
    "EPISODIC_NOTE_LIMIT",
    "FILE_SUMMARY_LIMIT",
    "RELEVANT_MEMORY_LIMIT",
    # 工具
    "now",
    "clip",
    "tokenize",
    "canonicalize_path",
    "file_freshness",
    # 记忆类型
    "WorkingContext",
    "EpisodicNote",
    "EpisodicNoteStore",
    "FileSummary",
    "FileSummaryStore",
    "DurableNote",
    "DurableMemoryStore",
    # 检索
    "retrieval_candidates",
    "retrieval_view",
    "render_memory_text",
    "is_effectively_empty",
    # 管理器
    "MemoryManager",
    "default_memory_state",
]
