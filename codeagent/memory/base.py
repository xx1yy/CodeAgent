"""记忆系统统一数据结构基类。

定义 MemoryItem 作为所有记忆条目的 Pydantic 基类，
以及共享的工具函数和常量。
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

# ── 常量 ──────────────────────────────────────────

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
RELEVANT_MEMORY_LIMIT = 3

# ── 工具函数 ──────────────────────────────────────


def now() -> str:
    """返回 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def clip(text: str, limit: int = 4000) -> str:
    """截断文本到指定长度，超过时追加截断提示。"""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def tokenize(text: str) -> Set[str]:
    """将文本拆分为小写 token 集合，用于关键词匹配。"""
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", str(text))}


def parse_timestamp(value: Any) -> float:
    """尝试将 ISO 时间戳解析为浮点数（epoch seconds），失败返回 0.0。"""
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def file_freshness(raw_path: str, workspace_root: Optional[str] = None) -> Optional[str]:
    """计算文件内容的 SHA256 指纹，用于判断文件是否已变更。"""
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def resolve_workspace_path(raw_path: str, workspace_root: Optional[str] = None) -> Optional[Path]:
    """将路径解析为工作区内的绝对路径，若不在工作区内则返回 None。"""
    path = Path(str(raw_path))
    if workspace_root is None:
        return path

    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path: str, workspace_root: Optional[str] = None) -> str:
    """将路径规范化为相对于工作区根目录的 POSIX 路径。"""
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None:
        return Path(str(raw_path)).as_posix()
    if workspace_root is None:
        return Path(str(raw_path)).as_posix()
    root = Path(workspace_root).resolve()
    return resolved.relative_to(root).as_posix()


def ensure_list(value: Any) -> list:
    """将各种输入统一转换为 list。"""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def dedupe_preserve_order(items: list) -> list:
    """去重并保持顺序。"""
    seen: set = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


# ── 基类 ──────────────────────────────────────────


class MemoryItem(BaseModel):
    """记忆项统一数据结构。

    所有记忆条目的基类，提供统一的身份标识、内容、记忆类型、
    时间戳和重要性评分。
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    content: str = ""
    memory_type: str = "generic"
    user_id: str = "default"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
