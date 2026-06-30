"""工作记忆 — 当前任务摘要和最近接触的文件。

轻量容器，保存 agent 在当前轮次的工作上下文快照。
内部直接存字符串和路径，不继承 MemoryItem。
"""

from typing import List, Optional

from ..base import WORKING_FILE_LIMIT, clip, canonicalize_path


class WorkingContext:
    """工作记忆上下文。

    管理当前任务摘要和最近接触的文件列表。
    纯容器角色，不属于 MemoryItem 体系。
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self._workspace_root = workspace_root
        self._task_summary: str = ""
        self._recent_files: List[str] = []

    # ── task_summary ───────────────────────────────

    @property
    def task_summary(self) -> str:
        return self._task_summary

    def set_task_summary(self, summary: str) -> "WorkingContext":
        self._task_summary = clip(str(summary).strip(), 300)
        return self

    # ── recent_files ───────────────────────────────

    @property
    def recent_files(self) -> List[str]:
        return list(self._recent_files)

    def remember_file(self, path: str) -> "WorkingContext":
        canon_path = canonicalize_path(path, self._workspace_root).strip()
        if not canon_path:
            return self
        self._recent_files = [p for p in self._recent_files if p != canon_path]
        self._recent_files.append(canon_path)
        self._recent_files = self._recent_files[-WORKING_FILE_LIMIT:]
        return self

    def file_exists(self, path: str) -> bool:
        canon_path = canonicalize_path(path, self._workspace_root).strip()
        return canon_path in self._recent_files

    def is_empty(self) -> bool:
        return not self._task_summary and not self._recent_files

    # ── 序列化 ─────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "task_summary": self._task_summary,
            "recent_files": list(self._recent_files),
        }

    def load_dict(self, data: dict) -> "WorkingContext":
        if not isinstance(data, dict):
            return self
        summary = str(data.get("task_summary", "")).strip()
        if summary:
            self.set_task_summary(summary)
        for path in data.get("recent_files", []):
            if str(path).strip():
                self.remember_file(str(path).strip())
        return self
