"""记忆管理器 — 统一接口，组合各记忆模块。

MemoryManager 是 LayeredMemory 的替代品（Pydantic 版本），
对外提供链式调用的统一 API，内部委派给各专用 Store。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    EPISODIC_NOTE_LIMIT,
    FILE_SUMMARY_LIMIT,
    RELEVANT_MEMORY_LIMIT,
    WORKING_FILE_LIMIT,
    MemoryItem,
    canonicalize_path,
    file_freshness,
    now,
)
from .retrieval import (
    is_effectively_empty,
    render_memory_text,
    retrieval_candidates,
    retrieval_view,
)
from .types.durable_note import DurableMemoryStore
from .types.episodic_note import EpisodicNoteStore
from .types.file_summary import FileSummaryStore
from .types.working_context import WorkingContext


class MemoryManager:
    """记忆管理器 — 统一接口，组合各记忆模块。

    用法::

        manager = MemoryManager(workspace_root="/path/to/repo")
        (manager
            .set_task_summary("Fix flaky tests")
            .remember_file("src/app.py")
            .append_note("Use mock for network calls", tags=("testing",))
            .set_file_summary("src/app.py", "Core app logic with API routes"))
        print(manager.render())
        related = manager.retrieve("testing")
    """

    def __init__(self, workspace_root: Optional[str] = None, state: Optional[dict] = None):
        self._workspace_root = workspace_root

        # 各记忆层
        self._working = WorkingContext(workspace_root)
        self._episodic = EpisodicNoteStore()
        self._file_summaries = FileSummaryStore(workspace_root)
        self._durable = (
            DurableMemoryStore(Path(workspace_root) / ".pico" / "memory")
            if workspace_root is not None
            else None
        )

        # 如果传入了旧版 state dict，尝试加载
        if state is not None:
            self._load_dict(state)

    # ── 属性 ───────────────────────────────────────

    @property
    def working(self) -> WorkingContext:
        return self._working

    @property
    def episodic(self) -> EpisodicNoteStore:
        return self._episodic

    @property
    def file_summaries(self) -> FileSummaryStore:
        return self._file_summaries

    @property
    def durable(self) -> Optional[DurableMemoryStore]:
        return self._durable

    # ── 工作记忆 ──────────────────────────────────

    def set_task_summary(self, summary: str) -> "MemoryManager":
        """设置当前任务摘要。"""
        self._working.set_task_summary(summary)
        return self

    def remember_file(self, path: str) -> "MemoryManager":
        """记录一个最近接触的文件。"""
        self._working.remember_file(path)
        return self

    # ── 情景笔记 ──────────────────────────────────

    def append_note(
        self,
        text: str,
        tags: Tuple[str, ...] = (),
        source: str = "",
        created_at: Optional[str] = None,
        kind: str = "episodic",
    ) -> "MemoryManager":
        """追加一条情景笔记。"""
        self._episodic.append(
            text, tags=tags, source=source, created_at=created_at, kind=kind
        )
        return self

    # ── 文件摘要 ──────────────────────────────────

    def set_file_summary(self, path: str, summary: str) -> "MemoryManager":
        """设置文件摘要。"""
        self._file_summaries.set(path, summary)
        return self

    def invalidate_file_summary(self, path: str) -> "MemoryManager":
        """主动移除指定文件的摘要。"""
        self._file_summaries.invalidate(path)
        return self

    def invalidate_stale_file_summaries(self) -> List[str]:
        """失效所有不新鲜的文件摘要。返回被失效的路径列表。"""
        return self._file_summaries.invalidate_stale()

    # ── 检索 ──────────────────────────────────────

    def retrieval_candidates(self, query: str, limit: int = RELEVANT_MEMORY_LIMIT) -> List[MemoryItem]:
        """从情景笔记和持久化记忆中联合召回相关条目。"""
        return retrieval_candidates(self._episodic, self._durable, query, limit=limit)

    def retrieve(self, query: str, limit: int = RELEVANT_MEMORY_LIMIT) -> str:
        """返回格式化的召回文本，供注入 prompt。"""
        return retrieval_view(self._episodic, self._durable, query, limit=limit)

    # ── 持久化记忆晋升 ────────────────────────────

    def promote_durable(self, promotions: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """将笔记晋升为持久化记忆。

        Args:
            promotions: [(topic, note_text), ...]

        Returns:
            (promoted, superseded)
        """
        if self._durable is None:
            return [], []
        return self._durable.promote(promotions)

    # ── 渲染 ──────────────────────────────────────

    def render(self) -> str:
        """渲染完整记忆仪表盘文本。"""
        return render_memory_text(
            self._working,
            self._file_summaries,
            self._episodic,
            self._durable,
            self._workspace_root,
        )

    def is_empty(self) -> bool:
        """判断记忆是否为空。"""
        return is_effectively_empty(self._working, self._file_summaries, self._episodic)

    # ── 序列化（兼容旧版 dict 格式） ──────────────

    def to_dict(self) -> dict:
        """序列化为 dict（兼容原始 memory.py 的格式）。"""
        # 构建 file_summaries 的旧版格式
        file_summaries = {}
        for path, s in self._file_summaries._summaries.items():
            file_summaries[path] = {
                "summary": s.content,
                "created_at": s.timestamp.isoformat()
                if hasattr(s.timestamp, "isoformat")
                else str(s.timestamp),
                "freshness": s.freshness,
            }

        # 构建 episodic_notes 的旧版格式
        episodic_notes = []
        for note in self._episodic._notes:
            episodic_notes.append(
                {
                    "text": note.content,
                    "tags": list(note.tags),
                    "source": note.source,
                    "created_at": note.timestamp.isoformat()
                    if hasattr(note.timestamp, "isoformat")
                    else str(note.timestamp),
                    "note_index": note.note_index,
                    "kind": note.kind,
                }
            )

        return {
            "working": self._working.to_dict(),
            "episodic_notes": episodic_notes,
            "file_summaries": file_summaries,
            "task": self._working.task_summary,
            "files": list(self._working.recent_files),
            "notes": self._episodic.note_texts,
            "next_note_index": self._episodic.next_index,
            "durable_topics": self._durable.topic_slugs() if self._durable is not None else [],
        }

    def _load_dict(self, state: dict) -> "MemoryManager":
        """从旧版 dict 加载状态。"""
        if not isinstance(state, dict):
            return self

        # 加载 working
        working = state.get("working")
        if isinstance(working, dict):
            self._working.load_dict(working)
        else:
            # 回退到顶层兼容字段
            task = str(state.get("task", "")).strip()
            if task:
                self._working.set_task_summary(task)
            for path in state.get("files", []):
                if str(path).strip():
                    self._working.remember_file(str(path).strip())

        # 加载 episodic_notes
        notes_data = state.get("episodic_notes", state.get("notes", []))
        self._episodic.load_dict(notes_data)

        # 加载 file_summaries
        fs_data = state.get("file_summaries", {})
        self._file_summaries.load_dict(fs_data)

        return self


def default_memory_state() -> dict:
    """返回初始记忆状态 dict（兼容旧版格式）。"""
    return {
        "working": {"task_summary": "", "recent_files": []},
        "episodic_notes": [],
        "file_summaries": {},
        "task": "",
        "files": [],
        "notes": [],
        "next_note_index": 0,
    }
