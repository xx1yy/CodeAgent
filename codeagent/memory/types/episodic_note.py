"""情景笔记 — 跨轮的标签化短时文本记录。

每条笔记是一个 EpisodicNote（继承 MemoryItem），
带 tags、source、kind 元信息，使用 FIFO 淘汰策略。
"""

from typing import Any, Dict, List, Optional, Tuple

from ..base import (
    EPISODIC_NOTE_LIMIT,
    MemoryItem,
    clip,
    dedupe_preserve_order,
    ensure_list,
    now,
)


class EpisodicNote(MemoryItem):
    """情景笔记记忆项。

    对应原始 memory.py 中 episodic_notes 列表里的每条笔记。
    """

    memory_type: str = "episodic"
    tags: List[str] = []
    source: str = ""
    note_index: int = 0
    kind: str = "episodic"


class EpisodicNoteStore:
    """情景笔记存储。

    管理带 tag 的笔记列表，支持追加、去重、FIFO 淘汰。
    """

    def __init__(self):
        self._notes: List[EpisodicNote] = []
        self._next_index: int = 0

    # ── 读 ─────────────────────────────────────────

    @property
    def notes(self) -> List[EpisodicNote]:
        return list(self._notes)

    @property
    def note_texts(self) -> List[str]:
        return [note.content for note in self._notes]

    @property
    def next_index(self) -> int:
        return self._next_index

    def __len__(self) -> int:
        return len(self._notes)

    def is_empty(self) -> bool:
        return len(self._notes) == 0

    # ── 写 ─────────────────────────────────────────

    def append(
        self,
        text: str,
        tags: Tuple[str, ...] = (),
        source: str = "",
        created_at: Optional[str] = None,
        kind: str = "episodic",
    ) -> "EpisodicNoteStore":
        """追加一条情景笔记。按内容去重，超过上限时淘汰最旧的。"""
        text = clip(str(text).strip(), 500)
        if not text:
            return self

        note = EpisodicNote(
            content=text,
            tags=dedupe_preserve_order(
                [str(tag).strip() for tag in ensure_list(tags) if str(tag).strip()]
            ),
            source=str(source).strip(),
            timestamp=created_at or now(),
            note_index=self._next_index,
            kind=str(kind).strip() or "episodic",
        )
        self._next_index += 1

        # 按内容去重：移除相同 text 的旧条目
        self._notes = [n for n in self._notes if n.content != note.content]
        self._notes.append(note)
        self._notes = self._notes[-EPISODIC_NOTE_LIMIT:]

        return self

    # ── 序列化 ─────────────────────────────────────

    def to_dict(self) -> List[Dict[str, Any]]:
        return [note.model_dump() for note in self._notes]

    def text_list(self) -> List[str]:
        return [note.content for note in self._notes]

    def load_dict(self, data: Any) -> "EpisodicNoteStore":
        """从原始 dict 列表加载（兼容旧版 dict 格式）。"""
        if not isinstance(data, list):
            return self

        notes: List[EpisodicNote] = []
        max_index = -1
        for item in data:
            if isinstance(item, str):
                item_text = str(item).strip()
                if not item_text:
                    continue
                note = EpisodicNote(
                    content=clip(item_text, 500),
                    note_index=self._next_index,
                )
                self._next_index += 1
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                note = EpisodicNote(
                    content=clip(text, 500),
                    tags=dedupe_preserve_order(
                        [str(t).strip() for t in ensure_list(item.get("tags", [])) if str(t).strip()]
                    ),
                    source=str(item.get("source", "")).strip(),
                    timestamp=str(item.get("created_at", "")).strip() or now(),
                    note_index=int(item.get("note_index", self._next_index)),
                    kind=str(item.get("kind", "episodic")).strip() or "episodic",
                )
                self._next_index = max(self._next_index, note.note_index + 1)
            else:
                continue
            notes.append(note)

        notes = notes[-EPISODIC_NOTE_LIMIT:]
        self._notes = notes
        return self
