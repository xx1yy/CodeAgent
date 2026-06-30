"""持久化记忆 — 存储在磁盘上的长时记忆主题笔记。

DurableMemoryStore 负责读写 .pico/memory/ 目录下的索引和主题文件。
每条笔记是一个 DurableNote（继承 MemoryItem）。
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import MemoryItem, now, tokenize, parse_timestamp

# ── 预设主题 ─────────────────────────────────────

DURABLE_TOPIC_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "project-conventions": {
        "title": "Project Conventions",
        "summary": "Stable repository conventions.",
        "tags": ["convention"],
    },
    "key-decisions": {
        "title": "Key Decisions",
        "summary": "Long-lived decisions and rationale anchors.",
        "tags": ["decision"],
    },
    "dependency-facts": {
        "title": "Dependency Facts",
        "summary": "Stable dependency and environment facts.",
        "tags": ["dependency"],
    },
    "user-preferences": {
        "title": "User Preferences",
        "summary": "Stable user preferences.",
        "tags": ["preference"],
    },
}


class DurableNote(MemoryItem):
    """持久化记忆笔记项。

    存储在磁盘上，可跨 session 保留。
    """

    memory_type: str = "durable"
    topic: str = ""
    tags: List[str] = []
    kind: str = "durable"


class DurableMemoryStore:
    """持久化记忆存储。

    管理 .pico/memory/ 目录下的主题索引 (MEMORY.md)
    和各主题笔记文件 (topics/<slug>.md)。
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = root
        if root is not None:
            self.index_path = root / "MEMORY.md"
            self.topics_dir = root / "topics"

    # ── 索引读写 ──────────────────────────────────

    def topic_slugs(self) -> List[str]:
        """返回所有已注册的主题 slug。"""
        return [topic["topic"] for topic in self.load_index()]

    def load_index(self) -> List[Dict[str, Any]]:
        """加载 MEMORY.md 索引文件。"""
        if not self._index_exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").splitlines()
        topics: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for raw in lines:
            line = raw.strip()
            match = re.match(r"- \[([^\]]+)\]\([^)]+\):\s*(.+)", line)
            if match:
                current = {
                    "topic": match.group(1).strip(),
                    "title": match.group(2).strip(),
                    "summary": "",
                    "tags": [],
                }
                topics.append(current)
                continue
            if current is None:
                continue
            summary_match = re.match(r"- summary:\s*(.+)", line)
            if summary_match:
                current["summary"] = summary_match.group(1).strip()
                continue
            tags_match = re.match(r"- tags:\s*(.+)", line)
            if tags_match:
                current["tags"] = [tag.strip() for tag in tags_match.group(1).split(",") if tag.strip()]
        return topics

    def _write_index(self, topics: List[Dict[str, Any]]) -> None:
        """写入 MEMORY.md 索引文件。"""
        self.root.mkdir(parents=True, exist_ok=True)
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        lines = ["# Durable Memory Index", ""]
        for topic in topics:
            lines.append(f"- [{topic['topic']}](topics/{topic['topic']}.md): {topic['title']}")
            lines.append(f"  - summary: {topic['summary']}")
            lines.append(f"  - tags: {', '.join(topic['tags'])}")
        self.index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _index_exists(self) -> bool:
        return self.root is not None and self.index_path.exists()

    # ── 主题笔记读写 ──────────────────────────────

    def load_topic_notes(self, topic: str) -> List[DurableNote]:
        """加载指定主题的所有笔记。"""
        if self.root is None:
            return []
        path = self.topics_dir / f"{topic}.md"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        notes: List[DurableNote] = []
        capture = False
        updated_at = ""
        tags: List[str] = []
        for raw in lines:
            line = raw.strip()
            if line.startswith("- tags:"):
                tags = [tag.strip() for tag in line.split(":", 1)[1].split(",") if tag.strip()]
            elif line.startswith("- updated_at:"):
                updated_at = line.split(":", 1)[1].strip()
            elif line == "## Notes":
                capture = True
            elif capture and line.startswith("- "):
                text = line[2:].strip()
                if text:
                    notes.append(
                        DurableNote(
                            content=text,
                            topic=topic,
                            tags=list(tags),
                            timestamp=updated_at or now(),
                            kind="durable",
                        )
                    )
        return notes

    def _write_topic(self, topic: str, note_texts: List[str]) -> None:
        """将主题笔记列表写回磁盘文件。"""
        if self.root is None:
            return
        meta = DURABLE_TOPIC_DEFAULTS.get(topic, {})
        lines = [
            f"# {meta.get('title', topic)}",
            "",
            f"- topic: {topic}",
            f"- summary: {meta.get('summary', '')}",
            f"- tags: {', '.join(meta.get('tags', []))}",
            f"- updated_at: {now()}",
            "",
            "## Notes",
        ]
        for note in note_texts:
            lines.append(f"- {note}")
        self.topics_dir.mkdir(parents=True, exist_ok=True)
        (self.topics_dir / f"{topic}.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )

    # ── 检索 ─────────────────────────────────────

    def retrieval_candidates(self, query: str, limit: int = 3) -> List[DurableNote]:
        """从所有主题中召回与 query 相关的笔记。"""
        if self.root is None:
            return []
        query_tokens = tokenize(query)
        ranked: List[Tuple[tuple, DurableNote]] = []
        for topic in self.load_index():
            notes = self.load_topic_notes(topic["topic"])
            for note in notes:
                note_tags = {tag.lower() for tag in note.tags}
                note_tokens = tokenize(note.content) | tokenize(
                    topic.get("title", "")
                ) | note_tags
                exact_tag_match = int(bool(query_tokens & note_tags))
                keyword_overlap = len(query_tokens & note_tokens)
                if exact_tag_match == 0 and keyword_overlap == 0:
                    continue
                recency = parse_timestamp(str(note.timestamp))
                ranked.append(((exact_tag_match, keyword_overlap, recency), note))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in ranked[:limit]]

    # ── 晋升 ─────────────────────────────────────

    @staticmethod
    def _subject_key(text: str) -> Optional[str]:
        """提取文本的主语，用于按主语去重。"""
        text = str(text).strip()
        patterns = (
            r"^(.+?)\s+is\s+.+$",
            r"^(.+?)\s+are\s+.+$",
            r"^(.+?)\s+uses?\s+.+$",
            r"^(.+?)\s+should\s+.+$",
            r"^(.+?)是.+$",
            r"^(.+?)使用.+$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, re.I)
            if match:
                subject = " ".join(tokenize(match.group(1)))
                return subject or None
        return None

    def promote(self, promotions: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
        """将笔记晋升为持久化记忆。

        Args:
            promotions: [(topic, note_text), ...] 列表

        Returns:
            (promoted, superseded): 成功晋升和被替换的笔记列表
        """
        if not promotions or self.root is None:
            return [], []

        topics = {topic["topic"]: topic for topic in self.load_index()}
        topic_notes: Dict[str, List[str]] = {
            slug: [note.content for note in self.load_topic_notes(slug)]
            for slug in topics
        }
        results: List[str] = []
        superseded: List[str] = []

        for topic, note_text in promotions:
            meta = DURABLE_TOPIC_DEFAULTS.get(topic, {})
            topics.setdefault(
                topic,
                {
                    "topic": topic,
                    "title": meta.get("title", topic),
                    "summary": meta.get("summary", ""),
                    "tags": list(meta.get("tags", [])),
                },
            )
            existing = topic_notes.setdefault(topic, [])
            if note_text in existing:
                continue

            new_subject = self._subject_key(note_text)
            replaced = False
            if new_subject:
                for index, old_text in enumerate(list(existing)):
                    if self._subject_key(old_text) == new_subject:
                        superseded.append(f"{topic}: {old_text} -> {note_text}")
                        existing[index] = note_text
                        replaced = True
                        break
            if not replaced:
                existing.append(note_text)
            results.append(f"{topic}: {note_text}")

        self._write_index([topics[slug] for slug in sorted(topics)])
        for slug, notes in topic_notes.items():
            self._write_topic(slug, notes)
        return results, superseded
