"""检索与渲染 — 从各记忆层联合召回并格式化为模型可读文本。"""

from typing import List, Optional

from .base import (
    RELEVANT_MEMORY_LIMIT,
    FILE_SUMMARY_LIMIT,
    MemoryItem,
    file_freshness,
    parse_timestamp,
    tokenize,
)
from .types.episodic_note import EpisodicNoteStore
from .types.durable_note import DurableMemoryStore
from .types.working_context import WorkingContext
from .types.file_summary import FileSummaryStore


def retrieval_candidates(
    episodic_store: EpisodicNoteStore,
    durable_store: Optional[DurableMemoryStore],
    query: str,
    limit: int = RELEVANT_MEMORY_LIMIT,
) -> List[MemoryItem]:
    """从情景笔记 + 持久化记忆两层联合召回。

    排序策略（从高到低）：
        1. tag 精确命中 > 未命中
        2. 关键词重叠数
        3. 时间新鲜度（情景笔记优先于持久化笔记）
    """
    query_tokens = tokenize(query)
    ranked: list = []

    # 情景层
    for note in episodic_store.notes:
        note_tags = {tag.lower() for tag in note.tags}
        note_tokens = tokenize(note.content) | tokenize(note.source) | note_tags
        exact_tag_match = int(bool(query_tokens & note_tags))
        keyword_overlap = len(query_tokens & note_tokens)
        if exact_tag_match == 0 and keyword_overlap == 0:
            continue
        recency = parse_timestamp(str(note.timestamp))
        ranked.append(((exact_tag_match, keyword_overlap, recency, note.note_index), note))

    # 持久化层
    if durable_store is not None and durable_store.root is not None:
        for note in durable_store.retrieval_candidates(query, limit=limit):
            note_tags = {tag.lower() for tag in note.tags}
            note_tokens = tokenize(note.content) | note_tags
            exact_tag_match = int(bool(query_tokens & note_tags))
            keyword_overlap = len(query_tokens & note_tokens)
            recency = parse_timestamp(str(note.timestamp))
            # note_index = -1 表示来自持久化层，排在相同匹配度的情景笔记之后
            ranked.append(((exact_tag_match, keyword_overlap, recency, -1), note))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [note for _, note in ranked[:limit]]


def retrieval_view(
    episodic_store: EpisodicNoteStore,
    durable_store: Optional[DurableMemoryStore],
    query: str,
    limit: int = RELEVANT_MEMORY_LIMIT,
) -> str:
    """返回格式化的召回文本，供注入 prompt。"""
    candidates = retrieval_candidates(episodic_store, durable_store, query, limit=limit)
    lines = ["Relevant memory:"]
    if not candidates:
        lines.append("- none")
    else:
        for note in candidates:
            lines.append(f"- {note.content}")
    return "\n".join(lines)


def render_memory_text(
    working: WorkingContext,
    file_store: FileSummaryStore,
    episodic_store: EpisodicNoteStore,
    durability_store: Optional[DurableMemoryStore],
    workspace_root: Optional[str] = None,
) -> str:
    """渲染完整记忆仪表盘（给模型看的紧凑摘要）。

    笔记正文默认不展开，只在相关召回时才按需拿出来。
    """
    lines = [
        "Memory:",
        f"- task: {working.task_summary or '-'}",
        f"- recent_files: {', '.join(working.recent_files) or '-'}",
    ]

    # 文件摘要：只展示最近文件中仍然新鲜的
    summaries = []
    for summary in file_store.get_fresh(working.recent_files):
        summaries.append(f"- {summary.file_path}: {summary.content}")
    if summaries:
        lines.append("- file_summaries:")
        lines.extend(f"  {line}" for line in summaries)
    else:
        lines.append("- file_summaries: -")

    lines.append(f"- episodic_notes: {len(episodic_store)}")

    durable_topics = durability_store.topic_slugs() if durability_store is not None else []
    lines.append(f"- durable_topics: {', '.join(durable_topics) or '-'}")

    return "\n".join(lines)


def is_effectively_empty(
    working: WorkingContext,
    file_store: FileSummaryStore,
    episodic_store: EpisodicNoteStore,
) -> bool:
    """判断记忆是否为空（没有任何有效内容）。"""
    return (
        not working.task_summary.strip()
        and not working.recent_files
        and episodic_store.is_empty()
        and file_store.is_empty()
    )
