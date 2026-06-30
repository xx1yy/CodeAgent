"""文件摘要 — 文件内容的短摘要缓存，附带新鲜度校验。

每条摘要是一个 FileSummary（继承 MemoryItem），
通过 SHA256 内容指纹判断文件是否已变更。
"""

from typing import Any, Dict, List, Optional

from ..base import FILE_SUMMARY_LIMIT, MemoryItem, clip, file_freshness, now


class FileSummary(MemoryItem):
    """文件摘要记忆项。

    保存文件内容的短摘要，以及写入时的文件 SHA256 指纹。
    """

    memory_type: str = "file_summary"
    file_path: str = ""
    freshness: Optional[str] = None


class FileSummaryStore:
    """文件摘要存储。

    管理文件摘要的读写、新鲜度校验和惰性失效。
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self._workspace_root = workspace_root
        self._summaries: Dict[str, FileSummary] = {}

    # ── 读 ─────────────────────────────────────────

    def get(self, path: str) -> Optional[FileSummary]:
        return self._summaries.get(path)

    def get_fresh(self, recent_files: list) -> List[FileSummary]:
        """返回最新 N 个文件中仍然新鲜的摘要。"""
        result = []
        for path in recent_files[:FILE_SUMMARY_LIMIT]:
            summary = self._summaries.get(path)
            if summary is None:
                continue
            if summary.freshness != file_freshness(path, self._workspace_root):
                continue
            result.append(summary)
        return result

    def __contains__(self, path: str) -> bool:
        return path in self._summaries

    def __len__(self) -> int:
        return len(self._summaries)

    def is_empty(self) -> bool:
        return len(self._summaries) == 0

    # ── 写 ─────────────────────────────────────────

    def set(self, path: str, summary: str) -> "FileSummaryStore":
        """设置一条文件摘要，同时记录当前文件新鲜度指纹。"""
        path = str(path).strip()
        summary = clip(str(summary).strip(), 500)
        if not path or not summary:
            return self

        self._summaries[path] = FileSummary(
            content=summary,
            file_path=path,
            freshness=file_freshness(path, self._workspace_root),
            timestamp=now(),
        )
        return self

    def invalidate(self, path: str) -> "FileSummaryStore":
        """主动移除指定文件的摘要。"""
        path = str(path).strip()
        self._summaries.pop(path, None)
        return self

    def invalidate_stale(self) -> List[str]:
        """惰性失效：对比所有摘要的 freshness 与文件当前 SHA256，不一致则移除。

        返回被失效的路径列表。
        """
        invalidated: List[str] = []
        for path, summary in list(self._summaries.items()):
            current = file_freshness(path, self._workspace_root)
            if summary.freshness == current:
                continue
            invalidated.append(path)
            del self._summaries[path]
        return invalidated

    # ── 序列化 ─────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            path: {
                "summary": s.content,
                "created_at": s.timestamp.isoformat() if hasattr(s.timestamp, "isoformat") else str(s.timestamp),
                "freshness": s.freshness,
            }
            for path, s in self._summaries.items()
        }

    def load_dict(self, data: Any) -> "FileSummaryStore":
        """从旧版 dict 格式加载。"""
        if not isinstance(data, dict):
            return self
        for path, raw in data.items():
            path = str(path).strip()
            if not path:
                continue
            if isinstance(raw, dict):
                text = clip(str(raw.get("summary", "")).strip(), 500)
                if not text:
                    continue
                freshness = raw.get("freshness")
                freshness = None if freshness in (None, "") else str(freshness).strip() or None
                self._summaries[path] = FileSummary(
                    content=text,
                    file_path=path,
                    freshness=freshness,
                    timestamp=str(raw.get("created_at", "")).strip() or now(),
                )
            elif isinstance(raw, str):
                text = clip(str(raw).strip(), 500)
                if not text:
                    continue
                self._summaries[path] = FileSummary(
                    content=text,
                    file_path=path,
                    timestamp=now(),
                )
        return self

