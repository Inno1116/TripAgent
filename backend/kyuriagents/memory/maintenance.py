"""Rule-based long-term memory maintenance."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kyuriagents.memory.types import MemoryRecord, MemoryScope

if TYPE_CHECKING:
    from kyuriagents.memory.service import MemoryService

_DEFAULT_MIN_GROUP_SIZE = 3
_DEFAULT_MAX_SUMMARY_CHARS = 1_200
_SUMMARY_TAG = "compressed"
_TRUNCATION_SUFFIX = "...[truncated]"


@dataclass(frozen=True, kw_only=True)
class MemoryCompactionConfig:
    """Controls for rule-based memory compaction.

    Args:
        min_group_size: Minimum active memories in one group before compaction.
        max_summary_chars: Maximum characters stored in the compacted summary.
    """

    min_group_size: int = _DEFAULT_MIN_GROUP_SIZE
    max_summary_chars: int = _DEFAULT_MAX_SUMMARY_CHARS

    def __post_init__(self) -> None:
        """Validate compaction settings."""
        if self.min_group_size <= 1:
            msg = "`min_group_size` must be greater than 1."
            raise ValueError(msg)
        if self.max_summary_chars <= 0:
            msg = "`max_summary_chars` must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class MemoryCompactionResult:
    """Result of one memory compaction pass.

    Args:
        created: New compacted memories.
        superseded_ids: Source memories marked as superseded.
        inspected_count: Number of active memories inspected.
    """

    created: tuple[MemoryRecord, ...]
    superseded_ids: tuple[str, ...]
    inspected_count: int


class MemoryMaintenanceService:
    """Maintain memory quality with deterministic compaction rules."""

    def __init__(
        self,
        service: MemoryService,
        *,
        config: MemoryCompactionConfig | None = None,
    ) -> None:
        """Initialize the maintenance service.

        Args:
            service: Memory service used for listing and upserting records.
            config: Optional compaction configuration.
        """
        self._service = service
        self._config = config or MemoryCompactionConfig()

    def compact_scope(self, scope: MemoryScope, *, limit: int = 1_000) -> MemoryCompactionResult:
        """Compact active memories visible in a scope.

        Args:
            scope: Scope whose memories should be compacted.
            limit: Maximum memories to inspect.

        Returns:
            Compaction result with created and superseded records.
        """
        memories = [memory for memory in self._service.list_memories(scope=scope, limit=limit) if memory.status == "active"]
        created: list[MemoryRecord] = []
        superseded: list[str] = []
        for group in _groups(memories):
            if len(group) < self._config.min_group_size:
                continue
            compacted = _compact_group(group, max_summary_chars=self._config.max_summary_chars)
            self._service.upsert_record(compacted)
            created.append(compacted)
            for memory in group:
                self._service.upsert_record(memory.with_status("superseded"))
                superseded.append(memory.memory_id)
        return MemoryCompactionResult(
            created=tuple(created),
            superseded_ids=tuple(superseded),
            inspected_count=len(memories),
        )


def _groups(memories: list[MemoryRecord]) -> list[list[MemoryRecord]]:
    grouped: dict[tuple[str, str, str, str | None], list[MemoryRecord]] = {}
    for memory in memories:
        if _SUMMARY_TAG in memory.tags:
            continue
        key = (memory.scope_type, memory.scope_id, memory.memory_type, memory.user_id)
        grouped.setdefault(key, []).append(memory)
    return list(grouped.values())


def _compact_group(group: list[MemoryRecord], *, max_summary_chars: int) -> MemoryRecord:
    ordered = sorted(group, key=lambda memory: (memory.importance, memory.confidence, memory.updated_at), reverse=True)
    head = ordered[0]
    summary = _truncate(" ".join(_memory_text(memory) for memory in ordered), max_summary_chars)
    content = f"Compacted {len(group)} {head.memory_type} memories. {summary}"
    return MemoryRecord(
        memory_id=_compacted_id(ordered),
        tenant_id=head.tenant_id,
        user_id=head.user_id,
        scope_type=head.scope_type,
        scope_id=head.scope_id,
        memory_type=head.memory_type,
        content=content,
        summary=summary,
        visibility=head.visibility,
        confidence=min(memory.confidence for memory in ordered),
        importance=max(memory.importance for memory in ordered),
        tags=tuple(dict.fromkeys((_SUMMARY_TAG, head.memory_type, *(tag for memory in ordered for tag in memory.tags)))),
        source_thread_id=head.source_thread_id,
        source_message_ids=tuple(dict.fromkeys(message_id for memory in ordered for message_id in memory.source_message_ids)),
        embedding_model=head.embedding_model,
        embedding_version=head.embedding_version,
        schema_version=head.schema_version,
    )


def _memory_text(memory: MemoryRecord) -> str:
    return memory.summary or memory.content


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    if limit <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:limit]
    return normalized[: limit - len(_TRUNCATION_SUFFIX)].rstrip() + _TRUNCATION_SUFFIX


def _compacted_id(memories: list[MemoryRecord]) -> str:
    joined = "|".join(memory.memory_id for memory in memories)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
    return f"mem_compact_{digest}"


__all__ = [
    "MemoryCompactionConfig",
    "MemoryCompactionResult",
    "MemoryMaintenanceService",
]
