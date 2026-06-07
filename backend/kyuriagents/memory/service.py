"""High-level helpers for dynamic memory operations."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kyuriagents.memory.compression import MemoryContextBudget, MemoryContextCompressor
from kyuriagents.memory.types import MemoryRecord, MemoryScope, MemorySearchResult, MemoryStore, MemoryWriteCandidate

if TYPE_CHECKING:
    from kyuriagents.memory.indexing import MemoryHybridSearcher, MemoryIndexer

_LOGGER = logging.getLogger(__name__)


def format_memory_context(
    results: list[MemorySearchResult],
    *,
    compressor: MemoryContextCompressor | None = None,
) -> str:
    """Format retrieved memories for prompt injection.

    Args:
        results: Ranked memory search results.
        compressor: Optional compressor used before formatting.

    Returns:
        XML-like context block containing only relevant memories.
    """
    if not results:
        return "<agent_long_term_memory>\n(No relevant long-term memory found.)\n</agent_long_term_memory>"

    compressed = (compressor or MemoryContextCompressor()).compress(results)
    lines = ["<agent_long_term_memory>"]
    for result in compressed.results:
        memory = result.memory
        if memory.summary and memory.summary.strip() != memory.content.strip():
            text = f"{memory.summary}: {memory.content}"
        else:
            text = memory.content or memory.summary
        lines.append(
            f"- [{memory.memory_type}; scope={memory.scope_type}/{memory.scope_id}; "
            f"confidence={memory.confidence:.2f}; importance={memory.importance:.2f}] {text}"
        )
    if compressed.omitted_count:
        lines.append(f"- [omitted] {compressed.omitted_count} additional memories were omitted by the memory context budget.")
    if compressed.truncated_count:
        lines.append(f"- [truncated] {compressed.truncated_count} memories were shortened to fit the memory context budget.")
    lines.append("</agent_long_term_memory>")
    return "\n".join(lines)


class MemoryService:
    """Small orchestration layer around a `MemoryStore`.

    LangMem or another extractor can produce `MemoryWriteCandidate` instances;
    this service stamps tenant, owner, and audit metadata before persistence.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        compressor: MemoryContextCompressor | None = None,
        hybrid_searcher: MemoryHybridSearcher | None = None,
        indexer: MemoryIndexer | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            store: Durable memory store.
            compressor: Optional memory context compressor.
            hybrid_searcher: Optional hybrid memory searcher.
            indexer: Optional retrieval index synchronizer.
        """
        self._store = store
        self._compressor = compressor or MemoryContextCompressor()
        self._hybrid_searcher = hybrid_searcher
        self._indexer = indexer

    def save_candidate(
        self,
        candidate: MemoryWriteCandidate,
        *,
        tenant_id: str,
        user_id: str | None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        """Persist one extracted memory candidate.

        Args:
            candidate: Proposed memory item.
            tenant_id: Tenant or organization identifier.
            user_id: Optional owner for private user memory.
            memory_id: Optional stable identifier supplied by an upstream store.

        Returns:
            Persisted memory record.
        """
        now = datetime.now(tz=UTC).isoformat()
        record = MemoryRecord(
            memory_id=memory_id or f"mem_{uuid.uuid4().hex}",
            tenant_id=tenant_id,
            user_id=user_id,
            scope_type=candidate.scope_type,
            scope_id=candidate.scope_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            summary=candidate.summary,
            visibility=candidate.visibility,
            confidence=candidate.confidence,
            importance=candidate.importance,
            tags=candidate.tags,
            source_thread_id=candidate.source_thread_id,
            source_message_ids=candidate.source_message_ids,
            created_at=now,
            updated_at=now,
            expires_at=candidate.expires_at,
        )
        return self.upsert_record(record)

    def upsert_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist one memory record and update the retrieval index.

        Args:
            record: Memory record.

        Returns:
            Persisted memory record.
        """
        saved = self._store.upsert(record)
        if self._indexer is not None:
            try:
                self._indexer.upsert([saved])
            except Exception as exc:  # noqa: BLE001  # memory store remains authoritative when retrieval indexes lag
                _LOGGER.warning("Failed to index long-term memory `%s`: %s", saved.memory_id, exc)
        return saved

    def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Search memory records visible to a caller.

        Args:
            query: User query or rewritten query.
            scope: Tenant and authorization filters.
            limit: Maximum number of records to return.

        Returns:
            Ranked memory results.
        """
        if self._hybrid_searcher is not None:
            try:
                results = self._hybrid_searcher.search(query, scope=scope, limit=limit)
                if results:
                    return results
            except Exception as exc:  # noqa: BLE001  # fallback to PostgreSQL when ES/Milvus is unavailable
                _LOGGER.warning("Hybrid memory search failed; falling back to PostgreSQL memory search: %s", exc)
        return self._store.search(query, scope=scope, limit=limit)

    def build_context(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int = 5,
    ) -> str:
        """Retrieve relevant memories and format them for prompt injection.

        Args:
            query: User query or rewritten query.
            scope: Tenant and authorization filters.
            limit: Maximum number of memories to include.

        Returns:
            Prompt-ready long-term memory context block.
        """
        return format_memory_context(self.search(query, scope=scope, limit=limit), compressor=self._compressor)

    def get(self, memory_id: str, *, scope: MemoryScope) -> MemoryRecord | None:
        """Load one visible memory record.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            Matching record, or `None`.
        """
        return self._store.get(memory_id, scope=scope)

    def list_memories(self, *, scope: MemoryScope, limit: int = 100) -> list[MemoryRecord]:
        """List memories visible to the caller.

        Args:
            scope: Tenant and authorization filters.
            limit: Maximum records to return.

        Returns:
            Visible memory records.
        """
        return self._store.list_memories(scope=scope, limit=limit)

    def delete(self, memory_id: str, *, scope: MemoryScope) -> bool:
        """Soft delete one visible memory record.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            `True` when a record was updated.
        """
        deleted = self._store.delete(memory_id, scope=scope)
        if deleted and self._indexer is not None:
            try:
                self._indexer.delete([memory_id])
            except Exception as exc:  # noqa: BLE001  # deleted memory remains hidden by PostgreSQL status filters
                _LOGGER.warning("Failed to delete long-term memory `%s` from retrieval indexes: %s", memory_id, exc)
        return deleted

    @property
    def memory_context_budget(self) -> MemoryContextBudget:
        """Return the active memory context budget."""
        return self._compressor.budget
