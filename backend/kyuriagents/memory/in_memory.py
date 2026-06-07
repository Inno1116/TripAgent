"""In-memory dynamic memory store for local development and unit tests."""

from __future__ import annotations

import re
from dataclasses import replace

from kyuriagents.memory.types import MemoryRecord, MemoryScope, MemorySearchResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}


def _lexical_score(query: str, memory: MemoryRecord) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    memory_tokens = _tokens(f"{memory.index_text} {' '.join(memory.tags)} {memory.memory_type}")
    if not memory_tokens:
        return 0.0
    return len(query_tokens & memory_tokens) / len(query_tokens)


class InMemoryMemoryStore:
    """Simple `MemoryStore` implementation backed by a dictionary."""

    def __init__(self, memories: list[MemoryRecord] | None = None) -> None:
        """Initialize the store.

        Args:
            memories: Optional seed records.
        """
        self._memories: dict[str, MemoryRecord] = {}
        for memory in memories or []:
            self.upsert(memory)

    def upsert(self, memory: MemoryRecord) -> MemoryRecord:
        """Create or replace a memory record.

        Args:
            memory: Record to persist.

        Returns:
            Persisted record.
        """
        self._memories[memory.memory_id] = memory
        return memory

    def get(self, memory_id: str, *, scope: MemoryScope) -> MemoryRecord | None:
        """Load one memory record when visible to the caller.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            Matching record, or `None`.
        """
        memory = self._memories.get(memory_id)
        if memory is None or not scope.matches(memory):
            return None
        return memory

    def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int,
    ) -> list[MemorySearchResult]:
        """Search visible memory records.

        Args:
            query: User query or rewritten query.
            scope: Tenant and authorization filters.
            limit: Maximum number of records to return.

        Returns:
            Ranked memory results.
        """
        if limit <= 0:
            msg = "`limit` must be positive."
            raise ValueError(msg)

        results: list[MemorySearchResult] = []
        for memory in self._memories.values():
            if not scope.matches(memory):
                continue
            lexical = _lexical_score(query, memory)
            if query and lexical == 0.0:
                continue
            score = lexical + (memory.importance * 0.10) + (memory.confidence * 0.05)
            results.append(MemorySearchResult(memory=memory, score=score, lexical_score=lexical))

        results.sort(key=lambda result: (result.score, result.memory.updated_at, result.memory.memory_id), reverse=True)
        return results[:limit]

    def list_memories(self, *, scope: MemoryScope, limit: int = 100) -> list[MemoryRecord]:
        """List visible memory records.

        Args:
            scope: Tenant and authorization filters.
            limit: Maximum number of records to return.

        Returns:
            Visible records in insertion order.
        """
        if limit <= 0:
            msg = "`limit` must be positive."
            raise ValueError(msg)
        return [memory for memory in self._memories.values() if scope.matches(memory)][:limit]

    def delete(self, memory_id: str, *, scope: MemoryScope) -> bool:
        """Soft delete a visible memory record.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            `True` when a record was updated.
        """
        memory = self.get(memory_id, scope=scope)
        if memory is None:
            return False
        self._memories[memory_id] = replace(memory, status="deleted")
        return True
