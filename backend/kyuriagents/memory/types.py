"""Shared contracts for dynamic agent memory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

from kyuriagents.rag import ChunkMetadata, DocumentChunk

MemoryScopeType = Literal["user", "project", "team", "global"]
"""Scope categories used to isolate memories."""

MemoryType = Literal[
    "preference",
    "fact",
    "rule",
    "decision",
    "workflow",
    "correction",
    "summary",
]
"""Semantic categories used by memory extraction and retrieval."""

MemoryStatus = Literal["active", "superseded", "deleted"]
"""Lifecycle state for a memory item."""

MemoryVisibility = Literal["private", "team", "public"]
"""Visibility levels for memory retrieval."""


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        msg = f"`{name}` must be between 0.0 and 1.0."
        raise ValueError(msg)


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


@dataclass(frozen=True, kw_only=True)
class MemoryScope:
    """Security and product scope for one memory operation.

    Args:
        tenant_id: Tenant or organization identifier.
        user_id: Optional user identifier for private memories.
        scope_types: Optional scope categories to search.
        scope_ids: Optional concrete scope identifiers to search.
        memory_types: Optional memory categories to search.
        tags: Optional tags that matching memories must include.
        visibility: Optional visibility filter.
        active_only: Whether superseded and deleted memories should be hidden.
    """

    tenant_id: str
    user_id: str | None = None
    scope_types: tuple[MemoryScopeType, ...] = ()
    scope_ids: tuple[str, ...] = ()
    memory_types: tuple[MemoryType, ...] = ()
    tags: tuple[str, ...] = ()
    visibility: MemoryVisibility | None = None
    active_only: bool = True

    def matches(self, memory: MemoryRecord, *, now: datetime | None = None) -> bool:
        """Return whether a memory is visible inside this scope.

        Args:
            memory: Candidate memory record.
            now: Optional reference time for expiration checks.

        Returns:
            `True` when the record satisfies authorization and configured
            filters.
        """
        checks = (
            memory.tenant_id == self.tenant_id,
            not self.active_only or memory.status == "active",
            not memory.is_expired(now=now),
            not self.scope_types or memory.scope_type in self.scope_types,
            not self.scope_ids or memory.scope_id in self.scope_ids,
            not self.memory_types or memory.memory_type in self.memory_types,
            self.visibility is None or memory.visibility == self.visibility,
            not self.tags or all(tag in memory.tags for tag in self.tags),
            self._can_read_owner(memory),
        )
        return all(checks)

    def to_filter_dict(self) -> dict[str, object]:
        """Convert this scope into JSON-serializable storage filters.

        Returns:
            Dictionary suitable for PostgreSQL, Elasticsearch, and Milvus
            adapter implementations.
        """
        data: dict[str, object] = {
            "tenant_id": self.tenant_id,
            "active_only": self.active_only,
        }
        if self.user_id is not None:
            data["user_id"] = self.user_id
        if self.scope_types:
            data["scope_types"] = list(self.scope_types)
        if self.scope_ids:
            data["scope_ids"] = list(self.scope_ids)
        if self.memory_types:
            data["memory_types"] = list(self.memory_types)
        if self.tags:
            data["tags"] = list(self.tags)
        if self.visibility is not None:
            data["visibility"] = self.visibility
        return data

    def _can_read_owner(self, memory: MemoryRecord) -> bool:
        if memory.user_id is not None and memory.user_id != self.user_id:
            return False
        return memory.visibility != "private" or memory.user_id == self.user_id


@dataclass(frozen=True, kw_only=True)
class MemoryRecord:
    """A durable long-term memory item.

    Args:
        memory_id: Stable memory identifier.
        tenant_id: Tenant or organization identifier.
        scope_type: Product scope category.
        scope_id: Concrete scope identifier.
        memory_type: Semantic memory category.
        content: Canonical memory content.
        summary: Optional shorter text for prompts and retrieval snippets.
        user_id: Optional owner for private user memory.
        visibility: Visibility level.
        confidence: Extraction confidence from 0.0 to 1.0.
        importance: Retrieval priority from 0.0 to 1.0.
        tags: Classification tags.
        status: Lifecycle state.
        source_thread_id: Thread where the memory was learned.
        source_message_ids: Message identifiers used as evidence.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 update timestamp.
        expires_at: Optional ISO 8601 expiration timestamp.
        embedding_model: Embedding model identifier.
        embedding_version: Embedding pipeline version.
        schema_version: Metadata schema version.
    """

    memory_id: str
    tenant_id: str
    scope_type: MemoryScopeType
    scope_id: str
    memory_type: MemoryType
    content: str
    summary: str = ""
    user_id: str | None = None
    visibility: MemoryVisibility = "private"
    confidence: float = 1.0
    importance: float = 0.5
    tags: tuple[str, ...] = field(default_factory=tuple)
    status: MemoryStatus = "active"
    source_thread_id: str | None = None
    source_message_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None
    embedding_model: str = ""
    embedding_version: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate numeric metadata after dataclass initialization."""
        if not self.content:
            msg = "`content` must not be empty."
            raise ValueError(msg)
        if not self.scope_id:
            msg = "`scope_id` must not be empty."
            raise ValueError(msg)
        if self.schema_version <= 0:
            msg = "`schema_version` must be positive."
            raise ValueError(msg)
        _validate_ratio("confidence", self.confidence)
        _validate_ratio("importance", self.importance)

    @property
    def index_text(self) -> str:
        """Return text that should be indexed for retrieval."""
        if self.summary:
            return f"{self.summary}\n\n{self.content}"
        return self.content

    @property
    def content_hash(self) -> str:
        """Return a stable hash of the canonical memory content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return whether this memory is past its expiration time.

        Args:
            now: Optional reference time.

        Returns:
            `True` if `expires_at` exists and is earlier than `now`.

        Raises:
            ValueError: If `expires_at` is not a valid ISO 8601 timestamp.
        """
        expires_at = _parse_timestamp(self.expires_at or "")
        if expires_at is None:
            return False
        reference = now or datetime.now(tz=UTC)
        return expires_at <= reference

    def to_rag_metadata(self) -> ChunkMetadata:
        """Convert this memory into RAG-compatible chunk metadata.

        Returns:
            `ChunkMetadata` using `source_type="memory"` so memory can flow
            through the same Elasticsearch, Milvus, fusion, and rerank path as
            document chunks.
        """
        tags = tuple(dict.fromkeys(("memory", self.memory_type, *self.tags)))
        return ChunkMetadata(
            chunk_id=f"memory:{self.memory_id}",
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            kb_id=f"memory:{self.scope_type}:{self.scope_id}",
            doc_id=self.memory_id,
            doc_version=f"{self.memory_id}:v{self.schema_version}",
            chunk_index=0,
            content_hash=self.content_hash,
            source_type="memory",
            source_uri=f"memory://{self.tenant_id}/{self.scope_type}/{self.scope_id}/{self.memory_id}",
            title=f"{self.memory_type} memory",
            section_path=f"{self.scope_type}/{self.scope_id}",
            language="unknown",
            tags=tags,
            visibility=self.visibility,
            created_at=self.created_at,
            updated_at=self.updated_at,
            embedding_model=self.embedding_model,
            embedding_version=self.embedding_version,
            schema_version=self.schema_version,
            is_active=self.status == "active",
        )

    def to_document_chunk(self, *, embedding: tuple[float, ...] = ()) -> DocumentChunk:
        """Convert this memory into a RAG `DocumentChunk`.

        Args:
            embedding: Optional precomputed embedding for vector stores.

        Returns:
            RAG chunk containing indexed memory text and metadata.
        """
        return DocumentChunk(
            text=self.index_text,
            metadata=self.to_rag_metadata(),
            embedding=embedding,
            keywords=tuple(dict.fromkeys((self.memory_type, *self.tags))),
        )

    def with_status(self, status: MemoryStatus) -> MemoryRecord:
        """Return a copy with an updated lifecycle status.

        Args:
            status: Replacement lifecycle state.

        Returns:
            New `MemoryRecord` instance.
        """
        return replace(self, status=status)


@dataclass(frozen=True, kw_only=True)
class MemorySearchResult:
    """A ranked memory search result.

    Args:
        memory: Matched memory record.
        score: Final retrieval score.
        lexical_score: Optional lexical score.
        semantic_score: Optional semantic score.
    """

    memory: MemoryRecord
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None

    @property
    def memory_id(self) -> str:
        """Return the stable memory identifier."""
        return self.memory.memory_id


@dataclass(frozen=True, kw_only=True)
class MemoryWriteCandidate:
    """A proposed memory item produced by extraction.

    Args:
        content: Canonical memory content.
        memory_type: Semantic memory category.
        scope_type: Product scope category.
        scope_id: Concrete scope identifier.
        summary: Optional shorter text for prompts and retrieval snippets.
        visibility: Visibility level.
        confidence: Extraction confidence from 0.0 to 1.0.
        importance: Retrieval priority from 0.0 to 1.0.
        tags: Classification tags.
        source_thread_id: Thread where the memory was learned.
        source_message_ids: Message identifiers used as evidence.
        expires_at: Optional ISO 8601 expiration timestamp.
    """

    content: str
    memory_type: MemoryType
    scope_type: MemoryScopeType
    scope_id: str
    summary: str = ""
    visibility: MemoryVisibility = "private"
    confidence: float = 0.8
    importance: float = 0.5
    tags: tuple[str, ...] = field(default_factory=tuple)
    source_thread_id: str | None = None
    source_message_ids: tuple[str, ...] = field(default_factory=tuple)
    expires_at: str | None = None


class MemoryStore(Protocol):
    """Protocol for durable memory stores."""

    def upsert(self, memory: MemoryRecord) -> MemoryRecord:
        """Create or replace a memory record.

        Args:
            memory: Record to persist.

        Returns:
            Persisted record.
        """
        ...

    def get(self, memory_id: str, *, scope: MemoryScope) -> MemoryRecord | None:
        """Load one memory record when visible to the caller.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            Matching record, or `None`.
        """
        ...

    def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int,
    ) -> list[MemorySearchResult]:
        """Search memory records visible to the caller.

        Args:
            query: User query or rewritten query.
            scope: Tenant and authorization filters.
            limit: Maximum number of records to return.

        Returns:
            Ranked memory search results.
        """
        ...

    def list_memories(self, *, scope: MemoryScope, limit: int = 100) -> list[MemoryRecord]:
        """List visible memory records.

        Args:
            scope: Tenant and authorization filters.
            limit: Maximum number of records to return.

        Returns:
            Visible memory records in store order.
        """
        ...

    def delete(self, memory_id: str, *, scope: MemoryScope) -> bool:
        """Soft delete a visible memory record.

        Args:
            memory_id: Stable memory identifier.
            scope: Tenant and authorization filters.

        Returns:
            `True` when a record was updated.
        """
        ...
