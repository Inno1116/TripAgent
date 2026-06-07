"""Metadata contracts for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Visibility = Literal["private", "team", "public"]
"""Visibility values used by retrieval filters and storage schemas."""


@dataclass(frozen=True, kw_only=True)
class RetrievalScope:
    """Security and product scope for one retrieval request.

    `tenant_id` is intentionally required so future multi-tenant deployments do
    not need a breaking API change. Single-tenant applications can use a stable
    value such as `"default"`.

    Args:
        tenant_id: Tenant or organization identifier.
        user_id: Optional user identifier for private knowledge bases.
        kb_ids: Optional knowledge-base identifiers to search.
        doc_ids: Optional document identifiers to restrict retrieval.
        languages: Optional ISO language tags to restrict retrieval.
        source_types: Optional document source types, such as `pdf` or `html`.
        tags: Optional tags that matching chunks must include.
        visibility: Optional visibility filter.
        active_only: Whether soft-deleted or inactive chunks should be hidden.
    """

    tenant_id: str
    user_id: str | None = None
    kb_ids: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    visibility: Visibility | None = None
    active_only: bool = True

    def matches(self, metadata: ChunkMetadata) -> bool:
        """Return whether a chunk belongs inside this retrieval scope.

        Args:
            metadata: Candidate chunk metadata.

        Returns:
            `True` when the chunk satisfies every configured filter.
        """
        checks = (
            metadata.tenant_id == self.tenant_id,
            metadata.user_id is None or metadata.user_id == self.user_id,
            not self.kb_ids or metadata.kb_id in self.kb_ids,
            not self.doc_ids or metadata.doc_id in self.doc_ids,
            not self.languages or metadata.language in self.languages,
            not self.source_types or metadata.source_type in self.source_types,
            self.visibility is None or metadata.visibility == self.visibility,
            not self.active_only or metadata.is_active,
            not self.tags or all(tag in metadata.tags for tag in self.tags),
        )
        return all(checks)

    def to_filter_dict(self) -> dict[str, object]:
        """Convert the scope into scalar filters for storage adapters.

        Returns:
            Dictionary using JSON-serializable values suitable for Milvus and
            Elasticsearch adapter implementations.
        """
        data: dict[str, object] = {
            "tenant_id": self.tenant_id,
            "active_only": self.active_only,
        }
        if self.user_id is not None:
            data["user_id"] = self.user_id
        if self.kb_ids:
            data["kb_ids"] = list(self.kb_ids)
        if self.doc_ids:
            data["doc_ids"] = list(self.doc_ids)
        if self.languages:
            data["languages"] = list(self.languages)
        if self.source_types:
            data["source_types"] = list(self.source_types)
        if self.tags:
            data["tags"] = list(self.tags)
        if self.visibility is not None:
            data["visibility"] = self.visibility
        return data


@dataclass(frozen=True, kw_only=True)
class ChunkMetadata:
    """Metadata attached to a single indexed chunk.

    The fields are intentionally storage-friendly: commonly filtered values are
    scalar columns for Milvus and Elasticsearch, while richer document ownership
    and lifecycle state belongs in PostgreSQL.

    Args:
        chunk_id: Stable chunk identifier.
        tenant_id: Tenant or organization identifier.
        kb_id: Knowledge-base identifier.
        doc_id: Source document identifier.
        doc_version: Source document version identifier.
        chunk_index: Zero-based chunk position in the document version.
        content_hash: Hash of normalized chunk text.
        source_type: Source type, such as `pdf`, `html`, `md`, or `docx`.
        source_uri: Original source URI or storage path.
        title: Source title or filename.
        section_path: Heading path within the source document.
        page_start: First source page covered by the chunk, if known.
        page_end: Last source page covered by the chunk, if known.
        char_start: Start character offset in normalized source text, if known.
        char_end: End character offset in normalized source text, if known.
        language: ISO language tag.
        tags: Classification tags.
        visibility: Visibility level.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 update timestamp.
        embedding_model: Embedding model identifier.
        embedding_version: Embedding pipeline version.
        schema_version: Metadata schema version.
        is_active: Whether the chunk should be visible to retrieval.
        user_id: Optional owner for private user uploads.
    """

    chunk_id: str
    tenant_id: str
    kb_id: str
    doc_id: str
    doc_version: str
    chunk_index: int
    content_hash: str
    source_type: str
    source_uri: str
    title: str = ""
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    language: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    visibility: Visibility = "private"
    created_at: str = ""
    updated_at: str = ""
    embedding_model: str = ""
    embedding_version: str = ""
    schema_version: int = 1
    is_active: bool = True
    user_id: str | None = None

    def to_milvus_fields(self) -> dict[str, object]:
        """Return scalar fields to store next to a Milvus vector.

        Returns:
            Dictionary containing filterable Milvus fields. The text content and
            vector are intentionally not included.
        """
        return {
            "chunk_id": self.chunk_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id or "",
            "kb_id": self.kb_id,
            "doc_id": self.doc_id,
            "doc_version": self.doc_version,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "title": self.title,
            "section_path": self.section_path,
            "page_start": self.page_start or 0,
            "page_end": self.page_end or 0,
            "char_start": self.char_start or 0,
            "char_end": self.char_end or 0,
            "language": self.language,
            "tags": list(self.tags),
            "visibility": self.visibility,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "schema_version": self.schema_version,
            "is_active": self.is_active,
        }

    def to_es_document(self, text: str) -> dict[str, object]:
        """Return an Elasticsearch document for keyword retrieval.

        Args:
            text: Chunk text to index.

        Returns:
            Dictionary containing searchable text and filterable metadata.
        """
        document = {
            **self.to_milvus_fields(),
            "chunk_text": text,
        }
        if not self.created_at:
            document.pop("created_at")
        if not self.updated_at:
            document.pop("updated_at")
        return document
