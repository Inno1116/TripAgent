"""Knowledge-base metadata stores for document ingestion."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from contextlib import AbstractContextManager

ParserMode = Literal["auto", "local", "mcp"]
Visibility = Literal["private", "team", "public"]
DocumentStatus = Literal["active", "processing", "failed", "archived", "deleted"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class _Cursor(Protocol):
    def execute(self, query: str, params: object | None = None) -> object:
        """Execute a SQL statement."""
        ...

    def fetchone(self) -> Mapping[str, object] | None:
        """Fetch one row."""
        ...

    def fetchall(self) -> list[Mapping[str, object]]:
        """Fetch all rows."""
        ...


class _Connection(Protocol):
    def cursor(self, *, row_factory: object) -> AbstractContextManager[_Cursor]:
        """Open a cursor context."""
        ...


@dataclass(frozen=True, kw_only=True)
class KnowledgeBaseRecord:
    """Knowledge base metadata visible to a user."""

    kb_id: str
    tenant_id: str
    name: str
    description: str = ""
    owner_user_id: str | None = None
    visibility: Visibility = "private"
    status: str = "active"
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class DocumentRecord:
    """Uploaded source document metadata."""

    doc_id: str
    tenant_id: str
    kb_id: str
    source_type: str
    source_uri: str
    owner_user_id: str | None = None
    file_name: str = ""
    mime_type: str = ""
    byte_size: int = 0
    title: str = ""
    language: str = "unknown"
    visibility: Visibility = "private"
    status: DocumentStatus = "processing"
    latest_version: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class IngestionJobRecord:
    """Queued document ingestion job."""

    job_id: str
    tenant_id: str
    kb_id: str
    source_uri: str
    doc_id: str | None = None
    requested_by_user_id: str | None = None
    parser_mode: ParserMode = "auto"
    progress: int = 0
    status: JobStatus = "queued"
    error_message: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class DocumentVersionRecord:
    """One indexed version of a document."""

    doc_version: str
    doc_id: str
    tenant_id: str
    content_hash: str
    parser_version: str
    chunker_version: str
    embedding_model: str
    embedding_version: str
    chunk_count: int = 0
    status: str = "pending"
    error_message: str | None = None
    created_at: str = ""
    indexed_at: str | None = None


class KnowledgeBaseStore(Protocol):
    """Store contract for knowledge-base metadata and ingestion jobs."""

    def ensure_identity(
        self,
        *,
        tenant_id: str,
        tenant_name: str,
        user_id: str,
        email: str,
        display_name: str,
    ) -> None:
        """Ensure RAG tenant and user rows exist."""
        ...

    def create_knowledge_base(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str,
        description: str = "",
        visibility: Visibility = "private",
        kb_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> KnowledgeBaseRecord:
        """Create a knowledge base."""
        ...

    def list_knowledge_bases(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[KnowledgeBaseRecord]:
        """List knowledge bases visible to a user."""
        ...

    def get_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord | None:
        """Load one knowledge base visible to a user."""
        ...

    def delete_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord | None:
        """Soft-delete one knowledge base owned by the user."""
        ...

    def create_document_job(
        self,
        *,
        tenant_id: str,
        user_id: str,
        kb_id: str,
        source_uri: str,
        source_type: str,
        file_name: str,
        mime_type: str,
        byte_size: int,
        title: str,
        visibility: Visibility,
        parser_mode: ParserMode,
        metadata: Mapping[str, object],
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """Create a processing document and queued ingestion job."""
        ...

    def list_documents(self, *, tenant_id: str, user_id: str, kb_id: str, limit: int = 100) -> list[DocumentRecord]:
        """List documents in a knowledge base."""
        ...

    def delete_document(self, *, tenant_id: str, user_id: str, kb_id: str, doc_id: str) -> DocumentRecord | None:
        """Soft-delete one document owned by the user."""
        ...

    def get_job(self, *, tenant_id: str, user_id: str, job_id: str) -> IngestionJobRecord | None:
        """Load one ingestion job visible to a user."""
        ...

    def claim_next_job(self) -> IngestionJobRecord | None:
        """Claim the next queued ingestion job."""
        ...

    def create_document_version(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        content_hash: str,
        parser_version: str,
        chunker_version: str,
        embedding_model: str,
        embedding_version: str,
    ) -> DocumentVersionRecord:
        """Create a pending document version."""
        ...

    def replace_chunks(self, *, chunks: list[Mapping[str, object]]) -> None:
        """Upsert chunk manifests into PostgreSQL."""
        ...

    def mark_job_succeeded(
        self,
        *,
        job_id: str,
        doc_id: str,
        doc_version: str,
        title: str,
        language: str,
        chunk_count: int,
    ) -> None:
        """Mark a job and document as indexed."""
        ...

    def mark_job_failed(self, *, job_id: str, doc_id: str | None, error_message: str) -> None:
        """Mark a job as failed."""
        ...

    def mark_stale_jobs_failed(self, *, cutoff: datetime, error_message: str) -> int:
        """Mark running jobs older than the cutoff as failed."""
        ...


class InMemoryKnowledgeBaseStore:
    """In-memory store used by API tests and local prototypes."""

    def __init__(self) -> None:
        """Initialize empty records."""
        self._knowledge_bases: dict[str, KnowledgeBaseRecord] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._jobs: dict[str, IngestionJobRecord] = {}
        self._versions: dict[str, DocumentVersionRecord] = {}
        self._chunks: dict[str, Mapping[str, object]] = {}

    def ensure_identity(
        self,
        *,
        tenant_id: str,
        tenant_name: str,
        user_id: str,
        email: str,
        display_name: str,
    ) -> None:
        """In-memory records do not need separate identity rows."""
        _ = (tenant_id, tenant_name, user_id, email, display_name)

    def create_knowledge_base(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str,
        description: str = "",
        visibility: Visibility = "private",
        kb_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> KnowledgeBaseRecord:
        """Create a knowledge base."""
        now = _now()
        record = KnowledgeBaseRecord(
            kb_id=kb_id or _id("kb"),
            tenant_id=tenant_id,
            owner_user_id=user_id,
            name=name,
            description=description,
            visibility=visibility,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._knowledge_bases[record.kb_id] = record
        return record

    def list_knowledge_bases(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[KnowledgeBaseRecord]:
        """List knowledge bases visible to a user."""
        records = [
            item
            for item in self._knowledge_bases.values()
            if item.tenant_id == tenant_id and item.status == "active" and (item.owner_user_id == user_id or item.visibility in {"team", "public"})
        ]
        records.sort(key=lambda item: (item.updated_at, item.kb_id), reverse=True)
        return records[:limit]

    def get_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord | None:
        """Load one knowledge base visible to a user."""
        record = self._knowledge_bases.get(kb_id)
        if record is None or record.tenant_id != tenant_id or record.status != "active":
            return None
        if record.owner_user_id != user_id and record.visibility not in {"team", "public"}:
            return None
        return record

    def delete_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord | None:
        """Soft-delete one knowledge base owned by the user."""
        record = self._knowledge_bases.get(kb_id)
        if record is None or record.tenant_id != tenant_id or record.owner_user_id != user_id or record.status != "active":
            return None
        now = _now()
        deleted = replace(record, status="archived", updated_at=now)
        self._knowledge_bases[kb_id] = deleted
        for doc_id, document in list(self._documents.items()):
            if document.tenant_id == tenant_id and document.kb_id == kb_id and document.status != "deleted":
                self._documents[doc_id] = replace(document, status="deleted", updated_at=now)
        for job_id, job in list(self._jobs.items()):
            if job.tenant_id == tenant_id and job.kb_id == kb_id and job.status in {"queued", "running"}:
                self._jobs[job_id] = replace(job, status="cancelled", finished_at=now, updated_at=now)
        self._deactivate_chunks(tenant_id=tenant_id, kb_id=kb_id)
        return deleted

    def create_document_job(
        self,
        *,
        tenant_id: str,
        user_id: str,
        kb_id: str,
        source_uri: str,
        source_type: str,
        file_name: str,
        mime_type: str,
        byte_size: int,
        title: str,
        visibility: Visibility,
        parser_mode: ParserMode,
        metadata: Mapping[str, object],
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """Create a processing document and queued ingestion job."""
        now = _now()
        doc = DocumentRecord(
            doc_id=_id("doc"),
            tenant_id=tenant_id,
            kb_id=kb_id,
            owner_user_id=user_id,
            source_type=source_type,
            source_uri=source_uri,
            file_name=file_name,
            mime_type=mime_type,
            byte_size=byte_size,
            title=title,
            visibility=visibility,
            metadata=dict(metadata),
            created_at=now,
            updated_at=now,
        )
        job = IngestionJobRecord(
            job_id=_id("job"),
            tenant_id=tenant_id,
            kb_id=kb_id,
            doc_id=doc.doc_id,
            requested_by_user_id=user_id,
            source_uri=source_uri,
            parser_mode=parser_mode,
            metadata=dict(metadata),
            created_at=now,
            updated_at=now,
        )
        self._documents[doc.doc_id] = doc
        self._jobs[job.job_id] = job
        return doc, job

    def list_documents(self, *, tenant_id: str, user_id: str, kb_id: str, limit: int = 100) -> list[DocumentRecord]:
        """List documents in a knowledge base."""
        if self.get_knowledge_base(tenant_id=tenant_id, user_id=user_id, kb_id=kb_id) is None:
            return []
        records = [item for item in self._documents.values() if item.tenant_id == tenant_id and item.kb_id == kb_id and item.status != "deleted"]
        records.sort(key=lambda item: (item.updated_at, item.doc_id), reverse=True)
        return records[:limit]

    def delete_document(self, *, tenant_id: str, user_id: str, kb_id: str, doc_id: str) -> DocumentRecord | None:
        """Soft-delete one document owned by the user."""
        kb = self._knowledge_bases.get(kb_id)
        document = self._documents.get(doc_id)
        if kb is None or kb.tenant_id != tenant_id or kb.owner_user_id != user_id or kb.status != "active":
            return None
        if document is None or document.tenant_id != tenant_id or document.kb_id != kb_id or document.status == "deleted":
            return None
        now = _now()
        deleted = replace(document, status="deleted", updated_at=now)
        self._documents[doc_id] = deleted
        for job_id, job in list(self._jobs.items()):
            if job.tenant_id == tenant_id and job.doc_id == doc_id and job.status in {"queued", "running"}:
                self._jobs[job_id] = replace(job, status="cancelled", finished_at=now, updated_at=now)
        self._deactivate_chunks(tenant_id=tenant_id, kb_id=kb_id, doc_id=doc_id)
        return deleted

    def get_job(self, *, tenant_id: str, user_id: str, job_id: str) -> IngestionJobRecord | None:
        """Load one ingestion job visible to a user."""
        job = self._jobs.get(job_id)
        if job is None or job.tenant_id != tenant_id:
            return None
        if job.requested_by_user_id != user_id and self.get_knowledge_base(tenant_id=tenant_id, user_id=user_id, kb_id=job.kb_id) is None:
            return None
        return job

    def claim_next_job(self) -> IngestionJobRecord | None:
        """Claim the next queued ingestion job."""
        queued = sorted((job for job in self._jobs.values() if job.status == "queued"), key=lambda item: item.created_at)
        if not queued:
            return None
        job = queued[0]
        claimed = replace(job, status="running", progress=5, started_at=_now(), updated_at=_now())
        self._jobs[job.job_id] = claimed
        return claimed

    def create_document_version(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        content_hash: str,
        parser_version: str,
        chunker_version: str,
        embedding_model: str,
        embedding_version: str,
    ) -> DocumentVersionRecord:
        """Create a pending document version."""
        record = DocumentVersionRecord(
            doc_version=_id("docv"),
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash=content_hash,
            parser_version=parser_version,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            embedding_version=embedding_version,
            created_at=_now(),
        )
        self._versions[record.doc_version] = record
        return record

    def replace_chunks(self, *, chunks: list[Mapping[str, object]]) -> None:
        """Upsert chunk manifests into memory."""
        for chunk in chunks:
            self._chunks[str(chunk["chunk_id"])] = dict(chunk)

    def mark_job_succeeded(
        self,
        *,
        job_id: str,
        doc_id: str,
        doc_version: str,
        title: str,
        language: str,
        chunk_count: int,
    ) -> None:
        """Mark a job and document as indexed."""
        now = _now()
        job = self._jobs[job_id]
        self._jobs[job_id] = replace(job, status="succeeded", progress=100, finished_at=now, updated_at=now)
        document = self._documents[doc_id]
        if document.status != "deleted":
            self._documents[doc_id] = replace(
                document,
                status="active",
                latest_version=doc_version,
                title=title,
                language=language,
                updated_at=now,
            )
        version = self._versions[doc_version]
        self._versions[doc_version] = replace(version, status="indexed", chunk_count=chunk_count, indexed_at=now)

    def mark_job_failed(self, *, job_id: str, doc_id: str | None, error_message: str) -> None:
        """Mark a job as failed."""
        now = _now()
        job = self._jobs[job_id]
        self._jobs[job_id] = replace(job, status="failed", error_message=error_message, finished_at=now, updated_at=now)
        if doc_id and doc_id in self._documents:
            document = self._documents[doc_id]
            if document.status != "deleted":
                self._documents[doc_id] = replace(document, status="failed", error_message=error_message, updated_at=now)

    def mark_stale_jobs_failed(self, *, cutoff: datetime, error_message: str) -> int:
        """Mark running jobs older than the cutoff as failed."""
        count = 0
        for job_id, job in list(self._jobs.items()):
            started = _parse_datetime(job.started_at)
            if job.status != "running" or started is None or started >= cutoff:
                continue
            self.mark_job_failed(job_id=job_id, doc_id=job.doc_id, error_message=error_message)
            count += 1
        return count

    def _deactivate_chunks(self, *, tenant_id: str, kb_id: str, doc_id: str | None = None) -> None:
        for chunk_id, chunk in list(self._chunks.items()):
            if chunk.get("tenant_id") != tenant_id or chunk.get("kb_id") != kb_id:
                continue
            if doc_id is not None and chunk.get("doc_id") != doc_id:
                continue
            updated = dict(chunk)
            updated["is_active"] = False
            self._chunks[chunk_id] = updated


class PostgresKnowledgeBaseStore:
    """PostgreSQL-backed knowledge-base metadata store."""

    def __init__(self, *, dsn: str, connection: _Connection | None = None) -> None:
        """Initialize the store.

        Args:
            dsn: PostgreSQL DSN.
            connection: Optional existing connection for tests.
        """
        self._dsn = dsn
        self._connection = connection

    def ensure_identity(
        self,
        *,
        tenant_id: str,
        tenant_name: str,
        user_id: str,
        email: str,
        display_name: str,
    ) -> None:
        """Ensure RAG tenant and user rows exist."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rag_tenants (tenant_id, name)
                VALUES (%(tenant_id)s, %(name)s)
                ON CONFLICT (tenant_id) DO UPDATE SET name = EXCLUDED.name
                """,
                {"tenant_id": tenant_id, "name": tenant_name},
            )
            cursor.execute(
                """
                INSERT INTO rag_users (user_id, tenant_id, external_user_id, display_name, email)
                VALUES (%(user_id)s, %(tenant_id)s, %(external_user_id)s, %(display_name)s, %(email)s)
                ON CONFLICT (user_id) DO UPDATE
                SET display_name = EXCLUDED.display_name, email = EXCLUDED.email, updated_at = now()
                """,
                {
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "external_user_id": user_id,
                    "display_name": display_name,
                    "email": email,
                },
            )

    def create_knowledge_base(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str,
        description: str = "",
        visibility: Visibility = "private",
        kb_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> KnowledgeBaseRecord:
        """Create a knowledge base."""
        resolved_id = kb_id or _id("kb")
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rag_knowledge_bases (kb_id, tenant_id, owner_user_id, name, description, visibility, metadata)
                VALUES (%(kb_id)s, %(tenant_id)s, %(owner_user_id)s, %(name)s, %(description)s, %(visibility)s, %(metadata)s)
                RETURNING *
                """,
                {
                    "kb_id": resolved_id,
                    "tenant_id": tenant_id,
                    "owner_user_id": user_id,
                    "name": name,
                    "description": description,
                    "visibility": visibility,
                    "metadata": _jsonb(dict(metadata or {})),
                },
            )
            row = cursor.fetchone()
        return _knowledge_base_from_row(_require_row(row))

    def list_knowledge_bases(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[KnowledgeBaseRecord]:
        """List knowledge bases visible to a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM rag_knowledge_bases
                WHERE tenant_id = %(tenant_id)s
                  AND status = 'active'
                  AND (owner_user_id = %(user_id)s OR visibility IN ('team', 'public'))
                ORDER BY updated_at DESC, kb_id DESC
                LIMIT %(limit)s
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "limit": limit},
            )
            rows = cursor.fetchall()
        return [_knowledge_base_from_row(row) for row in rows]

    def get_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord | None:
        """Load one knowledge base visible to a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM rag_knowledge_bases
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND status = 'active'
                  AND (owner_user_id = %(user_id)s OR visibility IN ('team', 'public'))
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "kb_id": kb_id},
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _knowledge_base_from_row(row)

    def delete_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord | None:
        """Soft-delete one knowledge base owned by the user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE rag_knowledge_bases
                SET status = 'archived', updated_at = now()
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND owner_user_id = %(user_id)s
                  AND status = 'active'
                RETURNING *
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "kb_id": kb_id},
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute(
                """
                UPDATE rag_documents
                SET status = 'deleted', updated_at = now()
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND status <> 'deleted'
                """,
                {"tenant_id": tenant_id, "kb_id": kb_id},
            )
            cursor.execute(
                """
                UPDATE rag_chunks
                SET is_active = false, updated_at = now()
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND is_active = true
                """,
                {"tenant_id": tenant_id, "kb_id": kb_id},
            )
            cursor.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = 'cancelled', finished_at = now(), updated_at = now()
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND status IN ('queued', 'running')
                """,
                {"tenant_id": tenant_id, "kb_id": kb_id},
            )
        return _knowledge_base_from_row(row)

    def create_document_job(
        self,
        *,
        tenant_id: str,
        user_id: str,
        kb_id: str,
        source_uri: str,
        source_type: str,
        file_name: str,
        mime_type: str,
        byte_size: int,
        title: str,
        visibility: Visibility,
        parser_mode: ParserMode,
        metadata: Mapping[str, object],
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """Create a processing document and queued ingestion job."""
        doc_id = _id("doc")
        job_id = _id("job")
        params = {
            "doc_id": doc_id,
            "job_id": job_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "user_id": user_id,
            "source_type": source_type,
            "source_uri": source_uri,
            "file_name": file_name,
            "mime_type": mime_type,
            "byte_size": byte_size,
            "title": title,
            "visibility": visibility,
            "parser_mode": parser_mode,
            "metadata": _jsonb(dict(metadata)),
        }
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rag_documents (
                    doc_id, tenant_id, kb_id, owner_user_id, source_type, source_uri,
                    file_name, mime_type, byte_size, title, visibility, metadata
                )
                VALUES (
                    %(doc_id)s, %(tenant_id)s, %(kb_id)s, %(user_id)s, %(source_type)s, %(source_uri)s,
                    %(file_name)s, %(mime_type)s, %(byte_size)s, %(title)s, %(visibility)s, %(metadata)s
                )
                RETURNING *
                """,
                params,
            )
            document_row = cursor.fetchone()
            cursor.execute(
                """
                INSERT INTO rag_ingestion_jobs (
                    job_id, tenant_id, kb_id, doc_id, requested_by_user_id, source_uri,
                    parser_mode, metadata
                )
                VALUES (
                    %(job_id)s, %(tenant_id)s, %(kb_id)s, %(doc_id)s, %(user_id)s, %(source_uri)s,
                    %(parser_mode)s, %(metadata)s
                )
                RETURNING *
                """,
                params,
            )
            job_row = cursor.fetchone()
        return _document_from_row(_require_row(document_row)), _job_from_row(_require_row(job_row))

    def list_documents(self, *, tenant_id: str, user_id: str, kb_id: str, limit: int = 100) -> list[DocumentRecord]:
        """List documents in a knowledge base."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT d.*
                FROM rag_documents d
                JOIN rag_knowledge_bases kb ON kb.kb_id = d.kb_id
                WHERE d.tenant_id = %(tenant_id)s
                  AND d.kb_id = %(kb_id)s
                  AND d.status <> 'deleted'
                  AND kb.status = 'active'
                  AND (kb.owner_user_id = %(user_id)s OR kb.visibility IN ('team', 'public'))
                ORDER BY d.updated_at DESC, d.doc_id DESC
                LIMIT %(limit)s
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "kb_id": kb_id, "limit": limit},
            )
            rows = cursor.fetchall()
        return [_document_from_row(row) for row in rows]

    def delete_document(self, *, tenant_id: str, user_id: str, kb_id: str, doc_id: str) -> DocumentRecord | None:
        """Soft-delete one document owned by the user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE rag_documents d
                SET status = 'deleted', updated_at = now()
                FROM rag_knowledge_bases kb
                WHERE d.tenant_id = %(tenant_id)s
                  AND d.kb_id = %(kb_id)s
                  AND d.doc_id = %(doc_id)s
                  AND d.status <> 'deleted'
                  AND kb.kb_id = d.kb_id
                  AND kb.tenant_id = d.tenant_id
                  AND kb.owner_user_id = %(user_id)s
                  AND kb.status = 'active'
                RETURNING d.*
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "kb_id": kb_id, "doc_id": doc_id},
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute(
                """
                UPDATE rag_chunks
                SET is_active = false, updated_at = now()
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND doc_id = %(doc_id)s
                  AND is_active = true
                """,
                {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id},
            )
            cursor.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = 'cancelled', finished_at = now(), updated_at = now()
                WHERE tenant_id = %(tenant_id)s
                  AND kb_id = %(kb_id)s
                  AND doc_id = %(doc_id)s
                  AND status IN ('queued', 'running')
                """,
                {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id},
            )
        return _document_from_row(row)

    def get_job(self, *, tenant_id: str, user_id: str, job_id: str) -> IngestionJobRecord | None:
        """Load one ingestion job visible to a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT j.*
                FROM rag_ingestion_jobs j
                JOIN rag_knowledge_bases kb ON kb.kb_id = j.kb_id
                WHERE j.tenant_id = %(tenant_id)s
                  AND j.job_id = %(job_id)s
                  AND (j.requested_by_user_id = %(user_id)s OR kb.owner_user_id = %(user_id)s OR kb.visibility IN ('team', 'public'))
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "job_id": job_id},
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _job_from_row(row)

    def claim_next_job(self) -> IngestionJobRecord | None:
        """Claim the next queued ingestion job."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                WITH candidate AS (
                    SELECT job_id
                    FROM rag_ingestion_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC, job_id ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE rag_ingestion_jobs j
                SET status = 'running', started_at = COALESCE(started_at, now()), progress = 5, updated_at = now()
                FROM candidate
                WHERE j.job_id = candidate.job_id
                RETURNING j.*
                """
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _job_from_row(row)

    def create_document_version(
        self,
        *,
        doc_id: str,
        tenant_id: str,
        content_hash: str,
        parser_version: str,
        chunker_version: str,
        embedding_model: str,
        embedding_version: str,
    ) -> DocumentVersionRecord:
        """Create a pending document version."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO rag_document_versions (
                    doc_version, doc_id, tenant_id, content_hash, parser_version,
                    chunker_version, embedding_model, embedding_version
                )
                VALUES (
                    %(doc_version)s, %(doc_id)s, %(tenant_id)s, %(content_hash)s, %(parser_version)s,
                    %(chunker_version)s, %(embedding_model)s, %(embedding_version)s
                )
                ON CONFLICT (doc_id, content_hash, embedding_model, embedding_version) DO UPDATE
                SET status = 'pending', error_message = NULL
                RETURNING *
                """,
                {
                    "doc_version": _id("docv"),
                    "doc_id": doc_id,
                    "tenant_id": tenant_id,
                    "content_hash": content_hash,
                    "parser_version": parser_version,
                    "chunker_version": chunker_version,
                    "embedding_model": embedding_model,
                    "embedding_version": embedding_version,
                },
            )
            row = cursor.fetchone()
        return _version_from_row(_require_row(row))

    def replace_chunks(self, *, chunks: list[Mapping[str, object]]) -> None:
        """Upsert chunk manifests into PostgreSQL."""
        with self._cursor() as cursor:
            for chunk in chunks:
                values = dict(chunk)
                values["chunk_text"] = str(values.get("chunk_text") or "")
                values["tags"] = _jsonb(values.get("tags") or [])
                cursor.execute(
                    """
                    INSERT INTO rag_chunks (
                        chunk_id, tenant_id, kb_id, doc_id, doc_version, user_id, chunk_index,
                        content_hash, chunk_text, source_type, source_uri, title, section_path, page_start,
                        page_end, char_start, char_end, language, tags, visibility,
                        embedding_model, embedding_version, schema_version, is_active
                    )
                    VALUES (
                        %(chunk_id)s, %(tenant_id)s, %(kb_id)s, %(doc_id)s, %(doc_version)s, %(user_id)s, %(chunk_index)s,
                        %(content_hash)s, %(chunk_text)s, %(source_type)s, %(source_uri)s, %(title)s, %(section_path)s, %(page_start)s,
                        %(page_end)s, %(char_start)s, %(char_end)s, %(language)s, %(tags)s, %(visibility)s,
                        %(embedding_model)s, %(embedding_version)s, %(schema_version)s, %(is_active)s
                    )
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET chunk_text = EXCLUDED.chunk_text,
                        title = EXCLUDED.title,
                        section_path = EXCLUDED.section_path,
                        tags = EXCLUDED.tags,
                        visibility = EXCLUDED.visibility,
                        is_active = EXCLUDED.is_active,
                        updated_at = now()
                    """,
                    values,
                )

    def mark_job_succeeded(
        self,
        *,
        job_id: str,
        doc_id: str,
        doc_version: str,
        title: str,
        language: str,
        chunk_count: int,
    ) -> None:
        """Mark a job and document as indexed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE rag_document_versions
                SET status = 'indexed', chunk_count = %(chunk_count)s, indexed_at = now()
                WHERE doc_version = %(doc_version)s
                """,
                {"doc_version": doc_version, "chunk_count": chunk_count},
            )
            cursor.execute(
                """
                UPDATE rag_documents
                SET status = 'active', latest_version = %(doc_version)s, title = %(title)s, language = %(language)s,
                    error_message = NULL, updated_at = now()
                WHERE doc_id = %(doc_id)s
                  AND status <> 'deleted'
                """,
                {"doc_id": doc_id, "doc_version": doc_version, "title": title, "language": language},
            )
            cursor.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = 'succeeded', progress = 100, error_message = NULL, finished_at = now(), updated_at = now()
                WHERE job_id = %(job_id)s
                """,
                {"job_id": job_id},
            )

    def mark_job_failed(self, *, job_id: str, doc_id: str | None, error_message: str) -> None:
        """Mark a job as failed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE rag_ingestion_jobs
                SET status = 'failed', error_message = %(error_message)s, finished_at = now(), updated_at = now()
                WHERE job_id = %(job_id)s
                """,
                {"job_id": job_id, "error_message": error_message},
            )
            if doc_id:
                cursor.execute(
                    """
                    UPDATE rag_documents
                    SET status = 'failed', error_message = %(error_message)s, updated_at = now()
                    WHERE doc_id = %(doc_id)s
                      AND status <> 'deleted'
                    """,
                    {"doc_id": doc_id, "error_message": error_message},
                )

    def mark_stale_jobs_failed(self, *, cutoff: datetime, error_message: str) -> int:
        """Mark running jobs older than the cutoff as failed."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                WITH stale AS (
                    UPDATE rag_ingestion_jobs
                    SET status = 'failed', error_message = %(error_message)s, finished_at = now(), updated_at = now()
                    WHERE status = 'running'
                      AND started_at < %(cutoff)s
                    RETURNING doc_id
                ),
                failed_docs AS (
                    UPDATE rag_documents d
                    SET status = 'failed', error_message = %(error_message)s, updated_at = now()
                    FROM stale
                    WHERE d.doc_id = stale.doc_id
                      AND d.status <> 'deleted'
                    RETURNING d.doc_id
                )
                SELECT
                    (SELECT count(*) FROM stale) AS failed_count,
                    (SELECT count(*) FROM failed_docs) AS failed_doc_count
                """,
                {"cutoff": cutoff, "error_message": error_message},
            )
            row = cursor.fetchone()
        if row is None:
            return 0
        value = row.get("failed_count", 0)
        if isinstance(value, int):
            return value
        return int(str(value))

    @contextmanager
    def _cursor(self) -> Iterator[_Cursor]:
        connection = self._connection
        if connection is not None:
            with connection.cursor(row_factory=_dict_row()) as cursor:
                yield cursor
            return
        with _connect(self._dsn) as connection, connection.cursor(row_factory=_dict_row()) as cursor:
            yield cursor


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _connect(dsn: str) -> AbstractContextManager[_Connection]:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use PostgreSQL knowledge bases."
        raise ImportError(msg) from exc
    return cast("AbstractContextManager[_Connection]", psycopg.connect(dsn))


def _dict_row() -> object:
    try:
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use PostgreSQL knowledge bases."
        raise ImportError(msg) from exc
    return dict_row


def _jsonb(value: object) -> object:
    try:
        from psycopg.types.json import Jsonb  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use PostgreSQL knowledge bases."
        raise ImportError(msg) from exc
    return Jsonb(value)


def _require_row(row: Mapping[str, object] | None) -> Mapping[str, object]:
    if row is None:
        msg = "PostgreSQL operation returned no row."
        raise RuntimeError(msg)
    return row


def _knowledge_base_from_row(row: Mapping[str, object]) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        kb_id=str(row["kb_id"]),
        tenant_id=str(row["tenant_id"]),
        owner_user_id=_optional_str(row.get("owner_user_id")),
        name=str(row["name"]),
        description=str(row.get("description") or ""),
        visibility=_visibility(row.get("visibility")),
        status=str(row.get("status", "active")),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _document_from_row(row: Mapping[str, object]) -> DocumentRecord:
    return DocumentRecord(
        doc_id=str(row["doc_id"]),
        tenant_id=str(row["tenant_id"]),
        kb_id=str(row["kb_id"]),
        owner_user_id=_optional_str(row.get("owner_user_id")),
        source_type=str(row["source_type"]),
        source_uri=str(row["source_uri"]),
        file_name=str(row.get("file_name") or ""),
        mime_type=str(row.get("mime_type") or ""),
        byte_size=_int_or_zero(row.get("byte_size")),
        title=str(row.get("title") or ""),
        language=str(row.get("language") or "unknown"),
        visibility=_visibility(row.get("visibility")),
        status=_document_status(row.get("status")),
        latest_version=_optional_str(row.get("latest_version")),
        error_message=_optional_str(row.get("error_message")),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _job_from_row(row: Mapping[str, object]) -> IngestionJobRecord:
    return IngestionJobRecord(
        job_id=str(row["job_id"]),
        tenant_id=str(row["tenant_id"]),
        kb_id=str(row["kb_id"]),
        doc_id=_optional_str(row.get("doc_id")),
        requested_by_user_id=_optional_str(row.get("requested_by_user_id")),
        source_uri=str(row["source_uri"]),
        parser_mode=_parser_mode(row.get("parser_mode")),
        progress=_int_or_zero(row.get("progress")),
        status=_job_status(row.get("status")),
        error_message=_optional_str(row.get("error_message")),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        started_at=_optional_str(row.get("started_at")),
        finished_at=_optional_str(row.get("finished_at")),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _version_from_row(row: Mapping[str, object]) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        doc_version=str(row["doc_version"]),
        doc_id=str(row["doc_id"]),
        tenant_id=str(row["tenant_id"]),
        content_hash=str(row["content_hash"]),
        parser_version=str(row["parser_version"]),
        chunker_version=str(row["chunker_version"]),
        embedding_model=str(row["embedding_model"]),
        embedding_version=str(row["embedding_version"]),
        chunk_count=_int_or_zero(row.get("chunk_count")),
        status=str(row.get("status", "pending")),
        error_message=_optional_str(row.get("error_message")),
        created_at=str(row.get("created_at", "")),
        indexed_at=_optional_str(row.get("indexed_at")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_zero(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(str(value))


def _visibility(value: object) -> Visibility:
    if value in {"private", "team", "public"}:
        return cast("Visibility", value)
    return "private"


def _document_status(value: object) -> DocumentStatus:
    if value in {"active", "processing", "failed", "archived", "deleted"}:
        return cast("DocumentStatus", value)
    return "processing"


def _job_status(value: object) -> JobStatus:
    if value in {"queued", "running", "succeeded", "failed", "cancelled"}:
        return cast("JobStatus", value)
    return "queued"


def _parser_mode(value: object) -> ParserMode:
    if value in {"auto", "local", "mcp"}:
        return cast("ParserMode", value)
    return "auto"


__all__ = [
    "DocumentRecord",
    "DocumentStatus",
    "DocumentVersionRecord",
    "InMemoryKnowledgeBaseStore",
    "IngestionJobRecord",
    "JobStatus",
    "KnowledgeBaseRecord",
    "KnowledgeBaseStore",
    "ParserMode",
    "PostgresKnowledgeBaseStore",
    "Visibility",
]
