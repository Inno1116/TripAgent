"""High-level service for uploading and indexing knowledge-base documents."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from kyuriagents.ingestion.indexing import HybridChunkIndexer
from kyuriagents.ingestion.parsers import DocumentParser, ParseRequest, build_document_parser
from kyuriagents.ingestion.redis_queue import IngestionJobQueue, default_ingestion_job_queue
from kyuriagents.ingestion.store import (
    DocumentRecord,
    IngestionJobRecord,
    InMemoryKnowledgeBaseStore,
    KnowledgeBaseRecord,
    KnowledgeBaseStore,
    ParserMode,
    PostgresKnowledgeBaseStore,
    Visibility,
)
from kyuriagents.rag._text import tokenize
from kyuriagents.rag.metadata import ChunkMetadata
from kyuriagents.rag.types import DocumentChunk
from kyuriagents.runtime.errors import public_error_message

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kyuriagents.runtime import AgentRuntimeConfig

_CHUNKER_VERSION = "fixed_char_window:v1"
_DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_SOURCE_TYPE_BY_MIME = {"application/pdf": "pdf", _DOCX_MIME_TYPE: "docx", "text/plain": "txt"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MIN_KEYWORD_LENGTH = 3
_LOGGER = logging.getLogger(__name__)


class ChunkIndexer(Protocol):
    """Minimal indexer contract used by the ingestion service."""

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Index prepared chunks."""
        ...

    def delete_knowledge_base(self, *, tenant_id: str, kb_id: str) -> None:
        """Remove indexed chunks for a knowledge base."""
        ...

    def delete_document(self, *, tenant_id: str, kb_id: str, doc_id: str) -> None:
        """Remove indexed chunks for a document."""
        ...


class KnowledgeBaseService:
    """Coordinate knowledge-base metadata, file storage, parsing, and indexing."""

    def __init__(
        self,
        *,
        config: AgentRuntimeConfig,
        store: KnowledgeBaseStore | None = None,
        parser: DocumentParser | None = None,
        indexer: ChunkIndexer | None = None,
        job_queue: IngestionJobQueue | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            config: Runtime configuration.
            store: Optional metadata store.
            parser: Optional document parser.
            indexer: Optional hybrid chunk indexer.
            job_queue: Optional Redis wake-up queue for ingestion workers.
        """
        self._config = config
        self._store = store or _default_store(config)
        self._parser = parser or build_document_parser(config)
        self._indexer = indexer or HybridChunkIndexer(config=config)
        self._job_queue = job_queue or default_ingestion_job_queue(config)
        self._upload_root = Path(config.upload_dir)

    @property
    def store(self) -> KnowledgeBaseStore:
        """Return the underlying metadata store."""
        return self._store

    def ensure_identity(
        self,
        *,
        tenant_id: str,
        tenant_name: str,
        user_id: str,
        email: str,
        display_name: str,
    ) -> None:
        """Ensure RAG identity rows exist."""
        self._store.ensure_identity(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            user_id=user_id,
            email=email,
            display_name=display_name,
        )

    def create_knowledge_base(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str,
        description: str = "",
        visibility: Visibility = "private",
        metadata: Mapping[str, object] | None = None,
    ) -> KnowledgeBaseRecord:
        """Create a knowledge base."""
        return self._store.create_knowledge_base(
            tenant_id=tenant_id,
            user_id=user_id,
            name=name.strip() or "未命名知识库",
            description=description,
            visibility=visibility,
            metadata=metadata,
        )

    def list_knowledge_bases(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[KnowledgeBaseRecord]:
        """List knowledge bases visible to the user."""
        return self._store.list_knowledge_bases(tenant_id=tenant_id, user_id=user_id, limit=limit)

    def upload_document(
        self,
        *,
        tenant_id: str,
        user_id: str,
        kb_id: str,
        filename: str,
        mime_type: str,
        content: bytes,
        parser_mode: ParserMode | None = None,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        """Store an uploaded document and create an ingestion job.

        Args:
            tenant_id: Tenant identifier.
            user_id: Uploading user identifier.
            kb_id: Target knowledge base.
            filename: Original file name.
            mime_type: Request MIME type.
            content: Raw file bytes.
            parser_mode: Optional parser mode override for the job.

        Returns:
            Created document and job records.
        """
        if not content:
            msg = "Uploaded document is empty."
            raise ValueError(msg)
        if len(content) > self._config.upload_max_bytes:
            msg = f"Uploaded document exceeds {self._config.upload_max_bytes} bytes."
            raise ValueError(msg)
        kb = self._store.get_knowledge_base(tenant_id=tenant_id, user_id=user_id, kb_id=kb_id)
        if kb is None:
            msg = "Knowledge base not found."
            raise LookupError(msg)
        normalized_mime = _normalize_mime_type(mime_type, filename=filename)
        source_type = _source_type(normalized_mime, filename=filename)
        safe_name = _safe_filename(filename)
        doc_dir = self._upload_root / tenant_id / user_id / kb_id / _content_hash(content)[:16]
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / safe_name
        file_path.write_bytes(content)
        source_uri = f"upload://{tenant_id}/{kb_id}/{file_path.parent.name}/{safe_name}"
        metadata = {
            "storage_path": str(file_path),
            "content_sha256": _content_hash(content),
            "filename": safe_name,
            "mime_type": normalized_mime,
            "visibility": kb.visibility,
        }
        document, job = self._store.create_document_job(
            tenant_id=tenant_id,
            user_id=user_id,
            kb_id=kb_id,
            source_uri=source_uri,
            source_type=source_type,
            file_name=safe_name,
            mime_type=normalized_mime,
            byte_size=len(content),
            title=safe_name,
            visibility=kb.visibility,
            parser_mode=parser_mode or self._config.ingestion_parser_mode,
            metadata=metadata,
        )
        self._enqueue_job(job.job_id)
        return document, job

    def list_documents(self, *, tenant_id: str, user_id: str, kb_id: str, limit: int = 100) -> list[DocumentRecord]:
        """List documents for a knowledge base."""
        return self._store.list_documents(tenant_id=tenant_id, user_id=user_id, kb_id=kb_id, limit=limit)

    def delete_knowledge_base(self, *, tenant_id: str, user_id: str, kb_id: str) -> KnowledgeBaseRecord:
        """Soft-delete a knowledge base and remove its retrieval chunks."""
        record = self._store.delete_knowledge_base(tenant_id=tenant_id, user_id=user_id, kb_id=kb_id)
        if record is None:
            msg = "Knowledge base not found."
            raise LookupError(msg)
        self._indexer.delete_knowledge_base(tenant_id=tenant_id, kb_id=kb_id)
        return record

    def delete_document(self, *, tenant_id: str, user_id: str, kb_id: str, doc_id: str) -> DocumentRecord:
        """Soft-delete a document and remove its retrieval chunks."""
        record = self._store.delete_document(tenant_id=tenant_id, user_id=user_id, kb_id=kb_id, doc_id=doc_id)
        if record is None:
            msg = "Document not found."
            raise LookupError(msg)
        self._indexer.delete_document(tenant_id=tenant_id, kb_id=kb_id, doc_id=doc_id)
        return record

    def get_job(self, *, tenant_id: str, user_id: str, job_id: str) -> IngestionJobRecord | None:
        """Load one job visible to a user."""
        return self._store.get_job(tenant_id=tenant_id, user_id=user_id, job_id=job_id)

    def fail_stale_jobs(self, *, max_age_seconds: int | None = None) -> int:
        """Mark running jobs that exceeded the configured timeout as failed."""
        timeout = self._config.ingestion_job_timeout_seconds if max_age_seconds is None else max_age_seconds
        if timeout <= 0:
            return 0
        cutoff = datetime.now(tz=UTC) - timedelta(seconds=timeout)
        return self._store.mark_stale_jobs_failed(
            cutoff=cutoff,
            error_message=f"Ingestion job exceeded {timeout} seconds.",
        )

    def process_next_job(
        self,
        *,
        wait_for_queue: bool = False,
        queue_timeout_seconds: int | None = None,
    ) -> IngestionJobRecord | None:
        """Process one queued ingestion job.

        Args:
            wait_for_queue: Whether to wait on the configured wake-up queue when
                PostgreSQL has no immediately queued jobs.
            queue_timeout_seconds: Optional queue wait timeout override.

        Returns:
            Claimed job, or `None` when no queued job exists.
        """
        job = self._store.claim_next_job()
        if job is None and wait_for_queue:
            self._wait_for_queued_job(timeout_seconds=queue_timeout_seconds)
            job = self._store.claim_next_job()
        if job is None:
            return None
        try:
            self._process_job(job)
        except Exception as exc:  # noqa: BLE001  # The worker must persist unexpected parser/indexer failures.
            self._store.mark_job_failed(job_id=job.job_id, doc_id=job.doc_id, error_message=public_error_message(exc))
        return job

    def _process_job(self, job: IngestionJobRecord) -> None:
        if not job.doc_id:
            msg = "Ingestion job is missing `doc_id`."
            raise ValueError(msg)
        storage_path = _storage_path(job.metadata)
        request = ParseRequest(
            file_path=storage_path,
            source_uri=job.source_uri,
            filename=str(job.metadata.get("filename") or storage_path.name),
            mime_type=str(job.metadata.get("mime_type") or _mime_from_suffix(storage_path.name)),
            metadata=dict(job.metadata),
        )
        parser = self._parser_for_job(job)
        if not parser.supports(request):
            msg = f"Parser `{parser.name}` does not support {request.mime_type or request.filename}."
            raise ValueError(msg)
        # 解析的真正地方，按页抽取
        parsed = parser.parse(request)
        text = parsed.text.strip()
        if not text:
            msg = "Parsed document is empty."
            raise ValueError(msg)
        version = self._store.create_document_version(
            doc_id=job.doc_id,
            tenant_id=job.tenant_id,
            content_hash=_content_hash(text.encode("utf-8")),
            parser_version=parser.version,
            chunker_version=_CHUNKER_VERSION,
            embedding_model=self._config.embedding_model,
            embedding_version=_embedding_version(self._config),
        )
        chunks = _document_chunks(
            parsed.sections,
            tenant_id=job.tenant_id,
            user_id=job.requested_by_user_id,
            kb_id=job.kb_id,
            doc_id=job.doc_id,
            doc_version=version.doc_version,
            source_uri=job.source_uri,
            source_type=_source_type(request.mime_type, filename=request.filename),
            title=parsed.title or request.filename,
            language=parsed.language,
            visibility=_visibility(job.metadata.get("visibility")),
            embedding_model=self._config.embedding_model,
            embedding_version=_embedding_version(self._config),
            chunk_chars=self._config.ingestion_chunk_chars,
            chunk_overlap=self._config.ingestion_chunk_overlap,
        )
        self._indexer.index(chunks)
        self._store.replace_chunks(chunks=[_chunk_manifest(chunk) for chunk in chunks])
        self._store.mark_job_succeeded(
            job_id=job.job_id,
            doc_id=job.doc_id,
            doc_version=version.doc_version,
            title=parsed.title or request.filename,
            language=parsed.language,
            chunk_count=len(chunks),
        )

    def _parser_for_job(self, job: IngestionJobRecord) -> DocumentParser:
        if job.parser_mode == self._config.ingestion_parser_mode:
            return self._parser
        return build_document_parser(replace(self._config, ingestion_parser_mode=job.parser_mode))

    def _enqueue_job(self, job_id: str) -> None:
        try:
            self._job_queue.enqueue(job_id)
        except Exception as exc:  # noqa: BLE001  # Redis wakeups are best-effort; PostgreSQL keeps the job durable.
            _LOGGER.warning("failed to publish ingestion job wakeup: %s", public_error_message(exc))

    def _wait_for_queued_job(self, *, timeout_seconds: int | None) -> None:
        timeout = self._config.ingestion_redis_block_timeout_seconds if timeout_seconds is None else timeout_seconds
        try:
            self._job_queue.wait_for_job(timeout_seconds=timeout)
        except Exception as exc:  # noqa: BLE001  # Worker falls back to PostgreSQL polling after Redis errors.
            _LOGGER.warning("failed to wait for ingestion job wakeup: %s", public_error_message(exc))


def _default_store(config: AgentRuntimeConfig) -> KnowledgeBaseStore:
    if config.postgres_dsn:
        return PostgresKnowledgeBaseStore(dsn=config.postgres_dsn)
    return InMemoryKnowledgeBaseStore()


def _document_chunks(
    sections: Sequence[object],
    *,
    tenant_id: str,
    user_id: str | None,
    kb_id: str,
    doc_id: str,
    doc_version: str,
    source_uri: str,
    source_type: str,
    title: str,
    language: str,
    visibility: Visibility,
    embedding_model: str,
    embedding_version: str,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    now = datetime.now(tz=UTC).isoformat()
    chunks: list[DocumentChunk] = []
    index = 0
    for section in sections:
        text = str(getattr(section, "text", "")).strip()
        if not text:
            continue
        for window, start, end in _chunk_windows(text, size=chunk_chars, overlap=chunk_overlap):
            content_hash = _content_hash(window.encode("utf-8"))
            metadata = ChunkMetadata(
                chunk_id=f"{doc_id}:{index}:{content_hash[:16]}",
                tenant_id=tenant_id,
                user_id=user_id,
                kb_id=kb_id,
                doc_id=doc_id,
                doc_version=doc_version,
                chunk_index=index,
                content_hash=content_hash,
                source_type=source_type,
                source_uri=source_uri,
                title=title,
                section_path=str(getattr(section, "title", "")),
                page_start=cast("int | None", getattr(section, "page_start", None)),
                page_end=cast("int | None", getattr(section, "page_end", None)),
                char_start=start,
                char_end=end,
                language=language,
                tags=(),
                visibility=visibility,
                created_at=now,
                updated_at=now,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
            )
            chunks.append(DocumentChunk(text=window, metadata=metadata, keywords=_keywords(window)))
            index += 1
    return chunks


def _chunk_windows(text: str, *, size: int, overlap: int) -> list[tuple[str, int, int]]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    windows: list[tuple[str, int, int]] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind("。", start, end), normalized.rfind(".", start, end), normalized.rfind(" ", start, end))
            if boundary > start + max(size // 2, 1):
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if chunk:
            windows.append((chunk, start, end))
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return windows


def _chunk_manifest(chunk: DocumentChunk) -> Mapping[str, object]:
    fields = chunk.metadata.to_milvus_fields()
    fields["chunk_text"] = chunk.text
    fields["tags"] = list(chunk.metadata.tags)
    return fields


def _keywords(text: str, *, limit: int = 24) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for token in tokenize(text):
        if len(token) < _MIN_KEYWORD_LENGTH or token in seen:
            continue
        seen.add(token)
        values.append(token)
        if len(values) >= limit:
            break
    return tuple(values)


def _storage_path(metadata: Mapping[str, object]) -> Path:
    raw = metadata.get("storage_path")
    if not raw:
        msg = "Ingestion job is missing the uploaded storage path."
        raise ValueError(msg)
    path = Path(str(raw))
    if not path.exists():
        msg = f"Uploaded source file does not exist: {path}"
        raise FileNotFoundError(msg)
    return path


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "document.pdf"
    name = _SAFE_NAME_RE.sub("_", name)
    if "." not in name:
        name = f"{name}.pdf"
    return name[:180]


def _normalize_mime_type(mime_type: str, *, filename: str) -> str:
    value = mime_type.split(";", 1)[0].strip().lower()
    if value:
        return value
    return _mime_from_suffix(filename)


def _mime_from_suffix(filename: str) -> str:
    suffix = filename.lower()
    if suffix.endswith(".pdf"):
        return "application/pdf"
    if suffix.endswith(".docx"):
        return _DOCX_MIME_TYPE
    if suffix.endswith((".txt", ".text")):
        return "text/plain"
    return "application/octet-stream"


def _source_type(mime_type: str, *, filename: str) -> str:
    if mime_type in _SOURCE_TYPE_BY_MIME:
        return _SOURCE_TYPE_BY_MIME[mime_type]
    suffix = filename.lower()
    if suffix.endswith(".pdf"):
        return "pdf"
    if suffix.endswith(".docx"):
        return "docx"
    if suffix.endswith((".txt", ".text")):
        return "txt"
    return "file"


def _visibility(value: object) -> Visibility:
    if value in {"private", "team", "public"}:
        return cast("Visibility", value)
    return "private"


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _embedding_version(config: AgentRuntimeConfig) -> str:
    if config.embedding_dimensions is None:
        return config.embedding_model
    return f"{config.embedding_model}:{config.embedding_dimensions}"


__all__ = [
    "DocumentRecord",
    "IngestionJobRecord",
    "KnowledgeBaseRecord",
    "KnowledgeBaseService",
]
