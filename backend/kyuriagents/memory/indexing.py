"""Hybrid retrieval helpers for long-term memory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from kyuriagents.memory.types import MemoryRecord, MemoryScope, MemorySearchResult, MemoryStore
from kyuriagents.rag import HybridRAGRetriever, RetrievalScope, RetrievedChunk

if TYPE_CHECKING:
    from kyuriagents.rag.types import DocumentChunk

EmbedText = Callable[[str], Sequence[float]]
"""Callable that embeds memory text."""

_HTTP_NOT_FOUND = 404


class MemoryIndexer(Protocol):
    """Protocol for synchronizing memories into a retrieval index."""

    def upsert(self, memories: Sequence[MemoryRecord]) -> None:
        """Index or replace memory records.

        Args:
            memories: Records to index.
        """
        ...

    def delete(self, memory_ids: Sequence[str]) -> None:
        """Remove memory records from the retrieval index.

        Args:
            memory_ids: Memory identifiers to remove.
        """
        ...


class _ElasticsearchBulkClient(Protocol):
    def bulk(self, *, operations: Sequence[dict[str, object]], refresh: bool) -> dict[str, object]:
        """Run an Elasticsearch bulk request."""
        ...


class _MilvusWriteClient(Protocol):
    def upsert(self, *, collection_name: str, data: Sequence[dict[str, object]]) -> object:
        """Upsert rows into a Milvus collection."""
        ...

    def delete(self, **kwargs: object) -> object:
        """Delete rows from a Milvus collection."""
        ...


class ElasticsearchMilvusMemoryIndexer:
    """Synchronize long-term memories into Elasticsearch and Milvus."""

    def __init__(
        self,
        *,
        es_index: str,
        milvus_collection: str,
        embed_text: EmbedText,
        es_url: str = "http://localhost:9200",
        milvus_uri: str = "http://localhost:19530",
        milvus_token: str | None = None,
        milvus_db: str | None = None,
        es_client: _ElasticsearchBulkClient | None = None,
        milvus_client: _MilvusWriteClient | None = None,
        refresh: bool = False,
    ) -> None:
        """Initialize the memory indexer.

        Args:
            es_index: Elasticsearch index for memory chunks.
            milvus_collection: Milvus collection for memory vectors.
            embed_text: Embedding function for memory text.
            es_url: Elasticsearch URL.
            milvus_uri: Milvus URI.
            milvus_token: Optional Milvus token.
            milvus_db: Optional Milvus database.
            es_client: Optional preconfigured Elasticsearch-compatible client.
            milvus_client: Optional preconfigured Milvus-compatible client.
            refresh: Whether Elasticsearch should refresh after bulk writes.
        """
        self._es_index = es_index
        self._milvus_collection = milvus_collection
        self._embed_text = embed_text
        self._es_client = es_client if es_client is not None else _build_elasticsearch_client(es_url)
        self._milvus_client = milvus_client if milvus_client is not None else _build_milvus_client(milvus_uri, milvus_token, milvus_db)
        self._refresh = refresh

    def upsert(self, memories: Sequence[MemoryRecord]) -> None:
        """Index or replace memory records.

        Args:
            memories: Records to index.
        """
        if not memories:
            return
        chunks = memory_records_to_chunks(memories, embed_text=self._embed_text)
        _bulk_index_elasticsearch(self._es_client, self._es_index, chunks, refresh=self._refresh)
        _upsert_milvus(self._milvus_client, self._milvus_collection, chunks)

    def delete(self, memory_ids: Sequence[str]) -> None:
        """Remove memory records from Elasticsearch and Milvus.

        Args:
            memory_ids: Memory identifiers to remove.
        """
        if not memory_ids:
            return
        chunk_ids = tuple(f"memory:{memory_id}" for memory_id in memory_ids)
        _bulk_delete_elasticsearch(self._es_client, self._es_index, chunk_ids, refresh=self._refresh)
        self._milvus_client.delete(
            collection_name=self._milvus_collection,
            filter=_milvus_chunk_id_filter(chunk_ids),
        )


def memory_records_to_chunks(
    memories: Sequence[MemoryRecord],
    *,
    embed_text: EmbedText | None = None,
) -> list[DocumentChunk]:
    """Convert memory records to RAG chunks.

    Args:
        memories: Memory records to index.
        embed_text: Optional embedding function.

    Returns:
        Document chunks suitable for Milvus and Elasticsearch indexing.
    """
    chunks: list[DocumentChunk] = []
    for memory in memories:
        embedding = ()
        if embed_text is not None:
            embedding = tuple(float(value) for value in embed_text(memory.index_text))
        chunks.append(memory.to_document_chunk(embedding=embedding))
    return chunks


class MemoryHybridSearcher:
    """Search indexed memory chunks with an existing hybrid RAG retriever."""

    def __init__(self, *, retriever: HybridRAGRetriever, store: MemoryStore) -> None:
        """Initialize the searcher.

        Args:
            retriever: Hybrid retriever configured over the memory index.
            store: Source-of-truth memory store for loading records.
        """
        self._retriever = retriever
        self._store = store

    def search(self, query: str, *, scope: MemoryScope, limit: int) -> list[MemorySearchResult]:
        """Search long-term memory through hybrid retrieval.

        Args:
            query: User query.
            scope: Memory scope.
            limit: Maximum results.

        Returns:
            Ranked memory search results.
        """
        chunks = self._retriever.retrieve(
            query,
            scope=_retrieval_scope(scope),
            top_k=limit,
        )
        return self._chunks_to_results(chunks, scope=scope)

    def _chunks_to_results(self, chunks: list[RetrievedChunk], *, scope: MemoryScope) -> list[MemorySearchResult]:
        results: list[MemorySearchResult] = []
        seen: set[str] = set()
        for chunk in chunks:
            memory_id = chunk.metadata.doc_id
            if memory_id in seen:
                continue
            seen.add(memory_id)
            memory = self._store.get(memory_id, scope=scope)
            if memory is None:
                continue
            score = _chunk_score(chunk) + memory.importance * 0.10 + memory.confidence * 0.05
            results.append(
                MemorySearchResult(
                    memory=memory,
                    score=score,
                    lexical_score=chunk.keyword_score,
                    semantic_score=chunk.vector_score,
                )
            )
        return results


def _retrieval_scope(scope: MemoryScope) -> RetrievalScope:
    kb_ids = tuple(f"memory:{scope_type}:{scope_id}" for scope_type in scope.scope_types for scope_id in scope.scope_ids)
    return RetrievalScope(
        tenant_id=scope.tenant_id,
        user_id=scope.user_id,
        kb_ids=kb_ids,
        source_types=("memory",),
        tags=scope.tags,
        visibility=scope.visibility,
        active_only=scope.active_only,
    )


def _chunk_score(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is not None:
        return chunk.rerank_score
    if chunk.fused_score:
        return chunk.fused_score
    if chunk.vector_score is not None:
        return chunk.vector_score
    if chunk.keyword_score is not None:
        return chunk.keyword_score
    return 0.0


def _bulk_index_elasticsearch(
    es: _ElasticsearchBulkClient,
    index: str,
    chunks: Sequence[DocumentChunk],
    *,
    refresh: bool,
) -> None:
    operations: list[dict[str, object]] = []
    for chunk in chunks:
        document = chunk.metadata.to_es_document(chunk.text)
        document["keywords"] = list(chunk.keywords)
        operations.append({"index": {"_index": index, "_id": chunk.metadata.chunk_id}})
        operations.append(document)
    response = es.bulk(operations=operations, refresh=refresh)
    _raise_for_bulk_errors(response, action="Elasticsearch memory indexing", allow_not_found=False)


def _bulk_delete_elasticsearch(
    es: _ElasticsearchBulkClient,
    index: str,
    chunk_ids: Sequence[str],
    *,
    refresh: bool,
) -> None:
    operations = [_delete_operation(index, chunk_id) for chunk_id in chunk_ids]
    response = es.bulk(operations=operations, refresh=refresh)
    _raise_for_bulk_errors(response, action="Elasticsearch memory deletion", allow_not_found=True)


def _upsert_milvus(
    milvus: _MilvusWriteClient,
    collection_name: str,
    chunks: Sequence[DocumentChunk],
) -> None:
    rows = []
    for chunk in chunks:
        row = chunk.metadata.to_milvus_fields()
        row["embedding"] = [float(value) for value in chunk.embedding]
        rows.append(row)
    milvus.upsert(collection_name=collection_name, data=rows)


def _raise_for_bulk_errors(response: dict[str, object], *, action: str, allow_not_found: bool) -> None:
    if not response.get("errors"):
        return
    items = cast("Sequence[dict[str, object]]", response.get("items", ()))
    if items and all(_allowed_bulk_item_error(item, allow_not_found=allow_not_found) for item in items):
        return
    msg = f"{action} failed: {response}"
    raise RuntimeError(msg)


def _allowed_bulk_item_error(item: dict[str, object], *, allow_not_found: bool) -> bool:
    if not allow_not_found:
        return False
    payload = next(iter(item.values()), {})
    if not isinstance(payload, dict):
        return False
    entry = cast("dict[str, object]", payload)
    status = entry.get("status")
    if not isinstance(status, int | str):
        return False
    return int(status) == _HTTP_NOT_FOUND


def _delete_operation(index: str, chunk_id: str) -> dict[str, object]:
    return {"delete": {"_index": index, "_id": chunk_id}}


def _milvus_chunk_id_filter(chunk_ids: Sequence[str]) -> str:
    quoted = ", ".join(f'"{_escape_milvus_string(chunk_id)}"' for chunk_id in chunk_ids)
    return f"chunk_id in [{quoted}]"


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_elasticsearch_client(url: str) -> _ElasticsearchBulkClient:
    try:
        from elasticsearch import Elasticsearch  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[rag]` or pass an Elasticsearch client."
        raise ImportError(msg) from exc
    return cast("_ElasticsearchBulkClient", Elasticsearch(url))


def _build_milvus_client(uri: str, token: str | None, db_name: str | None) -> _MilvusWriteClient:
    try:
        from pymilvus import MilvusClient  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[rag]` or pass a Milvus client."
        raise ImportError(msg) from exc
    if token is not None and db_name is not None:
        return cast("_MilvusWriteClient", MilvusClient(uri=uri, token=token, db_name=db_name))
    if token is not None:
        return cast("_MilvusWriteClient", MilvusClient(uri=uri, token=token))
    if db_name is not None:
        return cast("_MilvusWriteClient", MilvusClient(uri=uri, db_name=db_name))
    return cast("_MilvusWriteClient", MilvusClient(uri=uri))


__all__ = [
    "ElasticsearchMilvusMemoryIndexer",
    "EmbedText",
    "MemoryHybridSearcher",
    "MemoryIndexer",
    "memory_records_to_chunks",
]
