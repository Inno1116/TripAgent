"""Elasticsearch and Milvus indexing for ingested document chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from kyuriagents.rag.types import DocumentChunk
    from kyuriagents.runtime import AgentRuntimeConfig

    class _EmbeddingDocumentsClient(Protocol):
        def embed_documents(self, texts: list[str], *, chunk_size: int) -> list[list[float]]:
            """Embed multiple documents."""
            ...

    class _OpenAIEmbeddingsConstructor(Protocol):
        def __call__(
            self,
            *,
            model: str,
            dimensions: int | None,
            api_key: str,
            base_url: str,
            tiktoken_enabled: bool,
            check_embedding_ctx_length: bool,
            chunk_size: int,
        ) -> _EmbeddingDocumentsClient:
            """Create an embedding client."""
            ...

EmbedDocuments = Callable[[Sequence[str]], list[tuple[float, ...]]]


class _ElasticsearchClient(Protocol):
    def bulk(self, *, operations: Sequence[dict[str, object]], refresh: bool) -> Mapping[str, object]:
        """Bulk index documents."""
        ...

    def delete_by_query(self, *, index: str, query: dict[str, object], conflicts: str, refresh: bool) -> Mapping[str, object]:
        """Delete indexed documents matching a query."""
        ...


class _MilvusClient(Protocol):
    def delete(self, *, collection_name: str, filter: str) -> object:  # noqa: A002  # `filter` is the pymilvus keyword.
        """Delete rows matching a scalar filter."""
        ...

    def flush(self, collection_name: str) -> object:
        """Flush a collection."""
        ...

    def upsert(self, *, collection_name: str, data: Sequence[dict[str, object]]) -> object:
        """Upsert rows into a collection."""
        ...


class HybridChunkIndexer:
    """Index document chunks into Elasticsearch and Milvus."""

    def __init__(
        self,
        *,
        config: AgentRuntimeConfig,
        es_client: _ElasticsearchClient | None = None,
        milvus_client: _MilvusClient | None = None,
        embed_documents: EmbedDocuments | None = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config: Runtime configuration.
            es_client: Optional Elasticsearch-compatible client.
            milvus_client: Optional Milvus-compatible client.
            embed_documents: Optional document embedding function.
        """
        self._config = config
        self._es = es_client
        self._milvus = milvus_client
        self._embed_documents = embed_documents

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Embed and index chunks.

        Args:
            chunks: Prepared document chunks.

        Raises:
            RuntimeError: If Elasticsearch reports bulk indexing errors.
        """
        if not chunks:
            return
        embed_documents = self._embed_documents or _create_embed_documents(
            self._config,
            embedding_batch_size=self._config.ingestion_embedding_batch_size,
        )
        es = self._es or _elasticsearch(self._config)
        milvus = self._milvus or _milvus(self._config)
        embeddings = embed_documents([chunk.text for chunk in chunks])
        _bulk_index_elasticsearch(es, self._config.rag_es_index, chunks)
        _upsert_milvus(milvus, self._config.rag_milvus_collection, chunks, embeddings)
        milvus.flush(collection_name=self._config.rag_milvus_collection)

    def delete_knowledge_base(self, *, tenant_id: str, kb_id: str) -> None:
        """Remove all indexed chunks for one knowledge base."""
        self._delete_by_filter(tenant_id=tenant_id, kb_id=kb_id)

    def delete_document(self, *, tenant_id: str, kb_id: str, doc_id: str) -> None:
        """Remove indexed chunks for one uploaded document."""
        self._delete_by_filter(tenant_id=tenant_id, kb_id=kb_id, doc_id=doc_id)

    def _delete_by_filter(self, *, tenant_id: str, kb_id: str, doc_id: str | None = None) -> None:
        es = self._es or _elasticsearch(self._config)
        milvus = self._milvus or _milvus(self._config)
        es.delete_by_query(
            index=self._config.rag_es_index,
            query=_delete_query(tenant_id=tenant_id, kb_id=kb_id, doc_id=doc_id),
            conflicts="proceed",
            refresh=True,
        )
        milvus.delete(
            collection_name=self._config.rag_milvus_collection,
            filter=_milvus_delete_filter(tenant_id=tenant_id, kb_id=kb_id, doc_id=doc_id),
        )
        milvus.flush(collection_name=self._config.rag_milvus_collection)


def _bulk_index_elasticsearch(es: _ElasticsearchClient, index: str, chunks: Sequence[DocumentChunk]) -> None:
    operations: list[dict[str, object]] = []
    for chunk in chunks:
        document = chunk.metadata.to_es_document(chunk.text)
        document["keywords"] = list(chunk.keywords)
        operations.append({"index": {"_index": index, "_id": chunk.metadata.chunk_id}})
        operations.append(document)
    response = es.bulk(operations=operations, refresh=False)
    if response.get("errors"):
        msg = f"Elasticsearch bulk indexing failed: {response}"
        raise RuntimeError(msg)


def _upsert_milvus(
    milvus: _MilvusClient,
    collection_name: str,
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
) -> None:
    rows = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        row = chunk.metadata.to_milvus_fields()
        row["embedding"] = [float(value) for value in embedding]
        rows.append(row)
    milvus.upsert(collection_name=collection_name, data=rows)


def _delete_query(*, tenant_id: str, kb_id: str, doc_id: str | None) -> dict[str, object]:
    filters: list[dict[str, object]] = [
        {"term": {"tenant_id": tenant_id}},
        {"term": {"kb_id": kb_id}},
    ]
    if doc_id is not None:
        filters.append({"term": {"doc_id": doc_id}})
    return {"bool": {"filter": filters}}


def _milvus_delete_filter(*, tenant_id: str, kb_id: str, doc_id: str | None) -> str:
    filters = [f'tenant_id == "{_escape_milvus_value(tenant_id)}"', f'kb_id == "{_escape_milvus_value(kb_id)}"']
    if doc_id is not None:
        filters.append(f'doc_id == "{_escape_milvus_value(doc_id)}"')
    return " and ".join(filters)


def _escape_milvus_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _create_embed_documents(config: AgentRuntimeConfig, *, embedding_batch_size: int) -> EmbedDocuments:
    if not config.dashscope_api_key:
        msg = "Set `DASHSCOPE_API_KEY` before embedding ingested documents."
        raise ValueError(msg)
    try:
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` to use DashScope embeddings."
        raise ImportError(msg) from exc

    embedding_cls = cast("_OpenAIEmbeddingsConstructor", OpenAIEmbeddings)
    embeddings = embedding_cls(
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_base_url,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        chunk_size=embedding_batch_size,
    )

    def embed_documents(texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [
            tuple(float(value) for value in vector)
            for vector in embeddings.embed_documents(
                list(texts),
                chunk_size=embedding_batch_size,
            )
        ]

    return embed_documents


def _elasticsearch(config: AgentRuntimeConfig) -> _ElasticsearchClient:
    from elasticsearch import Elasticsearch  # noqa: PLC0415

    return cast("_ElasticsearchClient", Elasticsearch(config.rag_es_url))


def _milvus(config: AgentRuntimeConfig) -> _MilvusClient:
    from pymilvus import MilvusClient  # noqa: PLC0415

    return cast(
        "_MilvusClient",
        MilvusClient(
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token or "",
            db_name=config.rag_milvus_db or "",
        ),
    )


__all__ = ["EmbedDocuments", "HybridChunkIndexer"]
