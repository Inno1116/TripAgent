"""Local StratRAG indexing and retrieval evaluation helper."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from kyuriagents.rag import (
    DashScopeTextReranker,
    ElasticsearchKeywordStore,
    HybridRAGRetriever,
    MilvusVectorStore,
    PostgresChunkTextHydrator,
    evaluate_stratrag_retriever,
    load_stratrag_jsonl,
    stratrag_chunks,
)
from kyuriagents.runtime import AgentRuntimeConfig
from kyuriagents.runtime.dashscope import create_dashscope_embed_query

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from kyuriagents.rag import StratRAGExample
    from kyuriagents.rag.types import DocumentChunk

_T = TypeVar("_T")
_LOGGER = logging.getLogger("stratrag_rag_eval")
_EXPECTED_GOLD_DOCS = 2

EmbedDocuments = Callable[[Sequence[str]], list[tuple[float, ...]]]


class _ElasticsearchIndices(Protocol):
    def exists(self, *, index: str) -> object:
        """Return whether an index exists."""
        ...

    def delete(self, *, index: str) -> object:
        """Delete an index."""
        ...

    def create(
        self,
        *,
        index: str,
        settings: object,
        mappings: object,
    ) -> object:
        """Create an index."""
        ...

    def refresh(self, *, index: str) -> object:
        """Refresh an index."""
        ...


class _ElasticsearchClient(Protocol):
    indices: _ElasticsearchIndices

    def bulk(self, *, operations: Sequence[dict[str, object]], refresh: bool) -> Mapping[str, object]:
        """Bulk index documents."""
        ...


class _MilvusSchema(Protocol):
    def add_field(self, field_name: str, datatype: object, **kwargs: object) -> None:
        """Add a field to a collection schema."""
        ...


class _MilvusIndexParams(Protocol):
    def add_index(self, field_name: str, index_type: str, **kwargs: object) -> None:
        """Add an index definition."""
        ...


class _MilvusClient(Protocol):
    def has_collection(self, collection_name: str) -> object:
        """Return whether a collection exists."""
        ...

    def drop_collection(self, collection_name: str) -> object:
        """Drop a collection."""
        ...

    def prepare_index_params(self) -> _MilvusIndexParams:
        """Create index parameter container."""
        ...

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: _MilvusSchema,
        index_params: _MilvusIndexParams,
    ) -> object:
        """Create a collection."""
        ...

    def load_collection(self, collection_name: str) -> object:
        """Load a collection."""
        ...

    def flush(self, collection_name: str) -> object:
        """Flush a collection."""
        ...

    def upsert(self, *, collection_name: str, data: Sequence[dict[str, object]]) -> object:
        """Upsert rows into a collection."""
        ...


def main() -> None:
    """Run the StratRAG helper CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Index and evaluate StratRAG with the configured RAG runtime.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create Elasticsearch index and Milvus collection.")
    init.add_argument("--reset", action="store_true", help="Drop existing ES index and Milvus collection first.")

    ingest = subparsers.add_parser("ingest", help="Embed and upsert StratRAG chunks into ES and Milvus.")
    ingest.add_argument("--path", default="StratRAG/val.jsonl", help="Path to StratRAG JSONL.")
    ingest.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    ingest.add_argument("--batch-size", type=int, default=20, help="Local ES/Milvus upsert batch size.")
    ingest.add_argument("--embedding-batch-size", type=int, default=10, help="DashScope embedding request batch size.")

    evaluate = subparsers.add_parser("eval", help="Evaluate the configured hybrid retriever on StratRAG.")
    evaluate.add_argument("--path", default="StratRAG/val.jsonl", help="Path to StratRAG JSONL.")
    evaluate.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    evaluate.add_argument("--top-k", type=int, default=5, help="Retrieval depth.")
    evaluate.add_argument(
        "--gold-policy",
        choices=("dataset", "skip-empty", "first-two"),
        default="dataset",
        help="How to handle missing gold_doc_indices in local StratRAG files.",
    )

    args = parser.parse_args()
    config = AgentRuntimeConfig.from_env()

    if args.command == "init":
        init_indexes(config, reset=cast("bool", args.reset))
    elif args.command == "ingest":
        ingest_stratrag(
            config,
            path=Path(cast("str", args.path)),
            limit=cast("int | None", args.limit),
            batch_size=cast("int", args.batch_size),
            embedding_batch_size=cast("int", args.embedding_batch_size),
        )
    elif args.command == "eval":
        evaluate_stratrag(
            config,
            path=Path(cast("str", args.path)),
            limit=cast("int | None", args.limit),
            top_k=cast("int", args.top_k),
            gold_policy=cast("str", args.gold_policy),
        )


def init_indexes(config: AgentRuntimeConfig, *, reset: bool = False) -> None:
    """Create Elasticsearch index and Milvus collection for RAG chunks.

    Args:
        config: Runtime configuration.
        reset: Whether to drop existing objects before creating them.
    """
    _init_elasticsearch(config, reset=reset)
    _init_milvus(config, reset=reset)


def ingest_stratrag(
    config: AgentRuntimeConfig,
    *,
    path: Path,
    limit: int | None = None,
    batch_size: int = 32,
    embedding_batch_size: int = 10,
) -> None:
    """Embed and index StratRAG chunks into Elasticsearch and Milvus.

    Args:
        config: Runtime configuration.
        path: StratRAG JSONL path.
        limit: Optional maximum number of examples to ingest.
        batch_size: Local Elasticsearch and Milvus upsert batch size.
        embedding_batch_size: Maximum texts per DashScope embedding request.
    """
    _positive("batch_size", batch_size)
    _positive("embedding_batch_size", embedding_batch_size)
    examples = load_stratrag_jsonl(path, limit=limit)
    chunks = stratrag_chunks(
        examples,
        tenant_id=config.tenant_id,
        embedding_model=config.embedding_model,
        embedding_version=_embedding_version(config),
    )
    _LOGGER.info("loaded examples=%s chunks=%s", len(examples), len(chunks))

    es = _elasticsearch(config)
    milvus = _milvus(config)
    embed_documents = _create_embed_documents(config, embedding_batch_size=embedding_batch_size)

    indexed = 0
    for batch in _batched(chunks, batch_size):
        embeddings = embed_documents([chunk.text for chunk in batch])
        _bulk_index_elasticsearch(es, config.rag_es_index, batch)
        _upsert_milvus(milvus, config.rag_milvus_collection, batch, embeddings)
        _upsert_postgres_chunks(config, batch)
        indexed += len(batch)
        _LOGGER.info("indexed chunks=%s/%s", indexed, len(chunks))

    milvus.flush(collection_name=config.rag_milvus_collection)
    milvus.load_collection(collection_name=config.rag_milvus_collection)
    es.indices.refresh(index=config.rag_es_index)
    _LOGGER.info("ingest complete")


def evaluate_stratrag(
    config: AgentRuntimeConfig,
    *,
    path: Path,
    limit: int | None = None,
    top_k: int = 5,
    gold_policy: str = "dataset",
) -> None:
    """Evaluate hybrid retrieval on StratRAG and print aggregate metrics.

    Args:
        config: Runtime configuration.
        path: StratRAG JSONL path.
        limit: Optional maximum number of examples to evaluate.
        top_k: Retrieval depth.
        gold_policy: How to handle missing gold annotations.
    """
    examples = _apply_gold_policy(load_stratrag_jsonl(path, limit=limit), gold_policy)
    if not examples:
        msg = "No StratRAG examples available after applying gold policy."
        raise ValueError(msg)
    retriever = HybridRAGRetriever(
        vector_searcher=MilvusVectorStore(
            collection_name=config.rag_milvus_collection,
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token,
            db_name=config.rag_milvus_db,
            embed_query=create_dashscope_embed_query(config),
            search_params=_milvus_search_params(),
        ),
        keyword_searcher=ElasticsearchKeywordStore(
            index=config.rag_es_index,
            url=config.rag_es_url,
        ),
        chunk_hydrator=PostgresChunkTextHydrator(dsn=config.postgres_dsn) if config.postgres_dsn else None,
        reranker=DashScopeTextReranker(
            api_key=config.dashscope_api_key or "",
            model=config.rag_rerank_model,
            endpoint=config.rag_rerank_url,
            timeout_seconds=config.rag_rerank_timeout_seconds,
        )
        if config.rag_rerank_model
        else None,
    )
    result = evaluate_stratrag_retriever(
        examples,
        retriever,
        tenant_id=config.tenant_id,
        top_k=top_k,
    )
    _LOGGER.info(json.dumps(result.overall.to_dict(), ensure_ascii=False, indent=2))
    for question_type, metrics in sorted(result.by_question_type.items()):
        _LOGGER.info("%s %s", question_type, json.dumps(metrics.to_dict(), ensure_ascii=False, sort_keys=True))


def _apply_gold_policy(examples: Sequence[StratRAGExample], policy: str) -> list[StratRAGExample]:
    if policy == "dataset":
        return list(examples)
    if policy == "skip-empty":
        return [example for example in examples if example.gold_doc_indices]
    if policy == "first-two":
        return [replace(example, gold_doc_indices=(0, 1)) for example in examples if len(example.doc_pool) >= _EXPECTED_GOLD_DOCS]
    msg = "`gold_policy` must be one of: dataset, skip-empty, first-two."
    raise ValueError(msg)


def _init_elasticsearch(config: AgentRuntimeConfig, *, reset: bool) -> None:
    spec = _json_resource("kyuriagents.rag", "schemas/elasticsearch_index.json")
    es = _elasticsearch(config)
    exists = bool(es.indices.exists(index=config.rag_es_index))
    if exists and reset:
        es.indices.delete(index=config.rag_es_index)
        exists = False
    if exists:
        _LOGGER.info("elasticsearch index exists: %s", config.rag_es_index)
        return
    es.indices.create(
        index=config.rag_es_index,
        settings=spec["settings"],
        mappings=spec["mappings"],
    )
    _LOGGER.info("created elasticsearch index: %s", config.rag_es_index)


def _init_milvus(config: AgentRuntimeConfig, *, reset: bool) -> None:
    if config.embedding_dimensions is None:
        msg = "Set DASHSCOPE_EMBEDDING_DIMENSIONS before creating the Milvus collection."
        raise ValueError(msg)
    milvus = _milvus(config)
    exists = bool(milvus.has_collection(config.rag_milvus_collection))
    if exists and reset:
        milvus.drop_collection(config.rag_milvus_collection)
        exists = False
    if exists:
        _LOGGER.info("milvus collection exists: %s", config.rag_milvus_collection)
        return

    schema = _milvus_schema(config.embedding_dimensions)
    index_params = milvus.prepare_index_params()
    embedding = cast("dict[str, object]", _json_resource("kyuriagents.rag", "schemas/milvus_collection.json")["embedding"])
    index_params.add_index(
        field_name=str(embedding["field_name"]),
        index_type=str(embedding["index_type"]),
        metric_type=embedding["metric_type"],
        params=embedding["index_params"],
    )
    milvus.create_collection(
        collection_name=config.rag_milvus_collection,
        schema=schema,
        index_params=index_params,
    )
    milvus.load_collection(config.rag_milvus_collection)
    _LOGGER.info("created milvus collection: %s", config.rag_milvus_collection)


def _milvus_schema(dimension: int) -> _MilvusSchema:
    from pymilvus import DataType, MilvusClient  # noqa: PLC0415

    spec = _json_resource("kyuriagents.rag", "schemas/milvus_collection.json")
    schema = cast("_MilvusSchema", MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False))
    fields = cast("list[dict[str, object]]", spec["fields"])
    for field in fields:
        kwargs: dict[str, object] = {}
        if field.get("is_primary"):
            kwargs["is_primary"] = True
        if "max_length" in field:
            kwargs["max_length"] = field["max_length"]
        if "max_capacity" in field:
            kwargs["max_capacity"] = field["max_capacity"]
        if "element_type" in field:
            kwargs["element_type"] = _milvus_data_type(str(field["element_type"]))
        schema.add_field(
            field_name=str(field["name"]),
            datatype=_milvus_data_type(str(field["type"])),
            **kwargs,
        )
    embedding = cast("dict[str, object]", spec["embedding"])
    schema.add_field(
        field_name=str(embedding["field_name"]),
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )
    return schema


def _milvus_data_type(value: str) -> object:
    from pymilvus import DataType  # noqa: PLC0415

    return getattr(DataType, value)


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


def _upsert_postgres_chunks(config: AgentRuntimeConfig, chunks: Sequence[DocumentChunk]) -> None:
    if not config.postgres_dsn:
        return
    try:
        import psycopg  # noqa: PLC0415
        from psycopg.types.json import Jsonb  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to write StratRAG chunk text into PostgreSQL."
        raise ImportError(msg) from exc

    with psycopg.connect(config.postgres_dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO rag_tenants (tenant_id, name)
            VALUES (%(tenant_id)s, %(name)s)
            ON CONFLICT (tenant_id) DO UPDATE SET name = EXCLUDED.name
            """,
            {"tenant_id": config.tenant_id, "name": "StratRAG evaluation"},
        )
        for chunk in chunks:
            metadata = chunk.metadata
            cursor.execute(
                """
                INSERT INTO rag_knowledge_bases (kb_id, tenant_id, name, visibility, status, metadata)
                VALUES (%(kb_id)s, %(tenant_id)s, %(name)s, 'public', 'active', %(metadata)s)
                ON CONFLICT (kb_id) DO UPDATE
                SET name = EXCLUDED.name,
                    visibility = EXCLUDED.visibility,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                {
                    "kb_id": metadata.kb_id,
                    "tenant_id": metadata.tenant_id,
                    "name": metadata.kb_id,
                    "metadata": Jsonb({"source": "stratrag"}),
                },
            )
            cursor.execute(
                """
                INSERT INTO rag_documents (
                    doc_id, tenant_id, kb_id, source_type, source_uri, file_name,
                    title, language, visibility, status, latest_version, metadata
                )
                VALUES (
                    %(doc_id)s, %(tenant_id)s, %(kb_id)s, %(source_type)s, %(source_uri)s, %(file_name)s,
                    %(title)s, %(language)s, %(visibility)s, 'active', %(doc_version)s, %(metadata)s
                )
                ON CONFLICT (doc_id) DO UPDATE
                SET title = EXCLUDED.title,
                    language = EXCLUDED.language,
                    visibility = EXCLUDED.visibility,
                    status = EXCLUDED.status,
                    latest_version = EXCLUDED.latest_version,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                {
                    "doc_id": metadata.doc_id,
                    "tenant_id": metadata.tenant_id,
                    "kb_id": metadata.kb_id,
                    "source_type": metadata.source_type,
                    "source_uri": metadata.source_uri,
                    "file_name": metadata.doc_id,
                    "title": metadata.title,
                    "language": metadata.language,
                    "visibility": metadata.visibility,
                    "doc_version": metadata.doc_version,
                    "metadata": Jsonb({"source": "stratrag"}),
                },
            )
            cursor.execute(
                """
                INSERT INTO rag_document_versions (
                    doc_version, doc_id, tenant_id, content_hash, parser_version,
                    chunker_version, embedding_model, embedding_version, chunk_count, status, indexed_at
                )
                VALUES (
                    %(doc_version)s, %(doc_id)s, %(tenant_id)s, %(content_hash)s, 'stratrag:v1',
                    'stratrag-doc-pool:v1', %(embedding_model)s, %(embedding_version)s, 1, 'indexed', now()
                )
                ON CONFLICT (doc_id, content_hash, embedding_model, embedding_version) DO UPDATE
                SET status = 'indexed',
                    chunk_count = EXCLUDED.chunk_count,
                    indexed_at = now()
                """,
                {
                    "doc_version": metadata.doc_version,
                    "doc_id": metadata.doc_id,
                    "tenant_id": metadata.tenant_id,
                    "content_hash": metadata.content_hash,
                    "embedding_model": metadata.embedding_model,
                    "embedding_version": metadata.embedding_version,
                },
            )
            values = metadata.to_milvus_fields()
            values["chunk_text"] = chunk.text
            values["tags"] = Jsonb(list(metadata.tags))
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
                    tags = EXCLUDED.tags,
                    is_active = EXCLUDED.is_active,
                    updated_at = now()
                """,
                values,
            )


def _create_embed_documents(config: AgentRuntimeConfig, *, embedding_batch_size: int) -> EmbedDocuments:
    if not config.dashscope_api_key:
        msg = "Set DASHSCOPE_API_KEY before embedding StratRAG."
        raise ValueError(msg)
    try:
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` to use DashScope embeddings."
        raise ImportError(msg) from exc

    embeddings = OpenAIEmbeddings(
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


def _milvus_search_params() -> dict[str, object]:
    embedding = cast("dict[str, object]", _json_resource("kyuriagents.rag", "schemas/milvus_collection.json")["embedding"])
    return {
        "metric_type": embedding["metric_type"],
        "params": embedding["search_params"],
    }


def _embedding_version(config: AgentRuntimeConfig) -> str:
    if config.embedding_dimensions is None:
        return config.embedding_model
    return f"{config.embedding_model}:{config.embedding_dimensions}"


def _json_resource(package: str, path: str) -> dict[str, object]:
    text = resources.files(package).joinpath(path).read_text(encoding="utf-8")
    return cast("dict[str, object]", json.loads(text))


def _elasticsearch(config: AgentRuntimeConfig) -> _ElasticsearchClient:
    from elasticsearch import Elasticsearch  # noqa: PLC0415

    return cast("_ElasticsearchClient", Elasticsearch(config.rag_es_url))


def _milvus(config: AgentRuntimeConfig) -> _MilvusClient:
    from pymilvus import MilvusClient  # noqa: PLC0415

    return cast(
        "_MilvusClient",
        MilvusClient(
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token,
            db_name=config.rag_milvus_db,
        ),
    )


def _batched(items: Sequence[_T], size: int) -> Iterable[Sequence[_T]]:
    _positive("size", size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _positive(name: str, value: int) -> None:
    if value <= 0:
        msg = f"`{name}` must be positive."
        raise ValueError(msg)


if __name__ == "__main__":
    main()
