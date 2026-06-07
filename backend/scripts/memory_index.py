"""Create and rebuild Elasticsearch/Milvus indexes for long-term memory."""

from __future__ import annotations

import argparse
import json
import logging
from importlib import resources
from typing import TYPE_CHECKING, Protocol, cast

from kyuriagents.memory import (
    ElasticsearchMilvusMemoryIndexer,
    MemoryScope,
    PostgresMemoryStore,
)
from kyuriagents.runtime import AgentRuntimeConfig
from kyuriagents.runtime.dashscope import create_dashscope_embed_query

if TYPE_CHECKING:
    from kyuriagents.memory.types import MemoryScopeType

_LOGGER = logging.getLogger("memory_index")
_DEFAULT_REINDEX_LIMIT = 1_000


class _ElasticsearchIndices(Protocol):
    def exists(self, *, index: str) -> bool:
        """Return whether an Elasticsearch index exists."""
        ...

    def create(self, *, index: str, **kwargs: object) -> object:
        """Create an Elasticsearch index."""
        ...

    def delete(self, *, index: str) -> object:
        """Delete an Elasticsearch index."""
        ...


class _ElasticsearchClient(Protocol):
    indices: _ElasticsearchIndices


class _MilvusSchema(Protocol):
    def add_field(self, **kwargs: object) -> object:
        """Add a field to a Milvus collection schema."""
        ...


class _MilvusIndexParams(Protocol):
    def add_index(self, **kwargs: object) -> object:
        """Add an index definition."""
        ...


class _MilvusClient(Protocol):
    def has_collection(self, collection_name: str) -> bool:
        """Return whether a Milvus collection exists."""
        ...

    def drop_collection(self, collection_name: str) -> object:
        """Drop a Milvus collection."""
        ...

    def prepare_index_params(self) -> _MilvusIndexParams:
        """Create empty Milvus index params."""
        ...

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: _MilvusSchema,
        index_params: _MilvusIndexParams,
    ) -> object:
        """Create a Milvus collection."""
        ...

    def load_collection(self, collection_name: str) -> object:
        """Load a Milvus collection."""
        ...


def main() -> None:
    """Run the memory index maintenance command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Initialize or rebuild memory ES/Milvus indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create memory Elasticsearch index and Milvus collection.")
    init.add_argument("--reset", action="store_true", help="Drop existing memory indexes first.")

    reindex = subparsers.add_parser("reindex", help="Reindex visible PostgreSQL memory records.")
    reindex.add_argument("--limit", type=int, default=_DEFAULT_REINDEX_LIMIT, help="Maximum memory records to reindex.")
    reindex.add_argument("--scope-type", action="append", default=[], help="Optional scope type filter.")
    reindex.add_argument("--scope-id", action="append", default=[], help="Optional scope id filter.")

    args = parser.parse_args()
    config = AgentRuntimeConfig.from_env()
    if args.command == "init":
        init_memory_indexes(config, reset=bool(args.reset))
    elif args.command == "reindex":
        reindex_memory(
            config,
            limit=int(args.limit),
            scope_types=tuple(str(value) for value in args.scope_type),
            scope_ids=tuple(str(value) for value in args.scope_id),
        )


def init_memory_indexes(config: AgentRuntimeConfig, *, reset: bool = False) -> None:
    """Create the Elasticsearch and Milvus indexes used by memory retrieval.

    Args:
        config: Runtime configuration.
        reset: Whether to drop existing indexes before creating them.
    """
    _create_elasticsearch_index(_elasticsearch(config), config.memory_es_index, reset=reset)
    _create_milvus_collection(_milvus(config), config.memory_milvus_collection, config.embedding_dimensions, reset=reset)


def reindex_memory(
    config: AgentRuntimeConfig,
    *,
    limit: int,
    scope_types: tuple[str, ...] = (),
    scope_ids: tuple[str, ...] = (),
) -> None:
    """Reindex active PostgreSQL memory records into Elasticsearch and Milvus.

    Args:
        config: Runtime configuration.
        limit: Maximum memory records to reindex.
        scope_types: Optional memory scope type filters.
        scope_ids: Optional memory scope id filters.
    """
    if config.postgres_dsn is None:
        msg = "Set DEEPAGENTS_POSTGRES_DSN before reindexing memory."
        raise ValueError(msg)
    scope = MemoryScope(
        tenant_id=config.tenant_id,
        user_id=config.user_id,
        scope_types=cast("tuple[MemoryScopeType, ...]", scope_types),
        scope_ids=scope_ids,
    )
    store = PostgresMemoryStore(dsn=config.postgres_dsn)
    memories = store.list_memories(scope=scope, limit=limit)
    indexer = ElasticsearchMilvusMemoryIndexer(
        es_index=config.memory_es_index,
        es_url=config.rag_es_url,
        milvus_collection=config.memory_milvus_collection,
        milvus_uri=config.rag_milvus_uri,
        milvus_token=config.rag_milvus_token,
        milvus_db=config.rag_milvus_db,
        embed_text=create_dashscope_embed_query(config),
        refresh=True,
    )
    indexer.upsert(memories)
    _LOGGER.info("reindexed memory records: %s", len(memories))


def _create_elasticsearch_index(es: _ElasticsearchClient, index: str, *, reset: bool) -> None:
    exists = bool(es.indices.exists(index=index))
    if exists and reset:
        es.indices.delete(index=index)
        exists = False
    if exists:
        _LOGGER.info("elasticsearch memory index exists: %s", index)
        return
    spec = _json_resource("kyuriagents.rag", "schemas/elasticsearch_index.json")
    es.indices.create(
        index=index,
        settings=spec["settings"],
        mappings=spec["mappings"],
    )
    _LOGGER.info("created elasticsearch memory index: %s", index)


def _create_milvus_collection(
    milvus: _MilvusClient,
    collection_name: str,
    dimension: int | None,
    *,
    reset: bool,
) -> None:
    if dimension is None:
        msg = "Set DASHSCOPE_EMBEDDING_DIMENSIONS before creating the Milvus memory collection."
        raise ValueError(msg)
    exists = bool(milvus.has_collection(collection_name))
    if exists and reset:
        milvus.drop_collection(collection_name)
        exists = False
    if exists:
        _LOGGER.info("milvus memory collection exists: %s", collection_name)
        return

    schema = _milvus_schema(dimension)
    index_params = milvus.prepare_index_params()
    embedding = cast("dict[str, object]", _json_resource("kyuriagents.rag", "schemas/milvus_collection.json")["embedding"])
    index_params.add_index(
        field_name=str(embedding["field_name"]),
        index_type=str(embedding["index_type"]),
        metric_type=embedding["metric_type"],
        params=embedding["index_params"],
    )
    milvus.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    milvus.load_collection(collection_name)
    _LOGGER.info("created milvus memory collection: %s", collection_name)


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


def _json_resource(package: str, path: str) -> dict[str, object]:
    text = resources.files(package).joinpath(path).read_text(encoding="utf-8")
    return cast("dict[str, object]", json.loads(text))


def _elasticsearch(config: AgentRuntimeConfig) -> _ElasticsearchClient:
    from elasticsearch import Elasticsearch  # noqa: PLC0415

    return cast("_ElasticsearchClient", Elasticsearch(config.rag_es_url))


def _milvus(config: AgentRuntimeConfig) -> _MilvusClient:
    from pymilvus import MilvusClient  # noqa: PLC0415

    if config.rag_milvus_token is not None and config.rag_milvus_db is not None:
        return cast("_MilvusClient", MilvusClient(uri=config.rag_milvus_uri, token=config.rag_milvus_token, db_name=config.rag_milvus_db))
    if config.rag_milvus_token is not None:
        return cast("_MilvusClient", MilvusClient(uri=config.rag_milvus_uri, token=config.rag_milvus_token))
    if config.rag_milvus_db is not None:
        return cast("_MilvusClient", MilvusClient(uri=config.rag_milvus_uri, db_name=config.rag_milvus_db))
    return cast("_MilvusClient", MilvusClient(uri=config.rag_milvus_uri))


if __name__ == "__main__":
    main()
