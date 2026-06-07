"""Smoke test PostgreSQL-backed long-term memory and Agent memory tools."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage

from kyuriagents.memory import (
    ElasticsearchMilvusMemoryIndexer,
    MemoryHybridSearcher,
    MemoryRecord,
    MemoryScope,
    MemoryService,
    MemoryWriteCandidate,
    PostgresMemoryStore,
)
from kyuriagents.rag import ElasticsearchKeywordStore, HybridRAGRetriever, MilvusVectorStore
from kyuriagents.runtime import AgentRuntimeConfig, create_kyuri_agent
from kyuriagents.runtime.dashscope import create_dashscope_embed_query

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from kyuriagents.middleware.retrieval import RetrievalMode
    from kyuriagents.runtime.dashscope import EmbedQuery

_LOGGER = logging.getLogger("memory_smoke")
_THREAD_ID = "memory-smoke"
_MEMORY_ID = "mem_smoke_dashscope_rag"
_MEMORY_CONTENT = "The Kyuriagents project uses DashScope embeddings with Milvus vector search and Elasticsearch keyword search for hybrid RAG."
_PREVIEW_LIMIT = 500


def main() -> None:
    """Run memory smoke checks."""
    _load_runtime_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Verify PostgreSQL memory and Agent memory tools.")
    parser.add_argument("--thread-id", default=_THREAD_ID, help="Thread id used for tool audit lookup.")
    parser.add_argument("--hybrid", action="store_true", help="Also verify memory_chunks in Elasticsearch and Milvus.")
    args = parser.parse_args()

    config = _runtime_config(thread_id=args.thread_id, hybrid=bool(args.hybrid))
    store = PostgresMemoryStore(dsn=_dsn(config))
    service = MemoryService(store)
    scope = MemoryScope(tenant_id=config.tenant_id, user_id=config.user_id)
    embed_query = create_dashscope_embed_query(config) if args.hybrid else None

    saved = service.save_candidate(
        MemoryWriteCandidate(
            content=_MEMORY_CONTENT,
            memory_type="fact",
            scope_type="user",
            scope_id=config.user_id or config.tenant_id,
            summary="Kyuriagents uses DashScope, Milvus, and Elasticsearch for hybrid RAG.",
            importance=0.9,
            confidence=0.95,
            tags=("rag", "dashscope", "milvus", "elasticsearch"),
            source_thread_id=args.thread_id,
        ),
        tenant_id=config.tenant_id,
        user_id=config.user_id,
        memory_id=_MEMORY_ID,
    )
    if args.hybrid and embed_query is not None:
        indexer = _memory_indexer(config, embed_query)
        indexer.upsert([saved])
        hybrid_service = MemoryService(
            store,
            hybrid_searcher=MemoryHybridSearcher(
                retriever=_memory_retriever(config, embed_query),
                store=store,
            ),
        )
        hybrid_results = hybrid_service.search("Kyuriagents Milvus Elasticsearch DashScope hybrid RAG services", scope=scope, limit=3)
        _LOGGER.info("hybrid_indexed=%s", saved.memory_id)
        _LOGGER.info("hybrid_search_count=%s", len(hybrid_results))
        for index, result in enumerate(hybrid_results, start=1):
            _LOGGER.info("hybrid_result[%s]=%s score=%.4f", index, result.memory.memory_id, result.score)

    results = service.search("Kyuriagents hybrid RAG uses what services?", scope=scope, limit=3)
    _LOGGER.info("storage_saved=%s", saved.memory_id)
    _LOGGER.info("storage_search_count=%s", len(results))
    for index, result in enumerate(results, start=1):
        _LOGGER.info("storage_result[%s]=%s score=%.4f", index, result.memory.memory_id, result.score)

    before = _audit_count(config, args.thread_id)
    agent = cast(
        "Any",
        create_kyuri_agent(
            config,
            system_prompt=(
                "You are validating long-term memory. You must call `search_memory` before answering questions about prior project facts."
            ),
        ),
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="Please call search_memory and answer: Which services does Kyuriagents use for hybrid RAG?")]},
        config={
            "recursion_limit": 20,
            "configurable": {
                "tenant_id": config.tenant_id,
                "user_id": config.user_id,
                "thread_id": args.thread_id,
                "tool_thread_id": args.thread_id,
                "memory_scope_types": ["user"],
                "memory_scope_ids": [config.user_id or config.tenant_id],
            },
        },
    )
    messages = cast("list[BaseMessage]", result["messages"])
    tool_calls = _tool_calls(messages)
    tool_messages = [message for message in messages if message.type == "tool"]
    after = _audit_count(config, args.thread_id)

    _LOGGER.info("tool_call_count=%s", len(tool_calls))
    for index, tool_call in enumerate(tool_calls, start=1):
        _LOGGER.info("tool_call[%s]=%s args=%s", index, tool_call.get("name"), tool_call.get("args"))
    _LOGGER.info("tool_message_count=%s", len(tool_messages))
    for index, message in enumerate(tool_messages, start=1):
        _LOGGER.info("tool_message[%s]=%s", index, _preview(str(message.text)))
    _LOGGER.info("audit_delta=%s", None if before is None or after is None else after - before)
    checkpoints = _checkpoint_memories(service, scope=scope, thread_id=args.thread_id)
    _LOGGER.info("checkpoint_count=%s", len(checkpoints))
    for index, memory in enumerate(checkpoints, start=1):
        _LOGGER.info("checkpoint[%s]=%s %s", index, memory.memory_id, _preview(memory.content))
    _LOGGER.info("final=%s", _preview(str(messages[-1].text)))


def _runtime_config(*, thread_id: str, hybrid: bool) -> AgentRuntimeConfig:
    config = AgentRuntimeConfig.from_env()
    return replace(
        config,
        enable_rag=hybrid,
        enable_memory=True,
        enable_checkpointer=False,
        rag_mode=cast("RetrievalMode", "off"),
        memory_mode=cast("RetrievalMode", "tool"),
        thread_id=thread_id,
    )


def _memory_indexer(config: AgentRuntimeConfig, embed_query: EmbedQuery) -> ElasticsearchMilvusMemoryIndexer:
    return ElasticsearchMilvusMemoryIndexer(
        es_index=config.memory_es_index,
        es_url=config.rag_es_url,
        milvus_collection=config.memory_milvus_collection,
        milvus_uri=config.rag_milvus_uri,
        milvus_token=config.rag_milvus_token,
        milvus_db=config.rag_milvus_db,
        embed_text=embed_query,
        refresh=True,
    )


def _memory_retriever(config: AgentRuntimeConfig, embed_query: EmbedQuery) -> HybridRAGRetriever:
    return HybridRAGRetriever(
        vector_searcher=MilvusVectorStore(
            collection_name=config.memory_milvus_collection,
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token,
            db_name=config.rag_milvus_db,
            embed_query=embed_query,
        ),
        keyword_searcher=ElasticsearchKeywordStore(
            index=config.memory_es_index,
            url=config.rag_es_url,
        ),
    )


def _dsn(config: AgentRuntimeConfig) -> str:
    if config.postgres_dsn is None:
        msg = "Set DEEPAGENTS_POSTGRES_DSN before running memory smoke tests."
        raise ValueError(msg)
    return config.postgres_dsn


def _tool_calls(messages: list[BaseMessage]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for message in messages:
        raw = getattr(message, "tool_calls", None)
        if raw:
            calls.extend(cast("list[dict[str, object]]", raw))
    return calls


def _audit_count(config: AgentRuntimeConfig, thread_id: str) -> int | None:
    if not config.postgres_dsn:
        return None
    try:
        import psycopg  # noqa: PLC0415
    except ImportError:
        return None
    with psycopg.connect(config.postgres_dsn) as connection:
        row = connection.execute(
            "SELECT count(*) FROM agent_tool_calls WHERE thread_id = %s AND tool_name IN ('search_memory', 'save_memory', 'delete_memory')",
            (thread_id,),
        ).fetchone()
    if row is None:
        return None
    return int(row[0])


def _checkpoint_memories(service: MemoryService, *, scope: MemoryScope, thread_id: str) -> list[MemoryRecord]:
    return [
        memory
        for memory in service.list_memories(scope=scope, limit=100)
        if memory.source_thread_id == thread_id and "conversation-checkpoint" in memory.tags
    ]


def _preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= _PREVIEW_LIMIT:
        return normalized
    return normalized[: _PREVIEW_LIMIT - 14] + "...[truncated]"


def _load_runtime_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / "kyuriagents" / "runtime" / "runtime.env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


if __name__ == "__main__":
    main()
