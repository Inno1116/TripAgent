"""Factory for assembling a runnable Kyuri agent from runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain.agents import create_agent

from kyuriagents.memory import ElasticsearchMilvusMemoryIndexer, MemoryHybridSearcher, MemoryService, PostgresMemoryStore
from kyuriagents.middleware.retrieval import RetrievalMiddleware
from kyuriagents.rag import DashScopeTextReranker, ElasticsearchKeywordStore, HybridRAGRetriever, MilvusVectorStore, PostgresChunkTextHydrator
from kyuriagents.runtime.dashscope import EmbedQuery, create_dashscope_embed_query, create_dashscope_model
from kyuriagents.runtime.mcp import LoadedMCPTools, load_mcp_tools
from kyuriagents.tools import (
    PostgresToolAuditSink,
    ToolAuditSink,
    ToolDescriptor,
    ToolGovernanceMiddleware,
    ToolPolicy,
    ToolRegistry,
    default_tool_registry,
    merge_tool_sequences,
)
from kyuriagents.websearch import create_web_search_tools, web_search_tool_descriptors

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from kyuriagents.runtime.config import AgentRuntimeConfig


def create_kyuri_agent(
    config: AgentRuntimeConfig,
    *,
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    rag_retriever: HybridRAGRetriever | None = None,
    memory_service: MemoryService | None = None,
    embed_query: EmbedQuery | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_policy: ToolPolicy | None = None,
    tool_audit_sink: ToolAuditSink | None = None,
    mcp_tools: Sequence[BaseTool | Callable | dict[str, Any]] | LoadedMCPTools | None = None,
    mcp_descriptors: Sequence[ToolDescriptor] = (),
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    system_prompt: str | None = None,
    debug: bool = False,
    name: str | None = None,
) -> object:
    """Create a Kyuri agent with runtime RAG and memory wiring.

    Args:
        config: Runtime configuration.
        model: Optional prebuilt model. When omitted, DashScope is used.
        tools: Additional user tools.
        middleware: Additional middleware before retrieval wiring.
        rag_retriever: Optional prebuilt RAG retriever.
        memory_service: Optional prebuilt memory service.
        embed_query: Optional query embedding function.
        tool_registry: Optional registry for tool descriptors.
        tool_policy: Optional policy for tool calls.
        tool_audit_sink: Optional audit sink.
        mcp_tools: Optional preloaded MCP tools. When omitted and MCP is
            enabled, tools are loaded from `config.mcp_config_path`.
        mcp_descriptors: Optional descriptors for preloaded MCP tools.
        checkpointer: Optional LangGraph checkpointer.
        store: Optional LangGraph store.
        system_prompt: Optional user system prompt.
        debug: Whether to enable LangGraph debug mode.
        name: Optional agent name.

    Returns:
        Compiled LangChain/LangGraph agent.
    """
    resolved_model = model if model is not None else create_dashscope_model(config)
    resolved_embed_query = embed_query
    if config.enable_rag and rag_retriever is None:
        resolved_embed_query = resolved_embed_query or create_dashscope_embed_query(config)
        rag_retriever = _create_rag_retriever(config, resolved_embed_query)
    if config.enable_memory and memory_service is None:
        memory_service = _create_memory_service(
            config,
            hybrid_retriever=rag_retriever if config.enable_rag else None,
            embed_text=resolved_embed_query,
        )

    resolved_checkpointer = checkpointer
    resolved_store = store
    if config.enable_checkpointer and (resolved_checkpointer is None or resolved_store is None):
        pg_checkpointer, pg_store = _create_langgraph_postgres(config)
        resolved_checkpointer = resolved_checkpointer or pg_checkpointer
        resolved_store = resolved_store or pg_store

    retrieval = RetrievalMiddleware(
        rag_retriever=rag_retriever if config.enable_rag else None,
        memory_service=memory_service if config.enable_memory else None,
        rag_mode=config.rag_mode,
        memory_mode=config.memory_mode,
        defaults=config.retrieval_defaults(),
        memory_checkpoint_interval=config.memory_checkpoint_interval,
        memory_checkpoint_max_chars=config.memory_checkpoint_max_chars,
    )
    runtime_web_tools = create_web_search_tools(config) if config.enable_web_search else ()
    resolved_tools, governance = _build_tool_runtime(
        config,
        native_tools=tools,
        runtime_tools=runtime_web_tools,
        runtime_descriptors=web_search_tool_descriptors(
            timeout_seconds=max(1, int(max(config.web_search_timeout_seconds, config.web_fetch_timeout_seconds, config.web_render_timeout_seconds)))
        )
        if config.enable_web_search
        else (),
        middleware_tools=retrieval.tools,
        tool_registry=tool_registry,
        tool_policy=tool_policy,
        tool_audit_sink=tool_audit_sink,
        mcp_tools=mcp_tools,
        mcp_descriptors=mcp_descriptors,
    )
    resolved_middleware = [*middleware, retrieval]
    if governance is not None:
        resolved_middleware.append(governance)

    return create_agent(
        model=resolved_model,
        tools=resolved_tools,
        system_prompt=system_prompt,
        middleware=resolved_middleware,
        checkpointer=resolved_checkpointer,
        store=resolved_store,
        debug=debug,
        name=name,
    ).with_config(
        {
            "recursion_limit": 9_999,
            "metadata": {
                "ls_integration": "kyuriagents",
                "lc_agent_name": name,
            },
        }
    )


def _create_rag_retriever(config: AgentRuntimeConfig, embed_query: EmbedQuery) -> HybridRAGRetriever:
    return HybridRAGRetriever(
        vector_searcher=MilvusVectorStore(
            collection_name=config.rag_milvus_collection,
            uri=config.rag_milvus_uri,
            token=config.rag_milvus_token,
            db_name=config.rag_milvus_db,
            embed_query=embed_query,
        ),
        keyword_searcher=ElasticsearchKeywordStore(
            index=config.rag_es_index,
            url=config.rag_es_url,
        ),
        chunk_hydrator=_create_rag_chunk_hydrator(config),
        reranker=_create_rag_reranker(config),
    )


def _create_rag_chunk_hydrator(config: AgentRuntimeConfig) -> PostgresChunkTextHydrator | None:
    if not config.postgres_dsn:
        return None
    return PostgresChunkTextHydrator(dsn=config.postgres_dsn)


def _create_rag_reranker(config: AgentRuntimeConfig) -> DashScopeTextReranker | None:
    if not config.rag_rerank_model:
        return None
    return DashScopeTextReranker(
        api_key=config.dashscope_api_key or "",
        model=config.rag_rerank_model,
        endpoint=config.rag_rerank_url,
        timeout_seconds=config.rag_rerank_timeout_seconds,
    )


def _create_memory_service(
    config: AgentRuntimeConfig,
    *,
    hybrid_retriever: HybridRAGRetriever | None = None,
    embed_text: EmbedQuery | None = None,
) -> MemoryService:
    if not config.postgres_dsn:
        missing = ", ".join(config.missing_for_memory())
        msg = f"Missing settings for memory runtime: {missing}."
        raise ValueError(msg)
    store = PostgresMemoryStore(dsn=config.postgres_dsn)
    hybrid_searcher = None
    indexer = None
    if hybrid_retriever is not None and embed_text is not None:
        memory_retriever = _create_memory_retriever(config, embed_text)
        hybrid_searcher = MemoryHybridSearcher(retriever=memory_retriever, store=store)
        indexer = ElasticsearchMilvusMemoryIndexer(
            es_index=config.memory_es_index,
            es_url=config.rag_es_url,
            milvus_collection=config.memory_milvus_collection,
            milvus_uri=config.rag_milvus_uri,
            milvus_token=config.rag_milvus_token,
            milvus_db=config.rag_milvus_db,
            embed_text=embed_text,
        )
    return MemoryService(store, hybrid_searcher=hybrid_searcher, indexer=indexer)


def _create_memory_retriever(config: AgentRuntimeConfig, embed_query: EmbedQuery) -> HybridRAGRetriever:
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


def _build_tool_runtime(
    config: AgentRuntimeConfig,
    *,
    native_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    runtime_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    runtime_descriptors: Sequence[ToolDescriptor],
    middleware_tools: Sequence[BaseTool],
    tool_registry: ToolRegistry | None,
    tool_policy: ToolPolicy | None,
    tool_audit_sink: ToolAuditSink | None,
    mcp_tools: Sequence[BaseTool | Callable | dict[str, Any]] | LoadedMCPTools | None,
    mcp_descriptors: Sequence[ToolDescriptor],
) -> tuple[list[BaseTool | Callable | dict[str, Any]], ToolGovernanceMiddleware | None]:
    registry = tool_registry.copy() if tool_registry is not None else default_tool_registry()
    resolved_mcp_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None
    resolved_mcp_descriptors: Sequence[ToolDescriptor] = mcp_descriptors
    if config.enable_mcp:
        loaded = load_mcp_tools(config) if mcp_tools is None else mcp_tools
        if isinstance(loaded, LoadedMCPTools):
            resolved_mcp_tools = loaded.tools
            resolved_mcp_descriptors = (*resolved_mcp_descriptors, *loaded.descriptors)
        else:
            resolved_mcp_tools = loaded

    for tool in native_tools or ():
        _register_tool_if_missing(registry, tool)
    for tool in runtime_tools:
        _register_tool_if_missing(registry, tool, source="runtime")
    for tool in middleware_tools:
        _register_tool_if_missing(registry, tool, source="runtime")
    registry.register_many(runtime_descriptors, replace_existing=True)
    registry.register_many(resolved_mcp_descriptors, replace_existing=True)

    governance = None
    if config.enable_tools:
        resolved_audit_sink = tool_audit_sink
        if resolved_audit_sink is None and config.enable_tool_audit and config.postgres_dsn:
            resolved_audit_sink = PostgresToolAuditSink(dsn=config.postgres_dsn)
        governance = ToolGovernanceMiddleware(
            registry=registry,
            policy=tool_policy or config.tool_policy(),
            audit_sink=resolved_audit_sink,
            defaults=config.tool_defaults(),
        )

    return merge_tool_sequences(native_tools, runtime_tools, middleware_tools, resolved_mcp_tools), governance


def _register_tool_if_missing(
    registry: ToolRegistry,
    tool: BaseTool | Callable | dict[str, Any],
    *,
    source: str = "native",
) -> None:
    try:
        registry.register_tool(tool, source=cast("Any", source))
    except ValueError:
        return


def _create_langgraph_postgres(config: AgentRuntimeConfig) -> tuple[Checkpointer, BaseStore]:
    if not config.postgres_dsn:
        missing = ", ".join(config.missing_for_memory())
        msg = f"Missing settings for LangGraph PostgreSQL runtime: {missing}."
        raise ValueError(msg)
    try:
        import psycopg  # noqa: PLC0415
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
        from langgraph.store.postgres import PostgresStore  # noqa: PLC0415
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[memory]` to use PostgreSQL checkpointer/store."
        raise ImportError(msg) from exc

    connect = cast("Any", psycopg.connect)
    checkpointer_connection = connect(config.postgres_dsn, autocommit=True, row_factory=dict_row)
    store_connection = connect(config.postgres_dsn, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(checkpointer_connection)
    runtime_store = PostgresStore(store_connection)
    return checkpointer, runtime_store


__all__ = ["create_kyuri_agent"]
