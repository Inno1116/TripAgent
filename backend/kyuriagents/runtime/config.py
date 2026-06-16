"""Configuration model for assembling a runnable KyuriAgents agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from kyuriagents.middleware.retrieval import RetrievalMode, RuntimeContextDefaults
from kyuriagents.tools import (
    DEFAULT_ALLOWED_RISKS,
    DEFAULT_CONFIRMATION_RISKS,
    ToolContextDefaults,
    ToolPolicy,
    ToolRisk,
    parse_tool_names,
    parse_tool_risks,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


@dataclass(frozen=True, kw_only=True)
class AgentRuntimeConfig:
    """Runtime configuration for wiring model, RAG, traveler profile, and persistence.

    Args:
        tenant_id: Tenant or organization identifier.
        user_id: Optional user identifier.
        thread_id: Optional default thread identifier.
        dashscope_api_key: DashScope API key.
        dashscope_base_url: OpenAI-compatible DashScope base URL.
        chat_model: Chat model name.
        dashscope_enable_thinking: Optional DashScope thinking-mode switch.
            When set, it is sent as `extra_body.enable_thinking`.
        context_summary_model: Optional cheaper chat model for short-term
            summarization. When omitted, the main chat model is used.
        embedding_model: Embedding model name.
        embedding_dimensions: Optional embedding dimension override.
        postgres_dsn: Application PostgreSQL DSN.
        postgres_admin_dsn: Optional admin PostgreSQL DSN for database creation.
        postgres_database: Database name to create during bootstrap.
        enable_rag: Whether to wire RAG when dependencies are available.
        enable_travel_profile: Whether to wire structured traveler profile memory.
        enable_checkpointer: Whether to wire LangGraph PostgreSQL persistence.
        rag_mode: RAG middleware mode.
        rag_es_url: Elasticsearch URL.
        rag_es_index: Elasticsearch index.
        rag_milvus_uri: Milvus URI.
        rag_milvus_collection: Milvus collection name.
        rag_milvus_db: Optional Milvus database.
        rag_milvus_token: Optional Milvus token.
        rag_kb_ids: Optional default knowledge-base filters.
        rag_rerank_model: Optional DashScope rerank model. Use `off` in the
            environment to disable this quality layer.
        rag_rerank_url: DashScope rerank endpoint URL.
        rag_rerank_timeout_seconds: DashScope rerank request timeout.
        travel_profile_context_max_chars: Maximum traveler profile characters
            injected into model prompts.
        enable_context_summarization: Whether to enable short-term thread
            summarization before model calls.
        context_summary_trigger_tokens: Approximate input tokens that trigger
            short-term conversation summarization. Use `0` to disable the
            token trigger.
        context_summary_trigger_messages: Legacy message-count trigger for
            short-term conversation summarization. This is only used when the
            token trigger is disabled.
        context_summary_keep_messages: Number of recent messages preserved
            after short-term conversation summarization.
        redis_url: Redis URL used for pending turns and per-thread locks.
        pending_turn_ttl_seconds: Time-to-live for pending turn buffers.
        thread_lock_ttl_seconds: Time-to-live for one in-flight thread lock.
        context_window_tokens: Runtime context window used for local budgeting.
        reserved_output_tokens: Tokens reserved for model output.
        context_safety_ratio: Conservative multiplier applied to the context window.
        max_user_input_tokens: Maximum tokens accepted in one user message.
        max_rag_context_tokens: Maximum retrieved RAG context tokens.
        max_tool_result_tokens: Maximum one tool result tokens.
        tokenizer_model: Qwen tokenizer model used for local token counting.
        tokenizer_local_files_only: Whether tokenization may only use cached files.
        tokenizer_strict: Whether tokenizer loading failures should raise instead
            of falling back to approximate counting.
        enable_subagents: Whether the main agent should delegate RAG and web
            research through specialized subagents instead of seeing those raw
            tools directly.
        enable_web_search: Whether to expose public web search tools.
        enable_travel_tools: Whether to expose travel-domain tools backed by
            AMap MCP plus local budget estimation.
        amap_api_key: AMap API key used by the official AMap MCP endpoint.
        amap_mcp_url: Optional override for the AMap Streamable HTTP MCP URL.
        searxng_base_url: Base URL for the SearXNG instance.
        web_search_max_results: Maximum SearXNG results returned to one tool call.
        web_agent_max_search_calls: Maximum search calls a web subagent should
            spend on one delegated task.
        web_search_safe_search: SearXNG safe-search level.
        web_search_language: Optional SearXNG language code.
        web_search_fallback_engines: SearXNG engines tried one by one after
            aggregate search fails or returns no results.
        web_search_timeout_seconds: Timeout for SearXNG API calls.
        web_fetch_max_pages: Maximum pages opened by one research tool call.
        web_fetch_concurrency: Maximum parallel page fetches.
        web_fetch_retries: Number of retries for SearXNG and HTTP page fetches.
        web_fetch_timeout_seconds: Timeout for plain HTTP page fetches.
        web_fetch_max_bytes: Maximum bytes read from one page response.
        web_fetch_max_chars: Maximum extracted characters returned from one page.
            This also bounds the text a web subagent can inspect from one page.
        web_render_max_pages: Maximum pages rendered with Playwright fallback.
        web_render_timeout_seconds: Timeout for Playwright page rendering.
        api_admin_key: Optional bootstrap key for API admin endpoints.
        auth_token_ttl_days: Number of days before login tokens expire. Use `0`
            for non-expiring tokens in local development.
        api_cors_origins: Browser origins allowed to call the API.
        enable_tools: Whether to enable tool governance middleware.
        enable_mcp: Whether to load MCP tools during runtime startup.
        tool_allowed_risks: Risk classes allowed to execute.
        tool_confirmation_risks: Risk classes that require confirmation.
        tool_allow_requires_confirmation: Whether confirmation-gated tools may execute.
        tool_allowed_names: Optional tool allow-list.
        tool_denied_names: Explicit tool deny-list.
        enable_tool_audit: Whether tool calls should be audited.
        mcp_config_path: Optional MCP JSON config path.
        mcp_tool_name_prefix: Whether descriptor matching expects server-prefixed MCP tool names.
        upload_dir: Directory used for user-uploaded source documents.
        upload_max_bytes: Maximum request body size accepted by document upload endpoints.
        ingestion_parser_mode: Parser backend used by ingestion workers.
        ingestion_mcp_config_path: Optional MCP config path used only for document parsing.
        ingestion_mcp_tool_name: MCP tool name expected to parse one document.
        enable_ingestion_redis_queue: Whether uploads publish ingestion job ids
            to Redis so workers can wake without relying only on polling.
        ingestion_redis_queue_name: Redis list key used for ingestion job ids.
        ingestion_redis_block_timeout_seconds: Maximum seconds a worker blocks
            waiting for one Redis queue signal.
        ingestion_chunk_chars: Maximum characters per indexed document chunk.
        ingestion_chunk_overlap: Characters repeated between adjacent chunks.
        ingestion_embedding_batch_size: Maximum documents sent in one embedding request.
        ingestion_job_timeout_seconds: Maximum running time before a worker marks
            an ingestion job as failed. Use `0` to disable stale job cleanup.
    """

    tenant_id: str = "default"
    user_id: str | None = None
    thread_id: str | None = None
    dashscope_api_key: str | None = None
    dashscope_base_url: str = _DASHSCOPE_BASE_URL
    chat_model: str = "qwen-plus"
    dashscope_enable_thinking: bool | None = None
    context_summary_model: str | None = None
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int | None = None
    postgres_dsn: str | None = None
    postgres_admin_dsn: str | None = None
    postgres_database: str = "kyuriagents"
    enable_rag: bool = True
    enable_travel_profile: bool = True
    enable_checkpointer: bool = True
    rag_mode: RetrievalMode = "tool"
    rag_es_url: str = "http://localhost:9200"
    rag_es_index: str = "rag_chunks"
    rag_milvus_uri: str = "http://localhost:19530"
    rag_milvus_collection: str = "rag_chunks"
    rag_milvus_db: str | None = None
    rag_milvus_token: str | None = None
    rag_kb_ids: tuple[str, ...] = ()
    rag_rerank_model: str | None = "qwen3-vl-rerank"
    rag_rerank_url: str = _DASHSCOPE_RERANK_URL
    rag_rerank_timeout_seconds: float = 10.0
    travel_profile_context_max_chars: int = 4_000
    enable_context_summarization: bool = True
    context_summary_trigger_tokens: int = 100_000
    context_summary_trigger_messages: int = 0
    context_summary_keep_messages: int = 12
    redis_url: str = "redis://localhost:6379/0"
    pending_turn_ttl_seconds: int = 10 * 60
    thread_lock_ttl_seconds: int = 3 * 60
    context_window_tokens: int = 128_000
    reserved_output_tokens: int = 8_192
    context_safety_ratio: float = 0.85
    max_user_input_tokens: int = 12_800
    max_rag_context_tokens: int = 12_000
    max_tool_result_tokens: int = 6_000
    tokenizer_model: str = "Qwen/Qwen3-8B"
    tokenizer_local_files_only: bool = True
    tokenizer_strict: bool = False
    enable_subagents: bool = False
    enable_web_search: bool = False
    enable_travel_tools: bool = True
    amap_api_key: str | None = None
    amap_mcp_url: str | None = None
    searxng_base_url: str = "http://127.0.0.1:8888"
    web_search_max_results: int = 8
    web_search_query_plan_size: int = 3
    web_search_cache_ttl_seconds: int = 300
    web_search_rerank_candidates: int = 24
    web_agent_max_search_calls: int = 3
    web_search_safe_search: int = 0
    web_search_language: str = ""
    web_search_fallback_engines: tuple[str, ...] = ("duckduckgo", "bing", "baidu", "sogou")
    web_search_timeout_seconds: float = 8.0
    web_fetch_max_pages: int = 5
    web_fetch_concurrency: int = 3
    web_fetch_retries: int = 1
    web_fetch_timeout_seconds: float = 8.0
    web_fetch_max_bytes: int = 1_000_000
    web_fetch_max_chars: int = 3_000
    web_render_max_pages: int = 3
    web_render_timeout_seconds: float = 12.0
    api_admin_key: str | None = None
    auth_token_ttl_days: int = 30
    api_cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    enable_tools: bool = True
    enable_mcp: bool = False
    tool_allowed_risks: frozenset[ToolRisk] = DEFAULT_ALLOWED_RISKS
    tool_confirmation_risks: frozenset[ToolRisk] = DEFAULT_CONFIRMATION_RISKS
    tool_allow_requires_confirmation: bool = False
    tool_allowed_names: frozenset[str] = frozenset()
    tool_denied_names: frozenset[str] = frozenset()
    enable_tool_audit: bool = True
    mcp_config_path: str | None = None
    mcp_tool_name_prefix: bool = False
    upload_dir: str = ".kyuriagents/uploads"
    upload_max_bytes: int = 25 * 1024 * 1024
    ingestion_parser_mode: Literal["auto", "local", "mcp"] = "auto"
    ingestion_mcp_config_path: str | None = None
    ingestion_mcp_tool_name: str = "parse_document"
    enable_ingestion_redis_queue: bool = False
    ingestion_redis_queue_name: str = "kyuri:ingestion:jobs"
    ingestion_redis_block_timeout_seconds: int = 2
    ingestion_chunk_chars: int = 1_200
    ingestion_chunk_overlap: int = 180
    ingestion_embedding_batch_size: int = 10
    ingestion_job_timeout_seconds: int = 15 * 60

    def __post_init__(self) -> None:
        """Validate context window settings."""
        self._validate_summary_config()
        self._validate_runtime_budget_config()
        self._validate_rag_config()
        if self.upload_max_bytes <= 0:
            msg = "`upload_max_bytes` must be positive."
            raise ValueError(msg)
        self._validate_ingestion_config()

    def _validate_summary_config(self) -> None:
        """Validate short-term context summarization settings."""
        if self.context_summary_trigger_tokens < 0:
            msg = "`context_summary_trigger_tokens` must not be negative."
            raise ValueError(msg)
        if self.context_summary_trigger_messages < 0:
            msg = "`context_summary_trigger_messages` must not be negative."
            raise ValueError(msg)
        if self.context_summary_keep_messages <= 0:
            msg = "`context_summary_keep_messages` must be positive."
            raise ValueError(msg)
        if (
            self.context_summary_trigger_tokens == 0
            and self.context_summary_trigger_messages > 0
            and self.context_summary_trigger_messages <= self.context_summary_keep_messages
        ):
            msg = "`context_summary_trigger_messages` must be greater than `context_summary_keep_messages`."
            raise ValueError(msg)

    def _validate_runtime_budget_config(self) -> None:
        """Validate Redis and token budget runtime settings."""
        if self.pending_turn_ttl_seconds <= 0:
            msg = "`pending_turn_ttl_seconds` must be positive."
            raise ValueError(msg)
        if self.thread_lock_ttl_seconds <= 0:
            msg = "`thread_lock_ttl_seconds` must be positive."
            raise ValueError(msg)
        if self.context_window_tokens <= 0:
            msg = "`context_window_tokens` must be positive."
            raise ValueError(msg)
        if self.reserved_output_tokens < 0:
            msg = "`reserved_output_tokens` must not be negative."
            raise ValueError(msg)
        if not 0 < self.context_safety_ratio <= 1:
            msg = "`context_safety_ratio` must be between 0 and 1."
            raise ValueError(msg)
        if self.max_user_input_tokens <= 0:
            msg = "`max_user_input_tokens` must be positive."
            raise ValueError(msg)
        if self.max_rag_context_tokens <= 0:
            msg = "`max_rag_context_tokens` must be positive."
            raise ValueError(msg)
        if self.travel_profile_context_max_chars <= 0:
            msg = "`travel_profile_context_max_chars` must be positive."
            raise ValueError(msg)
        if self.max_tool_result_tokens <= 0:
            msg = "`max_tool_result_tokens` must be positive."
            raise ValueError(msg)
        self._validate_web_search_config()

    def _validate_rag_config(self) -> None:
        """Validate RAG-specific runtime settings."""
        if self.rag_rerank_model and not self.rag_rerank_url:
            msg = "`rag_rerank_url` must not be empty when rerank is enabled."
            raise ValueError(msg)
        if self.rag_rerank_timeout_seconds <= 0:
            msg = "`rag_rerank_timeout_seconds` must be positive."
            raise ValueError(msg)

    def _validate_web_search_config(self) -> None:
        """Validate web search runtime settings."""
        if not self.searxng_base_url:
            msg = "`searxng_base_url` must not be empty."
            raise ValueError(msg)
        self._validate_web_search_pipeline_config()
        self._validate_web_fetch_config()

    def _validate_web_search_pipeline_config(self) -> None:
        """Validate web search query planning and ranking settings."""
        if self.web_search_max_results <= 0:
            msg = "`web_search_max_results` must be positive."
            raise ValueError(msg)
        if self.web_search_query_plan_size <= 0:
            msg = "`web_search_query_plan_size` must be positive."
            raise ValueError(msg)
        if self.web_search_cache_ttl_seconds < 0:
            msg = "`web_search_cache_ttl_seconds` must not be negative."
            raise ValueError(msg)
        if self.web_search_rerank_candidates <= 0:
            msg = "`web_search_rerank_candidates` must be positive."
            raise ValueError(msg)
        if self.web_agent_max_search_calls <= 0:
            msg = "`web_agent_max_search_calls` must be positive."
            raise ValueError(msg)
        if self.web_search_safe_search < 0:
            msg = "`web_search_safe_search` must not be negative."
            raise ValueError(msg)

    def _validate_web_fetch_config(self) -> None:
        """Validate web page fetch and render settings."""
        if self.web_search_timeout_seconds <= 0 or self.web_fetch_timeout_seconds <= 0 or self.web_render_timeout_seconds <= 0:
            msg = "Web search and fetch timeouts must be positive."
            raise ValueError(msg)
        if self.web_fetch_retries < 0:
            msg = "`web_fetch_retries` must not be negative."
            raise ValueError(msg)
        if self.web_fetch_max_pages <= 0 or self.web_fetch_concurrency <= 0:
            msg = "Web page fetch limits must be positive."
            raise ValueError(msg)
        if self.web_fetch_max_bytes <= 0 or self.web_fetch_max_chars <= 0:
            msg = "Web fetch size limits must be positive."
            raise ValueError(msg)
        if self.web_render_max_pages < 0:
            msg = "`web_render_max_pages` must not be negative."
            raise ValueError(msg)

    def _validate_ingestion_config(self) -> None:
        """Validate ingestion-specific runtime settings."""
        if self.ingestion_parser_mode not in {"auto", "local", "mcp"}:
            msg = "`ingestion_parser_mode` must be one of: auto, local, mcp."
            raise ValueError(msg)
        if self.ingestion_chunk_chars <= 0:
            msg = "`ingestion_chunk_chars` must be positive."
            raise ValueError(msg)
        if self.ingestion_chunk_overlap < 0:
            msg = "`ingestion_chunk_overlap` must not be negative."
            raise ValueError(msg)
        if self.ingestion_chunk_overlap >= self.ingestion_chunk_chars:
            msg = "`ingestion_chunk_overlap` must be smaller than `ingestion_chunk_chars`."
            raise ValueError(msg)
        if self.ingestion_embedding_batch_size <= 0:
            msg = "`ingestion_embedding_batch_size` must be positive."
            raise ValueError(msg)
        if self.ingestion_job_timeout_seconds < 0:
            msg = "`ingestion_job_timeout_seconds` must not be negative."
            raise ValueError(msg)
        if self.ingestion_redis_block_timeout_seconds < 0:
            msg = "`ingestion_redis_block_timeout_seconds` must not be negative."
            raise ValueError(msg)

    @classmethod
    def from_env(
        cls,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AgentRuntimeConfig:
        """Create runtime config from environment variables.

        Args:
            tenant_id: Optional explicit tenant override.
            user_id: Optional explicit user override.
            thread_id: Optional explicit thread override.
            env: Environment mapping for tests or custom loaders.

        Returns:
            Parsed runtime configuration.
        """
        source = _runtime_env_source(env if env is not None else os.environ)
        return cls(
            tenant_id=tenant_id or _env(source, "DEEPAGENTS_TENANT_ID", "TENANT_ID", default="default"),
            user_id=user_id or _optional_env(source, "DEEPAGENTS_USER_ID", "USER_ID"),
            thread_id=thread_id or _optional_env(source, "DEEPAGENTS_THREAD_ID", "THREAD_ID"),
            dashscope_api_key=_optional_env(source, "DASHSCOPE_API_KEY"),
            dashscope_base_url=_env(source, "DASHSCOPE_BASE_URL", default=_DASHSCOPE_BASE_URL),
            chat_model=_env(source, "DASHSCOPE_CHAT_MODEL", "DEEPAGENTS_CHAT_MODEL", default="qwen-plus"),
            dashscope_enable_thinking=_optional_bool_env(source, "DASHSCOPE_ENABLE_THINKING", "DEEPAGENTS_DASHSCOPE_ENABLE_THINKING"),
            context_summary_model=_optional_env(source, "DEEPAGENTS_CONTEXT_SUMMARY_MODEL", "DASHSCOPE_CONTEXT_SUMMARY_MODEL"),
            embedding_model=_env(source, "DASHSCOPE_EMBEDDING_MODEL", "DEEPAGENTS_EMBEDDING_MODEL", default="text-embedding-v3"),
            embedding_dimensions=_optional_int_env(source, "DASHSCOPE_EMBEDDING_DIMENSIONS", "DEEPAGENTS_EMBEDDING_DIMENSIONS"),
            postgres_dsn=_optional_env(source, "DEEPAGENTS_POSTGRES_DSN", "RAG_POSTGRES_DSN"),
            postgres_admin_dsn=_optional_env(source, "DEEPAGENTS_POSTGRES_ADMIN_DSN", "POSTGRES_ADMIN_DSN"),
            postgres_database=_env(source, "DEEPAGENTS_POSTGRES_DATABASE", "POSTGRES_DATABASE", default="kyuriagents"),
            enable_rag=_bool_env(source, "DEEPAGENTS_ENABLE_RAG", default=True),
            enable_travel_profile=_bool_env(source, "KYURI_ENABLE_TRAVEL_PROFILE", "DEEPAGENTS_ENABLE_TRAVEL_PROFILE", default=True),
            enable_checkpointer=_bool_env(source, "DEEPAGENTS_ENABLE_CHECKPOINTER", default=True),
            rag_mode=_retrieval_mode(_env(source, "DEEPAGENTS_RAG_MODE", default="tool")),
            rag_es_url=_env(source, "RAG_ES_URL", default="http://localhost:9200"),
            rag_es_index=_env(source, "RAG_ES_INDEX", default="rag_chunks"),
            rag_milvus_uri=_env(source, "RAG_MILVUS_URI", default="http://localhost:19530"),
            rag_milvus_collection=_env(source, "RAG_MILVUS_COLLECTION", default="rag_chunks"),
            rag_milvus_db=_optional_env(source, "RAG_MILVUS_DB"),
            rag_milvus_token=_optional_env(source, "RAG_MILVUS_TOKEN"),
            rag_kb_ids=_tuple_env(source, "RAG_KB_IDS", "DEEPAGENTS_RAG_KB_IDS"),
            rag_rerank_model=_optional_model_env(
                source,
                "DEEPAGENTS_RAG_RERANK_MODEL",
                "DASHSCOPE_RERANK_MODEL",
                default="qwen3-vl-rerank",
            ),
            rag_rerank_url=_env(source, "DEEPAGENTS_RAG_RERANK_URL", "DASHSCOPE_RERANK_URL", default=_DASHSCOPE_RERANK_URL),
            rag_rerank_timeout_seconds=_float_env(source, "DEEPAGENTS_RAG_RERANK_TIMEOUT_SECONDS", "DASHSCOPE_RERANK_TIMEOUT_SECONDS", default=10.0),
            travel_profile_context_max_chars=_int_env(source, "KYURI_TRAVEL_PROFILE_CONTEXT_MAX_CHARS", "DEEPAGENTS_TRAVEL_PROFILE_CONTEXT_MAX_CHARS", default=4_000),
            enable_context_summarization=_bool_env(source, "DEEPAGENTS_ENABLE_CONTEXT_SUMMARIZATION", default=True),
            context_summary_trigger_tokens=_int_env(source, "DEEPAGENTS_CONTEXT_SUMMARY_TRIGGER_TOKENS", default=100_000),
            context_summary_trigger_messages=_int_env(source, "DEEPAGENTS_CONTEXT_SUMMARY_TRIGGER_MESSAGES", default=0),
            context_summary_keep_messages=_int_env(source, "DEEPAGENTS_CONTEXT_SUMMARY_KEEP_MESSAGES", default=12),
            redis_url=_env(source, "DEEPAGENTS_REDIS_URL", "REDIS_URL", default="redis://localhost:6379/0"),
            pending_turn_ttl_seconds=_int_env(source, "DEEPAGENTS_PENDING_TURN_TTL_SECONDS", default=10 * 60),
            thread_lock_ttl_seconds=_int_env(source, "DEEPAGENTS_THREAD_LOCK_TTL_SECONDS", default=3 * 60),
            context_window_tokens=_int_env(source, "KYURI_CONTEXT_WINDOW_TOKENS", "DEEPAGENTS_CONTEXT_WINDOW_TOKENS", default=128_000),
            reserved_output_tokens=_int_env(source, "KYURI_RESERVED_OUTPUT_TOKENS", "DEEPAGENTS_RESERVED_OUTPUT_TOKENS", default=8_192),
            context_safety_ratio=_float_env(source, "KYURI_CONTEXT_SAFETY_RATIO", "DEEPAGENTS_CONTEXT_SAFETY_RATIO", default=0.85),
            max_user_input_tokens=_int_env(source, "KYURI_MAX_USER_INPUT_TOKENS", "DEEPAGENTS_MAX_USER_INPUT_TOKENS", default=12_800),
            max_rag_context_tokens=_int_env(source, "KYURI_MAX_RAG_CONTEXT_TOKENS", "DEEPAGENTS_MAX_RAG_CONTEXT_TOKENS", default=12_000),
            max_tool_result_tokens=_int_env(source, "KYURI_MAX_TOOL_RESULT_TOKENS", "DEEPAGENTS_MAX_TOOL_RESULT_TOKENS", default=6_000),
            tokenizer_model=_env(source, "QWEN_TOKENIZER_MODEL", "KYURI_TOKENIZER_MODEL", default="Qwen/Qwen3-8B"),
            tokenizer_local_files_only=_bool_env(source, "QWEN_TOKENIZER_LOCAL_FILES_ONLY", default=True),
            tokenizer_strict=_bool_env(source, "QWEN_TOKENIZER_STRICT", default=False),
            enable_subagents=_bool_env(source, "DEEPAGENTS_ENABLE_SUBAGENTS", "KYURI_ENABLE_SUBAGENTS", default=False),
            enable_web_search=_bool_env(source, "DEEPAGENTS_ENABLE_WEB_SEARCH", "KYURI_ENABLE_WEB_SEARCH", default=False),
            enable_travel_tools=_bool_env(source, "KYURI_ENABLE_TRAVEL_TOOLS", "DEEPAGENTS_ENABLE_TRAVEL_TOOLS", default=True),
            amap_api_key=_optional_env(source, "AMAP_API_KEY", "DEEPAGENTS_AMAP_API_KEY", "KYURI_AMAP_API_KEY"),
            amap_mcp_url=_optional_env(source, "AMAP_MCP_URL", "DEEPAGENTS_AMAP_MCP_URL", "KYURI_AMAP_MCP_URL"),
            searxng_base_url=_env(source, "SEARXNG_BASE_URL", "DEEPAGENTS_SEARXNG_BASE_URL", default="http://127.0.0.1:8888"),
            web_search_max_results=_int_env(source, "DEEPAGENTS_WEB_SEARCH_MAX_RESULTS", default=8),
            web_search_query_plan_size=_int_env(source, "DEEPAGENTS_WEB_SEARCH_QUERY_PLAN_SIZE", default=3),
            web_search_cache_ttl_seconds=_int_env(source, "DEEPAGENTS_WEB_SEARCH_CACHE_TTL_SECONDS", default=300),
            web_search_rerank_candidates=_int_env(source, "DEEPAGENTS_WEB_SEARCH_RERANK_CANDIDATES", default=24),
            web_agent_max_search_calls=_int_env(source, "DEEPAGENTS_WEB_AGENT_MAX_SEARCH_CALLS", "KYURI_WEB_AGENT_MAX_SEARCH_CALLS", default=3),
            web_search_safe_search=_int_env(source, "DEEPAGENTS_WEB_SEARCH_SAFE_SEARCH", default=0),
            web_search_language=_env(source, "DEEPAGENTS_WEB_SEARCH_LANGUAGE", default=""),
            web_search_fallback_engines=_tuple_env(source, "DEEPAGENTS_WEB_SEARCH_FALLBACK_ENGINES") or ("duckduckgo", "bing", "baidu", "sogou"),
            web_search_timeout_seconds=_float_env(source, "DEEPAGENTS_WEB_SEARCH_TIMEOUT_SECONDS", default=8.0),
            web_fetch_max_pages=_int_env(source, "DEEPAGENTS_WEB_FETCH_MAX_PAGES", default=5),
            web_fetch_concurrency=_int_env(source, "DEEPAGENTS_WEB_FETCH_CONCURRENCY", default=3),
            web_fetch_retries=_int_env(source, "DEEPAGENTS_WEB_FETCH_RETRIES", default=1),
            web_fetch_timeout_seconds=_float_env(source, "DEEPAGENTS_WEB_FETCH_TIMEOUT_SECONDS", default=8.0),
            web_fetch_max_bytes=_int_env(source, "DEEPAGENTS_WEB_FETCH_MAX_BYTES", default=1_000_000),
            web_fetch_max_chars=_int_env(source, "DEEPAGENTS_WEB_FETCH_MAX_CHARS", default=3_000),
            web_render_max_pages=_int_env(source, "DEEPAGENTS_WEB_RENDER_MAX_PAGES", default=3),
            web_render_timeout_seconds=_float_env(source, "DEEPAGENTS_WEB_RENDER_TIMEOUT_SECONDS", default=12.0),
            api_admin_key=_optional_env(source, "DEEPAGENTS_API_ADMIN_KEY"),
            auth_token_ttl_days=_int_env(source, "DEEPAGENTS_AUTH_TOKEN_TTL_DAYS", default=30),
            api_cors_origins=_tuple_env(source, "DEEPAGENTS_API_CORS_ORIGINS") or ("http://127.0.0.1:5173", "http://localhost:5173"),
            enable_tools=_bool_env(source, "DEEPAGENTS_ENABLE_TOOLS", default=True),
            enable_mcp=_bool_env(source, "DEEPAGENTS_ENABLE_MCP", default=False),
            tool_allowed_risks=parse_tool_risks(source.get("DEEPAGENTS_TOOL_ALLOWED_RISKS"), default=DEFAULT_ALLOWED_RISKS),
            tool_confirmation_risks=parse_tool_risks(source.get("DEEPAGENTS_TOOL_CONFIRMATION_RISKS"), default=DEFAULT_CONFIRMATION_RISKS),
            tool_allow_requires_confirmation=_bool_env(source, "DEEPAGENTS_TOOL_ALLOW_REQUIRES_CONFIRMATION", default=False),
            tool_allowed_names=parse_tool_names(source.get("DEEPAGENTS_TOOL_ALLOWED_NAMES")),
            tool_denied_names=parse_tool_names(source.get("DEEPAGENTS_TOOL_DENIED_NAMES")),
            enable_tool_audit=_bool_env(source, "DEEPAGENTS_ENABLE_TOOL_AUDIT", default=True),
            mcp_config_path=_optional_env(source, "DEEPAGENTS_MCP_CONFIG_PATH"),
            mcp_tool_name_prefix=_bool_env(source, "DEEPAGENTS_MCP_TOOL_NAME_PREFIX", default=False),
            upload_dir=_env(source, "DEEPAGENTS_UPLOAD_DIR", default=".kyuriagents/uploads"),
            upload_max_bytes=_int_env(source, "DEEPAGENTS_UPLOAD_MAX_BYTES", default=25 * 1024 * 1024),
            ingestion_parser_mode=_ingestion_parser_mode(_env(source, "DEEPAGENTS_INGESTION_PARSER", default="auto")),
            ingestion_mcp_config_path=_optional_env(source, "DEEPAGENTS_INGESTION_MCP_CONFIG_PATH"),
            ingestion_mcp_tool_name=_env(source, "DEEPAGENTS_INGESTION_MCP_TOOL_NAME", default="parse_document"),
            enable_ingestion_redis_queue=_bool_env(source, "DEEPAGENTS_ENABLE_INGESTION_REDIS_QUEUE", default=False),
            ingestion_redis_queue_name=_env(source, "DEEPAGENTS_INGESTION_REDIS_QUEUE_NAME", default="kyuri:ingestion:jobs"),
            ingestion_redis_block_timeout_seconds=_int_env(source, "DEEPAGENTS_INGESTION_REDIS_BLOCK_TIMEOUT_SECONDS", default=2),
            ingestion_chunk_chars=_int_env(source, "DEEPAGENTS_INGESTION_CHUNK_CHARS", default=1_200),
            ingestion_chunk_overlap=_int_env(source, "DEEPAGENTS_INGESTION_CHUNK_OVERLAP", default=180),
            ingestion_embedding_batch_size=_int_env(source, "DEEPAGENTS_INGESTION_EMBEDDING_BATCH_SIZE", default=10),
            ingestion_job_timeout_seconds=_int_env(source, "DEEPAGENTS_INGESTION_JOB_TIMEOUT_SECONDS", default=15 * 60),
        )

    def retrieval_defaults(self) -> RuntimeContextDefaults:
        """Return defaults consumed by `RetrievalMiddleware`.

        Returns:
            Runtime context defaults for tenant, user, and knowledge bases.
        """
        return RuntimeContextDefaults(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
            kb_ids=self.rag_kb_ids,
        )

    def missing_for_model(self) -> tuple[str, ...]:
        """Return missing settings required to construct the DashScope model."""
        if self.dashscope_api_key:
            return ()
        return ("DASHSCOPE_API_KEY",)

    def missing_for_rag(self) -> tuple[str, ...]:
        """Return missing settings required to construct RAG components."""
        missing = list(self.missing_for_model())
        if not self.rag_es_url:
            missing.append("RAG_ES_URL")
        if not self.rag_milvus_uri:
            missing.append("RAG_MILVUS_URI")
        return tuple(missing)

    def missing_for_postgres(self) -> tuple[str, ...]:
        """Return missing settings required to construct PostgreSQL-backed runtime components."""
        if self.postgres_dsn:
            return ()
        return ("DEEPAGENTS_POSTGRES_DSN",)

    def missing_for_profile(self) -> tuple[str, ...]:
        """Return missing settings required to construct traveler profile components."""
        return self.missing_for_postgres()

    def tool_policy(self) -> ToolPolicy:
        """Return the configured tool policy."""
        return ToolPolicy(
            allowed_risks=self.tool_allowed_risks,
            confirmation_risks=self.tool_confirmation_risks,
            allow_requires_confirmation=self.tool_allow_requires_confirmation,
            allowed_tools=self.tool_allowed_names,
            denied_tools=self.tool_denied_names,
        )

    def tool_defaults(self) -> ToolContextDefaults:
        """Return defaults consumed by `ToolGovernanceMiddleware`."""
        return ToolContextDefaults(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            thread_id=self.thread_id,
        )

    def context_summary_trigger(self) -> tuple[Literal["tokens"], int] | tuple[Literal["messages"], int] | None:
        """Return the short-term summarization trigger for KyuriAgents."""
        if self.context_summary_trigger_tokens > 0:
            return ("tokens", self.context_summary_trigger_tokens)
        if self.context_summary_trigger_messages == 0:
            return None
        return ("messages", self.context_summary_trigger_messages)

    def context_summary_keep(self) -> tuple[Literal["messages"], int]:
        """Return the recent-message retention setting for summarization."""
        return ("messages", self.context_summary_keep_messages)


def _env(source: Mapping[str, str], *names: str, default: str) -> str:
    for name in names:
        value = source.get(name)
        if value:
            return value
    return default


def _runtime_env_source(source: Mapping[str, str]) -> Mapping[str, str]:
    aliases = dict(source)
    for name, value in source.items():
        if name.startswith("KYURIAGENTS_"):
            aliases.setdefault(f"DEEPAGENTS_{name.removeprefix('KYURIAGENTS_')}", value)
    return aliases


def _optional_env(source: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = source.get(name)
        if value:
            return value
    return None


def _optional_model_env(source: Mapping[str, str], *names: str, default: str | None) -> str | None:
    for name in names:
        if name not in source:
            continue
        value = source[name].strip()
        if value.lower() in {"", "0", "false", "none", "off"}:
            return None
        return value
    return default


def _bool_env(source: Mapping[str, str], *names: str, default: bool) -> bool:
    value = _optional_env(source, *names)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _optional_bool_env(source: Mapping[str, str], *names: str) -> bool | None:
    value = _optional_env(source, *names)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    msg = f"Boolean environment value must be true/false, got `{value}`."
    raise ValueError(msg)


def _optional_int_env(source: Mapping[str, str], *names: str) -> int | None:
    value = _optional_env(source, *names)
    if value is None:
        return None
    return int(value)


def _int_env(source: Mapping[str, str], *names: str, default: int) -> int:
    value = _optional_env(source, *names)
    if value is None:
        return default
    return int(value)


def _float_env(source: Mapping[str, str], *names: str, default: float) -> float:
    value = _optional_env(source, *names)
    if value is None:
        return default
    return float(value)


def _tuple_env(source: Mapping[str, str], *names: str) -> tuple[str, ...]:
    value = _optional_env(source, *names)
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _retrieval_mode(value: str) -> RetrievalMode:
    if value not in {"off", "auto", "tool", "hybrid"}:
        msg = "`RetrievalMode` must be one of: off, auto, tool, hybrid."
        raise ValueError(msg)
    return cast("RetrievalMode", value)


def _ingestion_parser_mode(value: str) -> Literal["auto", "local", "mcp"]:
    if value not in {"auto", "local", "mcp"}:
        msg = "`DEEPAGENTS_INGESTION_PARSER` must be one of: auto, local, mcp."
        raise ValueError(msg)
    return cast("Literal['auto', 'local', 'mcp']", value)
