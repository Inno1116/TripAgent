"""Middleware that connects RAG and structured traveler profiles to the agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime  # noqa: TC002  # StructuredTool detects runtime injection from this annotation.
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

from kyuriagents.middleware._utils import append_to_system_message
from kyuriagents.profile import TravelProfileService, format_travel_profile_context
from kyuriagents.rag import HybridRAGRetriever, RetrievalScope, RetrievedChunk

RetrievalMode = Literal["off", "auto", "tool", "hybrid"]
"""Runtime modes for automatic context injection and explicit tools."""

RagScopeResolver = Callable[[object], RetrievalScope]
"""Callable that derives a RAG scope from LangGraph runtime objects."""


@dataclass(frozen=True, kw_only=True)
class RuntimeContextDefaults:
    """Default tenant, user, thread, and knowledge-base values."""

    tenant_id: str = "default"
    user_id: str | None = None
    thread_id: str | None = None
    kb_ids: tuple[str, ...] = ()


def resolve_retrieval_scope(
    runtime: object,
    *,
    defaults: RuntimeContextDefaults | None = None,
) -> RetrievalScope:
    """Resolve a RAG retrieval scope from runtime context and config."""
    resolved_defaults = defaults or RuntimeContextDefaults()
    values = _runtime_values(runtime)
    tenant_id = _string_value(values.get("rag_tenant_id") or values.get("tenant_id")) or resolved_defaults.tenant_id
    user_id = _optional_string(values.get("rag_user_id") or values.get("user_id")) or resolved_defaults.user_id
    kb_ids = _string_tuple(values.get("rag_kb_ids") or values.get("kb_ids")) or resolved_defaults.kb_ids
    doc_ids = _string_tuple(values.get("rag_doc_ids") or values.get("doc_ids"))
    tags = _string_tuple(values.get("rag_tags") or values.get("tags"))
    languages = _string_tuple(values.get("rag_languages") or values.get("languages"))
    source_types = _string_tuple(values.get("rag_source_types") or values.get("source_types"))

    return RetrievalScope(
        tenant_id=tenant_id,
        user_id=user_id,
        kb_ids=kb_ids,
        doc_ids=doc_ids,
        tags=tags,
        languages=languages,
        source_types=source_types,
    )


def format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved knowledge-base chunks for prompt injection."""
    if not chunks:
        return "<retrieved_knowledge_base>\n(No relevant knowledge-base chunks found.)\n</retrieved_knowledge_base>"

    lines = ["<retrieved_knowledge_base>"]
    for chunk in chunks:
        metadata = chunk.metadata
        score = _chunk_score(chunk)
        title = metadata.title or metadata.doc_id
        source = metadata.source_uri or metadata.doc_id
        lines.append(f"- [{chunk.chunk_id}; score={score:.4f}; title={title}; source={source}] {chunk.text}")
    lines.append("</retrieved_knowledge_base>")
    return "\n".join(lines)


class RetrievalMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Connect hybrid RAG and structured traveler profile memory to a Kyuri agent."""

    def __init__(
        self,
        *,
        rag_retriever: HybridRAGRetriever | None = None,
        profile_service: TravelProfileService | None = None,
        rag_mode: RetrievalMode = "tool",
        profile_auto: bool = True,
        defaults: RuntimeContextDefaults | None = None,
        rag_scope: RetrievalScope | None = None,
        rag_scope_resolver: RagScopeResolver | None = None,
        rag_auto_top_k: int = 4,
        rag_tool_top_k: int = 8,
        min_rag_score: float = 0.0,
        profile_context_max_chars: int = 4_000,
    ) -> None:
        """Initialize the retrieval middleware."""
        self._rag_retriever = rag_retriever
        self._profile_service = profile_service
        self._rag_mode = rag_mode
        self._profile_auto = profile_auto
        self._defaults = defaults or RuntimeContextDefaults()
        self._rag_scope = rag_scope
        self._rag_scope_resolver = rag_scope_resolver
        self._rag_auto_top_k = _positive("rag_auto_top_k", rag_auto_top_k)
        self._rag_tool_top_k = _positive("rag_tool_top_k", rag_tool_top_k)
        self._min_rag_score = min_rag_score
        self._profile_context_max_chars = _positive("profile_context_max_chars", profile_context_max_chars)
        self.tools = self._build_tools()

    def _build_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        if self._rag_retriever is not None and _has_tool(self._rag_mode):
            tools.append(self._build_search_knowledge_base_tool())
        if self._profile_service is not None:
            tools.extend([self._build_get_travel_profile_tool(), self._build_update_travel_profile_tool()])
        return tools

    def _resolve_rag_scope(self, runtime: object) -> RetrievalScope:
        if self._rag_scope is not None:
            return self._rag_scope
        if self._rag_scope_resolver is not None:
            return self._rag_scope_resolver(runtime)
        return resolve_retrieval_scope(runtime, defaults=self._defaults)

    def _profile_identity(self, runtime: object) -> tuple[str, str] | None:
        values = _runtime_values(runtime)
        tenant_id = _string_value(values.get("profile_tenant_id") or values.get("tenant_id")) or self._defaults.tenant_id
        user_id = _optional_string(values.get("profile_user_id") or values.get("user_id")) or self._defaults.user_id
        if not tenant_id or not user_id:
            return None
        return tenant_id, user_id

    def _retrieve_rag_context(self, query: str, runtime: object, *, limit: int) -> str:
        if self._rag_retriever is None:
            return ""
        chunks = self._rag_retriever.retrieve(query, scope=self._resolve_rag_scope(runtime), top_k=limit)
        filtered = [chunk for chunk in chunks if _chunk_score(chunk) >= self._min_rag_score]
        if not filtered:
            return ""
        return format_rag_context(filtered)

    def _profile_context(self, runtime: object) -> str:
        if self._profile_service is None:
            return ""
        identity = self._profile_identity(runtime)
        if identity is None:
            return ""
        tenant_id, user_id = identity
        record = self._profile_service.get_profile(tenant_id=tenant_id, user_id=user_id)
        return format_travel_profile_context(record, max_chars=self._profile_context_max_chars)

    def _build_auto_context(self, request: ModelRequest[ContextT]) -> str:
        sections: list[str] = []
        if self._profile_auto:
            profile_context = self._profile_context(request.runtime)
            if profile_context:
                sections.append(profile_context)

        query = _latest_human_text(request.messages)
        if query and self._rag_retriever is not None and _has_auto(self._rag_mode):
            rag_context = self._retrieve_rag_context(query, request.runtime, limit=self._rag_auto_top_k)
            if rag_context:
                sections.append(rag_context)
        return "\n\n".join(sections)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject relevant context into a model request."""
        context = self._build_auto_context(request)
        if not context:
            return request
        system_message = append_to_system_message(request.system_message, context)
        return request.override(system_message=system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject retrieval/profile context before a synchronous model call."""
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Inject retrieval/profile context before an asynchronous model call."""
        return await handler(self.modify_request(request))

    def _build_search_knowledge_base_tool(self) -> BaseTool:
        middleware = self

        def search_knowledge_base(query: str, runtime: ToolRuntime, top_k: int | None = None) -> str:
            """Search the configured knowledge base for relevant document chunks."""
            limit = _positive("top_k", top_k or middleware._rag_tool_top_k)
            context = middleware._retrieve_rag_context(query, runtime, limit=limit)
            return context or "No relevant knowledge-base chunks found."

        return StructuredTool.from_function(
            name="search_knowledge_base",
            func=search_knowledge_base,
            description=(
                "Search the configured RAG knowledge base. Use this when the answer may depend on indexed documents, uploaded files, "
                "or domain knowledge outside the current conversation."
            ),
        )

    def _build_get_travel_profile_tool(self) -> BaseTool:
        middleware = self

        def get_travel_profile(runtime: ToolRuntime) -> dict[str, Any]:
            """Return the current structured traveler profile."""
            if middleware._profile_service is None:
                return {"error": "Traveler profile is not configured."}
            identity = middleware._profile_identity(runtime)
            if identity is None:
                return {"error": "No user profile identity is available."}
            tenant_id, user_id = identity
            record = middleware._profile_service.get_profile(tenant_id=tenant_id, user_id=user_id)
            return {
                "tenant_id": record.tenant_id,
                "user_id": record.user_id,
                "profile_version": record.profile_version,
                "profile_data": record.profile_data,
            }

        return StructuredTool.from_function(
            name="get_travel_profile",
            func=get_travel_profile,
            description="Read the structured traveler profile used for personalized travel planning.",
        )

    def _build_update_travel_profile_tool(self) -> BaseTool:
        middleware = self

        def update_travel_profile(
            profile_data: dict[str, Any],
            runtime: ToolRuntime,
            expected_version: int | None = None,
        ) -> dict[str, Any]:
            """Replace the structured traveler profile with a complete updated object."""
            if middleware._profile_service is None:
                return {"error": "Traveler profile is not configured."}
            identity = middleware._profile_identity(runtime)
            if identity is None:
                return {"error": "No user profile identity is available."}
            tenant_id, user_id = identity
            record = middleware._profile_service.update_profile(
                tenant_id=tenant_id,
                user_id=user_id,
                profile_data=profile_data,
                expected_version=expected_version,
            )
            return {
                "tenant_id": record.tenant_id,
                "user_id": record.user_id,
                "profile_version": record.profile_version,
                "profile_data": record.profile_data,
            }

        return StructuredTool.from_function(
            name="update_travel_profile",
            func=update_travel_profile,
            description=(
                "Overwrite the user's structured traveler profile. Use only when the user states stable travel preferences, hard constraints, "
                "trip state, or history facts. The input must be the complete profile object with sections: hard_constraints, "
                "dynamic_preferences, trip_state, and history_facts. Preserve still-valid existing fields."
            ),
        )


def _runtime_values(runtime: object) -> dict[str, object]:
    values: dict[str, object] = {}
    state = getattr(runtime, "state", None)
    context = getattr(runtime, "context", None)
    config = getattr(runtime, "config", None)

    if isinstance(state, Mapping):
        _merge_mapping(values, state)
    if isinstance(context, Mapping):
        _merge_mapping(values, context)
    if isinstance(config, Mapping):
        metadata = config.get("metadata")
        configurable = config.get("configurable")
        if isinstance(metadata, Mapping):
            _merge_mapping(values, metadata)
        if isinstance(configurable, Mapping):
            _merge_mapping(values, configurable)
    return values


def _merge_mapping(target: dict[str, object], source: Mapping[object, object]) -> None:
    target.update({key: value for key, value in source.items() if isinstance(key, str)})


def _string_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if item is not None and str(item))
    return (str(value),)


def _positive(name: str, value: int) -> int:
    if value <= 0:
        msg = f"`{name}` must be positive."
        raise ValueError(msg)
    return value


def _has_auto(mode: RetrievalMode) -> bool:
    return mode in {"auto", "hybrid"}


def _has_tool(mode: RetrievalMode) -> bool:
    return mode in {"tool", "hybrid"}


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


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.text)
    return ""


__all__ = [
    "RagScopeResolver",
    "RetrievalMiddleware",
    "RetrievalMode",
    "RuntimeContextDefaults",
    "format_rag_context",
    "resolve_retrieval_scope",
]
