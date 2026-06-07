"""Middleware that connects RAG and long-term memory to the main agent."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

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

from kyuriagents.memory import (
    MemoryScope,
    MemoryService,
    MemoryWriteCandidate,
    format_memory_context,
)
from kyuriagents.memory.types import MemoryScopeType, MemoryType, MemoryVisibility
from kyuriagents.middleware._utils import append_to_system_message
from kyuriagents.rag import HybridRAGRetriever, RetrievalScope, RetrievedChunk

RetrievalMode = Literal["off", "auto", "tool", "hybrid"]
"""Runtime modes for automatic context injection and explicit tools."""

RagScopeResolver = Callable[[object], RetrievalScope]
"""Callable that derives a RAG scope from LangGraph runtime objects."""

MemoryScopeResolver = Callable[[object], MemoryScope]
"""Callable that derives a memory scope from LangGraph runtime objects."""

_MEMORY_TYPES = frozenset(get_args(MemoryType))
_MEMORY_SCOPE_TYPES = frozenset(get_args(MemoryScopeType))
_MEMORY_VISIBILITIES = frozenset(get_args(MemoryVisibility))
_DEFAULT_MEMORY_CHECKPOINT_INTERVAL = 10
_DEFAULT_MEMORY_CHECKPOINT_MAX_CHARS = 3_000
_MAX_MEMORY_CHECKPOINT_SUMMARY_CHARS = 500
_MAX_MEMORY_CHECKPOINT_FACTS = 5
_MAX_MEMORY_CHECKPOINT_GOALS = 7
_MAX_MEMORY_CHECKPOINT_OUTCOMES = 5
_MAX_MEMORY_CHECKPOINT_BULLET_CHARS = 220
_SECRET_ASSIGNMENT_RE = re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\s*[:=]\s*([^\s,;]+)")
_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+")
_KEYLIKE_SECRET_RE = re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9][A-Za-z0-9._\-]{12,}\b")
_USER_NAME_RE = re.compile(r"(?i)\b(?:my name is|i am|i'm|call me)\s+([A-Z][A-Za-z0-9_\-]{1,40})\b")
_PREFERENCE_RE = re.compile(r"(?i)\b(?:i prefer|i like|i want|i wanna|i would like|please|could you|can you)\b")
_CORRECTION_RE = re.compile(r"(?i)\b(?:actually|correction|instead|not that|no thank you|forget that)\b")
_GOAL_RE = re.compile(r"(?i)\b(?:remember|summarize|test|check|tell me|show me|find|search|compare)\b")


@dataclass(frozen=True, kw_only=True)
class _CheckpointTurn:
    """Compact representation of one user turn for memory checkpoints."""

    user_text: str
    assistant_texts: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RuntimeContextDefaults:
    """Default tenant and user values for retrieval scopes.

    Args:
        tenant_id: Fallback tenant identifier when runtime config does not
            provide one.
        user_id: Optional fallback user identifier.
        thread_id: Optional fallback thread identifier.
        kb_ids: Optional fallback knowledge-base identifiers for RAG.
    """

    tenant_id: str = "default"
    user_id: str | None = None
    thread_id: str | None = None
    kb_ids: tuple[str, ...] = ()


def resolve_retrieval_scope(
    runtime: object,
    *,
    defaults: RuntimeContextDefaults | None = None,
) -> RetrievalScope:
    """Resolve a RAG retrieval scope from runtime context and config.

    Looks for `tenant_id`, `user_id`, and `kb_ids` in runtime context,
    metadata, and configurable values. `rag_tenant_id`, `rag_user_id`, and
    `rag_kb_ids` override the generic names when provided.

    Args:
        runtime: LangGraph runtime or tool runtime object.
        defaults: Fallback scope values.

    Returns:
        `RetrievalScope` for the current request.
    """
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


def resolve_memory_scope(
    runtime: object,
    *,
    defaults: RuntimeContextDefaults | None = None,
) -> MemoryScope:
    """Resolve a long-term memory scope from runtime context and config.

    Looks for `tenant_id`, `user_id`, `memory_scope_types`, `memory_scope_ids`,
    `memory_types`, and `memory_tags`.

    Args:
        runtime: LangGraph runtime or tool runtime object.
        defaults: Fallback scope values.

    Returns:
        `MemoryScope` for the current request.
    """
    resolved_defaults = defaults or RuntimeContextDefaults()
    values = _runtime_values(runtime)
    tenant_id = _string_value(values.get("memory_tenant_id") or values.get("tenant_id")) or resolved_defaults.tenant_id
    user_id = _optional_string(values.get("memory_user_id") or values.get("user_id")) or resolved_defaults.user_id
    scope_types = _memory_scope_types(values.get("memory_scope_types"))
    scope_ids = _string_tuple(values.get("memory_scope_ids"))
    memory_types = _memory_types(values.get("memory_types"))
    tags = _string_tuple(values.get("memory_tags") or values.get("tags"))

    return MemoryScope(
        tenant_id=tenant_id,
        user_id=user_id,
        scope_types=scope_types,
        scope_ids=scope_ids,
        memory_types=memory_types,
        tags=tags,
    )


def format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved knowledge-base chunks for prompt injection.

    Args:
        chunks: Ranked retrieved chunks.

    Returns:
        XML-like context block with source metadata and snippets.
    """
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
    """Connect hybrid RAG and long-term memory to a KyuriAgent.

    The middleware can automatically inject small Top-K context blocks before
    model calls, expose explicit `search_knowledge_base` and `search_memory`
    tools, and expose simple memory write/delete tools backed by `MemoryService`.

    Args:
        rag_retriever: Optional hybrid RAG retriever.
        memory_service: Optional long-term memory service.
        rag_mode: Whether RAG runs automatically, as a tool, both, or neither.
        memory_mode: Whether memory runs automatically, as tools, both, or neither.
        defaults: Fallback tenant/user/kb values.
        rag_scope: Optional static RAG scope.
        memory_scope: Optional static memory scope.
        rag_scope_resolver: Optional callable for dynamic RAG scope resolution.
        memory_scope_resolver: Optional callable for dynamic memory scope resolution.
        rag_auto_top_k: Maximum chunks to inject automatically.
        rag_tool_top_k: Default chunks returned by the RAG tool.
        memory_auto_top_k: Maximum memories to inject automatically.
        memory_tool_top_k: Default memories returned by the memory search tool.
        min_rag_score: Minimum RAG score for automatic injection.
        min_memory_score: Minimum memory score for automatic injection.
        memory_checkpoint_interval: Number of user turns between automatic
            long-term memory checkpoints. Set to `0` to disable.
        memory_checkpoint_max_chars: Maximum characters saved in one automatic
            memory checkpoint.
    """

    def __init__(
        self,
        *,
        rag_retriever: HybridRAGRetriever | None = None,
        memory_service: MemoryService | None = None,
        rag_mode: RetrievalMode = "tool",
        memory_mode: RetrievalMode = "hybrid",
        defaults: RuntimeContextDefaults | None = None,
        rag_scope: RetrievalScope | None = None,
        memory_scope: MemoryScope | None = None,
        rag_scope_resolver: RagScopeResolver | None = None,
        memory_scope_resolver: MemoryScopeResolver | None = None,
        rag_auto_top_k: int = 4,
        rag_tool_top_k: int = 8,
        memory_auto_top_k: int = 3,
        memory_tool_top_k: int = 10,
        min_rag_score: float = 0.0,
        min_memory_score: float = 0.0,
        memory_checkpoint_interval: int = _DEFAULT_MEMORY_CHECKPOINT_INTERVAL,
        memory_checkpoint_max_chars: int = _DEFAULT_MEMORY_CHECKPOINT_MAX_CHARS,
    ) -> None:
        """Initialize the retrieval middleware."""
        self._rag_retriever = rag_retriever
        self._memory_service = memory_service
        self._rag_mode = rag_mode
        self._memory_mode = memory_mode
        self._defaults = defaults or RuntimeContextDefaults()
        self._rag_scope = rag_scope
        self._memory_scope = memory_scope
        self._rag_scope_resolver = rag_scope_resolver
        self._memory_scope_resolver = memory_scope_resolver
        self._rag_auto_top_k = _positive("rag_auto_top_k", rag_auto_top_k)
        self._rag_tool_top_k = _positive("rag_tool_top_k", rag_tool_top_k)
        self._memory_auto_top_k = _positive("memory_auto_top_k", memory_auto_top_k)
        self._memory_tool_top_k = _positive("memory_tool_top_k", memory_tool_top_k)
        self._min_rag_score = min_rag_score
        self._min_memory_score = min_memory_score
        self._memory_checkpoint_interval = _non_negative("memory_checkpoint_interval", memory_checkpoint_interval)
        self._memory_checkpoint_max_chars = _positive("memory_checkpoint_max_chars", memory_checkpoint_max_chars)
        self.tools = self._build_tools()

    def _build_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        if self._rag_retriever is not None and _has_tool(self._rag_mode):
            tools.append(self._build_search_knowledge_base_tool())
        if self._memory_service is not None and _has_tool(self._memory_mode):
            tools.extend(
                [
                    self._build_search_memory_tool(),
                    self._build_save_memory_tool(),
                    self._build_delete_memory_tool(),
                ]
            )
        return tools

    def _resolve_rag_scope(self, runtime: object) -> RetrievalScope:
        if self._rag_scope is not None:
            return self._rag_scope
        if self._rag_scope_resolver is not None:
            return self._rag_scope_resolver(runtime)
        return resolve_retrieval_scope(runtime, defaults=self._defaults)

    def _resolve_memory_scope(self, runtime: object) -> MemoryScope:
        if self._memory_scope is not None:
            return self._memory_scope
        if self._memory_scope_resolver is not None:
            return self._memory_scope_resolver(runtime)
        return resolve_memory_scope(runtime, defaults=self._defaults)

    def _retrieve_rag_context(self, query: str, runtime: object, *, limit: int) -> str:
        if self._rag_retriever is None:
            return ""
        chunks = self._rag_retriever.retrieve(query, scope=self._resolve_rag_scope(runtime), top_k=limit)
        filtered = [chunk for chunk in chunks if _chunk_score(chunk) >= self._min_rag_score]
        if not filtered:
            return ""
        return format_rag_context(filtered)

    def _retrieve_memory_context(self, query: str, runtime: object, *, limit: int) -> str:
        if self._memory_service is None:
            return ""
        scope = self._resolve_memory_scope(runtime)
        results = self._memory_service.search(query, scope=scope, limit=limit)
        filtered = [result for result in results if result.score >= self._min_memory_score]
        if not filtered:
            return ""
        return format_memory_context(filtered)

    def _build_auto_context(self, request: ModelRequest[ContextT]) -> str:
        query = _latest_human_text(request.messages)
        if not query:
            return ""

        sections: list[str] = []
        if self._rag_retriever is not None and _has_auto(self._rag_mode):
            rag_context = self._retrieve_rag_context(query, request.runtime, limit=self._rag_auto_top_k)
            if rag_context:
                sections.append(rag_context)
        if self._memory_service is not None and _has_auto(self._memory_mode):
            memory_context = self._retrieve_memory_context(query, request.runtime, limit=self._memory_auto_top_k)
            if memory_context:
                sections.append(memory_context)
        return "\n\n".join(sections)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject relevant RAG and memory context into a model request.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with relevant retrieval context in the system
            message. If no context is retrieved, returns the original request.
        """
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
        """Inject retrieval context before a synchronous model call."""
        return handler(self.modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Inject retrieval context before an asynchronous model call."""
        return await handler(self.modify_request(request))

    def after_agent(self, state: AgentState, runtime: object) -> dict[str, Any] | None:
        """Persist a deterministic long-term memory checkpoint after the agent run."""
        self._maybe_save_memory_checkpoint(state.get("messages", []), runtime)
        return None

    async def aafter_agent(self, state: AgentState, runtime: object) -> dict[str, Any] | None:
        """Async variant of `after_agent`."""
        self._maybe_save_memory_checkpoint(state.get("messages", []), runtime)
        return None

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
                "Search the configured RAG knowledge base. Use this when the "
                "answer may depend on indexed documents, uploaded files, or "
                "domain knowledge outside the current conversation."
            ),
        )

    def _build_search_memory_tool(self) -> BaseTool:
        middleware = self

        def search_memory(query: str, runtime: ToolRuntime, top_k: int | None = None) -> str:
            """Search long-term memory for relevant user or project context."""
            limit = _positive("top_k", top_k or middleware._memory_tool_top_k)
            context = middleware._retrieve_memory_context(query, runtime, limit=limit)
            return context or "No relevant long-term memory found."

        return StructuredTool.from_function(
            name="search_memory",
            func=search_memory,
            description=("Search long-term memory for user preferences, durable facts, project rules, prior decisions, corrections, and workflows."),
        )

    def _build_save_memory_tool(self) -> BaseTool:
        middleware = self

        def save_memory(
            content: str,
            runtime: ToolRuntime,
            memory_type: str = "fact",
            scope_type: str = "user",
            scope_id: str = "",
            summary: str = "",
            importance: float = 0.5,
            confidence: float = 0.8,
            visibility: str = "private",
        ) -> str:
            """Save a durable memory item with tenant and user metadata."""
            if middleware._memory_service is None:
                return "Long-term memory is not configured."
            resolved_scope = middleware._resolve_memory_scope(runtime)
            resolved_memory_type = cast("MemoryType", _literal_value("memory_type", memory_type, _MEMORY_TYPES))
            resolved_scope_type = cast("MemoryScopeType", _literal_value("scope_type", scope_type, _MEMORY_SCOPE_TYPES))
            resolved_visibility = cast("MemoryVisibility", _literal_value("visibility", visibility, _MEMORY_VISIBILITIES))
            resolved_scope_id = scope_id or _default_memory_scope_id(resolved_scope, resolved_scope_type)
            record = middleware._memory_service.save_candidate(
                MemoryWriteCandidate(
                    content=content,
                    memory_type=resolved_memory_type,
                    scope_type=resolved_scope_type,
                    scope_id=resolved_scope_id,
                    summary=summary,
                    importance=importance,
                    confidence=confidence,
                    visibility=resolved_visibility,
                    source_thread_id=_thread_id(runtime) or middleware._defaults.thread_id,
                ),
                tenant_id=resolved_scope.tenant_id,
                user_id=resolved_scope.user_id,
            )
            return f"Saved memory `{record.memory_id}`."

        return StructuredTool.from_function(
            name="save_memory",
            func=save_memory,
            description=(
                "Save durable long-term memory. Use only for stable preferences, "
                "facts, project rules, decisions, corrections, or workflows. "
                "Never save secrets or one-time transient details."
            ),
        )

    def _build_delete_memory_tool(self) -> BaseTool:
        middleware = self

        def delete_memory(memory_id: str, runtime: ToolRuntime) -> str:
            """Delete a visible long-term memory item."""
            if middleware._memory_service is None:
                return "Long-term memory is not configured."
            deleted = middleware._memory_service.delete(memory_id, scope=middleware._resolve_memory_scope(runtime))
            if not deleted:
                return f"Memory `{memory_id}` was not found or is not visible."
            return f"Deleted memory `{memory_id}`."

        return StructuredTool.from_function(
            name="delete_memory",
            func=delete_memory,
            description="Soft-delete a long-term memory item when it is wrong, outdated, or no longer useful.",
        )

    def _maybe_save_memory_checkpoint(self, messages: list[AnyMessage], runtime: object) -> None:
        if self._memory_service is None or self._memory_checkpoint_interval == 0:
            return
        human_count = _human_message_count(messages)
        if human_count == 0 or human_count % self._memory_checkpoint_interval != 0:
            return

        segment = human_count // self._memory_checkpoint_interval
        memory_scope = self._resolve_memory_scope(runtime)
        scope_type = memory_scope.scope_types[0] if memory_scope.scope_types else cast("MemoryScopeType", "user")
        scope_id = _default_memory_scope_id(memory_scope, scope_type)
        thread_id = _thread_id(runtime) or self._defaults.thread_id
        thread_key = thread_id or _conversation_fingerprint(messages)
        memory_id = _memory_checkpoint_id(memory_scope.tenant_id, scope_type, scope_id, thread_key, segment)
        visibility = cast("MemoryVisibility", "private" if memory_scope.user_id is not None else "team")
        lookup_scope = MemoryScope(
            tenant_id=memory_scope.tenant_id,
            user_id=memory_scope.user_id,
            scope_types=(scope_type,),
            scope_ids=(scope_id,),
            active_only=False,
        )
        if self._memory_service.get(memory_id, scope=lookup_scope) is not None:
            return

        recent = _recent_turn_messages(messages, self._memory_checkpoint_interval)
        start_turn = ((segment - 1) * self._memory_checkpoint_interval) + 1
        content = _build_memory_checkpoint_content(
            recent,
            start_turn=start_turn,
            end_turn=human_count,
            max_chars=self._memory_checkpoint_max_chars,
        )
        summary = _truncate_text(f"Auto memory checkpoint for turns {start_turn}-{human_count}.", _MAX_MEMORY_CHECKPOINT_SUMMARY_CHARS)
        self._memory_service.save_candidate(
            MemoryWriteCandidate(
                content=content,
                memory_type="summary",
                scope_type=scope_type,
                scope_id=scope_id,
                summary=summary,
                importance=0.6,
                confidence=0.7,
                visibility=visibility,
                tags=("auto", "conversation-checkpoint"),
                source_thread_id=thread_id,
                source_message_ids=_message_ids(recent),
            ),
            tenant_id=memory_scope.tenant_id,
            user_id=memory_scope.user_id,
            memory_id=memory_id,
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


def _memory_types(value: object) -> tuple[MemoryType, ...]:
    return tuple(cast("MemoryType", _literal_value("memory_type", item, _MEMORY_TYPES)) for item in _string_tuple(value))


def _memory_scope_types(value: object) -> tuple[MemoryScopeType, ...]:
    return tuple(cast("MemoryScopeType", _literal_value("scope_type", item, _MEMORY_SCOPE_TYPES)) for item in _string_tuple(value))


def _literal_value(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        msg = f"`{name}` must be one of: {allowed_values}."
        raise ValueError(msg)
    return value


def _positive(name: str, value: int) -> int:
    if value <= 0:
        msg = f"`{name}` must be positive."
        raise ValueError(msg)
    return value


def _non_negative(name: str, value: int) -> int:
    if value < 0:
        msg = f"`{name}` must not be negative."
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


def _default_memory_scope_id(scope: MemoryScope, scope_type: MemoryScopeType) -> str:
    if scope.scope_ids:
        return scope.scope_ids[0]
    if scope_type == "user" and scope.user_id is not None:
        return scope.user_id
    return scope.tenant_id


def _thread_id(runtime: object) -> str | None:
    values = _runtime_values(runtime)
    return _optional_string(values.get("thread_id"))


def _human_message_count(messages: list[AnyMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, HumanMessage))


def _recent_turn_messages(messages: list[AnyMessage], interval: int) -> list[AnyMessage]:
    selected: list[AnyMessage] = []
    human_count = 0
    for message in reversed(messages):
        selected.append(message)
        if isinstance(message, HumanMessage):
            human_count += 1
            if human_count >= interval:
                break
    selected.reverse()
    return selected


def _build_memory_checkpoint_content(
    messages: list[AnyMessage],
    *,
    start_turn: int,
    end_turn: int,
    max_chars: int,
) -> str:
    turns = _checkpoint_turns(messages)
    sections = [f"Automatic conversation checkpoint for user turns {start_turn}-{end_turn}."]
    facts = _checkpoint_facts(turns)
    goals = _checkpoint_goals(turns)
    outcomes = _checkpoint_outcomes(turns)

    if facts:
        sections.append(_checkpoint_section("Durable user facts and preferences", facts))
    if goals:
        sections.append(_checkpoint_section("Recent user goals", goals))
    if outcomes:
        sections.append(_checkpoint_section("Assistant outcomes", outcomes))
    if len(sections) == 1:
        sections.append("- No durable user facts, preferences, goals, or decisions were identified in this checkpoint.")
    return _truncate_multiline_text("\n".join(sections), max_chars)


def _checkpoint_role(message: AnyMessage) -> str | None:
    if isinstance(message, HumanMessage):
        return "user"
    if message.type == "ai":
        return "assistant"
    return None


def _checkpoint_turns(messages: list[AnyMessage]) -> list[_CheckpointTurn]:
    turns: list[_CheckpointTurn] = []
    current_user = ""
    assistant_texts: list[str] = []
    seen_assistant: set[str] = set()

    def flush() -> None:
        nonlocal current_user, assistant_texts, seen_assistant
        if current_user:
            turns.append(_CheckpointTurn(user_text=current_user, assistant_texts=tuple(assistant_texts)))
        current_user = ""
        assistant_texts = []
        seen_assistant = set()

    for message in messages:
        role = _checkpoint_role(message)
        if role is None:
            continue
        text = _redact_sensitive(_message_text(message))
        if not text:
            continue
        if role == "user":
            flush()
            current_user = text
            continue
        fingerprint = _checkpoint_text_fingerprint(text)
        if fingerprint in seen_assistant:
            continue
        seen_assistant.add(fingerprint)
        assistant_texts.append(text)
    flush()
    return turns


def _checkpoint_facts(turns: list[_CheckpointTurn]) -> list[str]:
    facts: list[str] = []
    for turn in turns:
        user_text = turn.user_text
        name_match = _USER_NAME_RE.search(user_text)
        if name_match:
            facts.append(f"User identified themselves as {name_match.group(1)}.")
        if _PREFERENCE_RE.search(user_text):
            facts.append(f"User preference/request: {_checkpoint_bullet_text(user_text)}")
        if _CORRECTION_RE.search(user_text):
            facts.append(f"User correction: {_checkpoint_bullet_text(user_text)}")
    return _dedupe_bullets(facts, limit=_MAX_MEMORY_CHECKPOINT_FACTS)


def _checkpoint_goals(turns: list[_CheckpointTurn]) -> list[str]:
    goals: list[str] = []
    for turn in turns:
        text = turn.user_text
        if "?" in text or _PREFERENCE_RE.search(text) or _CORRECTION_RE.search(text) or _GOAL_RE.search(text):
            goals.append(f"User asked/needed: {_checkpoint_bullet_text(text)}")
    return _dedupe_bullets(goals, limit=_MAX_MEMORY_CHECKPOINT_GOALS)


def _checkpoint_outcomes(turns: list[_CheckpointTurn]) -> list[str]:
    outcomes: list[str] = []
    for turn in turns:
        if not turn.assistant_texts:
            continue
        outcomes.append(f"Assistant response: {_checkpoint_bullet_text(turn.assistant_texts[-1])}")
    return _dedupe_bullets(outcomes, limit=_MAX_MEMORY_CHECKPOINT_OUTCOMES)


def _checkpoint_section(title: str, bullets: list[str]) -> str:
    lines = [f"{title}:"]
    lines.extend(f"- {bullet}" for bullet in bullets)
    return "\n".join(lines)


def _checkpoint_bullet_text(text: str) -> str:
    return _truncate_text(text, _MAX_MEMORY_CHECKPOINT_BULLET_CHARS)


def _dedupe_bullets(bullets: list[str], *, limit: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for bullet in bullets:
        key = _checkpoint_text_fingerprint(bullet)
        if key in seen:
            continue
        seen.add(key)
        selected.append(bullet)
        if len(selected) >= limit:
            break
    return selected


def _checkpoint_text_fingerprint(text: str) -> str:
    return " ".join(text.casefold().split())


def _message_text(message: AnyMessage) -> str:
    value = getattr(message, "text", "")
    if isinstance(value, str) and value:
        return " ".join(value.split())
    return " ".join(str(message.content).split())


def _redact_sensitive(text: str) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    redacted = _BEARER_SECRET_RE.sub("Bearer [redacted]", redacted)
    return _KEYLIKE_SECRET_RE.sub("[redacted-secret]", redacted)


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    suffix = "...[truncated]"
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return normalized[: max_chars - len(suffix)].rstrip() + suffix


def _truncate_multiline_text(text: str, max_chars: int) -> str:
    normalized = "\n".join(line for line in (" ".join(line.split()) for line in text.splitlines()) if line)
    if len(normalized) <= max_chars:
        return normalized
    suffix = "...[truncated]"
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return normalized[: max_chars - len(suffix)].rstrip() + suffix


def _message_ids(messages: list[AnyMessage]) -> tuple[str, ...]:
    ids = [message.id for message in messages if message.id]
    return tuple(dict.fromkeys(ids))


def _conversation_fingerprint(messages: list[AnyMessage]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(message.type.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_message_text(message).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _memory_checkpoint_id(tenant_id: str, scope_type: str, scope_id: str, thread_key: str, segment: int) -> str:
    raw = f"{tenant_id}|{scope_type}|{scope_id}|{thread_key}|{segment}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"mem_checkpoint_{digest}"


__all__ = [
    "MemoryScopeResolver",
    "RagScopeResolver",
    "RetrievalMiddleware",
    "RetrievalMode",
    "RuntimeContextDefaults",
    "format_rag_context",
    "resolve_memory_scope",
    "resolve_retrieval_scope",
]
