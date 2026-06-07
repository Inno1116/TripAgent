"""FastAPI application factory for deployed KyuriAgents runtimes."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, TypeVar, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from kyuriagents.ingestion import KnowledgeBaseService
from kyuriagents.runtime import AgentRuntimeConfig, create_kyuri_agent
from kyuriagents.runtime.errors import public_error_message
from kyuriagents.runtime.token_budget import TokenBudgetExceeded, enforce_user_input_budget, messages_tokens, token_counter_from_config
from kyuriagents.server.identity import (
    AuthContext,
    DuplicateUserError,
    MessageRecord,
    PostgresUserCenter,
    ThreadRecord,
    ThreadSummaryRecord,
    ThreadSummaryUpdate,
    UserCenter,
)
from kyuriagents.server.pending import PendingTurn, PendingTurnStore, RedisPendingTurnStore, ThreadBusyError
from kyuriagents.server.summary import ThreadSummaryService, ThreadSummaryServiceProtocol
from kyuriagents.tasks import TaskRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from kyuriagents.tasks.types import TaskEventRecord, TaskRecord, TaskStepRecord


_TASK_STREAM_POLL_SECONDS = 0.2
_TASK_STREAM_EVENT_LIMIT = 500


class _Agent(Protocol):
    """Minimal runnable agent contract used by the API layer."""

    def invoke(self, input_data: object, /, config: Mapping[str, object] | None = None) -> object:
        """Invoke the agent with LangGraph-style input."""
        ...

    def stream(
        self,
        input_data: object,
        /,
        *,
        config: Mapping[str, object] | None = None,
        stream_mode: object | None = None,
    ) -> Iterable[object]:
        """Stream the agent with LangGraph-style input."""
        ...


class ThreadNotFoundError(LookupError):
    """Raised when a user cannot access the requested thread."""


@dataclass
class _StreamState:
    """Mutable state for one streaming chat response."""

    buffered_text_parts: list[str] = field(default_factory=list)
    emitted_text: bool = False
    initial_model_pending: bool = True
    tool_finished: bool = False
    tool_started: bool = False


@dataclass
class _TaskWorkerState:
    """Mutable state shared by the task stream worker and SSE loop."""

    done: threading.Event = field(default_factory=threading.Event)
    result: object | None = None
    error: BaseException | None = None


@dataclass(frozen=True, kw_only=True)
class _ChatContext:
    """Messages and optional rolling-summary update for one chat turn."""

    input_data: dict[str, list[object]]
    summary_update: ThreadSummaryUpdate | None = None


_MIN_PASSWORD_LENGTH = 8
_STREAM_ITEM_PAIR_LENGTH = 2
_STREAM_ITEM_TRIPLE_LENGTH = 3
_KnowledgeResult = TypeVar("_KnowledgeResult")
_DEFAULT_API_SYSTEM_PROMPT = """You are Kyuriagents, a careful agent with access to tools, RAG, and long-term memory.

Answer the user's question directly and in the user's language unless they ask otherwise.
When retrieved knowledge-base or memory context is available, ground your answer in that context and prefer concrete dates, names, and facts.
When web search results are available, cite source titles or URLs and distinguish current web evidence from local knowledge.
Do not use web search to locate pirated, leaked, cracked, or unauthorized download resources.
For comparison questions, state the conclusion first, then give the supporting facts.
Do not narrate internal reasoning, do not say what the user "appears to be asking", and do not mention search mechanics unless the user asks.
If the available context is insufficient, say so briefly and name the missing information.
"""
ParserMode = Literal["auto", "local", "mcp"]
Visibility = Literal["private", "team", "public"]


class RegisterRequest(BaseModel):
    """Request body for email/password registration."""

    email: str
    password: str = Field(min_length=_MIN_PASSWORD_LENGTH)
    display_name: str = ""
    tenant_id: str | None = None
    tenant_name: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class LoginRequest(BaseModel):
    """Request body for email/password login."""

    email: str
    password: str
    tenant_id: str | None = None


class TenantCreateRequest(BaseModel):
    """Request body for creating a tenant."""

    name: str
    tenant_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class UserCreateRequest(BaseModel):
    """Request body for creating a user."""

    tenant_id: str
    email: str
    display_name: str = ""
    user_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class APIKeyCreateRequest(BaseModel):
    """Request body for creating an API key."""

    tenant_id: str
    user_id: str
    name: str = ""
    expires_at: str | None = None


class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""

    title: str = ""
    thread_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class KnowledgeBaseCreateRequest(BaseModel):
    """Request body for creating a user knowledge base."""

    name: str
    description: str = ""
    visibility: Visibility = "private"
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request body for sending a user message."""

    message: str
    thread_id: str | None = None
    title: str = ""
    rag_enabled: bool | None = None
    web_search_enabled: bool | None = None


class TaskCreateRequest(BaseModel):
    """Request body for starting task mode."""

    goal: str
    thread_id: str | None = None
    title: str = ""
    intent: Literal["chat", "task", "rag_query", "memory_query", "clarify", "unsafe"] | None = "task"
    rag_enabled: bool | None = None
    web_search_enabled: bool | None = None


class TaskResumeRequest(BaseModel):
    """Request body for resuming a task waiting for human input."""

    message: str
    intent: Literal["chat", "task", "rag_query", "memory_query", "clarify", "unsafe"] | None = "task"
    rag_enabled: bool | None = None
    web_search_enabled: bool | None = None


class RecordResponse(BaseModel):
    """Generic JSON response for record-shaped objects."""

    data: dict[str, object]


class APIKeyResponse(BaseModel):
    """Response body for API key creation."""

    data: dict[str, object]
    api_key: str


class AuthTokenResponse(BaseModel):
    """Response body for email/password auth endpoints."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105  # OAuth token type label, not a credential.
    expires_at: str | None = None
    data: dict[str, object]


class ChatResponse(BaseModel):
    """Response body for chat calls."""

    thread_id: str
    message_id: str
    content: str


def create_app(
    *,
    config: AgentRuntimeConfig | None = None,
    user_center: UserCenter | None = None,
    knowledge_service: KnowledgeBaseService | None = None,
    task_runtime: TaskRuntime | None = None,
    agent_factory: Callable[..., object] | None = None,
    pending_turn_store: PendingTurnStore | None = None,
    thread_summary_service: ThreadSummaryServiceProtocol | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        config: Runtime configuration. Defaults to `AgentRuntimeConfig.from_env()`.
        user_center: Optional user center store. Defaults to PostgreSQL.
        knowledge_service: Optional knowledge-base ingestion service.
        task_runtime: Optional task-mode runtime.
        agent_factory: Optional agent factory for tests.
        pending_turn_store: Optional Redis-backed pending turn store.
        thread_summary_service: Optional persistent thread summary service.

    Returns:
        FastAPI app instance.

    Raises:
        ValueError: If PostgreSQL DSN is missing when no user center is supplied.
    """
    resolved_config = config or AgentRuntimeConfig.from_env()
    resolved_center = user_center or _postgres_user_center(resolved_config)
    resolved_knowledge = knowledge_service or KnowledgeBaseService(config=resolved_config)
    resolved_tasks = task_runtime or TaskRuntime.from_config(resolved_config)
    resolved_agent_factory = agent_factory or create_kyuri_agent
    resolved_pending_turn_store = pending_turn_store or RedisPendingTurnStore.from_config(resolved_config)
    resolved_summary_service = thread_summary_service or ThreadSummaryService(config=resolved_config)
    app = FastAPI(title="KyuriAgents API", version="0.1.0")
    _configure_cors(app, resolved_config)
    _register_auth_routes(app=app, config=resolved_config, user_center=resolved_center)
    _register_admin_routes(app=app, config=resolved_config, user_center=resolved_center)
    _register_user_routes(
        app=app,
        config=resolved_config,
        user_center=resolved_center,
        knowledge_service=resolved_knowledge,
        task_runtime=resolved_tasks,
        agent_factory=resolved_agent_factory,
        pending_turn_store=resolved_pending_turn_store,
        thread_summary_service=resolved_summary_service,
    )
    return app


def _configure_cors(app: FastAPI, config: AgentRuntimeConfig) -> None:
    """Allow configured browser origins to call the API."""
    if not config.api_cors_origins:
        return
    app.add_middleware(
        cast("Any", CORSMiddleware),
        allow_origins=list(config.api_cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_auth_routes(*, app: FastAPI, config: AgentRuntimeConfig, user_center: UserCenter) -> None:
    """Register public email/password authentication routes."""

    @app.post("/v1/auth/register", response_model=AuthTokenResponse)
    def register(request: RegisterRequest) -> AuthTokenResponse:
        tenant_id = request.tenant_id or config.tenant_id
        user_center.ensure_tenant(name=request.tenant_name or tenant_id, tenant_id=tenant_id)
        try:
            user = user_center.create_user_with_password(
                tenant_id=tenant_id,
                email=request.email,
                password=request.password,
                display_name=request.display_name,
                metadata=request.metadata,
            )
        except DuplicateUserError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A user with this email already exists.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        created = user_center.create_api_key(
            tenant_id=tenant_id,
            user_id=user.user_id,
            name="password-login",
            expires_at=_token_expires_at(config),
        )
        return _auth_token_response(user_center, created.raw_key)

    @app.post("/v1/auth/login", response_model=AuthTokenResponse)
    def login(request: LoginRequest) -> AuthTokenResponse:
        tenant_id = request.tenant_id or config.tenant_id
        try:
            user = user_center.authenticate_password(tenant_id=tenant_id, email=request.email, password=request.password)
        except ValueError:
            user = None
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
        created = user_center.create_api_key(
            tenant_id=tenant_id,
            user_id=user.user_id,
            name="password-login",
            expires_at=_token_expires_at(config),
        )
        return _auth_token_response(user_center, created.raw_key)


def _register_admin_routes(*, app: FastAPI, config: AgentRuntimeConfig, user_center: UserCenter) -> None:
    """Register tenant and API key administration routes."""

    def require_admin(x_admin_key: Annotated[str | None, Header(alias="X-Admin-Key")] = None) -> None:
        if not config.api_admin_key:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Admin API is not configured.")
        if x_admin_key != config.api_admin_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key.")

    @app.post("/v1/admin/tenants", response_model=RecordResponse, dependencies=[Depends(require_admin)])
    def create_tenant(request: TenantCreateRequest) -> RecordResponse:
        tenant = user_center.create_tenant(name=request.name, tenant_id=request.tenant_id, metadata=request.metadata)
        return RecordResponse(data=_record_dict(tenant))

    @app.post("/v1/admin/users", response_model=RecordResponse, dependencies=[Depends(require_admin)])
    def create_user(request: UserCreateRequest) -> RecordResponse:
        user = user_center.create_user(
            tenant_id=request.tenant_id,
            email=request.email,
            display_name=request.display_name,
            user_id=request.user_id,
            metadata=request.metadata,
        )
        return RecordResponse(data=_record_dict(user))

    @app.post("/v1/admin/api-keys", response_model=APIKeyResponse, dependencies=[Depends(require_admin)])
    def create_api_key(request: APIKeyCreateRequest) -> APIKeyResponse:
        created = user_center.create_api_key(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            name=request.name,
            expires_at=request.expires_at,
        )
        return APIKeyResponse(data=_record_dict(created.record), api_key=created.raw_key)


def _register_user_routes(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    knowledge_service: KnowledgeBaseService,
    task_runtime: TaskRuntime,
    agent_factory: Callable[..., object],
    pending_turn_store: PendingTurnStore,
    thread_summary_service: ThreadSummaryServiceProtocol,
) -> None:
    """Register authenticated user and chat routes."""
    require_auth = _make_auth_dependency(user_center)
    _register_user_metadata_routes(app=app, user_center=user_center, require_auth=require_auth)
    _register_knowledge_routes(app=app, config=config, knowledge_service=knowledge_service, require_auth=require_auth)
    _register_task_routes(app=app, config=config, user_center=user_center, task_runtime=task_runtime, require_auth=require_auth)
    _register_chat_route(
        app=app,
        config=config,
        user_center=user_center,
        agent_factory=agent_factory,
        pending_turn_store=pending_turn_store,
        thread_summary_service=thread_summary_service,
        require_auth=require_auth,
    )


def _make_auth_dependency(user_center: UserCenter) -> Callable[..., AuthContext]:
    """Build the FastAPI dependency that authenticates API keys."""

    def require_auth(
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> AuthContext:
        raw_key = _raw_api_key(authorization=authorization, x_api_key=x_api_key)
        if raw_key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key.")
        context = user_center.authenticate_api_key(raw_key)
        if context is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
        return context

    return require_auth


def _register_user_metadata_routes(
    *,
    app: FastAPI,
    user_center: UserCenter,
    require_auth: Callable[..., AuthContext],
) -> None:
    """Register authenticated user, thread, and message metadata routes."""
    auth_dependency = cast("AuthContext", Depends(require_auth))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/me", response_model=RecordResponse)
    def me(context: AuthContext = auth_dependency) -> RecordResponse:
        return RecordResponse(
            data={
                "tenant": _record_dict(context.tenant),
                "user": _record_dict(context.user),
                "api_key": {"key_id": context.api_key.key_id, "name": context.api_key.name, "key_prefix": context.api_key.key_prefix},
            }
        )

    @app.post("/v1/auth/logout", response_model=RecordResponse)
    def logout(context: AuthContext = auth_dependency) -> RecordResponse:
        revoked = user_center.revoke_api_key(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            key_id=context.api_key.key_id,
        )
        return RecordResponse(data={"revoked": revoked, "key_id": context.api_key.key_id})

    @app.post("/v1/auth/tokens/{key_id}/revoke", response_model=RecordResponse)
    def revoke_token(key_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        revoked = user_center.revoke_api_key(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            key_id=key_id,
        )
        if not revoked:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found.")
        return RecordResponse(data={"revoked": True, "key_id": key_id})

    _register_thread_routes(app=app, user_center=user_center, auth_dependency=auth_dependency)


def _register_thread_routes(
    *,
    app: FastAPI,
    user_center: UserCenter,
    auth_dependency: AuthContext,
) -> None:
    """Register authenticated thread and message metadata routes."""

    @app.post("/v1/threads", response_model=RecordResponse)
    def create_thread(request: ThreadCreateRequest, context: AuthContext = auth_dependency) -> RecordResponse:
        thread = user_center.create_thread(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            title=request.title,
            thread_id=request.thread_id,
            metadata=request.metadata,
        )
        return RecordResponse(data=_record_dict(thread))

    @app.get("/v1/threads", response_model=RecordResponse)
    def list_threads(context: AuthContext = auth_dependency, limit: int = 50) -> RecordResponse:
        threads = user_center.list_threads(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, limit=limit)
        return RecordResponse(data={"threads": [_record_dict(thread) for thread in threads]})

    @app.get("/v1/threads/{thread_id}/messages", response_model=RecordResponse)
    def list_messages(thread_id: str, context: AuthContext = auth_dependency, limit: int = 100) -> RecordResponse:
        try:
            thread = _require_thread(user_center, context=context, thread_id=thread_id)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        messages = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=limit)
        return RecordResponse(data={"messages": [_record_dict(message) for message in messages]})

    @app.delete("/v1/threads/{thread_id}", response_model=RecordResponse)
    def delete_thread(thread_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        deleted = user_center.delete_thread(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, thread_id=thread_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
        return RecordResponse(data={"deleted": True, "thread_id": thread_id})


def _register_knowledge_routes(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    knowledge_service: KnowledgeBaseService,
    require_auth: Callable[..., AuthContext],
) -> None:
    """Register knowledge-base upload and ingestion routes."""
    auth_dependency = cast("AuthContext", Depends(require_auth))

    @app.post("/v1/knowledge-bases", response_model=RecordResponse)
    def create_knowledge_base(request: KnowledgeBaseCreateRequest, context: AuthContext = auth_dependency) -> RecordResponse:
        kb = _call_knowledge_service(lambda: _create_knowledge_base_for_context(knowledge_service, request, context))
        return RecordResponse(data=_record_dict(kb))

    @app.get("/v1/knowledge-bases", response_model=RecordResponse)
    def list_knowledge_bases(context: AuthContext = auth_dependency, limit: int = 50) -> RecordResponse:
        items = _call_knowledge_service(
            lambda: knowledge_service.list_knowledge_bases(
                tenant_id=context.tenant.tenant_id,
                user_id=context.user.user_id,
                limit=limit,
            )
        )
        return RecordResponse(data={"knowledge_bases": [_record_dict(item) for item in items]})

    @app.delete("/v1/knowledge-bases/{kb_id}", response_model=RecordResponse)
    def delete_knowledge_base(kb_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        kb = _call_knowledge_service(
            lambda: knowledge_service.delete_knowledge_base(
                tenant_id=context.tenant.tenant_id,
                user_id=context.user.user_id,
                kb_id=kb_id,
            )
        )
        return RecordResponse(data=_record_dict(kb))

    @app.post("/v1/knowledge-bases/{kb_id}/documents", response_model=RecordResponse)
    async def upload_document(
        kb_id: str,
        request: Request,
        filename: Annotated[str, Query(min_length=1)],
        parser_mode: Annotated[str | None, Query()] = None,
        context: AuthContext = auth_dependency,
    ) -> RecordResponse:
        content = await request.body()
        document, job = _call_knowledge_service(
            lambda: _upload_document_for_context(
                knowledge_service,
                context=context,
                kb_id=kb_id,
                filename=filename,
                mime_type=request.headers.get("content-type", ""),
                content=content,
                parser_mode=_parser_mode(parser_mode, default=config.ingestion_parser_mode),
            )
        )
        return RecordResponse(data={"document": _record_dict(document), "job": _record_dict(job)})

    @app.get("/v1/knowledge-bases/{kb_id}/documents", response_model=RecordResponse)
    def list_documents(kb_id: str, context: AuthContext = auth_dependency, limit: int = 100) -> RecordResponse:
        documents = _call_knowledge_service(
            lambda: knowledge_service.list_documents(
                tenant_id=context.tenant.tenant_id,
                user_id=context.user.user_id,
                kb_id=kb_id,
                limit=limit,
            )
        )
        return RecordResponse(data={"documents": [_record_dict(document) for document in documents]})

    @app.delete("/v1/knowledge-bases/{kb_id}/documents/{doc_id}", response_model=RecordResponse)
    def delete_document(kb_id: str, doc_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        document = _call_knowledge_service(
            lambda: knowledge_service.delete_document(
                tenant_id=context.tenant.tenant_id,
                user_id=context.user.user_id,
                kb_id=kb_id,
                doc_id=doc_id,
            )
        )
        return RecordResponse(data=_record_dict(document))

    @app.get("/v1/ingestion/jobs/{job_id}", response_model=RecordResponse)
    def get_ingestion_job(job_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        job = knowledge_service.get_job(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            job_id=job_id,
        )
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
        return RecordResponse(data=_record_dict(job))


def _register_task_routes(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    task_runtime: TaskRuntime,
    require_auth: Callable[..., AuthContext],
) -> None:
    """Register task-mode planning and execution routes."""
    auth_dependency = cast("AuthContext", Depends(require_auth))

    @app.post("/v1/tasks", response_model=RecordResponse)
    def create_task(request: TaskCreateRequest, context: AuthContext = auth_dependency) -> RecordResponse:
        _raise_if_user_input_too_large(config, request.goal)
        try:
            chat_request = ChatRequest(message=request.goal, thread_id=request.thread_id, title=request.title)
            thread = _resolve_thread(user_center, context=context, request=chat_request)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        history = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=100)
        result = task_runtime.run(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            thread_id=thread.thread_id,
            goal=request.goal,
            title=request.title,
            messages=history,
            forced_intent=request.intent,
            disabled_tools=_disabled_tools_for_request(request),
        )
        if result.final_answer:
            user_center.append_turn(
                tenant_id=context.tenant.tenant_id,
                thread_id=thread.thread_id,
                user_id=context.user.user_id,
                user_content=request.goal,
                assistant_content=result.final_answer,
                user_metadata={"task_id": result.task.task_id, "task_mode": True},
                assistant_metadata={"task_id": result.task.task_id, "task_mode": True},
            )
        return RecordResponse(data=_task_payload(result))

    _register_task_stream_route(app=app, config=config, user_center=user_center, task_runtime=task_runtime, auth_dependency=auth_dependency)
    _register_task_resume_routes(app=app, config=config, user_center=user_center, task_runtime=task_runtime, auth_dependency=auth_dependency)

    @app.get("/v1/tasks", response_model=RecordResponse)
    def list_tasks(context: AuthContext = auth_dependency, limit: int = 50) -> RecordResponse:
        tasks = task_runtime.store.list_tasks(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, limit=limit)
        return RecordResponse(data={"tasks": [_record_dict(task) for task in tasks]})

    @app.get("/v1/tasks/{task_id}", response_model=RecordResponse)
    def get_task(task_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        task = _task_or_404(task_runtime, context=context, task_id=task_id)
        return RecordResponse(
            data={
                "task": _record_dict(task),
                "steps": [_record_dict(step) for step in task_runtime.store.list_steps(task_id=task.task_id)],
                "events": [_record_dict(event) for event in task_runtime.store.list_events(task_id=task.task_id)],
            }
        )

    @app.get("/v1/tasks/{task_id}/events", response_model=RecordResponse)
    def list_task_events(task_id: str, context: AuthContext = auth_dependency, limit: int = 200) -> RecordResponse:
        task = _task_or_404(task_runtime, context=context, task_id=task_id)
        return RecordResponse(data={"events": [_record_dict(event) for event in task_runtime.store.list_events(task_id=task.task_id, limit=limit)]})

    @app.post("/v1/tasks/{task_id}/cancel", response_model=RecordResponse)
    def cancel_task(task_id: str, context: AuthContext = auth_dependency) -> RecordResponse:
        task = _task_or_404(task_runtime, context=context, task_id=task_id)
        if task.status in {"succeeded", "failed", "cancelled"}:
            return RecordResponse(data=_record_dict(task))
        cancelled = task_runtime.store.update_task(task.task_id, status="cancelled", finished=True)
        task_runtime.store.add_event(task_id=task.task_id, event_type="cancelled", message="Task cancelled.")
        return RecordResponse(data=_record_dict(cancelled))


def _register_task_resume_routes(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    task_runtime: TaskRuntime,
    auth_dependency: AuthContext,
) -> None:
    """Register task human-in-the-loop resume endpoints."""

    @app.post("/v1/tasks/{task_id}/resume", response_model=RecordResponse)
    def resume_task(task_id: str, request: TaskResumeRequest, context: AuthContext = auth_dependency) -> RecordResponse:
        task, history = _prepare_task_resume(
            config=config,
            user_center=user_center,
            task_runtime=task_runtime,
            context=context,
            task_id=task_id,
            request=request,
        )
        result = task_runtime.run_existing_task(
            task=task,
            messages=history,
            forced_intent=request.intent,
            disabled_tools=_disabled_tools_for_request(request),
        )
        if result.final_answer:
            user_center.append_turn(
                tenant_id=context.tenant.tenant_id,
                thread_id=task.thread_id,
                user_id=context.user.user_id,
                user_content=request.message,
                assistant_content=result.final_answer,
                user_metadata={"task_id": result.task.task_id, "task_mode": True, "task_resume": True},
                assistant_metadata={"task_id": result.task.task_id, "task_mode": True, "task_resume": True},
            )
        return RecordResponse(data=_task_payload(result))

    @app.post("/v1/tasks/{task_id}/resume/stream")
    def resume_task_stream(task_id: str, request: TaskResumeRequest, context: AuthContext = auth_dependency) -> StreamingResponse:
        task, history = _prepare_task_resume(
            config=config,
            user_center=user_center,
            task_runtime=task_runtime,
            context=context,
            task_id=task_id,
            request=request,
        )
        return StreamingResponse(
            _task_event_source(
                user_center=user_center,
                task_runtime=task_runtime,
                context=context,
                task=task,
                messages=history,
                user_content=request.message,
                forced_intent=request.intent,
                disabled_tools=_disabled_tools_for_request(request),
                user_metadata={"task_id": task.task_id, "task_mode": True, "task_resume": True},
                assistant_metadata={"task_id": task.task_id, "task_mode": True, "task_resume": True},
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _register_task_stream_route(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    task_runtime: TaskRuntime,
    auth_dependency: AuthContext,
) -> None:
    """Register the streaming task endpoint."""

    @app.post("/v1/tasks/stream")
    def create_task_stream(request: TaskCreateRequest, context: AuthContext = auth_dependency) -> StreamingResponse:
        _raise_if_user_input_too_large(config, request.goal)
        try:
            chat_request = ChatRequest(message=request.goal, thread_id=request.thread_id, title=request.title)
            thread = _resolve_thread(user_center, context=context, request=chat_request)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        history = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=100)
        task = task_runtime.store.create_task(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            thread_id=thread.thread_id,
            goal=request.goal,
            title=request.title,
            intent="task",
        )
        task_runtime.store.add_event(task_id=task.task_id, event_type="created", message="Task created.")
        return StreamingResponse(
            _task_event_source(
                user_center=user_center,
                task_runtime=task_runtime,
                context=context,
                task=task,
                messages=history,
                user_content=request.goal,
                forced_intent=request.intent,
                disabled_tools=_disabled_tools_for_request(request),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _register_chat_route(
    *,
    app: FastAPI,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    agent_factory: Callable[..., object],
    pending_turn_store: PendingTurnStore,
    thread_summary_service: ThreadSummaryServiceProtocol,
    require_auth: Callable[..., AuthContext],
) -> None:
    """Register the chat endpoint."""
    auth_dependency = cast("AuthContext", Depends(require_auth))

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, context: AuthContext = auth_dependency) -> ChatResponse:
        _raise_if_user_input_too_large(config, request.message)
        try:
            thread = _resolve_thread(user_center, context=context, request=request)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        history = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=100)
        try:
            turn = pending_turn_store.start_turn(
                tenant_id=context.tenant.tenant_id,
                user_id=context.user.user_id,
                thread_id=thread.thread_id,
                user_message=request.message,
            )
        except ThreadBusyError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        user_message = _pending_user_message(context=context, thread=thread, content=request.message)
        agent_config = _pending_chat_runtime_config(config, context=context, thread=thread, request=request)
        agent = cast("_Agent", agent_factory(agent_config, system_prompt=_DEFAULT_API_SYSTEM_PROMPT))
        chat_context = _build_chat_context(
            config=agent_config,
            user_center=user_center,
            summary_service=thread_summary_service,
            context=context,
            thread=thread,
            history=history,
            user_message=user_message,
        )
        try:
            result = agent.invoke(
                chat_context.input_data,
                config=_graph_config(context=context, thread=thread),
            )
            content = _assistant_text(cast("Mapping[str, object]", result))
            _, assistant_message = user_center.append_turn(
                tenant_id=context.tenant.tenant_id,
                thread_id=thread.thread_id,
                user_id=context.user.user_id,
                user_content=request.message,
                assistant_content=content,
                user_message_id=user_message.message_id,
                user_metadata={"turn_id": turn.turn_id},
                assistant_metadata={"turn_id": turn.turn_id},
                summary_update=chat_context.summary_update,
            )
            pending_turn_store.mark_committed(turn)
        except Exception as exc:
            pending_turn_store.mark_failed(turn, public_error_message(exc))
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=public_error_message(exc)) from exc
        finally:
            pending_turn_store.release(turn)
        return ChatResponse(thread_id=thread.thread_id, message_id=assistant_message.message_id, content=content)

    @app.post("/v1/chat/stream")
    def chat_stream(request: ChatRequest, context: AuthContext = auth_dependency) -> StreamingResponse:
        _raise_if_user_input_too_large(config, request.message)
        try:
            thread = _resolve_thread(user_center, context=context, request=request)
        except ThreadNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
        history = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=thread.thread_id, limit=100)
        try:
            turn = pending_turn_store.start_turn(
                tenant_id=context.tenant.tenant_id,
                user_id=context.user.user_id,
                thread_id=thread.thread_id,
                user_message=request.message,
            )
        except ThreadBusyError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        user_message = _pending_user_message(context=context, thread=thread, content=request.message)
        agent_config = _pending_chat_runtime_config(config, context=context, thread=thread, request=request)
        agent = cast("_Agent", agent_factory(agent_config, system_prompt=_DEFAULT_API_SYSTEM_PROMPT))
        chat_context = _build_chat_context(
            config=agent_config,
            user_center=user_center,
            summary_service=thread_summary_service,
            context=context,
            thread=thread,
            history=history,
            user_message=user_message,
        )
        return StreamingResponse(
            _chat_event_source(
                agent=agent,
                user_center=user_center,
                pending_turn_store=pending_turn_store,
                turn=turn,
                user_message=user_message,
                context=context,
                thread=thread,
                user_content=request.message,
                input_data=chat_context.input_data,
                summary_update=chat_context.summary_update,
                graph_config=_graph_config(context=context, thread=thread),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _chat_runtime_config(
    config: AgentRuntimeConfig,
    *,
    context: AuthContext,
    thread: ThreadRecord,
    request: ChatRequest,
) -> AgentRuntimeConfig:
    runtime = replace(
        config,
        tenant_id=context.tenant.tenant_id,
        user_id=context.user.user_id,
        thread_id=thread.thread_id,
    )
    if request.rag_enabled is False:
        runtime = replace(runtime, rag_mode="off")
    if request.web_search_enabled is False:
        runtime = replace(runtime, enable_web_search=False)
    return runtime


def _pending_chat_runtime_config(
    config: AgentRuntimeConfig,
    *,
    context: AuthContext,
    thread: ThreadRecord,
    request: ChatRequest,
) -> AgentRuntimeConfig:
    runtime = _chat_runtime_config(config, context=context, thread=thread, request=request)
    memory_mode = "off" if runtime.memory_mode == "off" else "auto"
    return replace(
        runtime,
        enable_checkpointer=False,
        memory_checkpoint_interval=0,
        memory_mode=memory_mode,
    )


def _pending_user_message(*, context: AuthContext, thread: ThreadRecord, content: str) -> MessageRecord:
    return MessageRecord(
        message_id=f"msg_{uuid.uuid4().hex}",
        tenant_id=context.tenant.tenant_id,
        thread_id=thread.thread_id,
        user_id=context.user.user_id,
        role="user",
        content=content,
    )


def _chat_input(
    history: Sequence[MessageRecord],
    *,
    user_message: MessageRecord,
    agent_config: AgentRuntimeConfig,
) -> dict[str, list[object]]:
    return {
        "messages": _agent_messages(
            history,
            user_message=user_message,
            use_checkpointer=agent_config.enable_checkpointer,
        )
    }


def _build_chat_context(
    *,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    summary_service: ThreadSummaryServiceProtocol,
    context: AuthContext,
    thread: ThreadRecord,
    history: Sequence[MessageRecord],
    user_message: MessageRecord,
) -> _ChatContext:
    if config.enable_checkpointer or not config.enable_context_summarization:
        return _ChatContext(input_data=_chat_input(history, user_message=user_message, agent_config=config))

    summary = user_center.get_thread_summary(
        tenant_id=context.tenant.tenant_id,
        user_id=context.user.user_id,
        thread_id=thread.thread_id,
    )
    unsummarized = _messages_after_summary(history, summary)
    summary_update = _maybe_build_summary_update(
        config=config,
        summary_service=summary_service,
        summary=summary,
        unsummarized=unsummarized,
        user_message=user_message,
    )
    visible_summary = summary_update.summary if summary_update is not None else (summary.summary if summary is not None else "")
    visible_history = _visible_history(config=config, unsummarized=unsummarized, summary_update=summary_update)
    messages = _summary_messages(visible_summary) + _to_langchain_messages(visible_history)
    messages.append(HumanMessage(content=user_message.content, id=user_message.message_id))
    return _ChatContext(input_data={"messages": messages}, summary_update=summary_update)


def _messages_after_summary(history: Sequence[MessageRecord], summary: ThreadSummaryRecord | None) -> list[MessageRecord]:
    if summary is None:
        return list(history)
    return [message for message in history if message.message_seq == 0 or message.message_seq > summary.summarized_until_message_seq]


def _maybe_build_summary_update(
    *,
    config: AgentRuntimeConfig,
    summary_service: ThreadSummaryServiceProtocol,
    summary: ThreadSummaryRecord | None,
    unsummarized: Sequence[MessageRecord],
    user_message: MessageRecord,
) -> ThreadSummaryUpdate | None:
    compactable, _kept = _split_compactable_history(config, unsummarized)
    if not compactable:
        return None
    if not _should_summarize(config=config, summary=summary, unsummarized=unsummarized, user_message=user_message):
        return None
    existing_summary = summary.summary if summary is not None else ""
    updated_summary = summary_service.summarize(existing_summary=existing_summary, messages=compactable)
    if not updated_summary:
        return None
    summarized_until = max(message.message_seq for message in compactable)
    counter = token_counter_from_config(config)
    return ThreadSummaryUpdate(
        summary=updated_summary,
        summarized_until_message_seq=summarized_until,
        token_count=counter.count_text(updated_summary),
        metadata={"strategy": "rolling_thread_summary:v1", "compacted_messages": len(compactable)},
    )


def _should_summarize(
    *,
    config: AgentRuntimeConfig,
    summary: ThreadSummaryRecord | None,
    unsummarized: Sequence[MessageRecord],
    user_message: MessageRecord,
) -> bool:
    trigger = config.context_summary_trigger()
    if trigger is None:
        return False
    kind, value = trigger
    if kind == "messages":
        return len(unsummarized) + 1 >= value
    counter = token_counter_from_config(config)
    messages = _summary_messages(summary.summary if summary is not None else "") + _to_langchain_messages(unsummarized)
    messages.append(HumanMessage(content=user_message.content, id=user_message.message_id))
    return messages_tokens(cast("list[BaseMessage]", messages), counter) >= value


def _visible_history(
    *,
    config: AgentRuntimeConfig,
    unsummarized: Sequence[MessageRecord],
    summary_update: ThreadSummaryUpdate | None,
) -> list[MessageRecord]:
    if summary_update is None:
        return list(unsummarized)
    _compactable, kept = _split_compactable_history(config, unsummarized)
    return kept


def _split_compactable_history(config: AgentRuntimeConfig, history: Sequence[MessageRecord]) -> tuple[list[MessageRecord], list[MessageRecord]]:
    keep_count = max(config.context_summary_keep_messages, 1)
    if len(history) <= keep_count:
        return [], list(history)
    return list(history[:-keep_count]), list(history[-keep_count:])


def _summary_messages(summary: str) -> list[object]:
    text = summary.strip()
    if not text:
        return []
    return [SystemMessage(content=f"Persistent conversation summary so far:\n{text}")]


def _graph_config(*, context: AuthContext, thread: ThreadRecord) -> dict[str, dict[str, object]]:
    return {
        "configurable": {
            "tenant_id": context.tenant.tenant_id,
            "user_id": context.user.user_id,
            "thread_id": thread.thread_id,
            "tool_thread_id": thread.thread_id,
            "memory_scope_types": ["user"],
            "memory_scope_ids": [context.user.user_id],
        }
    }


def _chat_event_source(
    *,
    agent: _Agent,
    user_center: UserCenter,
    pending_turn_store: PendingTurnStore,
    turn: PendingTurn,
    user_message: MessageRecord,
    context: AuthContext,
    thread: ThreadRecord,
    user_content: str,
    input_data: Mapping[str, object],
    summary_update: ThreadSummaryUpdate | None,
    graph_config: Mapping[str, object],
) -> Iterator[str]:
    content_parts: list[str] = []
    final_content = ""
    seen_tools: set[str] = set()
    stream_state = _StreamState()
    yield _sse("message_start", {"thread_id": thread.thread_id})
    try:
        stream = agent.stream(input_data, config=graph_config, stream_mode=["messages", "updates"])
        for item in stream:
            mode, data = _stream_item_parts(item)
            if mode == "messages":
                for event in _message_stream_events(data, content_parts=content_parts, seen_tools=seen_tools, stream_state=stream_state):
                    _capture_pending_delta(event, pending_turn_store=pending_turn_store, turn=turn)
                    yield event
            elif mode == "updates":
                if not _tool_call_names_from_update(data):
                    final_content = _assistant_text_from_update(data) or final_content
                yield from _update_stream_events(data, content_parts=content_parts, seen_tools=seen_tools, stream_state=stream_state)
        content = final_content or "".join(content_parts)
        _, assistant_message = user_center.append_turn(
            tenant_id=context.tenant.tenant_id,
            thread_id=thread.thread_id,
            user_id=context.user.user_id,
            user_content=user_content,
            assistant_content=content,
            user_message_id=user_message.message_id,
            user_metadata={"turn_id": turn.turn_id},
            assistant_metadata={"turn_id": turn.turn_id},
            summary_update=summary_update,
        )
        pending_turn_store.mark_committed(turn)
        yield _sse(
            "done",
            {
                "thread_id": thread.thread_id,
                "message_id": assistant_message.message_id,
                "content": content,
                "replace": not stream_state.emitted_text,
            },
        )
    except Exception as exc:  # noqa: BLE001  # Streaming endpoints must send errors after headers are committed.
        pending_turn_store.mark_failed(turn, public_error_message(exc))
        yield _sse("error", {"detail": public_error_message(exc)})
    finally:
        pending_turn_store.release(turn)


def _task_event_source(
    *,
    user_center: UserCenter,
    task_runtime: TaskRuntime,
    context: AuthContext,
    task: TaskRecord,
    messages: Sequence[MessageRecord],
    user_content: str,
    forced_intent: Literal["chat", "task", "rag_query", "memory_query", "clarify", "unsafe"] | None,
    disabled_tools: Sequence[str],
    user_metadata: Mapping[str, object] | None = None,
    assistant_metadata: Mapping[str, object] | None = None,
) -> Iterator[str]:
    state = _TaskWorkerState()

    def run_task() -> None:
        try:
            state.result = task_runtime.run_existing_task(
                task=task,
                messages=messages,
                forced_intent=forced_intent,
                disabled_tools=disabled_tools,
            )
        except BaseException as exc:  # noqa: BLE001  # worker failures must be reported through SSE after headers are sent.
            state.error = exc
        finally:
            state.done.set()

    threading.Thread(target=run_task, daemon=True).start()
    seen_event_ids: set[str] = set()
    last_snapshot = ""
    yield _sse("task_start", _task_snapshot_payload(task_runtime=task_runtime, task=task))
    while not state.done.is_set():
        last_snapshot = yield from _task_progress_events(
            task_runtime=task_runtime,
            task=task,
            seen_event_ids=seen_event_ids,
            last_snapshot=last_snapshot,
        )
        time.sleep(_TASK_STREAM_POLL_SECONDS)
    last_snapshot = yield from _task_progress_events(
        task_runtime=task_runtime,
        task=task,
        seen_event_ids=seen_event_ids,
        last_snapshot=last_snapshot,
    )
    if state.error is not None:
        yield _sse("error", {"detail": public_error_message(state.error)})
        return
    result = state.result
    if result is None:
        yield _sse("error", {"detail": "Task finished without a result."})
        return
    payload = _task_payload(result)
    payload["thread_id"] = task.thread_id
    final_answer = str(payload.get("final_answer") or "")
    if final_answer:
        _, assistant_message = user_center.append_turn(
            tenant_id=context.tenant.tenant_id,
            thread_id=task.thread_id,
            user_id=context.user.user_id,
            user_content=user_content,
            assistant_content=final_answer,
            user_metadata=dict(user_metadata or {"task_id": task.task_id, "task_mode": True}),
            assistant_metadata=dict(assistant_metadata or {"task_id": task.task_id, "task_mode": True}),
        )
        payload["message_id"] = assistant_message.message_id
    yield _sse("done", payload)


def _task_progress_events(
    *,
    task_runtime: TaskRuntime,
    task: TaskRecord,
    seen_event_ids: set[str],
    last_snapshot: str,
) -> Iterator[str]:
    latest_task, steps, events = _task_stream_records(task_runtime=task_runtime, task=task)
    for event in events:
        if event.event_id in seen_event_ids:
            continue
        seen_event_ids.add(event.event_id)
        yield _sse("task_event", {"event": _record_dict(event)})
    payload = _task_snapshot_payload(task_runtime=task_runtime, task=latest_task, steps=steps, events=events)
    snapshot = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if snapshot != last_snapshot:
        yield _sse("task_snapshot", payload)
        return snapshot
    return last_snapshot


def _task_snapshot_payload(
    *,
    task_runtime: TaskRuntime,
    task: TaskRecord,
    steps: Sequence[TaskStepRecord] | None = None,
    events: Sequence[TaskEventRecord] | None = None,
) -> dict[str, object]:
    resolved_steps = list(steps) if steps is not None else task_runtime.store.list_steps(task_id=task.task_id)
    resolved_events = list(events) if events is not None else task_runtime.store.list_events(task_id=task.task_id, limit=_TASK_STREAM_EVENT_LIMIT)
    return {
        "task": _record_dict(task),
        "steps": [_record_dict(step) for step in resolved_steps],
        "events": [_record_dict(event) for event in resolved_events],
    }


def _task_stream_records(*, task_runtime: TaskRuntime, task: TaskRecord) -> tuple[TaskRecord, list[TaskStepRecord], list[TaskEventRecord]]:
    latest_task = task_runtime.store.get_task(tenant_id=task.tenant_id, user_id=task.user_id, task_id=task.task_id) or task
    steps = task_runtime.store.list_steps(task_id=task.task_id)
    events = task_runtime.store.list_events(task_id=task.task_id, limit=_TASK_STREAM_EVENT_LIMIT)
    return latest_task, steps, events


def _stream_item_parts(item: object) -> tuple[str, object]:
    if isinstance(item, tuple):
        if len(item) == _STREAM_ITEM_PAIR_LENGTH:
            mode, data = item
            return str(mode), data
        if len(item) == _STREAM_ITEM_TRIPLE_LENGTH:
            _, mode, data = item
            return str(mode), data
    return "", item


def _message_stream_events(
    data: object,
    *,
    content_parts: list[str],
    seen_tools: set[str],
    stream_state: _StreamState,
) -> Iterator[str]:
    chunk = _stream_message_chunk(data)
    if chunk is None:
        return
    tool_names = _message_tool_names(chunk)
    for tool_name in tool_names:
        if tool_name not in seen_tools:
            seen_tools.add(tool_name)
            yield _sse("status", {"text": _tool_status_text(tool_name), "tool": tool_name})
    if tool_names:
        stream_state.buffered_text_parts.clear()
        stream_state.initial_model_pending = False
        stream_state.tool_started = True
        return
    text = _message_delta_text(chunk)
    if text:
        if stream_state.initial_model_pending and not stream_state.tool_finished:
            stream_state.buffered_text_parts.append(text)
            return
        if stream_state.tool_started and not stream_state.tool_finished:
            return
        content_parts.append(text)
        stream_state.emitted_text = True
        yield _sse("delta", {"text": text})


def _update_stream_events(
    data: object,
    *,
    content_parts: list[str],
    seen_tools: set[str],
    stream_state: _StreamState,
) -> Iterator[str]:
    tool_call_names = _tool_call_names_from_update(data)
    for tool_name in tool_call_names:
        if tool_name not in seen_tools:
            seen_tools.add(tool_name)
            yield _sse("status", {"text": _tool_status_text(tool_name), "tool": tool_name})
    if tool_call_names:
        stream_state.buffered_text_parts.clear()
        stream_state.initial_model_pending = False
        stream_state.tool_started = True
    for tool_name in _tool_result_names(data):
        stream_state.initial_model_pending = False
        stream_state.tool_finished = True
        yield _sse("status", {"text": f"Finished {tool_name}.", "tool": tool_name})
    if stream_state.initial_model_pending and not stream_state.tool_started:
        yield from _flush_buffered_text(content_parts=content_parts, stream_state=stream_state)


def _flush_buffered_text(*, content_parts: list[str], stream_state: _StreamState) -> Iterator[str]:
    buffered = "".join(stream_state.buffered_text_parts)
    stream_state.buffered_text_parts.clear()
    stream_state.initial_model_pending = False
    if not buffered:
        return
    content_parts.append(buffered)
    stream_state.emitted_text = True
    yield _sse("delta", {"text": buffered})


def _stream_message_chunk(data: object) -> object | None:
    if isinstance(data, tuple) and data:
        return data[0]
    return data


def _message_delta_text(message: object) -> str:
    message_type = str(getattr(message, "type", ""))
    class_name = type(message).__name__
    if message_type == "tool" or ("AIMessage" not in class_name and message_type not in {"ai", "AIMessageChunk"}):
        return ""
    return _content_text(getattr(message, "content", ""))


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = cast("Mapping[str, object]", block).get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _message_tool_names(message: object) -> list[str]:
    names: list[str] = []
    for attr in ("tool_calls", "tool_call_chunks"):
        calls = getattr(message, attr, None)
        if not isinstance(calls, list):
            continue
        for call in calls:
            name = _tool_call_name(call)
            if name:
                names.append(name)
    return names


def _tool_call_name(call: object) -> str:
    if isinstance(call, Mapping):
        name = cast("Mapping[str, object]", call).get("name")
        if isinstance(name, str):
            return name
    name = getattr(call, "name", "")
    if isinstance(name, str):
        return name
    return ""


def _assistant_text_from_update(update: object) -> str:
    if not isinstance(update, Mapping):
        return ""
    for value in update.values():
        if isinstance(value, Mapping):
            text = _assistant_text(cast("Mapping[str, object]", value))
            if text:
                return text
    return _assistant_text(cast("Mapping[str, object]", update))


def _tool_call_names_from_update(update: object) -> list[str]:
    names: list[str] = []
    for message in _messages_from_update(update):
        names.extend(_message_tool_names(message))
    return names


def _tool_result_names(update: object) -> list[str]:
    names: list[str] = []
    for message in _messages_from_update(update):
        if getattr(message, "type", None) != "tool":
            continue
        name = getattr(message, "name", "")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _messages_from_update(update: object) -> list[object]:
    if not isinstance(update, Mapping):
        return []
    messages: list[object] = []
    for value in update.values():
        if not isinstance(value, Mapping):
            continue
        update_messages = cast("Mapping[str, object]", value).get("messages")
        if not isinstance(update_messages, list):
            continue
        messages.extend(update_messages)
    return messages


def _tool_status_text(tool_name: str) -> str:
    status_by_tool = {
        "search_knowledge_base": "正在检索知识库...",
        "search_memory": "正在检索长期记忆...",
        "web_search": "正在联网搜索...",
        "web_research": "正在阅读网页...",
        "web_fetch_page": "正在打开网页...",
        "save_memory": "正在保存记忆...",
        "delete_memory": "正在更新记忆...",
    }
    return status_by_tool.get(tool_name, f"正在调用 {tool_name}...")


def _disabled_tools_for_request(request: ChatRequest | TaskCreateRequest | TaskResumeRequest) -> tuple[str, ...]:
    disabled: list[str] = []
    if request.rag_enabled is False:
        disabled.extend(("search_knowledge_base", "rag_agent"))
    if request.web_search_enabled is False:
        disabled.extend(("web_search", "web_research", "web_fetch_page", "web_agent"))
    return tuple(disabled)


def _sse(event: str, payload: Mapping[str, object]) -> str:
    data = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _capture_pending_delta(event: str, *, pending_turn_store: PendingTurnStore, turn: PendingTurn) -> None:
    if not event.startswith("event: delta\n"):
        return
    _, _, raw = event.partition("data: ")
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    text = payload.get("text") if isinstance(payload, Mapping) else None
    if isinstance(text, str):
        pending_turn_store.append_delta(turn.turn_id, text)


def _raise_if_user_input_too_large(config: AgentRuntimeConfig, text: str) -> None:
    try:
        enforce_user_input_budget(config, text, token_counter_from_config(config))
    except TokenBudgetExceeded as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc


def _postgres_user_center(config: AgentRuntimeConfig) -> PostgresUserCenter:
    if not config.postgres_dsn:
        msg = "Set `DEEPAGENTS_POSTGRES_DSN` before starting the API server."
        raise ValueError(msg)
    return PostgresUserCenter(dsn=config.postgres_dsn)


def _call_knowledge_service(action: Callable[[], _KnowledgeResult]) -> _KnowledgeResult:
    try:
        return action()
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        _raise_if_postgres_schema_drift(exc)
        raise


def _create_knowledge_base_for_context(
    knowledge_service: KnowledgeBaseService,
    request: KnowledgeBaseCreateRequest,
    context: AuthContext,
) -> object:
    _ensure_rag_identity(knowledge_service, context)
    return knowledge_service.create_knowledge_base(
        tenant_id=context.tenant.tenant_id,
        user_id=context.user.user_id,
        name=request.name,
        description=request.description,
        visibility=request.visibility,
        metadata=request.metadata,
    )


def _upload_document_for_context(
    knowledge_service: KnowledgeBaseService,
    *,
    context: AuthContext,
    kb_id: str,
    filename: str,
    mime_type: str,
    content: bytes,
    parser_mode: ParserMode,
) -> tuple[object, object]:
    _ensure_rag_identity(knowledge_service, context)
    return knowledge_service.upload_document(
        tenant_id=context.tenant.tenant_id,
        user_id=context.user.user_id,
        kb_id=kb_id,
        filename=filename,
        mime_type=mime_type,
        content=content,
        parser_mode=parser_mode,
    )


def _ensure_rag_identity(knowledge_service: KnowledgeBaseService, context: AuthContext) -> None:
    knowledge_service.ensure_identity(
        tenant_id=context.tenant.tenant_id,
        tenant_name=context.tenant.name,
        user_id=context.user.user_id,
        email=context.user.email,
        display_name=context.user.display_name,
    )


def _raise_if_postgres_schema_drift(exc: Exception) -> None:
    if not _is_postgres_schema_drift(exc):
        return
    msg = (
        "PostgreSQL schema is out of date. Re-apply the runtime schema before using knowledge bases: "
        "`python scripts/bootstrap_runtime.py --skip-rag-index --skip-memory-index`."
    )
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg) from exc


def _is_postgres_schema_drift(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name not in {"UndefinedColumn", "UndefinedTable"}:
        return False
    text = str(exc)
    return any(token in text for token in ("rag_", "deepagent_", "checkpoint", "store"))


def _parser_mode(value: str | None, *, default: ParserMode) -> ParserMode:
    if value in (None, ""):
        return default
    if value in {"auto", "local", "mcp"}:
        return cast("ParserMode", value)
    msg = "`parser_mode` must be one of: auto, local, mcp."
    raise ValueError(msg)


def _raw_api_key(*, authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def _auth_token_response(user_center: UserCenter, raw_key: str) -> AuthTokenResponse:
    context = user_center.authenticate_api_key(raw_key)
    if context is None:
        msg = "Created API key could not be authenticated."
        raise RuntimeError(msg)
    return AuthTokenResponse(
        access_token=raw_key,
        expires_at=context.api_key.expires_at,
        data={
            "tenant": _record_dict(context.tenant),
            "user": _record_dict(context.user),
            "api_key": {"key_id": context.api_key.key_id, "name": context.api_key.name, "key_prefix": context.api_key.key_prefix},
        },
    )


def _token_expires_at(config: AgentRuntimeConfig) -> str | None:
    if config.auth_token_ttl_days <= 0:
        return None
    return (datetime.now(tz=UTC) + timedelta(days=config.auth_token_ttl_days)).isoformat()


def _resolve_thread(
    user_center: UserCenter,
    *,
    context: AuthContext,
    request: ChatRequest,
) -> ThreadRecord:
    if request.thread_id is None:
        return user_center.create_thread(
            tenant_id=context.tenant.tenant_id,
            user_id=context.user.user_id,
            title=request.title,
        )
    return _require_thread(user_center, context=context, thread_id=request.thread_id)


def _require_thread(
    user_center: UserCenter,
    *,
    context: AuthContext,
    thread_id: str,
) -> ThreadRecord:
    thread = user_center.get_thread(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, thread_id=thread_id)
    if thread is None:
        raise ThreadNotFoundError
    return thread


def _to_langchain_messages(messages: Sequence[MessageRecord]) -> list[object]:
    converted: list[object] = []
    for message in messages:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content, id=message.message_id))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content, id=message.message_id))
        elif message.role == "system":
            converted.append(SystemMessage(content=message.content, id=message.message_id))
    return converted


def _agent_messages(
    history: Sequence[MessageRecord],
    *,
    user_message: MessageRecord,
    use_checkpointer: bool,
) -> list[object]:
    current = HumanMessage(content=user_message.content, id=user_message.message_id)
    if use_checkpointer:
        return [current]
    return [*_to_langchain_messages(history), current]


def _assistant_text(result: Mapping[str, object]) -> str:
    messages = result.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = getattr(message, "text", "")
            if isinstance(text, str):
                return text
            content = getattr(message, "content", "")
            return str(content)
    return ""


def _task_or_404(task_runtime: TaskRuntime, *, context: AuthContext, task_id: str) -> TaskRecord:
    task = task_runtime.store.get_task(tenant_id=context.tenant.tenant_id, user_id=context.user.user_id, task_id=task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    return task


def _prepare_task_resume(
    *,
    config: AgentRuntimeConfig,
    user_center: UserCenter,
    task_runtime: TaskRuntime,
    context: AuthContext,
    task_id: str,
    request: TaskResumeRequest,
) -> tuple[TaskRecord, list[MessageRecord]]:
    _raise_if_user_input_too_large(config, request.message)
    task = _task_or_404(task_runtime, context=context, task_id=task_id)
    if task.status != "waiting_user":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task is not waiting for user input.")
    _require_thread(user_center, context=context, thread_id=task.thread_id)
    metadata, clarified_goal = _resume_task_metadata_and_goal(task, answer=request.message)
    resumed = task_runtime.store.update_task(
        task.task_id,
        goal=clarified_goal,
        status="queued",
        final_answer="",
        error_message=None,
        metadata=metadata,
    )
    task_runtime.store.add_event(
        task_id=task.task_id,
        event_type="hitl_resumed",
        message="Task resumed with user clarification.",
        payload={"answer": request.message, "goal": clarified_goal},
    )
    history = user_center.list_messages(tenant_id=context.tenant.tenant_id, thread_id=task.thread_id, limit=100)
    return resumed, history


def _resume_task_metadata_and_goal(task: TaskRecord, *, answer: str) -> tuple[dict[str, object], str]:
    metadata = dict(task.metadata)
    original_goal = str(metadata.get("original_goal") or task.goal)
    existing = metadata.get("hitl")
    hitl = dict(existing) if isinstance(existing, Mapping) else {}
    question = str(hitl.get("question") or task.final_answer or "")
    raw_answers = hitl.get("answers")
    answers = [dict(item) for item in raw_answers if isinstance(item, Mapping)] if isinstance(raw_answers, list) else []
    answers.append({"question": question, "content": answer, "created_at": datetime.now(tz=UTC).isoformat()})
    metadata["original_goal"] = original_goal
    metadata["hitl"] = {
        **hitl,
        "status": "resumed",
        "answers": answers,
        "last_answer": answer,
    }
    return metadata, _clarified_task_goal(original_goal=original_goal, answers=answers)


def _clarified_task_goal(*, original_goal: str, answers: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "[ORIGINAL TASK]",
        original_goal.strip(),
        "",
        "[USER CLARIFICATIONS]",
    ]
    for index, answer in enumerate(answers, start=1):
        question = str(answer.get("question") or "").strip()
        content = str(answer.get("content") or "").strip()
        if question:
            lines.append(f"{index}. Asked: {question}")
            lines.append(f"   User answered: {content}")
        else:
            lines.append(f"{index}. {content}")
    lines.extend(("", "Continue the original task using all clarifications above. Treat the clarified task as the current task."))
    return "\n".join(lines)


def _task_payload(result: object) -> dict[str, object]:
    payload = cast("Any", result)
    task = _task_record_dict(payload.task)
    return {
        "task": task,
        "steps": [_task_record_dict(step) for step in payload.steps],
        "events": [_record_dict(event) for event in payload.events],
        "final_answer": str(payload.final_answer),
    }


def _task_record_dict(record: object) -> dict[str, object]:
    data = _record_dict(record)
    error = data.get("error_message")
    if isinstance(error, str) and error:
        data["error_message"] = public_error_message(error)
    return data


def _record_dict(record: object) -> dict[str, object]:
    data = getattr(record, "__dict__", {})
    if not isinstance(data, dict):
        return {}
    hidden = {"key_hash", "password_hash"}
    return {key: value for key, value in data.items() if key not in hidden}


__all__ = ["create_app"]
