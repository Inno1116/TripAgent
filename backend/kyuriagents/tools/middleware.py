"""Middleware for tool policy enforcement and audit logging."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from kyuriagents.tools.policy import ToolPolicy
from kyuriagents.tools.registry import ToolRegistry, default_tool_registry
from kyuriagents.tools.types import ToolCallRecord, ToolCallStatus, ToolDescriptor

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langgraph.types import Command

    from kyuriagents.tools.audit import ToolAuditSink

_SUMMARY_LIMIT = 2_000


@dataclass(frozen=True, kw_only=True)
class ToolContextDefaults:
    """Fallback values used for tool audit records.

    Args:
        tenant_id: Tenant or organization identifier.
        user_id: Optional user identifier.
        thread_id: Optional thread identifier.
    """

    tenant_id: str = "default"
    user_id: str | None = None
    thread_id: str | None = None


class ToolGovernanceMiddleware(AgentMiddleware[Any, Any, Any]):
    """Apply policy and audit logging around every tool call.

    Args:
        registry: Tool descriptor registry.
        policy: Policy used to allow or block tool calls.
        audit_sink: Optional audit sink.
        defaults: Fallback tenant/user/thread values for audit logs.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        policy: ToolPolicy | None = None,
        audit_sink: ToolAuditSink | None = None,
        defaults: ToolContextDefaults | None = None,
    ) -> None:
        """Initialize the middleware."""
        self._registry = registry or default_tool_registry()
        self._policy = policy or ToolPolicy()
        self._audit_sink = audit_sink
        self._defaults = defaults or ToolContextDefaults()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Enforce policy and audit one synchronous tool call."""
        descriptor = self._descriptor(request)
        decision = self._policy.evaluate(descriptor)
        started = time.perf_counter()
        if not decision.allowed:
            message = self._blocked_message(request, decision.reason)
            self._record(request, descriptor, status="blocked", started=started, error=decision.reason, output=message)
            return message

        try:
            result = handler(request)
        except Exception as exc:
            self._record(request, descriptor, status="error", started=started, error=str(exc))
            raise

        self._record(request, descriptor, status=_result_status(result), started=started, output=result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Enforce policy and audit one asynchronous tool call."""
        descriptor = self._descriptor(request)
        decision = self._policy.evaluate(descriptor)
        started = time.perf_counter()
        if not decision.allowed:
            message = self._blocked_message(request, decision.reason)
            self._record(request, descriptor, status="blocked", started=started, error=decision.reason, output=message)
            return message

        try:
            result = await handler(request)
        except Exception as exc:
            self._record(request, descriptor, status="error", started=started, error=str(exc))
            raise

        self._record(request, descriptor, status=_result_status(result), started=started, output=result)
        return result

    def _descriptor(self, request: ToolCallRequest) -> ToolDescriptor:
        name = str(request.tool_call.get("name") or "")
        return self._registry.descriptor_for(name)

    def _blocked_message(self, request: ToolCallRequest, reason: str) -> ToolMessage:
        return ToolMessage(
            content=reason,
            tool_call_id=str(request.tool_call.get("id") or ""),
            name=str(request.tool_call.get("name") or ""),
            status="error",
        )

    def _record(
        self,
        request: ToolCallRequest,
        descriptor: ToolDescriptor,
        *,
        status: ToolCallStatus,
        started: float,
        error: str | None = None,
        output: object = None,
    ) -> None:
        if self._audit_sink is None:
            return
        values = _runtime_values(request.runtime)
        record = ToolCallRecord(
            call_id=str(request.tool_call.get("id") or ""),
            tenant_id=_string_value(values.get("tool_tenant_id") or values.get("tenant_id")) or self._defaults.tenant_id,
            user_id=_optional_string(values.get("tool_user_id") or values.get("user_id")) or self._defaults.user_id,
            thread_id=_optional_string(values.get("tool_thread_id") or values.get("thread_id")) or self._defaults.thread_id,
            tool_name=descriptor.name,
            source=descriptor.source,
            risk=descriptor.risk,
            status=status,
            input_summary=_summarize(request.tool_call.get("args", {})),
            output_summary=_summarize(output),
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
            error=error,
            created_at=datetime.now(tz=UTC).isoformat(),
            metadata={"requires_confirmation": descriptor.requires_confirmation, **descriptor.metadata},
        )
        self._audit_sink.record(record)


def _result_status(result: ToolMessage | Command) -> ToolCallStatus:
    if isinstance(result, ToolMessage) and result.status == "error":
        return "error"
    return "success"


def _summarize(value: object) -> str:
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= _SUMMARY_LIMIT:
        return text
    return text[: _SUMMARY_LIMIT - 14] + "...[truncated]"


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


__all__ = ["ToolContextDefaults", "ToolGovernanceMiddleware"]
