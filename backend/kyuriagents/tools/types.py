"""Shared types for governed runtime tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ToolRisk = Literal["read_only", "external_read", "write", "destructive", "network"]
"""Risk classes used by tool policy checks."""

ToolSource = Literal["builtin", "native", "mcp", "runtime"]
"""Origin of a tool exposed to the agent."""

ToolCallStatus = Literal["success", "error", "blocked"]
"""Lifecycle status recorded for audited tool calls."""


@dataclass(frozen=True, kw_only=True)
class ToolDescriptor:
    """Metadata used to apply policy before a tool executes.

    Args:
        name: Runtime tool name.
        description: Human-readable purpose for policy UIs and audit logs.
        risk: Risk class used by `ToolPolicy`.
        source: Tool origin.
        requires_confirmation: Whether the tool should be blocked until a
            human approval flow is wired.
        timeout_seconds: Optional execution budget for future adapters.
        tags: Optional classification tags.
        metadata: Additional JSON-serializable metadata.
    """

    name: str
    description: str = ""
    risk: ToolRisk = "read_only"
    source: ToolSource = "native"
    requires_confirmation: bool = False
    timeout_seconds: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate descriptor fields."""
        if not self.name:
            msg = "`name` must not be empty."
            raise ValueError(msg)
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            msg = "`timeout_seconds` must be positive when provided."
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class ToolPolicyDecision:
    """Decision returned by `ToolPolicy`.

    Args:
        allowed: Whether the tool may execute.
        reason: Human-readable reason for denials and audit logs.
    """

    allowed: bool
    reason: str = ""


@dataclass(frozen=True, kw_only=True)
class ToolCallRecord:
    """Audit record for one tool call.

    Args:
        call_id: Runtime tool call identifier.
        tenant_id: Tenant or organization identifier.
        user_id: Optional user identifier.
        thread_id: Optional conversation thread identifier.
        tool_name: Runtime tool name.
        source: Tool origin.
        risk: Risk class used for the decision.
        status: Final call status.
        input_summary: Redacted or truncated input summary.
        output_summary: Redacted or truncated output summary.
        duration_ms: Execution duration in milliseconds.
        error: Optional error or denial reason.
        created_at: ISO 8601 timestamp.
        metadata: Additional JSON-serializable metadata.
    """

    call_id: str
    tenant_id: str
    tool_name: str
    source: ToolSource
    risk: ToolRisk
    status: ToolCallStatus
    user_id: str | None = None
    thread_id: str | None = None
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: int = 0
    error: str | None = None
    created_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = [
    "ToolCallRecord",
    "ToolCallStatus",
    "ToolDescriptor",
    "ToolPolicyDecision",
    "ToolRisk",
    "ToolSource",
]
