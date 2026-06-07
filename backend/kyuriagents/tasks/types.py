"""Shared task runtime records and planning types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kyuriagents.tools.types import ToolRisk

TaskIntent = Literal["chat", "task", "rag_query", "memory_query", "clarify", "unsafe"]
TaskStatus = Literal["queued", "planning", "running", "waiting_user", "succeeded", "failed", "cancelled"]
TaskStepKind = Literal["think", "tool", "rag", "web", "process", "answer"]
TaskStepStatus = Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled"]
TaskEventType = Literal[
    "created",
    "intent",
    "context",
    "planned",
    "validated",
    "hitl_requested",
    "hitl_resumed",
    "step_started",
    "step_finished",
    "retry",
    "replanned",
    "skipped",
    "failed",
    "cancelled",
    "finished",
]
ObserverDecision = Literal["continue", "retry", "skip", "replan", "ask_user", "finish", "fail"]


@dataclass(frozen=True, kw_only=True)
class TaskRecord:
    """Persisted user task.

    Args:
        task_id: Stable task identifier.
        tenant_id: Tenant that owns the task.
        user_id: User that requested the task.
        thread_id: Conversation thread associated with the task.
        goal: User-provided task goal.
        intent: Routed intent.
        status: Current task lifecycle status.
        title: Human-readable task title.
        final_answer: Final answer when the task succeeds.
        error_message: Failure detail when the task fails.
        metadata: Additional JSON-compatible task metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        finished_at: Completion timestamp.
    """

    task_id: str
    tenant_id: str
    user_id: str
    thread_id: str
    goal: str
    intent: TaskIntent = "task"
    status: TaskStatus = "queued"
    title: str = ""
    final_answer: str = ""
    error_message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    finished_at: str | None = None


@dataclass(frozen=True, kw_only=True)
class TaskStepRecord:
    """Persisted executable plan step."""

    step_id: str
    task_id: str
    step_index: int
    kind: TaskStepKind
    title: str
    instruction: str = ""
    tool_name: str = ""
    input: dict[str, object] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    parallel_group: str = ""
    risk: ToolRisk = "read_only"
    requires_confirmation: bool = False
    status: TaskStepStatus = "pending"
    output: str = ""
    error_message: str | None = None
    attempts: int = 0
    created_at: str = ""
    updated_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True, kw_only=True)
class TaskEventRecord:
    """Persisted task event for progress timelines and audit."""

    event_id: str
    task_id: str
    event_type: TaskEventType
    message: str
    step_id: str | None = None
    payload: dict[str, object] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True, kw_only=True)
class PlannedStep:
    """Step produced by a planner before persistence."""

    kind: TaskStepKind
    title: str
    instruction: str = ""
    tool_name: str = ""
    input: dict[str, object] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    parallel_group: str = ""


@dataclass(frozen=True, kw_only=True)
class TaskPlan:
    """Structured task plan emitted by the planner."""

    goal: str
    steps: tuple[PlannedStep, ...]
    summary: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TaskContext:
    """Planner context produced by the context builder."""

    goal: str
    intent: TaskIntent
    tenant_id: str
    user_id: str
    thread_id: str
    recent_messages: tuple[dict[str, str], ...] = ()
    available_tools: tuple[dict[str, object], ...] = ()
    constraints: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ValidationResult:
    """Result of validating a generated plan."""

    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class StepObservation:
    """Result observed after one step attempt."""

    decision: ObserverDecision
    message: str = ""


__all__ = [
    "ObserverDecision",
    "PlannedStep",
    "StepObservation",
    "TaskContext",
    "TaskEventRecord",
    "TaskEventType",
    "TaskIntent",
    "TaskPlan",
    "TaskRecord",
    "TaskStatus",
    "TaskStepKind",
    "TaskStepRecord",
    "TaskStepStatus",
    "ValidationResult",
]
