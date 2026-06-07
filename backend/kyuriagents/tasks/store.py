"""Task runtime stores for planning and execution state."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from kyuriagents.tasks.types import TaskEventRecord, TaskEventType, TaskIntent, TaskRecord, TaskStatus, TaskStepKind, TaskStepRecord, TaskStepStatus

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

    from kyuriagents.tools.types import ToolRisk


class _Cursor(Protocol):
    def execute(self, query: str, params: Mapping[str, object] | None = None) -> object:
        """Execute a SQL statement."""
        ...

    def fetchone(self) -> Mapping[str, object] | None:
        """Fetch one row."""
        ...

    def fetchall(self) -> list[Mapping[str, object]]:
        """Fetch all rows."""
        ...


class _Connection(Protocol):
    def cursor(self, *, row_factory: object) -> AbstractContextManager[_Cursor]:
        """Open a cursor context."""
        ...


class TaskStore(Protocol):
    """Persistence contract for task runtime state."""

    def create_task(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        goal: str,
        title: str = "",
        intent: TaskIntent = "task",
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRecord:
        """Create a task."""
        ...

    def update_task(
        self,
        task_id: str,
        *,
        goal: str | None = None,
        status: TaskStatus | None = None,
        intent: TaskIntent | None = None,
        final_answer: str | None = None,
        error_message: str | None = None,
        metadata: Mapping[str, object] | None = None,
        finished: bool = False,
    ) -> TaskRecord:
        """Update a task."""
        ...

    def get_task(self, *, tenant_id: str, user_id: str, task_id: str) -> TaskRecord | None:
        """Load one visible task."""
        ...

    def list_tasks(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[TaskRecord]:
        """List tasks for a user."""
        ...

    def add_steps(self, *, task_id: str, steps: list[TaskStepRecord]) -> list[TaskStepRecord]:
        """Persist planned steps."""
        ...

    def update_step(
        self,
        step_id: str,
        *,
        status: TaskStepStatus | None = None,
        output: str | None = None,
        error_message: str | None = None,
        attempts: int | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> TaskStepRecord:
        """Update one step."""
        ...

    def list_steps(self, *, task_id: str) -> list[TaskStepRecord]:
        """List steps for a task."""
        ...

    def add_event(
        self,
        *,
        task_id: str,
        event_type: TaskEventType,
        message: str,
        step_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> TaskEventRecord:
        """Append a task event."""
        ...

    def list_events(self, *, task_id: str, limit: int = 200) -> list[TaskEventRecord]:
        """List task events."""
        ...


class InMemoryTaskStore:
    """In-memory task store for tests and local fallback."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._tasks: dict[str, TaskRecord] = {}
        self._steps: dict[str, TaskStepRecord] = {}
        self._events: dict[str, TaskEventRecord] = {}

    def create_task(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        goal: str,
        title: str = "",
        intent: TaskIntent = "task",
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRecord:
        """Create a task."""
        now = _now()
        record = TaskRecord(
            task_id=_id("task"),
            tenant_id=tenant_id,
            user_id=user_id,
            thread_id=thread_id,
            goal=goal,
            intent=intent,
            status="queued",
            title=title,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._tasks[record.task_id] = record
        return record

    def update_task(
        self,
        task_id: str,
        *,
        goal: str | None = None,
        status: TaskStatus | None = None,
        intent: TaskIntent | None = None,
        final_answer: str | None = None,
        error_message: str | None = None,
        metadata: Mapping[str, object] | None = None,
        finished: bool = False,
    ) -> TaskRecord:
        """Update a task."""
        task = self._tasks[task_id]
        updated = replace(
            task,
            goal=task.goal if goal is None else goal,
            status=status or task.status,
            intent=intent or task.intent,
            final_answer=task.final_answer if final_answer is None else final_answer,
            error_message=task.error_message if error_message is None else error_message,
            metadata=task.metadata if metadata is None else dict(metadata),
            updated_at=_now(),
            finished_at=_now() if finished else task.finished_at,
        )
        self._tasks[task_id] = updated
        return updated

    def get_task(self, *, tenant_id: str, user_id: str, task_id: str) -> TaskRecord | None:
        """Load one visible task."""
        task = self._tasks.get(task_id)
        if task is None or task.tenant_id != tenant_id or task.user_id != user_id:
            return None
        return task

    def list_tasks(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[TaskRecord]:
        """List tasks for a user."""
        records = [task for task in self._tasks.values() if task.tenant_id == tenant_id and task.user_id == user_id]
        records.sort(key=lambda task: task.created_at, reverse=True)
        return records[: _positive("limit", limit)]

    def add_steps(self, *, task_id: str, steps: list[TaskStepRecord]) -> list[TaskStepRecord]:
        """Persist planned steps."""
        _ = task_id
        now = _now()
        saved = [replace(step, created_at=step.created_at or now, updated_at=step.updated_at or now) for step in steps]
        for step in saved:
            self._steps[step.step_id] = step
        return saved

    def update_step(
        self,
        step_id: str,
        *,
        status: TaskStepStatus | None = None,
        output: str | None = None,
        error_message: str | None = None,
        attempts: int | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> TaskStepRecord:
        """Update one step."""
        step = self._steps[step_id]
        updated = replace(
            step,
            status=status or step.status,
            output=step.output if output is None else output,
            error_message=step.error_message if error_message is None else error_message,
            attempts=step.attempts if attempts is None else attempts,
            updated_at=_now(),
            started_at=_now() if started else step.started_at,
            finished_at=_now() if finished else step.finished_at,
        )
        self._steps[step_id] = updated
        return updated

    def list_steps(self, *, task_id: str) -> list[TaskStepRecord]:
        """List steps for a task."""
        records = [step for step in self._steps.values() if step.task_id == task_id]
        records.sort(key=lambda step: step.step_index)
        return records

    def add_event(
        self,
        *,
        task_id: str,
        event_type: TaskEventType,
        message: str,
        step_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> TaskEventRecord:
        """Append a task event."""
        record = TaskEventRecord(
            event_id=_id("event"),
            task_id=task_id,
            step_id=step_id,
            event_type=event_type,
            message=message,
            payload=dict(payload or {}),
            created_at=_now(),
        )
        self._events[record.event_id] = record
        return record

    def list_events(self, *, task_id: str, limit: int = 200) -> list[TaskEventRecord]:
        """List task events."""
        records = [event for event in self._events.values() if event.task_id == task_id]
        records.sort(key=lambda event: event.created_at)
        return records[-_positive("limit", limit) :]


class PostgresTaskStore:
    """PostgreSQL-backed task store."""

    def __init__(self, *, dsn: str, connection: _Connection | None = None) -> None:
        """Initialize the store.

        Args:
            dsn: PostgreSQL DSN.
            connection: Optional existing connection for tests.
        """
        self._dsn = dsn
        self._connection = connection

    def create_task(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        goal: str,
        title: str = "",
        intent: TaskIntent = "task",
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRecord:
        """Create a task."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_tasks (task_id, tenant_id, user_id, thread_id, goal, intent, title, metadata)
                VALUES (%(task_id)s, %(tenant_id)s, %(user_id)s, %(thread_id)s, %(goal)s, %(intent)s, %(title)s, %(metadata)s)
                RETURNING *
                """,
                {
                    "task_id": _id("task"),
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "goal": goal,
                    "intent": intent,
                    "title": title,
                    "metadata": _jsonb(dict(metadata or {})),
                },
            )
            row = cursor.fetchone()
        return _task_from_row(_require_row(row))

    def update_task(
        self,
        task_id: str,
        *,
        goal: str | None = None,
        status: TaskStatus | None = None,
        intent: TaskIntent | None = None,
        final_answer: str | None = None,
        error_message: str | None = None,
        metadata: Mapping[str, object] | None = None,
        finished: bool = False,
    ) -> TaskRecord:
        """Update a task."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE agent_tasks
                SET goal = COALESCE(%(goal)s, goal),
                    status = COALESCE(%(status)s, status),
                    intent = COALESCE(%(intent)s, intent),
                    final_answer = COALESCE(%(final_answer)s, final_answer),
                    error_message = %(error_message)s,
                    metadata = COALESCE(%(metadata)s, metadata),
                    finished_at = CASE WHEN %(finished)s THEN now() ELSE finished_at END,
                    updated_at = now()
                WHERE task_id = %(task_id)s
                RETURNING *
                """,
                {
                    "task_id": task_id,
                    "goal": goal,
                    "status": status,
                    "intent": intent,
                    "final_answer": final_answer,
                    "error_message": error_message,
                    "metadata": _jsonb(dict(metadata)) if metadata is not None else None,
                    "finished": finished,
                },
            )
            row = cursor.fetchone()
        return _task_from_row(_require_row(row))

    def get_task(self, *, tenant_id: str, user_id: str, task_id: str) -> TaskRecord | None:
        """Load one visible task."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM agent_tasks
                WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s AND task_id = %(task_id)s
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "task_id": task_id},
            )
            row = cursor.fetchone()
        return _task_from_row(row) if row is not None else None

    def list_tasks(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[TaskRecord]:
        """List tasks for a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM agent_tasks
                WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s
                ORDER BY created_at DESC, task_id DESC
                LIMIT %(limit)s
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "limit": _positive("limit", limit)},
            )
            rows = cursor.fetchall()
        return [_task_from_row(row) for row in rows]

    def add_steps(self, *, task_id: str, steps: list[TaskStepRecord]) -> list[TaskStepRecord]:
        """Persist planned steps."""
        saved: list[TaskStepRecord] = []
        with self._cursor() as cursor:
            for step in steps:
                cursor.execute(
                    """
                    INSERT INTO agent_task_steps (
                        step_id, task_id, step_index, kind, title, instruction, tool_name, input,
                        depends_on, parallel_group, risk, requires_confirmation
                    )
                    VALUES (
                        %(step_id)s, %(task_id)s, %(step_index)s, %(kind)s, %(title)s, %(instruction)s, %(tool_name)s,
                        %(input)s, %(depends_on)s, %(parallel_group)s, %(risk)s, %(requires_confirmation)s
                    )
                    RETURNING *
                    """,
                    _step_params(step, task_id=task_id),
                )
                saved.append(_step_from_row(_require_row(cursor.fetchone())))
        return saved

    def update_step(
        self,
        step_id: str,
        *,
        status: TaskStepStatus | None = None,
        output: str | None = None,
        error_message: str | None = None,
        attempts: int | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> TaskStepRecord:
        """Update one step."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE agent_task_steps
                SET status = COALESCE(%(status)s, status),
                    output = COALESCE(%(output)s, output),
                    error_message = %(error_message)s,
                    attempts = COALESCE(%(attempts)s, attempts),
                    started_at = CASE WHEN %(started)s THEN now() ELSE started_at END,
                    finished_at = CASE WHEN %(finished)s THEN now() ELSE finished_at END,
                    updated_at = now()
                WHERE step_id = %(step_id)s
                RETURNING *
                """,
                {
                    "step_id": step_id,
                    "status": status,
                    "output": output,
                    "error_message": error_message,
                    "attempts": attempts,
                    "started": started,
                    "finished": finished,
                },
            )
            row = cursor.fetchone()
        return _step_from_row(_require_row(row))

    def list_steps(self, *, task_id: str) -> list[TaskStepRecord]:
        """List steps for a task."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM agent_task_steps WHERE task_id = %(task_id)s ORDER BY step_index ASC",
                {"task_id": task_id},
            )
            rows = cursor.fetchall()
        return [_step_from_row(row) for row in rows]

    def add_event(
        self,
        *,
        task_id: str,
        event_type: TaskEventType,
        message: str,
        step_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> TaskEventRecord:
        """Append a task event."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_task_events (event_id, task_id, step_id, event_type, message, payload)
                VALUES (%(event_id)s, %(task_id)s, %(step_id)s, %(event_type)s, %(message)s, %(payload)s)
                RETURNING *
                """,
                {
                    "event_id": _id("event"),
                    "task_id": task_id,
                    "step_id": step_id,
                    "event_type": event_type,
                    "message": message,
                    "payload": _jsonb(dict(payload or {})),
                },
            )
            row = cursor.fetchone()
        return _event_from_row(_require_row(row))

    def list_events(self, *, task_id: str, limit: int = 200) -> list[TaskEventRecord]:
        """List task events."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM (
                    SELECT *
                    FROM agent_task_events
                    WHERE task_id = %(task_id)s
                    ORDER BY created_at DESC, event_id DESC
                    LIMIT %(limit)s
                ) latest_events
                ORDER BY created_at ASC
                """,
                {"task_id": task_id, "limit": _positive("limit", limit)},
            )
            rows = cursor.fetchall()
        return [_event_from_row(row) for row in rows]

    @contextmanager
    def _cursor(self) -> Iterator[_Cursor]:
        connection = self._connection
        if connection is not None:
            with connection.cursor(row_factory=_dict_row()) as cursor:
                yield cursor
            return
        with _connect(self._dsn) as connection, connection.cursor(row_factory=_dict_row()) as cursor:
            yield cursor


def new_step_record(
    *,
    task_id: str,
    step_index: int,
    kind: TaskStepKind,
    title: str,
    instruction: str = "",
    tool_name: str = "",
    input: dict[str, object] | None = None,  # noqa: A002  # record field name
    depends_on: tuple[str, ...] = (),
    parallel_group: str = "",
    risk: ToolRisk = "read_only",
    requires_confirmation: bool = False,
) -> TaskStepRecord:
    """Create a pending step record with a generated identifier.

    Args:
        task_id: Parent task identifier.
        step_index: Zero-based step index.
        kind: Step kind.
        title: Step title.
        instruction: Step instruction.
        tool_name: Tool name for tool steps.
        input: Tool input payload.
        depends_on: Step identifiers this step depends on.
        parallel_group: Optional parallel execution group.
        risk: Tool risk used for validation.
        requires_confirmation: Whether the step requires confirmation.

    Returns:
        Pending step record.
    """
    return TaskStepRecord(
        step_id=_id("step"),
        task_id=task_id,
        step_index=step_index,
        kind=kind,
        title=title,
        instruction=instruction,
        tool_name=tool_name,
        input=dict(input or {}),
        depends_on=depends_on,
        parallel_group=parallel_group,
        risk=risk,
        requires_confirmation=requires_confirmation,
    )


def _step_params(step: TaskStepRecord, *, task_id: str) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "task_id": task_id,
        "step_index": step.step_index,
        "kind": step.kind,
        "title": step.title,
        "instruction": step.instruction,
        "tool_name": step.tool_name,
        "input": _jsonb(dict(step.input)),
        "depends_on": _jsonb(list(step.depends_on)),
        "parallel_group": step.parallel_group,
        "risk": step.risk,
        "requires_confirmation": step.requires_confirmation,
    }


def _task_from_row(row: Mapping[str, object]) -> TaskRecord:
    return TaskRecord(
        task_id=str(row["task_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=str(row["user_id"]),
        thread_id=str(row["thread_id"]),
        goal=str(row["goal"]),
        intent=cast("TaskIntent", str(row.get("intent") or "task")),
        status=cast("TaskStatus", str(row.get("status") or "queued")),
        title=str(row.get("title") or ""),
        final_answer=str(row.get("final_answer") or ""),
        error_message=_optional_str(row.get("error_message")),
        metadata=_dict_value(row.get("metadata")),
        created_at=_str_time(row.get("created_at")),
        updated_at=_str_time(row.get("updated_at")),
        finished_at=_optional_time(row.get("finished_at")),
    )


def _step_from_row(row: Mapping[str, object]) -> TaskStepRecord:
    return TaskStepRecord(
        step_id=str(row["step_id"]),
        task_id=str(row["task_id"]),
        step_index=_int_value(row["step_index"], default=0),
        kind=cast("TaskStepKind", str(row["kind"])),
        title=str(row.get("title") or ""),
        instruction=str(row.get("instruction") or ""),
        tool_name=str(row.get("tool_name") or ""),
        input=_dict_value(row.get("input")),
        depends_on=tuple(str(item) for item in _list_value(row.get("depends_on"))),
        parallel_group=str(row.get("parallel_group") or ""),
        risk=cast("ToolRisk", str(row.get("risk") or "read_only")),
        requires_confirmation=bool(row.get("requires_confirmation")),
        status=cast("TaskStepStatus", str(row.get("status") or "pending")),
        output=str(row.get("output") or ""),
        error_message=_optional_str(row.get("error_message")),
        attempts=_int_value(row.get("attempts"), default=0),
        created_at=_str_time(row.get("created_at")),
        updated_at=_str_time(row.get("updated_at")),
        started_at=_optional_time(row.get("started_at")),
        finished_at=_optional_time(row.get("finished_at")),
    )


def _event_from_row(row: Mapping[str, object]) -> TaskEventRecord:
    return TaskEventRecord(
        event_id=str(row["event_id"]),
        task_id=str(row["task_id"]),
        step_id=_optional_str(row.get("step_id")),
        event_type=cast("TaskEventType", str(row["event_type"])),
        message=str(row.get("message") or ""),
        payload=_dict_value(row.get("payload")),
        created_at=_str_time(row.get("created_at")),
    )


def _connect(dsn: str) -> AbstractContextManager[_Connection]:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `psycopg` to use PostgreSQL task storage."
        raise ImportError(msg) from exc
    return cast("AbstractContextManager[_Connection]", psycopg.connect(dsn))


def _dict_row() -> object:
    try:
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `psycopg` to use PostgreSQL task storage."
        raise ImportError(msg) from exc
    return dict_row


def _jsonb(value: object) -> object:
    try:
        from psycopg.types.json import Jsonb  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `psycopg` to use PostgreSQL task storage."
        raise ImportError(msg) from exc
    return Jsonb(value)


def _require_row(row: Mapping[str, object] | None) -> Mapping[str, object]:
    if row is None:
        msg = "Expected PostgreSQL row."
        raise RuntimeError(msg)
    return row


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _positive(name: str, value: int) -> int:
    if value <= 0:
        msg = f"`{name}` must be positive."
        raise ValueError(msg)
    return value


def _int_value(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if not isinstance(value, str | bytes | bytearray):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _dict_value(value: object) -> dict[str, object]:
    return dict(cast("Mapping[str, object]", value)) if isinstance(value, Mapping) else {}


def _list_value(value: object) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _str_time(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


def _optional_time(value: object) -> str | None:
    if value is None:
        return None
    return _str_time(value)


__all__ = ["InMemoryTaskStore", "PostgresTaskStore", "TaskStore", "new_step_record"]
