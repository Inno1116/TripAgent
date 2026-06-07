"""Audit sinks for runtime tool calls."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import TracebackType

    from kyuriagents.tools.types import ToolCallRecord


class ToolAuditSink(Protocol):
    """Protocol implemented by tool audit sinks."""

    def record(self, call: ToolCallRecord) -> None:
        """Persist one audit record.

        Args:
            call: Tool call audit record.
        """
        ...


class InMemoryToolAuditSink:
    """In-memory audit sink for tests and local diagnostics."""

    def __init__(self) -> None:
        """Initialize an empty sink."""
        self.records: list[ToolCallRecord] = []

    def record(self, call: ToolCallRecord) -> None:
        """Append one audit record.

        Args:
            call: Tool call audit record.
        """
        self.records.append(call)


class PostgresToolAuditSink:
    """PostgreSQL-backed audit sink for production tool call logs."""

    def __init__(self, *, dsn: str | None = None, connection: _Connection | None = None) -> None:
        """Initialize the sink.

        Args:
            dsn: PostgreSQL connection string.
            connection: Optional existing psycopg-compatible connection.

        Raises:
            ValueError: If neither `dsn` nor `connection` is supplied.
        """
        if dsn is None and connection is None:
            msg = "Either `dsn` or `connection` must be provided."
            raise ValueError(msg)
        self._dsn = dsn
        self._connection = connection

    def record(self, call: ToolCallRecord) -> None:
        """Insert one audit record.

        Args:
            call: Tool call audit record.
        """
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_tool_calls (
                    call_id, tenant_id, user_id, thread_id, tool_name,
                    source, risk, status, input_summary, output_summary,
                    duration_ms, error, metadata, created_at
                )
                VALUES (
                    %(call_id)s, %(tenant_id)s, %(user_id)s, %(thread_id)s,
                    %(tool_name)s, %(source)s, %(risk)s, %(status)s,
                    %(input_summary)s, %(output_summary)s, %(duration_ms)s,
                    %(error)s, %(metadata)s, %(created_at)s
                )
                """,
                {
                    "call_id": call.call_id,
                    "tenant_id": call.tenant_id,
                    "user_id": call.user_id,
                    "thread_id": call.thread_id,
                    "tool_name": call.tool_name,
                    "source": call.source,
                    "risk": call.risk,
                    "status": call.status,
                    "input_summary": call.input_summary,
                    "output_summary": call.output_summary,
                    "duration_ms": call.duration_ms,
                    "error": call.error,
                    "metadata": _jsonb(call.metadata),
                    "created_at": call.created_at,
                },
            )

    @contextmanager
    def _cursor(self) -> Iterator[_Cursor]:
        if self._connection is not None:
            with self._connection.cursor(row_factory=_dict_row()) as cursor:
                yield cast("_Cursor", cursor)
            return

        with _connect(self._dsn or "") as connection, connection.cursor(row_factory=_dict_row()) as cursor:
            yield cursor


class _Cursor(Protocol):
    def execute(self, query: str, params: object = None) -> object:
        """Execute a SQL statement."""
        ...

    def fetchone(self) -> Mapping[str, object] | None:
        """Fetch one row."""
        ...

    def fetchall(self) -> Sequence[Mapping[str, object]]:
        """Fetch all rows."""
        ...


class _Connection(Protocol):
    def cursor(self, **kwargs: object) -> _CursorContext:
        """Create a cursor."""
        ...


class _CursorContext(Protocol):
    def __enter__(self) -> _Cursor:
        """Enter cursor context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit cursor context."""
        ...


class _ConnectionContext(Protocol):
    def __enter__(self) -> _Connection:
        """Enter connection context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit connection context."""
        ...


def _connect(dsn: str) -> _ConnectionContext:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use `PostgresToolAuditSink`."
        raise ImportError(msg) from exc
    return cast("_ConnectionContext", psycopg.connect(dsn))


def _dict_row() -> object:
    try:
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use `PostgresToolAuditSink`."
        raise ImportError(msg) from exc
    return dict_row


def _jsonb(value: Mapping[str, object]) -> object:
    try:
        from psycopg.types.json import Jsonb  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use `PostgresToolAuditSink`."
        raise ImportError(msg) from exc
    return Jsonb(value)


__all__ = [
    "InMemoryToolAuditSink",
    "PostgresToolAuditSink",
    "ToolAuditSink",
]
