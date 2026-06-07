"""PostgreSQL-backed long-term memory store."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

from kyuriagents.memory.types import MemoryRecord, MemoryScope, MemorySearchResult
from kyuriagents.rag._text import tokenize

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import TracebackType

    from kyuriagents.memory.types import MemoryScopeType, MemoryStatus, MemoryType, MemoryVisibility


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


class PostgresMemoryStore:
    """`MemoryStore` implementation backed by PostgreSQL."""

    def __init__(
        self,
        *,
        dsn: str | None = None,
        connection: _Connection | None = None,
    ) -> None:
        """Initialize the store.

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

    def upsert(self, memory: MemoryRecord) -> MemoryRecord:
        """Create or replace a memory record."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_memory_items (
                    memory_id, tenant_id, user_id, scope_type, scope_id,
                    memory_type, content, summary, confidence, importance,
                    status, visibility, source_thread_id, source_message_ids,
                    tags, embedding_model, embedding_version, schema_version,
                    expires_at, created_at, updated_at
                )
                VALUES (
                    %(memory_id)s, %(tenant_id)s, %(user_id)s, %(scope_type)s,
                    %(scope_id)s, %(memory_type)s, %(content)s, %(summary)s,
                    %(confidence)s, %(importance)s, %(status)s, %(visibility)s,
                    %(source_thread_id)s, %(source_message_ids)s, %(tags)s,
                    %(embedding_model)s, %(embedding_version)s,
                    %(schema_version)s, %(expires_at)s, %(created_at)s,
                    %(updated_at)s
                )
                ON CONFLICT (memory_id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    user_id = EXCLUDED.user_id,
                    scope_type = EXCLUDED.scope_type,
                    scope_id = EXCLUDED.scope_id,
                    memory_type = EXCLUDED.memory_type,
                    content = EXCLUDED.content,
                    summary = EXCLUDED.summary,
                    confidence = EXCLUDED.confidence,
                    importance = EXCLUDED.importance,
                    status = EXCLUDED.status,
                    visibility = EXCLUDED.visibility,
                    source_thread_id = EXCLUDED.source_thread_id,
                    source_message_ids = EXCLUDED.source_message_ids,
                    tags = EXCLUDED.tags,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_version = EXCLUDED.embedding_version,
                    schema_version = EXCLUDED.schema_version,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = EXCLUDED.updated_at
                RETURNING *
                """,
                _memory_params(memory),
            )
            row = cursor.fetchone()
        if row is None:
            msg = "PostgreSQL did not return the upserted memory row."
            raise RuntimeError(msg)
        return _row_to_memory(row)

    def get(self, memory_id: str, *, scope: MemoryScope) -> MemoryRecord | None:
        """Load one visible memory record."""
        where, params = _scope_where(scope)
        params["memory_id"] = memory_id
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM agent_memory_items
                WHERE memory_id = %(memory_id)s AND {where}
                LIMIT 1
                """,  # noqa: S608  # where clauses are assembled only from fixed fragments.
                params,
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    def search(
        self,
        query: str,
        *,
        scope: MemoryScope,
        limit: int,
    ) -> list[MemorySearchResult]:
        """Search visible memory records using PostgreSQL full-text ranking."""
        if limit <= 0:
            msg = "`limit` must be positive."
            raise ValueError(msg)

        where, params = _scope_where(scope)
        terms = _search_terms(query)
        params.update(
            {
                "query": query,
                "like_query": f"%{query}%",
                "like_terms": [f"%{term}%" for term in terms],
                "limit": limit,
            }
        )
        search_filter = ""
        if query:
            search_filter = """
                AND (
                    to_tsvector('simple', summary || ' ' || content) @@ plainto_tsquery('simple', %(query)s)
                    OR summary ILIKE %(like_query)s
                    OR content ILIKE %(like_query)s
                    OR summary ILIKE ANY(%(like_terms)s)
                    OR content ILIKE ANY(%(like_terms)s)
                )
            """

        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *,
                    (
                        CASE
                            WHEN %(query)s = '' THEN 0
                            ELSE ts_rank_cd(
                                to_tsvector('simple', summary || ' ' || content),
                                plainto_tsquery('simple', %(query)s)
                            )
                        END
                        + importance::float * 0.10
                        + confidence::float * 0.05
                    ) AS search_score
                FROM agent_memory_items
                WHERE {where}
                {search_filter}
                ORDER BY search_score DESC, updated_at DESC, memory_id DESC
                LIMIT %(limit)s
                """,  # noqa: S608  # where clauses are assembled only from fixed fragments.
                params,
            )
            rows = cursor.fetchall()
        return [
            MemorySearchResult(
                memory=_row_to_memory(row),
                score=_float_value(row.get("search_score"), default=0.0),
                lexical_score=_float_value(row.get("search_score"), default=0.0),
            )
            for row in rows
        ]

    def list_memories(self, *, scope: MemoryScope, limit: int = 100) -> list[MemoryRecord]:
        """List visible memory records."""
        if limit <= 0:
            msg = "`limit` must be positive."
            raise ValueError(msg)
        where, params = _scope_where(scope)
        params["limit"] = limit
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM agent_memory_items
                WHERE {where}
                ORDER BY updated_at DESC, memory_id DESC
                LIMIT %(limit)s
                """,  # noqa: S608  # where clauses are assembled only from fixed fragments.
                params,
            )
            rows = cursor.fetchall()
        return [_row_to_memory(row) for row in rows]

    def delete(self, memory_id: str, *, scope: MemoryScope) -> bool:
        """Soft delete a visible memory record."""
        where, params = _scope_where(scope)
        params["memory_id"] = memory_id
        with self._cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE agent_memory_items
                SET status = 'deleted', updated_at = now()
                WHERE memory_id = %(memory_id)s AND {where}
                RETURNING memory_id
                """,  # noqa: S608  # where clauses are assembled only from fixed fragments.
                params,
            )
            row = cursor.fetchone()
        return row is not None

    @contextmanager
    def _cursor(self) -> Iterator[_Cursor]:
        if self._connection is not None:
            with self._connection.cursor(row_factory=_dict_row()) as cursor:
                yield cast("_Cursor", cursor)
            return

        with _connect(self._dsn or "") as connection, connection.cursor(row_factory=_dict_row()) as cursor:
            yield cursor


def _connect(dsn: str) -> _ConnectionContext:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[memory]` or `psycopg` to use `PostgresMemoryStore`."
        raise ImportError(msg) from exc
    return cast("_ConnectionContext", psycopg.connect(dsn))


def _dict_row() -> object:
    try:
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[memory]` or `psycopg` to use `PostgresMemoryStore`."
        raise ImportError(msg) from exc
    return dict_row


def _jsonb(value: object) -> object:
    try:
        from psycopg.types.json import Jsonb  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[memory]` or `psycopg` to use `PostgresMemoryStore`."
        raise ImportError(msg) from exc
    return Jsonb(value)


def _search_terms(query: str) -> tuple[str, ...]:
    terms = tuple(dict.fromkeys(tokenize(query)))
    if terms:
        return terms
    stripped = query.strip()
    return (stripped,) if stripped else ()


def _memory_params(memory: MemoryRecord) -> dict[str, object]:
    now = datetime.now(tz=UTC).isoformat()
    return {
        "memory_id": memory.memory_id,
        "tenant_id": memory.tenant_id,
        "user_id": memory.user_id,
        "scope_type": memory.scope_type,
        "scope_id": memory.scope_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "summary": memory.summary,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "status": memory.status,
        "visibility": memory.visibility,
        "source_thread_id": memory.source_thread_id,
        "source_message_ids": _jsonb(list(memory.source_message_ids)),
        "tags": _jsonb(list(memory.tags)),
        "embedding_model": memory.embedding_model,
        "embedding_version": memory.embedding_version,
        "schema_version": memory.schema_version,
        "expires_at": memory.expires_at,
        "created_at": memory.created_at or now,
        "updated_at": memory.updated_at or now,
    }


def _scope_where(scope: MemoryScope) -> tuple[str, dict[str, object]]:
    clauses = [
        "tenant_id = %(tenant_id)s",
        "(expires_at IS NULL OR expires_at > now())",
        "(user_id IS NULL OR user_id = %(user_id)s)",
        "(visibility <> 'private' OR user_id = %(user_id)s OR (user_id IS NULL AND %(user_id)s IS NULL))",
    ]
    params: dict[str, object] = {
        "tenant_id": scope.tenant_id,
        "user_id": scope.user_id,
    }
    if scope.active_only:
        clauses.append("status = 'active'")
    if scope.scope_types:
        clauses.append("scope_type = ANY(%(scope_types)s)")
        params["scope_types"] = list(scope.scope_types)
    if scope.scope_ids:
        clauses.append("scope_id = ANY(%(scope_ids)s)")
        params["scope_ids"] = list(scope.scope_ids)
    if scope.memory_types:
        clauses.append("memory_type = ANY(%(memory_types)s)")
        params["memory_types"] = list(scope.memory_types)
    if scope.tags:
        clauses.append("tags ?& %(tags)s")
        params["tags"] = list(scope.tags)
    if scope.visibility is not None:
        clauses.append("visibility = %(visibility)s")
        params["visibility"] = scope.visibility
    return " AND ".join(clauses), params


def _row_to_memory(row: Mapping[str, object]) -> MemoryRecord:
    tags = _string_tuple(row.get("tags"))
    source_message_ids = _string_tuple(row.get("source_message_ids"))
    return MemoryRecord(
        memory_id=str(row["memory_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=_optional_str(row.get("user_id")),
        scope_type=cast("MemoryScopeType", row["scope_type"]),
        scope_id=str(row["scope_id"]),
        memory_type=cast("MemoryType", row["memory_type"]),
        content=str(row["content"]),
        summary=str(row.get("summary") or ""),
        visibility=cast("MemoryVisibility", row.get("visibility", "private")),
        confidence=_float_value(row.get("confidence"), default=1.0),
        importance=_float_value(row.get("importance"), default=0.5),
        tags=tags,
        status=cast("MemoryStatus", row.get("status", "active")),
        source_thread_id=_optional_str(row.get("source_thread_id")),
        source_message_ids=source_message_ids,
        created_at=_timestamp(row.get("created_at")),
        updated_at=_timestamp(row.get("updated_at")),
        expires_at=_optional_timestamp(row.get("expires_at")),
        embedding_model=str(row.get("embedding_model") or ""),
        embedding_version=str(row.get("embedding_version") or ""),
        schema_version=_int_value(row.get("schema_version"), default=1),
    )


def _float_value(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float | str):
        return float(value)
    return default


def _int_value(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int | float | str):
        return int(value)
    return default


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _timestamp(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _optional_timestamp(value: object) -> str | None:
    text = _timestamp(value)
    return text or None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = ["PostgresMemoryStore"]
