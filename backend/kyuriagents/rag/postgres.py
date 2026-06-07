"""PostgreSQL helpers for RAG retrieval."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from contextlib import AbstractContextManager

    from kyuriagents.rag.types import RetrievedChunk


class _Cursor(Protocol):
    def execute(self, query: str, params: object | None = None) -> object:
        """Execute a SQL statement."""
        ...

    def fetchall(self) -> list[Mapping[str, object]]:
        """Fetch all rows."""
        ...


class _Connection(Protocol):
    def cursor(self, *, row_factory: object) -> AbstractContextManager[_Cursor]:
        """Open a cursor context."""
        ...


class PostgresChunkTextHydrator:
    """Hydrate RAG chunk text from PostgreSQL by `chunk_id`.

    PostgreSQL is treated as the source of truth for original chunk text.
    Elasticsearch may keep a searchable copy, while Milvus only needs ids and
    vector metadata for ANN retrieval.
    """

    def __init__(self, *, dsn: str, connection: _Connection | None = None) -> None:
        """Initialize the hydrator.

        Args:
            dsn: PostgreSQL DSN.
            connection: Optional existing connection for tests.
        """
        self._dsn = dsn
        self._connection = connection

    def hydrate(self, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Fill missing candidate text from PostgreSQL.

        Args:
            candidates: Fused retrieval candidates.

        Returns:
            Candidates with PostgreSQL text when a row exists.
        """
        if not candidates:
            return []
        missing_ids = tuple(candidate.chunk_id for candidate in candidates if not candidate.text)
        if not missing_ids:
            return list(candidates)
        texts = self._load_texts(missing_ids)
        hydrated: list[RetrievedChunk] = []
        for candidate in candidates:
            text = texts.get(candidate.chunk_id)
            hydrated.append(candidate.with_text(text) if text else candidate)
        return hydrated

    def _load_texts(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, chunk_text
                FROM rag_chunks
                WHERE chunk_id = ANY(%(chunk_ids)s)
                  AND is_active = true
                """,
                {"chunk_ids": list(chunk_ids)},
            )
            rows = cursor.fetchall()
        return {str(row["chunk_id"]): str(row.get("chunk_text") or "") for row in rows}

    @contextmanager
    def _cursor(self) -> Iterator[_Cursor]:
        connection = self._connection
        if connection is not None:
            with connection.cursor(row_factory=_dict_row()) as cursor:
                yield cursor
            return
        with _connect(self._dsn) as connection, connection.cursor(row_factory=_dict_row()) as cursor:
            yield cursor


def _connect(dsn: str) -> AbstractContextManager[_Connection]:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to hydrate RAG chunks from PostgreSQL."
        raise ImportError(msg) from exc
    return cast("AbstractContextManager[_Connection]", psycopg.connect(dsn))


def _dict_row() -> object:
    try:
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to hydrate RAG chunks from PostgreSQL."
        raise ImportError(msg) from exc
    return dict_row


__all__ = ["PostgresChunkTextHydrator"]
