"""PostgreSQL bootstrap helpers for KyuriAgents runtime deployments."""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any, LiteralString


def create_postgres_database(
    *,
    admin_dsn: str,
    database: str,
    owner: str | None = None,
) -> bool:
    """Create the runtime PostgreSQL database if it does not already exist.

    Args:
        admin_dsn: DSN for a role allowed to create databases.
        database: Database name to create.
        owner: Optional owner role for the database.

    Returns:
        `True` when a database was created, `False` when it already existed.
    """
    try:
        import psycopg  # noqa: PLC0415
        from psycopg import sql  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to initialize PostgreSQL."
        raise ImportError(msg) from exc

    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (database,),
        ).fetchone()
        if exists is not None:
            return False

        query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
        if owner:
            query += sql.SQL(" OWNER {}").format(sql.Identifier(owner))
        connection.execute(query)
    return True


def apply_kyuriagents_postgres_schemas(
    *,
    dsn: str,
    include_rag: bool = True,
    include_memory: bool = True,
    include_tools: bool = True,
    include_api: bool = True,
    include_langgraph: bool = True,
) -> None:
    """Apply KyuriAgents and LangGraph PostgreSQL schemas.

    Args:
        dsn: Application PostgreSQL DSN.
        include_rag: Whether to apply RAG metadata tables.
        include_memory: Whether to apply dynamic memory tables.
        include_tools: Whether to apply tool and MCP governance tables.
        include_api: Whether to apply user center and API service tables.
        include_langgraph: Whether to run LangGraph checkpointer/store setup.
    """
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to initialize PostgreSQL."
        raise ImportError(msg) from exc

    with psycopg.connect(dsn, autocommit=True) as connection:
        if include_rag:
            connection.execute(cast("LiteralString", _resource_text("kyuriagents.rag", "schemas/postgres_schema.sql")))
        if include_memory:
            connection.execute(cast("LiteralString", _resource_text("kyuriagents.memory", "schemas/postgres_schema.sql")))
        if include_tools:
            connection.execute(cast("LiteralString", _resource_text("kyuriagents.tools", "schemas/postgres_schema.sql")))
        if include_api:
            connection.execute(cast("LiteralString", _resource_text("kyuriagents.server", "schemas/postgres_schema.sql")))

    if include_langgraph:
        _setup_langgraph_postgres(dsn)


def _setup_langgraph_postgres(dsn: str) -> None:
    try:
        import psycopg  # noqa: PLC0415
        from langgraph.checkpoint.postgres import PostgresSaver  # noqa: PLC0415
        from langgraph.store.postgres import PostgresStore  # noqa: PLC0415
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` to initialize LangGraph PostgreSQL tables."
        raise ImportError(msg) from exc

    connect = cast("Any", psycopg.connect)
    with connect(dsn, autocommit=True, row_factory=dict_row) as checkpointer_connection:
        PostgresSaver(checkpointer_connection).setup()
    with connect(dsn, autocommit=True, row_factory=dict_row) as store_connection:
        PostgresStore(store_connection).setup()


def _resource_text(package: str, path: str) -> str:
    return resources.files(package).joinpath(path).read_text(encoding="utf-8")


__all__ = ["apply_kyuriagents_postgres_schemas", "create_postgres_database"]
