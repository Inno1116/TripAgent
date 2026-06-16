"""Bootstrap runtime storage for a deployed Kyuriagents server."""

from __future__ import annotations

import argparse
import logging
import time
from typing import TYPE_CHECKING, TypeVar

from kyuriagents.runtime import AgentRuntimeConfig, apply_kyuriagents_postgres_schemas, create_postgres_database

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


def main() -> None:
    """Apply schemas and create runtime indexes."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap PostgreSQL, Elasticsearch, and Milvus runtime objects.")
    parser.add_argument(
        "--create-database",
        action="store_true",
        help="Create `DEEPAGENTS_POSTGRES_DATABASE` through `DEEPAGENTS_POSTGRES_ADMIN_DSN`.",
    )
    parser.add_argument("--reset-indexes", action="store_true", help="Drop and recreate Elasticsearch indexes and Milvus collections.")
    parser.add_argument("--skip-postgres", action="store_true", help="Skip PostgreSQL schema setup.")
    parser.add_argument("--skip-rag-index", action="store_true", help="Skip RAG Elasticsearch/Milvus setup.")
    parser.add_argument("--skip-memory-index", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempts", type=int, default=30, help="Retry attempts while backing services start.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between retry attempts.")
    args = parser.parse_args()

    config = AgentRuntimeConfig.from_env()
    if args.create_database:
        _retry(
            "create PostgreSQL database",
            lambda: _create_database(config),
            attempts=int(args.attempts),
            delay=float(args.delay),
        )
    if not args.skip_postgres:
        _retry(
            "apply PostgreSQL schemas",
            lambda: _apply_postgres(config),
            attempts=int(args.attempts),
            delay=float(args.delay),
        )
    if not args.skip_rag_index:
        _retry(
            "initialize RAG indexes",
            lambda: _init_rag_indexes(config, reset=bool(args.reset_indexes)),
            attempts=int(args.attempts),
            delay=float(args.delay),
        )


def _retry(label: str, action: Callable[[], _T], *, attempts: int, delay: float) -> _T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = action()
            _LOGGER.info("%s: ok", label)
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            _LOGGER.info("%s: waiting for dependencies (%s/%s): %s", label, attempt, attempts, exc)
            time.sleep(delay)
        else:
            return result
    if last_error is not None:
        raise last_error
    msg = f"{label} failed without an exception."
    raise RuntimeError(msg)


def _create_database(config: AgentRuntimeConfig) -> None:
    if not config.postgres_admin_dsn:
        msg = "Set DEEPAGENTS_POSTGRES_ADMIN_DSN before using --create-database."
        raise ValueError(msg)
    created = create_postgres_database(
        admin_dsn=config.postgres_admin_dsn,
        database=config.postgres_database,
    )
    _LOGGER.info("postgres database %s: %s", config.postgres_database, "created" if created else "exists")


def _apply_postgres(config: AgentRuntimeConfig) -> None:
    if not config.postgres_dsn:
        msg = "Set DEEPAGENTS_POSTGRES_DSN before bootstrapping PostgreSQL."
        raise ValueError(msg)
    apply_kyuriagents_postgres_schemas(dsn=config.postgres_dsn)


def _init_rag_indexes(config: AgentRuntimeConfig, *, reset: bool) -> None:
    from stratrag_rag_eval import init_indexes  # noqa: PLC0415

    init_indexes(config, reset=reset)


if __name__ == "__main__":
    main()
