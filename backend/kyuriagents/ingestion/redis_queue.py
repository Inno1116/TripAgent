"""Redis-backed wake-up queue for document ingestion workers."""

from __future__ import annotations

from collections import deque
from threading import Condition
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from kyuriagents.runtime import AgentRuntimeConfig


class IngestionJobQueue(Protocol):
    """Queue contract used to wake ingestion workers after uploads."""

    def enqueue(self, job_id: str) -> None:
        """Publish one queued job identifier."""
        ...

    def wait_for_job(self, *, timeout_seconds: int) -> str | None:
        """Wait for one queued job identifier."""
        ...


class NoopIngestionJobQueue:
    """Queue implementation used when Redis wakeups are disabled."""

    def enqueue(self, job_id: str) -> None:
        """Ignore one queued job identifier."""
        _ = job_id

    def wait_for_job(self, *, timeout_seconds: int) -> str | None:
        """Return immediately without claiming queue signals."""
        _ = timeout_seconds
        return None


class InMemoryIngestionJobQueue:
    """Small in-process queue for deterministic tests."""

    def __init__(self) -> None:
        """Initialize the queue."""
        self._condition = Condition()
        self._queue: deque[str] = deque()

    def enqueue(self, job_id: str) -> None:
        """Publish one queued job identifier."""
        with self._condition:
            self._queue.append(job_id)
            self._condition.notify()

    def wait_for_job(self, *, timeout_seconds: int) -> str | None:
        """Wait for one queued job identifier."""
        with self._condition:
            if not self._queue and timeout_seconds > 0:
                self._condition.wait(timeout=timeout_seconds)
            if not self._queue:
                return None
            return self._queue.popleft()


class RedisIngestionJobQueue:
    """Redis list queue used as a worker wake-up signal."""

    def __init__(self, *, url: str, queue_name: str) -> None:
        """Initialize the Redis-backed queue.

        Args:
            url: Redis connection URL.
            queue_name: Redis list key for ingestion job ids.
        """
        try:
            import redis  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
            msg = "Install `redis` to use Redis-backed ingestion wakeups."
            raise RuntimeError(msg) from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._queue_name = queue_name

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> RedisIngestionJobQueue:
        """Create a Redis queue from runtime configuration."""
        return cls(url=config.redis_url, queue_name=config.ingestion_redis_queue_name)

    def enqueue(self, job_id: str) -> None:
        """Publish one queued job identifier."""
        self._client.rpush(self._queue_name, job_id)

    def wait_for_job(self, *, timeout_seconds: int) -> str | None:
        """Block until a job identifier is published or the timeout expires."""
        result = self._client.blpop(self._queue_name, timeout=max(timeout_seconds, 0))
        if result is None:
            return None
        _key, job_id = cast("tuple[str, str]", result)
        return job_id


def default_ingestion_job_queue(config: AgentRuntimeConfig) -> IngestionJobQueue:
    """Return the configured ingestion wake-up queue."""
    if config.enable_ingestion_redis_queue:
        return RedisIngestionJobQueue.from_config(config)
    return NoopIngestionJobQueue()


__all__ = [
    "InMemoryIngestionJobQueue",
    "IngestionJobQueue",
    "NoopIngestionJobQueue",
    "RedisIngestionJobQueue",
    "default_ingestion_job_queue",
]
