"""Redis-backed pending turn state for clean chat commits."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from kyuriagents.runtime.config import AgentRuntimeConfig


class ThreadBusyError(RuntimeError):
    """Raised when another turn is already running for a thread."""


@dataclass(frozen=True, kw_only=True)
class PendingTurn:
    """Redis lease for one in-flight chat turn."""

    turn_id: str
    lock_key: str
    lock_token: str


class PendingTurnStore(Protocol):
    """Store pending chat turns outside the authoritative message table."""

    def start_turn(self, *, tenant_id: str, user_id: str, thread_id: str, user_message: str) -> PendingTurn:
        """Create one pending turn and acquire the per-thread lock."""
        ...

    def append_delta(self, turn_id: str, text: str) -> None:
        """Append one streamed text delta to the pending buffer."""
        ...

    def mark_committed(self, turn: PendingTurn) -> None:
        """Remove pending state after the DB commit succeeds."""
        ...

    def mark_failed(self, turn: PendingTurn, error: str) -> None:
        """Mark a pending turn as failed without touching authoritative messages."""
        ...

    def release(self, turn: PendingTurn) -> None:
        """Release the per-thread lock when the turn ends."""
        ...


class RedisPendingTurnStore:
    """Redis implementation for pending turns, stream buffers, and thread locks."""

    def __init__(self, *, url: str, ttl_seconds: int, lock_ttl_seconds: int, prefix: str = "kyuri") -> None:
        """Initialize the Redis pending-turn store.

        Args:
            url: Redis connection URL.
            ttl_seconds: TTL for pending turn metadata and streamed chunks.
            lock_ttl_seconds: TTL for the per-thread lock.
            prefix: Redis key prefix.
        """
        self._ttl_seconds = ttl_seconds
        self._lock_ttl_seconds = lock_ttl_seconds
        self._prefix = prefix.strip(":") or "kyuri"
        try:
            import redis  # noqa: PLC0415
        except ImportError as exc:
            msg = "Install `redis` to use clean pending-turn commits."
            raise ImportError(msg) from exc
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()

    @classmethod
    def from_config(cls, config: AgentRuntimeConfig) -> RedisPendingTurnStore:
        """Create the Redis store from runtime config.

        Args:
            config: Runtime configuration.

        Returns:
            Redis pending-turn store.
        """
        return cls(
            url=config.redis_url,
            ttl_seconds=config.pending_turn_ttl_seconds,
            lock_ttl_seconds=config.thread_lock_ttl_seconds,
        )

    def start_turn(self, *, tenant_id: str, user_id: str, thread_id: str, user_message: str) -> PendingTurn:
        """Create one pending turn and acquire the per-thread lock."""
        turn_id = f"turn_{secrets.token_hex(16)}"
        lock_key = self._thread_lock_key(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id)
        lock_token = secrets.token_urlsafe(24)
        acquired = self._client.set(lock_key, lock_token, nx=True, ex=self._lock_ttl_seconds)
        if not acquired:
            msg = "Another request is already running for this thread."
            raise ThreadBusyError(msg)
        payload = {
            "turn_id": turn_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "status": "running",
            "user_message": user_message,
            "created_at": _now(),
        }
        self._client.setex(self._turn_key(turn_id), self._ttl_seconds, json.dumps(payload, ensure_ascii=False))
        self._client.expire(self._buffer_key(turn_id), self._ttl_seconds)
        return PendingTurn(turn_id=turn_id, lock_key=lock_key, lock_token=lock_token)

    def append_delta(self, turn_id: str, text: str) -> None:
        """Append one streamed text delta to the pending buffer."""
        if not text:
            return
        key = self._buffer_key(turn_id)
        self._client.rpush(key, text)
        self._client.expire(key, self._ttl_seconds)

    def mark_committed(self, turn: PendingTurn) -> None:
        """Remove pending state after the DB commit succeeds."""
        self._client.delete(self._turn_key(turn.turn_id), self._buffer_key(turn.turn_id))

    def mark_failed(self, turn: PendingTurn, error: str) -> None:
        """Mark a pending turn as failed without touching authoritative messages."""
        key = self._turn_key(turn.turn_id)
        raw = self._client.get(key)
        payload = json.loads(raw) if raw else {"turn_id": turn.turn_id}
        payload.update({"status": "failed", "error": error, "failed_at": _now()})
        self._client.setex(key, self._ttl_seconds, json.dumps(payload, ensure_ascii=False))
        self._client.delete(self._buffer_key(turn.turn_id))

    def release(self, turn: PendingTurn) -> None:
        """Release the per-thread lock when the turn ends."""
        script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
          return redis.call("DEL", KEYS[1])
        end
        return 0
        """
        self._client.eval(script, 1, turn.lock_key, turn.lock_token)

    def _turn_key(self, turn_id: str) -> str:
        return f"{self._prefix}:pending_turn:{turn_id}"

    def _buffer_key(self, turn_id: str) -> str:
        return f"{self._prefix}:pending_turn:{turn_id}:chunks"

    def _thread_lock_key(self, *, tenant_id: str, user_id: str, thread_id: str) -> str:
        return f"{self._prefix}:thread_lock:{tenant_id}:{user_id}:{thread_id}"


class InMemoryPendingTurnStore:
    """In-memory pending-turn store for unit tests."""

    def __init__(self) -> None:
        """Initialize empty pending state."""
        self._turns: dict[str, dict[str, object]] = {}
        self._buffers: dict[str, list[str]] = {}
        self._locks: dict[str, str] = {}

    def start_turn(self, *, tenant_id: str, user_id: str, thread_id: str, user_message: str) -> PendingTurn:
        """Create one pending turn and acquire the per-thread lock."""
        turn_id = f"turn_{secrets.token_hex(16)}"
        lock_key = self._thread_lock_key(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id)
        if lock_key in self._locks:
            msg = "Another request is already running for this thread."
            raise ThreadBusyError(msg)
        lock_token = secrets.token_urlsafe(24)
        self._locks[lock_key] = lock_token
        self._turns[turn_id] = {
            "turn_id": turn_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "status": "running",
            "user_message": user_message,
            "created_at": _now(),
        }
        self._buffers[turn_id] = []
        return PendingTurn(turn_id=turn_id, lock_key=lock_key, lock_token=lock_token)

    def append_delta(self, turn_id: str, text: str) -> None:
        """Append one streamed text delta to the pending buffer."""
        if text:
            self._buffers.setdefault(turn_id, []).append(text)

    def mark_committed(self, turn: PendingTurn) -> None:
        """Remove pending state after the DB commit succeeds."""
        self._turns.pop(turn.turn_id, None)
        self._buffers.pop(turn.turn_id, None)

    def mark_failed(self, turn: PendingTurn, error: str) -> None:
        """Mark a pending turn as failed without touching authoritative messages."""
        payload = self._turns.setdefault(turn.turn_id, {"turn_id": turn.turn_id})
        payload.update({"status": "failed", "error": error, "failed_at": _now()})
        self._buffers.pop(turn.turn_id, None)

    def release(self, turn: PendingTurn) -> None:
        """Release the per-thread lock when the turn ends."""
        if self._locks.get(turn.lock_key) == turn.lock_token:
            self._locks.pop(turn.lock_key, None)

    def _thread_lock_key(self, *, tenant_id: str, user_id: str, thread_id: str) -> str:
        return f"memory:thread_lock:{tenant_id}:{user_id}:{thread_id}"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


__all__ = ["InMemoryPendingTurnStore", "PendingTurn", "PendingTurnStore", "RedisPendingTurnStore", "ThreadBusyError"]
