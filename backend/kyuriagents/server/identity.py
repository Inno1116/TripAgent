"""User center records and stores for the KyuriAgents API service."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from base64 import b64decode, b64encode
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from contextlib import AbstractContextManager

UserStatus = Literal["active", "disabled"]
APIKeyStatus = Literal["active", "revoked"]
ThreadStatus = Literal["active", "archived", "deleted"]
MessageRole = Literal["user", "assistant", "system", "tool"]

_API_KEY_PREFIX = "kya"
_PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"  # noqa: S105  # Hash format marker, not a secret.
_PASSWORD_HASH_PARTS = 4
_PASSWORD_HASH_ITERATIONS = 210_000
_MIN_PASSWORD_LENGTH = 8
_PASSWORD_SALT_BYTES = 16
_TOKEN_BYTES = 32


class DuplicateUserError(ValueError):
    """Raised when a user email is already registered within a tenant."""


class _Cursor(Protocol):
    """Small psycopg cursor protocol used for optional runtime typing."""

    def execute(self, query: str, params: Mapping[str, object] | None = None) -> object:
        """Execute a SQL statement."""
        ...

    def fetchone(self) -> Mapping[str, object] | None:
        """Fetch one row as a mapping."""
        ...

    def fetchall(self) -> list[Mapping[str, object]]:
        """Fetch all rows as mappings."""
        ...


class _Connection(Protocol):
    """Small psycopg connection protocol used for optional runtime typing."""

    def cursor(self, *, row_factory: object) -> AbstractContextManager[_Cursor]:
        """Open a cursor context."""
        ...


@dataclass(frozen=True, kw_only=True)
class TenantRecord:
    """Tenant or organization account.

    Args:
        tenant_id: Stable tenant identifier.
        name: Human-readable tenant name.
        metadata: JSON-compatible metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    tenant_id: str
    name: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class UserRecord:
    """User account within a tenant."""

    user_id: str
    tenant_id: str
    email: str
    display_name: str = ""
    status: UserStatus = "active"
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class APIKeyRecord:
    """Stored API key metadata. Raw keys are never persisted."""

    key_id: str
    tenant_id: str
    user_id: str
    name: str
    key_prefix: str
    key_hash: str
    status: APIKeyStatus = "active"
    created_at: str = ""
    expires_at: str | None = None
    last_used_at: str | None = None


@dataclass(frozen=True, kw_only=True)
class CreatedAPIKey:
    """API key creation result containing the one-time raw key."""

    record: APIKeyRecord
    raw_key: str


@dataclass(frozen=True, kw_only=True)
class AuthContext:
    """Authenticated tenant and user context."""

    tenant: TenantRecord
    user: UserRecord
    api_key: APIKeyRecord


@dataclass(frozen=True, kw_only=True)
class ThreadRecord:
    """Conversation thread owned by a user."""

    thread_id: str
    tenant_id: str
    user_id: str
    title: str = ""
    status: ThreadStatus = "active"
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class MessageRecord:
    """Persisted chat message."""

    message_id: str
    message_seq: int = 0
    tenant_id: str
    thread_id: str
    role: MessageRole
    content: str
    user_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True, kw_only=True)
class ThreadSummaryRecord:
    """Persisted rolling summary for one conversation thread."""

    thread_id: str
    tenant_id: str
    user_id: str
    summary: str = ""
    summarized_until_message_seq: int = 0
    summary_version: int = 1
    token_count: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class ThreadSummaryUpdate:
    """Candidate summary update committed with a successful turn."""

    summary: str
    summarized_until_message_seq: int
    token_count: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


class UserCenter(Protocol):
    """Store contract for users, API keys, threads, and messages."""

    def ensure_tenant(self, *, name: str, tenant_id: str, metadata: Mapping[str, object] | None = None) -> TenantRecord:
        """Create a tenant if it does not already exist."""
        ...

    def create_tenant(self, *, name: str, tenant_id: str | None = None, metadata: Mapping[str, object] | None = None) -> TenantRecord:
        """Create or update a tenant."""
        ...

    def create_user(
        self,
        *,
        tenant_id: str,
        email: str,
        display_name: str = "",
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UserRecord:
        """Create or update a user."""
        ...

    def create_user_with_password(
        self,
        *,
        tenant_id: str,
        email: str,
        password: str,
        display_name: str = "",
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UserRecord:
        """Create a password-login user."""
        ...

    def authenticate_password(self, *, tenant_id: str, email: str, password: str) -> UserRecord | None:
        """Authenticate a user with email and password."""
        ...

    def create_api_key(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str = "",
        expires_at: str | None = None,
    ) -> CreatedAPIKey:
        """Create an API key for a user."""
        ...

    def authenticate_api_key(self, raw_key: str) -> AuthContext | None:
        """Authenticate a raw API key."""
        ...

    def revoke_api_key(self, *, tenant_id: str, user_id: str, key_id: str) -> bool:
        """Revoke one API key owned by a user."""
        ...

    def create_thread(
        self,
        *,
        tenant_id: str,
        user_id: str,
        title: str = "",
        thread_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ThreadRecord:
        """Create a conversation thread."""
        ...

    def list_threads(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[ThreadRecord]:
        """List visible conversation threads."""
        ...

    def get_thread(self, *, tenant_id: str, user_id: str, thread_id: str) -> ThreadRecord | None:
        """Load one active thread visible to a user."""
        ...

    def delete_thread(self, *, tenant_id: str, user_id: str, thread_id: str) -> bool:
        """Soft-delete one active thread visible to a user."""
        ...

    def append_message(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        role: MessageRole,
        content: str,
        user_id: str | None = None,
        message_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MessageRecord:
        """Append one message to a thread."""
        ...

    def append_turn(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        user_metadata: Mapping[str, object] | None = None,
        assistant_metadata: Mapping[str, object] | None = None,
        summary_update: ThreadSummaryUpdate | None = None,
    ) -> tuple[MessageRecord, MessageRecord]:
        """Append user and assistant messages atomically."""
        ...

    def get_thread_summary(self, *, tenant_id: str, user_id: str, thread_id: str) -> ThreadSummaryRecord | None:
        """Load the rolling summary for a thread."""
        ...

    def list_messages(self, *, tenant_id: str, thread_id: str, limit: int = 100) -> list[MessageRecord]:
        """List thread messages."""
        ...


class InMemoryUserCenter:
    """In-memory `UserCenter` for tests and local prototypes."""

    def __init__(self) -> None:
        """Initialize empty in-memory stores."""
        self._tenants: dict[str, TenantRecord] = {}
        self._users: dict[str, UserRecord] = {}
        self._api_keys: dict[str, APIKeyRecord] = {}
        self._threads: dict[str, ThreadRecord] = {}
        self._messages: dict[str, MessageRecord] = {}
        self._message_seq = 0
        self._summaries: dict[str, ThreadSummaryRecord] = {}
        self._password_hashes: dict[str, str] = {}

    def ensure_tenant(self, *, name: str, tenant_id: str, metadata: Mapping[str, object] | None = None) -> TenantRecord:
        """Create a tenant if it does not already exist."""
        existing = self._tenants.get(tenant_id)
        if existing is not None:
            return existing
        return self.create_tenant(name=name, tenant_id=tenant_id, metadata=metadata)

    def create_tenant(self, *, name: str, tenant_id: str | None = None, metadata: Mapping[str, object] | None = None) -> TenantRecord:
        """Create or update a tenant."""
        now = _now()
        resolved_id = tenant_id or _id("tenant")
        record = TenantRecord(
            tenant_id=resolved_id,
            name=name,
            metadata=dict(metadata or {}),
            created_at=self._tenants.get(resolved_id, TenantRecord(tenant_id=resolved_id, name=name)).created_at or now,
            updated_at=now,
        )
        self._tenants[resolved_id] = record
        return record

    def create_user(
        self,
        *,
        tenant_id: str,
        email: str,
        display_name: str = "",
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UserRecord:
        """Create or update a user."""
        _require_tenant(self._tenants, tenant_id)
        normalized_email = normalize_email(email)
        now = _now()
        existing = next((user for user in self._users.values() if user.tenant_id == tenant_id and user.email == normalized_email), None)
        resolved_id = user_id or (existing.user_id if existing else _id("user"))
        record = UserRecord(
            user_id=resolved_id,
            tenant_id=tenant_id,
            email=normalized_email,
            display_name=display_name,
            metadata=dict(metadata or {}),
            created_at=self._users.get(resolved_id, UserRecord(user_id=resolved_id, tenant_id=tenant_id, email=normalized_email)).created_at or now,
            updated_at=now,
        )
        self._users[resolved_id] = record
        return record

    def create_user_with_password(
        self,
        *,
        tenant_id: str,
        email: str,
        password: str,
        display_name: str = "",
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UserRecord:
        """Create a password-login user."""
        _require_tenant(self._tenants, tenant_id)
        _validate_password(password)
        normalized_email = normalize_email(email)
        existing = next((user for user in self._users.values() if user.tenant_id == tenant_id and user.email == normalized_email), None)
        if existing is not None:
            msg = "A user with this email already exists."
            raise DuplicateUserError(msg)
        user = self.create_user(
            tenant_id=tenant_id,
            email=normalized_email,
            display_name=display_name,
            user_id=user_id,
            metadata=metadata,
        )
        self._password_hashes[user.user_id] = hash_password(password)
        return user

    def authenticate_password(self, *, tenant_id: str, email: str, password: str) -> UserRecord | None:
        """Authenticate a user with email and password."""
        normalized_email = normalize_email(email)
        user = next(
            (candidate for candidate in self._users.values() if candidate.tenant_id == tenant_id and candidate.email == normalized_email), None
        )
        if user is None or user.status != "active":
            return None
        password_hash = self._password_hashes.get(user.user_id)
        if password_hash is None or not verify_password(password, password_hash):
            return None
        return user

    def create_api_key(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str = "",
        expires_at: str | None = None,
    ) -> CreatedAPIKey:
        """Create an API key for a user."""
        _require_tenant(self._tenants, tenant_id)
        user = self._users.get(user_id)
        if user is None or user.tenant_id != tenant_id:
            msg = "`user_id` must belong to the tenant."
            raise ValueError(msg)
        raw_key = generate_api_key()
        record = APIKeyRecord(
            key_id=_id("key"),
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            key_prefix=api_key_prefix(raw_key),
            key_hash=hash_api_key(raw_key),
            created_at=_now(),
            expires_at=expires_at,
        )
        self._api_keys[record.key_id] = record
        return CreatedAPIKey(record=record, raw_key=raw_key)

    def authenticate_api_key(self, raw_key: str) -> AuthContext | None:
        """Authenticate a raw API key."""
        key_hash = hash_api_key(raw_key)
        now = datetime.now(tz=UTC)
        for key_id, key in self._api_keys.items():
            if key.key_hash != key_hash or key.status != "active" or _is_expired(key.expires_at, now=now):
                continue
            user = self._users.get(key.user_id)
            tenant = self._tenants.get(key.tenant_id)
            if user is None or tenant is None or user.status != "active":
                return None
            touched = replace(key, last_used_at=_now())
            self._api_keys[key_id] = touched
            return AuthContext(tenant=tenant, user=user, api_key=touched)
        return None

    def revoke_api_key(self, *, tenant_id: str, user_id: str, key_id: str) -> bool:
        """Revoke one API key owned by a user."""
        key = self._api_keys.get(key_id)
        if key is None or key.tenant_id != tenant_id or key.user_id != user_id or key.status != "active":
            return False
        self._api_keys[key_id] = replace(key, status="revoked")
        return True

    def create_thread(
        self,
        *,
        tenant_id: str,
        user_id: str,
        title: str = "",
        thread_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ThreadRecord:
        """Create a conversation thread."""
        _require_tenant(self._tenants, tenant_id)
        if user_id not in self._users:
            msg = "`user_id` must exist."
            raise ValueError(msg)
        now = _now()
        record = ThreadRecord(
            thread_id=thread_id or _id("thread"),
            tenant_id=tenant_id,
            user_id=user_id,
            title=title,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._threads[record.thread_id] = record
        return record

    def list_threads(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[ThreadRecord]:
        """List visible conversation threads."""
        _positive("limit", limit)
        threads = [
            thread for thread in self._threads.values() if thread.tenant_id == tenant_id and thread.user_id == user_id and thread.status == "active"
        ]
        threads.sort(key=lambda thread: (thread.updated_at, thread.thread_id), reverse=True)
        return threads[:limit]

    def get_thread(self, *, tenant_id: str, user_id: str, thread_id: str) -> ThreadRecord | None:
        """Load one active thread visible to a user."""
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id or thread.user_id != user_id or thread.status != "active":
            return None
        return thread

    def delete_thread(self, *, tenant_id: str, user_id: str, thread_id: str) -> bool:
        """Soft-delete one active thread visible to a user."""
        thread = self.get_thread(tenant_id=tenant_id, user_id=user_id, thread_id=thread_id)
        if thread is None:
            return False
        self._threads[thread_id] = replace(thread, status="deleted", updated_at=_now())
        return True

    def append_message(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        role: MessageRole,
        content: str,
        user_id: str | None = None,
        message_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MessageRecord:
        """Append one message to a thread."""
        thread = self._threads.get(thread_id)
        if thread is None or thread.tenant_id != tenant_id:
            msg = "`thread_id` must belong to the tenant."
            raise ValueError(msg)
        self._message_seq += 1
        record = MessageRecord(
            message_id=message_id or _id("msg"),
            message_seq=self._message_seq,
            tenant_id=tenant_id,
            thread_id=thread_id,
            user_id=user_id,
            role=role,
            content=content,
            metadata=dict(metadata or {}),
            created_at=_now(),
        )
        self._messages[record.message_id] = record
        self._threads[thread_id] = replace(thread, updated_at=record.created_at)
        return record

    def append_turn(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        user_metadata: Mapping[str, object] | None = None,
        assistant_metadata: Mapping[str, object] | None = None,
        summary_update: ThreadSummaryUpdate | None = None,
    ) -> tuple[MessageRecord, MessageRecord]:
        """Append user and assistant messages atomically."""
        user = self.append_message(
            tenant_id=tenant_id,
            thread_id=thread_id,
            user_id=user_id,
            role="user",
            content=user_content,
            message_id=user_message_id,
            metadata=user_metadata,
        )
        assistant = self.append_message(
            tenant_id=tenant_id,
            thread_id=thread_id,
            user_id=user_id,
            role="assistant",
            content=assistant_content,
            message_id=assistant_message_id,
            metadata=assistant_metadata,
        )
        if summary_update is not None:
            self._upsert_thread_summary(
                tenant_id=tenant_id,
                user_id=user_id,
                thread_id=thread_id,
                update=summary_update,
            )
        return user, assistant

    def get_thread_summary(self, *, tenant_id: str, user_id: str, thread_id: str) -> ThreadSummaryRecord | None:
        """Load the rolling summary for a thread."""
        summary = self._summaries.get(thread_id)
        if summary is None or summary.tenant_id != tenant_id or summary.user_id != user_id:
            return None
        return summary

    def list_messages(self, *, tenant_id: str, thread_id: str, limit: int = 100) -> list[MessageRecord]:
        """List thread messages."""
        _positive("limit", limit)
        messages = [message for message in self._messages.values() if message.tenant_id == tenant_id and message.thread_id == thread_id]
        messages.sort(key=lambda message: message.created_at)
        return messages[-limit:]

    def _upsert_thread_summary(
        self,
        *,
        tenant_id: str,
        user_id: str,
        thread_id: str,
        update: ThreadSummaryUpdate,
    ) -> None:
        existing = self._summaries.get(thread_id)
        version = (existing.summary_version + 1) if existing is not None else 1
        now = _now()
        self._summaries[thread_id] = ThreadSummaryRecord(
            thread_id=thread_id,
            tenant_id=tenant_id,
            user_id=user_id,
            summary=update.summary,
            summarized_until_message_seq=update.summarized_until_message_seq,
            summary_version=version,
            token_count=update.token_count,
            metadata=dict(update.metadata),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )


class PostgresUserCenter:
    """PostgreSQL-backed `UserCenter` implementation."""

    def __init__(self, *, dsn: str, connection: _Connection | None = None) -> None:
        """Initialize the store.

        Args:
            dsn: PostgreSQL DSN.
            connection: Optional existing psycopg connection for tests.
        """
        self._dsn = dsn
        self._connection = connection

    def ensure_tenant(self, *, name: str, tenant_id: str, metadata: Mapping[str, object] | None = None) -> TenantRecord:
        """Create a tenant if it does not already exist."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_tenants (tenant_id, name, metadata)
                VALUES (%(tenant_id)s, %(name)s, %(metadata)s)
                ON CONFLICT (tenant_id) DO NOTHING
                RETURNING *
                """,
                {"tenant_id": tenant_id, "name": name, "metadata": _jsonb(dict(metadata or {}))},
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "SELECT * FROM agent_tenants WHERE tenant_id = %(tenant_id)s",
                    {"tenant_id": tenant_id},
                )
                row = cursor.fetchone()
        return _tenant_from_row(_require_row(row))

    def create_tenant(self, *, name: str, tenant_id: str | None = None, metadata: Mapping[str, object] | None = None) -> TenantRecord:
        """Create or update a tenant."""
        resolved_id = tenant_id or _id("tenant")
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_tenants (tenant_id, name, metadata)
                VALUES (%(tenant_id)s, %(name)s, %(metadata)s)
                ON CONFLICT (tenant_id) DO UPDATE
                SET name = EXCLUDED.name, metadata = EXCLUDED.metadata, updated_at = now()
                RETURNING *
                """,
                {"tenant_id": resolved_id, "name": name, "metadata": _jsonb(dict(metadata or {}))},
            )
            row = cursor.fetchone()
        return _tenant_from_row(_require_row(row))

    def create_user(
        self,
        *,
        tenant_id: str,
        email: str,
        display_name: str = "",
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UserRecord:
        """Create or update a user."""
        resolved_id = user_id or _id("user")
        normalized_email = normalize_email(email)
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_users (user_id, tenant_id, email, display_name, metadata)
                VALUES (%(user_id)s, %(tenant_id)s, %(email)s, %(display_name)s, %(metadata)s)
                ON CONFLICT (tenant_id, email) DO UPDATE
                SET display_name = EXCLUDED.display_name, metadata = EXCLUDED.metadata, updated_at = now()
                RETURNING *
                """,
                {
                    "user_id": resolved_id,
                    "tenant_id": tenant_id,
                    "email": normalized_email,
                    "display_name": display_name,
                    "metadata": _jsonb(dict(metadata or {})),
                },
            )
            row = cursor.fetchone()
        return _user_from_row(_require_row(row))

    def create_user_with_password(
        self,
        *,
        tenant_id: str,
        email: str,
        password: str,
        display_name: str = "",
        user_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UserRecord:
        """Create a password-login user."""
        _validate_password(password)
        resolved_id = user_id or _id("user")
        normalized_email = normalize_email(email)
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_users (user_id, tenant_id, email, display_name, metadata, password_hash, password_updated_at)
                VALUES (%(user_id)s, %(tenant_id)s, %(email)s, %(display_name)s, %(metadata)s, %(password_hash)s, now())
                ON CONFLICT (tenant_id, email) DO NOTHING
                RETURNING *
                """,
                {
                    "user_id": resolved_id,
                    "tenant_id": tenant_id,
                    "email": normalized_email,
                    "display_name": display_name,
                    "metadata": _jsonb(dict(metadata or {})),
                    "password_hash": hash_password(password),
                },
            )
            row = cursor.fetchone()
        if row is None:
            msg = "A user with this email already exists."
            raise DuplicateUserError(msg)
        return _user_from_row(row)

    def authenticate_password(self, *, tenant_id: str, email: str, password: str) -> UserRecord | None:
        """Authenticate a user with email and password."""
        normalized_email = normalize_email(email)
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM agent_users
                WHERE tenant_id = %(tenant_id)s
                  AND email = %(email)s
                  AND status = 'active'
                """,
                {"tenant_id": tenant_id, "email": normalized_email},
            )
            row = cursor.fetchone()
            if row is None:
                return None
            password_hash = _optional_str(row.get("password_hash"))
            if password_hash is None or not verify_password(password, password_hash):
                return None
            cursor.execute("UPDATE agent_users SET last_login_at = now() WHERE user_id = %(user_id)s", {"user_id": row["user_id"]})
        return _user_from_row(row)

    def create_api_key(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str = "",
        expires_at: str | None = None,
    ) -> CreatedAPIKey:
        """Create an API key for a user."""
        raw_key = generate_api_key()
        params = {
            "key_id": _id("key"),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "name": name,
            "key_prefix": api_key_prefix(raw_key),
            "key_hash": hash_api_key(raw_key),
            "expires_at": expires_at,
        }
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_api_keys (key_id, tenant_id, user_id, name, key_prefix, key_hash, expires_at)
                VALUES (%(key_id)s, %(tenant_id)s, %(user_id)s, %(name)s, %(key_prefix)s, %(key_hash)s, %(expires_at)s)
                RETURNING *
                """,
                params,
            )
            row = cursor.fetchone()
        return CreatedAPIKey(record=_api_key_from_row(_require_row(row)), raw_key=raw_key)

    def authenticate_api_key(self, raw_key: str) -> AuthContext | None:
        """Authenticate a raw API key."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    k.key_id AS k_key_id,
                    k.tenant_id AS k_tenant_id,
                    k.user_id AS k_user_id,
                    k.name AS k_name,
                    k.key_prefix AS k_key_prefix,
                    k.key_hash AS k_key_hash,
                    k.status AS k_status,
                    k.created_at AS k_created_at,
                    k.expires_at AS k_expires_at,
                    k.last_used_at AS k_last_used_at,
                    t.tenant_id AS t_tenant_id,
                    t.name AS t_name,
                    t.metadata AS t_metadata,
                    t.created_at AS t_created_at,
                    t.updated_at AS t_updated_at,
                    u.user_id AS u_user_id,
                    u.tenant_id AS u_tenant_id,
                    u.email AS u_email,
                    u.display_name AS u_display_name,
                    u.status AS u_status,
                    u.metadata AS u_metadata,
                    u.created_at AS u_created_at,
                    u.updated_at AS u_updated_at
                FROM agent_api_keys k
                JOIN agent_tenants t ON t.tenant_id = k.tenant_id
                JOIN agent_users u ON u.user_id = k.user_id
                WHERE k.key_hash = %(key_hash)s
                  AND k.status = 'active'
                  AND u.status = 'active'
                  AND (k.expires_at IS NULL OR k.expires_at > now())
                """,
                {"key_hash": hash_api_key(raw_key)},
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute("UPDATE agent_api_keys SET last_used_at = now() WHERE key_id = %(key_id)s", {"key_id": row["k_key_id"]})
        return AuthContext(
            tenant=_tenant_from_prefixed_row(row, "t_"),
            user=_user_from_prefixed_row(row, "u_"),
            api_key=_api_key_from_prefixed_row(row, "k_"),
        )

    def revoke_api_key(self, *, tenant_id: str, user_id: str, key_id: str) -> bool:
        """Revoke one API key owned by a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE agent_api_keys
                SET status = 'revoked'
                WHERE tenant_id = %(tenant_id)s
                  AND user_id = %(user_id)s
                  AND key_id = %(key_id)s
                  AND status = 'active'
                RETURNING key_id
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "key_id": key_id},
            )
            row = cursor.fetchone()
        return row is not None

    def create_thread(
        self,
        *,
        tenant_id: str,
        user_id: str,
        title: str = "",
        thread_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ThreadRecord:
        """Create a conversation thread."""
        resolved_id = thread_id or _id("thread")
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_threads (thread_id, tenant_id, user_id, title, metadata)
                VALUES (%(thread_id)s, %(tenant_id)s, %(user_id)s, %(title)s, %(metadata)s)
                RETURNING *
                """,
                {
                    "thread_id": resolved_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "title": title,
                    "metadata": _jsonb(dict(metadata or {})),
                },
            )
            row = cursor.fetchone()
        return _thread_from_row(_require_row(row))

    def list_threads(self, *, tenant_id: str, user_id: str, limit: int = 50) -> list[ThreadRecord]:
        """List visible conversation threads."""
        _positive("limit", limit)
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM agent_threads
                WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s AND status = 'active'
                ORDER BY updated_at DESC, thread_id DESC
                LIMIT %(limit)s
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "limit": limit},
            )
            rows = cursor.fetchall()
        return [_thread_from_row(row) for row in rows]

    def get_thread(self, *, tenant_id: str, user_id: str, thread_id: str) -> ThreadRecord | None:
        """Load one active thread visible to a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM agent_threads
                WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s AND thread_id = %(thread_id)s AND status = 'active'
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "thread_id": thread_id},
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _thread_from_row(row)

    def delete_thread(self, *, tenant_id: str, user_id: str, thread_id: str) -> bool:
        """Soft-delete one active thread visible to a user."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                UPDATE agent_threads
                SET status = 'deleted', updated_at = now()
                WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s AND thread_id = %(thread_id)s AND status = 'active'
                RETURNING thread_id
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "thread_id": thread_id},
            )
            row = cursor.fetchone()
        return row is not None

    def append_message(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        role: MessageRole,
        content: str,
        user_id: str | None = None,
        message_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MessageRecord:
        """Append one message to a thread."""
        params = {
            "message_id": message_id or _id("msg"),
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "metadata": _jsonb(dict(metadata or {})),
        }
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_messages (message_id, tenant_id, thread_id, user_id, role, content, metadata)
                VALUES (%(message_id)s, %(tenant_id)s, %(thread_id)s, %(user_id)s, %(role)s, %(content)s, %(metadata)s)
                RETURNING *
                """,
                params,
            )
            row = cursor.fetchone()
            cursor.execute("UPDATE agent_threads SET updated_at = now() WHERE thread_id = %(thread_id)s", {"thread_id": thread_id})
        return _message_from_row(_require_row(row))

    def append_turn(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        user_id: str,
        user_content: str,
        assistant_content: str,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        user_metadata: Mapping[str, object] | None = None,
        assistant_metadata: Mapping[str, object] | None = None,
        summary_update: ThreadSummaryUpdate | None = None,
    ) -> tuple[MessageRecord, MessageRecord]:
        """Append user and assistant messages atomically."""
        user_params = {
            "message_id": user_message_id or _id("msg"),
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "role": "user",
            "content": user_content,
            "metadata": _jsonb(dict(user_metadata or {})),
        }
        assistant_params = {
            "message_id": assistant_message_id or _id("msg"),
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "role": "assistant",
            "content": assistant_content,
            "metadata": _jsonb(dict(assistant_metadata or {})),
        }
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO agent_messages (message_id, tenant_id, thread_id, user_id, role, content, metadata)
                VALUES (%(message_id)s, %(tenant_id)s, %(thread_id)s, %(user_id)s, %(role)s, %(content)s, %(metadata)s)
                RETURNING *
                """,
                user_params,
            )
            user_row = cursor.fetchone()
            cursor.execute(
                """
                INSERT INTO agent_messages (message_id, tenant_id, thread_id, user_id, role, content, metadata)
                VALUES (%(message_id)s, %(tenant_id)s, %(thread_id)s, %(user_id)s, %(role)s, %(content)s, %(metadata)s)
                RETURNING *
                """,
                assistant_params,
            )
            assistant_row = cursor.fetchone()
            if summary_update is not None:
                cursor.execute(
                    """
                    INSERT INTO agent_thread_summaries (
                        thread_id, tenant_id, user_id, summary,
                        summarized_until_message_seq, summary_version, token_count, metadata
                    )
                    VALUES (
                        %(thread_id)s, %(tenant_id)s, %(user_id)s, %(summary)s,
                        %(summarized_until_message_seq)s, 1, %(token_count)s, %(metadata)s
                    )
                    ON CONFLICT (thread_id) DO UPDATE
                    SET summary = EXCLUDED.summary,
                        summarized_until_message_seq = EXCLUDED.summarized_until_message_seq,
                        summary_version = agent_thread_summaries.summary_version + 1,
                        token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                    """,
                    {
                        "thread_id": thread_id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "summary": summary_update.summary,
                        "summarized_until_message_seq": summary_update.summarized_until_message_seq,
                        "token_count": summary_update.token_count,
                        "metadata": _jsonb(dict(summary_update.metadata)),
                    },
                )
            cursor.execute("UPDATE agent_threads SET updated_at = now() WHERE thread_id = %(thread_id)s", {"thread_id": thread_id})
        return _message_from_row(_require_row(user_row)), _message_from_row(_require_row(assistant_row))

    def get_thread_summary(self, *, tenant_id: str, user_id: str, thread_id: str) -> ThreadSummaryRecord | None:
        """Load the rolling summary for a thread."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM agent_thread_summaries
                WHERE tenant_id = %(tenant_id)s
                  AND user_id = %(user_id)s
                  AND thread_id = %(thread_id)s
                """,
                {"tenant_id": tenant_id, "user_id": user_id, "thread_id": thread_id},
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _thread_summary_from_row(row)

    def list_messages(self, *, tenant_id: str, thread_id: str, limit: int = 100) -> list[MessageRecord]:
        """List thread messages."""
        _positive("limit", limit)
        with self._cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM (
                    SELECT *
                    FROM agent_messages
                    WHERE tenant_id = %(tenant_id)s AND thread_id = %(thread_id)s
                    ORDER BY message_seq DESC
                    LIMIT %(limit)s
                ) latest_messages
                ORDER BY message_seq ASC
                """,
                {"tenant_id": tenant_id, "thread_id": thread_id, "limit": limit},
            )
            rows = cursor.fetchall()
        return [_message_from_row(row) for row in rows]

    @contextmanager
    def _cursor(self) -> Iterator[_Cursor]:
        connection = self._connection
        if connection is not None:
            with connection.cursor(row_factory=_dict_row()) as cursor:
                yield cursor
            return
        with _connect(self._dsn) as connection, connection.cursor(row_factory=_dict_row()) as cursor:
            yield cursor


def generate_api_key() -> str:
    """Generate a raw API key."""
    secret = secrets.token_urlsafe(_TOKEN_BYTES)
    return f"{_API_KEY_PREFIX}_{secret}"


def api_key_prefix(raw_key: str) -> str:
    """Return the display prefix for a raw API key."""
    return raw_key[:16]


def hash_api_key(raw_key: str) -> str:
    """Hash a raw API key for storage."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    """Normalize an email address for account lookup."""
    normalized = email.strip().lower()
    if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        msg = "`email` must be a valid email address."
        raise ValueError(msg)
    return normalized


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    _validate_password(password)
    salt = secrets.token_bytes(_PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PASSWORD_HASH_ITERATIONS)
    return "$".join(
        [
            _PASSWORD_HASH_ALGORITHM,
            str(_PASSWORD_HASH_ITERATIONS),
            b64encode(salt).decode("ascii"),
            b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""
    parts = password_hash.split("$")
    if len(parts) != _PASSWORD_HASH_PARTS or parts[0] != _PASSWORD_HASH_ALGORITHM:
        return False
    try:
        iterations = int(parts[1])
        salt = b64decode(parts[2].encode("ascii"), validate=True)
        expected = b64decode(parts[3].encode("ascii"), validate=True)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return secrets.compare_digest(actual, expected)


def _validate_password(password: str) -> None:
    if len(password) < _MIN_PASSWORD_LENGTH:
        msg = "`password` must be at least 8 characters."
        raise ValueError(msg)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _positive(name: str, value: int) -> None:
    if value <= 0:
        msg = f"`{name}` must be positive."
        raise ValueError(msg)


def _require_tenant(tenants: Mapping[str, TenantRecord], tenant_id: str) -> None:
    if tenant_id not in tenants:
        msg = "`tenant_id` must exist."
        raise ValueError(msg)


def _is_expired(expires_at: str | None, *, now: datetime) -> bool:
    if expires_at is None:
        return False
    parsed = datetime.fromisoformat(expires_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed <= now


def _connect(dsn: str) -> AbstractContextManager[_Connection]:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use `PostgresUserCenter`."
        raise ImportError(msg) from exc
    return cast("AbstractContextManager[_Connection]", psycopg.connect(dsn))


def _dict_row() -> object:
    try:
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use `PostgresUserCenter`."
        raise ImportError(msg) from exc
    return dict_row


def _jsonb(value: object) -> object:
    try:
        from psycopg.types.json import Jsonb  # noqa: PLC0415
    except ImportError as exc:
        msg = "Install `kyuriagents[runtime]` or `psycopg` to use `PostgresUserCenter`."
        raise ImportError(msg) from exc
    return Jsonb(value)


def _require_row(row: Mapping[str, object] | None) -> Mapping[str, object]:
    if row is None:
        msg = "PostgreSQL operation returned no row."
        raise RuntimeError(msg)
    return row


def _tenant_from_row(row: Mapping[str, object]) -> TenantRecord:
    return TenantRecord(
        tenant_id=str(row["tenant_id"]),
        name=str(row["name"]),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _tenant_from_prefixed_row(row: Mapping[str, object], prefix: str) -> TenantRecord:
    return TenantRecord(
        tenant_id=str(row[f"{prefix}tenant_id"]),
        name=str(row[f"{prefix}name"]),
        metadata=cast("Mapping[str, object]", row.get(f"{prefix}metadata") or {}),
        created_at=str(row.get(f"{prefix}created_at", "")),
        updated_at=str(row.get(f"{prefix}updated_at", "")),
    )


def _user_from_row(row: Mapping[str, object]) -> UserRecord:
    return UserRecord(
        user_id=str(row["user_id"]),
        tenant_id=str(row["tenant_id"]),
        email=str(row["email"]),
        display_name=str(row.get("display_name", "")),
        status=cast("UserStatus", row.get("status", "active")),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _user_from_prefixed_row(row: Mapping[str, object], prefix: str) -> UserRecord:
    return UserRecord(
        user_id=str(row[f"{prefix}user_id"]),
        tenant_id=str(row[f"{prefix}tenant_id"]),
        email=str(row[f"{prefix}email"]),
        display_name=str(row.get(f"{prefix}display_name", "")),
        status=cast("UserStatus", row.get(f"{prefix}status", "active")),
        metadata=cast("Mapping[str, object]", row.get(f"{prefix}metadata") or {}),
        created_at=str(row.get(f"{prefix}created_at", "")),
        updated_at=str(row.get(f"{prefix}updated_at", "")),
    )


def _api_key_from_row(row: Mapping[str, object]) -> APIKeyRecord:
    return APIKeyRecord(
        key_id=str(row["key_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=str(row["user_id"]),
        name=str(row.get("name", "")),
        key_prefix=str(row["key_prefix"]),
        key_hash=str(row["key_hash"]),
        status=cast("APIKeyStatus", row.get("status", "active")),
        created_at=str(row.get("created_at", "")),
        expires_at=_optional_str(row.get("expires_at")),
        last_used_at=_optional_str(row.get("last_used_at")),
    )


def _api_key_from_prefixed_row(row: Mapping[str, object], prefix: str) -> APIKeyRecord:
    return APIKeyRecord(
        key_id=str(row[f"{prefix}key_id"]),
        tenant_id=str(row[f"{prefix}tenant_id"]),
        user_id=str(row[f"{prefix}user_id"]),
        name=str(row.get(f"{prefix}name", "")),
        key_prefix=str(row[f"{prefix}key_prefix"]),
        key_hash=str(row[f"{prefix}key_hash"]),
        status=cast("APIKeyStatus", row.get(f"{prefix}status", "active")),
        created_at=str(row.get(f"{prefix}created_at", "")),
        expires_at=_optional_str(row.get(f"{prefix}expires_at")),
        last_used_at=_optional_str(row.get(f"{prefix}last_used_at")),
    )


def _thread_from_row(row: Mapping[str, object]) -> ThreadRecord:
    return ThreadRecord(
        thread_id=str(row["thread_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=str(row["user_id"]),
        title=str(row.get("title", "")),
        status=cast("ThreadStatus", row.get("status", "active")),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _message_from_row(row: Mapping[str, object]) -> MessageRecord:
    return MessageRecord(
        message_id=str(row["message_id"]),
        message_seq=_int_or_zero(row.get("message_seq")),
        tenant_id=str(row["tenant_id"]),
        thread_id=str(row["thread_id"]),
        user_id=_optional_str(row.get("user_id")),
        role=cast("MessageRole", row["role"]),
        content=str(row["content"]),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
    )


def _thread_summary_from_row(row: Mapping[str, object]) -> ThreadSummaryRecord:
    return ThreadSummaryRecord(
        thread_id=str(row["thread_id"]),
        tenant_id=str(row["tenant_id"]),
        user_id=str(row["user_id"]),
        summary=str(row.get("summary") or ""),
        summarized_until_message_seq=_int_or_zero(row.get("summarized_until_message_seq")),
        summary_version=_int_or_zero(row.get("summary_version")) or 1,
        token_count=_int_or_zero(row.get("token_count")),
        metadata=cast("Mapping[str, object]", row.get("metadata") or {}),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _int_or_zero(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(str(value))


__all__ = [
    "APIKeyRecord",
    "AuthContext",
    "CreatedAPIKey",
    "DuplicateUserError",
    "InMemoryUserCenter",
    "MessageRecord",
    "PostgresUserCenter",
    "TenantRecord",
    "ThreadRecord",
    "ThreadSummaryRecord",
    "ThreadSummaryUpdate",
    "UserCenter",
    "UserRecord",
    "api_key_prefix",
    "generate_api_key",
    "hash_api_key",
    "hash_password",
    "normalize_email",
    "verify_password",
]
