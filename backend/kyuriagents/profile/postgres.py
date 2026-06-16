"""PostgreSQL-backed structured traveler profile store."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from kyuriagents.profile.types import TravelProfileRecord, normalize_profile_data

if TYPE_CHECKING:
    from typing import LiteralString


class PostgresTravelProfileStore:
    """Store one structured traveler profile per tenant/user."""

    def __init__(self, *, dsn: str) -> None:
        """Initialize the store."""
        self._dsn = dsn

    def get(self, *, tenant_id: str, user_id: str) -> TravelProfileRecord | None:
        """Load a profile."""
        import psycopg  # noqa: PLC0415
        from psycopg.rows import dict_row  # noqa: PLC0415

        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            row = connection.execute(
                """
                SELECT tenant_id, user_id, profile_data, profile_version, created_at, updated_at
                FROM user_travel_profiles
                WHERE tenant_id = %s AND user_id = %s
                """,
                (tenant_id, user_id),
            ).fetchone()
        return _row_to_profile(row) if row is not None else None

    def upsert(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_data: dict[str, object],
        expected_version: int | None = None,
    ) -> TravelProfileRecord:
        """Create or replace a profile, optionally enforcing an expected version."""
        import psycopg  # noqa: PLC0415
        from psycopg.rows import dict_row  # noqa: PLC0415
        from psycopg.types.json import Jsonb  # noqa: PLC0415

        normalized = normalize_profile_data(profile_data)
        with psycopg.connect(self._dsn, row_factory=dict_row) as connection:
            with connection.transaction():
                existing = connection.execute(
                    """
                    SELECT profile_version
                    FROM user_travel_profiles
                    WHERE tenant_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (tenant_id, user_id),
                ).fetchone()
                if existing is not None and expected_version is not None and int(existing["profile_version"]) != expected_version:
                    msg = f"Traveler profile version conflict: expected {expected_version}, found {existing['profile_version']}."
                    raise ValueError(msg)
                if existing is None:
                    row = connection.execute(
                        """
                        INSERT INTO user_travel_profiles (tenant_id, user_id, profile_data, profile_version)
                        VALUES (%s, %s, %s, 1)
                        RETURNING tenant_id, user_id, profile_data, profile_version, created_at, updated_at
                        """,
                        (tenant_id, user_id, Jsonb(normalized)),
                    ).fetchone()
                else:
                    row = connection.execute(
                        """
                        UPDATE user_travel_profiles
                        SET profile_data = %s,
                            profile_version = profile_version + 1,
                            updated_at = now()
                        WHERE tenant_id = %s AND user_id = %s
                        RETURNING tenant_id, user_id, profile_data, profile_version, created_at, updated_at
                        """,
                        (Jsonb(normalized), tenant_id, user_id),
                    ).fetchone()
        if row is None:
            msg = "PostgreSQL did not return the saved traveler profile."
            raise RuntimeError(msg)
        return _row_to_profile(row)


def _row_to_profile(row: Mapping[str, Any]) -> TravelProfileRecord:
    return TravelProfileRecord(
        tenant_id=str(row["tenant_id"]),
        user_id=str(row["user_id"]),
        profile_data=normalize_profile_data(row.get("profile_data") if isinstance(row.get("profile_data"), Mapping) else {}),
        profile_version=int(row.get("profile_version") or 1),
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


__all__ = ["PostgresTravelProfileStore"]
