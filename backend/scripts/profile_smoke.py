"""Smoke test structured traveler profile storage."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import psycopg

from kyuriagents.profile import PostgresTravelProfileStore, TravelProfileService, format_travel_profile_context
from kyuriagents.runtime import AgentRuntimeConfig


def main() -> None:
    """Upsert and read one structured traveler profile."""
    _load_runtime_env()
    config = AgentRuntimeConfig.from_env()
    if not config.postgres_dsn:
        msg = "Set DEEPAGENTS_POSTGRES_DSN before running profile smoke tests."
        raise ValueError(msg)
    user_id = config.user_id or "local-user"
    _ensure_smoke_identity(dsn=config.postgres_dsn, tenant_id=config.tenant_id, user_id=user_id)
    service = TravelProfileService(PostgresTravelProfileStore(dsn=config.postgres_dsn))
    profile = {
        "hard_constraints": {"mobility": "避免过于消耗体力的路线"},
        "dynamic_preferences": {"interests": ["历史古迹", "本地美食"], "budget": "中等预算"},
        "trip_state": {"last_test": "游客画像 smoke 测试"},
        "history_facts": ["用户测试过结构化游客画像存储。"],
    }
    saved = service.update_profile(tenant_id=config.tenant_id, user_id=user_id, profile_data=profile)
    loaded = service.get_profile(tenant_id=config.tenant_id, user_id=user_id)
    print(
        json.dumps(
            {
                "tenant_id": loaded.tenant_id,
                "user_id": loaded.user_id,
                "profile_version": loaded.profile_version,
                "profile_data": loaded.profile_data,
                "context_preview": format_travel_profile_context(loaded, max_chars=800),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert saved.profile_version == loaded.profile_version


def _ensure_smoke_identity(*, dsn: str, tenant_id: str, user_id: str) -> None:
    """Create the smoke tenant/user when the local database lacks them."""
    local_part = re.sub(r"[^a-zA-Z0-9_.+-]+", "-", user_id).strip("-") or "local-user"
    email = f"{local_part}@profile-smoke.local"
    with psycopg.connect(dsn) as connection:
        connection.execute(
            """
            INSERT INTO agent_tenants (tenant_id, name)
            VALUES (%s, %s)
            ON CONFLICT (tenant_id) DO NOTHING
            """,
            (tenant_id, tenant_id),
        )
        connection.execute(
            """
            INSERT INTO agent_users (user_id, tenant_id, email, display_name, metadata)
            VALUES (%s, %s, %s, %s, '{}'::jsonb)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (user_id, tenant_id, email, "Profile Smoke User"),
        )
        connection.commit()


def _load_runtime_env() -> None:
    root = Path(__file__).resolve().parents[1]
    for env_path in (root / "runtime.env", root / "kyuriagents" / "runtime" / "runtime.env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    main()
