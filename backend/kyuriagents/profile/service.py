"""Service layer for structured traveler profiles."""

from __future__ import annotations

import json
from typing import Protocol

from kyuriagents.profile.types import TravelProfileRecord, default_profile_data, normalize_profile_data


class TravelProfileStore(Protocol):
    """Durable store contract for traveler profiles."""

    def get(self, *, tenant_id: str, user_id: str) -> TravelProfileRecord | None:
        """Load a profile."""
        ...

    def upsert(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_data: dict[str, object],
        expected_version: int | None = None,
    ) -> TravelProfileRecord:
        """Create or replace a profile."""
        ...


class TravelProfileService:
    """Small orchestration layer around structured traveler profile storage."""

    def __init__(self, store: TravelProfileStore) -> None:
        """Initialize the service."""
        self._store = store

    def get_profile(self, *, tenant_id: str, user_id: str) -> TravelProfileRecord:
        """Return the user profile, falling back to an empty profile object."""
        record = self._store.get(tenant_id=tenant_id, user_id=user_id)
        if record is not None:
            return record
        return TravelProfileRecord(
            tenant_id=tenant_id,
            user_id=user_id,
            profile_data=default_profile_data(),
            profile_version=0,
        )

    def update_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
        profile_data: dict[str, object],
        expected_version: int | None = None,
    ) -> TravelProfileRecord:
        """Replace the structured traveler profile after validation."""
        return self._store.upsert(
            tenant_id=tenant_id,
            user_id=user_id,
            profile_data=normalize_profile_data(profile_data),
            expected_version=expected_version,
        )


def format_travel_profile_context(record: TravelProfileRecord, *, max_chars: int = 4_000) -> str:
    """Format a traveler profile for system-prompt injection."""
    profile = normalize_profile_data(record.profile_data)
    if _is_empty_profile(profile):
        return ""
    payload = {
        "profile_version": record.profile_version,
        "profile_data": profile,
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(body) > max_chars:
        suffix = "\n...[truncated]"
        body = body[: max(0, max_chars - len(suffix))].rstrip() + suffix
    return (
        "<current_traveler_profile>\n"
        "以下结构化游客画像是旅行规划中的权威长期记忆。请优先遵守 hard_constraints，"
        "在相关场景中参考 dynamic_preferences，并仅在用户明确表达稳定偏好、约束或旅行状态时更新画像。\n"
        f"{body}\n"
        "</current_traveler_profile>"
    )


def _is_empty_profile(profile: dict[str, object]) -> bool:
    return not any(bool(profile.get(key)) for key in ("hard_constraints", "dynamic_preferences", "trip_state", "history_facts"))
