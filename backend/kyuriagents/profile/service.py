"""Service layer for structured traveler profiles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from kyuriagents.profile.types import JsonValue, ProfileCandidate, TravelProfileRecord, default_profile_data, normalize_profile_data


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

    def apply_candidates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        candidates: Sequence[ProfileCandidate],
    ) -> TravelProfileRecord | None:
        """Merge explicit profile candidates into the latest stored profile."""
        if not candidates:
            return None
        current = self.get_profile(tenant_id=tenant_id, user_id=user_id)
        merged = merge_profile_candidates(current.profile_data, candidates)
        if merged == normalize_profile_data(current.profile_data):
            return None
        return self.update_profile(
            tenant_id=tenant_id,
            user_id=user_id,
            profile_data=merged,
            expected_version=current.profile_version if current.profile_version > 0 else None,
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


def merge_profile_candidates(profile_data: dict[str, object], candidates: Sequence[ProfileCandidate]) -> dict[str, JsonValue]:
    """Apply field-level candidates while preserving unrelated profile data."""
    profile = normalize_profile_data(profile_data)
    for candidate in candidates:
        if candidate.section == "history_facts":
            _merge_history_facts(profile, candidate)
            continue
        section = profile.get(candidate.section)
        values = dict(section) if isinstance(section, dict) else {}
        if candidate.operation == "set":
            values[candidate.field] = candidate.value
        elif candidate.operation == "append":
            values[candidate.field] = _append_value(values.get(candidate.field), candidate.value)
        else:
            _remove_value(values, candidate.field, candidate.value)
        profile[candidate.section] = values
    return normalize_profile_data(profile)


def _merge_history_facts(profile: dict[str, JsonValue], candidate: ProfileCandidate) -> None:
    existing = profile.get("history_facts")
    facts = list(existing) if isinstance(existing, list) else []
    if candidate.operation == "set":
        profile["history_facts"] = _as_unique_list(candidate.value)
        return
    if candidate.operation == "append":
        profile["history_facts"] = _append_value(facts, candidate.value)
        return
    removals = _as_unique_list(candidate.value)
    profile["history_facts"] = [fact for fact in facts if fact not in removals]


def _append_value(existing: JsonValue, value: JsonValue) -> list[JsonValue]:
    values = list(existing) if isinstance(existing, list) else ([] if existing is None else [existing])
    for item in _as_unique_list(value):
        if item not in values:
            values.append(item)
    return values


def _remove_value(values: dict[str, JsonValue], field: str, value: JsonValue) -> None:
    if value is None:
        values.pop(field, None)
        return
    existing = values.get(field)
    if isinstance(existing, list):
        removals = _as_unique_list(value)
        remaining = [item for item in existing if item not in removals]
        if remaining:
            values[field] = remaining
        else:
            values.pop(field, None)
    elif existing == value:
        values.pop(field, None)


def _as_unique_list(value: JsonValue) -> list[JsonValue]:
    raw = value if isinstance(value, list) else ([] if value is None else [value])
    result: list[JsonValue] = []
    for item in raw:
        if item not in result:
            result.append(item)
    return result
