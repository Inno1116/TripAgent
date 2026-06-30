"""Types and validation helpers for structured traveler profiles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ProfileSection: TypeAlias = Literal["hard_constraints", "dynamic_preferences", "trip_state", "history_facts"]
ProfileOperation: TypeAlias = Literal["set", "append", "remove"]
ProfileScope: TypeAlias = Literal["long_term", "current_trip"]

_PROFILE_KEYS = ("hard_constraints", "dynamic_preferences", "trip_state", "history_facts")
_DEFAULT_PROFILE: dict[str, JsonValue] = {
    "hard_constraints": {},
    "dynamic_preferences": {},
    "trip_state": {},
    "history_facts": [],
}
_MAX_PROFILE_JSON_CHARS = 12_000


@dataclass(frozen=True, kw_only=True)
class TravelProfileRecord:
    """Persisted structured traveler profile for one user."""

    tenant_id: str
    user_id: str
    profile_data: dict[str, JsonValue]
    profile_version: int = 1
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, kw_only=True)
class ProfileCandidate:
    """One explicit user-sourced traveler profile change candidate."""

    section: ProfileSection
    field: str
    operation: ProfileOperation
    value: JsonValue = None
    scope: ProfileScope = "long_term"
    source_text: str = ""


def default_profile_data() -> dict[str, JsonValue]:
    """Return an empty traveler profile object."""
    return {key: _copy_json_value(value) for key, value in _DEFAULT_PROFILE.items()}


def normalize_profile_data(value: Mapping[str, object] | None) -> dict[str, JsonValue]:
    """Validate and normalize a traveler profile JSON object.

    Args:
        value: Raw profile data from a tool call or database row.

    Returns:
        Normalized profile data with stable top-level sections.

    Raises:
        ValueError: If the object is too large or has unsupported top-level keys.
    """
    raw = dict(value or {})
    unknown = sorted(key for key in raw if key not in _PROFILE_KEYS)
    if unknown:
        msg = f"Unsupported traveler profile section(s): {', '.join(unknown)}."
        raise ValueError(msg)
    profile = default_profile_data()
    for key in _PROFILE_KEYS:
        if key in raw:
            profile[key] = _json_value(raw[key])
    text = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    if len(text) > _MAX_PROFILE_JSON_CHARS:
        msg = f"Traveler profile is too large ({len(text)} chars, max {_MAX_PROFILE_JSON_CHARS})."
        raise ValueError(msg)
    return profile


def normalize_profile_candidate(value: Mapping[str, object]) -> ProfileCandidate:
    """Validate one structured traveler profile change candidate."""
    section = str(value.get("section") or "").strip()
    if section not in _PROFILE_KEYS:
        msg = f"Unsupported traveler profile section: {section or '(empty)'}."
        raise ValueError(msg)
    operation = str(value.get("operation") or "set").strip().lower()
    if operation not in {"set", "append", "remove"}:
        msg = f"Unsupported traveler profile operation: {operation or '(empty)'}."
        raise ValueError(msg)
    scope = str(value.get("scope") or ("current_trip" if section == "trip_state" else "long_term")).strip().lower()
    if scope not in {"long_term", "current_trip"}:
        msg = f"Unsupported traveler profile scope: {scope or '(empty)'}."
        raise ValueError(msg)
    expected_scope = "current_trip" if section == "trip_state" else "long_term"
    if scope != expected_scope:
        msg = f"Traveler profile section `{section}` requires scope `{expected_scope}`."
        raise ValueError(msg)
    field = str(value.get("field") or "").strip()
    if not field:
        msg = "Traveler profile candidate field must not be empty."
        raise ValueError(msg)
    source_text = str(value.get("source_text") or "").strip()
    if not source_text:
        msg = "Traveler profile candidate source_text must not be empty."
        raise ValueError(msg)
    return ProfileCandidate(
        section=cast("ProfileSection", section),
        field=field,
        operation=cast("ProfileOperation", operation),
        value=_json_value(value.get("value")),
        scope=cast("ProfileScope", scope),
        source_text=source_text,
    )


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return cast("JsonValue", value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    return str(value)


def _copy_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value
