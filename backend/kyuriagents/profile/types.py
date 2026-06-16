"""Types and validation helpers for structured traveler profiles."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

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
