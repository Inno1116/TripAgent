"""Runtime date and time context helpers."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DEFAULT_TIMEZONE = "Asia/Hong_Kong"


def current_time_context(*, timezone_name: str | None = None) -> dict[str, str]:
    """Return the current runtime date/time as prompt-safe metadata."""
    resolved_timezone = timezone_name or os.getenv("KYURI_RUNTIME_TIMEZONE") or os.getenv("DEEPAGENTS_RUNTIME_TIMEZONE") or _DEFAULT_TIMEZONE
    tz = _timezone(resolved_timezone)
    now = datetime.now(tz)
    return {
        "current_date": now.date().isoformat(),
        "current_datetime": now.isoformat(timespec="seconds"),
        "current_year": str(now.year),
        "timezone": resolved_timezone,
        "weekday": now.strftime("%A"),
    }


def format_time_context_block(*, timezone_name: str | None = None) -> str:
    """Format current runtime date/time instructions for model prompts."""
    context = current_time_context(timezone_name=timezone_name)
    return (
        "Runtime date context:\n"
        f"- Current date: {context['current_date']}\n"
        f"- Current datetime: {context['current_datetime']}\n"
        f"- Timezone: {context['timezone']}\n"
        f"- Weekday: {context['weekday']}\n"
        "Treat relative dates such as today, tomorrow, this week, this month, holidays, and upcoming seasons "
        "relative to this runtime date. Do not infer the current date from model training data. "
        "For travel and weather questions, use the user's supplied travel dates when available; if dates are missing, "
        "ask briefly or state that weather can only be checked for currently available forecast windows."
    )


def _timezone(name: str) -> timezone | ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        if name in {"Asia/Hong_Kong", "Asia/Shanghai", "UTC+08:00", "+08:00"}:
            return timezone(timedelta(hours=8), name=name)
        return timezone.utc
