"""AMap MCP backed travel services and budget estimation helpers."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from kyuriagents.runtime.time_context import current_time_context

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from kyuriagents.runtime import AgentRuntimeConfig

_AMAP_MCP_URL = "https://mcp.amap.com/mcp?key={api_key}"
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class TravelToolError(RuntimeError):
    """Raised when a travel tool cannot be executed."""


@dataclass(frozen=True, kw_only=True)
class BudgetProfile:
    """Simple per-person budget defaults."""

    hotel_per_night: int
    meal_per_day: int
    local_transport_per_day: int
    ticket_per_day: int


_BUDGET_PROFILES: dict[str, BudgetProfile] = {
    "economy": BudgetProfile(hotel_per_night=280, meal_per_day=100, local_transport_per_day=45, ticket_per_day=80),
    "medium": BudgetProfile(hotel_per_night=520, meal_per_day=180, local_transport_per_day=70, ticket_per_day=120),
    "comfortable": BudgetProfile(hotel_per_night=900, meal_per_day=320, local_transport_per_day=120, ticket_per_day=180),
    "luxury": BudgetProfile(hotel_per_night=1600, meal_per_day=600, local_transport_per_day=220, ticket_per_day=300),
}


class AmapTravelService:
    """Small wrapper around official AMap MCP tools.

    The wrapper keeps planner-facing tool output stable even when the upstream
    MCP server returns provider-specific JSON or text.
    """

    def __init__(self, config: AgentRuntimeConfig) -> None:
        """Initialize the service."""
        self._config = config
        self._client: object | None = None
        self._tools: dict[str, BaseTool] | None = None

    def search_poi(self, *, city: str, keywords: str, citylimit: bool = True, max_results: int = 10) -> dict[str, object]:
        """Search points of interest in a city."""
        payload: dict[str, object] = {"keywords": keywords, "city": city, "citylimit": citylimit}
        result = self._invoke(("maps_text_search", "amap_maps_text_search"), payload)
        return {
            "tool": "amap_search_poi",
            "query": {"city": city, "keywords": keywords, "citylimit": citylimit, "max_results": max_results},
            "runtime": current_time_context(),
            "items": _limit_items(_extract_items(result, keys=("pois", "data", "results")), max_results),
            "raw": _compact_raw(result),
        }

    def get_weather(self, *, city: str) -> dict[str, object]:
        """Fetch weather forecast for a city."""
        result = self._invoke(("maps_weather", "amap_maps_weather"), {"city": city})
        return {
            "tool": "amap_get_weather",
            "query": {"city": city},
            "runtime": current_time_context(),
            "forecasts": _extract_items(result, keys=("forecasts", "lives", "data", "results")),
            "raw": _compact_raw(result),
        }

    def get_poi_detail(self, *, poi_id: str) -> dict[str, object]:
        """Fetch detail for a point of interest."""
        result = self._invoke(("maps_search_detail", "amap_maps_search_detail"), {"id": poi_id})
        return {
            "tool": "amap_get_poi_detail",
            "query": {"poi_id": poi_id},
            "runtime": current_time_context(),
            "detail": _extract_mapping(result),
            "raw": _compact_raw(result),
        }

    def plan_route(
        self,
        *,
        origin: str,
        destination: str,
        mode: str = "transit",
        origin_city: str = "",
        destination_city: str = "",
        city: str = "",
    ) -> dict[str, object]:
        """Plan a route between two places.

        Address-based tools are tried first. If the MCP server only exposes
        coordinate-based tools, the wrapper geocodes both endpoints and retries.
        """
        normalized_mode = _route_mode(mode)
        address_payload: dict[str, object] = {"origin_address": origin, "destination_address": destination}
        if origin_city:
            address_payload["origin_city"] = origin_city
        if destination_city:
            address_payload["destination_city"] = destination_city
        if city:
            address_payload.setdefault("origin_city", city)
            address_payload.setdefault("destination_city", city)
        try:
            result = self._invoke(_route_aliases(normalized_mode, by_address=True), address_payload)
        except TravelToolError:
            origin_location = self._geocode(origin, city=origin_city or city)
            destination_location = self._geocode(destination, city=destination_city or city)
            coordinate_payload = {"origin": origin_location, "destination": destination_location}
            if city:
                coordinate_payload["city"] = city
            result = self._invoke(_route_aliases(normalized_mode, by_address=False), coordinate_payload)
        return {
            "tool": "amap_plan_route",
            "query": {
                "origin": origin,
                "destination": destination,
                "mode": normalized_mode,
                "origin_city": origin_city,
                "destination_city": destination_city,
                "city": city,
            },
            "runtime": current_time_context(),
            "route": _extract_mapping(result),
            "raw": _compact_raw(result),
        }

    def create_trip_map(self, *, trip_name: str, days: Sequence[Mapping[str, object]]) -> dict[str, object]:
        """Create or prepare map visualization data for a trip."""
        payload = {"name": trip_name, "days": list(days)}
        try:
            result = self._invoke(
                (
                    "maps_create_trip_map",
                    "maps_trip_plan",
                    "maps_travel_map",
                    "amap_maps_create_trip_map",
                ),
                payload,
            )
            return {
                "tool": "amap_create_trip_map",
                "trip_name": trip_name,
                "runtime": current_time_context(),
                "map": _extract_mapping(result),
                "markers": _markers_from_days(days),
                "raw": _compact_raw(result),
            }
        except TravelToolError as exc:
            return {
                "tool": "amap_create_trip_map",
                "trip_name": trip_name,
                "runtime": current_time_context(),
                "markers": _markers_from_days(days),
                "missing": ["AMap MCP trip-map tool was unavailable; returned frontend-ready marker data instead."],
                "failures": [str(exc)],
            }

    def _geocode(self, address: str, *, city: str = "") -> str:
        payload: dict[str, object] = {"address": address}
        if city:
            payload["city"] = city
        result = self._invoke(("maps_geo", "amap_maps_geo"), payload)
        location = _first_location(result)
        if not location:
            msg = f"Could not geocode `{address}` through AMap MCP."
            raise TravelToolError(msg)
        return location

    def _invoke(self, aliases: Sequence[str], payload: Mapping[str, object]) -> object:
        tools = self._get_tools()
        tool = _resolve_tool(tools, aliases)
        if tool is None:
            available = ", ".join(sorted(tools))
            wanted = ", ".join(aliases)
            msg = f"AMap MCP tool not found. Wanted one of: {wanted}. Available: {available or '<none>'}."
            raise TravelToolError(msg)
        try:
            result = _invoke_tool(tool, dict(payload))
        except Exception as exc:  # noqa: BLE001
            msg = f"AMap MCP tool `{tool.name}` failed: {exc}"
            raise TravelToolError(msg) from exc
        return _normalize_result(result)

    def _get_tools(self) -> dict[str, BaseTool]:
        if self._tools is not None:
            return self._tools
        api_key = self._config.amap_api_key
        if not api_key:
            msg = "Set AMAP_API_KEY before using AMap travel tools."
            raise TravelToolError(msg)
        try:
            client_module = import_module("langchain_mcp_adapters.client")
        except ImportError as exc:
            msg = "Install `langchain-mcp-adapters` before using AMap MCP travel tools."
            raise TravelToolError(msg) from exc
        client_cls = client_module.MultiServerMCPClient
        endpoint = self._config.amap_mcp_url or _AMAP_MCP_URL.format(api_key=api_key)
        connection = {"transport": "streamable_http", "url": endpoint}
        client = client_cls({"amap": connection})
        self._client = client
        self._tools = {tool.name: tool for tool in _run_async(lambda: client.get_tools())}
        return self._tools


def estimate_budget(
    *,
    days: int,
    travelers: int = 1,
    budget_level: str = "medium",
    city: str = "",
    attraction_count: int = 0,
    route_count: int = 0,
) -> dict[str, object]:
    """Estimate a rough travel budget without real-time ticket or hotel prices."""
    resolved_days = max(1, int(days))
    resolved_travelers = max(1, int(travelers))
    profile_key = _budget_key(budget_level)
    profile = _BUDGET_PROFILES[profile_key]
    nights = max(0, resolved_days - 1)
    city_multiplier = _city_multiplier(city)
    attractions = attraction_count if attraction_count > 0 else resolved_days * 2
    routes = route_count if route_count > 0 else resolved_days * 2
    hotel = round(profile.hotel_per_night * nights * resolved_travelers * city_multiplier)
    meals = round(profile.meal_per_day * resolved_days * resolved_travelers * city_multiplier)
    local_transport = round(profile.local_transport_per_day * resolved_days * resolved_travelers * city_multiplier + routes * 10)
    tickets = round(profile.ticket_per_day * resolved_days * resolved_travelers * city_multiplier + attractions * 20)
    total = hotel + meals + local_transport + tickets
    return {
        "tool": "estimate_travel_budget",
        "runtime": current_time_context(),
        "city": city,
        "budget_level": profile_key,
        "travelers": resolved_travelers,
        "days": resolved_days,
        "nights": nights,
        "currency": "CNY",
        "items": {
            "hotel": hotel,
            "meals": meals,
            "local_transport": local_transport,
            "tickets_and_activities": tickets,
        },
        "total": total,
        "note": "This is a rule-based estimate for planning only, not live ticket, hotel, flight, or rail pricing.",
    }


def format_travel_payload(payload: Mapping[str, object]) -> str:
    """Format travel tool output for LLM consumption."""
    return "<travel_tool_result>\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n</travel_tool_result>"


def _invoke_tool(tool: BaseTool, payload: Mapping[str, object]) -> object:
    """Invoke MCP tools that may be async-only."""
    async_invoke = getattr(tool, "ainvoke", None)
    if callable(async_invoke):
        return _run_async(lambda: async_invoke(dict(payload)))
    return tool.invoke(dict(payload))


def _run_async(factory: object) -> Any:
    """Run an async factory from sync code, even inside an active event loop."""
    async_factory = cast("Any", factory)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_factory())
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(async_factory())
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=runner, name="kyuriagents-amap-async", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise cast("BaseException", result["error"])
    return result.get("value")


def _resolve_tool(tools: Mapping[str, BaseTool], aliases: Sequence[str]) -> BaseTool | None:
    alias_set = {alias.lower() for alias in aliases}
    for name, tool in tools.items():
        lower_name = name.lower()
        if lower_name in alias_set or any(lower_name.endswith(f"_{alias}") for alias in alias_set):
            return tool
    return None


def _normalize_result(result: object) -> object:
    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    if isinstance(content, Mapping | list):
        return content
    text = str(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match is not None:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return text


def _extract_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(cast("Mapping[str, object]", value))
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return {"items": [dict(cast("Mapping[str, object]", item)) for item in value]}
    return {}


def _extract_items(value: object, *, keys: Sequence[str]) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if not isinstance(value, Mapping):
        return []
    mapping = cast("Mapping[str, object]", value)
    for key in keys:
        item = mapping.get(key)
        if isinstance(item, list):
            return item
        if isinstance(item, Mapping):
            nested = _extract_items(item, keys=keys)
            if nested:
                return nested
    return []


def _limit_items(items: Sequence[object], limit: int) -> list[object]:
    return list(items[: max(1, limit)])


def _compact_raw(value: object, *, max_chars: int = 2_000) -> object:
    if isinstance(value, Mapping | list):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    if len(text) <= max_chars:
        return value
    return f"{text[:max_chars]}...[truncated]"


def _route_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"walk", "walking", "步行"}:
        return "walking"
    if normalized in {"drive", "driving", "car", "自驾", "驾车"}:
        return "driving"
    if normalized in {"bike", "bicycling", "cycling", "骑行"}:
        return "bicycling"
    return "transit"


def _route_aliases(mode: str, *, by_address: bool) -> tuple[str, ...]:
    suffix = "_by_address" if by_address else ""
    if mode == "walking":
        return (f"maps_direction_walking{suffix}", f"amap_maps_direction_walking{suffix}")
    if mode == "driving":
        return (f"maps_direction_driving{suffix}", f"amap_maps_direction_driving{suffix}")
    if mode == "bicycling":
        return (f"maps_direction_bicycling{suffix}", f"amap_maps_direction_bicycling{suffix}")
    return (
        f"maps_direction_transit_integrated{suffix}",
        f"amap_maps_direction_transit_integrated{suffix}",
    )


def _first_location(value: object) -> str:
    items = _extract_items(value, keys=("geocodes", "pois", "data", "results"))
    for item in items:
        if isinstance(item, Mapping):
            raw = item.get("location")
            if raw:
                return str(raw)
    if isinstance(value, Mapping):
        raw = value.get("location")
        if raw:
            return str(raw)
    return ""


def _markers_from_days(days: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    for day_index, day in enumerate(days, start=1):
        raw_points = day.get("points") or day.get("attractions") or []
        if not isinstance(raw_points, list):
            continue
        for point_index, point in enumerate(raw_points, start=1):
            if not isinstance(point, Mapping):
                continue
            marker = {
                "day": day.get("day") or day.get("day_index") or day_index,
                "order": point_index,
                "name": str(point.get("name") or point.get("title") or ""),
                "address": str(point.get("address") or ""),
                "location": point.get("location") or {"lng": point.get("lng"), "lat": point.get("lat")},
            }
            markers.append(marker)
    return markers


def _budget_key(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"经济", "经济型", "省钱", "budget", "low"}:
        return "economy"
    if normalized in {"舒适", "comfortable", "comfort"}:
        return "comfortable"
    if normalized in {"豪华", "luxury", "high"}:
        return "luxury"
    return "medium"


def _city_multiplier(city: str) -> float:
    if any(name in city for name in ("北京", "上海", "深圳", "香港", "澳门", "东京", "大阪", "京都")):
        return 1.25
    if any(name in city for name in ("成都", "重庆", "杭州", "广州", "南京", "西安")):
        return 1.1
    return 1.0


__all__ = ["AmapTravelService", "TravelToolError", "estimate_budget", "format_travel_payload"]
