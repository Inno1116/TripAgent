"""LangChain and task-mode adapters for travel tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from kyuriagents.tools import ToolDescriptor
from kyuriagents.travel.service import AmapTravelService, estimate_budget, format_travel_payload

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from langchain_core.tools import BaseTool

    from kyuriagents.runtime import AgentRuntimeConfig
    from kyuriagents.tasks.runtime import TaskExecutionContext, ToolHandler


def create_travel_tools(config: AgentRuntimeConfig) -> list[BaseTool]:
    """Create LangChain tools for travel planning."""
    service = AmapTravelService(config)

    def amap_search_poi(city: str, keywords: str, citylimit: bool = True, max_results: int = 10) -> str:
        """Search AMap POIs such as attractions, hotels, restaurants, stations, and business areas."""
        return format_travel_payload(service.search_poi(city=city, keywords=keywords, citylimit=citylimit, max_results=max_results))

    def amap_get_weather(city: str) -> str:
        """Get AMap weather forecast for a destination city."""
        return format_travel_payload(service.get_weather(city=city))

    def amap_plan_route(
        origin: str,
        destination: str,
        mode: str = "transit",
        origin_city: str = "",
        destination_city: str = "",
        city: str = "",
    ) -> str:
        """Plan an AMap route between two addresses or places."""
        return format_travel_payload(
            service.plan_route(
                origin=origin,
                destination=destination,
                mode=mode,
                origin_city=origin_city,
                destination_city=destination_city,
                city=city,
            )
        )

    def amap_get_poi_detail(poi_id: str) -> str:
        """Get detail for one AMap POI id."""
        return format_travel_payload(service.get_poi_detail(poi_id=poi_id))

    def amap_create_trip_map(trip_name: str, days: list[dict[str, object]]) -> str:
        """Create map-ready marker data or an AMap trip-map result from planned days."""
        return format_travel_payload(service.create_trip_map(trip_name=trip_name, days=days))

    def estimate_travel_budget(
        days: int,
        travelers: int = 1,
        budget_level: str = "medium",
        city: str = "",
        attraction_count: int = 0,
        route_count: int = 0,
    ) -> str:
        """Estimate rough travel cost for hotels, meals, local transport, and tickets."""
        return format_travel_payload(
            estimate_budget(
                days=days,
                travelers=travelers,
                budget_level=budget_level,
                city=city,
                attraction_count=attraction_count,
                route_count=route_count,
            )
        )

    return [
        StructuredTool.from_function(amap_search_poi, name="amap_search_poi", description=travel_tool_description("amap_search_poi")),
        StructuredTool.from_function(amap_get_weather, name="amap_get_weather", description=travel_tool_description("amap_get_weather")),
        StructuredTool.from_function(amap_plan_route, name="amap_plan_route", description=travel_tool_description("amap_plan_route")),
        StructuredTool.from_function(amap_get_poi_detail, name="amap_get_poi_detail", description=travel_tool_description("amap_get_poi_detail")),
        StructuredTool.from_function(amap_create_trip_map, name="amap_create_trip_map", description=travel_tool_description("amap_create_trip_map")),
        StructuredTool.from_function(estimate_travel_budget, name="estimate_travel_budget", description=travel_tool_description("estimate_travel_budget")),
    ]


def create_travel_tool_handlers(config: AgentRuntimeConfig) -> dict[str, ToolHandler]:
    """Create task-mode handlers for travel tools."""
    service = AmapTravelService(config)

    def amap_search_poi(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        del context
        return format_travel_payload(
            service.search_poi(
                city=_str_input(input_data.get("city")),
                keywords=_str_input(input_data.get("keywords") or input_data.get("query")),
                citylimit=_bool_input(input_data.get("citylimit"), default=True),
                max_results=_int_input(input_data.get("max_results"), default=10),
            )
        )

    def amap_get_weather(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        return format_travel_payload(service.get_weather(city=_str_input(input_data.get("city") or context.goal)))

    def amap_plan_route(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        del context
        return format_travel_payload(
            service.plan_route(
                origin=_str_input(input_data.get("origin") or input_data.get("from")),
                destination=_str_input(input_data.get("destination") or input_data.get("to")),
                mode=_str_input(input_data.get("mode"), default="transit"),
                origin_city=_str_input(input_data.get("origin_city")),
                destination_city=_str_input(input_data.get("destination_city")),
                city=_str_input(input_data.get("city")),
            )
        )

    def amap_get_poi_detail(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        del context
        return format_travel_payload(service.get_poi_detail(poi_id=_str_input(input_data.get("poi_id") or input_data.get("id"))))

    def amap_create_trip_map(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        del context
        return format_travel_payload(
            service.create_trip_map(
                trip_name=_str_input(input_data.get("trip_name") or input_data.get("name"), default="Trip"),
                days=_days_input(input_data.get("days")),
            )
        )

    def estimate_travel_budget(context: TaskExecutionContext, input_data: Mapping[str, object]) -> str:
        del context
        return format_travel_payload(
            estimate_budget(
                days=_int_input(input_data.get("days"), default=3),
                travelers=_int_input(input_data.get("travelers"), default=1),
                budget_level=_str_input(input_data.get("budget_level") or input_data.get("level"), default="medium"),
                city=_str_input(input_data.get("city")),
                attraction_count=_int_input(input_data.get("attraction_count"), default=0),
                route_count=_int_input(input_data.get("route_count"), default=0),
            )
        )

    return {
        "amap_search_poi": amap_search_poi,
        "amap_get_weather": amap_get_weather,
        "amap_plan_route": amap_plan_route,
        "amap_get_poi_detail": amap_get_poi_detail,
        "amap_create_trip_map": amap_create_trip_map,
        "estimate_travel_budget": estimate_travel_budget,
    }


def travel_tool_descriptors(*, timeout_seconds: int | None = None) -> Sequence[ToolDescriptor]:
    """Return governance descriptors for travel tools."""
    return tuple(
        ToolDescriptor(
            name=name,
            description=travel_tool_description(name),
            risk="external_read" if name.startswith("amap_") else "read_only",
            source="runtime",
            timeout_seconds=timeout_seconds,
            tags=("travel", "amap") if name.startswith("amap_") else ("travel", "budget"),
        )
        for name in (
            "amap_search_poi",
            "amap_get_weather",
            "amap_plan_route",
            "amap_get_poi_detail",
            "amap_create_trip_map",
            "estimate_travel_budget",
        )
    )


def travel_tool_description(name: str) -> str:
    """Return a planner-facing tool description."""
    descriptions = {
        "amap_search_poi": (
            "Search AMap points of interest for travel planning. Use for attractions, restaurants, hotels, stations, "
            "shopping areas, and anime/location-themed spots. Required input: city, keywords."
        ),
        "amap_get_weather": "Get AMap weather forecast for a destination city. Use before arranging outdoor-heavy itineraries.",
        "amap_plan_route": (
            "Plan route, distance, and duration between two places with AMap. Use for ordering attractions and estimating "
            "city transport time. Inputs: origin, destination, mode, and optional city fields."
        ),
        "amap_get_poi_detail": "Fetch detail for one AMap POI id returned by amap_search_poi.",
        "amap_create_trip_map": "Create map-ready markers or an AMap trip-map result from finalized itinerary days.",
        "estimate_travel_budget": "Estimate rough travel budget from days, travelers, city, budget level, attractions, and routes.",
    }
    return descriptions.get(name, "Travel planning tool.")


def _str_input(value: object, *, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _int_input(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return max(0, value)
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return default


def _bool_input(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _days_input(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


__all__ = ["create_travel_tool_handlers", "create_travel_tools", "travel_tool_descriptors"]
