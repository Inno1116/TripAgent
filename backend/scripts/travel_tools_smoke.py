"""Smoke-test KyuriAgents travel tools without starting the API server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kyuriagents.runtime.config import AgentRuntimeConfig
from kyuriagents.travel import AmapTravelService, estimate_budget, format_travel_payload


def main() -> None:
    """Run one travel tool smoke test."""
    _load_runtime_env()
    parser = argparse.ArgumentParser(description="Smoke-test AMap travel tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    poi = subparsers.add_parser("poi", help="Search AMap POIs.")
    poi.add_argument("--city", default="北京")
    poi.add_argument("--keywords", default="故宫")
    poi.add_argument("--max-results", type=int, default=5)

    weather = subparsers.add_parser("weather", help="Fetch city weather.")
    weather.add_argument("--city", default="北京")

    route = subparsers.add_parser("route", help="Plan a route.")
    route.add_argument("--origin", default="天安门")
    route.add_argument("--destination", default="故宫博物院")
    route.add_argument("--city", default="北京")
    route.add_argument("--mode", default="walking", choices=("walking", "transit", "driving", "bicycling"))

    budget = subparsers.add_parser("budget", help="Estimate a travel budget without network access.")
    budget.add_argument("--city", default="北京")
    budget.add_argument("--days", type=int, default=4)
    budget.add_argument("--travelers", type=int, default=1)
    budget.add_argument("--level", default="medium")

    args = parser.parse_args()
    config = AgentRuntimeConfig.from_env()
    service = AmapTravelService(config)
    if args.command == "poi":
        print(format_travel_payload(service.search_poi(city=args.city, keywords=args.keywords, max_results=args.max_results)))
    elif args.command == "weather":
        print(format_travel_payload(service.get_weather(city=args.city)))
    elif args.command == "route":
        print(format_travel_payload(service.plan_route(origin=args.origin, destination=args.destination, city=args.city, mode=args.mode)))
    elif args.command == "budget":
        print(format_travel_payload(estimate_budget(city=args.city, days=args.days, travelers=args.travelers, budget_level=args.level)))


def _load_runtime_env() -> None:
    root = Path(__file__).resolve().parents[1]
    for env_path in (root / "runtime.env", root / "kyuriagents" / "runtime" / "runtime.env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip())


if __name__ == "__main__":
    main()
