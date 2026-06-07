"""LangChain tool adapters for web search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from kyuriagents.tools import ToolDescriptor
from kyuriagents.websearch.service import (
    WebSearchService,
    blocked_query_reason,
    format_fetched_page,
    format_web_research,
    format_web_search_results,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

    from kyuriagents.runtime import AgentRuntimeConfig


def create_web_search_tools(config: AgentRuntimeConfig) -> list[BaseTool]:
    """Create compatibility runtime web tools.

    Args:
        config: Runtime configuration.

    Returns:
        LangChain tools for SearXNG search and page reading.
    """
    service = WebSearchService(config)
    search_tool = _create_web_search_tool(service)
    fetch_static_tool = _create_web_fetch_static_tool(service)
    render_tool = _create_web_render_page_tool(service)

    def web_research(query: str, max_results: int | None = None, max_pages: int | None = None) -> str:
        """Search the public web and read several relevant pages with citations."""
        reason = blocked_query_reason(query)
        if reason:
            return reason
        try:
            result = service.research(query, max_results=max_results, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001  # Tool output should be a friendly recoverable failure.
            return f"Web research failed: {exc}"
        return format_web_research(result)

    def web_fetch_page(url: str) -> str:
        """Read one public HTTP(S) page and return a short sourced excerpt."""
        try:
            page = service.fetch_url(url)
        except Exception as exc:  # noqa: BLE001  # Tool output should be a friendly recoverable failure.
            return f"Web page fetch failed: {exc}"
        return format_fetched_page(page)

    return [
        search_tool,
        StructuredTool.from_function(
            web_research,
            name="web_research",
            description=(
                "Search the public web, then concurrently read several top pages. Use this when the answer needs "
                "evidence from multiple web pages. Returns sourced excerpts and per-page fetch errors."
            ),
        ),
        StructuredTool.from_function(
            web_fetch_page,
            name="web_fetch_page",
            description=(
                "Fetch and read a specific public HTTP(S) URL. Uses a browser render fallback when simple HTTP "
                "extraction is insufficient. Returns a concise excerpt with the source URL."
            ),
        ),
        fetch_static_tool,
        render_tool,
    ]


def create_web_agent_tools(config: AgentRuntimeConfig) -> list[BaseTool]:
    """Create focused web tools for the web research subagent.

    Args:
        config: Runtime configuration.

    Returns:
        Search, static fetch, and browser-render tools.
    """
    service = WebSearchService(config)
    return [
        _create_web_search_tool(service),
        _create_web_fetch_static_tool(service),
        _create_web_render_page_tool(service),
    ]


def _create_web_search_tool(service: WebSearchService) -> BaseTool:
    def web_search(query: str, max_results: int | None = None) -> str:
        """Search the public web through SearXNG and return sourced snippets."""
        reason = blocked_query_reason(query)
        if reason:
            return reason
        try:
            results = service.search_with_diagnostics(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001  # Tool output should be a friendly recoverable failure.
            return f"Web search failed: {exc}"
        return format_web_search_results(results, query=query)

    return StructuredTool.from_function(
        web_search,
        name="web_search",
        description=(
            "Search the public web through SearXNG. Use this for current events, public websites, "
            "official documentation, or when local knowledge is insufficient. Returns titles, URLs, and snippets."
        ),
    )


def _create_web_fetch_static_tool(service: WebSearchService) -> BaseTool:
    def web_fetch_static(url: str) -> str:
        """Fetch one public HTTP(S) URL with plain HTTP extraction only."""
        try:
            page = service.fetch_static_url(url)
        except Exception as exc:  # noqa: BLE001  # Tool output should be a friendly recoverable failure.
            return f"Static web page fetch failed: {exc}"
        return format_fetched_page(page)

    return StructuredTool.from_function(
        web_fetch_static,
        name="web_fetch_static",
        description=(
            "Fetch and extract text from a specific public HTTP(S) URL with plain HTTP only. "
            "Use this before browser rendering. Returns a concise excerpt with the source URL."
        ),
    )


def _create_web_render_page_tool(service: WebSearchService) -> BaseTool:
    def web_render_page(url: str) -> str:
        """Render one public HTTP(S) URL with Playwright and extract text."""
        try:
            page = service.render_url(url)
        except Exception as exc:  # noqa: BLE001  # Tool output should be a friendly recoverable failure.
            return f"Rendered web page fetch failed: {exc}"
        return format_fetched_page(page)

    return StructuredTool.from_function(
        web_render_page,
        name="web_render_page",
        description=(
            "Render a specific public HTTP(S) URL with Playwright and extract visible text. "
            "Use only when static fetching is empty, blocked, or clearly misses dynamic content."
        ),
    )


def web_search_tool_descriptors(*, timeout_seconds: int | None = None) -> Sequence[ToolDescriptor]:
    """Return governance metadata for web search tools."""
    return (
        ToolDescriptor(
            name="web_search",
            description="Public web search through SearXNG.",
            risk="external_read",
            source="runtime",
            timeout_seconds=timeout_seconds,
            tags=("web", "search", "searxng"),
        ),
        ToolDescriptor(
            name="web_research",
            description="Public web search plus bounded concurrent page reading.",
            risk="external_read",
            source="runtime",
            timeout_seconds=timeout_seconds,
            tags=("web", "search", "browser"),
        ),
        ToolDescriptor(
            name="web_fetch_page",
            description="Read one public web page.",
            risk="external_read",
            source="runtime",
            timeout_seconds=timeout_seconds,
            tags=("web", "browser"),
        ),
        ToolDescriptor(
            name="web_fetch_static",
            description="Read one public web page using plain HTTP extraction only.",
            risk="external_read",
            source="runtime",
            timeout_seconds=timeout_seconds,
            tags=("web", "browser", "static"),
        ),
        ToolDescriptor(
            name="web_render_page",
            description="Render one public web page with Playwright.",
            risk="external_read",
            source="runtime",
            timeout_seconds=timeout_seconds,
            tags=("web", "browser", "render"),
        ),
    )


__all__ = ["create_web_agent_tools", "create_web_search_tools", "web_search_tool_descriptors"]
