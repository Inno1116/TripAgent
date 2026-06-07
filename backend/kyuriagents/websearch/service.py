"""SearXNG-backed web search and page reading helpers."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

if TYPE_CHECKING:
    from kyuriagents.runtime import AgentRuntimeConfig

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 Kyuriagents/1.0"
_TEXT_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)
_SKIP_TAGS = {
    "aside",
    "button",
    "canvas",
    "footer",
    "form",
    "header",
    "iframe",
    "nav",
    "noscript",
    "script",
    "select",
    "style",
    "svg",
    "template",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_MIN_TEXT_CHARS_BEFORE_RENDER = 500
_DEFAULT_PAGE_EXCERPT_CHARS = 1_800
_DEFAULT_SEARCH_SNIPPET_CHARS = 500
_MIN_CJK_QUERY_CHARS_FOR_LOOKUP_GUARD = 2
_MIN_SINGLE_CHAR_LOOKUP_RESULTS = 3
_MAX_QUERY_PLAN_SIZE = 5
_CACHE_VERSION = "v3"
_PAGE_BLOCKED_MARKERS = (
    "403 forbidden",
    "access denied",
    "captcha",
    "cloudflare",
    "verify you are human",
    "\u9a8c\u8bc1\u7801",
    "\u8bbf\u95ee\u53d7\u9650",
    "\u8bf7\u767b\u5f55",
)
_PAGE_JS_REQUIRED_MARKERS = (
    "enable javascript",
    "requires javascript",
    "javascript is disabled",
    "please enable js",
    "\u8bf7\u542f\u7528javascript",
    "\u8bf7\u5f00\u542fjavascript",
    "\u6b63\u5728\u52a0\u8f7d",
)
_CHARSET_RE = re.compile(br"charset\s*=\s*['\"]?([a-zA-Z0-9._-]+)", re.IGNORECASE)
_LOW_CONFIDENCE_ENCODINGS = {"ascii", "iso-8859-1", "latin-1"}
_TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "spm",
    "from",
    "source",
    "fbclid",
    "gclid",
    "wfr",
    "for",
}
_LOW_VALUE_DOMAINS = (
    "global.bing.com/dict",
    "iciba.com",
    "hanyuguoxue.com",
    "hancibao.com",
    "zdic.net",
)
_LOW_VALUE_TITLE_MARKERS = (
    "字典",
    "词典",
    "拼音",
    "笔顺",
    "部首",
    "dictionary",
)
_AUTHORITY_DOMAIN_MARKERS = (
    ".gov.cn",
    ".edu.cn",
    "mfa.gov.cn",
    "www.gov.cn",
    "people.com.cn",
    "xinhuanet.com",
    "news.cn",
    "reuters.com",
    "apnews.com",
    "bbc.com",
)
_ENGINE_SHORTCUTS = {
    "baidu": "bd",
    "bing": "bi",
    "duckduckgo": "ddg",
    "sogou": "sg",
}
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_SINGLE_CHAR_LOOKUP_MARKERS = (
    "字典",
    "拼音",
    "笔顺",
    "部首",
    "汉字",
    "漢字",
    "解释",
    "解釋",
    "意思",
    "新华",
    "漢典",
)
_PIRACY_QUERY_RE = re.compile(
    "(?i)(盗版|破解版|破解资源|网盘资源|资源链接|下载链接|夸克网盘资源|百度网盘资源|magnet:?|torrent|leaked\\s+(?:key|token|credential|download))"
)


@dataclass(frozen=True, kw_only=True)
class WebSearchResult:
    """One SearXNG search result."""

    title: str
    url: str
    snippet: str = ""
    score: float | None = None
    source: str = "searxng"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class WebSearchResponse:
    """Search results with query-planning and quality diagnostics."""

    query: str
    planned_queries: tuple[str, ...] = ()
    results: tuple[WebSearchResult, ...] = ()
    raw_result_count: int = 0
    deduped_count: int = 0
    filtered_count: int = 0
    cache_hit: bool = False
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class FetchedPage:
    """Result of fetching or rendering one web page."""

    requested_url: str
    url: str
    title: str = ""
    text: str = ""
    snippet: str = ""
    status: Literal["fetched", "failed", "skipped"] = "failed"
    method: Literal["http", "playwright", "none"] = "none"
    error: str = ""
    content_type: str = ""
    text_chars: int = 0
    returned_chars: int = 0
    truncated: bool = False
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class WebResearchResult:
    """Combined search results and fetched page excerpts."""

    query: str
    search_results: tuple[WebSearchResult, ...] = ()
    pages: tuple[FetchedPage, ...] = ()
    error: str = ""


class WebSearchService:
    """Search SearXNG and read selected public web pages.

    Args:
        config: Runtime configuration with SearXNG and web-fetch limits.
    """

    def __init__(self, config: AgentRuntimeConfig) -> None:
        """Initialize the service."""
        self._config = config
        self._cache = _RedisSearchCache(config)

    def search(self, query: str, *, max_results: int | None = None) -> tuple[WebSearchResult, ...]:
        """Search SearXNG and return normalized results.

        Args:
            query: Search query.
            max_results: Optional result limit.

        Returns:
            Search results, deduplicated by URL.
        """
        return self.search_with_diagnostics(query, max_results=max_results).results

    def search_with_diagnostics(self, query: str, *, max_results: int | None = None) -> WebSearchResponse:
        """Search with query planning, deduplication, reranking, and diagnostics.

        Args:
            query: Search query.
            max_results: Optional result limit.

        Returns:
            Search response containing final results and pipeline diagnostics.
        """
        limit = _positive_limit(max_results, default=self._config.web_search_max_results)
        cached = self._cache.get(query=query, max_results=limit)
        if cached is not None:
            return cached

        planned_queries = _plan_search_queries(query, max_queries=self._config.web_search_query_plan_size)
        raw_count = 0
        failures: list[str] = []
        candidates: list[WebSearchResult] = []
        per_query_limit = max(limit, min(self._config.web_search_rerank_candidates, limit * 2))
        with ThreadPoolExecutor(max_workers=min(len(planned_queries), _MAX_QUERY_PLAN_SIZE), thread_name_prefix="kyuri-web-search") as executor:
            futures = {
                executor.submit(self._searxng_json, planned_query, max_results=per_query_limit): planned_query
                for planned_query in planned_queries
            }
            for future in as_completed(futures):
                planned_query = futures[future]
                try:
                    data = future.result()
                except Exception as exc:  # noqa: BLE001  # One planned query should not abort the whole search.
                    failures.append(f"{planned_query}: {exc}")
                    continue
                raw_results = data.get("results", [])
                if not isinstance(raw_results, Sequence):
                    failures.append(f"{planned_query}: invalid results")
                    continue
                raw_count += len(raw_results)
                candidates.extend(_results_from_raw(raw_results, planned_query=planned_query))

        deduped = _dedupe_results(candidates)
        gated = [_with_quality_metadata(query, result) for result in deduped]
        kept = [result for result in gated if not bool(result.metadata.get("filtered"))]
        filtered_count = len(gated) - len(kept)
        ranked = _rerank_web_results(query, kept, config=self._config, limit=limit)
        response = WebSearchResponse(
            query=query,
            planned_queries=planned_queries,
            results=tuple(ranked),
            raw_result_count=raw_count,
            deduped_count=len(deduped),
            filtered_count=filtered_count,
            failures=tuple(failures),
        )
        self._cache.set(response, max_results=limit)
        return response

    def _legacy_search(self, query: str, *, max_results: int | None = None) -> tuple[WebSearchResult, ...]:
        limit = _positive_limit(max_results, default=self._config.web_search_max_results)
        data = self._searxng_json(query, max_results=limit)
        raw_results = data.get("results", [])
        if not isinstance(raw_results, Sequence):
            return ()
        return tuple(_results_from_raw(raw_results, planned_query=query)[:limit])

    def research(
        self,
        query: str,
        *,
        max_results: int | None = None,
        max_pages: int | None = None,
    ) -> WebResearchResult:
        """Search and read several top pages concurrently.

        Args:
            query: Research query.
            max_results: Search result limit.
            max_pages: Maximum pages to read.

        Returns:
            Search results plus fetched page excerpts.
        """
        results = self.search(query, max_results=max_results)
        page_limit = min(_positive_limit(max_pages, default=self._config.web_fetch_max_pages), len(results))
        pages = self.fetch_results(results[:page_limit])
        return WebResearchResult(query=query, search_results=results, pages=pages)

    def fetch_url(self, url: str) -> FetchedPage:
        """Fetch one URL, using Playwright if HTTP extraction is not enough.

        Args:
            url: Public HTTP(S) URL.

        Returns:
            Fetched page or structured failure.
        """
        result = WebSearchResult(title=url, url=url)
        page = self._fetch_http(result)
        if not _should_render_after_static(page):
            return page
        rendered = self._fetch_playwright(result)
        return rendered if rendered.status == "fetched" else page

    def fetch_static_url(self, url: str) -> FetchedPage:
        """Fetch one URL with plain HTTP extraction only.

        Args:
            url: Public HTTP(S) URL.

        Returns:
            Fetched page or structured failure.
        """
        return self._fetch_http(WebSearchResult(title=url, url=url))

    def render_url(self, url: str) -> FetchedPage:
        """Render one URL with Playwright.

        Args:
            url: Public HTTP(S) URL.

        Returns:
            Rendered page or structured failure.
        """
        return self._fetch_playwright(WebSearchResult(title=url, url=url))

    def fetch_results(self, results: Sequence[WebSearchResult]) -> tuple[FetchedPage, ...]:
        """Fetch search results with bounded concurrency and render fallback."""
        if not results:
            return ()
        max_workers = max(1, min(self._config.web_fetch_concurrency, len(results)))
        pages: list[FetchedPage | None] = [None] * len(results)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kyuri-web-fetch") as executor:
            futures = {executor.submit(self._fetch_http, result): index for index, result in enumerate(results)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    pages[index] = future.result()
                except Exception as exc:  # noqa: BLE001  # Per-page failures should not abort the whole research tool.
                    pages[index] = FetchedPage(
                        requested_url=results[index].url,
                        url=results[index].url,
                        title=results[index].title,
                        snippet=results[index].snippet,
                        status="failed",
                        error=str(exc),
                    )

        rendered_count = 0
        render_limit = max(0, self._config.web_render_max_pages)
        resolved_pages = [page or _failed_page(result, "Unknown page fetch failure.") for page, result in zip(pages, results, strict=True)]
        for index, page in enumerate(resolved_pages):
            if rendered_count >= render_limit:
                break
            if not _should_render_after_static(page):
                continue
            rendered = self._fetch_playwright(results[index])
            rendered_count += 1
            if rendered.status == "fetched":
                resolved_pages[index] = rendered
        return tuple(resolved_pages)

    def _searxng_json(self, query: str, *, max_results: int) -> Mapping[str, object]:
        del max_results
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency is installed in runtime extra.
            msg = "Install `httpx` to use SearXNG web search."
            raise RuntimeError(msg) from exc

        base_url = self._config.searxng_base_url.rstrip("/")
        params: dict[str, str] = {
            "q": query,
            "format": "json",
            "categories": "general",
            "safesearch": str(self._config.web_search_safe_search),
        }
        if self._config.web_search_language:
            params["language"] = self._config.web_search_language
        failures: list[str] = []
        for engine in (None, *self._config.web_search_fallback_engines):
            label = engine or "aggregate"
            payload = self._try_searxng_json(httpx, base_url=base_url, params=params, engine=engine)
            if isinstance(payload, Exception):
                failures.append(f"{label}: {payload}")
                continue
            payload_map = cast("Mapping[str, object]", payload)
            if _has_usable_searxng_results(payload_map, query=query):
                return payload_map
            failures.append(f"{label}: no usable results")
        msg = f"SearXNG search failed: {'; '.join(failures) or 'all fallback engines returned no results'}"
        raise RuntimeError(msg)

    def _try_searxng_json(
        self,
        httpx_module: object,
        *,
        base_url: str,
        params: Mapping[str, str],
        engine: str | None,
    ) -> Mapping[str, object] | Exception:
        httpx = cast("Any", httpx_module)
        request_params = dict(params)
        if engine:
            shortcut = _ENGINE_SHORTCUTS.get(engine.lower(), engine.lower())
            request_params["q"] = f"!{shortcut} {request_params['q']}"
        error: Exception | None = None
        for _ in range(self._config.web_fetch_retries + 1):
            try:
                with httpx.Client(
                    timeout=float(self._config.web_search_timeout_seconds),
                    follow_redirects=True,
                    headers={"User-Agent": _USER_AGENT},
                ) as client:
                    response = client.get(f"{base_url}/search", params=request_params)
                    response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    return TypeError("SearXNG returned a non-object JSON payload.")
                return cast("Mapping[str, object]", payload)
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                error = exc
        return error or RuntimeError("Unknown SearXNG search error.")

    def _fetch_http(self, result: WebSearchResult) -> FetchedPage:
        allowed, reason = _public_http_url_allowed(result.url)
        if not allowed:
            return FetchedPage(
                requested_url=result.url,
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                status="skipped",
                error=reason,
            )
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - dependency is installed in runtime extra.
            return _failed_page(result, f"Install `httpx` to fetch web pages: {exc}")

        error: Exception | None = None
        for _ in range(self._config.web_fetch_retries + 1):
            try:
                chunks: list[bytes] = []
                total = 0
                download_truncated = False
                with (
                    httpx.Client(
                        timeout=float(self._config.web_fetch_timeout_seconds),
                        follow_redirects=True,
                        headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.2"},
                    ) as client,
                    client.stream("GET", result.url) as response,
                ):
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not any(item in content_type for item in _TEXT_CONTENT_TYPES):
                        return _failed_page(result, f"Unsupported content type `{content_type}`.")
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > self._config.web_fetch_max_bytes:
                            download_truncated = True
                            break
                        chunks.append(chunk)
                    final_url = str(response.url)
                    encoding = response.encoding
                raw = _decode_response_body(b"".join(chunks), encoding)
                extracted = _extract_text(raw)
                title = extracted.title or result.title
                text, text_truncated = _truncate_with_status(extracted.text, self._config.web_fetch_max_chars)
                quality_flags = _page_quality_flags(
                    extracted.text,
                    raw_html=raw,
                    download_truncated=download_truncated,
                    text_truncated=text_truncated,
                )
                return FetchedPage(
                    requested_url=result.url,
                    url=_normalize_url(final_url),
                    title=title,
                    text=text,
                    snippet=result.snippet,
                    status="fetched",
                    method="http",
                    content_type=content_type,
                    text_chars=len(extracted.text),
                    returned_chars=len(text),
                    truncated=text_truncated,
                    quality_flags=quality_flags,
                )
            except Exception as exc:  # noqa: BLE001  # A failed page becomes one structured failure.
                error = exc
        return _failed_page(result, str(error))

    def _fetch_playwright(self, result: WebSearchResult) -> FetchedPage:
        allowed, reason = _public_http_url_allowed(result.url)
        if not allowed:
            return FetchedPage(
                requested_url=result.url,
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                status="skipped",
                error=reason,
            )
        try:
            from playwright.sync_api import sync_playwright  # noqa: PLC0415
        except ImportError as exc:
            return _failed_page(result, f"Playwright is not installed: {exc}")

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(user_agent=_USER_AGENT)
                page.goto(result.url, wait_until="domcontentloaded", timeout=int(self._config.web_render_timeout_seconds * 1000))
                html = page.content()
                title = _clean_text(page.title()) or result.title
                final_url = _normalize_url(page.url)
                browser.close()
            extracted = _extract_text(html)
            text, text_truncated = _truncate_with_status(extracted.text, self._config.web_fetch_max_chars)
            quality_flags = _page_quality_flags(
                extracted.text,
                raw_html=html,
                download_truncated=False,
                text_truncated=text_truncated,
            )
            return FetchedPage(
                requested_url=result.url,
                url=final_url,
                title=title or extracted.title or result.title,
                text=text,
                snippet=result.snippet,
                status="fetched",
                method="playwright",
                content_type="text/html; rendered=playwright",
                text_chars=len(extracted.text),
                returned_chars=len(text),
                truncated=text_truncated,
                quality_flags=quality_flags,
            )
        except Exception as exc:  # noqa: BLE001  # Exposed as per-page failure, not a whole tool crash.
            return _failed_page(result, f"Playwright render failed: {exc}")


class _RedisSearchCache:
    def __init__(self, config: AgentRuntimeConfig) -> None:
        self._ttl = config.web_search_cache_ttl_seconds
        self._prefix = "kyuri:websearch"
        self._client: object | None = None
        if self._ttl <= 0:
            return
        try:
            import redis  # noqa: PLC0415

            self._client = redis.Redis.from_url(config.redis_url, decode_responses=True)
        except Exception:  # noqa: BLE001  # Search cache is an optional optimization.
            self._client = None

    def get(self, *, query: str, max_results: int) -> WebSearchResponse | None:
        if self._client is None or self._ttl <= 0:
            return None
        try:
            raw = self._client.get(self._key(query, max_results))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001  # Cache errors should not affect search.
            return None
        if not isinstance(raw, str) or not raw:
            return None
        with suppress(json.JSONDecodeError, TypeError, ValueError):
            response = _response_from_dict(json.loads(raw))
            return replace(response, cache_hit=True)
        return None

    def set(self, response: WebSearchResponse, *, max_results: int) -> None:
        if self._client is None or self._ttl <= 0:
            return
        if not response.results:
            return
        try:
            self._client.setex(  # type: ignore[attr-defined]
                self._key(response.query, max_results),
                self._ttl,
                json.dumps(_response_to_dict(response), ensure_ascii=False),
            )
        except Exception:  # noqa: BLE001  # Cache errors should not affect search.
            return

    def _key(self, query: str, max_results: int) -> str:
        digest = hashlib.sha256(f"{_CACHE_VERSION}:{query}:{max_results}".encode()).hexdigest()
        return f"{self._prefix}:{digest}"


def _response_to_dict(response: WebSearchResponse) -> dict[str, object]:
    return {
        "query": response.query,
        "planned_queries": list(response.planned_queries),
        "results": [_result_to_dict(result) for result in response.results],
        "raw_result_count": response.raw_result_count,
        "deduped_count": response.deduped_count,
        "filtered_count": response.filtered_count,
        "failures": list(response.failures),
    }


def _response_from_dict(value: object) -> WebSearchResponse:
    data = _as_mapping(value)
    raw_results = data.get("results", [])
    results = []
    if isinstance(raw_results, Sequence) and not isinstance(raw_results, str | bytes):
        results = [_result_from_dict(item) for item in raw_results]
    return WebSearchResponse(
        query=_string_value(data.get("query")),
        planned_queries=tuple(_metadata_strings(data.get("planned_queries"))),
        results=tuple(result for result in results if result is not None),
        raw_result_count=int(_float_value(data.get("raw_result_count")) or 0),
        deduped_count=int(_float_value(data.get("deduped_count")) or 0),
        filtered_count=int(_float_value(data.get("filtered_count")) or 0),
        failures=tuple(_metadata_strings(data.get("failures"))),
    )


def _result_to_dict(result: WebSearchResult) -> dict[str, object]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "score": result.score,
        "source": result.source,
        "metadata": result.metadata,
    }


def _result_from_dict(value: object) -> WebSearchResult | None:
    data = _as_mapping(value)
    url = _string_value(data.get("url"))
    if not url:
        return None
    metadata = data.get("metadata")
    return WebSearchResult(
        title=_string_value(data.get("title")) or url,
        url=url,
        snippet=_string_value(data.get("snippet")),
        score=_float_value(data.get("score")),
        source=_string_value(data.get("source")) or "searxng",
        metadata=_dict_value(metadata),
    )


def blocked_query_reason(query: str) -> str:
    """Return a policy reason when web search should not run for a query."""
    if _PIRACY_QUERY_RE.search(query):
        return (
            "Web search is disabled for requests seeking pirated, leaked, cracked, "
            "or unauthorized download links. Ask for official sources or legal alternatives instead."
        )
    return ""


def _plan_search_queries(query: str, *, max_queries: int) -> tuple[str, ...]:
    normalized = _clean_text(query)
    if not normalized:
        return ("",)
    planned: list[str] = [normalized]
    if _CJK_RE.search(normalized):
        _append_unique(planned, _official_query(normalized))
        _append_unique(planned, _synonym_query(normalized))
        _append_unique(planned, _source_focused_query(normalized))
    return tuple(item for item in planned if item)[: min(max_queries, _MAX_QUERY_PLAN_SIZE)]


def _official_query(query: str) -> str:
    if any(marker in query for marker in ("官方", "公告", "确认", "证实")):
        return query
    return f"{query} 官方 公告 确认"


def _synonym_query(query: str) -> str:
    replacements = (
        ("访问中国", "访华"),
        ("外国领导人", "外国元首 领导人"),
        ("夏天", "夏季"),
        ("举办地点", "举办城市"),
    )
    rewritten = query
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    return rewritten


def _source_focused_query(query: str) -> str:
    if any(marker in query for marker in ("新华社", "人民日报", "外交部", "央视")):
        return query
    return f"{query} 新华社 人民日报 外交部"


def _append_unique(items: list[str], value: str) -> None:
    cleaned = _clean_text(value)
    if cleaned and cleaned not in items:
        items.append(cleaned)


def _results_from_raw(raw_results: Sequence[object], *, planned_query: str) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            continue
        raw_map = cast("Mapping[str, object]", raw)
        url = _string_value(raw_map.get("url"))
        if not url:
            continue
        normalized = _normalize_url(url)
        title = _string_value(raw_map.get("title")) or normalized
        snippet = _string_value(raw_map.get("content")) or _string_value(raw_map.get("snippet"))
        score = _float_value(raw_map.get("score"))
        engines = raw_map.get("engines")
        results.append(
            WebSearchResult(
                title=_clean_text(title),
                url=normalized,
                snippet=_truncate(_clean_text(snippet), _DEFAULT_SEARCH_SNIPPET_CHARS),
                score=score,
                metadata={
                    "planned_query": planned_query,
                    "engines": list(engines) if isinstance(engines, Sequence) and not isinstance(engines, str | bytes) else [],
                },
            )
        )
    return results


def _dedupe_results(results: Sequence[WebSearchResult]) -> list[WebSearchResult]:
    merged: dict[str, WebSearchResult] = {}
    for result in results:
        existing = merged.get(result.url)
        if existing is None:
            merged[result.url] = result
            continue
        merged[result.url] = _merge_duplicate_result(existing, result)
    return list(merged.values())


def _merge_duplicate_result(first: WebSearchResult, second: WebSearchResult) -> WebSearchResult:
    first_score = first.score or 0.0
    second_score = second.score or 0.0
    winner = second if second_score > first_score else first
    planned_queries = {
        item
        for item in (
            str(first.metadata.get("planned_query", "")),
            str(second.metadata.get("planned_query", "")),
            *_metadata_strings(first.metadata.get("matched_queries")),
            *_metadata_strings(second.metadata.get("matched_queries")),
        )
        if item
    }
    metadata = {
        **winner.metadata,
        "matched_queries": sorted(planned_queries),
        "duplicate_count": _int_metadata(first.metadata, "duplicate_count", default=1)
        + _int_metadata(second.metadata, "duplicate_count", default=1),
    }
    snippet = winner.snippet or first.snippet or second.snippet
    return replace(winner, snippet=snippet, metadata=metadata)


def _metadata_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(str(item) for item in value if item)
    return ()


def _dict_value(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _int_metadata(metadata: Mapping[str, object], key: str, *, default: int) -> int:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        with suppress(ValueError):
            return int(value)
    return default


def _float_metadata(metadata: Mapping[str, object], key: str, *, default: float) -> float:
    value = metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        with suppress(ValueError):
            return float(value)
    return default


def _with_quality_metadata(query: str, result: WebSearchResult) -> WebSearchResult:
    score, flags = _quality_score(query, result)
    filtered = _should_filter_result(result, flags)
    metadata = {
        **result.metadata,
        "quality_score": round(score, 4),
        "quality_flags": flags,
        "filtered": filtered,
    }
    return replace(result, metadata=metadata)


def _quality_score(query: str, result: WebSearchResult) -> tuple[float, tuple[str, ...]]:
    flags: list[str] = []
    haystack = f"{result.title} {result.snippet}".lower()
    terms = _query_terms(query)
    matched = sum(1 for term in terms if term.lower() in haystack)
    coverage = matched / max(1, len(terms))
    score = 0.25 + coverage
    if result.score is not None:
        score += min(float(result.score), 5.0) / 20.0
    host_path = _host_path(result.url)
    if any(marker in host_path for marker in _AUTHORITY_DOMAIN_MARKERS):
        score += 0.35
        flags.append("authority_domain")
    if _is_homepage(result.url):
        score -= 0.25
        flags.append("generic_homepage")
    if any(marker in host_path for marker in _LOW_VALUE_DOMAINS) or any(marker.lower() in haystack for marker in _LOW_VALUE_TITLE_MARKERS):
        score -= 0.6
        flags.append("low_value_lookup")
    if not result.snippet:
        score -= 0.15
        flags.append("missing_snippet")
    return max(score, 0.0), tuple(flags)


def _should_filter_result(result: WebSearchResult, flags: Sequence[str]) -> bool:
    del result
    return "low_value_lookup" in flags


def _query_terms(query: str) -> tuple[str, ...]:
    cjk_groups = re.findall(r"[\u3400-\u9fff]{2,}", query)
    latin_terms = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", query)
    terms = [*cjk_groups, *latin_terms]
    return tuple(dict.fromkeys(terms))


def _host_path(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.netloc.lower()}{parsed.path.lower()}"


def _is_homepage(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    return not path or path.lower() in {"index.html", "index.htm", "home"}


def _rerank_web_results(
    query: str,
    results: Sequence[WebSearchResult],
    *,
    config: AgentRuntimeConfig,
    limit: int,
) -> list[WebSearchResult]:
    candidates = sorted(
        results,
        key=lambda item: (-float(item.metadata.get("quality_score", 0.0)), item.url),
    )[: config.web_search_rerank_candidates]
    if not candidates:
        return []
    scores = _dashscope_rerank_scores(query, candidates, config=config)
    ranked: list[WebSearchResult] = []
    for index, result in enumerate(candidates):
        rerank_score = scores.get(index)
        quality_score = _float_metadata(result.metadata, "quality_score", default=0.0)
        final_score = rerank_score if rerank_score is not None else quality_score
        metadata = {
            **result.metadata,
            "rerank_score": rerank_score,
            "final_score": round(final_score, 4),
            "ranker": "dashscope" if rerank_score is not None else "quality_score",
        }
        ranked.append(replace(result, metadata=metadata))
    return sorted(
        ranked,
        key=lambda item: (-float(item.metadata.get("final_score", 0.0)), item.url),
    )[:limit]


def _dashscope_rerank_scores(
    query: str,
    results: Sequence[WebSearchResult],
    *,
    config: AgentRuntimeConfig,
) -> dict[int, float]:
    if not config.dashscope_api_key or not config.rag_rerank_model:
        return {}
    headers = {
        "Authorization": f"Bearer {config.dashscope_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.rag_rerank_model,
        "input": {
            "query": {"text": query},
            "documents": [{"text": _web_result_document(result)} for result in results],
        },
    }
    try:
        import httpx  # noqa: PLC0415

        response = httpx.post(config.rag_rerank_url, headers=headers, json=body, timeout=config.rag_rerank_timeout_seconds)
        response.raise_for_status()
        return _parse_rerank_scores(response.json())
    except Exception:  # noqa: BLE001  # Web search must remain available when rerank quota or network fails.
        return {}


def _web_result_document(result: WebSearchResult) -> str:
    return "\n".join(
        part
        for part in (
            f"title: {result.title}",
            f"url: {result.url}",
            f"snippet: {result.snippet}",
        )
        if part
    )


def _parse_rerank_scores(payload: object) -> dict[int, float]:
    data = _as_mapping(payload)
    output = _as_mapping(data.get("output"))
    raw_results = output.get("results")
    if raw_results is None:
        raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return {}
    scores: dict[int, float] = {}
    for raw in raw_results:
        item = _as_mapping(raw)
        index = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if index is None or score is None:
            continue
        with suppress(ValueError):
            scores[int(str(index))] = float(str(score))
    return scores


def _as_mapping(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}


def format_web_search_results(results: Sequence[WebSearchResult] | WebSearchResponse, *, query: str = "") -> str:
    """Format SearXNG results for an LLM tool message."""
    response = results if isinstance(results, WebSearchResponse) else None
    if response is not None:
        final_results: Sequence[WebSearchResult] = response.results
    else:
        final_results = cast("Sequence[WebSearchResult]", results)
    display_query = query or (response.query if response is not None else "")
    if not final_results and response is None:
        return "No web search results found."
    header = f'<web_search_results query="{_xml_escape(display_query)}">' if display_query else "<web_search_results>"
    parts = [header]
    if response is not None:
        parts.append(
            "\n".join(
                (
                    "<diagnostics>",
                    f"planned_queries: {_xml_escape('; '.join(response.planned_queries))}",
                    f"raw_results: {response.raw_result_count}; deduped: {response.deduped_count}; "
                    f"filtered: {response.filtered_count}; cache_hit: {str(response.cache_hit).lower()}",
                    f"failures: {_xml_escape('; '.join(response.failures)) if response.failures else 'none'}",
                    "</diagnostics>",
                )
            )
        )
    if not final_results:
        parts.append("No web search results found.")
    for index, result in enumerate(final_results, start=1):
        quality_flags = result.metadata.get("quality_flags", ())
        flags = ", ".join(_metadata_strings(quality_flags))
        score = result.metadata.get("final_score", result.metadata.get("quality_score", ""))
        planned_query = _string_value(result.metadata.get("planned_query"))
        parts.append(
            "\n".join(
                (
                    f"- [{index}] {_xml_escape(result.title)}",
                    f"  url: {result.url}",
                    f"  score: {score}",
                    f"  planned_query: {_xml_escape(planned_query)}",
                    f"  quality_flags: {_xml_escape(flags or 'none')}",
                    f"  snippet: {_xml_escape(result.snippet)}",
                )
            )
        )
    parts.append("</web_search_results>")
    return "\n".join(parts)


def format_fetched_page(page: FetchedPage, *, index: int = 1) -> str:
    """Format one fetched page for an LLM tool message."""
    flags = ", ".join(page.quality_flags) or "none"
    lines = [
        f"# Web Page Fetch Result {index}",
        "",
        f"- Status: {page.status}",
        f"- Method: {_display_fetch_method(page.method)}",
        f"- Title: {page.title or '(untitled)'}",
        f"- URL: {page.url}",
        f"- Requested URL: {page.requested_url}",
        f"- Content type: {page.content_type or 'unknown'}",
        f"- Text chars: {page.text_chars}",
        f"- Returned chars: {page.returned_chars}",
        f"- Truncated: {str(page.truncated).lower()}",
        f"- Quality flags: {flags}",
    ]
    if page.status != "fetched":
        lines.extend(
            (
                f"- Error: {page.error or 'Page could not be fetched.'}",
                f"- Search snippet: {page.snippet}",
            )
        )
        return "\n".join(lines)
    lines.extend(("", "## Extracted Content", "", page.text or "(no extracted content)"))
    return "\n".join(lines)


def format_web_research(result: WebResearchResult) -> str:
    """Format combined search and page-reading output for an LLM."""
    if result.error:
        return f"<web_research_error>{_xml_escape(result.error)}</web_research_error>"
    if not result.search_results:
        return "No web search results found."
    parts = [f'<web_research_results query="{_xml_escape(result.query)}">']
    parts.append("<search_results>")
    for index, search_result in enumerate(result.search_results, start=1):
        parts.append(f"- [{index}] {_xml_escape(search_result.title)} | {search_result.url} | {_xml_escape(search_result.snippet)}")
    parts.append("</search_results>")
    parts.append("<opened_pages>")
    for index, page in enumerate(result.pages, start=1):
        parts.append(format_fetched_page(page, index=index))
    parts.append("</opened_pages>")
    parts.append("</web_research_results>")
    return "\n".join(parts)


@dataclass(frozen=True, kw_only=True)
class _ExtractedText:
    title: str
    text: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.lower()
        if normalized == "title":
            self._in_title = True
        if normalized in _SKIP_TAGS:
            self._skip_depth += 1
        if normalized in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "title":
            self._in_title = False
        if normalized in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if normalized in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        self.parts.append(text)


def _extract_text(html: str) -> _ExtractedText:
    parser = _TextExtractor()
    with suppress(Exception):
        parser.feed(html)
    title = _clean_text(" ".join(parser.title_parts))
    text = _clean_multiline(" ".join(parser.parts))
    return _ExtractedText(title=title, text=text)


def _clean_multiline(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = _clean_text(raw)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    return _truncate_with_status(text, max_chars)[0]


def _truncate_with_status(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return f"{text[:max_chars].rstrip()}\n...[truncated]", True


def _page_quality_flags(
    text: str,
    *,
    raw_html: str,
    download_truncated: bool,
    text_truncated: bool,
) -> tuple[str, ...]:
    flags: list[str] = []
    if not text.strip():
        flags.append("empty_text")
    elif len(text) < _MIN_TEXT_CHARS_BEFORE_RENDER:
        flags.append("too_short")
    marker_text = _marker_text(f"{text[:4000]}\n{raw_html[:8000]}")
    if any(_marker_text(marker) in marker_text for marker in _PAGE_BLOCKED_MARKERS):
        flags.append("blocked_or_verification")
    if any(_marker_text(marker) in marker_text for marker in _PAGE_JS_REQUIRED_MARKERS):
        flags.append("maybe_js_required")
    if download_truncated:
        flags.append("download_truncated")
    if text_truncated:
        flags.append("text_truncated")
    return tuple(dict.fromkeys(flags))


def _marker_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _should_render_after_static(page: FetchedPage) -> bool:
    if page.status == "skipped":
        return False
    if page.status != "fetched":
        return True
    return bool({"empty_text", "too_short", "maybe_js_required", "blocked_or_verification"} & set(page.quality_flags))


def _display_fetch_method(method: str) -> str:
    if method == "http":
        return "static"
    if method == "playwright":
        return "playwright"
    return method or "none"


def _decode_response_body(body: bytes, declared_encoding: str | None) -> str:
    sniffed = _sniff_html_charset(body)
    candidates = _encoding_candidates(sniffed, declared_encoding)
    best = ""
    best_score: int | None = None
    for encoding in candidates:
        with suppress(LookupError):
            decoded = body.decode(encoding, errors="replace")
            score = decoded.count("\ufffd")
            if best_score is None or score < best_score:
                best = decoded
                best_score = score
                if score == 0:
                    break
    return best or body.decode("utf-8", errors="replace")


def _sniff_html_charset(body: bytes) -> str | None:
    match = _CHARSET_RE.search(body[:4096])
    if match is None:
        return None
    return match.group(1).decode("ascii", errors="ignore").strip() or None


def _encoding_candidates(sniffed: str | None, declared: str | None) -> tuple[str, ...]:
    candidates: list[str] = []
    for encoding in (sniffed, declared, "utf-8", "gb18030", "gbk"):
        if not encoding:
            continue
        normalized = encoding.strip().lower()
        if normalized in _LOW_CONFIDENCE_ENCODINGS and sniffed:
            continue
        if normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _float_value(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _positive_limit(value: int | None, *, default: int) -> int:
    if value is None or value <= 0:
        return default
    return min(value, default)


def _has_usable_searxng_results(payload: Mapping[str, object], *, query: str) -> bool:
    results = payload.get("results")
    if not isinstance(results, Sequence) or isinstance(results, str | bytes) or len(results) == 0:
        return False
    return not _looks_like_single_character_lookup(query, results)


def _looks_like_single_character_lookup(query: str, results: Sequence[object]) -> bool:
    cjk_chars = _CJK_RE.findall(query)
    if len(cjk_chars) < _MIN_CJK_QUERY_CHARS_FOR_LOOKUP_GUARD:
        return False
    first = cjk_chars[0]
    checked = 0
    lookup_hits = 0
    for raw in results[:5]:
        if not isinstance(raw, Mapping):
            continue
        checked += 1
        raw_map = cast("Mapping[str, object]", raw)
        title = _clean_text(_string_value(raw_map.get("title")))
        snippet = _clean_text(_string_value(raw_map.get("content")) or _string_value(raw_map.get("snippet")))
        haystack = f"{title} {snippet}"
        if _lookup_title_startswith(title, first) and any(marker in haystack for marker in _SINGLE_CHAR_LOOKUP_MARKERS):
            lookup_hits += 1
    return checked >= _MIN_SINGLE_CHAR_LOOKUP_RESULTS and lookup_hits >= _MIN_SINGLE_CHAR_LOOKUP_RESULTS


def _lookup_title_startswith(title: str, char: str) -> bool:
    return title.lstrip("\u300a\u3008\u300c\u300e\u3010([\uff08 ")[:1] == char


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_QUERY_PARAMS and not key.lower().startswith("utm_")
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def _failed_page(result: WebSearchResult, error: str) -> FetchedPage:
    return FetchedPage(
        requested_url=result.url,
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        status="failed",
        error=error,
    )


def _public_http_url_allowed(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "Only HTTP(S) URLs are allowed."
    if not parsed.hostname:
        return False, "URL is missing a hostname."
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"}:
        return False, "Localhost URLs are not allowed for web fetching."
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return False, f"Could not resolve hostname: {exc}."
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False, "Private, local, reserved, or link-local addresses are not allowed."
    return True, ""


def _xml_escape(text: object) -> str:
    raw = str(text)
    return raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
