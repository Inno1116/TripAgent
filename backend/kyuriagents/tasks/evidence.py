"""Evidence-agent adapters used by task-mode step execution."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from kyuriagents.rag import (
    DashScopeTextReranker,
    ElasticsearchKeywordStore,
    HybridRAGRetriever,
    MilvusVectorStore,
    PostgresChunkTextHydrator,
    RetrievalScope,
)
from kyuriagents.runtime.evidence import EvidenceFinding, EvidencePackage, EvidenceSource
from kyuriagents.tools import ToolDescriptor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kyuriagents.rag.types import RetrievedChunk
    from kyuriagents.runtime import AgentRuntimeConfig
    from kyuriagents.websearch.service import (
        FetchedPage,
        WebSearchResponse,
        WebSearchResult,
        WebSearchService,
    )

_LOGGER = logging.getLogger(__name__)
_EXCERPT_CHARS = 600
_MAX_FINDINGS = 8


@dataclass(frozen=True, kw_only=True)
class EvidenceRequest:
    """Input passed from one task step into an evidence agent.

    Args:
        query: Query or instruction for the evidence agent.
        tenant_id: Tenant that owns the task.
        user_id: User that requested the task.
        thread_id: Conversation thread associated with the task.
        goal: Original task goal.
        input: Raw step input supplied by the planner.
    """

    query: str
    tenant_id: str
    user_id: str
    thread_id: str
    goal: str
    input: Mapping[str, object]


class EvidenceAgent(Protocol):
    """Protocol implemented by task-mode information agents."""

    @property
    def descriptor(self) -> ToolDescriptor:
        """Return planner-visible metadata for this evidence agent."""
        ...

    def run(self, request: EvidenceRequest) -> EvidencePackage:
        """Return a structured evidence package for a task step."""
        ...


class RAGEvidenceAgent:
    """Search private knowledge bases and return concise evidence."""

    def __init__(self, config: AgentRuntimeConfig) -> None:
        """Initialize the agent.

        Args:
            config: Runtime configuration.
        """
        self._config = config
        self._retriever: HybridRAGRetriever | None = None
        self._descriptor = ToolDescriptor(
            name="rag_agent",
            description=(
                "Search uploaded or indexed knowledge-base documents and return a structured evidence package. "
                "Use for private files, PDFs, papers, user documents, or local knowledge-base questions."
            ),
            risk="read_only",
            source="runtime",
            tags=("rag", "knowledge_base", "evidence"),
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        """Return planner-visible metadata for RAG evidence retrieval."""
        return self._descriptor

    def run(self, request: EvidenceRequest) -> EvidencePackage:
        """Retrieve knowledge-base chunks for a task step."""
        retriever = self._get_retriever()
        top_k = _int_input(request.input.get("top_k"), default=6)
        scope = RetrievalScope(tenant_id=request.tenant_id, user_id=request.user_id, kb_ids=self._config.rag_kb_ids)
        try:
            chunks = retriever.retrieve(request.query, scope=scope, top_k=top_k)
        except Exception as exc:  # noqa: BLE001  # Retrieval outages should become evidence gaps, not task crashes.
            return EvidencePackage(
                conclusion="Knowledge-base retrieval failed before evidence could be collected.",
                missing=[f"No knowledge-base evidence could be verified for: {request.query}"],
                failures=[str(exc)],
            )
        if not chunks:
            return EvidencePackage(
                conclusion="No relevant knowledge-base chunks were found.",
                missing=[f"No indexed document evidence matched: {request.query}"],
            )
        sources = [_source_from_chunk(chunk) for chunk in chunks]
        findings = [_finding_from_chunk(chunk, index=index) for index, chunk in enumerate(chunks[:_MAX_FINDINGS])]
        return EvidencePackage(
            conclusion=f"Found {len(chunks)} relevant knowledge-base chunks for the task.",
            findings=findings,
            sources=sources,
        )

    def _get_retriever(self) -> HybridRAGRetriever:
        retriever = self._retriever
        if retriever is not None:
            return retriever
        from kyuriagents.runtime.dashscope import create_dashscope_embed_query  # noqa: PLC0415

        config = self._config
        retriever = HybridRAGRetriever(
            vector_searcher=MilvusVectorStore(
                collection_name=config.rag_milvus_collection,
                uri=config.rag_milvus_uri,
                token=config.rag_milvus_token,
                db_name=config.rag_milvus_db,
                embed_query=create_dashscope_embed_query(config),
            ),
            keyword_searcher=ElasticsearchKeywordStore(index=config.rag_es_index, url=config.rag_es_url),
            chunk_hydrator=PostgresChunkTextHydrator(dsn=config.postgres_dsn) if config.postgres_dsn else None,
            reranker=DashScopeTextReranker(
                api_key=config.dashscope_api_key or "",
                model=config.rag_rerank_model,
                endpoint=config.rag_rerank_url,
                timeout_seconds=config.rag_rerank_timeout_seconds,
            )
            if config.rag_rerank_model
            else None,
        )
        self._retriever = retriever
        return retriever


class WebEvidenceAgent:
    """Search public web sources and return a bounded evidence package."""

    def __init__(self, config: AgentRuntimeConfig) -> None:
        """Initialize the agent.

        Args:
            config: Runtime configuration.
        """
        self._config = config
        self._service: WebSearchService | None = None
        timeout = max(1, int(max(config.web_search_timeout_seconds, config.web_fetch_timeout_seconds, config.web_render_timeout_seconds)))
        self._descriptor = ToolDescriptor(
            name="web_agent",
            description=(
                "Search current public web pages, open a bounded set of promising URLs, and return a structured "
                "evidence package with sources. Use for current events, official websites, schedules, prices, "
                "or public information not guaranteed to be in the local knowledge base."
            ),
            risk="external_read",
            source="runtime",
            timeout_seconds=timeout,
            tags=("web", "search", "evidence"),
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        """Return planner-visible metadata for web evidence retrieval."""
        return self._descriptor

    def run(self, request: EvidenceRequest) -> EvidencePackage:
        """Search and read public web pages for a task step.

        Normal search and page failures are represented inside the returned
        evidence package so one bad web call does not abort the entire plan.
        """
        from kyuriagents.websearch.service import blocked_query_reason  # noqa: PLC0415

        reason = blocked_query_reason(request.query)
        if reason:
            return EvidencePackage(conclusion="The web query was blocked by policy.", failures=[reason])
        service = self._get_service()
        max_results = _int_input(request.input.get("max_results"), default=self._config.web_search_max_results)
        max_pages = min(_int_input(request.input.get("max_pages"), default=self._config.web_fetch_max_pages), self._config.web_fetch_max_pages)
        try:
            response = service.search_with_diagnostics(request.query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001  # Search outages should become evidence gaps, not task crashes.
            return EvidencePackage(
                conclusion="Public web search failed before sources could be collected.",
                missing=[f"No web evidence could be verified for: {request.query}"],
                failures=[str(exc)],
            )
        pages = self._fetch_pages(service, response.results[:max_pages])
        return _package_from_web_response(request.query, response=response, pages=pages)

    def _get_service(self) -> WebSearchService:
        service = self._service
        if service is not None:
            return service
        from kyuriagents.websearch import WebSearchService  # noqa: PLC0415

        service = WebSearchService(self._config)
        self._service = service
        return service

    def _fetch_pages(self, service: WebSearchService, results: Sequence[WebSearchResult]) -> tuple[FetchedPage, ...]:
        if not results:
            return ()
        try:
            return service.fetch_results(results)
        except Exception as exc:  # noqa: BLE001  # Keep search snippets even if page opening fails globally.
            _LOGGER.info("Web evidence page fetch failed; continuing with snippets: %s", exc)
            return ()


def create_evidence_agents(config: AgentRuntimeConfig) -> dict[str, EvidenceAgent]:
    """Create task-mode evidence agents enabled by runtime configuration.

    Args:
        config: Runtime configuration.

    Returns:
        Evidence agents keyed by step kind.
    """
    agents: dict[str, EvidenceAgent] = {}
    if config.enable_rag:
        agents["rag"] = RAGEvidenceAgent(config)
    if config.enable_web_search:
        agents["web"] = WebEvidenceAgent(config)
    return agents


def format_evidence_package(package: EvidencePackage) -> str:
    """Serialize an evidence package for step output storage and final answering."""
    return "<evidence_package>\n" + json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n</evidence_package>"


def _package_from_web_response(query: str, *, response: WebSearchResponse, pages: Sequence[FetchedPage]) -> EvidencePackage:
    sources: list[EvidenceSource] = []
    findings: list[EvidenceFinding] = []
    failures = list(response.failures)
    page_by_url = {_canonical_url(page.url or page.requested_url): page for page in pages}
    for result in response.results:
        page = page_by_url.get(_canonical_url(result.url))
        text = _page_text(page) if page is not None else ""
        source_index = len(sources)
        sources.append(_source_from_web_result(result, page=page, text=text))
        claim = _web_claim(result, page=page, text=text)
        findings.append(EvidenceFinding(claim=claim, source_indices=[source_index], confidence=_web_confidence(result, page=page)))
        if page is not None and page.status != "fetched":
            failures.append(f"{result.url}: {page.error or page.status}")
    if not sources:
        return EvidencePackage(
            conclusion="No public web search results were found.",
            missing=[f"No public web evidence matched: {query}"],
            failures=failures,
        )
    missing = [] if any((source.quote or "").strip() for source in sources) else ["Search returned URLs, but no page excerpts could be verified."]
    return EvidencePackage(
        conclusion=f"Found {len(sources)} public web sources for the task.",
        findings=findings[:_MAX_FINDINGS],
        sources=sources,
        missing=missing,
        failures=failures,
    )


def _source_from_chunk(chunk: RetrievedChunk) -> EvidenceSource:
    metadata = chunk.metadata
    title = metadata.title or metadata.source_uri or chunk.chunk_id
    return EvidenceSource(
        title=title,
        url=metadata.source_uri or f"chunk:{chunk.chunk_id}",
        source_type="knowledge_base",
        quote=_excerpt(chunk.text or title),
    )


def _finding_from_chunk(chunk: RetrievedChunk, *, index: int) -> EvidenceFinding:
    score = chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score
    return EvidenceFinding(
        claim=_excerpt(chunk.text or chunk.metadata.title or chunk.chunk_id, max_chars=260),
        source_indices=[index],
        confidence=_confidence(score),
    )


def _source_from_web_result(result: WebSearchResult, *, page: FetchedPage | None, text: str) -> EvidenceSource:
    title = (page.title if page is not None and page.title else result.title) or result.url
    quote = _excerpt(text or result.snippet)
    return EvidenceSource(title=title, url=result.url, source_type="web", quote=quote)


def _web_claim(result: WebSearchResult, *, page: FetchedPage | None, text: str) -> str:
    if text:
        return _excerpt(text, max_chars=260)
    if page is not None and page.status != "fetched":
        return f"Page could not be fully fetched; search snippet says: {_excerpt(result.snippet, max_chars=180)}"
    return _excerpt(result.snippet or result.title or result.url, max_chars=260)


def _web_confidence(result: WebSearchResult, *, page: FetchedPage | None) -> float:
    score = _float_value(result.metadata.get("final_score"), default=result.score or 0.5)
    confidence = _confidence(score)
    if page is None:
        return max(0.35, confidence - 0.1)
    if page.status != "fetched":
        return max(0.25, confidence - 0.25)
    if "too_short" in page.quality_flags:
        return max(0.35, confidence - 0.15)
    return confidence


def _page_text(page: FetchedPage | None) -> str:
    if page is None or page.status != "fetched":
        return ""
    return page.text


def _excerpt(text: str, *, max_chars: int = _EXCERPT_CHARS) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[:max_chars]}...[truncated]"


def _int_input(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return max(1, value)
    if not isinstance(value, str | bytes | bytearray):
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, resolved)


def _float_value(value: object, *, default: float) -> float:
    if isinstance(value, float | int):
        return float(value)
    if not isinstance(value, str | bytes | bytearray):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence(score: float | None) -> float:
    if score is None:
        return 0.5
    if 0.0 <= score <= 1.0:
        return max(0.2, min(1.0, score))
    return max(0.2, min(1.0, score / (abs(score) + 1.0)))


def _canonical_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


__all__ = [
    "EvidenceAgent",
    "EvidenceRequest",
    "RAGEvidenceAgent",
    "WebEvidenceAgent",
    "create_evidence_agents",
    "format_evidence_package",
]
