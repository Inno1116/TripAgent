"""Reranker implementations for hybrid retrieval."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, cast

from kyuriagents.rag._text import tokenize

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from kyuriagents.rag.types import RetrievedChunk

_DEFAULT_FUSED_WEIGHT = 0.25
_DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
_LOGGER = logging.getLogger(__name__)


class _HttpResponse(Protocol):
    def raise_for_status(self) -> None:
        """Raise when the HTTP response is not successful."""
        ...

    def json(self) -> object:
        """Return parsed JSON response."""
        ...


class _HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
    ) -> _HttpResponse:
        """Post a JSON request."""
        ...


class FusedScoreReranker:
    """Reranker that trusts the hybrid fusion score."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Sort candidates by fused score.

        Args:
            query: Original user query. Accepted for protocol compatibility.
            candidates: Candidate chunks after hybrid fusion.
            limit: Maximum number of chunks to return.

        Returns:
            Final ranked chunks with `rerank_score` populated.
        """
        del query
        ranked = [candidate.with_scores(rerank_score=candidate.fused_score) for candidate in candidates]
        return sorted(ranked, key=lambda item: (-_score_or_zero(item.rerank_score), item.chunk_id))[:limit]


class LexicalReranker:
    """Dependency-free reranker for local validation and fallback deployments."""

    def __init__(self, *, fused_weight: float = _DEFAULT_FUSED_WEIGHT) -> None:
        """Initialize the reranker.

        Args:
            fused_weight: Weight added from the upstream fusion score.
        """
        self._fused_weight = fused_weight

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Rerank by query-token overlap plus a small fused-score prior.

        Args:
            query: Original user query.
            candidates: Candidate chunks after hybrid fusion.
            limit: Maximum number of chunks to return.

        Returns:
            Final ranked chunks with `rerank_score` populated.
        """
        query_terms = set(tokenize(query))
        ranked: list[RetrievedChunk] = []
        for candidate in candidates:
            candidate_terms = set(tokenize(f"{candidate.text} {candidate.metadata.title} {candidate.metadata.section_path}"))
            overlap = len(query_terms & candidate_terms)
            score = float(overlap) + (candidate.fused_score * self._fused_weight)
            ranked.append(candidate.with_scores(rerank_score=score))
        return sorted(ranked, key=lambda item: (-_score_or_zero(item.rerank_score), item.chunk_id))[:limit]


class DashScopeTextReranker:
    """DashScope text reranker with fused-score fallback."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-vl-rerank",
        endpoint: str = _DASHSCOPE_RERANK_URL,
        timeout_seconds: float = 10.0,
        fallback: FusedScoreReranker | None = None,
        client: _HttpClient | None = None,
    ) -> None:
        """Initialize the DashScope reranker.

        Args:
            api_key: DashScope API key.
            model: DashScope rerank model name.
            endpoint: DashScope rerank endpoint URL.
            timeout_seconds: HTTP request timeout.
            fallback: Fallback reranker used when the API is unavailable.
            client: Optional HTTP client for tests.
        """
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._fallback = fallback or FusedScoreReranker()
        self._client = client

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Rerank candidates through DashScope.

        Args:
            query: Original user query.
            candidates: Candidate chunks after hybrid fusion.
            limit: Maximum number of chunks to return.

        Returns:
            Final ranked chunks. Falls back to fused score on API errors.
        """
        if not candidates:
            return []
        if not self._api_key:
            return self._fallback.rerank(query, candidates, limit=limit)
        try:
            scores = self._request_scores(query, candidates)
        except Exception as exc:  # noqa: BLE001  # Rerank is an optional quality layer; retrieval should fail open.
            _LOGGER.warning("DashScope rerank failed; falling back to fused score: %s", exc)
            return self._fallback.rerank(query, candidates, limit=limit)
        ranked = [candidate.with_scores(rerank_score=scores.get(index, candidate.fused_score)) for index, candidate in enumerate(candidates)]
        return sorted(
            ranked,
            key=lambda item: (
                -_score_or_zero(item.rerank_score),
                -item.fused_score,
                item.chunk_id,
            ),
        )[:limit]

    def _request_scores(self, query: str, candidates: Sequence[RetrievedChunk]) -> dict[int, float]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "input": {
                "query": {"text": query},
                "documents": [{"text": _document_text(candidate)} for candidate in candidates],
            },
        }
        response = self._post(headers=headers, body=body)
        response.raise_for_status()
        return _parse_dashscope_scores(response.json())

    def _post(self, *, headers: Mapping[str, str], body: Mapping[str, object]) -> _HttpResponse:
        if self._client is not None:
            return self._client.post(self._endpoint, headers=headers, json=body)
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            msg = "Install `kyuriagents[runtime]` or `httpx` to use DashScope rerank."
            raise ImportError(msg) from exc
        return cast("_HttpResponse", httpx.post(self._endpoint, headers=headers, json=body, timeout=self._timeout_seconds))


def _score_or_zero(score: float | None) -> float:
    return 0.0 if score is None else score


def _document_text(candidate: RetrievedChunk) -> str:
    text = candidate.text.strip()
    if text:
        return text
    metadata = candidate.metadata
    fallback = " ".join(part for part in (metadata.title, metadata.section_path, metadata.source_uri) if part)
    return fallback or candidate.chunk_id


def _parse_dashscope_scores(payload: object) -> dict[int, float]:
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
        scores[int(str(index))] = float(str(score))
    return scores


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, dict):
        return cast("Mapping[str, object]", value)
    return {}
