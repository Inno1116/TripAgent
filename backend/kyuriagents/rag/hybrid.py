"""Hybrid retrieval pipeline for RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from kyuriagents.rag.query import IdentityQueryRewriter, QueryRewriter
from kyuriagents.rag.rerank import FusedScoreReranker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kyuriagents.rag.metadata import RetrievalScope
    from kyuriagents.rag.types import ChunkHydrator, KeywordSearcher, Reranker, RetrievedChunk, VectorSearcher

_DEFAULT_VECTOR_CANDIDATES = 50
_DEFAULT_KEYWORD_CANDIDATES = 50
_DEFAULT_RERANK_CANDIDATES = 80
_DEFAULT_TOP_K = 5
_DEFAULT_RRF_K = 60
_DEFAULT_WEIGHT = 1.0
_LOGGER = logging.getLogger(__name__)

_SearchKind = Literal["vector", "keyword"]


@dataclass(frozen=True, kw_only=True)
class HybridSearchConfig:
    """Configuration for hybrid retrieval.

    Args:
        vector_candidates: Number of candidates to request from vector search.
        keyword_candidates: Number of candidates to request from keyword search.
        rerank_candidates: Number of fused candidates to send to reranking.
        top_k: Number of final chunks returned by default.
        rrf_k: Reciprocal rank fusion constant.
        vector_weight: Weight applied to vector ranks during fusion.
        keyword_weight: Weight applied to keyword ranks during fusion.
    """

    vector_candidates: int = _DEFAULT_VECTOR_CANDIDATES
    keyword_candidates: int = _DEFAULT_KEYWORD_CANDIDATES
    rerank_candidates: int = _DEFAULT_RERANK_CANDIDATES
    top_k: int = _DEFAULT_TOP_K
    rrf_k: int = _DEFAULT_RRF_K
    vector_weight: float = _DEFAULT_WEIGHT
    keyword_weight: float = _DEFAULT_WEIGHT

    def __post_init__(self) -> None:
        """Validate numeric configuration."""
        positive_ints = {
            "vector_candidates": self.vector_candidates,
            "keyword_candidates": self.keyword_candidates,
            "rerank_candidates": self.rerank_candidates,
            "top_k": self.top_k,
            "rrf_k": self.rrf_k,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                msg = f"{name} must be greater than 0"
                raise ValueError(msg)
        if self.vector_weight < 0.0 or self.keyword_weight < 0.0:
            msg = "fusion weights must be non-negative"
            raise ValueError(msg)
        if self.vector_weight == 0.0 and self.keyword_weight == 0.0:
            msg = "at least one fusion weight must be greater than 0"
            raise ValueError(msg)


class HybridRAGRetriever:
    """Online RAG retrieval pipeline.

    The pipeline runs query rewriting, Milvus-style vector retrieval,
    Elasticsearch-style keyword retrieval, reciprocal rank fusion, reranking,
    and final Top-K selection.
    """

    def __init__(
        self,
        *,
        vector_searcher: VectorSearcher,
        keyword_searcher: KeywordSearcher,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        chunk_hydrator: ChunkHydrator | None = None,
        config: HybridSearchConfig | None = None,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_searcher: Semantic search adapter.
            keyword_searcher: Keyword search adapter.
            query_rewriter: Optional query rewriter.
            reranker: Optional reranker.
            chunk_hydrator: Optional source-of-truth text hydrator.
            config: Optional retrieval configuration.
        """
        self._vector_searcher = vector_searcher
        self._keyword_searcher = keyword_searcher
        self._query_rewriter = query_rewriter or IdentityQueryRewriter()
        self._reranker = reranker or FusedScoreReranker()
        self._chunk_hydrator = chunk_hydrator
        self._config = config or HybridSearchConfig()

    def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        history: Sequence[str] = (),
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve final Top-K chunks for a user query.

        Args:
            query: User query.
            scope: Tenant and authorization filters.
            history: Optional recent conversation turns for query rewriting.
            top_k: Optional override for final result count.

        Returns:
            Reranked chunks.
        """
        final_limit = self._resolve_top_k(top_k)
        rewrite = self._query_rewriter.rewrite(query, history=history)
        if not rewrite.search_queries:
            return []

        ranked_lists: list[tuple[_SearchKind, list[RetrievedChunk]]] = []
        for search_query in rewrite.search_queries:
            ranked_lists.append(
                (
                    "vector",
                    self._vector_searcher.search(
                        search_query,
                        scope=scope,
                        limit=self._config.vector_candidates,
                    ),
                )
            )
            ranked_lists.append(
                (
                    "keyword",
                    self._keyword_searcher.search(
                        search_query,
                        scope=scope,
                        limit=self._config.keyword_candidates,
                    ),
                )
            )

        fused = self._fuse(ranked_lists)
        hydrated = self._hydrate(fused)
        rerank_input = hydrated[: self._config.rerank_candidates]
        return self._reranker.rerank(
            rewrite.rewritten_query,
            rerank_input,
            limit=final_limit,
        )

    def _hydrate(self, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._chunk_hydrator is None:
            return candidates
        try:
            return self._chunk_hydrator.hydrate(candidates)
        except Exception as exc:  # noqa: BLE001  # Retrieval can still use ES text or metadata if hydration is temporarily down.
            _LOGGER.warning("RAG chunk hydration failed; continuing with raw retrieval hits: %s", exc)
            return candidates

    def _resolve_top_k(self, top_k: int | None) -> int:
        final_limit = self._config.top_k if top_k is None else top_k
        if final_limit <= 0:
            msg = "top_k must be greater than 0"
            raise ValueError(msg)
        return final_limit

    def _fuse(
        self,
        ranked_lists: Sequence[tuple[_SearchKind, Sequence[RetrievedChunk]]],
    ) -> list[RetrievedChunk]:
        candidates: dict[str, RetrievedChunk] = {}
        fused_scores: dict[str, float] = {}
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        for kind, results in ranked_lists:
            weight = self._weight_for(kind)
            if weight == 0.0:
                continue
            for rank, candidate in enumerate(results, start=1):
                chunk_id = candidate.chunk_id
                existing = candidates.get(chunk_id)
                if existing is None or (not existing.text and candidate.text):
                    candidates[chunk_id] = candidate
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (weight / float(self._config.rrf_k + rank))
                if kind == "vector" and candidate.vector_score is not None:
                    vector_scores[chunk_id] = max(candidate.vector_score, vector_scores.get(chunk_id, candidate.vector_score))
                if kind == "keyword" and candidate.keyword_score is not None:
                    keyword_scores[chunk_id] = max(candidate.keyword_score, keyword_scores.get(chunk_id, candidate.keyword_score))

        fused: list[RetrievedChunk] = []
        for chunk_id, candidate in candidates.items():
            fused.append(
                replace(
                    candidate,
                    vector_score=vector_scores.get(chunk_id),
                    keyword_score=keyword_scores.get(chunk_id),
                    fused_score=fused_scores[chunk_id],
                    rerank_score=None,
                )
            )
        return sorted(
            fused,
            key=lambda item: (
                -item.fused_score,
                -_score_or_zero(item.vector_score),
                -_score_or_zero(item.keyword_score),
                item.chunk_id,
            ),
        )

    def _weight_for(self, kind: _SearchKind) -> float:
        if kind == "vector":
            return self._config.vector_weight
        return self._config.keyword_weight


def _score_or_zero(score: float | None) -> float:
    return 0.0 if score is None else score
