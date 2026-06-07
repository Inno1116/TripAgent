"""Shared types and protocols for RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kyuriagents.rag.metadata import ChunkMetadata, RetrievalScope


@dataclass(frozen=True, kw_only=True)
class DocumentChunk:
    """A chunk available to retrieval stores.

    Args:
        text: Chunk text.
        metadata: Chunk metadata.
        embedding: Optional precomputed embedding used by vector stores.
        keywords: Optional keywords produced by the offline indexing pipeline.
    """

    text: str
    metadata: ChunkMetadata
    embedding: tuple[float, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RetrievedChunk:
    """A candidate chunk returned by retrieval.

    Args:
        text: Chunk text.
        metadata: Chunk metadata.
        vector_score: Raw vector search score, if present.
        keyword_score: Raw keyword search score, if present.
        fused_score: Hybrid fusion score.
        rerank_score: Reranker score.
    """

    text: str
    metadata: ChunkMetadata
    vector_score: float | None = None
    keyword_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        """Return the stable chunk identifier."""
        return self.metadata.chunk_id

    def with_scores(
        self,
        *,
        vector_score: float | None = None,
        keyword_score: float | None = None,
        fused_score: float | None = None,
        rerank_score: float | None = None,
    ) -> RetrievedChunk:
        """Return a copy with updated score fields.

        Args:
            vector_score: Replacement vector score.
            keyword_score: Replacement keyword score.
            fused_score: Replacement fused score.
            rerank_score: Replacement rerank score.

        Returns:
            New `RetrievedChunk` instance.
        """
        return replace(
            self,
            vector_score=self.vector_score if vector_score is None else vector_score,
            keyword_score=self.keyword_score if keyword_score is None else keyword_score,
            fused_score=self.fused_score if fused_score is None else fused_score,
            rerank_score=self.rerank_score if rerank_score is None else rerank_score,
        )

    def with_text(self, text: str) -> RetrievedChunk:
        """Return a copy with replacement text.

        Args:
            text: Hydrated chunk text.

        Returns:
            New `RetrievedChunk` instance.
        """
        return replace(self, text=text)


class ChunkHydrator(Protocol):
    """Protocol implemented by stores that hydrate chunk text by id."""

    def hydrate(self, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Hydrate missing chunk text.

        Args:
            candidates: Fused retrieval candidates.

        Returns:
            Candidates with text filled when available.
        """
        ...


class VectorSearcher(Protocol):
    """Protocol implemented by semantic vector stores such as Milvus."""

    def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Search for semantically similar chunks.

        Args:
            query: Rewritten query text.
            scope: Tenant and authorization filters.
            limit: Maximum number of candidates to return.

        Returns:
            Ranked vector candidates.
        """
        ...


class KeywordSearcher(Protocol):
    """Protocol implemented by keyword stores such as Elasticsearch."""

    def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Search for keyword matches.

        Args:
            query: Rewritten query text.
            scope: Tenant and authorization filters.
            limit: Maximum number of candidates to return.

        Returns:
            Ranked keyword candidates.
        """
        ...


class Reranker(Protocol):
    """Protocol implemented by cross-encoders or API rerankers."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Rerank hybrid retrieval candidates.

        Args:
            query: Original user query.
            candidates: Candidate chunks after hybrid fusion.
            limit: Maximum number of final chunks to return.

        Returns:
            Final ranked chunks.
        """
        ...
