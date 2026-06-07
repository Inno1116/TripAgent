"""In-memory retrieval stores for local development and unit tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from kyuriagents.rag._text import cosine_similarity, tokenize
from kyuriagents.rag.types import RetrievedChunk

if TYPE_CHECKING:
    from kyuriagents.rag.metadata import RetrievalScope
    from kyuriagents.rag.types import DocumentChunk

EmbeddingFunction = Callable[[str], Sequence[float]]
"""Callable that embeds a query string into a vector."""


class InMemoryVectorStore:
    """Small vector store useful before Milvus is installed.

    This store is deterministic and dependency-free. Production deployments
    should provide a `VectorSearcher` adapter backed by Milvus.
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        embed_query: EmbeddingFunction,
    ) -> None:
        """Initialize the vector store.

        Args:
            chunks: Indexed chunks with precomputed embeddings.
            embed_query: Query embedding function.
        """
        self._chunks = tuple(chunks)
        self._embed_query = embed_query

    def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Search chunks by cosine similarity.

        Args:
            query: Rewritten query text.
            scope: Tenant and authorization filters.
            limit: Maximum number of candidates.

        Returns:
            Ranked candidates with `vector_score` populated.
        """
        if limit <= 0:
            return []
        query_embedding = tuple(float(value) for value in self._embed_query(query))
        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            if not chunk.embedding or not scope.matches(chunk.metadata):
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            if score <= 0.0:
                continue
            scored.append(
                RetrievedChunk(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    vector_score=score,
                )
            )
        return sorted(scored, key=lambda item: (-_score_or_zero(item.vector_score), item.chunk_id))[:limit]


class InMemoryKeywordStore:
    """Small keyword store useful before Elasticsearch is installed."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        """Initialize the keyword store.

        Args:
            chunks: Indexed chunks.
        """
        self._chunks = tuple(chunks)

    def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Search chunks by lexical overlap.

        Args:
            query: Rewritten query text.
            scope: Tenant and authorization filters.
            limit: Maximum number of candidates.

        Returns:
            Ranked candidates with `keyword_score` populated.
        """
        if limit <= 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scored: list[RetrievedChunk] = []
        for chunk in self._chunks:
            if not scope.matches(chunk.metadata):
                continue
            score = self._score_chunk(query_terms, chunk)
            if score <= 0.0:
                continue
            scored.append(
                RetrievedChunk(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    keyword_score=score,
                )
            )
        return sorted(scored, key=lambda item: (-_score_or_zero(item.keyword_score), item.chunk_id))[:limit]

    @staticmethod
    def _score_chunk(query_terms: tuple[str, ...], chunk: DocumentChunk) -> float:
        searchable = " ".join(
            (
                chunk.text,
                chunk.metadata.title,
                chunk.metadata.section_path,
                " ".join(chunk.keywords),
            )
        )
        counts = Counter(tokenize(searchable))
        unique_query_terms = tuple(dict.fromkeys(query_terms))
        overlap = sum(counts[term] for term in unique_query_terms)
        if overlap == 0:
            return 0.0
        return float(overlap) / float(len(unique_query_terms))


def _score_or_zero(score: float | None) -> float:
    return 0.0 if score is None else score
