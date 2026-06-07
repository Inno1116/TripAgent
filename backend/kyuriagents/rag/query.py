"""Query rewriting contracts for online RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, kw_only=True)
class QueryRewrite:
    """A rewritten user query and optional retrieval expansions.

    Args:
        original_query: Raw user query.
        rewritten_query: Main query used for reranking and display.
        search_queries: Queries sent to keyword and vector stores.
    """

    original_query: str
    rewritten_query: str
    search_queries: tuple[str, ...]


class QueryRewriter(Protocol):
    """Protocol for components that rewrite user queries before retrieval."""

    def rewrite(
        self,
        query: str,
        *,
        history: Sequence[str] = (),
    ) -> QueryRewrite:
        """Rewrite a query for retrieval.

        Args:
            query: User query.
            history: Optional recent conversation turns.

        Returns:
            Rewritten query bundle.
        """
        ...


class IdentityQueryRewriter:
    """Minimal query rewriter used when no LLM rewrite service is configured."""

    def __init__(self, aliases: Mapping[str, Sequence[str]] | None = None) -> None:
        """Initialize the rewriter.

        Args:
            aliases: Optional term expansions. For example, `{"auth":
                ["authentication", "login"]}` adds expanded search queries when
                the original query contains `auth`.
        """
        self._aliases = {key.lower(): tuple(values) for key, values in (aliases or {}).items()}

    def rewrite(
        self,
        query: str,
        *,
        history: Sequence[str] = (),
    ) -> QueryRewrite:
        """Normalize whitespace and apply deterministic alias expansion.

        Args:
            query: User query.
            history: Optional recent conversation turns. The identity rewriter
                accepts this for API compatibility but does not use it.

        Returns:
            Rewritten query bundle.
        """
        del history
        rewritten = " ".join(query.split())
        expansions = self._expand_aliases(rewritten)
        search_queries = (rewritten, *expansions) if rewritten else ()
        return QueryRewrite(
            original_query=query,
            rewritten_query=rewritten,
            search_queries=search_queries,
        )

    def _expand_aliases(self, query: str) -> tuple[str, ...]:
        words = {word.lower() for word in query.split()}
        expansions: list[str] = []
        for key, values in self._aliases.items():
            if key not in words:
                continue
            expansions.extend(query.replace(key, value) for value in values)
        return tuple(dict.fromkeys(expansions))
