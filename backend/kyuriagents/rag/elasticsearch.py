"""Elasticsearch keyword retrieval adapter."""

from __future__ import annotations

from typing import Protocol, cast

from kyuriagents.rag.metadata import ChunkMetadata, RetrievalScope, Visibility
from kyuriagents.rag.types import RetrievedChunk

try:
    from elasticsearch import Elasticsearch as _Elasticsearch
except ImportError:
    _Elasticsearch = None


class _ElasticsearchClient(Protocol):
    def search(self, **kwargs: object) -> dict[str, object]:
        """Run an Elasticsearch search request."""
        ...


_DEFAULT_TEXT_FIELDS = ("title^2", "section_path^2", "chunk_text", "summary", "keywords")
_UNOWNED_USER_ID = ""


class ElasticsearchKeywordStore:
    """Keyword retrieval adapter backed by Elasticsearch.

    The `elasticsearch` Python package is optional. Pass an already-created
    client in tests, or install `kyuriagents[rag]` for the production client.
    """

    def __init__(
        self,
        *,
        index: str,
        url: str = "http://localhost:9200",
        client: _ElasticsearchClient | None = None,
        text_fields: tuple[str, ...] = _DEFAULT_TEXT_FIELDS,
    ) -> None:
        """Initialize the keyword store.

        Args:
            index: Elasticsearch index name.
            url: Elasticsearch URL.
            client: Optional preconfigured Elasticsearch-compatible client.
            text_fields: Fields used by the `multi_match` keyword query.
        """
        self._index = index
        self._client = client if client is not None else _build_client(url)
        self._text_fields = text_fields

    def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Search Elasticsearch for keyword candidates.

        Args:
            query: Rewritten query text.
            scope: Tenant and authorization filters.
            limit: Maximum number of candidates.

        Returns:
            Ranked candidates with `keyword_score` populated.
        """
        if limit <= 0 or not query.strip():
            return []
        response = self._client.search(
            index=self._index,
            size=limit,
            query=_build_query(query, scope, self._text_fields),
            source=True,
        )
        hits_root = cast("dict[str, object]", response.get("hits", {}))
        hits = cast("list[dict[str, object]]", hits_root.get("hits", []))
        return [_hit_to_retrieved_chunk(hit) for hit in hits]


def _build_client(url: str) -> _ElasticsearchClient:
    if _Elasticsearch is None:
        msg = "Install `kyuriagents[rag]` or pass an Elasticsearch client."
        raise ImportError(msg)
    return cast("_ElasticsearchClient", _Elasticsearch(url))


def _build_query(
    query: str,
    scope: RetrievalScope,
    text_fields: tuple[str, ...],
) -> dict[str, object]:
    filters: list[dict[str, object]] = [{"term": {"tenant_id": scope.tenant_id}}]
    if scope.active_only:
        filters.append({"term": {"is_active": True}})
    if scope.kb_ids:
        filters.append({"terms": {"kb_id": list(scope.kb_ids)}})
    if scope.doc_ids:
        filters.append({"terms": {"doc_id": list(scope.doc_ids)}})
    if scope.languages:
        filters.append({"terms": {"language": list(scope.languages)}})
    if scope.source_types:
        filters.append({"terms": {"source_type": list(scope.source_types)}})
    if scope.visibility is not None:
        filters.append({"term": {"visibility": scope.visibility}})
    if scope.tags:
        filters.extend({"term": {"tags": tag}} for tag in scope.tags)
    filters.append(_user_filter(scope.user_id))
    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": list(text_fields),
                        "type": "best_fields",
                    }
                }
            ],
            "filter": filters,
        }
    }


def _user_filter(user_id: str | None) -> dict[str, object]:
    if user_id is None:
        return {"term": {"user_id": _UNOWNED_USER_ID}}
    return {
        "bool": {
            "should": [
                {"term": {"user_id": _UNOWNED_USER_ID}},
                {"term": {"user_id": user_id}},
            ],
            "minimum_should_match": 1,
        }
    }


def _hit_to_retrieved_chunk(hit: dict[str, object]) -> RetrievedChunk:
    source = cast("dict[str, object]", hit.get("_source", {}))
    return RetrievedChunk(
        text=str(source.get("chunk_text", "")),
        metadata=_metadata_from_source(source),
        keyword_score=_float_or_none(hit.get("_score")),
    )


def _metadata_from_source(source: dict[str, object]) -> ChunkMetadata:
    tags = source.get("tags", ())
    user_id = source.get("user_id") or None
    return ChunkMetadata(
        chunk_id=str(source["chunk_id"]),
        tenant_id=str(source["tenant_id"]),
        user_id=str(user_id) if user_id is not None else None,
        kb_id=str(source["kb_id"]),
        doc_id=str(source["doc_id"]),
        doc_version=str(source["doc_version"]),
        chunk_index=int(source["chunk_index"]),
        content_hash=str(source["content_hash"]),
        source_type=str(source["source_type"]),
        source_uri=str(source["source_uri"]),
        title=str(source.get("title", "")),
        section_path=str(source.get("section_path", "")),
        page_start=_optional_int(source.get("page_start")),
        page_end=_optional_int(source.get("page_end")),
        char_start=_optional_int(source.get("char_start")),
        char_end=_optional_int(source.get("char_end")),
        language=str(source.get("language", "unknown")),
        tags=tuple(str(tag) for tag in tags),
        visibility=_visibility(source.get("visibility")),
        created_at=str(source.get("created_at", "")),
        updated_at=str(source.get("updated_at", "")),
        embedding_model=str(source.get("embedding_model", "")),
        embedding_version=str(source.get("embedding_version", "")),
        schema_version=int(source.get("schema_version", 1)),
        is_active=bool(source.get("is_active", True)),
    )


def _visibility(value: object) -> Visibility:
    if value in ("private", "team", "public"):
        return value
    return "private"


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
