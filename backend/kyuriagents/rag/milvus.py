"""Milvus vector retrieval adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

from kyuriagents.rag.metadata import ChunkMetadata, RetrievalScope, Visibility
from kyuriagents.rag.types import RetrievedChunk

EmbeddingFunction = Callable[[str], Sequence[float]]

try:
    from pymilvus import MilvusClient as _MilvusClient
except ImportError:
    _MilvusClient = None


class _MilvusSearchClient(Protocol):
    def search(self, **kwargs: object) -> list[list[object]]:
        """Run a Milvus vector search request."""
        ...


_DEFAULT_OUTPUT_FIELDS = (
    "chunk_id",
    "tenant_id",
    "user_id",
    "kb_id",
    "doc_id",
    "doc_version",
    "chunk_index",
    "content_hash",
    "source_type",
    "source_uri",
    "title",
    "section_path",
    "page_start",
    "page_end",
    "char_start",
    "char_end",
    "language",
    "tags",
    "visibility",
    "created_at",
    "updated_at",
    "embedding_model",
    "embedding_version",
    "schema_version",
    "is_active",
)
_UNOWNED_USER_ID = ""


class MilvusVectorStore:
    """Semantic retrieval adapter backed by Milvus."""

    def __init__(
        self,
        *,
        collection_name: str,
        embed_query: EmbeddingFunction,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        db_name: str | None = None,
        client: _MilvusSearchClient | None = None,
        output_fields: tuple[str, ...] = _DEFAULT_OUTPUT_FIELDS,
        search_params: dict[str, object] | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Milvus collection name.
            embed_query: Query embedding function.
            uri: Milvus URI. `http://localhost:19530` works with `MilvusClient`.
            token: Optional Milvus token.
            db_name: Optional Milvus database name.
            client: Optional preconfigured Milvus-compatible client.
            output_fields: Scalar fields returned with each hit.
            search_params: Optional Milvus search parameters.
        """
        self._collection_name = collection_name
        self._embed_query = embed_query
        self._client = client if client is not None else _build_client(uri, token, db_name)
        self._output_fields = output_fields
        self._search_params = search_params

    def search(
        self,
        query: str,
        *,
        scope: RetrievalScope,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Search Milvus for vector candidates.

        Args:
            query: Rewritten query text.
            scope: Tenant and authorization filters.
            limit: Maximum number of candidates.

        Returns:
            Ranked candidates with `vector_score` populated.
        """
        if limit <= 0 or not query.strip():
            return []
        embedding = [float(value) for value in self._embed_query(query)]
        if not embedding:
            return []
        kwargs: dict[str, object] = {
            "collection_name": self._collection_name,
            "data": [embedding],
            "filter": build_milvus_filter(scope),
            "limit": limit,
            "output_fields": list(self._output_fields),
        }
        if self._search_params is not None:
            kwargs["search_params"] = self._search_params
        raw = self._client.search(**kwargs)
        hits = raw[0] if raw else []
        return [_hit_to_retrieved_chunk(hit) for hit in hits]


def build_milvus_filter(scope: RetrievalScope) -> str:
    """Build a Milvus scalar filter expression from retrieval scope."""
    filters = [f'tenant_id == "{_escape(scope.tenant_id)}"']
    if scope.active_only:
        filters.append("is_active == true")
    if scope.kb_ids:
        filters.append(_in_filter("kb_id", scope.kb_ids))
    if scope.doc_ids:
        filters.append(_in_filter("doc_id", scope.doc_ids))
    if scope.languages:
        filters.append(_in_filter("language", scope.languages))
    if scope.source_types:
        filters.append(_in_filter("source_type", scope.source_types))
    if scope.visibility is not None:
        filters.append(f'visibility == "{_escape(scope.visibility)}"')
    if scope.tags:
        filters.extend(f'ARRAY_CONTAINS(tags, "{_escape(tag)}")' for tag in scope.tags)
    filters.append(_user_filter(scope.user_id))
    return " and ".join(filters)


def _build_client(uri: str, token: str | None, db_name: str | None) -> _MilvusSearchClient:
    if _MilvusClient is None:
        msg = "Install `kyuriagents[rag]` or pass a Milvus client."
        raise ImportError(msg)
    kwargs: dict[str, object] = {"uri": uri}
    if token is not None:
        kwargs["token"] = token
    if db_name is not None:
        kwargs["db_name"] = db_name
    return cast("_MilvusSearchClient", _MilvusClient(**kwargs))


def _hit_to_retrieved_chunk(hit: object) -> RetrievedChunk:
    entity = _extract_entity(hit)
    return RetrievedChunk(
        text=str(entity.get("chunk_text", "")),
        metadata=_metadata_from_entity(entity),
        vector_score=_extract_score(hit),
    )


def _extract_entity(hit: object) -> dict[str, object]:
    if isinstance(hit, dict):
        entity = hit.get("entity") or hit.get("fields") or hit
        return cast("dict[str, object]", dict(entity))
    entity = getattr(hit, "entity", None)
    if isinstance(entity, dict):
        return cast("dict[str, object]", dict(entity))
    get = getattr(hit, "get", None)
    if callable(get):
        raw = get("entity") or get("fields") or {}
        return cast("dict[str, object]", dict(raw))
    return {}


def _extract_score(hit: object) -> float | None:
    get = hit.get if isinstance(hit, dict) else getattr(hit, "get", None)
    value = get("distance", get("score")) if callable(get) else getattr(hit, "distance", getattr(hit, "score", None))
    if value is None:
        return None
    return float(value)


def _metadata_from_entity(entity: dict[str, object]) -> ChunkMetadata:
    tags = entity.get("tags", ())
    user_id = entity.get("user_id") or None
    return ChunkMetadata(
        chunk_id=str(entity["chunk_id"]),
        tenant_id=str(entity["tenant_id"]),
        user_id=str(user_id) if user_id is not None else None,
        kb_id=str(entity["kb_id"]),
        doc_id=str(entity["doc_id"]),
        doc_version=str(entity["doc_version"]),
        chunk_index=int(entity["chunk_index"]),
        content_hash=str(entity["content_hash"]),
        source_type=str(entity["source_type"]),
        source_uri=str(entity["source_uri"]),
        title=str(entity.get("title", "")),
        section_path=str(entity.get("section_path", "")),
        page_start=_optional_int(entity.get("page_start")),
        page_end=_optional_int(entity.get("page_end")),
        char_start=_optional_int(entity.get("char_start")),
        char_end=_optional_int(entity.get("char_end")),
        language=str(entity.get("language", "unknown")),
        tags=tuple(str(tag) for tag in tags),
        visibility=_visibility(entity.get("visibility")),
        created_at=str(entity.get("created_at", "")),
        updated_at=str(entity.get("updated_at", "")),
        embedding_model=str(entity.get("embedding_model", "")),
        embedding_version=str(entity.get("embedding_version", "")),
        schema_version=int(entity.get("schema_version", 1)),
        is_active=bool(entity.get("is_active", True)),
    )


def _visibility(value: object) -> Visibility:
    if value in ("private", "team", "public"):
        return value
    return "private"


def _optional_int(value: object) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _user_filter(user_id: str | None) -> str:
    if user_id is None:
        return f'user_id == "{_UNOWNED_USER_ID}"'
    return f'(user_id == "{_UNOWNED_USER_ID}" or user_id == "{_escape(user_id)}")'


def _in_filter(field: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f'"{_escape(value)}"' for value in values)
    return f"{field} in [{quoted}]"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
