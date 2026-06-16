"""Run retrieval-only evaluation for the Beijing tourism RAG test set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from eval_common import (
    aggregate_metric_dicts,
    default_embedding_file,
    default_eval_file,
    default_results_file,
    infer_scope_from_kb,
    load_eval_cases,
    load_query_embeddings,
    load_runtime_env,
    normalize_query,
    score_ranking,
    write_jsonl,
)
from kyuriagents.rag import (
    DashScopeTextReranker,
    ElasticsearchKeywordStore,
    FusedScoreReranker,
    HybridRAGRetriever,
    HybridSearchConfig,
    MilvusVectorStore,
    PostgresChunkTextHydrator,
    RetrievalScope,
)
from kyuriagents.rag.types import RetrievedChunk
from kyuriagents.runtime import AgentRuntimeConfig

Mode = Literal["vector", "hybrid", "hybrid_rerank"]
_MODES: tuple[Mode, ...] = ("vector", "hybrid", "hybrid_rerank")


class CachedQueryEmbedder:
    """Embedding function backed by a JSONL cache."""

    def __init__(self, embeddings: Mapping[str, Sequence[float]]) -> None:
        """Initialize the cached embedder."""
        self._embeddings = {normalize_query(key): tuple(float(value) for value in values) for key, values in embeddings.items()}

    def __call__(self, query: str) -> tuple[float, ...]:
        """Return a cached embedding for a query."""
        normalized = normalize_query(query)
        embedding = self._embeddings.get(normalized)
        if embedding is None:
            msg = f"Missing cached query embedding for: {normalized!r}. Run prepare_query_embeddings.py first."
            raise KeyError(msg)
        return embedding


def main() -> None:
    """Run retrieval evaluation against Milvus and Elasticsearch."""
    parser = argparse.ArgumentParser(description="Evaluate tourism RAG retrieval modes.")
    parser.add_argument("--eval-file", type=Path, default=default_eval_file(), help="Annotated tourism eval JSONL.")
    parser.add_argument("--embeddings-file", type=Path, default=default_embedding_file(), help="Cached query embedding JSONL.")
    parser.add_argument("--output", type=Path, default=default_results_file(), help="Per-question retrieval result JSONL.")
    parser.add_argument("--kb-id", default="", help="Knowledge base id. Defaults to RAG_KB_IDS or inferred from eval setup.")
    parser.add_argument("--tenant-id", default="", help="Tenant id override.")
    parser.add_argument("--user-id", default="", help="User id override; needed for private uploaded documents.")
    parser.add_argument("--top-k", type=int, default=5, help="Final Top-K used for metrics.")
    parser.add_argument("--candidate-k", type=int, default=40, help="Vector/keyword candidates before fusion.")
    parser.add_argument("--rerank-candidates", type=int, default=40, help="Hybrid candidates sent to reranker.")
    parser.add_argument(
        "--modes",
        default=",".join(_MODES),
        help="Comma-separated modes: vector,hybrid,hybrid_rerank.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        msg = "--top-k must be positive"
        raise ValueError(msg)

    load_runtime_env()
    config = AgentRuntimeConfig.from_env()
    cases = load_eval_cases(args.eval_file)
    embedding_records = load_query_embeddings(args.embeddings_file)
    cached_embeddings = {record.normalized_query: record.embedding for record in embedding_records.values()}
    embed_query = CachedQueryEmbedder(cached_embeddings)

    kb_id = args.kb_id or (config.rag_kb_ids[0] if config.rag_kb_ids else "")
    if not kb_id:
        msg = "Set --kb-id or RAG_KB_IDS before running tourism retrieval evaluation."
        raise ValueError(msg)
    tenant_id = args.tenant_id
    user_id = args.user_id
    if config.postgres_dsn and (not tenant_id or not user_id):
        inferred_tenant, inferred_user = infer_scope_from_kb(dsn=config.postgres_dsn, kb_id=kb_id)
        tenant_id = tenant_id or inferred_tenant
        user_id = user_id or inferred_user
    tenant_id = tenant_id or config.tenant_id
    user_id = user_id or config.user_id

    scope = RetrievalScope(tenant_id=tenant_id, user_id=user_id, kb_ids=(kb_id,))
    hydrator = PostgresChunkTextHydrator(dsn=config.postgres_dsn) if config.postgres_dsn else None
    vector_store = MilvusVectorStore(
        collection_name=config.rag_milvus_collection,
        uri=config.rag_milvus_uri,
        token=config.rag_milvus_token,
        db_name=config.rag_milvus_db,
        embed_query=embed_query,
    )
    keyword_store = ElasticsearchKeywordStore(index=config.rag_es_index, url=config.rag_es_url)

    selected_modes = _parse_modes(args.modes)
    rows: list[dict[str, object]] = []
    for mode in selected_modes:
        retriever = _retriever_for_mode(
            mode,
            config=config,
            vector_store=vector_store,
            keyword_store=keyword_store,
            hydrator=hydrator,
            candidate_k=args.candidate_k,
            rerank_candidates=args.rerank_candidates,
            top_k=args.top_k,
        )
        for case in cases:
            retrieved = _retrieve(
                mode,
                retriever=retriever,
                vector_store=vector_store,
                hydrator=hydrator,
                case_query=case.question,
                scope=scope,
                top_k=args.top_k,
            )
            retrieved_ids = [chunk.chunk_id for chunk in retrieved]
            metrics = _metrics_for(retrieved_ids, case.relevant_chunk_ids, k=args.top_k)
            rows.append(
                {
                    "mode": mode,
                    "id": case.id,
                    "type": case.type,
                    "difficulty": case.difficulty,
                    "question": case.question,
                    "gold_chunk_ids": list(case.relevant_chunk_ids),
                    "retrieved": [_retrieved_row(chunk, rank=rank) for rank, chunk in enumerate(retrieved, start=1)],
                    "metrics": metrics,
                }
            )

    write_jsonl(args.output, rows)
    summary = _summary(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


def _retriever_for_mode(
    mode: Mode,
    *,
    config: AgentRuntimeConfig,
    vector_store: MilvusVectorStore,
    keyword_store: ElasticsearchKeywordStore,
    hydrator: PostgresChunkTextHydrator | None,
    candidate_k: int,
    rerank_candidates: int,
    top_k: int,
) -> HybridRAGRetriever | None:
    if mode == "vector":
        return None
    if mode == "hybrid_rerank" and config.dashscope_api_key and config.rag_rerank_model:
        reranker = DashScopeTextReranker(
            api_key=config.dashscope_api_key,
            model=config.rag_rerank_model,
            endpoint=config.rag_rerank_url,
            timeout_seconds=config.rag_rerank_timeout_seconds,
        )
    else:
        reranker = FusedScoreReranker()
    return HybridRAGRetriever(
        vector_searcher=vector_store,
        keyword_searcher=keyword_store,
        reranker=reranker,
        chunk_hydrator=hydrator,
        config=HybridSearchConfig(
            vector_candidates=candidate_k,
            keyword_candidates=candidate_k,
            rerank_candidates=rerank_candidates,
            top_k=top_k,
        ),
    )


def _retrieve(
    mode: Mode,
    *,
    retriever: HybridRAGRetriever | None,
    vector_store: MilvusVectorStore,
    hydrator: PostgresChunkTextHydrator | None,
    case_query: str,
    scope: RetrievalScope,
    top_k: int,
) -> list[RetrievedChunk]:
    if mode == "vector":
        chunks = vector_store.search(case_query, scope=scope, limit=top_k)
        return hydrator.hydrate(chunks) if hydrator is not None else chunks
    if retriever is None:
        msg = f"Mode {mode} requires a retriever."
        raise ValueError(msg)
    return retriever.retrieve(case_query, scope=scope, top_k=top_k)


def _metrics_for(retrieved_ids: Sequence[str], gold_ids: Sequence[str], *, k: int) -> dict[str, float]:
    metrics_at_k = score_ranking(retrieved_ids, gold_ids, k=k)
    metrics_at_3 = score_ranking(retrieved_ids, gold_ids, k=min(3, k))
    return {
        "recall_at_3": metrics_at_3["recall"],
        "recall_at_k": metrics_at_k["recall"],
        "mrr_at_k": metrics_at_k["mrr"],
        "ndcg_at_k": metrics_at_k["ndcg"],
    }


def _retrieved_row(chunk: RetrievedChunk, *, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.metadata.title,
        "chunk_index": chunk.metadata.chunk_index,
        "vector_score": chunk.vector_score,
        "keyword_score": chunk.keyword_score,
        "fused_score": chunk.fused_score,
        "rerank_score": chunk.rerank_score,
        "text": chunk.text,
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    grouped_type: dict[tuple[str, str], list[Mapping[str, float]]] = defaultdict(list)
    for row in rows:
        mode = str(row["mode"])
        question_type = str(row["type"])
        metrics = row["metrics"]
        if not isinstance(metrics, dict):
            continue
        metric_values = {str(key): float(value) for key, value in metrics.items()}
        grouped[mode].append(metric_values)
        grouped_type[(mode, question_type)].append(metric_values)
    return {
        "count": len(rows),
        "overall": {mode: aggregate_metric_dicts(values) for mode, values in sorted(grouped.items())},
        "by_type": {
            mode: {
                question_type: aggregate_metric_dicts(grouped_type[(mode, question_type)])
                for question_type in sorted({key_type for key_mode, key_type in grouped_type if key_mode == mode})
            }
            for mode in sorted(grouped)
        },
    }


def _parse_modes(raw: str) -> tuple[Mode, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    invalid = sorted(set(values) - set(_MODES))
    if invalid:
        msg = f"Invalid modes: {invalid}. Allowed: {_MODES}"
        raise ValueError(msg)
    return values  # type: ignore[return-value]


if __name__ == "__main__":
    main()
