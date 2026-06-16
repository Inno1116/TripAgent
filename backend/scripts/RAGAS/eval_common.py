"""Shared helpers for the Beijing tourism RAG evaluation scripts."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_RAGAS_DIR = Path(__file__).resolve().parent
_DEFAULT_EVAL_FILE = Path(__file__).resolve().parent / "tourism_eval.jsonl"
_DEFAULT_EMBEDDING_FILE = Path(__file__).resolve().parent / "tourism_query_embeddings.jsonl"
_DEFAULT_RESULTS_FILE = Path(__file__).resolve().parent / "tourism_retrieval_results.jsonl"


@dataclass(frozen=True, kw_only=True)
class TourismEvalCase:
    """One manually annotated tourism RAG evaluation case."""

    id: str
    type: str
    question: str
    ground_truth_answer: str
    relevant_doc_ids: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...]
    difficulty: str
    raw: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class QueryEmbeddingRecord:
    """Cached embedding for one evaluation query."""

    id: str
    query: str
    normalized_query: str
    embedding_model: str
    embedding_version: str
    dimensions: int
    embedding: tuple[float, ...]


def default_eval_file() -> Path:
    """Return the default tourism eval JSONL path."""
    return _DEFAULT_EVAL_FILE


def default_embedding_file() -> Path:
    """Return the default cached query embedding JSONL path."""
    return _DEFAULT_EMBEDDING_FILE


def default_results_file() -> Path:
    """Return the default retrieval result JSONL path."""
    return _DEFAULT_RESULTS_FILE


def load_runtime_env() -> None:
    """Load evaluation environment values.

    `RAGAS.env` is intentionally loaded last and overwrites process values so
    offline evaluation can use a separate model/key from the running service.
    """
    for env_path in (_ROOT / "runtime.env", _ROOT / "kyuriagents" / "runtime" / "runtime.env"):
        if not env_path.exists():
            continue
        _load_env_file(env_path, override=False)
    ragas_env = _RAGAS_DIR / "RAGAS.env"
    if ragas_env.exists():
        _load_env_file(ragas_env, override=True)


def _load_env_file(path: Path, *, override: bool) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        key = name.strip()
        value = value.strip()
        if not value:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def load_eval_cases(path: Path | str = _DEFAULT_EVAL_FILE) -> list[TourismEvalCase]:
    """Load manually annotated tourism RAG eval cases from JSONL."""
    eval_path = Path(path)
    cases: list[TourismEvalCase] = []
    for line_number, line in enumerate(eval_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        relevant_chunks = raw.get("relevant_chunks")
        if not isinstance(relevant_chunks, list) or not relevant_chunks:
            msg = f"{eval_path}:{line_number} has no relevant_chunks"
            raise ValueError(msg)
        chunk_ids: list[str] = []
        for chunk in relevant_chunks:
            if not isinstance(chunk, dict):
                msg = f"{eval_path}:{line_number} relevant_chunks must contain objects"
                raise ValueError(msg)
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                msg = f"{eval_path}:{line_number} relevant chunk is missing chunk_id"
                raise ValueError(msg)
            chunk_ids.append(chunk_id)
        cases.append(
            TourismEvalCase(
                id=str(raw["id"]),
                type=str(raw["type"]),
                question=str(raw["question"]),
                ground_truth_answer=str(raw["ground_truth_answer"]),
                relevant_doc_ids=tuple(str(value) for value in raw.get("relevant_doc_ids", ())),
                relevant_chunk_ids=tuple(dict.fromkeys(chunk_ids)),
                difficulty=str(raw["difficulty"]),
                raw=raw,
            )
        )
    return cases


def load_query_embeddings(path: Path | str = _DEFAULT_EMBEDDING_FILE) -> dict[str, QueryEmbeddingRecord]:
    """Load cached query embeddings keyed by normalized query."""
    embedding_path = Path(path)
    records: dict[str, QueryEmbeddingRecord] = {}
    if not embedding_path.exists():
        return records
    for line_number, line in enumerate(embedding_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        embedding = tuple(float(value) for value in raw.get("embedding", ()))
        normalized_query = normalize_query(str(raw.get("normalized_query") or raw.get("query") or ""))
        if not normalized_query or not embedding:
            msg = f"{embedding_path}:{line_number} has invalid cached embedding"
            raise ValueError(msg)
        records[normalized_query] = QueryEmbeddingRecord(
            id=str(raw.get("id") or ""),
            query=str(raw.get("query") or ""),
            normalized_query=normalized_query,
            embedding_model=str(raw.get("embedding_model") or ""),
            embedding_version=str(raw.get("embedding_version") or ""),
            dimensions=int(raw.get("dimensions") or len(embedding)),
            embedding=embedding,
        )
    return records


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSON-serializable rows as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def normalize_query(query: str) -> str:
    """Normalize query text the same way the identity query rewriter does."""
    return " ".join(query.split())


def embedding_version(*, model: str, dimensions: int | None) -> str:
    """Return the embedding version string used by ingestion metadata."""
    if dimensions is None:
        return model
    return f"{model}:{dimensions}"


def infer_scope_from_kb(*, dsn: str, kb_id: str) -> tuple[str, str | None]:
    """Infer tenant and owner user id for a knowledge base."""
    try:
        import psycopg  # noqa: PLC0415
        from psycopg.rows import dict_row  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - runtime dependency guard.
        msg = "Install psycopg to infer evaluation scope from PostgreSQL."
        raise ImportError(msg) from exc
    with psycopg.connect(dsn, row_factory=dict_row) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT tenant_id, owner_user_id
            FROM rag_knowledge_bases
            WHERE kb_id = %s
            """,
            (kb_id,),
        )
        row = cursor.fetchone()
    if row is None:
        msg = f"Knowledge base not found: {kb_id}"
        raise ValueError(msg)
    return str(row["tenant_id"]), str(row["owner_user_id"]) if row.get("owner_user_id") else None


def score_ranking(retrieved_chunk_ids: Sequence[str], gold_chunk_ids: Sequence[str], *, k: int) -> dict[str, float]:
    """Compute recall, MRR, and NDCG for one ranked chunk list."""
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    gold = tuple(dict.fromkeys(gold_chunk_ids))
    if not gold:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    top = tuple(retrieved_chunk_ids[:k])
    gold_set = set(gold)
    hits = [chunk_id for chunk_id in top if chunk_id in gold_set]
    recall = len(set(hits)) / float(len(gold_set))
    mrr = 0.0
    for rank, chunk_id in enumerate(top, start=1):
        if chunk_id in gold_set:
            mrr = 1.0 / float(rank)
            break
    dcg = 0.0
    for rank, chunk_id in enumerate(top, start=1):
        if chunk_id in gold_set:
            dcg += 1.0 / _log2(rank + 1)
    ideal_hits = min(len(gold_set), k)
    ideal_dcg = sum(1.0 / _log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def aggregate_metric_dicts(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries."""
    if not values:
        return {}
    keys = sorted({key for item in values for key in item})
    return {key: sum(float(item.get(key, 0.0)) for item in values) / float(len(values)) for key in keys}


def _log2(value: int) -> float:
    # tiny helper avoids importing math in hot scoring loops from multiple files.
    import math

    return math.log2(value)
