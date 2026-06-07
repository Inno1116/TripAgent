"""StratRAG dataset loading and retrieval evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from kyuriagents.rag.hybrid import HybridRAGRetriever, HybridSearchConfig
from kyuriagents.rag.in_memory import EmbeddingFunction, InMemoryKeywordStore, InMemoryVectorStore
from kyuriagents.rag.metadata import ChunkMetadata, RetrievalScope
from kyuriagents.rag.types import DocumentChunk, RetrievedChunk

QuestionType = Literal["bridge", "comparison", "yes-no", "multi-hop"]
"""Question types defined by StratRAG."""


@dataclass(frozen=True, kw_only=True)
class StratRAGDocument:
    """One candidate document in a StratRAG row.

    Args:
        doc_id: Original dataset document identifier.
        text: Title and paragraph body.
        source: HotpotQA paragraph title.
    """

    doc_id: str
    text: str
    source: str


@dataclass(frozen=True, kw_only=True)
class StratRAGExample:
    """One StratRAG retrieval example.

    Args:
        example_id: Dataset example identifier.
        query: Multi-hop question.
        reference_answer: Ground-truth answer string.
        doc_pool: Candidate documents for this query.
        gold_doc_indices: Indices of gold documents inside `doc_pool`.
        split: Dataset split.
        question_type: StratRAG question type.
        created_at: Original dataset timestamp.
        provenance: Original dataset provenance metadata.
    """

    example_id: str
    query: str
    reference_answer: str
    doc_pool: tuple[StratRAGDocument, ...]
    gold_doc_indices: tuple[int, ...]
    split: str
    question_type: QuestionType
    created_at: str = ""
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def kb_id(self) -> str:
        """Return the per-example knowledge base id used for scoped retrieval.

        Returns:
            Knowledge base identifier for this example's candidate pool.
        """
        return f"stratrag:{self.split}:{self.example_id}"

    def gold_doc_ids(self) -> tuple[str, ...]:
        """Return normalized document identifiers for the gold documents.

        Returns:
            Normalized document identifiers for gold documents.
        """
        return tuple(_normalized_doc_id(self.example_id, self.doc_pool[index].doc_id) for index in self.gold_doc_indices)


@dataclass(frozen=True, kw_only=True)
class StratRAGExampleResult:
    """Retrieval metrics for one StratRAG example.

    Args:
        example_id: Dataset example identifier.
        question_type: StratRAG question type.
        retrieved_doc_ids: Ranked retrieved document identifiers.
        gold_doc_ids: Gold document identifiers.
        recall_at_1: Recall@1.
        recall_at_2: Recall@2.
        recall_at_5: Recall@5.
        mrr: Reciprocal rank of the first retrieved gold document.
        ndcg_at_5: NDCG@5 with binary document relevance.
    """

    example_id: str
    question_type: QuestionType
    retrieved_doc_ids: tuple[str, ...]
    gold_doc_ids: tuple[str, ...]
    recall_at_1: float
    recall_at_2: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float


@dataclass(frozen=True, kw_only=True)
class StratRAGAggregateResult:
    """Aggregated StratRAG retrieval metrics.

    Args:
        count: Number of evaluated examples.
        recall_at_1: Mean Recall@1.
        recall_at_2: Mean Recall@2.
        recall_at_5: Mean Recall@5.
        mrr: Mean reciprocal rank.
        ndcg_at_5: Mean NDCG@5.
    """

    count: int
    recall_at_1: float
    recall_at_2: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable metrics.

        Returns:
            Metrics dictionary suitable for JSON storage.
        """
        return {
            "count": self.count,
            "recall_at_1": self.recall_at_1,
            "recall_at_2": self.recall_at_2,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "ndcg_at_5": self.ndcg_at_5,
        }


@dataclass(frozen=True, kw_only=True)
class StratRAGEvaluation:
    """Complete StratRAG evaluation result.

    Args:
        overall: Aggregate metrics across all examples.
        by_question_type: Aggregate metrics grouped by question type.
        examples: Per-example metrics.
    """

    overall: StratRAGAggregateResult
    by_question_type: dict[QuestionType, StratRAGAggregateResult]
    examples: tuple[StratRAGExampleResult, ...]


def load_stratrag_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
    shuffle_docs: bool = False,
    seed: int = 42,
) -> list[StratRAGExample]:
    """Load StratRAG examples from local JSONL.

    Args:
        path: Path to `train.jsonl` or `val.jsonl`.
        limit: Optional maximum number of rows to load.
        shuffle_docs: Whether to shuffle each `doc_pool` and recompute gold
            indices. Useful for training or sanity checks against position bias.
        seed: Deterministic shuffle seed.

    Returns:
        Parsed examples.
    """
    examples: list[StratRAGExample] = []
    rng = random.Random(seed)  # noqa: S311  # deterministic dataset shuffling, not security-sensitive
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            examples.append(parse_stratrag_row(json.loads(line), shuffle_docs=shuffle_docs, rng=rng))
    return examples


def parse_stratrag_row(
    row: dict[str, object],
    *,
    shuffle_docs: bool = False,
    rng: random.Random | None = None,
) -> StratRAGExample:
    """Parse one StratRAG JSON row.

    Args:
        row: JSON object following the StratRAG README schema.
        shuffle_docs: Whether to shuffle `doc_pool`.
        rng: Random generator used when shuffling.

    Returns:
        Parsed example.
    """
    raw_pool = _object_list(row.get("doc_pool"), name="doc_pool")
    docs = tuple(
        StratRAGDocument(
            doc_id=str(item.get("doc_id", "")),
            text=str(item.get("text", "")),
            source=str(item.get("source", "")),
        )
        for item in raw_pool
    )
    gold_doc_indices = tuple(_int_value(index) for index in _object_sequence(row.get("gold_doc_indices"), name="gold_doc_indices"))
    if shuffle_docs:
        docs, gold_doc_indices = _shuffle_pool(
            docs,
            gold_doc_indices,
            rng=rng or random.Random(42),  # noqa: S311  # deterministic dataset shuffling, not security-sensitive
        )

    metadata = _object_dict(row.get("metadata"))
    return StratRAGExample(
        example_id=str(row.get("id", "")),
        query=str(row.get("query", "")),
        reference_answer=str(row.get("reference_answer", "")),
        doc_pool=docs,
        gold_doc_indices=gold_doc_indices,
        split=str(metadata.get("split", "")),
        question_type=_question_type(metadata.get("question_type")),
        created_at=str(row.get("created_at", "")),
        provenance=_object_dict(row.get("provenance")),
    )


def stratrag_chunks(
    examples: Sequence[StratRAGExample],
    *,
    tenant_id: str = "stratrag",
    embedding_model: str = "",
    embedding_version: str = "",
) -> list[DocumentChunk]:
    """Convert StratRAG examples into RAG chunks.

    Each StratRAG row becomes a separate knowledge base via `example.kb_id`.
    This preserves the official setting: the retriever must choose from that
    row's 15 candidate documents, not from the whole dataset.

    Args:
        examples: StratRAG examples.
        tenant_id: Tenant used for evaluation indexing.
        embedding_model: Embedding model identifier.
        embedding_version: Embedding pipeline version.

    Returns:
        RAG chunks ready for vector and keyword indexing.
    """
    chunks: list[DocumentChunk] = []
    for example in examples:
        gold_indices = set(example.gold_doc_indices)
        for index, doc in enumerate(example.doc_pool):
            content_hash = _hash_text(doc.text)
            normalized_doc_id = _normalized_doc_id(example.example_id, doc.doc_id)
            chunk_id = f"{example.example_id}:{index}:{content_hash[:16]}"
            tags = tuple(
                item
                for item in (
                    "stratrag",
                    example.split,
                    example.question_type,
                    "gold" if index in gold_indices else "distractor",
                )
                if item
            )
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                kb_id=example.kb_id,
                doc_id=normalized_doc_id,
                doc_version=f"{example.example_id}:{content_hash[:16]}:v1",
                chunk_index=index,
                content_hash=content_hash,
                source_type="stratrag",
                source_uri=f"stratrag://{example.split}/{example.example_id}/{doc.doc_id}",
                title=doc.source,
                section_path=example.question_type,
                language="en",
                tags=tags,
                visibility="public",
                created_at=example.created_at,
                updated_at=example.created_at,
                embedding_model=embedding_model,
                embedding_version=embedding_version,
            )
            chunks.append(
                DocumentChunk(
                    text=doc.text,
                    metadata=metadata,
                    keywords=tuple(token for token in (doc.source, example.question_type) if token),
                )
            )
    return chunks


def attach_embeddings(
    chunks: Sequence[DocumentChunk],
    *,
    embed_text: EmbeddingFunction,
) -> list[DocumentChunk]:
    """Return chunks with embeddings produced from their text.

    Args:
        chunks: Chunks to embed.
        embed_text: Embedding function.

    Returns:
        Chunks with `embedding` populated.
    """
    return [
        DocumentChunk(
            text=chunk.text,
            metadata=chunk.metadata,
            embedding=tuple(float(value) for value in embed_text(chunk.text)),
            keywords=chunk.keywords,
        )
        for chunk in chunks
    ]


def build_in_memory_stratrag_retriever(
    examples: Sequence[StratRAGExample],
    *,
    embed_text: EmbeddingFunction,
    tenant_id: str = "stratrag",
    config: HybridSearchConfig | None = None,
) -> HybridRAGRetriever:
    """Build an in-memory hybrid retriever for quick StratRAG experiments.

    Args:
        examples: StratRAG examples.
        embed_text: Embedding function used for both docs and queries.
        tenant_id: Tenant used in evaluation scopes.
        config: Optional hybrid retrieval config.

    Returns:
        In-memory hybrid retriever.
    """
    chunks = attach_embeddings(stratrag_chunks(examples, tenant_id=tenant_id), embed_text=embed_text)
    return HybridRAGRetriever(
        vector_searcher=InMemoryVectorStore(chunks, embed_query=embed_text),
        keyword_searcher=InMemoryKeywordStore(chunks),
        config=config,
    )


def evaluate_stratrag_retriever(
    examples: Sequence[StratRAGExample],
    retriever: HybridRAGRetriever,
    *,
    tenant_id: str = "stratrag",
    top_k: int = 5,
) -> StratRAGEvaluation:
    """Evaluate a retriever on StratRAG examples.

    Args:
        examples: StratRAG examples.
        retriever: Retriever with matching StratRAG chunks already indexed.
        tenant_id: Tenant used for evaluation scopes.
        top_k: Retrieval depth. Metrics are computed up to 5 by default.

    Returns:
        Complete evaluation result.
    """
    if top_k <= 0:
        msg = "`top_k` must be positive."
        raise ValueError(msg)

    results: list[StratRAGExampleResult] = []
    for example in examples:
        retrieved = retriever.retrieve(
            example.query,
            scope=RetrievalScope(tenant_id=tenant_id, kb_ids=(example.kb_id,), visibility="public"),
            top_k=top_k,
        )
        results.append(score_stratrag_example(example, retrieved))
    return aggregate_stratrag_results(results)


def score_stratrag_example(example: StratRAGExample, retrieved: Sequence[RetrievedChunk | str]) -> StratRAGExampleResult:
    """Score one StratRAG example.

    Args:
        example: StratRAG example.
        retrieved: Ranked retrieved chunks or document identifiers.

    Returns:
        Per-example metrics.
    """
    retrieved_doc_ids = _retrieved_doc_ids(retrieved)
    gold_doc_ids = example.gold_doc_ids()
    return StratRAGExampleResult(
        example_id=example.example_id,
        question_type=example.question_type,
        retrieved_doc_ids=retrieved_doc_ids,
        gold_doc_ids=gold_doc_ids,
        recall_at_1=recall_at_k(retrieved_doc_ids, gold_doc_ids, k=1),
        recall_at_2=recall_at_k(retrieved_doc_ids, gold_doc_ids, k=2),
        recall_at_5=recall_at_k(retrieved_doc_ids, gold_doc_ids, k=5),
        mrr=mean_reciprocal_rank(retrieved_doc_ids, gold_doc_ids),
        ndcg_at_5=ndcg_at_k(retrieved_doc_ids, gold_doc_ids, k=5),
    )


def aggregate_stratrag_results(results: Sequence[StratRAGExampleResult]) -> StratRAGEvaluation:
    """Aggregate StratRAG per-example metrics.

    Args:
        results: Per-example metrics.

    Returns:
        Overall and question-type aggregates.
    """
    grouped: dict[QuestionType, list[StratRAGExampleResult]] = defaultdict(list)
    for result in results:
        grouped[result.question_type].append(result)
    return StratRAGEvaluation(
        overall=_aggregate(results),
        by_question_type={question_type: _aggregate(items) for question_type, items in grouped.items()},
        examples=tuple(results),
    )


def recall_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], *, k: int) -> float:
    """Compute Recall@k for multi-gold retrieval.

    Args:
        retrieved_doc_ids: Ranked retrieved document identifiers.
        gold_doc_ids: Gold document identifiers.
        k: Retrieval depth.

    Returns:
        Fraction of gold documents retrieved in the first `k` positions.
    """
    if not gold_doc_ids:
        return 0.0
    hits = len(set(gold_doc_ids) & set(retrieved_doc_ids[:k]))
    return float(hits) / float(len(set(gold_doc_ids)))


def mean_reciprocal_rank(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> float:
    """Compute reciprocal rank of the first retrieved gold document.

    Args:
        retrieved_doc_ids: Ranked retrieved document identifiers.
        gold_doc_ids: Gold document identifiers.

    Returns:
        Reciprocal rank for the first relevant document, or `0.0`.
    """
    gold = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / float(rank)
    return 0.0


def ndcg_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], *, k: int) -> float:
    """Compute binary NDCG@k.

    Args:
        retrieved_doc_ids: Ranked retrieved document identifiers.
        gold_doc_ids: Gold document identifiers.
        k: Retrieval depth.

    Returns:
        Binary NDCG score in the first `k` positions.
    """
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _aggregate(results: Sequence[StratRAGExampleResult]) -> StratRAGAggregateResult:
    if not results:
        return StratRAGAggregateResult(count=0, recall_at_1=0.0, recall_at_2=0.0, recall_at_5=0.0, mrr=0.0, ndcg_at_5=0.0)
    count = len(results)
    return StratRAGAggregateResult(
        count=count,
        recall_at_1=sum(result.recall_at_1 for result in results) / count,
        recall_at_2=sum(result.recall_at_2 for result in results) / count,
        recall_at_5=sum(result.recall_at_5 for result in results) / count,
        mrr=sum(result.mrr for result in results) / count,
        ndcg_at_5=sum(result.ndcg_at_5 for result in results) / count,
    )


def _retrieved_doc_ids(retrieved: Sequence[RetrievedChunk | str]) -> tuple[str, ...]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        doc_id = item.metadata.doc_id if isinstance(item, RetrievedChunk) else item
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return tuple(doc_ids)


def _shuffle_pool(
    docs: tuple[StratRAGDocument, ...],
    gold_doc_indices: tuple[int, ...],
    *,
    rng: random.Random,
) -> tuple[tuple[StratRAGDocument, ...], tuple[int, ...]]:
    indexed = list(enumerate(docs))
    rng.shuffle(indexed)
    shuffled_docs = tuple(doc for _old_index, doc in indexed)
    gold_original = set(gold_doc_indices)
    shuffled_gold = tuple(index for index, (old_index, _doc) in enumerate(indexed) if old_index in gold_original)
    return shuffled_docs, shuffled_gold


def _normalized_doc_id(example_id: str, raw_doc_id: str) -> str:
    return f"{example_id}:{_hash_text(raw_doc_id)[:16]}"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _object_list(value: object, *, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        msg = f"`{name}` must be a list of objects."
        raise TypeError(msg)
    return [cast("dict[str, object]", item) for item in value]


def _object_sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"`{name}` must be a sequence."
        raise TypeError(msg)
    return value


def _int_value(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    msg = "Expected an integer-compatible value."
    raise TypeError(msg)


def _object_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "Expected an object."
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _question_type(value: object) -> QuestionType:
    if value not in {"bridge", "comparison", "yes-no", "multi-hop"}:
        msg = "`question_type` must be one of: bridge, comparison, yes-no, multi-hop."
        raise ValueError(msg)
    return cast("QuestionType", value)


__all__ = [
    "QuestionType",
    "StratRAGAggregateResult",
    "StratRAGDocument",
    "StratRAGEvaluation",
    "StratRAGExample",
    "StratRAGExampleResult",
    "aggregate_stratrag_results",
    "attach_embeddings",
    "build_in_memory_stratrag_retriever",
    "evaluate_stratrag_retriever",
    "load_stratrag_jsonl",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "parse_stratrag_row",
    "recall_at_k",
    "score_stratrag_example",
    "stratrag_chunks",
]
