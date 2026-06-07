"""Experimental RAG primitives for KyuriAgents.

This package contains the online retrieval path for a hybrid RAG system:
query rewriting, keyword/vector retrieval, rank fusion, reranking, and Top-K
selection. Offline document parsing and embedding jobs should write canonical
chunk text into PostgreSQL, searchable copies into Elasticsearch, and vector
metadata into Milvus.
"""

from kyuriagents.rag.elasticsearch import ElasticsearchKeywordStore
from kyuriagents.rag.hybrid import HybridRAGRetriever, HybridSearchConfig
from kyuriagents.rag.in_memory import InMemoryKeywordStore, InMemoryVectorStore
from kyuriagents.rag.metadata import ChunkMetadata, RetrievalScope
from kyuriagents.rag.milvus import MilvusVectorStore
from kyuriagents.rag.postgres import PostgresChunkTextHydrator
from kyuriagents.rag.query import IdentityQueryRewriter, QueryRewrite, QueryRewriter
from kyuriagents.rag.rerank import DashScopeTextReranker, FusedScoreReranker, LexicalReranker
from kyuriagents.rag.stratrag import (
    StratRAGAggregateResult,
    StratRAGDocument,
    StratRAGEvaluation,
    StratRAGExample,
    StratRAGExampleResult,
    aggregate_stratrag_results,
    attach_embeddings,
    build_in_memory_stratrag_retriever,
    evaluate_stratrag_retriever,
    load_stratrag_jsonl,
    mean_reciprocal_rank,
    ndcg_at_k,
    parse_stratrag_row,
    recall_at_k,
    score_stratrag_example,
    stratrag_chunks,
)
from kyuriagents.rag.types import (
    ChunkHydrator,
    DocumentChunk,
    KeywordSearcher,
    Reranker,
    RetrievedChunk,
    VectorSearcher,
)

__all__ = [
    "ChunkHydrator",
    "ChunkMetadata",
    "DashScopeTextReranker",
    "DocumentChunk",
    "ElasticsearchKeywordStore",
    "FusedScoreReranker",
    "HybridRAGRetriever",
    "HybridSearchConfig",
    "IdentityQueryRewriter",
    "InMemoryKeywordStore",
    "InMemoryVectorStore",
    "KeywordSearcher",
    "LexicalReranker",
    "MilvusVectorStore",
    "PostgresChunkTextHydrator",
    "QueryRewrite",
    "QueryRewriter",
    "Reranker",
    "RetrievalScope",
    "RetrievedChunk",
    "StratRAGAggregateResult",
    "StratRAGDocument",
    "StratRAGEvaluation",
    "StratRAGExample",
    "StratRAGExampleResult",
    "VectorSearcher",
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
