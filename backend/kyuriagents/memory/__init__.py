"""Experimental dynamic memory primitives for KyuriAgents.

This package defines storage-neutral contracts for short-term summaries and
long-term user/project memory. LangMem can be layered on top for extraction and
maintenance, while KyuriAgents keeps tenant isolation, audit metadata, and
deployment schemas under its own control.
"""

from kyuriagents.memory.compression import CompressedMemoryContext, MemoryContextBudget, MemoryContextCompressor
from kyuriagents.memory.in_memory import InMemoryMemoryStore
from kyuriagents.memory.indexing import (
    ElasticsearchMilvusMemoryIndexer,
    MemoryHybridSearcher,
    MemoryIndexer,
    memory_records_to_chunks,
)
from kyuriagents.memory.langmem import build_langmem_namespace, create_langmem_memory_tools
from kyuriagents.memory.maintenance import MemoryCompactionConfig, MemoryCompactionResult, MemoryMaintenanceService
from kyuriagents.memory.postgres import PostgresMemoryStore
from kyuriagents.memory.service import MemoryService, format_memory_context
from kyuriagents.memory.types import (
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
    MemoryWriteCandidate,
)

__all__ = [
    "CompressedMemoryContext",
    "ElasticsearchMilvusMemoryIndexer",
    "InMemoryMemoryStore",
    "MemoryCompactionConfig",
    "MemoryCompactionResult",
    "MemoryContextBudget",
    "MemoryContextCompressor",
    "MemoryHybridSearcher",
    "MemoryIndexer",
    "MemoryMaintenanceService",
    "MemoryRecord",
    "MemoryScope",
    "MemorySearchResult",
    "MemoryService",
    "MemoryStore",
    "MemoryWriteCandidate",
    "PostgresMemoryStore",
    "build_langmem_namespace",
    "create_langmem_memory_tools",
    "format_memory_context",
    "memory_records_to_chunks",
]
