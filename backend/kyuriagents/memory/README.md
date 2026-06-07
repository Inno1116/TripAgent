# KyuriAgents Dynamic Memory

This package contains storage-neutral primitives for long-term agent memory.

The intended production layering is:

1. LangGraph short-term memory keeps thread state and checkpoints in PostgreSQL.
2. LangMem extracts and manages durable memory candidates.
3. KyuriAgents stores authoritative memory metadata, audit events, and access rules in PostgreSQL.
4. Searchable memory text is indexed as `source_type="memory"` through the same Milvus and Elasticsearch path used by RAG.

Use `MemoryRecord.to_document_chunk()` when indexing memory into the hybrid RAG
pipeline. Use `MemoryService.build_context()` to inject only relevant Top-K
memories into a prompt. Use `create_langmem_memory_tools()` when wiring LangMem
tools into a LangGraph agent.

## Agent Runtime

Use `RetrievalMiddleware` to expose long-term memory to the main agent:

```python
from kyuriagents import AgentRuntimeConfig, create_kyuri_agent
from kyuriagents.middleware.retrieval import RetrievalMiddleware, RuntimeContextDefaults

config = AgentRuntimeConfig(enable_memory=True, memory_mode="hybrid")
agent = create_kyuri_agent(
    config,
    model=model,
    middleware=[
        RetrievalMiddleware(
            memory_service=memory_service,
            memory_mode="hybrid",
            defaults=RuntimeContextDefaults(tenant_id="default", user_id="user-1"),
        )
    ],
)
```

`hybrid` mode injects a small Top-K memory block automatically and also exposes
`search_memory`, `save_memory`, and `delete_memory` tools.
