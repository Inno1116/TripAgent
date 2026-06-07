# KyuriAgents RAG

This package contains the online retrieval path for hybrid RAG:

1. Rewrite the user query.
2. Retrieve candidates from a vector store such as Milvus.
3. Retrieve candidates from a keyword store such as Elasticsearch.
4. Fuse ranks with reciprocal rank fusion.
5. Rerank candidates.
6. Return final Top-K chunks.

The offline indexing pipeline is intentionally separate. It should parse source
documents, chunk text, create embeddings, write vectors to Milvus, write keyword
documents to Elasticsearch, and write authoritative metadata to PostgreSQL.

Production integrations should implement `VectorSearcher`, `KeywordSearcher`,
and `Reranker`. The included in-memory stores are for local development and
unit tests while Milvus or Elasticsearch are not installed.

For local services, install the optional clients and wire the adapters:

```bash
uv add 'kyuriagents[rag]'
```

```python
from kyuriagents.rag import ElasticsearchKeywordStore, MilvusVectorStore

vector_store = MilvusVectorStore(
    collection_name="rag_chunks",
    uri="http://localhost:19530",
    embed_query=embed_query,
)
keyword_store = ElasticsearchKeywordStore(
    index="rag_chunks",
    url="http://localhost:9200",
)
```

Every online retrieval call should pass a `RetrievalScope`. Even single-tenant
deployments should use a stable tenant id such as `default` so future
multi-tenant migrations do not require changing the retrieval API.

## Agent Runtime

Use `create_kyuri_agent` or `RetrievalMiddleware` to expose the retriever to the main agent:

```python
from kyuriagents import AgentRuntimeConfig, create_kyuri_agent
from kyuriagents.middleware.retrieval import RetrievalMiddleware, RuntimeContextDefaults

config = AgentRuntimeConfig(enable_rag=True, rag_mode="tool")
agent = create_kyuri_agent(
    config,
    model=model,
    middleware=[
        RetrievalMiddleware(
            rag_retriever=retriever,
            rag_mode="tool",
            defaults=RuntimeContextDefaults(tenant_id="default"),
        )
    ],
)
```

The middleware adds a `search_knowledge_base` tool. If `rag_mode` is `auto` or
`hybrid`, it also injects a small Top-K context block before model calls.
