# KyuriAgents RAG Schemas

These files are deployment assets for the hybrid RAG path:

- `postgres_schema.sql`: source-of-truth metadata, access control, document versions, chunk manifests, and ingestion jobs.
- `milvus_collection.json`: Milvus collection contract for vectors and scalar filters.
- `elasticsearch_index.json`: Elasticsearch keyword index mapping.
- `rag.env.example`: local development configuration template.

Before creating the Milvus collection, set `embedding.dimension` to the dimension
of the embedding model used by the offline indexing job. Keep
`embedding_model`, `embedding_version`, and `schema_version` populated on every
chunk so OmniEval runs can compare retrieval quality across indexing variants.

Online retrieval should always pass a `RetrievalScope` with `tenant_id`. In a
future multi-user deployment, tenant and access filters must be pushed down into
Milvus and Elasticsearch before reranking.

Your local defaults are:

```txt
Elasticsearch: http://localhost:9200
Milvus: http://localhost:19530
PostgreSQL: postgresql://kyuriagents:change-me@localhost:5432/kyuriagents_rag
```
