# KyuriAgents Memory Schemas

These files are deployment assets for dynamic long-term memory:

- `postgres_schema.sql`: authoritative memory metadata, audit events, access rules, links, and session summaries.
- `memory.env.example`: local development configuration template.

LangGraph's `PostgresSaver` and `PostgresStore` manage their own checkpoint and
store tables. Run their setup/migrations separately, then apply this schema for
the product-level memory metadata that KyuriAgents owns.

Searchable memory content should also be indexed into Milvus and Elasticsearch
using `MemoryRecord.to_document_chunk()` so OmniEval can compare document RAG and
memory retrieval behavior consistently.
