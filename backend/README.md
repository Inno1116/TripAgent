# KyuriAgents Backend

This backend contains the standalone KyuriAgents service code. It no longer
uses the original DeepAgent graph factory; `create_kyuri_agent` assembles the
runtime directly with LangChain/LangGraph `create_agent`, explicit tools, and
KyuriAgents middleware.

## What Is Included

- FastAPI service and user center APIs
- LangChain/LangGraph agent factory
- Hybrid RAG over Elasticsearch and Milvus
- PostgreSQL-backed structured traveler profiles and context summaries
- Redis pending-turn and ingestion queues
- Task planning runtime
- SearXNG and Playwright web search tools
- Document ingestion for PDF, Word, and plain text

## Local Start

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\backend).Path
.\libs\deepagents\.venv\Scripts\python.exe backend\scripts\api_server.py
```

For a clean environment, install the package:

```powershell
cd backend
python -m pip install -e .
python scripts\api_server.py
```

Runtime settings are read from `DEEPAGENTS_*`, `RAG_*`, `DASHSCOPE_*`, and
`KYURI_*` environment variables.
