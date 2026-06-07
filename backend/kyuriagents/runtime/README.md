# KyuriAgents Runtime

This package wires the SDK primitives into a runnable deployment shape:

- DashScope chat and embedding clients through the OpenAI-compatible API.
- PostgreSQL schema/bootstrap helpers.
- `PostgresMemoryStore` for durable long-term memory.
- `RetrievalMiddleware` with RAG and memory enabled from one config object.
- `ToolGovernanceMiddleware` with policy checks and optional PostgreSQL audit logs.
- Optional MCP tool loading via `langchain-mcp-adapters`.

Minimal usage:

```python
from kyuriagents.runtime import AgentRuntimeConfig, create_kyuri_agent

config = AgentRuntimeConfig.from_env(
    tenant_id="default",
    user_id="local-user",
)
agent = create_kyuri_agent(config)
```

Before first startup, create the database and apply schemas:

```python
from kyuriagents.runtime import (
    AgentRuntimeConfig,
    apply_kyuriagents_postgres_schemas,
    create_postgres_database,
)

config = AgentRuntimeConfig.from_env()
if config.postgres_admin_dsn:
    create_postgres_database(
        admin_dsn=config.postgres_admin_dsn,
        database=config.postgres_database,
        owner="kyuriagents",
    )
if config.postgres_dsn:
    apply_kyuriagents_postgres_schemas(dsn=config.postgres_dsn)
```

API keys should come from environment variables, never from checked-in files.

API server usage:

```bash
pip install "kyuriagents[api,runtime]"
python scripts/api_server.py
```

Set `DEEPAGENTS_API_ADMIN_KEY` before using admin bootstrap endpoints:

- `POST /v1/admin/tenants`
- `POST /v1/admin/users`
- `POST /v1/admin/api-keys`

Public email/password auth endpoints are available without email verification:

- `POST /v1/auth/register`
- `POST /v1/auth/login`
- `POST /v1/auth/logout`
- `POST /v1/auth/tokens/{key_id}/revoke`

Register and login return an `access_token`. Normal application requests use `Authorization: Bearer <access_token>` or
`X-API-Key: <api_key>`.
Login tokens expire after `DEEPAGENTS_AUTH_TOKEN_TTL_DAYS` days. Set it to `0`
to issue non-expiring local-development tokens.
Browser clients must come from an allowed origin. Development defaults are
`http://127.0.0.1:5173` and `http://localhost:5173`; override them with
`DEEPAGENTS_API_CORS_ORIGINS`.

Long-term memory can use a separate hybrid retrieval index so normal RAG
knowledge bases do not mix with user memories. Defaults are:

```bash
MEMORY_ES_INDEX=memory_chunks
MEMORY_MILVUS_COLLECTION=memory_chunks
DEEPAGENTS_MEMORY_CHECKPOINT_INTERVAL=10
DEEPAGENTS_MEMORY_CHECKPOINT_MAX_CHARS=3000
DEEPAGENTS_ENABLE_CONTEXT_SUMMARIZATION=true
DEEPAGENTS_CONTEXT_SUMMARY_MODEL=
DEEPAGENTS_CONTEXT_SUMMARY_TRIGGER_MESSAGES=40
DEEPAGENTS_CONTEXT_SUMMARY_KEEP_MESSAGES=12
```

`DEEPAGENTS_MEMORY_CHECKPOINT_INTERVAL` controls deterministic long-term
memory checkpoints. The default writes one `summary` memory every 10 user
turns. Set it to `0` to disable automatic checkpoints.

Short-term context summarization is separate from long-term memory. When a
thread reaches `DEEPAGENTS_CONTEXT_SUMMARY_TRIGGER_MESSAGES`, older messages
are summarized for the next model call and only
`DEEPAGENTS_CONTEXT_SUMMARY_KEEP_MESSAGES` recent messages stay verbatim. The
raw thread state remains checkpointed; this only manages what the model sees in
its active context window.
Set `DEEPAGENTS_CONTEXT_SUMMARY_MODEL` to use a cheaper DashScope-compatible
chat model for summaries. Leave it empty to reuse the main chat model.

Create those indexes before enabling hybrid memory search:

```bash
python scripts/memory_index.py init
```

If memories already exist in PostgreSQL, rebuild the hybrid memory index:

```bash
python scripts/memory_index.py reindex
```

MCP usage is optional. Set `DEEPAGENTS_ENABLE_MCP=true` and point
`DEEPAGENTS_MCP_CONFIG_PATH` at a JSON file shaped like `mcp.json.example`.
Secrets in that file should be referenced as `${ENV_VAR}` placeholders.
