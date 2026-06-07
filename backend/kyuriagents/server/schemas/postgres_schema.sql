-- User center and API service tables for deployed KyuriAgents runtimes.
-- RAG vectors live in Milvus; memory and tool tables live in their own schemas.

CREATE TABLE IF NOT EXISTS agent_tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_users (
    user_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    password_hash TEXT,
    password_updated_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

ALTER TABLE IF EXISTS agent_users
    ADD COLUMN IF NOT EXISTS password_hash TEXT;

ALTER TABLE IF EXISTS agent_users
    ADD COLUMN IF NOT EXISTS password_updated_at TIMESTAMPTZ;

ALTER TABLE IF EXISTS agent_users
    ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS agent_api_keys (
    key_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT '',
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_api_keys_tenant_user_idx ON agent_api_keys(tenant_id, user_id);
CREATE INDEX IF NOT EXISTS agent_api_keys_prefix_idx ON agent_api_keys(key_prefix);

CREATE TABLE IF NOT EXISTS agent_threads (
    thread_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_threads_tenant_user_updated_idx ON agent_threads(tenant_id, user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id TEXT PRIMARY KEY,
    message_seq BIGSERIAL UNIQUE,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL REFERENCES agent_threads(thread_id) ON DELETE CASCADE,
    user_id TEXT REFERENCES agent_users(user_id) ON DELETE SET NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS agent_messages
    ADD COLUMN IF NOT EXISTS message_seq BIGSERIAL;

CREATE UNIQUE INDEX IF NOT EXISTS agent_messages_message_seq_uidx ON agent_messages(message_seq);
CREATE INDEX IF NOT EXISTS agent_messages_thread_seq_idx ON agent_messages(tenant_id, thread_id, message_seq ASC);

CREATE TABLE IF NOT EXISTS agent_thread_summaries (
    thread_id TEXT PRIMARY KEY REFERENCES agent_threads(thread_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    summary TEXT NOT NULL DEFAULT '',
    summarized_until_message_seq BIGINT NOT NULL DEFAULT 0,
    summary_version INTEGER NOT NULL DEFAULT 1 CHECK (summary_version > 0),
    token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_thread_summaries_tenant_user_idx
    ON agent_thread_summaries(tenant_id, user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL REFERENCES agent_threads(thread_id) ON DELETE CASCADE,
    goal TEXT NOT NULL,
    intent TEXT NOT NULL DEFAULT 'task'
        CHECK (intent IN ('chat', 'task', 'rag_query', 'memory_query', 'clarify', 'unsafe')),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'planning', 'running', 'waiting_user', 'succeeded', 'failed', 'cancelled')),
    title TEXT NOT NULL DEFAULT '',
    final_answer TEXT NOT NULL DEFAULT '',
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_tasks_user_created_idx
    ON agent_tasks(tenant_id, user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_tasks_thread_created_idx
    ON agent_tasks(tenant_id, thread_id, created_at DESC);
CREATE INDEX IF NOT EXISTS agent_tasks_status_idx
    ON agent_tasks(tenant_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_task_steps (
    step_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL CHECK (step_index >= 0),
    kind TEXT NOT NULL CHECK (kind IN ('think', 'tool', 'rag', 'web', 'process', 'answer')),
    title TEXT NOT NULL DEFAULT '',
    instruction TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL DEFAULT '',
    input JSONB NOT NULL DEFAULT '{}'::jsonb,
    depends_on JSONB NOT NULL DEFAULT '[]'::jsonb,
    parallel_group TEXT NOT NULL DEFAULT '',
    risk TEXT NOT NULL DEFAULT 'read_only'
        CHECK (risk IN ('read_only', 'external_read', 'write', 'destructive', 'network')),
    requires_confirmation BOOLEAN NOT NULL DEFAULT false,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped', 'cancelled')),
    output TEXT NOT NULL DEFAULT '',
    error_message TEXT,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    UNIQUE (task_id, step_index)
);

ALTER TABLE IF EXISTS agent_task_steps
    DROP CONSTRAINT IF EXISTS agent_task_steps_kind_check;
ALTER TABLE IF EXISTS agent_task_steps
    ADD CONSTRAINT agent_task_steps_kind_check
        CHECK (kind IN ('think', 'tool', 'rag', 'web', 'process', 'answer'));

CREATE INDEX IF NOT EXISTS agent_task_steps_task_idx
    ON agent_task_steps(task_id, step_index);
CREATE INDEX IF NOT EXISTS agent_task_steps_tool_idx
    ON agent_task_steps(tool_name, status);

CREATE TABLE IF NOT EXISTS agent_task_events (
    event_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
    step_id TEXT REFERENCES agent_task_steps(step_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_task_events_task_created_idx
    ON agent_task_events(task_id, created_at ASC);
