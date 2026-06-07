-- KyuriAgents RAG metadata schema for PostgreSQL 15+.
-- Vector data lives in Milvus. PostgreSQL stores canonical chunk text while
-- Elasticsearch keeps a searchable copy for keyword retrieval.
-- PostgreSQL is the source of truth for users, tenants, knowledge bases,
-- document lifecycle, chunk manifests, access rules, and offline ingestion jobs.

CREATE OR REPLACE FUNCTION rag_touch_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TABLE IF NOT EXISTS rag_tenants (
    tenant_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'disabled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_users (
    user_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    external_user_id VARCHAR(255),
    display_name VARCHAR(255),
    email VARCHAR(320),
    status VARCHAR(16) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'disabled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_rag_users_tenant_external UNIQUE (tenant_id, external_user_id)
);

CREATE INDEX IF NOT EXISTS idx_rag_users_tenant_status
    ON rag_users (tenant_id, status);

CREATE TABLE IF NOT EXISTS rag_knowledge_bases (
    kb_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    owner_user_id VARCHAR(64) REFERENCES rag_users (user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    visibility VARCHAR(16) NOT NULL DEFAULT 'private'
        CHECK (visibility IN ('private', 'team', 'public')),
    status VARCHAR(16) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived', 'disabled')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS rag_knowledge_bases
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_rag_kb_tenant_visibility
    ON rag_knowledge_bases (tenant_id, visibility, status);
CREATE INDEX IF NOT EXISTS idx_rag_kb_owner
    ON rag_knowledge_bases (owner_user_id);

CREATE TABLE IF NOT EXISTS rag_documents (
    doc_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    kb_id VARCHAR(64) NOT NULL REFERENCES rag_knowledge_bases (kb_id),
    owner_user_id VARCHAR(64) REFERENCES rag_users (user_id),
    source_type VARCHAR(32) NOT NULL,
    source_uri TEXT NOT NULL,
    file_name VARCHAR(512) NOT NULL DEFAULT '',
    mime_type VARCHAR(128) NOT NULL DEFAULT '',
    byte_size BIGINT NOT NULL DEFAULT 0 CHECK (byte_size >= 0),
    title VARCHAR(512) NOT NULL DEFAULT '',
    language VARCHAR(16) NOT NULL DEFAULT 'unknown',
    visibility VARCHAR(16) NOT NULL DEFAULT 'private'
        CHECK (visibility IN ('private', 'team', 'public')),
    status VARCHAR(16) NOT NULL DEFAULT 'processing'
        CHECK (status IN ('active', 'processing', 'failed', 'archived', 'deleted')),
    latest_version VARCHAR(64),
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS rag_documents
    ADD COLUMN IF NOT EXISTS file_name VARCHAR(512) NOT NULL DEFAULT '';
ALTER TABLE IF EXISTS rag_documents
    ADD COLUMN IF NOT EXISTS mime_type VARCHAR(128) NOT NULL DEFAULT '';
ALTER TABLE IF EXISTS rag_documents
    ADD COLUMN IF NOT EXISTS byte_size BIGINT NOT NULL DEFAULT 0 CHECK (byte_size >= 0);
ALTER TABLE IF EXISTS rag_documents
    ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE IF EXISTS rag_documents
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_rag_documents_tenant_kb
    ON rag_documents (tenant_id, kb_id, status);
CREATE INDEX IF NOT EXISTS idx_rag_documents_owner
    ON rag_documents (owner_user_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_source
    ON rag_documents (tenant_id, source_type, left(source_uri, 512));

CREATE TABLE IF NOT EXISTS rag_document_versions (
    doc_version VARCHAR(64) PRIMARY KEY,
    doc_id VARCHAR(64) NOT NULL REFERENCES rag_documents (doc_id),
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    content_hash CHAR(64) NOT NULL,
    parser_version VARCHAR(64) NOT NULL,
    chunker_version VARCHAR(64) NOT NULL,
    embedding_model VARCHAR(128) NOT NULL,
    embedding_version VARCHAR(64) NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'indexed', 'failed', 'superseded')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at TIMESTAMPTZ,
    CONSTRAINT uq_rag_document_versions_hash
        UNIQUE (doc_id, content_hash, embedding_model, embedding_version)
);

CREATE INDEX IF NOT EXISTS idx_rag_document_versions_doc_status
    ON rag_document_versions (doc_id, status);
CREATE INDEX IF NOT EXISTS idx_rag_document_versions_tenant_status
    ON rag_document_versions (tenant_id, status);

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    kb_id VARCHAR(64) NOT NULL REFERENCES rag_knowledge_bases (kb_id),
    doc_id VARCHAR(64) NOT NULL REFERENCES rag_documents (doc_id),
    doc_version VARCHAR(64) NOT NULL REFERENCES rag_document_versions (doc_version),
    user_id VARCHAR(64) REFERENCES rag_users (user_id),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content_hash CHAR(64) NOT NULL,
    chunk_text TEXT NOT NULL DEFAULT '',
    source_type VARCHAR(32) NOT NULL,
    source_uri TEXT NOT NULL,
    title VARCHAR(512) NOT NULL DEFAULT '',
    section_path VARCHAR(1024) NOT NULL DEFAULT '',
    page_start INTEGER CHECK (page_start IS NULL OR page_start >= 0),
    page_end INTEGER CHECK (page_end IS NULL OR page_end >= 0),
    char_start INTEGER CHECK (char_start IS NULL OR char_start >= 0),
    char_end INTEGER CHECK (char_end IS NULL OR char_end >= 0),
    language VARCHAR(16) NOT NULL DEFAULT 'unknown',
    tags JSONB,
    visibility VARCHAR(16) NOT NULL DEFAULT 'private'
        CHECK (visibility IN ('private', 'team', 'public')),
    embedding_model VARCHAR(128) NOT NULL,
    embedding_version VARCHAR(64) NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1 CHECK (schema_version > 0),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_rag_chunks_doc_index UNIQUE (doc_version, chunk_index)
);

ALTER TABLE IF EXISTS rag_chunks
    ADD COLUMN IF NOT EXISTS chunk_text TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_rag_chunks_scope
    ON rag_chunks (tenant_id, kb_id, is_active, visibility);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc
    ON rag_chunks (doc_id, doc_version);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_user
    ON rag_chunks (user_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_hash
    ON rag_chunks (content_hash);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_tags
    ON rag_chunks USING GIN (tags);

CREATE TABLE IF NOT EXISTS rag_access_rules (
    rule_id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    kb_id VARCHAR(64) NOT NULL REFERENCES rag_knowledge_bases (kb_id),
    subject_type VARCHAR(16) NOT NULL
        CHECK (subject_type IN ('user', 'group', 'tenant')),
    subject_id VARCHAR(128) NOT NULL,
    permission VARCHAR(16) NOT NULL DEFAULT 'read'
        CHECK (permission IN ('read', 'write', 'admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_rag_access_rule
        UNIQUE (tenant_id, kb_id, subject_type, subject_id, permission)
);

CREATE INDEX IF NOT EXISTS idx_rag_access_subject
    ON rag_access_rules (tenant_id, subject_type, subject_id);

CREATE TABLE IF NOT EXISTS rag_ingestion_jobs (
    job_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    kb_id VARCHAR(64) NOT NULL REFERENCES rag_knowledge_bases (kb_id),
    doc_id VARCHAR(64) REFERENCES rag_documents (doc_id),
    requested_by_user_id VARCHAR(64) REFERENCES rag_users (user_id),
    source_uri TEXT NOT NULL,
    parser_mode VARCHAR(16) NOT NULL DEFAULT 'auto'
        CHECK (parser_mode IN ('auto', 'local', 'mcp')),
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    status VARCHAR(16) NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE IF EXISTS rag_ingestion_jobs
    ADD COLUMN IF NOT EXISTS parser_mode VARCHAR(16) NOT NULL DEFAULT 'auto'
        CHECK (parser_mode IN ('auto', 'local', 'mcp'));
ALTER TABLE IF EXISTS rag_ingestion_jobs
    ADD COLUMN IF NOT EXISTS progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100);
ALTER TABLE IF EXISTS rag_ingestion_jobs
    ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_rag_ingestion_jobs_scope
    ON rag_ingestion_jobs (tenant_id, kb_id, status);
CREATE INDEX IF NOT EXISTS idx_rag_ingestion_jobs_user
    ON rag_ingestion_jobs (requested_by_user_id);

CREATE TABLE IF NOT EXISTS rag_eval_runs (
    run_id VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL REFERENCES rag_tenants (tenant_id),
    dataset_name VARCHAR(128) NOT NULL,
    split VARCHAR(32) NOT NULL DEFAULT '',
    retriever_name VARCHAR(128) NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_eval_runs_tenant_dataset
    ON rag_eval_runs (tenant_id, dataset_name, split, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_eval_runs_status
    ON rag_eval_runs (tenant_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS rag_eval_results (
    run_id VARCHAR(128) NOT NULL REFERENCES rag_eval_runs (run_id) ON DELETE CASCADE,
    example_id VARCHAR(128) NOT NULL,
    question_type VARCHAR(32) NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    gold_doc_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieved_doc_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    recall_at_1 NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    recall_at_2 NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    recall_at_5 NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    mrr NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    ndcg_at_5 NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, example_id)
);

CREATE INDEX IF NOT EXISTS idx_rag_eval_results_question_type
    ON rag_eval_results (run_id, question_type);
CREATE INDEX IF NOT EXISTS idx_rag_eval_results_recall
    ON rag_eval_results (run_id, recall_at_2, recall_at_5);

DROP TRIGGER IF EXISTS trg_rag_tenants_updated_at ON rag_tenants;
CREATE TRIGGER trg_rag_tenants_updated_at
    BEFORE UPDATE ON rag_tenants
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_rag_users_updated_at ON rag_users;
CREATE TRIGGER trg_rag_users_updated_at
    BEFORE UPDATE ON rag_users
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_rag_knowledge_bases_updated_at ON rag_knowledge_bases;
CREATE TRIGGER trg_rag_knowledge_bases_updated_at
    BEFORE UPDATE ON rag_knowledge_bases
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_rag_documents_updated_at ON rag_documents;
CREATE TRIGGER trg_rag_documents_updated_at
    BEFORE UPDATE ON rag_documents
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_rag_chunks_updated_at ON rag_chunks;
CREATE TRIGGER trg_rag_chunks_updated_at
    BEFORE UPDATE ON rag_chunks
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_rag_ingestion_jobs_updated_at ON rag_ingestion_jobs;
CREATE TRIGGER trg_rag_ingestion_jobs_updated_at
    BEFORE UPDATE ON rag_ingestion_jobs
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();

DROP TRIGGER IF EXISTS trg_rag_eval_runs_updated_at ON rag_eval_runs;
CREATE TRIGGER trg_rag_eval_runs_updated_at
    BEFORE UPDATE ON rag_eval_runs
    FOR EACH ROW EXECUTE FUNCTION rag_touch_updated_at();
