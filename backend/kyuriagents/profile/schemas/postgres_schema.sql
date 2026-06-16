-- Structured traveler profile memory for Kyuri TripAgent.

CREATE TABLE IF NOT EXISTS user_travel_profiles (
    tenant_id TEXT NOT NULL REFERENCES agent_tenants(tenant_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES agent_users(user_id) ON DELETE CASCADE,
    profile_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    profile_version INTEGER NOT NULL DEFAULT 1 CHECK (profile_version > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE INDEX IF NOT EXISTS user_travel_profiles_updated_idx
    ON user_travel_profiles(tenant_id, updated_at DESC);
