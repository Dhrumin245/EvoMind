CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    salt TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    role TEXT NOT NULL DEFAULT 'admin',
    scopes_json TEXT NOT NULL DEFAULT '["*"]',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    last_used_at TEXT,
    expires_at TEXT,
    revoked_at TEXT,
    expired_at TEXT,
    rotated_at TEXT,
    rotated_from_key_id TEXT,
    replaced_by_key_id TEXT
);

ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'admin';
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scopes_json TEXT NOT NULL DEFAULT '["*"]';
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS expires_at TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS updated_at TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS revoked_at TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS expired_at TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rotated_at TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rotated_from_key_id TEXT;
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS replaced_by_key_id TEXT;

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_status
ON api_keys (tenant_id, status);

CREATE TABLE IF NOT EXISTS user_accounts (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    scopes_json TEXT NOT NULL DEFAULT '["*"]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_user_accounts_tenant_status
ON user_accounts (tenant_id, status);

CREATE TABLE IF NOT EXISTS user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_salt TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    expires_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_status
ON user_sessions (user_id, status);

CREATE TABLE IF NOT EXISTS tenant_limits (
    tenant_id TEXT PRIMARY KEY,
    requests_per_minute INTEGER NOT NULL,
    requests_per_day INTEGER NOT NULL,
    max_jobs INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limit_counters (
    tenant_id TEXT PRIMARY KEY,
    minute_window TEXT NOT NULL,
    minute_count INTEGER NOT NULL DEFAULT 0,
    day_window TEXT NOT NULL,
    day_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_logs (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    key_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    route_template TEXT,
    status_code INTEGER NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    job_id TEXT,
    billing_tier TEXT NOT NULL DEFAULT 'unclassified',
    billed_tokens INTEGER NOT NULL DEFAULT 0,
    unit_price_inr DOUBLE PRECISION NOT NULL DEFAULT 0,
    estimated_cost_inr DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS route_template TEXT;
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS billing_tier TEXT NOT NULL DEFAULT 'unclassified';
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS billed_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS unit_price_inr DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE usage_logs ADD COLUMN IF NOT EXISTS estimated_cost_inr DOUBLE PRECISION NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_usage_logs_tenant_created
ON usage_logs (tenant_id, created_at);

CREATE TABLE IF NOT EXISTS tenant_billing_accounts (
    tenant_id TEXT PRIMARY KEY,
    currency TEXT NOT NULL DEFAULT 'INR',
    balance_inr DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS billing_ledger (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    amount_inr DOUBLE PRECISION NOT NULL,
    balance_after_inr DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    description TEXT NOT NULL DEFAULT '',
    reference_type TEXT,
    reference_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_billing_ledger_tenant_created
ON billing_ledger (tenant_id, created_at);

CREATE TABLE IF NOT EXISTS billing_topups (
    topup_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    amount_inr DOUBLE PRECISION NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    receipt TEXT NOT NULL,
    provider_order_id TEXT NOT NULL,
    provider_payment_id TEXT,
    description TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    paid_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_topups_provider_order
ON billing_topups (provider, provider_order_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_topups_provider_payment
ON billing_topups (provider, provider_payment_id);

CREATE TABLE IF NOT EXISTS jobs (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    name TEXT NOT NULL,
    base_dir TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_tenant_created
ON jobs (tenant_id, created_at);

CREATE TABLE IF NOT EXISTS job_runtime_claims (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_job_runtime_claims_lease
ON job_runtime_claims (lease_expires_at, updated_at);

CREATE TABLE IF NOT EXISTS job_commands (
    command_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    command_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    worker_id TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_commands_status_created
ON job_commands (status, created_at);

CREATE TABLE IF NOT EXISTS job_runtime_status (
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    worker_id TEXT,
    status_payload_json TEXT NOT NULL DEFAULT '{}',
    last_error TEXT,
    active_command_id TEXT,
    command_type TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, job_id)
);

CREATE TABLE IF NOT EXISTS runtime_workers (
    worker_id TEXT NOT NULL,
    worker_type TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (worker_id, worker_type)
);

CREATE INDEX IF NOT EXISTS idx_runtime_workers_type_lease
ON runtime_workers (worker_type, lease_expires_at, updated_at);

CREATE TABLE IF NOT EXISTS job_events (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_events_tenant_job_created
ON job_events (tenant_id, job_id, created_at);

CREATE TABLE IF NOT EXISTS webhooks (
    webhook_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    secret TEXT,
    subscribed_events_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_delivery_at TEXT,
    last_delivery_status TEXT,
    last_delivery_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_webhooks_tenant_status
ON webhooks (tenant_id, status);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    delivery_id TEXT PRIMARY KEY,
    webhook_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    next_retry_at TEXT,
    delivered_at TEXT,
    last_error TEXT,
    processing_started_at TEXT,
    claim_token TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

ALTER TABLE webhook_deliveries ADD COLUMN IF NOT EXISTS processing_started_at TEXT;
ALTER TABLE webhook_deliveries ADD COLUMN IF NOT EXISTS claim_token TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_event
ON webhook_deliveries (webhook_id, event_id);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_due
ON webhook_deliveries (status, next_retry_at, updated_at);

CREATE TABLE IF NOT EXISTS webhook_delivery_attempts (
    attempt_id TEXT PRIMARY KEY,
    delivery_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    response_status_code INTEGER,
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_attempts_delivery
ON webhook_delivery_attempts (delivery_id, attempt_number);

INSERT INTO schema_migrations (migration_id)
VALUES ('001_init_schema')
ON CONFLICT (migration_id) DO NOTHING;
