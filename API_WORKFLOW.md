# EvoMind API Workflow

## Overview

The EvoMind API is now structured as a commercial, tenant-aware service.

It supports:

- API key authentication
- Tenant isolation
- Job-based training and inference
- Per-tenant usage logging
- Per-tenant request limits and job quotas
- Backward-compatible default-job routes
- Queue-driven training execution via a separate worker process

At a high level:

1. A customer authenticates with an API key.
2. The API resolves the customer to a `tenant_id`.
3. The tenant creates one or more jobs.
4. Each job gets its own isolated trainer state, metrics, checkpoints, and genomes.
5. The customer uses job-scoped endpoints for training, metrics, checkpoints, and inference.

Training execution is no longer performed inside the API process.

The API now acts as a control plane:

1. authenticated requests enqueue training commands
2. a separate worker process claims commands from the shared runtime database
3. the worker runs trainers in its own process and writes runtime status back to the database

The repository also includes a GitHub Actions CI workflow at `.github/workflows/api-ci.yml`.

It currently runs:

- dependency installation from `requirements.txt`
- Python compile checks for the API modules
- the full `unittest` suite under `tests/`


## Windows Deployment

The production deployment path is Windows-native. The API and worker run from a Python virtual environment on the Windows host, and deployment automation is handled by PowerShell plus Windows Scheduled Tasks.

The repo includes:

- `deploy/windows/deploy.ps1` to mirror the repository into the production directory, create or update `.venv`, install dependencies, register the API and worker scheduled tasks, start them, and wait for readiness
- `deploy/windows/run-service.ps1` as the long-running task entrypoint for the API and worker
- `deploy/windows/backup.ps1` to run production backups from the Windows host
- `deploy/Caddyfile` for a Windows Caddy reverse proxy in front of `127.0.0.1:8000`
- `.env.production.example` showing Windows-style production settings

Production host prerequisites:

- Windows Server or Windows 11 with PowerShell 5.1+ or PowerShell 7+
- Python 3.13 installed and available as `python`, or configured with `PYTHON_EXE`
- a reachable PostgreSQL database, either managed or installed on Windows
- a GitHub Actions self-hosted runner installed on the production host with labels `Windows` and `production`
- optional Caddy for TLS termination, installed as a Windows service or managed separately

Prepare the deployment directory on the Windows host:

```powershell
New-Item -ItemType Directory -Force C:\EvoMind
Copy-Item .env.production.example C:\EvoMind\.env.production
New-Item -ItemType Directory -Force C:\EvoMind\secrets
Set-Content -NoNewline C:\EvoMind\secrets\control_plane_db_url.txt "postgresql://evomind:replace-with-password@db.example.com:5432/evomind?sslmode=require"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" | Set-Content -NoNewline C:\EvoMind\secrets\webhook_secret_key.txt
```

Edit `C:\EvoMind\.env.production` for the real hostname, CORS origins, storage paths, and secret file paths.

Manual deployment from a checkout:

```powershell
.\deploy\windows\deploy.ps1 -SourcePath $PWD -DeployPath C:\EvoMind
```

This creates or updates:

- `EvoMindWorker`, a Windows Scheduled Task running `python -m api.worker`
- `EvoMindApi`, a Windows Scheduled Task running `python -m uvicorn api.server:app --host 127.0.0.1 --port 8000`
- runtime logs under `C:\EvoMind\output_logs`

Control-plane storage is now configurable with environment variables:

- `EVOMIND_CONTROL_PLANE_DB_URL`
- `EVOMIND_CONTROL_PLANE_DB_URL_FILE`
- `EVOMIND_CONTROL_PLANE_DB_HOST`
- `EVOMIND_CONTROL_PLANE_DB_PORT`
- `EVOMIND_CONTROL_PLANE_DB_NAME`
- `EVOMIND_CONTROL_PLANE_DB_USER`
- `EVOMIND_CONTROL_PLANE_DB_PASSWORD`
- `EVOMIND_CONTROL_PLANE_DB_PASSWORD_FILE`
- `EVOMIND_CONTROL_PLANE_DB_SSLMODE`
- `EVOMIND_API_AUTH_DB_URL`
- `EVOMIND_API_EVENTS_DB_URL`
- `EVOMIND_API_JOBS_DB_URL`

If `EVOMIND_ENV=production`, SQLite is rejected by default unless explicitly overridden with:

- `EVOMIND_ALLOW_SQLITE_IN_PRODUCTION=true`

Artifact and local fallback storage are still configurable with:

- `EVOMIND_DATA_DIR`
- `EVOMIND_BACKUP_DIR`
- `EVOMIND_TENANT_ROOT_DIR`
- `EVOMIND_API_AUTH_DB`
- `EVOMIND_API_EVENTS_DB`
- `EVOMIND_API_JOBS_DB`

For production, the intended model is:

- PostgreSQL for auth, events, usage, and runtime coordination
- Windows filesystem directories for tenant artifacts and backups
- tenant job directories under `EVOMIND_TENANT_ROOT_DIR` for checkpoints and artifacts
- `EVOMIND_SITE_ADDRESS` set to the public hostname handled by Caddy or another Windows reverse proxy
- webhook secrets encrypted at rest via `EVOMIND_WEBHOOK_SECRET_KEY` or `EVOMIND_WEBHOOK_SECRET_KEY_FILE`
- periodic backups created with `deploy/windows/backup.ps1` or `python -m api.backup create`

## Operations Automation

The repo now includes production automation under `.github/workflows/`:

- `windows-release.yml` packages a Windows deployment archive on `main`, tags, or manual dispatch
- `deploy-production.yml` deploys a chosen Git ref on the self-hosted Windows production runner, then waits for `/health/readiness`
- `auto-deploy-production.yml` deploys automatically after a successful Windows release from `main`
- `production-backup.yml` runs a scheduled backup job on the self-hosted Windows production runner and prunes old archives
- `backup-restore-drill.yml` runs a weekly Windows restore drill without Linux service containers

The production backup job uses:

- `python scripts/backup_job.py` for archive creation, manifest verification, and retention pruning
- `python scripts/backup_restore_drill.py` for a full create-mutate-restore verification path

Required GitHub Actions secrets for deployment and backups:

- `DEPLOY_PATH`

`DEPLOY_PATH` should point to the persistent production directory on the Windows host, for example `C:\EvoMind`.

Recommended release sequence:

1. Merge to `main` and let `Windows Release` produce the deployment archive.
2. Run `Deploy Production` with the target ref, or let `Auto Deploy Production` run after the release workflow.
3. Confirm readiness and scheduled backups on the host.
4. Review the weekly `Backup Restore Drill` workflow to confirm restore integrity.

The backup CLI stores tenant artifacts plus SQLite snapshots when SQLite is in use locally:

```powershell
python -m api.backup create
python -m api.backup restore --input backups/evomind-backup-<timestamp>.tar.gz --force
```

Run restore with the API and worker stopped so local SQLite fallback files and tenant artifacts can be replaced safely.

SQLite fallback databases are still opened in WAL mode with a busy timeout and foreign keys enabled so the API and worker can coordinate more reliably on a single host.


## Core Concepts

### Tenant

A tenant represents one customer or organization.

Each tenant has:

- its own API keys
- its own request limits
- its own usage history
- its own jobs

### Job

A job is one isolated training and inference workspace inside a tenant.

Each job has its own:

- training state
- training metrics
- checkpoints
- genomes selected for inference

### Default Job

For backward compatibility, the older endpoints such as `/train/status` and `/agent/action` still exist.

Those routes now operate on the tenant's `default` job.


## Request Lifecycle

### 1. Authentication

The customer sends one of:

- `X-API-Key: <key>`
- `Authorization: Bearer <key>`

The API validates the key in `api/auth.py`.

If valid, the API resolves:

- `key_id`
- `tenant_id`
- key status

### 2. Rate Limit Check

Before the request is handled, the tenant's request limits are checked.

Current controls:

- requests per minute
- requests per day
- maximum jobs per tenant

If the tenant exceeds request limits, the API returns `429 Too Many Requests`.

If the tenant exceeds its max job quota, the API returns `403`.

### 3. Usage Logging

After the request completes, the API logs usage metadata:

- tenant id
- key id
- HTTP method
- path
- response status code
- duration in ms
- job id if present

This supports billing, auditing, and analytics.

### 4. Tenant and Job Resolution

If the endpoint is job-scoped, the API loads the requested job for that tenant.

If the endpoint is a compatibility route, the API resolves the tenant's `default` job.

### 5. Job-Scoped Execution

The API loads or reuses:

- an `EvoTrainer`
- an `AgentInterface`

for that tenant/job pair.

Those are isolated in storage and in memory from other jobs and tenants.


## Storage Layout

Per-job state is stored under:

`data/tenants/{tenant_id}/jobs/{job_id}/`

Typical files include:

- `config.json`
- `expirement_state.json`
- `metrices.csv`
- `coevolution_state.json`
- `checkpoints/...`

This means one customer's training state does not mix with another customer's state.


## API Layers

### `api/auth.py`

Responsible for:

- API key storage and validation
- tenant limit storage
- usage log storage
- rate limit enforcement
- CLI management commands for keys and limits

### `api/job_manager.py`

Responsible for:

- creating jobs
- listing jobs
- loading job metadata
- runtime claims for active workers
- command queue persistence for start/resume/stop
- runtime status snapshots written by workers
- loading trainer snapshots for read-only API requests

### `api/trainer.py`

Responsible for:

- initializing training state
- starting and stopping training
- resuming from checkpoints
- saving checkpoints
- producing training status
- exposing metrics and insights

### `api/interface.py`

Responsible for:

- selecting best genomes
- returning genome summaries
- running single inference
- running batch inference

### `api/server.py`

Responsible for:

- FastAPI route definitions
- wiring auth to endpoints
- mapping training control routes to queue submissions
- reading runtime status produced by workers
- compatibility routes
- usage logging middleware

### `api/worker.py`

Responsible for:

- claiming queued training commands
- running `EvoTrainer` instances outside the API process
- heartbeating runtime leases
- publishing runtime status for polling endpoints
- emitting actual `job.started`, `job.resumed`, and `job.stopped` events


## Endpoint Groups

## 1. Health

### `GET /health`

Purpose:

- verify the API service is up

Returns:

- service health status

Main fields:

- `status`
- `message`
- `uptime_seconds`

### `GET /health/readiness`

Purpose:

- verify that runtime dependencies are operational for serving production traffic

Readiness now requires:

- API auth database connectivity
- runtime coordination database connectivity
- at least one active external training worker heartbeat
- event database connectivity
- webhook delivery worker running inside the API process

Behavior:

- returns `200` with `status="ready"` when all dependencies are healthy
- returns `503` with `status="not_ready"` when any dependency is unhealthy


## 2. Usage and Commercial Limits

### `GET /usage/limits`

Purpose:

- return the tenant's configured commercial limits

Returns:

- `tenant_id`
- `requests_per_minute`
- `requests_per_day`
- `max_jobs`

### `GET /usage/summary`

Purpose:

- return the tenant's API consumption summary

Returns:

- `tenant_id`
- `requests_last_minute`
- `requests_last_day`
- `requests_total`
- `requests_per_minute_limit`
- `requests_per_day_limit`
- `max_jobs`
- `remaining_this_minute`
- `remaining_today`


## 3. Job Management

### `POST /jobs`

Purpose:

- create a new isolated job for the authenticated tenant

Input:

- optional `job_id`
- optional `name`

Returns:

- job metadata

Main fields:

- `job_id`
- `tenant_id`
- `name`
- `base_dir`
- `created_at`
- `updated_at`
- `status`
- `generation`

### `GET /jobs`

Purpose:

- list all jobs owned by the authenticated tenant

Returns:

- count
- list of job summaries

### `GET /jobs/{job_id}`

Purpose:

- return metadata for one specific job

Returns:

- same job summary fields as `POST /jobs`


## 4. Training Control

### `POST /jobs/{job_id}/train/start`

Purpose:

- queue a training start request for the selected job

Returns:

- current training status snapshot

Notes:

- the request is accepted by the API immediately
- the separate worker process performs the actual start
- if the job is not already running, the returned status is typically `queued`

### `POST /jobs/{job_id}/train/stop`

Purpose:

- queue a training stop request for the selected job

Returns:

- current training status snapshot

### `POST /jobs/{job_id}/train/resume`

Purpose:

- queue a resume request from a checkpoint

Input:

- `checkpoint_path`

Notes:

- `checkpoint_path` must resolve inside the job's checkpoint directory
- relative paths are resolved from that directory
- absolute paths outside that directory are rejected

Returns:

- current training status snapshot


## 5. Training Metrics and Monitoring

### `GET /jobs/{job_id}/train/status`

Purpose:

- main dashboard endpoint for current training state

Notes:

- if a worker owns the job, this endpoint returns worker-published runtime status
- if a start or resume command is queued and no worker has claimed the job yet, status is `queued`

Returns:

- current generation
- current stage
- current fitness summary
- learning summary
- behavior summary
- diversity summary
- neural health summary
- system summary

Returned metric groups:

- `fitness.prey.best`
- `fitness.prey.average`
- `fitness.predator.best`
- `fitness.predator.average`
- `learning.adaptability`
- `learning.meta_effectiveness`
- `learning.performance_change`
- `learning.instability`
- `behavior.success_rate`
- `behavior.stability`
- `behavior.novelty`
- `diversity.prey_species`
- `diversity.predator_species`
- `neural_health.dead_connections`
- `neural_health.saturation`
- `system.evaluation_time_sec`
- `system.status`
- `system.uptime_seconds`
- `system.last_update`

Also includes flat backward-compatible fields:

- `status`
- `best_prey_fitness`
- `best_predator_fitness`
- `mean_prey_fitness`
- `mean_predator_fitness`
- `curriculum_stage`
- `total_generations_trained`
- `uptime_seconds`
- `last_update`

### `GET /jobs/{job_id}/train/insights`

Purpose:

- provide trend-friendly metrics over recent generations

Query params:

- `last_n`

Returns:

- `window`
- `source`
- `fitness_trend`
- `diversity_trend`
- `learning_trend`

`fitness_trend` items contain:

- `generation`
- `stage`
- `prey_best`
- `prey_average`
- `predator_best`
- `predator_average`

`diversity_trend` items contain:

- `generation`
- `stage`
- `prey_species`
- `predator_species`

`learning_trend` items contain:

- `generation`
- `stage`
- `adaptability`
- `meta_effectiveness`
- `performance_change`
- `instability`

### `GET /jobs/{job_id}/train/metrics`

Purpose:

- expose raw training metric rows for analytics, export, or dashboards

Query params:

- `limit`
- `offset`
- `since_generation`

Returns:

- `source`
- `total`
- `count`
- `limit`
- `offset`
- `since_generation`
- `items`

`items` contain the full generation metric payloads available in memory or from the metrics CSV.

Use this endpoint when the customer wants the most complete training metrics.


## 6. Checkpoints

### `GET /jobs/{job_id}/train/checkpoints`

Purpose:

- list available checkpoints for a job

Returns per checkpoint:

- `checkpoint_path`
- `generation`
- `saved_at_utc`
- `config_path`
- `experiment_path`
- `metrics_path`
- `marker_exists`

### `POST /jobs/{job_id}/train/checkpoints`

Purpose:

- create a new checkpoint on demand

Input:

- optional custom relative `path`

Returns:

- one checkpoint summary with the same fields as above


## 7. Agent Inference

### `POST /jobs/{job_id}/agent/action`

Purpose:

- run one inference call using the selected best genome for that job

Input:

- `observation`
- `genome_type`
- optional `generation`
- optional `max_action_length`

Returns:

- `action`
- `genome_id`
- `genome_fitness`
- `genome_type`
- `generation`
- `confidence`

### `POST /jobs/{job_id}/agent/action/batch`

Purpose:

- run batch inference over multiple observations

Input:

- `observations`
- `genome_type`
- optional `generation`
- optional `max_action_length`

Returns:

- `genome_id`
- `genome_type`
- `generation`
- `genome_fitness`
- `confidence`
- `batch_size`
- `actions`


## 8. Agent and Genome Metadata

### `GET /jobs/{job_id}/agent/info`

Purpose:

- return metadata about the selected best agent

Query params:

- `genome_type`
- optional `generation`

Returns:

- `available`
- `genome_id`
- `genome_type`
- `fitness`
- `generation`
- `source`
- `gene_count`
- `input_size`
- `output_size`
- `architecture`

### `GET /jobs/{job_id}/genomes`

Purpose:

- list top available genomes for the selected job

Query params:

- optional `genome_type`
- `limit_per_type`

Returns:

- `count`
- `items`

Each item contains:

- `genome_id`
- `genome_type`
- `fitness`
- `generation`
- `source`
- `gene_count`
- `input_size`
- `output_size`
- `architecture`

### `GET /jobs/{job_id}/genomes/{genome_id}`

Purpose:

- return metadata for one genome

Returns:

- same fields as each genome item above


## Compatibility Routes

The following older routes still exist:

- `/train/start`
- `/train/stop`
- `/train/status`
- `/train/insights`
- `/train/metrics`
- `/train/resume`
- `/train/checkpoints`
- `/agent/action`
- `/agent/action/batch`
- `/agent/info`
- `/genomes`
- `/genomes/{genome_id}`

These routes now operate on the authenticated tenant's `default` job.

This keeps older client integrations working while the newer job-based API is preferred.


## Which Endpoint Should A Customer Use

### For live dashboard status

Use:

- `GET /jobs/{job_id}/train/status`

### For trend graphs

Use:

- `GET /jobs/{job_id}/train/insights`

### For raw metrics export or analytics

Use:

- `GET /jobs/{job_id}/train/metrics`

### For saved training checkpoints

Use:

- `GET /jobs/{job_id}/train/checkpoints`

### For best model metadata

Use:

- `GET /jobs/{job_id}/agent/info`
- `GET /jobs/{job_id}/genomes`

### For single inference

Use:

- `POST /jobs/{job_id}/agent/action`

### For batch inference

Use:

- `POST /jobs/{job_id}/agent/action/batch`

### For commercial monitoring

Use:

- `GET /usage/limits`
- `GET /usage/summary`


## Recommended Customer Flow

1. Create or obtain an API key.
2. Call `GET /usage/limits` to know account limits.
3. Create a job with `POST /jobs`.
4. Start training with `POST /jobs/{job_id}/train/start`.
5. Poll `GET /jobs/{job_id}/train/status`.
6. Fetch trend data with `GET /jobs/{job_id}/train/insights`.
7. Export detailed metrics with `GET /jobs/{job_id}/train/metrics`.
8. Inspect evolved genomes with `GET /jobs/{job_id}/genomes`.
9. Run inference with `POST /jobs/{job_id}/agent/action` or `/batch`.
10. Monitor API consumption with `GET /usage/summary`.


## Operational Notes

- Authenticated requests are rate-limited.
- Authenticated requests are logged for usage tracking.
- Jobs are isolated per tenant.
- Job creation is quota-limited.
- Compatibility routes exist, but new integrations should prefer job-scoped routes.
- Training execution requires a separate worker process:
  `python -m api.worker`
