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

At a high level:

1. A customer authenticates with an API key.
2. The API resolves the customer to a `tenant_id`.
3. The tenant creates one or more jobs.
4. Each job gets its own isolated trainer state, metrics, checkpoints, and genomes.
5. The customer uses job-scoped endpoints for training, metrics, checkpoints, and inference.


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
- caching per-job trainers and agent interfaces

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
- mapping routes to trainer and agent operations
- compatibility routes
- usage logging middleware


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

- start background training for the selected job

Returns:

- full training status snapshot

### `POST /jobs/{job_id}/train/stop`

Purpose:

- stop training and save a checkpoint

Returns:

- full training status snapshot

### `POST /jobs/{job_id}/train/resume`

Purpose:

- resume training from a checkpoint

Input:

- `checkpoint_path`

Returns:

- full training status snapshot


## 5. Training Metrics and Monitoring

### `GET /jobs/{job_id}/train/status`

Purpose:

- main dashboard endpoint for current training state

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

