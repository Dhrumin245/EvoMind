import { expect, test } from '@playwright/test';

async function mockConsoleApi(page: import('@playwright/test').Page) {
  await page.route('**/api/health', async (route) => {
    await route.fulfill({ json: { status: 'healthy', message: 'ok', uptime_seconds: 12 } });
  });
  await page.route('**/api/health/readiness', async (route) => {
    await route.fulfill({
      json: {
        status: 'ready',
        message: 'ready',
        uptime_seconds: 12,
        components: [{ name: 'training_worker', healthy: false, detail: 'no active workers' }],
      },
    });
  });
  await page.route('**/api/usage/limits', async (route) => {
    await route.fulfill({ json: { tenant_id: 'default', requests_per_minute: 120, requests_per_day: 10000, max_jobs: 5 } });
  });
  await page.route('**/api/usage/summary', async (route) => {
    await route.fulfill({
      json: {
        tenant_id: 'default',
        requests_last_minute: 0,
        requests_last_day: 0,
        requests_total: 0,
        requests_per_minute_limit: 120,
        requests_per_day_limit: 10000,
        max_jobs: 5,
        remaining_this_minute: 120,
        remaining_today: 10000,
        estimated_cost_last_day_inr: 0,
        estimated_cost_total_inr: 0,
      },
    });
  });
  await page.route('**/api/usage/billing-tiers', async (route) => {
    await route.fulfill({ json: { count: 1, items: [{ method: 'GET', route_template: '/jobs', billing_tier: 'admin_free', unit_name: 'request', unit_price_inr: 0, description: 'List jobs' }] } });
  });
  await page.route('**/api/usage/export**', async (route) => {
    await route.fulfill({ json: { tenant_id: 'default', count: 0, items: [] } });
  });
  await page.route('**/api/jobs', async (route) => {
    if (route.request().method() === 'POST') {
      await route.fulfill({ json: { job_id: 'test', tenant_id: 'default', name: 'test', base_dir: 'data/test', created_at: '2026-05-18T00:00:00Z', updated_at: '2026-05-18T00:00:00Z', status: 'created', generation: 0 } });
      return;
    }
    await route.fulfill({ json: { count: 1, items: [{ job_id: 'test', tenant_id: 'default', name: 'test', base_dir: 'data/test', created_at: '2026-05-18T00:00:00Z', updated_at: '2026-05-18T00:00:00Z', status: 'queued', generation: 0 }] } });
  });
  await page.route('**/api/jobs/**/train/status', async (route) => {
    await route.fulfill({
      json: {
        generation: 0,
        stage: 'unknown',
        fitness: { prey: { best: 0, average: 0 }, predator: { best: 0, average: 0 } },
        learning: { adaptability: 0, meta_effectiveness: 0, performance_change: 0, instability: 0 },
        behavior: { success_rate: 0, stability: 0, novelty: 0 },
        diversity: { prey_species: 0, predator_species: 0 },
        neural_health: { dead_connections: 0, saturation: 0 },
        system: { evaluation_time_sec: 0, status: 'queued', uptime_seconds: 0, last_update: '2026-05-18T00:00:00Z' },
        status: 'queued',
        best_prey_fitness: 0,
        best_predator_fitness: 0,
        mean_prey_fitness: 0,
        mean_predator_fitness: 0,
        curriculum_stage: 'unknown',
        total_generations_trained: 0,
        uptime_seconds: 0,
        last_update: '2026-05-18T00:00:00Z',
      },
    });
  });
  await page.route('**/api/jobs/**/events**', async (route) => {
    await route.fulfill({ json: { count: 0, items: [] } });
  });
  await page.route('**/api/jobs/**/train/checkpoints**', async (route) => {
    await route.fulfill({ json: { count: 0, items: [] } });
  });
  await page.route('**/api/auth/keys', async (route) => {
    if (route.request().method() === 'POST') {
      await route.fulfill({
        json: {
          key: { key_id: 'key_1', name: 'test-key', tenant_id: 'default', status: 'active', role: 'reader', scopes: ['jobs:read'], created_at: '2026-05-18T00:00:00Z', updated_at: '2026-05-18T00:00:00Z', last_used_at: null, expires_at: null, revoked_at: null, expired_at: null, rotated_at: null, rotated_from_key_id: null, replaced_by_key_id: null },
          api_key: 'evm_key_1_secret',
        },
      });
      return;
    }
    await route.fulfill({ json: { tenant_id: 'default', count: 0, items: [] } });
  });
  await page.route('**/api/billing/account', async (route) => {
    await route.fulfill({ json: { tenant_id: 'default', currency: 'INR', available_credit_inr: 0, outstanding_amount_inr: 0, total_credited_inr: 0, total_debited_inr: 0, prepaid_required: false } });
  });
  await page.route('**/api/billing/ledger', async (route) => {
    await route.fulfill({ json: { tenant_id: 'default', count: 0, items: [] } });
  });
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('evomind.apiKey', 'evm_test_key'));
  await mockConsoleApi(page);
});

test('billing page exposes Razorpay checkout button', async ({ page }) => {
  await page.goto('/console/billing');
  await expect(page.getByRole('button', { name: /pay with razorpay/i })).toBeVisible();
});

test('operations page handles missing or empty API data', async ({ page }) => {
  await page.goto('/console/operations');
  await expect(page.getByRole('heading', { name: 'Operations' })).toBeVisible();
  await expect(page.getByText('training_worker')).toBeVisible();
});

test('api key page creates a key', async ({ page }) => {
  await page.goto('/console/api-keys');
  await page.getByPlaceholder('Name').fill('test-key');
  await page.getByRole('button', { name: /generate/i }).click();
  await expect(page.getByText('evm_key_1_secret')).toBeVisible();
});

test('training page shows queued state', async ({ page }) => {
  await page.goto('/console/training');
  await expect(page.getByRole('cell', { name: 'queued' })).toBeVisible();
});
