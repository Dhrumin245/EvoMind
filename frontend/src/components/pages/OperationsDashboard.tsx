import { Download, HeartPulse, ServerCog } from 'lucide-react';
import { apiClient, getErrorMessage } from '../../api/client';
import { useBillingTiers, useHealth, useReadiness, useUsageExport, useUsageLimits, useUsageSummary } from '../../api/hooks';
import { formatDuration, formatNumber } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { MetricCard } from '../common/MetricCard';
import { StatusBadge } from '../common/StatusBadge';

export function OperationsDashboard() {
  const pushToast = useAuthStore((state) => state.pushToast);
  const health = useHealth();
  const readiness = useReadiness();
  const limits = useUsageLimits();
  const summary = useUsageSummary();
  const tiers = useBillingTiers();
  const usageExport = useUsageExport(7, 100);
  const readinessComponents = readiness.data?.components || [];
  const billingTierItems = tiers.data?.items || [];
  const usageRows = usageExport.data?.items || [];

  const exportCsv = async () => {
    try {
      const response = await apiClient.get('/usage/export', { params: { format: 'csv', days: 7 }, responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'evomind-usage-7d.csv';
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      pushToast({ title: 'Usage export failed', description: getErrorMessage(error), tone: 'error' });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">Operations</h1>
          <p className="mt-1 text-sm text-muted-foreground">Health, readiness, usage limits, billing tiers, and recent API usage.</p>
        </div>
        <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={exportCsv} type="button">
          <Download className="h-4 w-4" aria-hidden="true" />
          Export 7d CSV
        </button>
      </div>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Health" value={health.data?.status || 'unknown'} detail={health.data?.message} />
        <MetricCard label="Readiness" value={readiness.data?.status || 'unknown'} detail={readiness.data?.message} />
        <MetricCard label="Uptime" value={formatDuration(health.data?.uptime_seconds)} />
        <MetricCard label="Tenant" value={summary.data?.tenant_id || limits.data?.tenant_id || 'unknown'} />
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <div className="flex items-center gap-2">
            <HeartPulse className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="text-base font-semibold">Readiness Components</h2>
          </div>
          <div className="mt-4 space-y-3">
            {readinessComponents.map((component) => (
              <div className="flex items-start justify-between gap-3 rounded-md border border-border p-3" key={component.name}>
                <div>
                  <p className="font-medium">{component.name}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{component.detail || 'No details reported.'}</p>
                </div>
                <StatusBadge status={component.healthy ? 'ready' : 'error'} />
              </div>
            ))}
            {!readinessComponents.length && <p className="py-6 text-center text-sm text-muted-foreground">No readiness data available.</p>}
          </div>
        </div>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <div className="flex items-center gap-2">
            <ServerCog className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="text-base font-semibold">Usage Summary</h2>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <MetricCard label="Last minute" value={`${summary.data?.requests_last_minute ?? 0}/${summary.data?.requests_per_minute_limit ?? limits.data?.requests_per_minute ?? 0}`} />
            <MetricCard label="Today" value={`${summary.data?.requests_last_day ?? 0}/${summary.data?.requests_per_day_limit ?? limits.data?.requests_per_day ?? 0}`} />
            <MetricCard label="Remaining minute" value={summary.data?.remaining_this_minute ?? 0} />
            <MetricCard label="Remaining today" value={summary.data?.remaining_today ?? 0} />
            <MetricCard label="Total requests" value={summary.data?.requests_total ?? 0} />
            <MetricCard label="Max jobs" value={summary.data?.max_jobs ?? limits.data?.max_jobs ?? 0} />
            <MetricCard label="Cost today" value={`INR ${formatNumber(summary.data?.estimated_cost_last_day_inr)}`} />
            <MetricCard label="Cost total" value={`INR ${formatNumber(summary.data?.estimated_cost_total_inr)}`} />
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Billing Tiers</h2>
          <div className="mt-4 max-h-96 overflow-auto table-scroll">
            <table className="w-full min-w-[680px] text-left text-sm">
              <thead className="border-b border-border text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Route</th>
                  <th className="py-2 pr-3">Tier</th>
                  <th className="py-2 pr-3">Unit</th>
                  <th className="py-2 pr-3">INR</th>
                </tr>
              </thead>
              <tbody>
                {billingTierItems.map((tier) => (
                  <tr className="border-b border-border last:border-0" key={`${tier.method}-${tier.route_template}`}>
                    <td className="py-3 pr-3">
                      <p className="font-mono text-xs">{tier.method} {tier.route_template}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{tier.description}</p>
                    </td>
                    <td className="py-3 pr-3">{tier.billing_tier}</td>
                    <td className="py-3 pr-3">{tier.unit_name}</td>
                    <td className="py-3 pr-3">{formatNumber(tier.unit_price_inr, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!billingTierItems.length && <p className="py-8 text-center text-sm text-muted-foreground">No billing tiers available.</p>}
          </div>
        </div>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Recent Usage Export</h2>
          <div className="mt-4 max-h-96 overflow-auto table-scroll">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="border-b border-border text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Path</th>
                  <th className="py-2 pr-3">Status</th>
                  <th className="py-2 pr-3">Duration</th>
                  <th className="py-2 pr-3">Cost</th>
                </tr>
              </thead>
              <tbody>
                {usageRows.map((row, index) => (
                  <tr className="border-b border-border last:border-0" key={`${row.created_at}-${index}`}>
                    <td className="py-3 pr-3">
                      <p className="font-mono text-xs">{row.method} {row.route_template || row.path}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{row.created_at}</p>
                    </td>
                    <td className="py-3 pr-3">{row.status_code}</td>
                    <td className="py-3 pr-3">{formatNumber(row.duration_ms, 1)} ms</td>
                    <td className="py-3 pr-3">INR {formatNumber(row.estimated_cost_inr, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!usageRows.length && <p className="py-8 text-center text-sm text-muted-foreground">No usage rows available.</p>}
          </div>
        </div>
      </section>
    </div>
  );
}
