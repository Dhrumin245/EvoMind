import { useMetrics, useTrainInsights, useTrainStatus } from '../../api/hooks';
import { formatNumber } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { LineChartAutoRefresh } from '../common/LineChartAutoRefresh';
import { MetricCard } from '../common/MetricCard';

const fitnessSeries = [
  { key: 'best_prey_fitness', name: 'Best prey', color: '#0f766e' },
  { key: 'mean_prey_fitness', name: 'Avg prey', color: '#14b8a6' },
  { key: 'best_predator_fitness', name: 'Best predator', color: '#7c3aed' },
  { key: 'mean_predator_fitness', name: 'Avg predator', color: '#a78bfa' },
];

const diversitySeries = [
  { key: 'prey_species', name: 'Prey species', color: '#0ea5e9', type: 'area' as const },
  { key: 'predator_species', name: 'Predator species', color: '#f59e0b', type: 'area' as const },
];

export function MetricsDashboard() {
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const autoRefreshMs = useAuthStore((state) => state.autoRefreshMs);
  const status = useTrainStatus(selectedJobId, autoRefreshMs);
  const metrics = useMetrics(selectedJobId, autoRefreshMs);
  const insights = useTrainInsights(selectedJobId, 25, autoRefreshMs * 2);
  const rows = metrics.data?.items || [];
  const current = status.data;

  const learning = current?.learning;
  const behavior = current?.behavior;
  const neural = current?.neural_health;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">Metrics</h1>
          <p className="mt-1 text-sm text-muted-foreground">Auto-refreshing fitness, learning, behavior, diversity, and neural health.</p>
        </div>
        <p className="text-sm text-muted-foreground">Job {selectedJobId}</p>
      </div>

      <section className="rounded-md border border-border bg-card p-4 shadow-panel">
        <h2 className="text-base font-semibold">Fitness Curves</h2>
        <div className="mt-4">
          <LineChartAutoRefresh data={rows} series={fitnessSeries} height={320} />
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Learning Metrics</h2>
          <div className="mt-4 grid gap-3">
            {[
              ['Adaptability', learning?.adaptability],
              ['Meta effectiveness', learning?.meta_effectiveness],
              ['Performance change', learning?.performance_change],
              ['Instability', learning?.instability],
            ].map(([label, value]) => (
              <div key={String(label)}>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{label}</span>
                  <span className="font-medium">{formatNumber(Number(value))}</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-muted">
                  <div className="h-2 rounded-full bg-primary" style={{ width: `${Math.min(100, Math.abs(Number(value) || 0) * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-3 xl:grid-cols-1">
          <MetricCard label="Success rate" value={`${formatNumber((behavior?.success_rate || 0) * 100, 1)}%`} />
          <MetricCard label="Stability" value={`${formatNumber((behavior?.stability || 0) * 100, 1)}%`} />
          <MetricCard label="Novelty" value={`${formatNumber((behavior?.novelty || 0) * 100, 1)}%`} />
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Species Diversity</h2>
          <div className="mt-4">
            <LineChartAutoRefresh data={rows} series={diversitySeries} height={280} />
          </div>
        </div>
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Neural Health</h2>
          <div className="mt-5 space-y-5">
            <div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Dead connections</span>
                <span className="font-medium">{neural?.dead_connections ?? 0}</span>
              </div>
              <div className="mt-2 h-3 rounded-full bg-muted">
                <div className="h-3 rounded-full bg-destructive" style={{ width: `${Math.min(100, neural?.dead_connections || 0)}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Saturation</span>
                <span className="font-medium">{neural?.saturation ?? 0}</span>
              </div>
              <div className="mt-2 h-3 rounded-full bg-muted">
                <div className="h-3 rounded-full bg-warning" style={{ width: `${Math.min(100, neural?.saturation || 0)}%` }} />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-panel">
        <h2 className="text-base font-semibold">Training Insights</h2>
        <p className="mt-1 text-sm text-muted-foreground">Raw insight payload from the backend training runtime.</p>
        <pre className="mt-4 max-h-96 overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-50">
          {JSON.stringify(insights.data || {}, null, 2)}
        </pre>
      </section>
    </div>
  );
}
