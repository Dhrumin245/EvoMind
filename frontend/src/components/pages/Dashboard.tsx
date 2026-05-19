import { Pause, Play, Save, Square } from 'lucide-react';
import { useState } from 'react';
import { getErrorMessage } from '../../api/client';
import { useCreateCheckpoint, useJobCommand, useJobEvents, useJobs, useTrainStatus } from '../../api/hooks';
import { formatDateTime, formatDuration, formatNumber } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { ConfirmDialog } from '../common/ConfirmDialog';
import { MetricCard } from '../common/MetricCard';
import { StatusBadge } from '../common/StatusBadge';

export function Dashboard() {
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const autoRefreshMs = useAuthStore((state) => state.autoRefreshMs);
  const pushToast = useAuthStore((state) => state.pushToast);
  const [confirmStop, setConfirmStop] = useState(false);
  const jobs = useJobs();
  const status = useTrainStatus(selectedJobId, autoRefreshMs);
  const events = useJobEvents(selectedJobId, autoRefreshMs);
  const command = useJobCommand(selectedJobId);
  const checkpoint = useCreateCheckpoint(selectedJobId);

  const current = status.data;

  const runCommand = (action: 'start' | 'stop') => {
    command.mutate(action, {
      onSuccess: () => pushToast({ title: `Training ${action} requested`, tone: 'success' }),
      onError: (error) => pushToast({ title: 'Training command failed', description: getErrorMessage(error), tone: 'error' }),
    });
  };

  const createCheckpoint = () => {
    checkpoint.mutate(undefined, {
      onSuccess: () => pushToast({ title: 'Checkpoint requested', tone: 'success' }),
      onError: (error) => pushToast({ title: 'Checkpoint failed', description: getErrorMessage(error), tone: 'error' }),
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">Dashboard</h1>
          <p className="mt-1 text-sm text-muted-foreground">Tenant training status, key metrics, and recent lifecycle events.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground" onClick={() => runCommand('start')} type="button">
            <Play className="h-4 w-4" aria-hidden="true" />
            Start
          </button>
          <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" disabled type="button">
            <Pause className="h-4 w-4" aria-hidden="true" />
            Pause
          </button>
          <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={() => setConfirmStop(true)} type="button">
            <Square className="h-4 w-4" aria-hidden="true" />
            Stop
          </button>
          <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={createCheckpoint} type="button">
            <Save className="h-4 w-4" aria-hidden="true" />
            Checkpoint
          </button>
        </div>
      </div>

      <section className="grid gap-4 lg:grid-cols-[1.2fr_2fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-base font-semibold">System Status</h2>
            <StatusBadge status={current?.system.status || current?.status || 'unknown'} />
          </div>
          <dl className="mt-4 grid gap-3 text-sm">
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Job</dt>
              <dd className="font-medium">{selectedJobId}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Uptime</dt>
              <dd className="font-medium">{formatDuration(current?.system.uptime_seconds || current?.uptime_seconds)}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Generation</dt>
              <dd className="font-medium">{current?.generation ?? 0}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Last update</dt>
              <dd className="text-right font-medium">{formatDateTime(current?.system.last_update || current?.last_update)}</dd>
            </div>
          </dl>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Best prey" value={formatNumber(current?.fitness.prey.best ?? current?.best_prey_fitness)} trend="up" />
          <MetricCard label="Best predator" value={formatNumber(current?.fitness.predator.best ?? current?.best_predator_fitness)} trend="up" />
          <MetricCard label="Average fitness" value={formatNumber(((current?.fitness.prey.average || 0) + (current?.fitness.predator.average || 0)) / 2)} />
          <MetricCard label="Learning score" value={formatNumber(current?.learning.meta_effectiveness)} detail={`Stage ${current?.stage || current?.curriculum_stage || 'unknown'}`} />
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Recent Job Events</h2>
          <div className="mt-3 max-h-96 overflow-auto table-scroll">
            {(events.data?.items || []).map((event) => (
              <div className="flex gap-3 border-b border-border py-3 last:border-0" key={event.event_id}>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{event.event_type}</p>
                  <p className="text-xs text-muted-foreground">{formatDateTime(event.created_at)}</p>
                </div>
                <StatusBadge status={event.job_id} />
              </div>
            ))}
            {!events.data?.items.length && <p className="py-8 text-center text-sm text-muted-foreground">No events available.</p>}
          </div>
        </div>
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Jobs</h2>
          <p className="mt-2 text-3xl font-semibold">{jobs.data?.count ?? 0}</p>
          <p className="mt-1 text-sm text-muted-foreground">Tracked jobs for the current tenant.</p>
        </div>
      </section>

      <ConfirmDialog
        open={confirmStop}
        title="Stop training?"
        description="This queues a stop command for the selected job. You can start training again later."
        confirmLabel="Stop"
        pending={command.isPending}
        onCancel={() => setConfirmStop(false)}
        onConfirm={() => {
          setConfirmStop(false);
          runCommand('stop');
        }}
      />
    </div>
  );
}
