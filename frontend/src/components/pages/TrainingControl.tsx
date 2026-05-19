import { Clipboard, Eye, Pause, Play, RotateCcw, Save, Square } from 'lucide-react';
import { FormEvent, useMemo, useState } from 'react';
import { getErrorMessage } from '../../api/client';
import { useCheckpoints, useCreateCheckpoint, useCreateJob, useJobCommand, useJobEvents, useJobs, useTrainStatus } from '../../api/hooks';
import type { JobSummary } from '../../api/types';
import { formatDateTime } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { ConfirmDialog } from '../common/ConfirmDialog';
import { StatusBadge } from '../common/StatusBadge';

export function TrainingControl() {
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const setSelectedJobId = useAuthStore((state) => state.setSelectedJobId);
  const pushToast = useAuthStore((state) => state.pushToast);
  const [statusFilter, setStatusFilter] = useState('all');
  const [dateFrom, setDateFrom] = useState('');
  const [detailJob, setDetailJob] = useState<JobSummary | null>(null);
  const [confirmStop, setConfirmStop] = useState<string | null>(null);
  const [newJobName, setNewJobName] = useState('');
  const [newJobId, setNewJobId] = useState('');
  const [resumePath, setResumePath] = useState('');
  const [checkpointPath, setCheckpointPath] = useState('');

  const jobs = useJobs();
  const createJob = useCreateJob();
  const command = useJobCommand(selectedJobId);
  const checkpoint = useCreateCheckpoint(selectedJobId);
  const checkpoints = useCheckpoints(selectedJobId);
  const status = useTrainStatus(selectedJobId);
  const events = useJobEvents(selectedJobId);

  const filteredJobs = useMemo(() => {
    return (jobs.data?.items || []).filter((job) => {
      const statusMatches = statusFilter === 'all' || job.status === statusFilter;
      const dateMatches = !dateFrom || new Date(job.created_at) >= new Date(dateFrom);
      return statusMatches && dateMatches;
    });
  }, [dateFrom, jobs.data?.items, statusFilter]);

  const submitJob = (event: FormEvent) => {
    event.preventDefault();
    createJob.mutate(
      { name: newJobName || undefined, job_id: newJobId || undefined },
      {
        onSuccess: (job) => {
          setSelectedJobId(job.job_id);
          setNewJobName('');
          setNewJobId('');
          pushToast({ title: 'Job created', description: job.job_id, tone: 'success' });
        },
        onError: (error) => pushToast({ title: 'Job creation failed', description: getErrorMessage(error), tone: 'error' }),
      },
    );
  };

  const runCommand = (action: 'start' | 'stop' | { command: 'resume'; checkpoint_path: string }) => {
    command.mutate(action, {
      onSuccess: () => pushToast({ title: 'Training command queued', tone: 'success' }),
      onError: (error) => pushToast({ title: 'Training command failed', description: getErrorMessage(error), tone: 'error' }),
    });
  };

  const createCheckpoint = () => {
    checkpoint.mutate(checkpointPath || undefined, {
      onSuccess: (response) => pushToast({ title: 'Checkpoint created', description: response.checkpoint.checkpoint_path, tone: 'success' }),
      onError: (error) => pushToast({ title: 'Checkpoint failed', description: getErrorMessage(error), tone: 'error' }),
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-normal">Training Control</h1>
        <p className="mt-1 text-sm text-muted-foreground">Create jobs, queue training commands, and inspect job activity.</p>
      </div>

      <section className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-base font-semibold">Jobs</h2>
            <div className="flex flex-wrap gap-2">
              <select className="h-9 rounded-md border border-border bg-background px-3 text-sm" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                <option value="all">All statuses</option>
                <option value="running">Running</option>
                <option value="queued">Queued</option>
                <option value="stopped">Stopped</option>
                <option value="paused">Paused</option>
                <option value="error">Error</option>
              </select>
              <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} />
            </div>
          </div>
          <div className="mt-4 overflow-auto table-scroll">
            <table className="w-full min-w-[760px] border-collapse text-left text-sm">
              <thead className="border-b border-border text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Job ID</th>
                  <th className="py-2 pr-3">Status</th>
                  <th className="py-2 pr-3">Created</th>
                  <th className="py-2 pr-3">Stage</th>
                  <th className="py-2 pr-3">Last update</th>
                  <th className="py-2 pr-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredJobs.map((job) => (
                  <tr className="border-b border-border last:border-0" key={job.job_id}>
                    <td className="py-3 pr-3 font-medium">{job.job_id}</td>
                    <td className="py-3 pr-3"><StatusBadge status={job.status} /></td>
                    <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(job.created_at)}</td>
                    <td className="py-3 pr-3">{job.generation}</td>
                    <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(job.updated_at)}</td>
                    <td className="py-3 pr-3">
                      <div className="flex gap-1">
                        <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => setSelectedJobId(job.job_id)} type="button" title="Select">
                          <Play className="h-4 w-4" aria-hidden="true" />
                        </button>
                        <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => setDetailJob(job)} type="button" title="Details">
                          <Eye className="h-4 w-4" aria-hidden="true" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!filteredJobs.length && <p className="py-8 text-center text-sm text-muted-foreground">No jobs match the filters.</p>}
          </div>
        </div>

        <div className="space-y-4">
          <form className="rounded-md border border-border bg-card p-4 shadow-panel" onSubmit={submitJob}>
            <h2 className="text-base font-semibold">Start New Job</h2>
            <div className="mt-4 grid gap-3">
              <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Environment or job name" value={newJobName} onChange={(event) => setNewJobName(event.target.value)} />
              <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Optional job id" value={newJobId} onChange={(event) => setNewJobId(event.target.value)} />
              <select className="h-9 rounded-md border border-border bg-background px-3 text-sm" defaultValue="default">
                <option value="default">Default curriculum stage</option>
                <option value="arena">Arena pack</option>
                <option value="deterministic">Deterministic environment</option>
              </select>
              <button className="h-9 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50" disabled={createJob.isPending} type="submit">
                {createJob.isPending ? 'Creating...' : 'Create Job'}
              </button>
            </div>
          </form>

          <div className="rounded-md border border-border bg-card p-4 shadow-panel">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-base font-semibold">Selected Job</h2>
              <StatusBadge status={status.data?.status || 'unknown'} />
            </div>
            <p className="mt-2 text-sm text-muted-foreground">{selectedJobId || 'No job selected'}</p>
            <div className="mt-4 grid grid-cols-2 gap-2">
              <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50" disabled={!selectedJobId} onClick={() => runCommand('start')} type="button">
                <Play className="h-4 w-4" aria-hidden="true" />
                Start
              </button>
              <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border px-3 text-sm" disabled type="button">
                <Pause className="h-4 w-4" aria-hidden="true" />
                Pause
              </button>
              <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border px-3 text-sm disabled:opacity-50" disabled={!selectedJobId} onClick={() => setConfirmStop(selectedJobId)} type="button">
                <Square className="h-4 w-4" aria-hidden="true" />
                Stop
              </button>
              <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border px-3 text-sm disabled:opacity-50" disabled={!selectedJobId} onClick={createCheckpoint} type="button">
                <Save className="h-4 w-4" aria-hidden="true" />
                Checkpoint
              </button>
            </div>
            <div className="mt-3 grid gap-2">
              <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Checkpoint path for resume" value={resumePath} onChange={(event) => setResumePath(event.target.value)} />
              <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border px-3 text-sm disabled:opacity-50" onClick={() => runCommand({ command: 'resume', checkpoint_path: resumePath })} type="button" disabled={!selectedJobId || !resumePath}>
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
                Resume
              </button>
              <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Optional checkpoint save path" value={checkpointPath} onChange={(event) => setCheckpointPath(event.target.value)} />
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-panel">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-base font-semibold">Checkpoints</h2>
            <p className="mt-1 text-sm text-muted-foreground">Browse saved checkpoint markers and resume directly from a selected path.</p>
          </div>
          <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={() => checkpoints.refetch()} type="button">
            Refresh
          </button>
        </div>
        <div className="mt-4 overflow-auto table-scroll">
          <table className="w-full min-w-[860px] text-left text-sm">
            <thead className="border-b border-border text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3">Path</th>
                <th className="py-2 pr-3">Generation</th>
                <th className="py-2 pr-3">Saved</th>
                <th className="py-2 pr-3">Marker</th>
                <th className="py-2 pr-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {(checkpoints.data?.items || []).map((item) => (
                <tr className="border-b border-border last:border-0" key={item.checkpoint_path}>
                  <td className="max-w-md break-all py-3 pr-3 font-mono text-xs">{item.checkpoint_path}</td>
                  <td className="py-3 pr-3">{item.generation}</td>
                  <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(item.saved_at_utc)}</td>
                  <td className="py-3 pr-3"><StatusBadge status={item.marker_exists ? 'ready' : 'error'} /></td>
                  <td className="py-3 pr-3">
                    <div className="flex gap-2">
                      <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => setResumePath(item.checkpoint_path)} type="button" title="Use path">
                        <Clipboard className="h-4 w-4" aria-hidden="true" />
                      </button>
                      <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => runCommand({ command: 'resume', checkpoint_path: item.checkpoint_path })} type="button" title="Resume">
                        <RotateCcw className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!checkpoints.data?.items.length && <p className="py-8 text-center text-sm text-muted-foreground">No checkpoints found for this job.</p>}
        </div>
      </section>

      {detailJob && (
        <div className="fixed inset-0 z-40 grid place-items-center bg-slate-950/35 p-4">
          <div className="max-h-[85vh] w-full max-w-2xl overflow-auto rounded-md border border-border bg-card p-5 shadow-xl">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">{detailJob.name}</h2>
              <button className="rounded-md border border-border px-3 py-2 text-sm" onClick={() => setDetailJob(null)} type="button">Close</button>
            </div>
            <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
              <div><dt className="text-muted-foreground">Job ID</dt><dd className="font-medium">{detailJob.job_id}</dd></div>
              <div><dt className="text-muted-foreground">Status</dt><dd><StatusBadge status={detailJob.status} /></dd></div>
              <div><dt className="text-muted-foreground">Created</dt><dd>{formatDateTime(detailJob.created_at)}</dd></div>
              <div><dt className="text-muted-foreground">Updated</dt><dd>{formatDateTime(detailJob.updated_at)}</dd></div>
            </dl>
            <h3 className="mt-6 text-sm font-semibold">Recent Events</h3>
            <div className="mt-2 space-y-2">
              {(events.data?.items || []).slice(0, 8).map((event) => (
                <div className="rounded-md border border-border p-3 text-sm" key={event.event_id}>
                  <p className="font-medium">{event.event_type}</p>
                  <p className="text-xs text-muted-foreground">{formatDateTime(event.created_at)}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={Boolean(confirmStop)}
        title="Stop training?"
        description={`Queue a stop command for ${confirmStop || selectedJobId}.`}
        confirmLabel="Stop"
        pending={command.isPending}
        onCancel={() => setConfirmStop(null)}
        onConfirm={() => {
          setConfirmStop(null);
          runCommand('stop');
        }}
      />
    </div>
  );
}
