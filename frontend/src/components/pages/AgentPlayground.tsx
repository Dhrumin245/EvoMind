import { Bot, Play, Send } from 'lucide-react';
import { FormEvent, useMemo, useState } from 'react';
import { getErrorMessage } from '../../api/client';
import { useAgentAction, useAgentInfo, useBatchAgentAction } from '../../api/hooks';
import type { GenomeType } from '../../api/types';
import { formatNumber } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { MetricCard } from '../common/MetricCard';
import { StatusBadge } from '../common/StatusBadge';

function parseVector(value: string): number[] {
  const trimmed = value.trim();
  if (!trimmed) {
    return [];
  }
  const parsed = trimmed.startsWith('[')
    ? JSON.parse(trimmed)
    : trimmed.split(',').map((part) => Number(part.trim()));
  if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== 'number' || Number.isNaN(item))) {
    throw new Error('Observation must be a numeric array or comma-separated numbers.');
  }
  return parsed;
}

function parseBatch(value: string): number[][] {
  const parsed = JSON.parse(value);
  if (
    !Array.isArray(parsed) ||
    parsed.some((row) => !Array.isArray(row) || row.some((item) => typeof item !== 'number' || Number.isNaN(item)))
  ) {
    throw new Error('Batch input must be a JSON array of numeric arrays.');
  }
  return parsed;
}

export function AgentPlayground() {
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const pushToast = useAuthStore((state) => state.pushToast);
  const [genomeType, setGenomeType] = useState<GenomeType>('prey');
  const [generation, setGeneration] = useState('');
  const [maxActionLength, setMaxActionLength] = useState('4');
  const [observation, setObservation] = useState('0.2, 0.8, 0.1');
  const [batchInput, setBatchInput] = useState('[[0.2,0.8,0.1],[0.4,0.1,0.9]]');
  const agentInfo = useAgentInfo(selectedJobId, genomeType, generation ? Number(generation) : null);
  const action = useAgentAction(selectedJobId);
  const batchAction = useBatchAgentAction(selectedJobId);

  const selectedGeneration = useMemo(() => {
    if (!generation) {
      return undefined;
    }
    const value = Number(generation);
    return Number.isFinite(value) ? value : undefined;
  }, [generation]);

  const submitAction = (event: FormEvent) => {
    event.preventDefault();
    try {
      action.mutate(
        {
          observation: parseVector(observation),
          genome_type: genomeType,
          generation: selectedGeneration,
          max_action_length: Number(maxActionLength) || 4,
        },
        {
          onError: (error) => pushToast({ title: 'Agent request failed', description: getErrorMessage(error), tone: 'error' }),
        },
      );
    } catch (error) {
      pushToast({ title: 'Invalid observation', description: getErrorMessage(error), tone: 'error' });
    }
  };

  const submitBatch = (event: FormEvent) => {
    event.preventDefault();
    try {
      batchAction.mutate(
        {
          observations: parseBatch(batchInput),
          genome_type: genomeType,
          generation: selectedGeneration,
          max_action_length: Number(maxActionLength) || 4,
        },
        {
          onError: (error) => pushToast({ title: 'Batch request failed', description: getErrorMessage(error), tone: 'error' }),
        },
      );
    } catch (error) {
      pushToast({ title: 'Invalid batch input', description: getErrorMessage(error), tone: 'error' });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">Agent Playground</h1>
          <p className="mt-1 text-sm text-muted-foreground">Call agent info, single action, and batch action endpoints for the selected job.</p>
        </div>
        <p className="text-sm text-muted-foreground">Job {selectedJobId}</p>
      </div>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Available" value={agentInfo.data?.available ? 'Yes' : 'No'} />
        <MetricCard label="Genome" value={agentInfo.data?.genome_id || 'None'} />
        <MetricCard label="Fitness" value={formatNumber(agentInfo.data?.fitness)} />
        <MetricCard label="Generation" value={agentInfo.data?.generation ?? 0} />
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <form className="rounded-md border border-border bg-card p-4 shadow-panel" onSubmit={submitAction}>
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="text-base font-semibold">Single Action</h2>
          </div>
          <div className="mt-4 grid gap-3">
            <select className="h-9 rounded-md border border-border bg-background px-3 text-sm" value={genomeType} onChange={(event) => setGenomeType(event.target.value as GenomeType)}>
              <option value="prey">Prey genome</option>
              <option value="predator">Predator genome</option>
            </select>
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Optional generation" value={generation} onChange={(event) => setGeneration(event.target.value)} />
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" max="10" min="1" type="number" value={maxActionLength} onChange={(event) => setMaxActionLength(event.target.value)} />
            <textarea className="min-h-28 rounded-md border border-border bg-background px-3 py-2 text-sm" value={observation} onChange={(event) => setObservation(event.target.value)} />
            <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50" disabled={action.isPending} type="submit">
              <Play className="h-4 w-4" aria-hidden="true" />
              {action.isPending ? 'Running...' : 'Run Action'}
            </button>
          </div>
        </form>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-base font-semibold">Action Response</h2>
            <StatusBadge status={action.data ? 'ready' : 'idle'} />
          </div>
          <pre className="mt-4 max-h-96 overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-50">
            {JSON.stringify(action.data || { action: [], genome_id: null }, null, 2)}
          </pre>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <form className="rounded-md border border-border bg-card p-4 shadow-panel" onSubmit={submitBatch}>
          <div className="flex items-center gap-2">
            <Send className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="text-base font-semibold">Batch Actions</h2>
          </div>
          <textarea className="mt-4 min-h-40 w-full rounded-md border border-border bg-background px-3 py-2 text-sm" value={batchInput} onChange={(event) => setBatchInput(event.target.value)} />
          <button className="mt-3 inline-flex h-9 items-center justify-center gap-2 rounded-md border border-border px-3 text-sm disabled:opacity-50" disabled={batchAction.isPending} type="submit">
            Run Batch
          </button>
        </form>
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Batch Response</h2>
          <pre className="mt-4 max-h-96 overflow-auto rounded-md bg-slate-950 p-4 text-xs text-slate-50">
            {JSON.stringify(batchAction.data || { actions: [] }, null, 2)}
          </pre>
        </div>
      </section>
    </div>
  );
}
