import { cn } from '../../lib';

const statusClasses: Record<string, string> = {
  running: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  queued: 'border-sky-200 bg-sky-50 text-sky-700',
  paused: 'border-amber-200 bg-amber-50 text-amber-700',
  stopped: 'border-slate-200 bg-slate-50 text-slate-700',
  error: 'border-red-200 bg-red-50 text-red-700',
  active: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  revoked: 'border-slate-200 bg-slate-50 text-slate-700',
  expired: 'border-red-200 bg-red-50 text-red-700',
  rotated: 'border-violet-200 bg-violet-50 text-violet-700',
};

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  return (
    <span
      className={cn(
        'inline-flex h-6 items-center rounded-md border px-2 text-xs font-medium capitalize',
        statusClasses[normalized] || 'border-slate-200 bg-slate-50 text-slate-700',
      )}
    >
      {status}
    </span>
  );
}
