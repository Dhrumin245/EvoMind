import { ArrowDownRight, ArrowUpRight } from 'lucide-react';
import { cn } from '../../lib';

interface MetricCardProps {
  label: string;
  value: string | number;
  detail?: string;
  trend?: 'up' | 'down' | 'neutral';
}

export function MetricCard({ label, value, detail, trend = 'neutral' }: MetricCardProps) {
  const TrendIcon = trend === 'down' ? ArrowDownRight : ArrowUpRight;
  return (
    <div className="rounded-md border border-border bg-card p-4 shadow-panel">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-muted-foreground">{label}</p>
        {trend !== 'neutral' && (
          <TrendIcon
            className={cn('h-4 w-4', trend === 'up' ? 'text-success' : 'text-destructive')}
            aria-hidden="true"
          />
        )}
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-normal text-card-foreground">{value}</div>
      {detail && <p className="mt-1 text-sm text-muted-foreground">{detail}</p>}
    </div>
  );
}
