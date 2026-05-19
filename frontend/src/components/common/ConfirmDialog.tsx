import { AlertTriangle, X } from 'lucide-react';

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  pending?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = 'Confirm',
  pending = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/35 p-4">
      <div className="w-full max-w-md rounded-md border border-border bg-card p-5 shadow-xl">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-md bg-red-50 text-destructive">
              <AlertTriangle className="h-5 w-5" aria-hidden="true" />
            </span>
            <h2 className="text-base font-semibold">{title}</h2>
          </div>
          <button className="rounded-md p-1 text-muted-foreground hover:bg-muted" onClick={onCancel} type="button">
            <X className="h-4 w-4" aria-hidden="true" />
            <span className="sr-only">Close</span>
          </button>
        </div>
        <p className="mt-4 text-sm leading-6 text-muted-foreground">{description}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button className="rounded-md border border-border px-3 py-2 text-sm hover:bg-muted" onClick={onCancel} type="button">
            Cancel
          </button>
          <button
            className="rounded-md bg-destructive px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
            disabled={pending}
            onClick={onConfirm}
            type="button"
          >
            {pending ? 'Working...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
