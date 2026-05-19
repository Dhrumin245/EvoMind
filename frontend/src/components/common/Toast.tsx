import { CheckCircle2, X, XCircle } from 'lucide-react';
import { useAuthStore } from '../../store/authStore';
import { cn } from '../../lib';

export function ToastViewport() {
  const toasts = useAuthStore((state) => state.toasts);
  const dismissToast = useAuthStore((state) => state.dismissToast);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-[min(360px,calc(100vw-2rem))] flex-col gap-2">
      {toasts.map((toast) => {
        const Icon = toast.tone === 'error' ? XCircle : CheckCircle2;
        return (
          <div
            key={toast.id}
            className={cn(
              'rounded-md border bg-card p-3 shadow-lg',
              toast.tone === 'error' ? 'border-red-200' : 'border-border',
            )}
          >
            <div className="flex items-start gap-3">
              <Icon className={cn('mt-0.5 h-4 w-4', toast.tone === 'error' ? 'text-destructive' : 'text-success')} />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">{toast.title}</p>
                {toast.description && <p className="mt-1 text-sm text-muted-foreground">{toast.description}</p>}
              </div>
              <button className="rounded-md p-1 text-muted-foreground hover:bg-muted" onClick={() => dismissToast(toast.id)} type="button">
                <X className="h-4 w-4" aria-hidden="true" />
                <span className="sr-only">Dismiss</span>
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
