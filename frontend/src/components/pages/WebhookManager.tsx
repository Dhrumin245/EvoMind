import { FormEvent, useState } from 'react';
import { Trash2, Webhook } from 'lucide-react';
import { getErrorMessage } from '../../api/client';
import { useCreateWebhook, useDeleteWebhook, useWebhookDeliveries, useWebhooks } from '../../api/hooks';
import { formatDateTime } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { ConfirmDialog } from '../common/ConfirmDialog';
import { StatusBadge } from '../common/StatusBadge';

export function WebhookManager() {
  const pushToast = useAuthStore((state) => state.pushToast);
  const webhooks = useWebhooks();
  const createWebhook = useCreateWebhook();
  const deleteWebhook = useDeleteWebhook();
  const [url, setUrl] = useState('');
  const [description, setDescription] = useState('');
  const [events, setEvents] = useState('');
  const [secret, setSecret] = useState('');
  const [selectedWebhook, setSelectedWebhook] = useState('');
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const deliveries = useWebhookDeliveries(selectedWebhook);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    createWebhook.mutate(
      {
        url,
        description: description || undefined,
        subscribed_events: events ? events.split(',').map((item) => item.trim()).filter(Boolean) : [],
        secret: secret || undefined,
      },
      {
        onSuccess: (webhook) => {
          setUrl('');
          setDescription('');
          setEvents('');
          setSecret('');
          setSelectedWebhook(webhook.webhook_id);
          pushToast({ title: 'Webhook created', description: webhook.url, tone: 'success' });
        },
        onError: (error) => pushToast({ title: 'Webhook creation failed', description: getErrorMessage(error), tone: 'error' }),
      },
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-normal">Webhooks</h1>
        <p className="mt-1 text-sm text-muted-foreground">Register event targets and inspect delivery history.</p>
      </div>

      <section className="grid gap-4 xl:grid-cols-[0.85fr_1.35fr]">
        <form className="rounded-md border border-border bg-card p-4 shadow-panel" onSubmit={submit}>
          <div className="flex items-center gap-2">
            <Webhook className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="text-base font-semibold">Create Webhook</h2>
          </div>
          <div className="mt-4 grid gap-3">
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="https://example.com/evomind" required value={url} onChange={(event) => setUrl(event.target.value)} />
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Description" value={description} onChange={(event) => setDescription(event.target.value)} />
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Events, comma-separated. Empty means all." value={events} onChange={(event) => setEvents(event.target.value)} />
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Optional signing secret" type="password" value={secret} onChange={(event) => setSecret(event.target.value)} />
            <button className="h-9 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50" disabled={createWebhook.isPending} type="submit">
              {createWebhook.isPending ? 'Creating...' : 'Create Webhook'}
            </button>
          </div>
        </form>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Registered Webhooks</h2>
          <div className="mt-4 overflow-auto table-scroll">
            <table className="w-full min-w-[780px] text-left text-sm">
              <thead className="border-b border-border text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Target</th>
                  <th className="py-2 pr-3">Status</th>
                  <th className="py-2 pr-3">Events</th>
                  <th className="py-2 pr-3">Last delivery</th>
                  <th className="py-2 pr-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(webhooks.data?.items || []).map((item) => (
                  <tr className="border-b border-border last:border-0" key={item.webhook_id}>
                    <td className="py-3 pr-3">
                      <button className="break-all text-left font-medium hover:text-primary" onClick={() => setSelectedWebhook(item.webhook_id)} type="button">
                        {item.url}
                      </button>
                      <p className="mt-1 text-xs text-muted-foreground">{item.description || item.webhook_id}</p>
                    </td>
                    <td className="py-3 pr-3"><StatusBadge status={item.status} /></td>
                    <td className="py-3 pr-3 text-muted-foreground">{item.subscribed_events.length ? item.subscribed_events.join(', ') : 'All'}</td>
                    <td className="py-3 pr-3">
                      <p>{item.last_delivery_status || 'Never'}</p>
                      <p className="text-xs text-muted-foreground">{formatDateTime(item.last_delivery_at)}</p>
                    </td>
                    <td className="py-3 pr-3">
                      <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => setDeleteId(item.webhook_id)} type="button" title="Delete">
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!webhooks.data?.items.length && <p className="py-8 text-center text-sm text-muted-foreground">No webhooks registered.</p>}
          </div>
        </div>
      </section>

      <section className="rounded-md border border-border bg-card p-4 shadow-panel">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h2 className="text-base font-semibold">Delivery History</h2>
          <select className="h-9 rounded-md border border-border bg-background px-3 text-sm" value={selectedWebhook} onChange={(event) => setSelectedWebhook(event.target.value)}>
            <option value="">Select webhook</option>
            {(webhooks.data?.items || []).map((item) => (
              <option value={item.webhook_id} key={item.webhook_id}>{item.description || item.url}</option>
            ))}
          </select>
        </div>
        <div className="mt-4 overflow-auto table-scroll">
          <table className="w-full min-w-[780px] text-left text-sm">
            <thead className="border-b border-border text-xs uppercase text-muted-foreground">
              <tr>
                <th className="py-2 pr-3">Event</th>
                <th className="py-2 pr-3">Status</th>
                <th className="py-2 pr-3">Attempts</th>
                <th className="py-2 pr-3">Delivered</th>
                <th className="py-2 pr-3">Last error</th>
              </tr>
            </thead>
            <tbody>
              {(deliveries.data?.items || []).map((delivery) => (
                <tr className="border-b border-border last:border-0" key={delivery.delivery_id}>
                  <td className="py-3 pr-3">
                    <p className="font-medium">{delivery.event_type}</p>
                    <p className="text-xs text-muted-foreground">{delivery.job_id}</p>
                  </td>
                  <td className="py-3 pr-3"><StatusBadge status={delivery.status} /></td>
                  <td className="py-3 pr-3">{delivery.attempt_count}/{delivery.max_attempts}</td>
                  <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(delivery.delivered_at)}</td>
                  <td className="py-3 pr-3 text-muted-foreground">{delivery.last_error || 'None'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!selectedWebhook && <p className="py-8 text-center text-sm text-muted-foreground">Select a webhook to view deliveries.</p>}
          {selectedWebhook && !deliveries.data?.items.length && <p className="py-8 text-center text-sm text-muted-foreground">No deliveries recorded.</p>}
        </div>
      </section>

      <ConfirmDialog
        open={Boolean(deleteId)}
        title="Delete webhook?"
        description="Delivery history remains available in backend storage, but this target will stop receiving events."
        confirmLabel="Delete"
        pending={deleteWebhook.isPending}
        onCancel={() => setDeleteId(null)}
        onConfirm={() => {
          if (!deleteId) {
            return;
          }
          deleteWebhook.mutate(deleteId, {
            onSuccess: () => pushToast({ title: 'Webhook deleted', tone: 'success' }),
            onError: (error) => pushToast({ title: 'Delete failed', description: getErrorMessage(error), tone: 'error' }),
          });
          if (selectedWebhook === deleteId) {
            setSelectedWebhook('');
          }
          setDeleteId(null);
        }}
      />
    </div>
  );
}
