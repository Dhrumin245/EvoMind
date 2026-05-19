import { Clipboard, KeyRound, Trash2 } from 'lucide-react';
import { FormEvent, useState } from 'react';
import { getErrorMessage } from '../../api/client';
import { useApiKeys, useCreateApiKey, useDeleteApiKey } from '../../api/hooks';
import { formatDateTime } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { ConfirmDialog } from '../common/ConfirmDialog';
import { StatusBadge } from '../common/StatusBadge';

export function ApiKeyManager() {
  const apiKeys = useApiKeys();
  const createApiKey = useCreateApiKey();
  const deleteApiKey = useDeleteApiKey();
  const pushToast = useAuthStore((state) => state.pushToast);
  const [name, setName] = useState('');
  const [role, setRole] = useState('reader');
  const [scopes, setScopes] = useState('');
  const [newKey, setNewKey] = useState('');
  const [revokeId, setRevokeId] = useState<string | null>(null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    createApiKey.mutate(
      {
        name,
        role,
        scopes: scopes ? scopes.split(',').map((scope) => scope.trim()).filter(Boolean) : undefined,
      },
      {
        onSuccess: (response) => {
          setName('');
          setScopes('');
          setNewKey(response.api_key);
          pushToast({ title: 'API key created', description: response.key.name, tone: 'success' });
        },
        onError: (error) => pushToast({ title: 'API key creation failed', description: getErrorMessage(error), tone: 'error' }),
      },
    );
  };

  const copy = async (value: string) => {
    await navigator.clipboard.writeText(value);
    pushToast({ title: 'Copied to clipboard', tone: 'success' });
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-normal">API Keys</h1>
        <p className="mt-1 text-sm text-muted-foreground">Create, copy, and revoke tenant-scoped credentials.</p>
      </div>

      {newKey && (
        <section className="rounded-md border border-emerald-200 bg-emerald-50 p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-emerald-800">New key</h2>
              <p className="mt-1 break-all font-mono text-sm text-emerald-900">{newKey}</p>
            </div>
            <button className="inline-flex h-9 items-center gap-2 rounded-md border border-emerald-300 px-3 text-sm text-emerald-900" onClick={() => copy(newKey)} type="button">
              <Clipboard className="h-4 w-4" aria-hidden="true" />
              Copy
            </button>
          </div>
        </section>
      )}

      <section className="grid gap-4 xl:grid-cols-[1fr_1.5fr]">
        <form className="rounded-md border border-border bg-card p-4 shadow-panel" onSubmit={submit}>
          <h2 className="text-base font-semibold">Generate Key</h2>
          <div className="mt-4 grid gap-3">
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Name" required value={name} onChange={(event) => setName(event.target.value)} />
            <select className="h-9 rounded-md border border-border bg-background px-3 text-sm" value={role} onChange={(event) => setRole(event.target.value)}>
              <option value="reader">Reader</option>
              <option value="operator">Operator</option>
              <option value="admin">Admin</option>
            </select>
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" placeholder="Optional scopes, comma-separated" value={scopes} onChange={(event) => setScopes(event.target.value)} />
            <button className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50" disabled={createApiKey.isPending} type="submit">
              <KeyRound className="h-4 w-4" aria-hidden="true" />
              {createApiKey.isPending ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </form>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Existing Keys</h2>
          <div className="mt-4 overflow-auto table-scroll">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="border-b border-border text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Key</th>
                  <th className="py-2 pr-3">Status</th>
                  <th className="py-2 pr-3">Role</th>
                  <th className="py-2 pr-3">Created</th>
                  <th className="py-2 pr-3">Last used</th>
                  <th className="py-2 pr-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {(apiKeys.data?.items || []).map((key) => (
                  <tr className="border-b border-border last:border-0" key={key.key_id}>
                    <td className="py-3 pr-3">
                      <p className="font-medium">{key.name}</p>
                      <p className="font-mono text-xs text-muted-foreground">{key.key_id.slice(0, 16)}...</p>
                    </td>
                    <td className="py-3 pr-3"><StatusBadge status={key.status} /></td>
                    <td className="py-3 pr-3 capitalize">{key.role}</td>
                    <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(key.created_at)}</td>
                    <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(key.last_used_at)}</td>
                    <td className="py-3 pr-3">
                      <button className="rounded-md border border-border p-2 hover:bg-muted" onClick={() => setRevokeId(key.key_id)} type="button" title="Revoke">
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!apiKeys.data?.items.length && <p className="py-8 text-center text-sm text-muted-foreground">No API keys available.</p>}
          </div>
        </div>
      </section>

      <ConfirmDialog
        open={Boolean(revokeId)}
        title="Revoke API key?"
        description="This immediately disables the credential. Existing clients using this key will fail authentication."
        confirmLabel="Revoke"
        pending={deleteApiKey.isPending}
        onCancel={() => setRevokeId(null)}
        onConfirm={() => {
          if (!revokeId) {
            return;
          }
          deleteApiKey.mutate(revokeId, {
            onSuccess: () => pushToast({ title: 'API key revoked', tone: 'success' }),
            onError: (error) => pushToast({ title: 'Revoke failed', description: getErrorMessage(error), tone: 'error' }),
          });
          setRevokeId(null);
        }}
      />
    </div>
  );
}
