import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { getErrorMessage } from '../../api/client';
import { useLogin, useRegister } from '../../api/hooks';
import { useAuthStore } from '../../store/authStore';

export function AuthPage() {
  const navigate = useNavigate();
  const login = useLogin();
  const register = useRegister();
  const setSession = useAuthStore((state) => state.setSession);
  const pushToast = useAuthStore((state) => state.pushToast);
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [tenantId, setTenantId] = useState('');
  const [name, setName] = useState('');

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const mutation = mode === 'login' ? login : register;
    mutation.mutate(
      mode === 'login'
        ? { email, password }
        : { email, password, tenant_id: tenantId || undefined, name: name || undefined },
      {
        onSuccess: (response) => {
          setSession(response.token, response.user.email, response.user.tenant_id);
          pushToast({ title: mode === 'login' ? 'Signed in' : 'Account created', description: response.user.tenant_id, tone: 'success' });
          navigate('/console');
        },
        onError: (error) => pushToast({ title: 'Authentication failed', description: getErrorMessage(error), tone: 'error' }),
      },
    );
  };

  const pending = login.isPending || register.isPending;

  return (
    <main className="grid min-h-screen place-items-center bg-background px-4 text-foreground">
      <form className="w-full max-w-md rounded-md border border-border bg-card p-6 shadow-panel" onSubmit={submit}>
        <Link className="text-sm text-muted-foreground hover:text-foreground" to="/">Back to website</Link>
        <h1 className="mt-4 text-2xl font-semibold tracking-normal">{mode === 'login' ? 'Sign in' : 'Create account'}</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          {mode === 'login' ? 'Use your EvoMind account to manage jobs, billing, and API keys.' : 'Create a tenant account with an admin session.'}
        </p>

        <div className="mt-6 grid gap-3">
          {mode === 'register' && (
            <>
              <input className="h-10 rounded-md border border-border bg-background px-3 text-sm" placeholder="Name" value={name} onChange={(event) => setName(event.target.value)} />
              <input className="h-10 rounded-md border border-border bg-background px-3 text-sm" placeholder="Tenant ID, optional" value={tenantId} onChange={(event) => setTenantId(event.target.value.toLowerCase())} />
            </>
          )}
          <input className="h-10 rounded-md border border-border bg-background px-3 text-sm" placeholder="Email" required type="email" value={email} onChange={(event) => setEmail(event.target.value)} />
          <input className="h-10 rounded-md border border-border bg-background px-3 text-sm" placeholder="Password" required type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          <button className="h-10 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50" disabled={pending} type="submit">
            {pending ? 'Please wait...' : mode === 'login' ? 'Sign in' : 'Register'}
          </button>
        </div>

        <button
          className="mt-4 text-sm text-primary"
          onClick={() => setMode((value) => (value === 'login' ? 'register' : 'login'))}
          type="button"
        >
          {mode === 'login' ? 'Need an account? Register' : 'Already have an account? Sign in'}
        </button>
      </form>
    </main>
  );
}
