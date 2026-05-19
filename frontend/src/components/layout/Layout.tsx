import { NavLink, Outlet } from 'react-router-dom';
import { Activity, BarChart3, Bot, CreditCard, Dna, Home, KeyRound, LayoutDashboard, LogOut, Menu, PlaySquare, ServerCog, UserRound, Webhook } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useJobs, useLogout } from '../../api/hooks';
import { useAuthStore } from '../../store/authStore';
import { cn } from '../../lib';
import { ToastViewport } from '../common/Toast';

const navItems = [
  { to: '/console', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/console/training', label: 'Training', icon: PlaySquare },
  { to: '/console/metrics', label: 'Metrics', icon: BarChart3 },
  { to: '/console/genomes', label: 'Genomes', icon: Dna },
  { to: '/console/agent', label: 'Agent', icon: Bot },
  { to: '/console/api-keys', label: 'API Keys', icon: KeyRound },
  { to: '/console/webhooks', label: 'Webhooks', icon: Webhook },
  { to: '/console/billing', label: 'Billing', icon: CreditCard },
  { to: '/console/operations', label: 'Operations', icon: ServerCog },
];

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const apiKey = useAuthStore((state) => state.apiKey);
  const sessionToken = useAuthStore((state) => state.sessionToken);
  const userEmail = useAuthStore((state) => state.userEmail);
  const tenantId = useAuthStore((state) => state.tenantId);
  const setApiKey = useAuthStore((state) => state.setApiKey);
  const selectedJobId = useAuthStore((state) => state.selectedJobId);
  const setSelectedJobId = useAuthStore((state) => state.setSelectedJobId);
  const clearAuth = useAuthStore((state) => state.clearAuth);
  const logout = useLogout();
  const jobs = useJobs();

  useEffect(() => {
    const items = jobs.data?.items || [];
    if (!items.length) {
      return;
    }
    if (!items.some((job) => job.job_id === selectedJobId)) {
      setSelectedJobId(items[0].job_id);
    }
  }, [jobs.data?.items, selectedJobId, setSelectedJobId]);

  const signOut = () => {
    logout.mutate(undefined, {
      onSettled: () => clearAuth(),
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-30 border-b border-border bg-card">
        <div className="flex h-16 items-center gap-3 px-4 lg:px-6">
          <button className="rounded-md border border-border p-2 lg:hidden" onClick={() => setSidebarOpen((value) => !value)} type="button">
            <Menu className="h-5 w-5" aria-hidden="true" />
            <span className="sr-only">Toggle navigation</span>
          </button>
          <div className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-md bg-primary text-primary-foreground">
              <Activity className="h-5 w-5" aria-hidden="true" />
            </span>
            <div>
              <p className="text-sm font-semibold">EvoMind</p>
              <p className="text-xs text-muted-foreground">Control Plane</p>
            </div>
          </div>
          <NavLink className="ml-auto hidden h-9 items-center gap-2 rounded-md border border-border px-3 text-sm text-muted-foreground hover:bg-muted hover:text-foreground sm:inline-flex" to="/">
            <Home className="h-4 w-4" aria-hidden="true" />
            Website
          </NavLink>
          {sessionToken ? (
            <div className="flex min-w-0 max-w-md items-center gap-2">
              <div className="hidden min-w-0 text-right text-xs sm:block">
                <p className="truncate font-medium">{userEmail}</p>
                <p className="truncate text-muted-foreground">{tenantId}</p>
              </div>
              <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={signOut} type="button">
                <LogOut className="h-4 w-4" aria-hidden="true" />
                Sign out
              </button>
            </div>
          ) : (
            <div className="flex w-full max-w-md items-center gap-2">
              <NavLink className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm text-muted-foreground hover:bg-muted hover:text-foreground" to="/login">
                <UserRound className="h-4 w-4" aria-hidden="true" />
                Login
              </NavLink>
              <input
                className="h-9 min-w-0 flex-1 rounded-md border border-border bg-background px-3 text-sm"
                onChange={(event) => setApiKey(event.target.value)}
                placeholder="API key"
                type="password"
                value={apiKey}
              />
            </div>
          )}
        </div>
      </header>

      <div className="flex">
        <aside
          className={cn(
            'fixed inset-y-16 left-0 z-20 w-64 border-r border-border bg-card p-3 transition-transform lg:sticky lg:top-16 lg:h-[calc(100vh-4rem)] lg:translate-x-0',
            sidebarOpen ? 'translate-x-0' : '-translate-x-full',
          )}
        >
          <nav className="space-y-1">
            {navItems.map((item) => (
              <NavLink
                className={({ isActive }) =>
                  cn(
                    'flex h-10 items-center gap-3 rounded-md px-3 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground',
                    isActive && 'bg-muted text-foreground',
                  )
                }
                key={item.to}
                onClick={() => setSidebarOpen(false)}
                to={item.to}
              >
                <item.icon className="h-4 w-4" aria-hidden="true" />
                {item.label}
              </NavLink>
            ))}
          </nav>
        </aside>
        <main className="min-w-0 flex-1 p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
      <ToastViewport />
    </div>
  );
}
