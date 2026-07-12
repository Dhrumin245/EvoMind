import { lazy, Suspense, type ReactNode } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { RouteErrorBoundary } from './components/common/RouteErrorBoundary';

const AgentPlayground = lazy(() => import('./components/pages/AgentPlayground').then((module) => ({ default: module.AgentPlayground })));
const ApiKeyManager = lazy(() => import('./components/pages/ApiKeyManager').then((module) => ({ default: module.ApiKeyManager })));
const AuthPage = lazy(() => import('./components/pages/AuthPage').then((module) => ({ default: module.AuthPage })));
const BillingDashboard = lazy(() => import('./components/pages/BillingDashboard').then((module) => ({ default: module.BillingDashboard })));
const Dashboard = lazy(() => import('./components/pages/Dashboard').then((module) => ({ default: module.Dashboard })));
const GenomeManager = lazy(() => import('./components/pages/GenomeManager').then((module) => ({ default: module.GenomeManager })));
const MetricsDashboard = lazy(() => import('./components/pages/MetricsDashboard').then((module) => ({ default: module.MetricsDashboard })));
const OperationsDashboard = lazy(() => import('./components/pages/OperationsDashboard').then((module) => ({ default: module.OperationsDashboard })));
const TrainingControl = lazy(() => import('./components/pages/TrainingControl').then((module) => ({ default: module.TrainingControl })));
const WebhookManager = lazy(() => import('./components/pages/WebhookManager').then((module) => ({ default: module.WebhookManager })));
const Website = lazy(() => import('./components/pages/Website').then((module) => ({ default: module.Website })));
const CampaignDashboard = lazy(() => import('./CampaignDashboard'));

function PageLoader() {
  return (
    <div className="grid min-h-screen place-items-center bg-background text-sm text-muted-foreground">
      Loading...
    </div>
  );
}

function withSuspense(element: ReactNode) {
  return <Suspense fallback={<PageLoader />}>{element}</Suspense>;
}

const router = createBrowserRouter([
  { path: '/', element: withSuspense(<Website />), errorElement: <RouteErrorBoundary /> },
  { path: '/campaign', element: withSuspense(<CampaignDashboard />), errorElement: <RouteErrorBoundary /> },
  { path: '/login', element: withSuspense(<AuthPage />), errorElement: <RouteErrorBoundary /> },
  {
    path: '/console',
    element: <Layout />,
    errorElement: <RouteErrorBoundary />,
    children: [
      { index: true, element: withSuspense(<Dashboard />) },
      { path: 'training', element: withSuspense(<TrainingControl />) },
      { path: 'metrics', element: withSuspense(<MetricsDashboard />) },
      { path: 'genomes', element: withSuspense(<GenomeManager />) },
      { path: 'agent', element: withSuspense(<AgentPlayground />) },
      { path: 'api-keys', element: withSuspense(<ApiKeyManager />) },
      { path: 'webhooks', element: withSuspense(<WebhookManager />) },
      { path: 'billing', element: withSuspense(<BillingDashboard />) },
      { path: 'operations', element: withSuspense(<OperationsDashboard />) },
    ],
  },
]);

export function App() {
  return <RouterProvider router={router} />;
}
