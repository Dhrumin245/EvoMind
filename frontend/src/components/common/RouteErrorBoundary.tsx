import { isRouteErrorResponse, Link, useRouteError } from 'react-router-dom';

function errorMessage(error: unknown) {
  if (isRouteErrorResponse(error)) {
    return error.statusText || error.data?.message || `Request failed with ${error.status}`;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred.';
}

export function RouteErrorBoundary() {
  const error = useRouteError();

  return (
    <main className="grid min-h-screen place-items-center bg-background px-4 text-foreground">
      <section className="w-full max-w-xl rounded-md border border-border bg-card p-6 shadow-panel">
        <p className="text-sm font-semibold uppercase text-destructive">Application Error</p>
        <h1 className="mt-3 text-2xl font-semibold tracking-normal">This page could not be rendered.</h1>
        <p className="mt-3 text-sm leading-6 text-muted-foreground">{errorMessage(error)}</p>
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            className="h-9 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground"
            onClick={() => window.location.reload()}
            type="button"
          >
            Reload
          </button>
          <Link className="inline-flex h-9 items-center rounded-md border border-border px-3 text-sm" to="/console">
            Console
          </Link>
          <Link className="inline-flex h-9 items-center rounded-md border border-border px-3 text-sm" to="/">
            Website
          </Link>
        </div>
      </section>
    </main>
  );
}
