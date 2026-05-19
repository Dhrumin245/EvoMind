# EvoMind

EvoMind is an evolutionary AI training control plane with a FastAPI backend, worker runtime, and React dashboard.

## Backend

Start the API locally:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Create an API key:

```bash
python -m api.auth create --name local-admin --tenant default
```

## Frontend

The dashboard lives in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

Vite serves the UI on `http://localhost:3000` and proxies `/api` to `http://localhost:8000`.

## Docker

Run the Linux container stack:

```bash
docker compose up --build
```

The frontend is exposed on port `3000` and the API on port `8000`. `deploy/Caddyfile` can be used by a Linux Caddy service to route `/api/*` to the API and all other paths to the frontend.
