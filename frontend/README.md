# EvoMind Frontend

React + TypeScript dashboard for EvoMind training operations.

## Local Development

1. Install dependencies:
   ```bash
   npm install
   ```
2. Set an API key in `frontend/.env.local` or paste it into the dashboard header.
3. Start Vite:
   ```bash
   npm run dev
   ```

Vite proxies `/api` to `http://localhost:8000` by default.

## Build

```bash
npm run build
```

The production Docker image serves the static build on port `3000` and proxies `/api` to the backend service.

## Production Configuration

Use `frontend/.env.production.example` as the template for frontend build-time configuration.

```bash
cp frontend/.env.production.example frontend/.env.production
```

Set `VITE_API_BASE_URL` to your deployed API origin, or keep `/api` when a reverse proxy routes API requests from the same domain.

Razorpay Checkout is loaded by the billing page at runtime from `https://checkout.razorpay.com/v1/checkout.js`. The secret Razorpay values must only be configured on the backend; never expose `EVOMIND_RAZORPAY_KEY_SECRET` in frontend env files.
