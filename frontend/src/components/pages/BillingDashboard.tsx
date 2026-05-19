import { Download } from 'lucide-react';
import { useState } from 'react';
import { apiClient, getErrorMessage } from '../../api/client';
import { useBillingAccount, useBillingLedger, useConfirmTopup, useCreateTopup, useJobs } from '../../api/hooks';
import type { BillingTopupConfirmResponse, BillingTopupResponse } from '../../api/types';
import { formatDateTime, formatNumber } from '../../lib';
import { useAuthStore } from '../../store/authStore';
import { MetricCard } from '../common/MetricCard';

const RAZORPAY_CHECKOUT_SRC = 'https://checkout.razorpay.com/v1/checkout.js';

type RazorpaySuccessResponse = {
  razorpay_payment_id: string;
  razorpay_order_id: string;
  razorpay_signature: string;
};

type RazorpayFailureResponse = {
  error?: {
    code?: string;
    description?: string;
    reason?: string;
    metadata?: {
      order_id?: string;
      payment_id?: string;
    };
  };
};

type RazorpayCheckoutOptions = {
  key: string;
  amount: number;
  currency: string;
  name: string;
  description: string;
  order_id: string;
  handler: (response: RazorpaySuccessResponse) => void | Promise<void>;
  notes?: Record<string, string>;
  theme?: { color?: string };
  modal?: { ondismiss?: () => void };
};

type RazorpayCheckout = {
  open: () => void;
  on: (event: 'payment.failed', callback: (response: RazorpayFailureResponse) => void) => void;
};

declare global {
  interface Window {
    Razorpay?: new (options: RazorpayCheckoutOptions) => RazorpayCheckout;
  }
}

let razorpayScriptPromise: Promise<void> | null = null;

function loadRazorpayCheckout() {
  if (window.Razorpay) {
    return Promise.resolve();
  }

  if (!razorpayScriptPromise) {
    razorpayScriptPromise = new Promise((resolve, reject) => {
      const existing = document.querySelector<HTMLScriptElement>(`script[src="${RAZORPAY_CHECKOUT_SRC}"]`);
      if (existing) {
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener('error', () => reject(new Error('Unable to load Razorpay Checkout.')), { once: true });
        return;
      }

      const script = document.createElement('script');
      script.src = RAZORPAY_CHECKOUT_SRC;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Unable to load Razorpay Checkout.'));
      document.body.appendChild(script);
    });
  }

  return razorpayScriptPromise;
}

export function BillingDashboard() {
  const account = useBillingAccount();
  const ledger = useBillingLedger();
  const jobs = useJobs();
  const createTopupMutation = useCreateTopup();
  const confirmTopupMutation = useConfirmTopup();
  const pushToast = useAuthStore((state) => state.pushToast);
  const [amount, setAmount] = useState('1000');
  const [topup, setTopup] = useState<BillingTopupResponse | null>(null);
  const [lastPayment, setLastPayment] = useState<BillingTopupConfirmResponse | null>(null);
  const [isCheckoutOpen, setIsCheckoutOpen] = useState(false);

  const openRazorpayCheckout = async (order: BillingTopupResponse) => {
    if (!order.checkout_key_id) {
      throw new Error('Razorpay checkout key is missing. Check EVOMIND_RAZORPAY_KEY_ID.');
    }

    await loadRazorpayCheckout();
    if (!window.Razorpay) {
      throw new Error('Razorpay Checkout did not initialize.');
    }

    const checkout = new window.Razorpay({
      key: order.checkout_key_id,
      amount: Math.round(order.amount_inr * 100),
      currency: order.currency || 'INR',
      name: 'EvoMind',
      description: `Prepaid credit top-up ${order.receipt}`,
      order_id: order.provider_order_id,
      notes: {
        tenant_id: order.tenant_id,
        topup_id: order.topup_id,
        receipt: order.receipt,
      },
      theme: {
        color: '#00f5a0',
      },
      modal: {
        ondismiss: () => {
          setIsCheckoutOpen(false);
          pushToast({ title: 'Payment cancelled', tone: 'default' });
        },
      },
      handler: async (response) => {
        try {
          const confirmation = await confirmTopupMutation.mutateAsync({
            razorpay_order_id: response.razorpay_order_id,
            razorpay_payment_id: response.razorpay_payment_id,
            razorpay_signature: response.razorpay_signature,
          });
          setLastPayment(confirmation);
          pushToast({
            title: confirmation.credited ? 'Payment verified' : 'Payment already processed',
            description: `Balance INR ${formatNumber(confirmation.balance_inr)}`,
            tone: 'success',
          });
        } catch (error) {
          pushToast({ title: 'Payment verification failed', description: getErrorMessage(error), tone: 'error' });
        } finally {
          setIsCheckoutOpen(false);
        }
      },
    });

    checkout.on('payment.failed', (response) => {
      setIsCheckoutOpen(false);
      pushToast({
        title: 'Payment failed',
        description: response.error?.description || response.error?.reason || response.error?.code || 'Razorpay reported a failed payment.',
        tone: 'error',
      });
    });

    setIsCheckoutOpen(true);
    checkout.open();
  };

  const startPayment = async () => {
    try {
      const numericAmount = Number(amount);
      if (!Number.isFinite(numericAmount) || numericAmount < 10) {
        throw new Error('Top-up amount must be at least INR 10.');
      }

      const order = await createTopupMutation.mutateAsync({
        amount_inr: numericAmount,
        description: 'Dashboard top-up',
      });
      setTopup(order);
      setLastPayment(null);
      await openRazorpayCheckout(order);
    } catch (error) {
      setIsCheckoutOpen(false);
      pushToast({ title: 'Payment could not start', description: getErrorMessage(error), tone: 'error' });
    }
  };

  const exportCsv = async () => {
    const response = await apiClient.get('/usage/export', { params: { format: 'csv', days: 30 }, responseType: 'blob' });
    const url = URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'evomind-usage.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-normal">Billing</h1>
          <p className="mt-1 text-sm text-muted-foreground">Prepaid balance, Razorpay checkout, usage indicators, and transaction history.</p>
        </div>
        <button className="inline-flex h-9 items-center gap-2 rounded-md border border-border px-3 text-sm" onClick={exportCsv} type="button">
          <Download className="h-4 w-4" aria-hidden="true" />
          Export CSV
        </button>
      </div>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Available balance" value={`INR ${formatNumber(account.data?.available_credit_inr)}`} />
        <MetricCard label="Outstanding" value={`INR ${formatNumber(account.data?.outstanding_amount_inr)}`} />
        <MetricCard label="Jobs run" value={jobs.data?.count ?? 0} />
        <MetricCard label="Compute hours" value="0" detail="Worker runtime export pending" />
      </section>

      <section className="grid gap-4 xl:grid-cols-[0.8fr_1.6fr]">
        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Top Up</h2>
          <p className="mt-1 text-sm text-muted-foreground">Create a Razorpay order and complete payment in the Checkout popup.</p>
          <div className="mt-4 grid gap-3">
            <input className="h-9 rounded-md border border-border bg-background px-3 text-sm" min="10" step="1" type="number" value={amount} onChange={(event) => setAmount(event.target.value)} />
            <button
              className="h-9 rounded-md bg-primary px-3 text-sm font-medium text-primary-foreground disabled:opacity-50"
              disabled={createTopupMutation.isPending || confirmTopupMutation.isPending || isCheckoutOpen}
              onClick={startPayment}
              type="button"
            >
              {createTopupMutation.isPending || isCheckoutOpen ? 'Opening Checkout...' : 'Pay with Razorpay'}
            </button>
          </div>
          {topup && (
            <div className="mt-4 rounded-md border border-border bg-background p-3 text-sm">
              <p className="font-medium">Order {topup.provider_order_id}</p>
              <p className="mt-1 text-muted-foreground">Receipt {topup.receipt} - {topup.currency} {formatNumber(topup.amount_inr)}</p>
              <p className="mt-1 text-muted-foreground">Checkout key {topup.checkout_key_id || 'not configured'}</p>
            </div>
          )}
          {lastPayment && (
            <div className="mt-4 rounded-md border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-900">
              <p className="font-medium">{lastPayment.credited ? 'Payment credited' : 'Payment already processed'}</p>
              <p className="mt-1">Payment {lastPayment.payment_id || 'unknown'} - Balance INR {formatNumber(lastPayment.balance_inr)}</p>
            </div>
          )}
        </div>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel">
          <h2 className="text-base font-semibold">Checkout Status</h2>
          <p className="mt-1 text-sm text-muted-foreground">Successful Checkout responses are verified by the backend before credit is added.</p>
          <div className="mt-4 grid gap-3 text-sm">
            <div className="flex justify-between gap-3">
              <span className="text-muted-foreground">Order</span>
              <span className="break-all text-right font-mono">{topup?.provider_order_id || 'Not created'}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-muted-foreground">Verification</span>
              <span>{confirmTopupMutation.isPending ? 'Verifying...' : lastPayment ? 'Verified' : 'Waiting'}</span>
            </div>
            <div className="flex justify-between gap-3">
              <span className="text-muted-foreground">Provider</span>
              <span>{topup?.provider || 'Razorpay'}</span>
            </div>
          </div>
        </div>

        <div className="rounded-md border border-border bg-card p-4 shadow-panel xl:col-span-2">
          <h2 className="text-base font-semibold">Transaction Ledger</h2>
          <div className="mt-4 overflow-auto table-scroll">
            <table className="w-full min-w-[680px] text-left text-sm">
              <thead className="border-b border-border text-xs uppercase text-muted-foreground">
                <tr>
                  <th className="py-2 pr-3">Date</th>
                  <th className="py-2 pr-3">Type</th>
                  <th className="py-2 pr-3">Amount</th>
                  <th className="py-2 pr-3">Balance</th>
                  <th className="py-2 pr-3">Description</th>
                </tr>
              </thead>
              <tbody>
                {(ledger.data?.items || []).map((entry) => (
                  <tr className="border-b border-border last:border-0" key={entry.entry_id}>
                    <td className="py-3 pr-3 text-muted-foreground">{formatDateTime(entry.created_at)}</td>
                    <td className="py-3 pr-3">{entry.entry_type}</td>
                    <td className="py-3 pr-3">INR {formatNumber(entry.amount_inr)}</td>
                    <td className="py-3 pr-3">INR {formatNumber(entry.balance_after_inr)}</td>
                    <td className="py-3 pr-3 text-muted-foreground">{entry.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!ledger.data?.items.length && <p className="py-8 text-center text-sm text-muted-foreground">No billing entries yet.</p>}
          </div>
        </div>
      </section>
    </div>
  );
}
