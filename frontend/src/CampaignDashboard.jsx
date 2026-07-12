import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Bar, ComposedChart } from 'recharts';
import { Play, Pause, SkipForward, Dna, FlaskConical, Sparkles } from 'lucide-react';

const T = {
  bg: '#11181A',
  panel: '#1A2326',
  panelAlt: '#161E20',
  border: '#2A3538',
  borderSoft: '#212B2E',
  amber: '#E8A33D',
  amberDim: '#8A6526',
  lavender: '#9B8AC4',
  lavenderDim: '#5C5476',
  coral: '#E0725C',
  parchment: '#EDE6D6',
  muted: '#7E9290',
  mutedDim: '#4E5E5C',
};

const API_BASE = 'http://localhost:8000';

const DEMO_STATUS = {
  campaign_id: 'camp_demo0001', tenant_id: 'demo', name: 'Q3 Signup Campaign',
  status: 'running', generation: 12, total_generations: 25, species_count: 6,
  best_fitness: 21.4, total_impressions_served: 68400,
  best_creative: {
    genome_id: 'ad_5c19a2', headline: 'Your competitors already found this',
    image_style: 'bold flat illustration, high contrast', cta: 'Claim your spot',
    tone: 'urgent and direct', color_scheme: 'navy and gold',
    optional_traits: { urgency_badge: true, social_proof_line: true, secondary_cta: false },
    fitness: 21.4,
  },
};

const DEMO_HISTORY = [
  { generation: 0, best_fitness: 8.2, species_count: 0, impressions_this_generation: 6000 },
  { generation: 1, best_fitness: 9.6, species_count: 4, impressions_this_generation: 6000 },
  { generation: 2, best_fitness: 9.1, species_count: 6, impressions_this_generation: 6000 },
  { generation: 3, best_fitness: 11.8, species_count: 5, impressions_this_generation: 6000 },
  { generation: 4, best_fitness: 12.4, species_count: 7, impressions_this_generation: 6000 },
  { generation: 5, best_fitness: 14.0, species_count: 6, impressions_this_generation: 6000 },
  { generation: 6, best_fitness: 13.5, species_count: 8, impressions_this_generation: 6000 },
  { generation: 7, best_fitness: 15.9, species_count: 7, impressions_this_generation: 6000 },
  { generation: 8, best_fitness: 17.2, species_count: 6, impressions_this_generation: 6000 },
  { generation: 9, best_fitness: 16.6, species_count: 8, impressions_this_generation: 6000 },
  { generation: 10, best_fitness: 18.8, species_count: 7, impressions_this_generation: 6000 },
  { generation: 11, best_fitness: 20.1, species_count: 6, impressions_this_generation: 6000 },
  { generation: 12, best_fitness: 21.4, species_count: 6, impressions_this_generation: 6000 },
];

const DEMO_CREATIVES = [
  { genome_id: 'ad_5c19a2', headline: 'Your competitors already found this', image_style: 'bold flat illustration, high contrast', cta: 'Claim your spot', tone: 'urgent and direct', color_scheme: 'navy and gold', optional_traits: { urgency_badge: true, social_proof_line: true, secondary_cta: false }, fitness: 21.4 },
  { genome_id: 'ad_881f30', headline: 'Finally, software that respects your time', image_style: 'lifestyle photo, warm natural light', cta: 'Get a demo', tone: 'warm and reassuring', color_scheme: 'soft pastel', optional_traits: { urgency_badge: false, social_proof_line: true, secondary_cta: true }, fitness: 19.8 },
  { genome_id: 'ad_204b7e', headline: 'Join thousands who already switched', image_style: 'screenshot-driven UI mockup', cta: 'Start free trial', tone: 'data-driven and precise', color_scheme: 'high-contrast black/white', optional_traits: { urgency_badge: true, social_proof_line: false, secondary_cta: false }, fitness: 18.3 },
  { genome_id: 'ad_63aa11', headline: 'Stop overpaying for the same results', image_style: 'before/after comparison layout', cta: 'See pricing', tone: 'urgent and direct', color_scheme: 'navy and gold', optional_traits: { urgency_badge: true, social_proof_line: false, secondary_cta: true }, fitness: 16.7 },
  { genome_id: 'ad_a0129d', headline: 'The smarter way to get this done', image_style: 'minimalist product shot on white', cta: 'Try it now', tone: 'playful and confident', color_scheme: 'brand primary plus neon accent', optional_traits: { urgency_badge: false, social_proof_line: false, secondary_cta: false }, fitness: 14.1 },
  { genome_id: 'ad_f7d420', headline: 'Built for people who hate wasting money', image_style: 'lifestyle photo, warm natural light', cta: 'Claim your spot', tone: 'data-driven and precise', color_scheme: 'soft pastel', optional_traits: { urgency_badge: true, social_proof_line: true, secondary_cta: true }, fitness: 12.9 },
];

const STATUS_STYLE = {
  running: { color: T.amber, label: 'RUNNING', glow: '0 0 12px rgba(232,163,61,0.35)' },
  paused: { color: T.muted, label: 'PAUSED', glow: 'none' },
  queued: { color: T.mutedDim, label: 'QUEUED', glow: 'none' },
  stopped: { color: T.lavender, label: 'COMPLETE', glow: '0 0 12px rgba(155,138,196,0.3)' },
  error: { color: T.coral, label: 'ERROR', glow: '0 0 12px rgba(224,114,92,0.35)' },
};

function StatCounter({ label, value, accent }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: T.mutedDim }}>{label}</span>
      <span className="font-mono text-xl" style={{ color: accent || T.parchment }}>{value}</span>
    </div>
  );
}

function SpecimenCard({ creative, rank }) {
  const badges = Object.entries(creative.optional_traits || {}).filter(([, v]) => v).map(([k]) => k.replace(/_/g, ' '));
  return (
    <div className="relative rounded-lg p-4 pt-5" style={{ background: T.panel, border: `1px solid ${T.border}` }}>
      {/* punch-hole notches */}
      <div className="absolute -top-1.5 left-4 w-3 h-3 rounded-full" style={{ background: T.bg, border: `1px solid ${T.border}` }} />
      <div className="absolute -top-1.5 right-4 w-3 h-3 rounded-full" style={{ background: T.bg, border: `1px solid ${T.border}` }} />

      <div className="flex items-start justify-between mb-3">
        <span className="font-mono text-[10px] px-1.5 py-0.5 rounded" style={{ color: T.mutedDim, border: `1px solid ${T.borderSoft}` }}>
          #{String(rank).padStart(2, '0')} · {creative.genome_id}
        </span>
        <span className="font-mono text-sm" style={{ color: T.amber }}>{creative.fitness.toFixed(1)}</span>
      </div>

      <p className="font-sans text-base leading-snug mb-3" style={{ color: T.parchment }}>
        &ldquo;{creative.headline}&rdquo;
      </p>

      <div className="space-y-1.5 mb-3">
        {[
          ['IMAGE', creative.image_style],
          ['CTA', creative.cta],
          ['TONE', creative.tone],
          ['PALETTE', creative.color_scheme],
        ].map(([label, val]) => (
          <div key={label} className="flex gap-2 text-xs">
            <span className="font-mono w-16 shrink-0" style={{ color: T.mutedDim }}>{label}</span>
            <span className="font-sans" style={{ color: T.muted }}>{val}</span>
          </div>
        ))}
      </div>

      {badges.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {badges.map((b) => (
            <span key={b} className="font-mono text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded-full"
                  style={{ color: T.lavender, border: `1px solid ${T.lavenderDim}` }}>
              + {b}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ControlButton({ icon: Icon, label, onClick, disabled, active }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-md font-mono text-xs uppercase tracking-wide transition-opacity disabled:opacity-40"
      style={{
        background: active ? T.amberDim : T.panelAlt,
        border: `1px solid ${active ? T.amber : T.border}`,
        color: active ? T.bg : T.parchment,
      }}
    >
      <Icon size={13} />
      {label}
    </button>
  );
}

export default function CampaignDashboard({ campaignId = 'camp_demo0001', tenantId = 'demo' }) {
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [creatives, setCreatives] = useState([]);
  const [isDemo, setIsDemo] = useState(false);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);

  const headers = { 'X-Tenant-Id': tenantId };

  const fetchAll = useCallback(async () => {
    try {
      const [sRes, hRes, cRes] = await Promise.all([
        fetch(`${API_BASE}/campaigns/${campaignId}`, { headers }),
        fetch(`${API_BASE}/campaigns/${campaignId}/history`, { headers }),
        fetch(`${API_BASE}/campaigns/${campaignId}/creatives?limit=6`, { headers }),
      ]);
      if (!sRes.ok || !hRes.ok || !cRes.ok) throw new Error('API unreachable');
      const s = await sRes.json();
      const h = await hRes.json();
      const c = await cRes.json();
      setStatus(s);
      setHistory(h.history);
      setCreatives(c.creatives);
      setIsDemo(false);
    } catch (e) {
      setStatus(DEMO_STATUS);
      setHistory(DEMO_HISTORY);
      setCreatives(DEMO_CREATIVES);
      setIsDemo(true);
    } finally {
      setLoading(false);
    }
  }, [campaignId]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const doAction = async (action) => {
    if (isDemo) return;
    setPending(true);
    try {
      await fetch(`${API_BASE}/campaigns/${campaignId}/${action}`, { method: 'POST', headers });
      await fetchAll();
    } finally {
      setPending(false);
    }
  };

  if (loading || !status) {
    return <div className="p-8 font-mono text-sm" style={{ background: T.bg, color: T.muted }}>Loading campaign…</div>;
  }

  const st = STATUS_STYLE[status.status] || STATUS_STYLE.queued;

  return (
    <div className="min-h-full w-full p-5" style={{ background: T.bg }}>
      {isDemo && (
        <div className="mb-4 px-3 py-2 rounded font-mono text-[11px]" style={{ background: T.panelAlt, border: `1px solid ${T.border}`, color: T.mutedDim }}>
          DEMO DATA — connect to a running campaign API at {API_BASE} to see live results.
        </div>
      )}

      {/* Instrument panel header */}
      <div className="rounded-xl p-5 mb-5" style={{ background: T.panel, border: `1px solid ${T.border}` }}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Dna size={18} style={{ color: T.amber }} />
            <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: T.mutedDim }}>
              Evolutionary Ad Creative Lab
            </span>
          </div>
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full font-mono text-[10px] uppercase tracking-widest"
               style={{ color: st.color, border: `1px solid ${st.color}`, boxShadow: st.glow }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: st.color }} />
            {st.label}
          </div>
        </div>

        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-sans text-2xl mb-1" style={{ color: T.parchment }}>{status.name}</h1>
            <span className="font-mono text-xs" style={{ color: T.mutedDim }}>{status.campaign_id}</span>
          </div>

          <div className="flex gap-6">
            <StatCounter label="Generation" value={`${status.generation} / ${status.total_generations}`} />
            <StatCounter label="Species" value={status.species_count} accent={T.lavender} />
            <StatCounter label="Best Fitness" value={status.best_fitness.toFixed(1)} accent={T.amber} />
            <StatCounter label="Impressions" value={status.total_impressions_served.toLocaleString()} />
          </div>

          <div className="flex gap-2">
            <ControlButton icon={Play} label="Start" onClick={() => doAction('start')} disabled={pending || status.status === 'running'} active={status.status === 'running'} />
            <ControlButton icon={Pause} label="Pause" onClick={() => doAction('pause')} disabled={pending || status.status !== 'running'} />
            <ControlButton icon={SkipForward} label="Step" onClick={() => doAction('step')} disabled={pending || status.status !== 'running'} />
          </div>
        </div>
      </div>

      {/* Growth curve */}
      <div className="rounded-xl p-5 mb-5" style={{ background: T.panel, border: `1px solid ${T.border}` }}>
        <div className="flex items-center gap-2 mb-4">
          <Sparkles size={14} style={{ color: T.amber }} />
          <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: T.mutedDim }}>
            Fitness &amp; Species — per generation
          </span>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={history} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid stroke={T.borderSoft} vertical={false} />
            <XAxis dataKey="generation" tick={{ fill: T.mutedDim, fontSize: 10, fontFamily: 'monospace' }} axisLine={{ stroke: T.border }} tickLine={false} />
            <YAxis yAxisId="fitness" tick={{ fill: T.mutedDim, fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
            <YAxis yAxisId="species" orientation="right" tick={{ fill: T.mutedDim, fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: T.panelAlt, border: `1px solid ${T.border}`, borderRadius: 6, fontFamily: 'monospace', fontSize: 11 }}
                     labelStyle={{ color: T.parchment }} />
            <Bar yAxisId="species" dataKey="species_count" fill={T.lavenderDim} radius={[2, 2, 0, 0]} barSize={12} name="species" />
            <Line yAxisId="fitness" type="monotone" dataKey="best_fitness" stroke={T.amber} strokeWidth={2} dot={false} name="best fitness" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Specimen board */}
      <div className="rounded-xl p-5" style={{ background: T.panel, border: `1px solid ${T.border}` }}>
        <div className="flex items-center gap-2 mb-4">
          <FlaskConical size={14} style={{ color: T.lavender }} />
          <span className="font-mono text-[10px] uppercase tracking-widest" style={{ color: T.mutedDim }}>
            Top Specimens — this generation
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {creatives.map((c, i) => <SpecimenCard key={c.genome_id} creative={c} rank={i + 1} />)}
        </div>
      </div>
    </div>
  );
}
