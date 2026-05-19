import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  Bot,
  Braces,
  CheckCircle2,
  Cpu,
  Database,
  Dna,
  Gauge,
  GitBranch,
  KeyRound,
  Layers3,
  LineChart,
  LockKeyhole,
  Network,
  Play,
  Server,
  ShieldCheck,
  TerminalSquare,
  Webhook,
  type LucideIcon,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

type Feature = {
  title: string;
  body: string;
  icon: LucideIcon;
  tone: 'blue' | 'green' | 'orange' | 'cyan';
  metric?: string;
};

type ApiSection = {
  id: string;
  method: 'get' | 'post';
  label: string;
  title: string;
  description: string;
  code: string;
  baseUrl?: string;
};

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;800&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

  .evomind-nexus {
    --black: #020408;
    --deep: #050b12;
    --panel: #07111c;
    --card: #091827;
    --card-strong: #0d2232;
    --border: #12304a;
    --blue: #0a84ff;
    --blue-soft: rgba(10, 132, 255, 0.22);
    --green: #00f5a0;
    --green-soft: rgba(0, 245, 160, 0.2);
    --orange: #ff7a1a;
    --orange-soft: rgba(255, 122, 26, 0.2);
    --cyan: #00d4ff;
    --cyan-soft: rgba(0, 212, 255, 0.18);
    --text: #c8dff0;
    --text-dim: #7391aa;
    --text-soft: #8fb1c8;
    --text-bright: #eef8ff;
    min-height: 100vh;
    overflow-x: hidden;
    background:
      radial-gradient(circle at 16% 8%, rgba(10, 132, 255, 0.16), transparent 28rem),
      radial-gradient(circle at 88% 12%, rgba(0, 245, 160, 0.1), transparent 24rem),
      linear-gradient(180deg, var(--black), #03070d 42%, var(--black));
    color: var(--text);
    font-family: 'Rajdhani', Inter, ui-sans-serif, system-ui, sans-serif;
  }

  .evomind-nexus * {
    box-sizing: border-box;
  }

  .evomind-nexus a {
    color: inherit;
  }

  .nexus-grid-bg {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(10, 132, 255, 0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(10, 132, 255, 0.045) 1px, transparent 1px);
    background-size: 58px 58px;
    mask-image: linear-gradient(180deg, black, rgba(0, 0, 0, 0.62), transparent 92%);
  }

  .nexus-scanline {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.02) 0,
      rgba(255, 255, 255, 0.02) 1px,
      transparent 1px,
      transparent 5px
    );
    opacity: 0.25;
  }

  .nexus-shell {
    position: relative;
    z-index: 1;
  }

  .nexus-nav {
    position: sticky;
    top: 0;
    z-index: 40;
    border-bottom: 1px solid var(--border);
    background: rgba(2, 4, 8, 0.86);
    backdrop-filter: blur(18px);
  }

  .nexus-nav-inner {
    display: flex;
    align-items: center;
    gap: 24px;
    width: min(1180px, calc(100% - 32px));
    height: 64px;
    margin: 0 auto;
  }

  .nexus-logo {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-family: 'Orbitron', monospace;
    font-size: 18px;
    font-weight: 800;
    text-decoration: none;
    color: var(--text-bright);
  }

  .nexus-logo-dot {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: var(--green);
    box-shadow: 0 0 18px var(--green);
    animation: pulse-dot 2.4s ease-in-out infinite;
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.72); }
  }

  .nexus-nav-links {
    display: flex;
    align-items: center;
    gap: 24px;
    margin-left: auto;
  }

  .nexus-nav-links a,
  .nexus-link {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
    text-decoration: none;
    color: var(--text-dim);
    transition: color 160ms ease;
  }

  .nexus-nav-links a:hover,
  .nexus-link:hover {
    color: var(--green);
  }

  .nexus-console-btn,
  .nexus-primary-btn,
  .nexus-secondary-btn,
  .nexus-action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    min-height: 42px;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 0 18px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    text-decoration: none;
    text-transform: uppercase;
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, color 180ms ease;
  }

  .nexus-console-btn {
    border-color: var(--blue);
    color: var(--blue);
    background: rgba(10, 132, 255, 0.08);
  }

  .nexus-primary-btn {
    border-color: var(--green);
    background: var(--green);
    color: #00120b;
    box-shadow: 0 0 34px rgba(0, 245, 160, 0.2);
  }

  .nexus-secondary-btn {
    border-color: var(--orange);
    color: var(--orange);
    background: rgba(255, 122, 26, 0.06);
  }

  .nexus-action-btn {
    width: 100%;
    border-color: var(--blue);
    color: var(--blue);
    background: transparent;
  }

  .nexus-primary-btn:hover,
  .nexus-secondary-btn:hover,
  .nexus-console-btn:hover,
  .nexus-action-btn:hover {
    transform: translateY(-2px);
  }

  .nexus-primary-btn:hover {
    box-shadow: 0 0 42px rgba(0, 245, 160, 0.32);
  }

  .nexus-secondary-btn:hover {
    background: var(--orange);
    color: #090400;
    box-shadow: 0 0 34px rgba(255, 122, 26, 0.2);
  }

  .nexus-hero {
    min-height: calc(100vh - 64px);
    display: grid;
    align-items: center;
    padding: 74px 0 44px;
  }

  .nexus-container {
    width: min(1180px, calc(100% - 32px));
    margin: 0 auto;
  }

  .nexus-hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 0.92fr) minmax(360px, 0.78fr);
    align-items: center;
    gap: 48px;
  }

  .nexus-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    color: var(--green);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
  }

  .nexus-line {
    width: 40px;
    height: 1px;
    background: currentColor;
    opacity: 0.55;
  }

  .nexus-title {
    margin: 22px 0 0;
    font-family: 'Orbitron', monospace;
    font-size: 72px;
    line-height: 0.96;
    font-weight: 800;
    color: var(--text-bright);
  }

  .nexus-title .blue { color: var(--blue); text-shadow: 0 0 36px var(--blue-soft); }
  .nexus-title .green { color: var(--green); text-shadow: 0 0 36px var(--green-soft); }
  .nexus-title .orange { color: var(--orange); text-shadow: 0 0 36px var(--orange-soft); }
  .nexus-title .block { display: block; }

  .nexus-subtitle {
    margin: 18px 0 0;
    font-family: 'Orbitron', monospace;
    font-size: 18px;
    font-weight: 500;
    color: var(--text-soft);
  }

  .nexus-hero-copy {
    max-width: 660px;
    margin: 22px 0 0;
    color: var(--text);
    font-size: 19px;
    line-height: 1.7;
  }

  .nexus-hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 34px;
  }

  .nexus-orbit-panel {
    position: relative;
    min-height: 520px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background:
      linear-gradient(180deg, rgba(7, 17, 28, 0.9), rgba(2, 4, 8, 0.92)),
      radial-gradient(circle at 50% 50%, rgba(0, 245, 160, 0.1), transparent 20rem);
    overflow: hidden;
    box-shadow: inset 0 0 80px rgba(10, 132, 255, 0.06), 0 0 60px rgba(0, 0, 0, 0.24);
  }

  .nexus-orbit-panel::before,
  .nexus-orbit-panel::after {
    content: '';
    position: absolute;
    width: 74px;
    height: 74px;
    pointer-events: none;
  }

  .nexus-orbit-panel::before {
    top: -1px;
    left: -1px;
    border-top: 2px solid var(--green);
    border-left: 2px solid var(--green);
  }

  .nexus-orbit-panel::after {
    right: -1px;
    bottom: -1px;
    border-right: 2px solid var(--orange);
    border-bottom: 2px solid var(--orange);
  }

  .nexus-orbit {
    position: absolute;
    inset: 70px;
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 50%;
    animation: slow-spin 24s linear infinite;
  }

  .nexus-orbit.two {
    inset: 116px;
    border-color: rgba(0, 245, 160, 0.18);
    animation-duration: 18s;
    animation-direction: reverse;
  }

  .nexus-orbit.three {
    inset: 168px;
    border-color: rgba(255, 122, 26, 0.2);
    animation-duration: 30s;
  }

  @keyframes slow-spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .nexus-node {
    position: absolute;
    width: 13px;
    height: 13px;
    border-radius: 999px;
    background: var(--green);
    box-shadow: 0 0 24px var(--green);
  }

  .nexus-node.blue { background: var(--blue); box-shadow: 0 0 24px var(--blue); }
  .nexus-node.orange { background: var(--orange); box-shadow: 0 0 24px var(--orange); }

  .nexus-core {
    position: absolute;
    inset: 50%;
    display: grid;
    width: 170px;
    height: 170px;
    place-items: center;
    transform: translate(-50%, -50%);
    border: 1px solid rgba(0, 245, 160, 0.3);
    border-radius: 8px;
    background: rgba(2, 4, 8, 0.72);
    box-shadow: 0 0 50px rgba(0, 245, 160, 0.12);
  }

  .nexus-core-icon {
    display: grid;
    width: 76px;
    height: 76px;
    place-items: center;
    border: 1px solid rgba(0, 212, 255, 0.26);
    border-radius: 8px;
    color: var(--cyan);
    background: rgba(0, 212, 255, 0.08);
  }

  .nexus-core-label {
    margin-top: 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: var(--text-bright);
  }

  .nexus-terminal {
    position: absolute;
    left: 24px;
    right: 24px;
    bottom: 24px;
    border: 1px solid rgba(18, 48, 74, 0.9);
    border-radius: 8px;
    background: rgba(2, 4, 8, 0.88);
    padding: 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
  }

  .terminal-green { color: var(--green); }
  .terminal-blue { color: var(--blue); }
  .terminal-orange { color: var(--orange); }
  .terminal-dim { color: var(--text-dim); }

  .nexus-section {
    padding: 92px 0;
  }

  .nexus-band {
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    background: rgba(5, 11, 18, 0.88);
  }

  .nexus-section-header {
    max-width: 720px;
    margin: 0 auto 52px;
    text-align: center;
  }

  .nexus-section-title {
    margin: 16px 0 0;
    font-family: 'Orbitron', monospace;
    font-size: 40px;
    line-height: 1.14;
    font-weight: 700;
    color: var(--text-bright);
  }

  .nexus-section-copy {
    margin: 16px auto 0;
    color: var(--text-soft);
    font-size: 17px;
    line-height: 1.7;
  }

  .nexus-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border: 1px solid var(--border);
    background: var(--panel);
  }

  .nexus-stat {
    padding: 26px 22px;
    text-align: center;
    border-right: 1px solid var(--border);
  }

  .nexus-stat:last-child {
    border-right: 0;
  }

  .nexus-stat-value {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 800;
    color: var(--text-bright);
  }

  .nexus-stat-value.blue { color: var(--blue); }
  .nexus-stat-value.green { color: var(--green); }
  .nexus-stat-value.orange { color: var(--orange); }
  .nexus-stat-value.cyan { color: var(--cyan); }

  .nexus-stat-label {
    margin-top: 7px;
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
  }

  .nexus-about-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(360px, 0.88fr);
    gap: 44px;
    align-items: center;
  }

  .nexus-copy-stack {
    display: grid;
    gap: 20px;
    color: var(--text-soft);
    font-size: 16px;
    line-height: 1.78;
  }

  .nexus-copy-stack strong {
    color: var(--text-bright);
  }

  .nexus-matrix {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px;
    background: var(--panel);
  }

  .nexus-matrix-title {
    margin-bottom: 18px;
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
  }

  .nexus-meter {
    display: grid;
    grid-template-columns: 106px 1fr 46px;
    align-items: center;
    gap: 12px;
    margin-top: 13px;
  }

  .nexus-meter-label,
  .nexus-meter-value {
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
  }

  .nexus-meter-track {
    height: 22px;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--card);
  }

  .nexus-meter-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, transparent, var(--blue));
  }

  .nexus-meter-fill.green { background: linear-gradient(90deg, transparent, var(--green)); }
  .nexus-meter-fill.orange { background: linear-gradient(90deg, transparent, var(--orange)); }
  .nexus-meter-fill.cyan { background: linear-gradient(90deg, transparent, var(--cyan)); }

  .nexus-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
  }

  .nexus-card-grid.four {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .nexus-card {
    min-height: 100%;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: rgba(7, 17, 28, 0.92);
    padding: 26px;
    transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
  }

  .nexus-card:hover {
    transform: translateY(-4px);
    border-color: rgba(143, 177, 200, 0.4);
    background: rgba(9, 24, 39, 0.96);
  }

  .nexus-icon {
    display: grid;
    width: 48px;
    height: 48px;
    place-items: center;
    border: 1px solid currentColor;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.035);
  }

  .nexus-icon.blue { color: var(--blue); }
  .nexus-icon.green { color: var(--green); }
  .nexus-icon.orange { color: var(--orange); }
  .nexus-icon.cyan { color: var(--cyan); }

  .nexus-card-title {
    margin: 20px 0 0;
    color: var(--text-bright);
    font-family: 'Orbitron', monospace;
    font-size: 15px;
    font-weight: 700;
  }

  .nexus-card-copy {
    margin-top: 12px;
    color: var(--text-soft);
    font-size: 15px;
    line-height: 1.65;
  }

  .nexus-card-metric {
    display: inline-flex;
    margin-top: 18px;
    border: 1px solid currentColor;
    border-radius: 4px;
    padding: 5px 10px;
    color: var(--green);
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
  }

  .nexus-adv-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 2px;
    border: 1px solid var(--border);
    background: var(--border);
  }

  .nexus-adv-item {
    display: flex;
    gap: 20px;
    min-height: 178px;
    padding: 30px;
    background: var(--panel);
  }

  .nexus-adv-number {
    color: rgba(238, 248, 255, 0.16);
    font-family: 'Orbitron', monospace;
    font-size: 38px;
    font-weight: 800;
    line-height: 1;
  }

  .nexus-api-layout {
    display: grid;
    grid-template-columns: 250px minmax(0, 1fr);
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
  }

  .nexus-api-sidebar {
    border-right: 1px solid var(--border);
    background: var(--card);
    padding: 20px 0;
  }

  .nexus-api-label {
    padding: 0 20px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
  }

  .nexus-api-tab {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    border: 0;
    border-left: 2px solid transparent;
    background: transparent;
    padding: 12px 18px;
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    text-align: left;
    cursor: pointer;
    transition: color 160ms ease, background 160ms ease, border-color 160ms ease;
  }

  .nexus-api-tab:hover,
  .nexus-api-tab.active {
    color: var(--green);
    border-left-color: var(--green);
    background: rgba(0, 245, 160, 0.06);
  }

  .nexus-method {
    display: inline-flex;
    min-width: 42px;
    justify-content: center;
    border-radius: 4px;
    padding: 3px 7px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
  }

  .nexus-method.get {
    color: var(--green);
    background: rgba(0, 245, 160, 0.14);
  }

  .nexus-method.post {
    color: var(--blue);
    background: rgba(10, 132, 255, 0.14);
  }

  .nexus-api-content {
    padding: 30px;
  }

  .nexus-endpoint-title {
    color: var(--text-bright);
    font-family: 'Orbitron', monospace;
    font-size: 20px;
    font-weight: 700;
  }

  .nexus-endpoint-copy {
    max-width: 760px;
    margin-top: 10px;
    color: var(--text-soft);
    font-size: 15px;
    line-height: 1.7;
  }

  .nexus-code {
    position: relative;
    margin-top: 22px;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--black);
    padding: 22px;
    color: #d8fbff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
  }

  .nexus-code::before {
    content: 'HTTP';
    position: absolute;
    top: 10px;
    right: 14px;
    color: var(--text-dim);
    font-size: 10px;
  }

  .nexus-base-url {
    margin-top: 18px;
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
  }

  .nexus-base-url span {
    color: var(--green);
  }

  .nexus-guide-grid {
    display: grid;
    grid-template-columns: minmax(0, 0.9fr) minmax(360px, 1.1fr);
    gap: 42px;
  }

  .nexus-step-list {
    display: grid;
    gap: 12px;
  }

  .nexus-step {
    display: grid;
    grid-template-columns: 42px 1fr;
    gap: 14px;
    align-items: start;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
    padding: 16px;
  }

  .nexus-step-number {
    display: grid;
    width: 36px;
    height: 36px;
    place-items: center;
    border-radius: 6px;
    background: var(--orange);
    color: #110600;
    font-family: 'Orbitron', monospace;
    font-size: 13px;
    font-weight: 800;
  }

  .nexus-pricing-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
  }

  .nexus-tier {
    position: relative;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--panel);
    padding: 30px;
  }

  .nexus-tier.featured {
    border-color: var(--green);
    box-shadow: inset 0 0 60px rgba(0, 245, 160, 0.04), 0 0 38px rgba(0, 245, 160, 0.12);
  }

  .nexus-tier-badge {
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
  }

  .nexus-tier-name {
    margin-top: 16px;
    color: var(--text-bright);
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 800;
  }

  .nexus-tier-copy {
    min-height: 72px;
    margin-top: 12px;
    color: var(--text-soft);
    line-height: 1.65;
  }

  .nexus-tier ul {
    display: grid;
    gap: 10px;
    margin: 22px 0 28px;
    padding: 0;
    list-style: none;
  }

  .nexus-tier li {
    display: flex;
    gap: 9px;
    color: var(--text);
    font-size: 14px;
  }

  .nexus-footer {
    border-top: 1px solid var(--border);
    background: #010204;
    padding: 34px 0;
  }

  .nexus-footer-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    width: min(1180px, calc(100% - 32px));
    margin: 0 auto;
    color: var(--text-dim);
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
  }

  @media (max-width: 1040px) {
    .nexus-hero-grid,
    .nexus-about-grid,
    .nexus-guide-grid {
      grid-template-columns: 1fr;
    }

    .nexus-title {
      font-size: 58px;
    }

    .nexus-orbit-panel {
      min-height: 460px;
    }

    .nexus-card-grid,
    .nexus-card-grid.four,
    .nexus-pricing-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 760px) {
    .nexus-nav-inner {
      height: auto;
      min-height: 64px;
      flex-wrap: wrap;
      padding: 12px 0;
    }

    .nexus-nav-links {
      display: none;
    }

    .nexus-console-btn {
      margin-left: auto;
      padding: 0 13px;
    }

    .nexus-hero {
      padding-top: 52px;
    }

    .nexus-title {
      font-size: 42px;
    }

    .nexus-subtitle {
      font-size: 15px;
    }

    .nexus-hero-copy {
      font-size: 17px;
    }

    .nexus-orbit-panel {
      min-height: 390px;
    }

    .nexus-orbit {
      inset: 58px;
    }

    .nexus-orbit.two {
      inset: 96px;
    }

    .nexus-orbit.three {
      inset: 132px;
    }

    .nexus-core {
      width: 138px;
      height: 138px;
    }

    .nexus-terminal {
      left: 14px;
      right: 14px;
      bottom: 14px;
      font-size: 11px;
    }

    .nexus-section {
      padding: 70px 0;
    }

    .nexus-section-title {
      font-size: 30px;
    }

    .nexus-stats,
    .nexus-card-grid,
    .nexus-card-grid.four,
    .nexus-adv-grid,
    .nexus-pricing-grid {
      grid-template-columns: 1fr;
    }

    .nexus-stat {
      border-right: 0;
      border-bottom: 1px solid var(--border);
    }

    .nexus-stat:last-child {
      border-bottom: 0;
    }

    .nexus-adv-item {
      min-height: auto;
      padding: 24px;
    }

    .nexus-api-layout {
      grid-template-columns: 1fr;
    }

    .nexus-api-sidebar {
      border-right: 0;
      border-bottom: 1px solid var(--border);
    }

    .nexus-meter {
      grid-template-columns: 88px 1fr;
    }

    .nexus-meter-value {
      grid-column: 2;
    }

    .nexus-footer-inner {
      align-items: flex-start;
      flex-direction: column;
    }
  }
`;

const performance: Feature[] = [
  {
    title: 'Evolutionary training',
    body: 'Run prey and predator populations through curriculum stages, score fitness, preserve champions, and continue from checkpoints.',
    icon: Dna,
    tone: 'green',
    metric: 'Population loop',
  },
  {
    title: 'Agent inference',
    body: 'Query the best evolved genome with an observation vector and receive a bounded action vector for a simulator or application.',
    icon: Bot,
    tone: 'blue',
    metric: 'Action API',
  },
  {
    title: 'Runtime telemetry',
    body: 'Monitor generation progress, learning stability, novelty, species diversity, and job status from live API data.',
    icon: LineChart,
    tone: 'cyan',
    metric: 'Live metrics',
  },
  {
    title: 'Tenant control',
    body: 'Create isolated jobs, manage scoped keys, inspect usage, enforce limits, and keep experiments separated by tenant.',
    icon: Server,
    tone: 'orange',
    metric: 'Scoped access',
  },
  {
    title: 'Recoverable runs',
    body: 'Checkpoint and resume commands keep long-running training sessions auditable instead of tied to one terminal session.',
    icon: GitBranch,
    tone: 'green',
    metric: 'Checkpoints',
  },
  {
    title: 'Operational surface',
    body: 'The console exposes dashboards, genome catalogs, billing controls, API keys, and worker readiness in one interface.',
    icon: Gauge,
    tone: 'blue',
    metric: 'Console ready',
  },
];

const advantages = [
  {
    number: '01',
    title: 'Experiment isolation',
    body: 'Every job gets separate state, metrics, checkpoints, and genome catalogs, so training runs do not overwrite each other.',
  },
  {
    number: '02',
    title: 'Production guardrails',
    body: 'API keys, request IDs, readiness checks, body limits, webhook signing, and usage logging protect the operational API.',
  },
  {
    number: '03',
    title: 'Observable evolution',
    body: 'Operators can watch fitness, diversity, behavior, job commands, runtime status, and neural health as the system evolves.',
  },
  {
    number: '04',
    title: 'Direct deployment path',
    body: 'The best genomes can be served behind authenticated endpoints instead of remaining offline artifacts in a training folder.',
  },
  {
    number: '05',
    title: 'Extensible backend',
    body: 'FastAPI, worker modules, storage adapters, and typed schemas keep the training runtime practical to extend.',
  },
  {
    number: '06',
    title: 'Operator-friendly UI',
    body: 'The website routes users into a working console for training controls, metrics, genome management, keys, and billing.',
  },
];

const useCases: Feature[] = [
  {
    title: 'Adaptive simulation agents',
    body: 'Train policies that react to changing environment dynamics and export action behavior through the runtime API.',
    icon: Activity,
    tone: 'green',
  },
  {
    title: 'Research pipelines',
    body: 'Run repeatable population experiments, compare generations, and inspect outcomes without rebuilding control scripts.',
    icon: Database,
    tone: 'cyan',
  },
  {
    title: 'Internal AI platforms',
    body: 'Give teams authenticated access to training jobs, metrics, billing summaries, and webhooks from one frontend.',
    icon: Layers3,
    tone: 'blue',
  },
  {
    title: 'Automation workflows',
    body: 'Trigger external systems from lifecycle events and use API keys to connect EvoMind to schedulers and dashboards.',
    icon: Webhook,
    tone: 'orange',
  },
];

const apiSections: ApiSection[] = [
  {
    id: 'auth',
    method: 'get',
    label: 'Auth keys',
    title: 'List API keys',
    description: 'Review tenant-scoped API keys. Protected routes accept X-API-Key or an Authorization bearer token.',
    code: `curl http://localhost:8000/auth/keys \\
  -H "X-API-Key: evomind_your_key"`,
    baseUrl: 'http://localhost:8000',
  },
  {
    id: 'jobs',
    method: 'post',
    label: 'Create job',
    title: 'Create an isolated training job',
    description: 'Create a tenant-owned job before queueing training, checkpoint, resume, or inference operations.',
    code: `curl -X POST http://localhost:8000/jobs \\
  -H "X-API-Key: evomind_your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"job_id":"default","name":"Default run"}'`,
    baseUrl: 'http://localhost:8000',
  },
  {
    id: 'train',
    method: 'post',
    label: 'Start train',
    title: 'Queue training start',
    description: 'Send a lifecycle command to the worker runtime and then watch status and metrics refresh from the console.',
    code: `curl -X POST http://localhost:8000/jobs/default/train/start \\
  -H "X-API-Key: evomind_your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"generations":200,"resume":false}'`,
    baseUrl: 'http://localhost:8000',
  },
  {
    id: 'metrics',
    method: 'get',
    label: 'Metrics',
    title: 'Fetch training metrics',
    description: 'Read generation metrics for charts, diagnostics, or external monitoring.',
    code: `curl "http://localhost:8000/jobs/default/train/metrics?limit=100" \\
  -H "X-API-Key: evomind_your_key"`,
    baseUrl: 'http://localhost:8000',
  },
  {
    id: 'agent',
    method: 'post',
    label: 'Agent action',
    title: 'Run the best genome',
    description: 'Submit an observation vector and get an action vector from the selected evolved agent.',
    code: `curl -X POST http://localhost:8000/jobs/default/agent/action \\
  -H "X-API-Key: evomind_your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"observation":[0.2,0.8,0.1],"genome_type":"prey","max_action_length":4}'`,
    baseUrl: 'http://localhost:8000',
  },
  {
    id: 'webhooks',
    method: 'post',
    label: 'Webhooks',
    title: 'Register lifecycle webhooks',
    description: 'Send job events to external automation, dashboards, or deployment systems.',
    code: `curl -X POST http://localhost:8000/webhooks \\
  -H "X-API-Key: evomind_your_key" \\
  -H "Content-Type: application/json" \\
  -d '{"url":"https://example.com/evomind","events":["job.started","job.completed"]}'`,
    baseUrl: 'http://localhost:8000',
  },
];

const guideSteps = [
  'Start the FastAPI backend and keep it reachable on port 8000.',
  'Create an admin API key, then paste it into the console header.',
  'Create or select a job, queue training, and watch metrics update.',
  'Inspect genomes, checkpoint runs, and call the agent action endpoint from your client.',
];

const tiers = [
  {
    badge: 'Local',
    name: 'Developer',
    copy: 'Run EvoMind on a workstation for experiments, demos, and API integration testing.',
    features: ['React website and console', 'FastAPI backend', 'Local storage', 'Manual worker process'],
  },
  {
    badge: 'Recommended',
    name: 'Team Lab',
    copy: 'Operate shared tenant-scoped jobs with API keys, webhooks, metrics, backups, and usage review.',
    features: ['Multi-job control plane', 'Checkpoint and resume', 'Billing dashboard', 'Webhook delivery history'],
    featured: true,
  },
  {
    badge: 'Scale',
    name: 'Production',
    copy: 'Deploy the service behind the provided container and reverse-proxy configuration.',
    features: ['Docker deployment', 'Readiness endpoints', 'Request limits', 'Backup and restore drills'],
  },
];

const meters = [
  { label: 'Training', value: 92, tone: 'green' },
  { label: 'Inference', value: 86, tone: 'blue' },
  { label: 'Telemetry', value: 94, tone: 'cyan' },
  { label: 'Security', value: 90, tone: 'orange' },
  { label: 'Recovery', value: 82, tone: 'green' },
] as const;

function SectionHeader({
  eyebrow,
  title,
  body,
}: {
  eyebrow: string;
  title: React.ReactNode;
  body?: string;
}) {
  return (
    <div className="nexus-section-header">
      <div className="nexus-eyebrow">
        <span className="nexus-line" />
        {eyebrow}
        <span className="nexus-line" />
      </div>
      <h2 className="nexus-section-title">{title}</h2>
      {body ? <p className="nexus-section-copy">{body}</p> : null}
    </div>
  );
}

function FeatureCard({ feature }: { feature: Feature }) {
  const Icon = feature.icon;

  return (
    <article className="nexus-card">
      <div className={`nexus-icon ${feature.tone}`}>
        <Icon size={23} aria-hidden="true" />
      </div>
      <h3 className="nexus-card-title">{feature.title}</h3>
      <p className="nexus-card-copy">{feature.body}</p>
      {feature.metric ? <span className="nexus-card-metric">{feature.metric}</span> : null}
    </article>
  );
}

export function Website() {
  const [activeApi, setActiveApi] = useState(apiSections[0].id);
  const currentApi = useMemo(
    () => apiSections.find((section) => section.id === activeApi) ?? apiSections[0],
    [activeApi],
  );

  return (
    <div className="evomind-nexus">
      <style>{styles}</style>
      <div className="nexus-grid-bg" aria-hidden="true" />
      <div className="nexus-scanline" aria-hidden="true" />

      <div className="nexus-shell">
        <header className="nexus-nav">
          <div className="nexus-nav-inner">
            <a className="nexus-logo" href="#top">
              <span className="nexus-logo-dot" />
              EVO<span style={{ color: 'var(--blue)' }}>MIND</span>
            </a>
            <nav className="nexus-nav-links" aria-label="Website navigation">
              <a href="#model">Model</a>
              <a href="#performance">Performance</a>
              <a href="#advantages">Advantages</a>
              <a href="#api-docs">API Docs</a>
              <a href="#deploy">Deploy</a>
            </nav>
            <Link className="nexus-console-btn" to="/console">
              <TerminalSquare size={16} aria-hidden="true" />
              Console
            </Link>
          </div>
        </header>

        <main id="top">
          <section className="nexus-hero">
            <div className="nexus-container nexus-hero-grid">
              <div>
                <div className="nexus-eyebrow">
                  <span className="nexus-line" />
                  Evolutionary AI Infrastructure
                </div>
                <h1 className="nexus-title">
                  <span className="blue">EVO</span>
                  <span className="green">MIND</span>
                  <span className="block orange">CONTROL</span>
                </h1>
                <p className="nexus-subtitle">TRAINING RUNTIME / AGENT API / OPERATIONS CONSOLE</p>
                <p className="nexus-hero-copy">
                  EvoMind evolves adaptive agents, tracks every generation, preserves the strongest genomes,
                  and exposes a secured API plus frontend console for operating experiments from launch to deployment.
                </p>
                <div className="nexus-hero-actions">
                  <Link className="nexus-primary-btn" to="/console">
                    <Play size={16} aria-hidden="true" />
                    Open Console
                  </Link>
                  <a className="nexus-secondary-btn" href="#api-docs">
                    <BookOpen size={16} aria-hidden="true" />
                    Explore API
                  </a>
                </div>
              </div>

              <div className="nexus-orbit-panel" aria-label="EvoMind system visual">
                <div className="nexus-orbit">
                  <span className="nexus-node" style={{ top: 24, left: '50%' }} />
                  <span className="nexus-node blue" style={{ right: 36, top: '58%' }} />
                </div>
                <div className="nexus-orbit two">
                  <span className="nexus-node orange" style={{ left: 30, top: '42%' }} />
                  <span className="nexus-node" style={{ right: 54, top: 36 }} />
                </div>
                <div className="nexus-orbit three">
                  <span className="nexus-node blue" style={{ left: '48%', bottom: -6 }} />
                </div>
                <div className="nexus-core">
                  <div>
                    <div className="nexus-core-icon">
                      <Dna size={38} aria-hidden="true" />
                    </div>
                    <div className="nexus-core-label">EVOLUTION CORE</div>
                  </div>
                </div>
                <div className="nexus-terminal">
                  <div>
                    <span className="terminal-green">$</span> evomind job start default
                  </div>
                  <div>
                    <span className="terminal-blue">generation</span> 128
                    <span className="terminal-dim"> / </span>
                    <span className="terminal-orange">fitness</span> 0.82
                  </div>
                  <div>
                    <span className="terminal-green">ready</span>
                    <span className="terminal-dim"> API, metrics, genomes, billing, webhooks</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="nexus-container" aria-label="Project statistics">
            <div className="nexus-stats">
              {[
                ['1.7.0', 'API version', 'blue'],
                ['2', 'Genome types', 'green'],
                ['Live', 'Metrics stream', 'orange'],
                ['Tenant', 'Access model', 'cyan'],
              ].map(([value, label, tone]) => (
                <div className="nexus-stat" key={label}>
                  <div className={`nexus-stat-value ${tone}`}>{value}</div>
                  <div className="nexus-stat-label">{label}</div>
                </div>
              ))}
            </div>
          </section>

          <section id="model" className="nexus-section">
            <div className="nexus-container">
              <SectionHeader
                eyebrow="01 / Project Model"
                title={
                  <>
                    What is <span style={{ color: 'var(--blue)' }}>EvoMind</span>?
                  </>
                }
                body="A complete platform for evolving, observing, and serving adaptive agents through authenticated APIs."
              />
              <div className="nexus-about-grid">
                <div className="nexus-copy-stack">
                  <p>
                    <strong>EvoMind</strong> combines evolutionary training, genome management, a FastAPI backend,
                    worker health checks, checkpoint storage, billing controls, usage limits, and a React console.
                  </p>
                  <p>
                    The system trains prey and predator populations, evaluates fitness, saves usable genomes, and
                    lets clients call the best evolved agent through an action endpoint with a simple observation vector.
                  </p>
                  <p>
                    The frontend is not just a brochure. It routes operators into dashboards for jobs, training
                    commands, metrics, genomes, API keys, billing, and operational readiness.
                  </p>
                </div>

                <div className="nexus-matrix">
                  <div className="nexus-matrix-title">Runtime capability matrix</div>
                  {meters.map((meter) => (
                    <div className="nexus-meter" key={meter.label}>
                      <div className="nexus-meter-label">{meter.label}</div>
                      <div className="nexus-meter-track">
                        <div
                          className={`nexus-meter-fill ${meter.tone}`}
                          style={{ width: `${meter.value}%` }}
                        />
                      </div>
                      <div className="nexus-meter-value">{meter.value}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section id="performance" className="nexus-section nexus-band">
            <div className="nexus-container">
              <SectionHeader
                eyebrow="02 / Performance Surface"
                title={
                  <>
                    What It <span style={{ color: 'var(--green)' }}>Performs</span>
                  </>
                }
                body="EvoMind turns environment feedback into evolved behavior, then keeps the training and serving loop observable."
              />
              <div className="nexus-card-grid">
                {performance.map((feature) => (
                  <FeatureCard feature={feature} key={feature.title} />
                ))}
              </div>
            </div>
          </section>

          <section id="advantages" className="nexus-section">
            <div className="nexus-container">
              <SectionHeader
                eyebrow="03 / Advantages"
                title={
                  <>
                    Why choose <span style={{ color: 'var(--orange)' }}>EvoMind</span>?
                  </>
                }
                body="The project wraps research-grade evolutionary loops in the pieces needed to operate them as a service."
              />
              <div className="nexus-adv-grid">
                {advantages.map((item) => (
                  <article className="nexus-adv-item" key={item.number}>
                    <div className="nexus-adv-number">{item.number}</div>
                    <div>
                      <h3 className="nexus-card-title" style={{ marginTop: 0 }}>
                        {item.title}
                      </h3>
                      <p className="nexus-card-copy">{item.body}</p>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="nexus-section nexus-band">
            <div className="nexus-container">
              <SectionHeader
                eyebrow="04 / Use Cases"
                title={
                  <>
                    Built for <span style={{ color: 'var(--blue)' }}>adaptive systems</span>
                  </>
                }
              />
              <div className="nexus-card-grid four">
                {useCases.map((feature) => (
                  <FeatureCard feature={feature} key={feature.title} />
                ))}
              </div>
            </div>
          </section>

          <section id="api-docs" className="nexus-section">
            <div className="nexus-container">
              <SectionHeader
                eyebrow="05 / API Reference"
                title={
                  <>
                    Developer <span style={{ color: 'var(--green)' }}>API Docs</span>
                  </>
                }
                body="Core backend routes for authentication, jobs, training lifecycle, metrics, agent actions, and webhooks."
              />
              <div className="nexus-api-layout">
                <div className="nexus-api-sidebar">
                  <div className="nexus-api-label">Endpoints</div>
                  {apiSections.map((section) => (
                    <button
                      className={`nexus-api-tab ${currentApi.id === section.id ? 'active' : ''}`}
                      key={section.id}
                      onClick={() => setActiveApi(section.id)}
                      type="button"
                    >
                      <span className={`nexus-method ${section.method}`}>{section.method}</span>
                      {section.label}
                    </button>
                  ))}
                </div>
                <div className="nexus-api-content">
                  <div className="nexus-endpoint-title">{currentApi.title}</div>
                  <p className="nexus-endpoint-copy">{currentApi.description}</p>
                  <pre className="nexus-code">{currentApi.code}</pre>
                  <div className="nexus-base-url">
                    BASE URL: <span>{currentApi.baseUrl}</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section id="deploy" className="nexus-section nexus-band">
            <div className="nexus-container nexus-guide-grid">
              <div>
                <div className="nexus-eyebrow">
                  <span className="nexus-line" />
                  06 / User Guide
                </div>
                <h2 className="nexus-section-title">
                  Run the stack, authenticate, and operate jobs.
                </h2>
                <p className="nexus-section-copy" style={{ marginLeft: 0 }}>
                  The website is the public entry point. The console becomes live when the backend is running
                  and a valid API key is configured.
                </p>
                <pre className="nexus-code">
{`python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
python -m api.auth create --name local-admin --tenant default
cd frontend
npm install
npm run dev`}
                </pre>
              </div>
              <div className="nexus-step-list">
                {guideSteps.map((step, index) => (
                  <div className="nexus-step" key={step}>
                    <div className="nexus-step-number">{index + 1}</div>
                    <p className="nexus-card-copy" style={{ marginTop: 0 }}>
                      {step}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="nexus-section">
            <div className="nexus-container">
              <SectionHeader
                eyebrow="07 / Deployment Modes"
                title={
                  <>
                    From local lab to <span style={{ color: 'var(--orange)' }}>production runtime</span>
                  </>
                }
              />
              <div className="nexus-pricing-grid">
                {tiers.map((tier) => (
                  <article className={`nexus-tier ${tier.featured ? 'featured' : ''}`} key={tier.name}>
                    <div className="nexus-tier-badge">{tier.badge}</div>
                    <h3 className="nexus-tier-name">{tier.name}</h3>
                    <p className="nexus-tier-copy">{tier.copy}</p>
                    <ul>
                      {tier.features.map((feature) => (
                        <li key={feature}>
                          <CheckCircle2 size={16} color="var(--green)" aria-hidden="true" />
                          {feature}
                        </li>
                      ))}
                    </ul>
                    <Link
                      className={tier.featured ? 'nexus-primary-btn' : 'nexus-action-btn'}
                      to="/console"
                    >
                      Open Console
                      <ArrowRight size={16} aria-hidden="true" />
                    </Link>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="nexus-section nexus-band">
            <div className="nexus-container nexus-about-grid">
              <div>
                <div className="nexus-eyebrow">
                  <span className="nexus-line" />
                  Interactive Console
                </div>
                <h2 className="nexus-section-title">The live API frontend is built in.</h2>
                <p className="nexus-section-copy" style={{ marginLeft: 0 }}>
                  Use the console to manage training, inspect metrics, compare genomes, create API keys,
                  review billing, and check service readiness against the real backend.
                </p>
                <div className="nexus-hero-actions">
                  <Link className="nexus-primary-btn" to="/console">
                    Launch Console
                    <ArrowRight size={16} aria-hidden="true" />
                  </Link>
                </div>
              </div>
              <div className="nexus-matrix">
                <div className="nexus-matrix-title">Console modules</div>
                <div className="nexus-card-grid" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
                  {[
                    [BarChart3, 'Metrics', 'Fitness and diversity'],
                    [Dna, 'Genomes', 'Prey and predator catalogs'],
                    [KeyRound, 'API keys', 'Scoped credentials'],
                    [Network, 'Readiness', 'Runtime dependencies'],
                    [Braces, 'Agent API', 'Observation to action'],
                    [LockKeyhole, 'Controls', 'Tenant-aware access'],
                  ].map(([Icon, title, body]) => {
                    const PreviewIcon = Icon as LucideIcon;
                    return (
                      <article className="nexus-card" key={String(title)} style={{ padding: 18 }}>
                        <div className="nexus-icon cyan" style={{ width: 40, height: 40 }}>
                          <PreviewIcon size={19} aria-hidden="true" />
                        </div>
                        <h3 className="nexus-card-title">{String(title)}</h3>
                        <p className="nexus-card-copy">{String(body)}</p>
                      </article>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        </main>

        <footer className="nexus-footer">
          <div className="nexus-footer-inner">
            <div className="nexus-logo" style={{ fontSize: 15 }}>
              EVO<span style={{ color: 'var(--blue)' }}>MIND</span>
            </div>
            <div>EvoMind evolutionary AI training, inference, and operations.</div>
            <div style={{ display: 'flex', gap: 18 }}>
              <a className="nexus-link" href="#api-docs">
                API Docs
              </a>
              <Link className="nexus-link" to="/console">
                Console
              </Link>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
