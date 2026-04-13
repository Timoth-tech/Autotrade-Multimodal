"use client";

import { useEffect, useState, useRef, useCallback } from 'react';

const API = "http://127.0.0.1:8000";

// ─── Types ───────────────────────────────────────────────────────
interface Factor { name: string; weight: number; value: string; positive: boolean }
interface HeatmapTile { symbol: string; pl_pct: number; market_value: number; qty: string; current_price: number }
interface DrawdownPoint { date: string; drawdown: number }
interface EquityPoint { date: string; equity: number; profit_loss: number }
interface Position { symbol: string; qty: string; current_price: number; unrealized_pl: number }

// ─── Confidence Gauge Component ──────────────────────────────────
function ConfidenceGauge({ score, sentiment }: { score: number; sentiment: string }) {
  const color = score >= 60 ? 'var(--green)' : score <= 40 ? 'var(--red)' : 'var(--yellow)';
  const circumference = 2 * Math.PI * 52;
  const offset = circumference - (score / 100) * circumference;
  return (
    <div className="gauge-ring">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
        <circle cx="60" cy="60" r="52" fill="none" stroke={color} strokeWidth="8"
          strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 1s ease' }} />
      </svg>
      <div className="gauge-value" style={{ color }}>
        {score}
        <span className="gauge-label" style={{ color: 'var(--text-secondary)' }}>{sentiment}</span>
      </div>
    </div>
  );
}

// ─── Factor Bar Component ────────────────────────────────────────
function FactorBar({ factor }: { factor: Factor }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
      <span style={{ width: '70px', fontSize: '0.72rem', color: 'var(--text-secondary)', flexShrink: 0 }}>{factor.name}</span>
      <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
        <div style={{
          width: `${Math.min(factor.weight, 100)}%`, height: '100%', borderRadius: '3px',
          background: factor.positive ? 'var(--green)' : 'var(--red)',
          transition: 'width 0.8s ease'
        }} />
      </div>
      <span style={{ fontSize: '0.7rem', color: factor.positive ? 'var(--green)' : 'var(--red)', width: '55px', textAlign: 'right', flexShrink: 0 }}>
        {factor.value}
      </span>
    </div>
  );
}

// ─── Mini Sparkline Component ────────────────────────────────────
function Sparkline({ data, width = 200, height = 50, color = 'var(--green)' }: { data: number[]; width?: number; height?: number; color?: string }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const step = width / (data.length - 1);
  const points = data.map((v, i) => `${i * step},${height - ((v - min) / range) * (height - 4)}`).join(' ');
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`sg-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${height} ${points} ${width},${height}`} fill={`url(#sg-${color})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

// ─── Heatmap Tile ────────────────────────────────────────────────
function HeatTile({ tile }: { tile: HeatmapTile }) {
  const intensity = Math.min(Math.abs(tile.pl_pct) / 10, 1);
  const bg = tile.pl_pct >= 0
    ? `rgba(34, 197, 94, ${0.1 + intensity * 0.5})`
    : `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;
  const border = tile.pl_pct >= 0 ? `rgba(34,197,94,${0.3 + intensity * 0.3})` : `rgba(239,68,68,${0.3 + intensity * 0.3})`;
  return (
    <div className="heatmap-tile" style={{ background: bg, border: `1px solid ${border}` }}>
      <div style={{ fontSize: '0.9rem', fontWeight: 700 }}>{tile.symbol}</div>
      <div style={{ fontSize: '1.1rem', fontWeight: 800, color: tile.pl_pct >= 0 ? 'var(--green)' : 'var(--red)' }}>
        {tile.pl_pct >= 0 ? '+' : ''}{tile.pl_pct}%
      </div>
      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>${tile.market_value.toLocaleString()}</div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────
export default function CommandCenter() {
  // State
  const [portfolio, setPortfolio] = useState<any>({ equity: 0, change_pct: 0, positions: [], status: "loading" });
  const [confidence, setConfidence] = useState<any>({ score: 50, sentiment: "Loading", factors: [] });
  const [heatmap, setHeatmap] = useState<HeatmapTile[]>([]);
  const [drawdown, setDrawdown] = useState<any>({ max_drawdown: 0, current_drawdown: 0, series: [] });
  const [newsSentiment, setNewsSentiment] = useState<any[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [activities, setActivities] = useState<any[]>([]);
  const [autopilot, setAutopilot] = useState<any>({ status: "idle" });
  const [strategyMode, setStrategyMode] = useState("balanced");
  const [autopilotEnabled, setAutopilotEnabled] = useState(true);
  const [logs, setLogs] = useState<{ time: string; node: string; msg: string; type: string }[]>([
    { time: new Date().toLocaleTimeString(), node: 'System', msg: 'Command Center Online', type: 'info' }
  ]);
  const [chatMessages, setChatMessages] = useState<{ sender: string; text: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [panicConfirm, setPanicConfirm] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  // ─── Data Fetching ──────────────────────────────────────────────
  const fetchAll = useCallback(async () => {
    try {
      const [pRes, cRes, hRes, dRes, eRes, aRes, sRes, nRes] = await Promise.allSettled([
        fetch(`${API}/api/portfolio`), fetch(`${API}/api/confidence`),
        fetch(`${API}/api/portfolio/heatmap`), fetch(`${API}/api/portfolio/drawdown`),
        fetch(`${API}/api/portfolio/history`), fetch(`${API}/api/activities`),
        fetch(`${API}/api/agent/status`), fetch(`${API}/api/news/sentiment`)
      ]);
      if (pRes.status === 'fulfilled') { const d = await pRes.value.json(); setPortfolio(d); }
      if (cRes.status === 'fulfilled') {
        const d = await cRes.value.json();
        setConfidence(d);
        setStrategyMode(d.strategy_mode || "balanced");
        setAutopilotEnabled(d.autopilot_enabled !== false);
      }
      if (hRes.status === 'fulfilled') { const d = await hRes.value.json(); if (d.tiles) setHeatmap(d.tiles); }
      if (dRes.status === 'fulfilled') { const d = await dRes.value.json(); setDrawdown(d); }
      if (eRes.status === 'fulfilled') { const d = await eRes.value.json(); if (d.history) setEquityHistory(d.history); }
      if (aRes.status === 'fulfilled') { const d = await aRes.value.json(); if (d.activities) setActivities(d.activities); }
      if (sRes.status === 'fulfilled') { const d = await sRes.value.json(); setAutopilot(d); }
      if (nRes?.status === 'fulfilled') { const d = await nRes.value.json(); if (d.news) setNewsSentiment(d.news); }
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  // ─── Actions ────────────────────────────────────────────────────
  const addLog = (node: string, msg: string, type: string = 'info') => {
    setLogs(prev => [{ time: new Date().toLocaleTimeString(), node, msg, type }, ...prev].slice(0, 50));
  };

  const changeStrategy = async (mode: string) => {
    await fetch(`${API}/api/strategy/mode?mode=${mode}`, { method: 'POST' });
    setStrategyMode(mode);
    addLog('System', `Stratégie changée → ${mode.toUpperCase()}`, 'info');
  };

  const handlePanic = async () => {
    if (!panicConfirm) { setPanicConfirm(true); return; }
    setPanicConfirm(false);
    try {
      await fetch(`${API}/api/panic`, { method: 'POST' });
      setAutopilotEnabled(false);
      addLog('PANIC', '🚨 ARRÊT D\'URGENCE — Toutes positions liquidées', 'error');
      fetchAll();
    } catch { addLog('System', 'Erreur panic switch', 'error'); }
  };

  const handleResume = async () => {
    await fetch(`${API}/api/panic/resume`, { method: 'POST' });
    setAutopilotEnabled(true);
    addLog('System', '✅ Autopilot réactivé', 'info');
  };

  const triggerAgent = async (symbol: string = "") => {
    setIsAgentRunning(true);
    addLog('System', symbol ? `Analyse directe de ${symbol}...` : 'Lancement Autopilot Multi-Ticker...', 'info');
    try {
      const res = await fetch(`${API}/api/agent/trigger?symbol=${symbol}`);
      const data = await res.json();
      if (data.status === "success") {
        data.messages?.forEach((m: string) => addLog('Agent', m, m.includes('TRADE') ? 'trade' : 'info'));
        setTimeout(fetchAll, 2000);
      } else {
        addLog('System', `Erreur: ${data.message}`, 'error');
      }
    } catch { addLog('System', 'Backend inaccessible', 'error'); }
    setIsAgentRunning(false);
  };

  const sendChat = async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput;
    setChatMessages(p => [...p, { sender: 'user', text: msg }]);
    setChatInput(""); setIsChatLoading(true);
    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, portfolio_context: `Equity: $${portfolio.equity}` })
      });
      const data = await res.json();
      setChatMessages(p => [...p, { sender: 'ia', text: data.status === 'success' ? data.reply : data.message }]);
    } catch { setChatMessages(p => [...p, { sender: 'ia', text: 'Connexion perdue.' }]); }
    setIsChatLoading(false);
  };

  // ─── Render ─────────────────────────────────────────────────────
  const statusColor = autopilot.status === 'completed' ? 'var(--green)' : autopilot.status === 'market_closed' ? 'var(--yellow)' : autopilot.status === 'paused' ? 'var(--red)' : 'var(--accent)';
  const statusText = autopilot.status === 'completed' ? `Cycle #${autopilot.cycle} → ${autopilot.decision} ${autopilot.symbol}` :
    autopilot.status === 'market_closed' ? `Fermé ${autopilot.next_open ? `(${autopilot.next_open})` : ''}` :
    autopilot.status === 'paused' ? 'PAUSE (Panic)' :
    autopilot.status === 'error' ? 'Erreur' : 'En attente';

  return (
    <div style={{ minHeight: '100vh', padding: '1.5rem 2rem', background: 'var(--bg-primary)', backgroundImage: 'radial-gradient(circle at 10% 20%, rgba(99,102,241,0.04), transparent 30%), radial-gradient(circle at 90% 80%, rgba(34,197,94,0.03), transparent 30%)' }}>

      {/* ═══ HEADER ═══ */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
        <div>
          <h1 style={{ fontSize: '1.6rem', fontWeight: 800, letterSpacing: '-0.03em', background: 'linear-gradient(135deg, #fff 0%, #6366f1 50%, #22c55e 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            AutoTrade Command Center
          </h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Autonomous Multi-Ticker AI Trading System</p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 0.9rem', borderRadius: '20px', border: `1px solid ${statusColor}`, background: `${statusColor}15`, fontSize: '0.75rem', fontWeight: 600, color: statusColor }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: statusColor, animation: 'pulse 2s infinite' }} />
            {statusText}
          </div>
          {!autopilotEnabled ? (
            <button className="btn btn-accent btn-sm" onClick={handleResume}>▶ Réactiver</button>
          ) : (
            <button className="btn btn-sm" onClick={() => triggerAgent("")} disabled={isAgentRunning}
              style={{ background: isAgentRunning ? 'var(--bg-tertiary)' : 'var(--accent)', borderColor: 'var(--accent)', color: '#fff' }}>
              {isAgentRunning ? '⏳ Analyse...' : '🌐 Lancer Cycle'}
            </button>
          )}
        </div>
      </header>

      {/* ═══ TOP ROW: Confidence + Strategy + Panic ═══ */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>

        {/* Confidence Score */}
        <div className="card" style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
          <ConfidenceGauge score={confidence.score || 50} sentiment={confidence.sentiment || 'N/A'} />
          <div style={{ flex: 1 }}>
            <div className="card-title">📊 Score de Confiance</div>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.6rem' }}>
              <div>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Win Rate</span>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--green)' }}>{confidence.win_rate || 0}%</div>
              </div>
              <div>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Daily P&L</span>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: (confidence.daily_change || 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {(confidence.daily_change || 0) >= 0 ? '+' : ''}{confidence.daily_change || 0}%
                </div>
              </div>
            </div>
            {(confidence.factors || []).map((f: Factor, i: number) => <FactorBar key={i} factor={f} />)}
          </div>
        </div>

        {/* Strategy Mode */}
        <div className="card">
          <div className="card-title">⚙️ Mode Stratégique</div>
          <div className="strategy-switch" style={{ marginBottom: '1rem' }}>
            {['conservative', 'balanced', 'aggressive'].map(m => (
              <button key={m} className={strategyMode === m ? 'active' : ''} onClick={() => changeStrategy(m)}>
                {m === 'conservative' ? '🛡️ Prudent' : m === 'balanced' ? '⚖️ Équilibré' : '🔥 Agressif'}
              </button>
            ))}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {strategyMode === 'conservative' ? 'Allocation max 3%. Stop-Loss serré. Uniquement les setups A+.' :
             strategyMode === 'aggressive' ? 'Allocation jusqu\'à 10%. Accepte plus de risque pour plus de rendement.' :
             'Allocation 1-10% modulaire. Équilibre risque/rendement selon la conviction.'}
          </div>
          <div style={{ marginTop: '0.8rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <span className="tag" style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}>Gemma-4 31B</span>
            <span className="tag" style={{ background: 'var(--green-dim)', color: 'var(--green)' }}>Multi-Ticker</span>
            <span className="tag" style={{ background: 'var(--cyan-dim)', color: 'var(--cyan)' }}>ChromaDB</span>
          </div>
        </div>

        {/* Panic Switch + Drawdown */}
        <div className="card">
          <div className="card-title">🚨 Gestion du Risque</div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start', marginBottom: '0.8rem' }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Drawdown Max</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--red)' }}>{drawdown.max_drawdown || 0}%</div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>Drawdown Actuel</div>
              <div style={{ fontSize: '1.3rem', fontWeight: 800, color: (drawdown.current_drawdown || 0) < -5 ? 'var(--red)' : 'var(--yellow)' }}>
                {drawdown.current_drawdown || 0}%
              </div>
            </div>
          </div>
          {drawdown.series?.length > 1 && (
            <Sparkline data={drawdown.series.map((d: DrawdownPoint) => d.drawdown)} width={260} height={40} color="var(--red)" />
          )}
          <button onClick={handlePanic}
            className={panicConfirm ? "btn btn-danger" : "btn"}
            style={{ width: '100%', marginTop: '0.8rem', justifyContent: 'center', animation: panicConfirm ? 'glow 1s infinite' : 'none' }}>
            {panicConfirm ? '⚠️ CONFIRMER — TOUT LIQUIDER' : '🚨 Panic Switch'}
          </button>
          {panicConfirm && <div style={{ fontSize: '0.65rem', color: 'var(--red)', textAlign: 'center', marginTop: '0.3rem' }}>Cliquez à nouveau pour confirmer. Toutes les positions seront fermées.</div>}
        </div>
      </div>

      {/* ═══ MIDDLE ROW: Equity Chart + Heatmap ═══ */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem', marginBottom: '1rem' }}>

        {/* Equity Chart */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <div className="card-title" style={{ margin: 0 }}>📈 Courbe d'Équité (1 mois)</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
              ${portfolio.equity?.toLocaleString("en-US", { minimumFractionDigits: 2 }) || '0.00'}
              <span style={{ fontSize: '0.8rem', fontWeight: 600, marginLeft: '0.5rem', color: (portfolio.change_pct || 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {(portfolio.change_pct || 0) >= 0 ? '▲' : '▼'} {Math.abs(portfolio.change_pct || 0)}%
              </span>
            </div>
          </div>
          {equityHistory.length > 1 ? (
            <div style={{ position: 'relative', height: '160px', background: 'rgba(0,0,0,0.15)', borderRadius: '8px', padding: '0.8rem', overflow: 'hidden' }}>
              <svg viewBox={`0 0 ${equityHistory.length * 12} 100`} style={{ width: '100%', height: '100%' }} preserveAspectRatio="none">
                {(() => {
                  const eqs = equityHistory.map(h => h.equity);
                  const min = Math.min(...eqs); const max = Math.max(...eqs); const range = max - min || 1;
                  const pts = eqs.map((eq, i) => `${i * 12},${100 - ((eq - min) / range) * 90}`).join(' ');
                  const isUp = eqs[eqs.length - 1] >= eqs[0];
                  const c = isUp ? '#22c55e' : '#ef4444';
                  return (<>
                    <defs>
                      <linearGradient id="eqG" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={c} stopOpacity="0.25" />
                        <stop offset="100%" stopColor={c} stopOpacity="0" />
                      </linearGradient>
                    </defs>
                    <polygon points={`0,100 ${pts} ${(eqs.length - 1) * 12},100`} fill="url(#eqG)" />
                    <polyline points={pts} fill="none" stroke={c} strokeWidth="1.5" />
                  </>);
                })()}
              </svg>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                <span>{equityHistory[0]?.date}</span>
                <span>{equityHistory[equityHistory.length - 1]?.date}</span>
              </div>
            </div>
          ) : <div style={{ height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>Données en chargement...</div>}
        </div>

        {/* Portfolio Heatmap */}
        <div className="card">
          <div className="card-title">🗺️ Heatmap Portefeuille</div>
          {heatmap.length === 0 ? (
            <div style={{ height: '160px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>Aucune position</div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(heatmap.length, 3)}, 1fr)`, gap: '0.5rem' }}>
              {heatmap.map(t => <HeatTile key={t.symbol} tile={t} />)}
            </div>
          )}
        </div>
      </div>

      {/* ═══ BOTTOM ROW: Positions + Activities + Logs + Chat + News ═══ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>

        {/* Positions */}
        <div className="card">
          <div className="card-title">💼 Positions Ouvertes</div>
          {portfolio.status === 'success' && portfolio.positions?.length > 0 ? (
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {portfolio.positions.map((p: Position) => (
                <div key={p.symbol} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0', borderBottom: '1px solid var(--border)' }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{p.symbol}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{p.qty} × ${p.current_price}</div>
                  </div>
                  <div style={{ textAlign: 'right', fontWeight: 700, fontSize: '0.85rem', color: p.unrealized_pl >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {p.unrealized_pl >= 0 ? '+' : ''}{p.unrealized_pl?.toFixed(2)}$
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '2rem 0', textAlign: 'center' }}>
              {portfolio.status === 'loading' ? 'Chargement...' : portfolio.status === 'error' ? 'Connexion perdue' : 'Aucune position'}
            </div>
          )}
          <div style={{ marginTop: '0.8rem', display: 'flex', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Cash: ${portfolio.cash?.toLocaleString() || '—'}</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>BP: ${portfolio.buying_power?.toLocaleString() || '—'}</span>
          </div>
        </div>

        {/* Activities */}
        <div className="card">
          <div className="card-title">📋 Historique des Trades</div>
          {activities.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '2rem 0', textAlign: 'center' }}>Aucun trade récent</div>
          ) : (
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
              {activities.map((a, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.45rem 0', borderBottom: '1px solid var(--border)' }}>
                  <div>
                    <span style={{ fontWeight: 700, color: a.side === 'buy' ? 'var(--green)' : 'var(--red)', fontSize: '0.75rem' }}>
                      {a.side?.toUpperCase()}
                    </span>
                    <span style={{ fontWeight: 600, marginLeft: '0.3rem', fontSize: '0.85rem' }}>{a.symbol}</span>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{a.time ? new Date(a.time).toLocaleString() : ''}</div>
                  </div>
                  <div style={{ textAlign: 'right', fontSize: '0.8rem' }}>
                    <div>{a.qty} actions</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>@ ${a.price}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Logs + Chat */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', maxHeight: '420px' }}>
          <div className="card-title">🧠 Thoughts & Chat</div>

          {/* Logs */}
          <div style={{ flex: 1, overflowY: 'auto', marginBottom: '0.5rem', minHeight: 0 }}>
            {logs.slice(0, 15).map((log, i) => (
              <div key={i} style={{ padding: '0.35rem 0', borderBottom: '1px solid var(--border)', fontSize: '0.78rem' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>{log.time}</span>
                {log.type === 'trade' && <span className="tag" style={{ background: 'var(--green-dim)', color: 'var(--green)', marginLeft: '0.3rem' }}>TRADE</span>}
                {log.type === 'error' && <span className="tag" style={{ background: 'var(--red-dim)', color: 'var(--red)', marginLeft: '0.3rem' }}>ERR</span>}
                <div style={{ color: 'var(--text-secondary)', marginTop: '0.15rem', lineHeight: 1.4, wordBreak: 'break-word' }}>{log.msg}</div>
              </div>
            ))}
          </div>

          {/* Chat */}
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: '0.5rem' }}>
            <div style={{ maxHeight: '120px', overflowY: 'auto', marginBottom: '0.4rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              {chatMessages.map((m, i) => (
                <div key={i} style={{ alignSelf: m.sender === 'user' ? 'flex-end' : 'flex-start' }}>
                  <div className="chat-bubble" style={{ background: m.sender === 'user' ? 'var(--accent)' : 'var(--bg-tertiary)', color: '#fff' }}>
                    {m.text}
                  </div>
                </div>
              ))}
              {isChatLoading && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Gemma-4 réfléchit...</div>}
              <div ref={chatEndRef} />
            </div>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <input type="text" value={chatInput} onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && sendChat()}
                placeholder="Demande à l'IA..." style={{ flex: 1 }} />
              <button className="btn btn-accent btn-sm" onClick={sendChat} disabled={isChatLoading}>Envoyer</button>
            </div>
          </div>
        </div>

        {/* 📰 Market Intelligence (FinBERT) */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', maxHeight: '420px' }}>
          <div className="card-title">📰 Intelligence Marché (FinBERT)</div>
          {newsSentiment.length === 0 ? (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center' }}>
              En attente d'extraction web...
            </div>
          ) : (
            <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {newsSentiment.map((n, i) => {
                const isBull = n.label === 'positive';
                const isBear = n.label === 'negative';
                const col = isBull ? 'var(--green)' : isBear ? 'var(--red)' : 'var(--text-muted)';
                const bg = isBull ? 'var(--green-dim)' : isBear ? 'var(--red-dim)' : 'rgba(255,255,255,0.05)';
                return (
                  <div key={i} style={{ padding: '0.6rem', border: '1px solid var(--border)', borderRadius: '6px', background: 'var(--bg-secondary)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.3rem' }}>
                      <span className="tag" style={{ background: bg, color: col, fontSize: '0.65rem' }}>
                        {n.label.toUpperCase()}
                      </span>
                      <span style={{ fontSize: '0.65rem', color: col, fontWeight: 700 }}>
                        Score: {(n.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div style={{ fontSize: '0.78rem', lineHeight: 1.4, color: 'var(--text-secondary)' }}>
                      {n.title}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
