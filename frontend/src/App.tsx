import React, { useState, useRef, useEffect } from 'react'
import type { TicketInput, StreamPayload } from './types'
import './App.css'

/* =========================
   CONFIG & BASE MODELS
   ========================= */
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const emptyTicket: TicketInput = { account: '', issue: '', channel: 'web', user_id: 'anon' }

/* =========================
   HELPERS (capitalize, copy, mode, digits)
   ========================= */
const capFirst = (s?: string | null) => (s && s.length ? s[0].toUpperCase() + s.slice(1) : s ?? '—')
function copy(text: string) { try { navigator.clipboard?.writeText(text ?? '') } catch { /* noop */ } }
function mode<T>(arr: T[]): T | undefined {
  if (!arr.length) return undefined
  const m = new Map<T, number>()
  let best: T | undefined
  let bestN = 0
  for (const v of arr) {
    const n = (m.get(v) || 0) + 1
    m.set(v, n)
    if (n > bestN) { bestN = n; best = v }
  }
  return best
}
function extractFirstDigits(v: any): number | undefined {
  if (v === null || v === undefined) return undefined
  const m = String(v).trim().match(/\d+/)
  return m ? Number(m[0]) : undefined
}

function getUrgencyCodeFromTicket(t: any): number | undefined {
  if (!t) return undefined
  // try common keys first (with different casings)
  const candidates = [
    'urgency_code','Urgency_Code','URGENCY_CODE',
    'code','Code','CODE',
    'urgency','Urgency','URGENCY',
    'inferred_urgency_code','inferredUrgencyCode','InferredUrgencyCode','INFERRED_URGENCY_CODE',
    'UrgencyCode','URGENCYCODE'
  ]
  for (const k of candidates) {
    const n = extractFirstDigits((t as any)[k])
    if (n !== undefined) return n
  }
  // last resort: scan all keys that look like code/urgency
  for (const [k, v] of Object.entries(t)) {
    if (/urgency|code/i.test(k)) {
      const n = extractFirstDigits(v)
      if (n !== undefined) return n
    }
  }
  return undefined
}


/* =========================
   RICH TEXT RENDERER (keeps your bullets neat)
   ========================= */
function renderFormatted(text: string) {
  if (!text || !text.trim()) return <div className="subtle">—</div>

  const lines = text.split(/\r?\n/)
  const blocks: JSX.Element[] = []
  let list: null | { ordered: boolean; items: string[] } = null

  // NEW: keep ordered numbering across multiple <ol> blocks
  let orderedCounter = 0

  const flushList = () => {
    if (!list) return
    const items = list.items.slice()
    const isOrdered = list.ordered
    list = null

    if (isOrdered) {
      const start = orderedCounter + 1
      orderedCounter += items.length
      blocks.push(
        <ol key={`list-${blocks.length}`} className="rich-list" start={start}>
          {items.map((it, idx) => (<li key={idx}>{renderInline(it)}</li>))}
        </ol>
      )
    } else {
      blocks.push(
        <ul key={`list-${blocks.length}`} className="rich-list">
          {items.map((it, idx) => (<li key={idx}>{renderInline(it)}</li>))}
        </ul>
      )
    }
  }

  function renderInline(s: string): Array<JSX.Element | string> {
    return s.split(/(\*\*[^*]+?\*\*)/g).map((part, i) => {
      if (/^\*\*[^*]+?\*\*$/.test(part)) return <strong key={i}>{part.slice(2, -2)}</strong>
      return part
    })
  }

  for (const raw of lines) {
    const line = raw.trimRight()
    const trimmed = line.trim()
    if (trimmed === '') { if (list) continue; continue }

    // bullets: -, *, •, –
    const bulletMatch = trimmed.match(/^([-*•–])\s+(.*)$/)
    // ordered: 1. or 1)
    const orderedMatch = trimmed.match(/^(\d+)[\.)]\s+(.*)$/)

    if (bulletMatch || orderedMatch) {
      const isOrdered = !!orderedMatch
      const content = isOrdered ? orderedMatch![2] : bulletMatch![2]
      if (!list || list.ordered !== isOrdered) { flushList(); list = { ordered: isOrdered, items: [] } }
      list.items.push(content)
      continue
    }

    // headings like "Resolution:"
    const headingMatch = trimmed.match(/^(auto-?response|resolution|steps?)\s*:?\s*$/i)
    if (headingMatch) {
      flushList()
      // NEW: reset ordered numbering when a new heading starts
      orderedCounter = 0
      blocks.push(
        <h4 key={`h-${blocks.length}`} className="rich-h">
          {headingMatch[1].replace(/^\w/, c => c.toUpperCase())}
        </h4>
      )
      continue
    }

    flushList()
    blocks.push(<p key={`p-${blocks.length}`} className="rich-p">{renderInline(trimmed)}</p>)
  }

  flushList()
  return <div className="rich">{blocks}</div>
}


/* =========================
   SMALL DISPLAY PIECES
   ========================= */
function Kpi({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="kpi">
      <div className="label" style={{ fontWeight: 700 }}>{label}</div>
      <div className="value">{value ?? '—'}</div>
    </div>
  )
}

function Steps({ lines }: { lines: string[] }) {
  const classFor = (l: string) => {
    if (l.startsWith('Intake')) return 'chip intake'
    if (l.startsWith('Understanding')) return 'chip understanding'
    if (l.startsWith('Prioritization')) return 'chip prioritization'
    if (l.startsWith('Routing')) return 'chip routing'
    if (l.startsWith('Knowledge')) return 'chip knowledge'
    if (l.startsWith('Resolution')) return 'chip resolution'
    if (l.startsWith('Done')) return 'chip done'
    if (l.startsWith('Error')) return 'chip error'
    return 'chip'
  }
  return (
    <div className="steps">
      {lines.map((l, i) => (
        <div key={i} className={classFor(l)}>
          <span className="dot" /> {l}
        </div>
      ))}
    </div>
  )
}

/* =========================
   SUMMARY BLOCK (RIGHT SIDE OUTPUT)
   ========================= */
function SummaryBlock({ s }: { s?: StreamPayload }) {
  if (!s) return null

  // Capitalise nice-to-read fields
  const sentiment = capFirst(s.sentiment)
  const intent = capFirst(s.intent)

  // (1) KPI: Average urgency code (rounded) across ALL similar tickets
  const urgencyCode = React.useMemo(() => {
    const codes = (s?.similar_tickets ?? [])
      .map((t: any) => getUrgencyCodeFromTicket(t))
      .filter((n): n is number => typeof n === 'number' && Number.isFinite(n))
    if (!codes.length) return undefined
    const avg = Math.round(codes.reduce((a, b) => a + b, 0) / codes.length)
    return String(avg)
  }, [s?.similar_tickets])

  const urgencyCodeDisplay = urgencyCode?.toString().trim() || undefined

  // Visuals for similarity & level badge
  const sim = typeof s.resolution_similarity_score === 'number' ? s.resolution_similarity_score : undefined
  const simPct = sim !== undefined ? Math.max(0, Math.min(100, Math.round(sim * 100))) : undefined
  const level = (s.priority_level ?? '').toLowerCase()
  const levelClass = level === 'high' ? 'badge red' : level === 'normal' ? 'badge amber' : level ? 'badge green' : ''

  return (
    <div className="card">
      <h3 style={{ fontWeight: 800 }}>🔎 Result Summary</h3>

      {/* KPIs row */}
      <div className="kpis" style={{ margin: '8px 0 12px' }}>
        <Kpi label="Ticket ID" value={s.ticket_id ?? '—'} />
        <Kpi label="Priority" value={capFirst(s.priority)} />
        <Kpi label="Category" value={capFirst(s.category)} />
        <Kpi label="Sentiment" value={sentiment} />
      </div>

      {/* --- Section: Intent / Assigned / Urgency (bold titles + spacing) --- */}
      <div className="summary-section">
        <div className="summary-title">Intent</div>
        <div className="summary-value">{intent}</div>
      </div>
      <div className="summary-section">
        <div className="summary-title">Assigned To</div>
        <div className="summary-value">{s.assigned_to ?? '—'}</div>
      </div>
      <div className="summary-section">
        <div className="summary-title">Inferred Urgency</div>
        <div className="summary-value">{capFirst(s.inferred_urgency)}</div>
      </div>

      {/* --- Section: Two-column long text (Auto-response / Resolution) --- */}
      <div className="two-col summary-section">
        <div>
          <div className="summary-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Auto-response</span>
            <button className="btn" type="button" onClick={() => copy(s.auto_response ?? '')}>Copy</button>
          </div>
          <div className="scroll">{renderFormatted(s.auto_response ?? '')}</div>
        </div>
        <div>
          <div className="summary-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Resolution</span>
            <button className="btn" type="button" onClick={() => copy(s.resolution ?? '')}>Copy</button>
          </div>
          <div className="scroll">{renderFormatted(s.resolution ?? '')}</div>
        </div>
      </div>

      {/* --- Section: Metrics + Similar tickets --- */}
      <div className="summary-section">
        <div className="kpis" style={{ marginBottom: 8 }}>
          <Kpi
            label="Resolution Sim."
            value={
              simPct !== undefined ? (
                <div>
                  <div style={{ fontWeight: 700 }}>{sim.toFixed(2)}</div>
                  <div className="meter" style={{ marginTop: 6 }}><span style={{ width: `${simPct}%` }} /></div>
                </div>
              ) : '—'
            }
          />
          <Kpi label="Urgency Code" value={urgencyCodeDisplay ?? '—'} />
          <Kpi label="Level" value={<span className={levelClass || undefined}>{capFirst(s.priority_level)}</span>} />
          <Kpi label="Similar Count" value={s.similar_tickets?.length ?? 0} />
        </div>

        <div className="subtle" style={{ marginBottom: 6 }}>Similar Tickets</div>
        <div className="tickets-list scroll" style={{ maxHeight: '30vh' }}>
          {(s.similar_tickets ?? []).length === 0 && <div className="subtle">No matches.</div>}
          {(s.similar_tickets ?? []).map((t, idx) => (
            <details key={idx} className="ticket" style={{ marginBottom: 8, background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: 10 }}>
              <summary style={{ cursor: 'pointer', display: 'flex', gap: 10, alignItems: 'center' }}>
                <span className="badge">Past Ticket #{idx + 1}</span>
                {(() => {
                  const code = getUrgencyCodeFromTicket(t as any)
                  return code !== undefined
                    ? <span className="badge amber">Urgency Code: {code}</span>
                    : null
                })()}
              </summary>
              <div style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>{(t as any).full ?? ''}</div>
              {(t as any).solution && (
                <div style={{ marginTop: 8 }}>
                  <div className="subtle">Solution</div>
                  <div className="code">{(t as any).solution}</div>
                </div>
              )}
            </details>
          ))}
        </div>
      </div>
    </div>
  )
}

/* =========================
   MAIN APP
   ========================= */
export default function App() {
  // form state
  const [tickets, setTickets] = useState<TicketInput[]>([{ ...emptyTicket }])
  const [busy, setBusy] = useState(false)
  const [logs, setLogs] = useState<Record<number, string[]>>({})
  const [summaries, setSummaries] = useState<Record<number, StreamPayload>>({})

  // measure intake panel height (to size steps panel)
  const intakeRef = useRef<HTMLFormElement | null>(null)
  const [intakeHeight, setIntakeHeight] = useState<number | undefined>(undefined)
  useEffect(() => {
    const measure = () => { if (intakeRef.current) setIntakeHeight(intakeRef.current.clientHeight) }
    measure()
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [])

  // helpers to edit tickets
  const add = () => setTickets(t => [...t, { ...emptyTicket }])
  const remove = (i: number) => setTickets(t => t.filter((_, idx) => idx !== i))
  const update = (i: number, patch: Partial<TicketInput>) =>
    setTickets(t => t.map((x, idx) => (idx === i ? { ...x, ...patch } : x)))

  // submit -> SSE stream
  const submitAll = async (e: React.FormEvent) => {
    e.preventDefault()
    const valid = tickets.filter(t => t.account.trim() && t.issue.trim())
    if (!valid.length) return alert('Please fill at least one ticket with Account and Issue.')

    setBusy(true); setLogs({}); setSummaries({})
    for (let i = 0; i < valid.length; i++) {
      // eslint-disable-next-line no-await-in-loop
      await new Promise<void>((resolve) => {
        const t = valid[i]
        const params = new URLSearchParams({
          user_id: t.user_id ?? 'anon',
          channel: t.channel ?? 'web',
          account: t.account,
          issue: t.issue,
          contact: t.contact ?? '',
          project: t.project ?? '',
          area: t.area ?? '',
          input_category: t.input_category ?? ''
        })

        const es = new EventSource(`${API_BASE}/api/triage-stream?${params.toString()}`)
        const step = (name: string, text: string) =>
          setLogs(L => ({ ...L, [i]: [...(L[i] ?? []), `${name}: ${text}`] }))

        es.addEventListener('intake', () => step('Intake', 'Processing metadata'))
        es.addEventListener('understanding', () => step('Understanding', 'Analyzing intent, entities, sentiment, category'))
        es.addEventListener('prioritization', () => step('Prioritization', 'Determining priority'))
        es.addEventListener('routing', () => step('Routing', 'Selecting team'))
        es.addEventListener('knowledge', () => step('Knowledge', 'Generating auto-response'))
        es.addEventListener('resolution', () => step('Resolution', 'Drafting resolution'))
        es.addEventListener('similar', (ev) => {
          const payload: StreamPayload = JSON.parse((ev as MessageEvent).data)
          setSummaries(S => ({ ...S, [i]: { ...(S[i] ?? {}), ...payload } }))
          step('Similarity', 'Retrieved similar tickets')
        })
        es.addEventListener('done', (ev) => {
          // Normalize so UI always has a similar_tickets array
          const raw: StreamPayload = JSON.parse((ev as MessageEvent).data)
          const normalized: StreamPayload = {
            ...raw,
            similar_tickets: (raw as any).similar_tickets ?? (raw as any).similar ?? [],
          } as any

          setSummaries(S => ({ ...S, [i]: { ...(S[i] ?? {}), ...normalized } }))
          step('Done', 'Ticket completed')
          es.close(); resolve()
        })

        es.onerror = () => { step('Error', 'Stream error – check backend'); es.close(); resolve() }
      })
    }
    setBusy(false)
  }

  /* ---------- RENDER ---------- */
  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="brand">
          {/* Logo (40x40) */}
          <img
            src="/my-logo.png"
            alt="App logo"
            style={{ width: '40px', height: '40px', objectFit: 'contain', borderRadius: '8px', flex: '0 0 40px' }}
          />
          <div>
            <div className="title">STAR — Smart Triage & Automated Routing</div>
            <div className="subtle">Smarter triage for faster resolutions and happier customers.</div>
          </div>
        </div>
        <div className="pill">API: {API_BASE}</div>
      </header>

      {/* Info: What it does (sits between header and grid) */}
      <div className="card" style={{ marginTop: 12, marginBottom: 24 }}>
        <h3>ℹ️ STAR — What it does</h3>
        <p className="subtle" style={{ marginTop: 6 }}>
          The STAR system ingests incoming tickets and immediately applies AI-powered analysis. It detects the intent and key details, compares them against past cases, and determines the urgency level. Each ticket is then prioritised and routed to the right person at the right time. Along the way, the system generates auto-response drafts and explains each step for full transparency. The result is faster resolutions, reduced workload for support teams, and a better experience for customers.
        </p>
      </div>

      {/* Top grid: left intake, right steps */}
      <div className="main">
        {/* Left: Intake form */}
        <form className="card" onSubmit={submitAll} ref={intakeRef}>
          <h3>📝 Ticket Intake</h3>

          {tickets.map((t, i) => (
            <div key={i} className="card" style={{ marginBottom: 12, padding: 14 }}>
              <div className="row">
                <div>
                  <label>Account <span className="subtle">(required)</span></label>
                  <input value={t.account} onChange={e => update(i, { account: e.target.value })} placeholder="Client / Department" required />
                </div>
                <div>
                  <label>Phone / E-Mail</label>
                  <input value={t.contact ?? ''} onChange={e => update(i, { contact: e.target.value })} placeholder="+44… or name@company.com" />
                </div>
              </div>

              <div className="row">
                <div>
                  <label>Project</label>
                  <input value={t.project ?? ''} onChange={e => update(i, { project: e.target.value })} placeholder="Optional" />
                </div>
                <div>
                  <label>Area</label>
                  <input value={t.area ?? ''} onChange={e => update(i, { area: e.target.value })} placeholder="e.g., Accounts Payable" />
                </div>
              </div>

              <div className="row">
                <div>
                  <label>Category (internal code)</label>
                  <input value={t.input_category ?? ''} onChange={e => update(i, { input_category: e.target.value })} placeholder="e.g., TECH-AUTH-01" />
                </div>
                <div>
                  <label>Channel</label>
                  <input value={t.channel ?? 'web'} onChange={e => update(i, { channel: e.target.value })} />
                </div>
              </div>

              <div>
                <label>Issue <span className="subtle">(required)</span></label>
                <textarea rows={5} value={t.issue} onChange={e => update(i, { issue: e.target.value })} placeholder="Describe the problem clearly…" required />
              </div>

              <div className="btns">
                {tickets.length > 1 && <button className="btn" type="button" onClick={() => remove(i)}>Remove</button>}
                {i === tickets.length - 1 && <button className="btn" type="button" onClick={add}>+ Add another ticket</button>}
              </div>
            </div>
          ))}

          <div className="btns">
            <button className="btn primary" disabled={busy} type="submit">{busy ? 'Processing…' : 'Submit All'}</button>
            <button className="btn" type="button" onClick={() => setTickets([{ ...emptyTicket }])} disabled={busy}>Reset</button>
          </div>
        </form>

        {/* Right: Agent steps (height matched to left card) */}
        <div className="right-sticky">
          <div className="card">
            <h3>🧭 Agent Steps</h3>
            <div className="subtle" style={{ marginTop: -4, marginBottom: 10 }}>
              Live view of each stage as your ticket streams through the system.
            </div>
            <div className="scroll" style={{ maxHeight: intakeHeight ? intakeHeight - 120 : '35vh', overflowY: 'auto' }}>
              {tickets.map((_, i) => (
                <div key={i} style={{ marginBottom: 12 }}>
                  <div className="subtle" style={{ marginBottom: 6 }}>Ticket #{i + 1}</div>
                  <Steps lines={logs[i] ?? []} />
                </div>
              ))}
              {tickets.length === 0 && <div className="subtle">No tickets yet.</div>}
            </div>
          </div>
        </div>
      </div>

      {/* Full-width summary*/}
      <div style={{ marginTop: 20 }}>
        {tickets.map((_, i) => (
          <div key={i} style={{ marginBottom: 20 }}>
            <SummaryBlock s={summaries[i]} />
          </div>
        ))}
      </div>
    </div>
  )
}
