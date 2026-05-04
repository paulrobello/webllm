// chat-metrics.js — per-turn metrics, context bar math, session totals,
// sparkline render.

const NUM = new Intl.NumberFormat("en-US");

export function contextBarState(used, max) {
  if (max <= 0) return "neutral";
  // Compare against the displayed rounded percent so the visual threshold
  // matches the bar/label state (avoids 94.995% display showing as amber).
  const pctRounded = Math.round((used / max) * 100);
  if (pctRounded >= 95) return "red";
  if (pctRounded >= 80) return "amber";
  return "neutral";
}

export function formatContext(used, max) {
  const pct = max > 0 ? Math.round((used / max) * 100) : 0;
  return `${NUM.format(used)} / ${NUM.format(max)} — ${pct}%`;
}

export function formatLastTurn(m) {
  return `last: ${Math.round(m.ttftMs)}ms TTFT · ${tps(m.outputTokens, m.decodeMs).toFixed(1)} tok/s · ${(m.totalMs / 1000).toFixed(1)}s · ${m.outputTokens} tok`;
}

export function formatSession(s) {
  const avg = s.totalDecodeMs > 0 ? (s.totalOutputTokens / (s.totalDecodeMs / 1000)) : 0;
  return `session: ${s.turns} turns · ${NUM.format(s.totalOutputTokens)} tok · ⌀ ${avg.toFixed(1)} tok/s`;
}

function tps(tokens, ms) {
  return ms > 0 ? tokens / (ms / 1000) : 0;
}

/** Mutable session totals — owned by chat-page. */
export function newSessionTotals() {
  return { turns: 0, totalOutputTokens: 0, totalDecodeMs: 0, history: [] };
}

export function addTurn(session, m) {
  session.turns += 1;
  session.totalOutputTokens += m.outputTokens;
  session.totalDecodeMs += m.decodeMs;
  session.history.push(tps(m.outputTokens, m.decodeMs));
  if (session.history.length > 20) session.history.shift();
}

/** Apply state + width + label to the context-bar DOM trio. */
export function applyContextBar(barInner, labelEl, used, max) {
  const pct = max > 0 ? Math.min(100, (used / max) * 100) : 0;
  barInner.style.width = `${pct}%`;
  barInner.classList.remove("amber", "red");
  const state = contextBarState(used, max);
  if (state !== "neutral") barInner.classList.add(state);
  labelEl.textContent = `context ${formatContext(used, max)}`;
  labelEl.classList.remove("amber", "red");
  if (state !== "neutral") labelEl.classList.add(state);
}

/** Render a 20-point sparkline of decode tok/s into a canvas. */
export function renderSparkline(canvas, history) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;
  const max = Math.max(...history) * 1.1 || 1;
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * (w - 2) + 1;
    const y = h - 1 - (v / max) * (h - 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#4a90e2";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}
