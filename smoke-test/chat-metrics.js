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
	const avg =
		s.totalDecodeMs > 0 ? s.totalOutputTokens / (s.totalDecodeMs / 1000) : 0;
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
	// Match the bitmap to the CSS box at devicePixelRatio so the line
	// stays crisp on retina (a 120×24 bitmap upscaled to 120×24 CSS px is
	// sharp on 1×, blurry on 2×). Uses clientWidth/Height so the size
	// tracks any responsive CSS overrides.
	const dpr = globalThis.devicePixelRatio || 1;
	const cssW = canvas.clientWidth || canvas.width;
	const cssH = canvas.clientHeight || canvas.height;
	if (
		canvas.width !== Math.round(cssW * dpr) ||
		canvas.height !== Math.round(cssH * dpr)
	) {
		canvas.width = Math.round(cssW * dpr);
		canvas.height = Math.round(cssH * dpr);
	}
	ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	ctx.clearRect(0, 0, cssW, cssH);
	if (history.length === 0) return;

	// Y-axis: tight min/max with padding so small fluctuations are visible.
	// For a single point, draw a centered horizontal line.
	const lo = Math.min(...history);
	const hi = Math.max(...history);
	const range = Math.max(hi - lo, hi * 0.1, 1); // never collapse to zero
	const yMin = lo - range * 0.15;
	const yMax = hi + range * 0.15;
	const span = yMax - yMin || 1;

	const padX = 2;
	const padY = 3;
	const innerW = cssW - padX * 2;
	const innerH = cssH - padY * 2;
	const xAt = (i) =>
		history.length > 1
			? padX + (i / (history.length - 1)) * innerW
			: padX + innerW / 2;
	const yAt = (v) => padY + innerH - ((v - yMin) / span) * innerH;

	// Fill area under the line for body. Translucent so the pill background
	// shows through but the trend reads at a glance.
	ctx.beginPath();
	ctx.moveTo(padX, padY + innerH);
	history.forEach((v, i) => {
		ctx.lineTo(xAt(i), yAt(v));
	});
	ctx.lineTo(xAt(history.length - 1), padY + innerH);
	ctx.closePath();
	ctx.fillStyle = "rgba(74, 144, 226, 0.25)";
	ctx.fill();

	// Stroke the trend line.
	ctx.beginPath();
	history.forEach((v, i) => {
		const x = xAt(i);
		const y = yAt(v);
		if (i === 0) ctx.moveTo(x, y);
		else ctx.lineTo(x, y);
	});
	if (history.length === 1) {
		// Force a visible mark for a single sample.
		ctx.lineTo(xAt(0) + 0.01, yAt(history[0]));
	}
	ctx.strokeStyle = "#6db4ff";
	ctx.lineWidth = 1.75;
	ctx.lineJoin = "round";
	ctx.stroke();

	// Most-recent point marker so the latest reading pops.
	const last = history[history.length - 1];
	ctx.beginPath();
	ctx.arc(xAt(history.length - 1), yAt(last), 2.25, 0, Math.PI * 2);
	ctx.fillStyle = "#a4cfff";
	ctx.fill();
}
