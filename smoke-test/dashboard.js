import {
	buildEmbeddingCosineChartData,
	buildEmbeddingLatencyChartData,
	buildEmbeddingThroughputChartData,
	buildTempSweepChartData,
	DIM_NAMES,
	tempBucket,
} from "./dashboard-charts.js";

// ── state ────────────────────────────────────────────────────────
const state = {
	runsByRunId: new Map(),
	runningByRunId: new Map(),
	failedByRunId: new Map(),
	sortKey: "timestamp",
	sortDir: "desc",
	thinkingFilter: "all",
	textFilter: "",
	selected: new Set(),
	freshRunIds: new Set(),
	freshEvalIds: new Set(),
	freshBarKeys: new Set(),
	// Evals
	evalsByEvalId: new Map(),
	runningEvalsByEvalId: new Map(),
	evalSortKey: "timestamp",
	evalSortDir: "desc",
	// System profiles
	systemProfilesById: new Map(),
	// Dim-card bar animation: remember which model cards we've already
	// rendered so only freshly-added cards animate from 0 to value.
	renderedDimModels: new Set(),
	// Historical eval series for score-over-time chart (fetched from server).
	evalSeries: [],
};

// ── SSE connection ────────────────────────────────────────────────
let eventSource = null;
let lastSeq = 0;

function setStatus(kind, text, meta = "") {
	const host = document.querySelector(".status");
	host.className = `status ${kind}`;
	document.getElementById("status-text").textContent = text;
	document.getElementById("status-meta").textContent = meta;
}

function connect() {
	const url = lastSeq > 0 ? `/stream?lastSeq=${lastSeq}` : "/stream";
	const es = new EventSource(url);
	eventSource = es;

	es.addEventListener("open", () => {
		setStatus("ok", "connected");
	});

	const kinds = [
		"snapshot",
		"run_started",
		"run_complete",
		"run_failed",
		"eval_started",
		"eval_task_complete",
		"eval_complete",
		"eval_failed",
		"reset",
	];
	for (const kind of kinds) {
		es.addEventListener(kind, (ev) => handleEvent(ev));
	}

	es.addEventListener("error", () => {
		setStatus("err", "reconnecting…");
		// native EventSource auto-reconnects; no manual retry needed
	});
}

function handleEvent(ev) {
	try {
		const evt = JSON.parse(ev.data);
		switch (evt.kind) {
			case "snapshot":
				// Snapshot is authoritative — reset lastSeq so a backend restart
				// (which restarts seq numbering at 1) doesn't leave us stranded.
				lastSeq = typeof evt.seq === "number" ? evt.seq : 0;
				state.runsByRunId.clear();
				for (const run of evt.payload.runs ?? []) {
					state.runsByRunId.set(run.runId, run);
				}
				state.evalsByEvalId.clear();
				for (const rep of evt.payload.evals ?? []) {
					state.evalsByEvalId.set(rep.evalId, rep);
				}
				state.systemProfilesById.clear();
				for (const sys of evt.payload.systemProfiles ?? []) {
					state.systemProfilesById.set(sys.systemId, sys);
				}
				state.runningByRunId.clear();
				state.runningEvalsByEvalId.clear();
				state.failedByRunId.clear();
				// Pre-seed "rendered" tracking so the initial snapshot paints
				// without flashing/animating every existing card+row.
				state.renderedDimModels.clear();
				render();
				return;
			case "run_started":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runningByRunId.set(evt.payload.runId, evt.payload);
				renderRunning();
				break;
			case "run_complete":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runningByRunId.delete(evt.payload.runId);
				state.failedByRunId.delete(evt.payload.runId);
				state.runsByRunId.set(evt.payload.runId, evt.payload);
				flashFresh(evt.payload.runId);
				flashFreshBar(evt.payload.profile ?? evt.payload.model);
				render();
				break;
			case "run_failed":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runningByRunId.delete(evt.payload.runId);
				state.failedByRunId.set(evt.payload.runId, evt.payload);
				renderRunning();
				renderTable();
				break;
			case "eval_started":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runningEvalsByEvalId.set(evt.payload.evalId, {
					...evt.payload,
					completedTasks: 0,
					passedTasks: 0,
				});
				renderRunning();
				renderEvals();
				break;
			case "eval_task_complete":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				{
					const live = state.runningEvalsByEvalId.get(evt.payload.evalId);
					if (live) {
						live.completedTasks = (live.completedTasks ?? 0) + 1;
						if (evt.payload.score >= 0.5) {
							live.passedTasks = (live.passedTasks ?? 0) + 1;
						}
						renderRunning();
						renderEvals();
					}
				}
				break;
			case "eval_complete":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runningEvalsByEvalId.delete(evt.payload.evalId);
				state.evalsByEvalId.set(evt.payload.evalId, evt.payload);
				flashFreshEval(evt.payload.evalId);
				// Invalidate the cached `/evals/series` snapshot so the score-
				// over-time chart re-fetches and includes this fresh eval.
				seriesLoaded = false;
				renderRunning();
				renderEvals();
				break;
			case "eval_failed":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runningEvalsByEvalId.delete(evt.payload.evalId);
				renderRunning();
				renderEvals();
				break;
			case "reset":
				if (typeof evt.seq === "number") lastSeq = Math.max(lastSeq, evt.seq);
				state.runsByRunId.clear();
				state.runningByRunId.clear();
				state.failedByRunId.clear();
				state.evalsByEvalId.clear();
				state.runningEvalsByEvalId.clear();
				state.selected.clear();
				state.renderedDimModels.clear();
				state.evalSeries = [];
				seriesLoaded = false;
				render();
				break;
		}
	} catch (err) {
		console.error("bad SSE payload", err, ev);
	}
}

/**
 * Animate every bar fill within `scope` from its current width to the
 * `data-width` percentage. Bars are rendered with style="width:0%" +
 * data-width="<pct>"; this kicks them to their target on the next
 * animation frame so the CSS `transition` property fires.
 */
function animateBars(scope) {
	const root = scope ?? document;
	const bars = root.querySelectorAll("[data-width]");
	if (bars.length === 0) return;
	// For freshly-inserted bars, the "width:0%" starting state and the
	// target width can coalesce into one paint — the browser never registers
	// the 0% origin, so the CSS transition has no delta to animate. Force a
	// layout read on each bar to commit the starting width, then flip to
	// target on the next frame.
	for (const el of bars) void el.offsetWidth;
	requestAnimationFrame(() => {
		for (const el of bars) {
			const w = el.getAttribute("data-width");
			if (w !== null) el.style.width = `${w}%`;
		}
	});
}

function systemShortLabel(systemId) {
	if (!systemId) return null;
	const sys = state.systemProfilesById.get(systemId);
	if (!sys) return systemId.slice(0, 8);
	const gpu = sys.gpuArchitecture || sys.gpuVendor || sys.gpuDevice;
	const chrome = sys.chromeVersion ? `Chrome ${sys.chromeVersion}` : null;
	return [gpu, chrome].filter(Boolean).join(" · ") || systemId.slice(0, 8);
}

function systemPill(systemId) {
	if (!systemId) return "";
	const label = systemShortLabel(systemId);
	const title = systemId;
	return `<span class="pill system" title="${escapeHtml(title)}" data-system-id="${escapeHtml(systemId)}">${escapeHtml(label)}</span>`;
}

function flashFresh(runId) {
	state.freshRunIds.add(runId);
	setTimeout(() => {
		state.freshRunIds.delete(runId);
		const tr = document.querySelector(`tr[data-run-id="${runId}"]`);
		tr?.classList.remove("fresh");
	}, 1800);
}

function flashFreshEval(evalId) {
	state.freshEvalIds.add(evalId);
	setTimeout(() => {
		state.freshEvalIds.delete(evalId);
		const tr = document.querySelector(`tr[data-eval-id="${evalId}"]`);
		tr?.classList.remove("fresh");
	}, 1800);
}

function flashFreshBar(key) {
	if (!key) return;
	state.freshBarKeys.add(key);
	setTimeout(() => {
		state.freshBarKeys.delete(key);
		const el = document.querySelector(
			`.bar-row[data-key="${cssEscape(key)}"]`,
		);
		el?.classList.remove("fresh");
	}, 1800);
}

// Minimal CSS.escape fallback (older browsers/test envs lack it).
function cssEscape(value) {
	if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
		return CSS.escape(value);
	}
	return String(value).replace(/["\\]/g, "\\$&");
}

// ── rendering ────────────────────────────────────────────────────

function render() {
	renderRunning();
	renderChart();
	renderToolChart();
	renderScatterChart();
	renderDimGroupedChart();
	renderEmbeddingCosineChart();
	renderEmbeddingLatencyChart();
	renderEmbeddingThroughputChart();
	renderTempSweepChart();
	renderThinkingDeltaChart();
	renderTtftChart();
	renderFinishChart();
	renderTable();
	renderCompareButton();
	renderEvals();
}

function renderRunning() {
	const host = document.getElementById("running-list");
	const speedRuns = Array.from(state.runningByRunId.values());
	const evalRuns = Array.from(state.runningEvalsByEvalId.values());
	if (speedRuns.length === 0 && evalRuns.length === 0) {
		host.className = "running-list empty";
		host.textContent = "nothing running";
		return;
	}
	if (host.classList.contains("empty")) {
		host.textContent = "";
	}
	host.className = "running-list";

	// Diff existing items by key so we can mutate individual fields
	// (progress bar width, label text) without tearing the DOM down —
	// that's what makes the bar animate smoothly task-to-task.
	const existing = new Map();
	for (const el of host.querySelectorAll(".running-item[data-key]")) {
		existing.set(el.dataset.key, el);
	}
	const seen = new Set();

	for (const item of speedRuns) {
		const key = `speed:${item.runId}`;
		seen.add(key);
		const label = item.profile ?? item.model;
		let el = existing.get(key);
		if (!el) {
			el = document.createElement("div");
			el.className = "running-item";
			el.dataset.key = key;
			host.appendChild(el);
		}
		el.innerHTML = `
			<span class="spinner"></span>
			<span class="pill phase">speed</span>
			<strong>${escapeHtml(label)}</strong>
			<span class="dim">${escapeHtml(item.model)}</span>
			<span class="pill ${item.thinking === "on" ? "think-on" : ""}">${item.thinking}</span>
			<span class="dim">${escapeHtml(item.prompt)}</span>
		`;
	}

	for (const item of evalRuns) {
		const key = `eval:${item.evalId}`;
		seen.add(key);
		const label = item.label ?? item.modelId;
		const completed = item.completedTasks ?? 0;
		const total = item.totalTasks ?? 0;
		const passed = item.passedTasks ?? 0;
		const pct = total > 0 ? Math.min(100, (completed / total) * 100) : 0;
		let el = existing.get(key);
		if (!el) {
			el = document.createElement("div");
			el.className = "running-item";
			el.dataset.key = key;
			el.innerHTML = `
				<span class="spinner"></span>
				<span class="pill phase accuracy">accuracy</span>
				<strong class="eval-label"></strong>
				<span class="dim eval-model"></span>
				<div class="eval-progress" role="progressbar" aria-valuemin="0">
					<div class="eval-progress-fill" style="width:0%"></div>
					<div class="eval-progress-label"></div>
				</div>
				<span class="dim eval-passing"></span>
			`;
			host.appendChild(el);
		}
		el.querySelector(".eval-label").textContent = label;
		el.querySelector(".eval-model").textContent = item.modelId;
		const progress = el.querySelector(".eval-progress");
		progress.setAttribute("aria-valuemax", String(total));
		progress.setAttribute("aria-valuenow", String(completed));
		el.querySelector(".eval-progress-fill").style.width = `${pct.toFixed(2)}%`;
		el.querySelector(".eval-progress-label").textContent = `${completed}/${total}`;
		el.querySelector(".eval-passing").textContent = `${passed} passing`;
	}

	// Drop items that have stopped running since last render.
	for (const [key, el] of existing) {
		if (!seen.has(key)) el.remove();
	}
}

// ── Chart.js integration ─────────────────────────────────────────────
// Chart instances are module-local so renders update in place (keeps
// animations smooth and avoids leaking canvas listeners on refresh).
let speedChartInstance = null;
let toolChartInstance = null;
let scatterChartInstance = null;
let dimGroupedChartInstance = null;
let tempSweepChartInstance = null;
let thinkingDeltaChartInstance = null;
let ttftChartInstance = null;
let finishChartInstance = null;
let seriesChartInstance = null;

// Pulled from dashboard.css so Chart.js datasets match the legend chips.
const CHART_COLORS = {
	blue: "#58a6ff",
	purple: "#bc8cff",
	green: "#3fb950",
	yellow: "#d29922",
	red: "#f85149",
	text: "#c9d1d9",
	muted: "#8b949e",
	grid: "rgba(139, 148, 158, 0.15)",
};

function baseChartOptions({ xTitle, tooltipLabel } = {}) {
	return {
		indexAxis: "y",
		responsive: true,
		maintainAspectRatio: false,
		animation: { duration: 400 },
		layout: { padding: { left: 6, right: 16, top: 4, bottom: 4 } },
		scales: {
			x: {
				beginAtZero: true,
				ticks: { color: CHART_COLORS.muted, font: { size: 11 } },
				grid: { color: CHART_COLORS.grid, drawBorder: false },
				title: xTitle
					? { display: true, text: xTitle, color: CHART_COLORS.muted }
					: undefined,
			},
			y: {
				ticks: { color: CHART_COLORS.muted, font: { size: 11 } },
				grid: { display: false, drawBorder: false },
			},
		},
		plugins: {
			legend: { display: false },
			tooltip: {
				backgroundColor: "#161b22",
				titleColor: CHART_COLORS.text,
				bodyColor: CHART_COLORS.text,
				borderColor: "#30363d",
				borderWidth: 1,
				callbacks: tooltipLabel ? { label: tooltipLabel } : undefined,
			},
		},
	};
}

// Each bar row needs vertical space; chart-host height scales with count
// so 14-row charts don't cram rows together at a default 360px canvas.
function sizeChartHost(canvas, rowCount) {
	const host = canvas.parentElement;
	if (!host) return;
	const rowHeight = 30;
	const padding = 60;
	host.style.height = `${Math.max(180, rowCount * rowHeight + padding)}px`;
}

function renderChart() {
	const canvas = document.getElementById("chart");
	const empty = document.getElementById("chart-empty");
	if (!canvas) return;

	const runs = Array.from(state.runsByRunId.values());
	if (runs.length === 0) {
		if (empty) empty.hidden = false;
		canvas.hidden = true;
		if (speedChartInstance) {
			speedChartInstance.destroy();
			speedChartInstance = null;
		}
		return;
	}
	if (empty) empty.hidden = true;
	canvas.hidden = false;

	// Latest per (profile || model).
	const latestByKey = new Map();
	for (const run of runs) {
		const key = run.profile ?? run.model;
		const prev = latestByKey.get(key);
		if (!prev || prev.timestamp < run.timestamp) latestByKey.set(key, run);
	}
	const rows = Array.from(latestByKey.entries()).map(([label, run]) => ({
		label,
		oneShot: run.oneShot?.tokensPerSecond ?? 0,
		interactive: run.interactive?.tokensPerSecond ?? 0,
	}));
	rows.sort((a, b) => b.oneShot - a.oneShot);

	const labels = rows.map((r) => r.label);
	const oneShotData = rows.map((r) => r.oneShot);
	const interactiveData = rows.map((r) => r.interactive);

	if (!speedChartInstance) {
		speedChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: {
				labels,
				datasets: [
					{
						label: "one-shot",
						data: oneShotData,
						backgroundColor: CHART_COLORS.blue,
						borderRadius: 3,
						barPercentage: 0.45,
						categoryPercentage: 0.9,
					},
					{
						label: "interactive",
						data: interactiveData,
						backgroundColor: CHART_COLORS.purple,
						borderRadius: 3,
						barPercentage: 0.45,
						categoryPercentage: 0.9,
					},
				],
			},
			options: baseChartOptions({
				xTitle: "tokens / sec",
				tooltipLabel: (ctx) =>
					`${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)} tok/s`,
			}),
		});
	} else {
		speedChartInstance.data.labels = labels;
		speedChartInstance.data.datasets[0].data = oneShotData;
		speedChartInstance.data.datasets[1].data = interactiveData;
		speedChartInstance.update();
	}
	sizeChartHost(canvas, labels.length);
}

/**
 * Tool-calling accuracy by profile. Two bars per profile (pass rate vs.
 * mean score), sorted by pass rate, latest eval per (profile || model)
 * wins. They diverge when partial-credit scoring kicks in.
 */
function renderToolChart() {
	const canvas = document.getElementById("tool-chart");
	const host = document.getElementById("tool-host");
	const empty = document.getElementById("tool-empty");
	if (!canvas) return;

	// Collect latest eval per key that has a tool-calling dim.
	const latestByKey = new Map();
	for (const ev of state.evalsByEvalId.values()) {
		const tc = ev.dimensions?.["tool-calling"];
		if (!tc || !tc.total) continue;
		const key = ev.profile ?? ev.modelId;
		const prev = latestByKey.get(key);
		if (!prev || prev.timestamp < ev.timestamp) latestByKey.set(key, ev);
	}

	if (latestByKey.size === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (toolChartInstance) {
			toolChartInstance.destroy();
			toolChartInstance = null;
		}
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	const rows = Array.from(latestByKey.entries()).map(([key, ev]) => {
		const tc = ev.dimensions["tool-calling"];
		const passRate = tc.total > 0 ? tc.passed / tc.total : 0;
		const meanScore = tc.score ?? 0;
		return {
			key,
			passPct: Math.round(passRate * 100),
			scorePct: Math.round(meanScore * 100),
			passed: tc.passed,
			total: tc.total,
		};
	});
	rows.sort((a, b) => b.passPct - a.passPct || b.scorePct - a.scorePct);

	const labels = rows.map((r) => r.key);
	const passData = rows.map((r) => r.passPct);
	const scoreData = rows.map((r) => r.scorePct);
	const tooltipExtras = rows.map((r) => `${r.passed}/${r.total}`);

	if (!toolChartInstance) {
		toolChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: {
				labels,
				datasets: [
					{
						label: "pass rate",
						data: passData,
						backgroundColor: CHART_COLORS.green,
						borderRadius: 3,
						barPercentage: 0.45,
						categoryPercentage: 0.9,
					},
					{
						label: "mean score",
						data: scoreData,
						backgroundColor: CHART_COLORS.yellow,
						borderRadius: 3,
						barPercentage: 0.45,
						categoryPercentage: 0.9,
					},
				],
			},
			options: baseChartOptions({
				xTitle: "% (0–100)",
				tooltipLabel: (ctx) => {
					const extra =
						ctx.dataset.label === "pass rate"
							? ` (${ctx.chart.$toolExtras?.[ctx.dataIndex] ?? ""})`
							: "";
					return `${ctx.dataset.label}: ${ctx.parsed.x}%${extra}`;
				},
			}),
		});
	} else {
		toolChartInstance.data.labels = labels;
		toolChartInstance.data.datasets[0].data = passData;
		toolChartInstance.data.datasets[1].data = scoreData;
		toolChartInstance.update();
	}
	// Stash per-row pass count so the tooltip can show "6/12" alongside
	// the percentage without re-looking-up the source eval.
	toolChartInstance.$toolExtras = tooltipExtras;
	// Force tooltip to see the same x axis (0–100 pct) even if the
	// underlying scales config was inherited from baseChartOptions.
	if (toolChartInstance.options.scales?.x) {
		toolChartInstance.options.scales.x.max = 100;
	}
	sizeChartHost(canvas, labels.length);
}

// ── 13. Accuracy × Speed scatter ────────────────────────────────────

function renderScatterChart() {
	const canvas = document.getElementById("scatter-chart");
	const host = document.getElementById("scatter-host");
	const empty = document.getElementById("scatter-empty");
	if (!canvas) return;

	// Join speed runs (latest per key) with evals (latest per key).
	const runByKey = new Map();
	for (const run of state.runsByRunId.values()) {
		const key = run.profile ?? run.model;
		const prev = runByKey.get(key);
		if (!prev || prev.timestamp < run.timestamp) runByKey.set(key, run);
	}
	const evalByKey = new Map();
	for (const ev of state.evalsByEvalId.values()) {
		const key = ev.profile ?? ev.modelId;
		const prev = evalByKey.get(key);
		if (!prev || prev.timestamp < ev.timestamp) evalByKey.set(key, ev);
	}

	// Group by `(modelId, thinking)` so the legend turns into a colour key
	// that respects thinking mode — same model running thinking-on and
	// thinking-off produces visibly different scores and shouldn't share a
	// colour. Matches the convention used by the temperature-sweep and
	// per-dimension grouped charts. Tooltip still spells out the full
	// profile label and thinking mode for each individual dot.
	const pointsByKey = new Map();
	for (const [key, run] of runByKey) {
		const ev = evalByKey.get(key);
		if (!ev) continue;
		const avgTps = ((run.oneShot?.tokensPerSecond ?? 0) + (run.interactive?.tokensPerSecond ?? 0)) / 2;
		if (avgTps === 0) continue;
		const modelId = ev.modelId ?? run.model ?? key;
		const thinking = ev.thinking === "on" ? "on" : "off";
		const groupKey = `${modelId}::${thinking}`;
		if (!pointsByKey.has(groupKey)) {
			pointsByKey.set(groupKey, { modelId, thinking, points: [] });
		}
		pointsByKey.get(groupKey).points.push({
			x: avgTps,
			y: Math.round((ev.overall ?? 0) * 100),
			label: key,
			think: thinking,
		});
	}

	if (pointsByKey.size === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (scatterChartInstance) { scatterChartInstance.destroy(); scatterChartInstance = null; }
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	const palette = [
		CHART_COLORS.blue,
		CHART_COLORS.purple,
		CHART_COLORS.green,
		CHART_COLORS.yellow,
		CHART_COLORS.orange ?? "#fb923c",
		"#f472b6",
		"#22d3ee",
		"#a78bfa",
	];
	// Sort keys for stable colour assignment across renders. thinking-off
	// keeps the bare model id as its dataset label so non-thinking-capable
	// models read the same as before.
	const sortedKeys = Array.from(pointsByKey.keys()).sort();
	const datasets = sortedKeys.map((groupKey, i) => {
		const { modelId, thinking, points } = pointsByKey.get(groupKey);
		const label = thinking === "on" ? `${modelId} (think)` : modelId;
		return {
			label,
			data: points,
			backgroundColor: palette[i % palette.length],
			borderColor: palette[i % palette.length],
			pointRadius: 6,
			pointHoverRadius: 8,
		};
	});
	const data = { datasets };

	const options = {
		responsive: true,
		maintainAspectRatio: false,
		animation: { duration: 400 },
		scales: {
			x: {
				title: { display: true, text: "avg tokens/sec", color: CHART_COLORS.muted },
				ticks: { color: CHART_COLORS.muted },
				grid: { color: CHART_COLORS.grid },
			},
			y: {
				title: { display: true, text: "overall accuracy %", color: CHART_COLORS.muted },
				min: 0, max: 100,
				ticks: { color: CHART_COLORS.muted },
				grid: { color: CHART_COLORS.grid },
			},
		},
		plugins: {
			legend: {
				display: true,
				position: "top",
				labels: {
					color: CHART_COLORS.muted,
					boxWidth: 10,
					font: { size: 11 },
				},
			},
			tooltip: {
				backgroundColor: "#161b22",
				titleColor: CHART_COLORS.text,
				bodyColor: CHART_COLORS.text,
				borderColor: "#30363d",
				borderWidth: 1,
				callbacks: {
					label: (ctx) => {
						const p = ctx.raw;
						return `${p.label}: ${p.x.toFixed(1)} tok/s · ${p.y}% accuracy (think ${p.think})`;
					},
				},
			},
		},
	};

	if (!scatterChartInstance) {
		scatterChartInstance = new Chart(canvas.getContext("2d"), { type: "scatter", data, options });
	} else {
		scatterChartInstance.data = data;
		scatterChartInstance.update();
	}
}

// ── 14. Per-dimension grouped bars ──────────────────────────────────

function renderDimGroupedChart() {
	const canvas = document.getElementById("dim-grouped-chart");
	const host = document.getElementById("dim-grouped-host");
	const empty = document.getElementById("dim-grouped-empty");
	if (!canvas) return;

	// Latest cold eval per (modelId, thinking).
	//
	// `thinking` is part of the key because Qwen-style models share a
	// single modelId across thinking-on and thinking-off runs but score
	// substantively differently. Without this split the row labelled
	// e.g. `qwen3-0.6b-q4f16` would silently render whichever mode
	// happened to land last — making thinking-on / thinking-off
	// indistinguishable in the comparison.
	//
	// Embedding-only evals are excluded entirely: they live in the
	// dedicated "Embeddings" section below (cosine + latency + throughput
	// charts), and including them here would just draw four null bars
	// next to a single embedding bar.
	const latestColdByKey = new Map();
	for (const ev of state.evalsByEvalId.values()) {
		const t = ev.params?.temperature;
		const bucket = tempBucket(t);
		// Use cold profiles only (or unspecified temp treated as cold).
		if (bucket !== null && bucket !== "cold") continue;
		const dims = Object.keys(ev.dimensions ?? {});
		if (dims.length === 1 && dims[0] === "embedding") continue;
		const thinking = ev.thinking === "on" ? "on" : "off";
		const key = `${ev.modelId}::${thinking}`;
		const prev = latestColdByKey.get(key);
		if (!prev || prev.timestamp < ev.timestamp) latestColdByKey.set(key, ev);
	}

	if (latestColdByKey.size === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (dimGroupedChartInstance) { dimGroupedChartInstance.destroy(); dimGroupedChartInstance = null; }
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	// Drop the "embedding" dimension here — it has its own dedicated
	// section below. Keeping it in this chart would just draw an empty
	// fifth column for every generative model.
	const dimNames = DIM_NAMES.filter((d) => d !== "embedding");
	const dimColors = [
		CHART_COLORS.green,
		CHART_COLORS.blue,
		CHART_COLORS.purple,
		CHART_COLORS.yellow,
	];
	const keys = Array.from(latestColdByKey.keys());
	// thinking-off keeps the bare model id as its label so non-thinking
	// models (Llama, TinyLlama, arctic-embed-…) read the same as before.
	const labels = keys.map((k) => {
		const ev = latestColdByKey.get(k);
		return ev?.thinking === "on"
			? `${ev.modelId} (think)`
			: (ev?.modelId ?? k);
	});

	const datasets = dimNames.map((dim, i) => ({
		label: dim,
		// Return null (not 0) for dimensions a model wasn't actually scored
		// on — embedding-only models would otherwise render as four "0%"
		// bars on tool-calling/reasoning/instruction-following/semantic-
		// reasoning, falsely implying they did poorly there.
		data: keys.map((k) => {
			const ev = latestColdByKey.get(k);
			const ds = ev?.dimensions?.[dim];
			return ds && (ds.total ?? 0) > 0
				? Math.round((ds.score ?? 0) * 100)
				: null;
		}),
		backgroundColor: dimColors[i],
		borderRadius: 3,
		barPercentage: 0.7,
		categoryPercentage: 0.85,
	}));

	if (!dimGroupedChartInstance) {
		dimGroupedChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: { labels, datasets },
			options: {
				...baseChartOptions({ xTitle: "% score" }),
				plugins: {
					legend: {
						display: true,
						position: "top",
						labels: { color: CHART_COLORS.muted, boxWidth: 10, font: { size: 11 } },
					},
					tooltip: {
						backgroundColor: "#161b22",
						titleColor: CHART_COLORS.text,
						bodyColor: CHART_COLORS.text,
						borderColor: "#30363d",
						borderWidth: 1,
					},
				},
			},
		});
	} else {
		dimGroupedChartInstance.data.labels = labels;
		dimGroupedChartInstance.data.datasets = datasets;
		dimGroupedChartInstance.update();
	}
	sizeChartHost(canvas, labels.length);
}

let embeddingCosineChartInstance = null;

function renderEmbeddingCosineChart() {
	const canvas = document.getElementById("embedding-cosine-chart");
	const host = document.getElementById("embedding-cosine-host");
	const empty = document.getElementById("embedding-cosine-empty");
	if (!canvas) return;

	const evals = Array.from(state.evalsByEvalId.values());
	const data = buildEmbeddingCosineChartData(evals);

	if (data.labels.length === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (embeddingCosineChartInstance) {
			embeddingCosineChartInstance.destroy();
			embeddingCosineChartInstance = null;
		}
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	const palette = [
		CHART_COLORS.blue,
		CHART_COLORS.green,
		CHART_COLORS.purple,
		CHART_COLORS.yellow,
		CHART_COLORS.orange ?? "#fb923c",
		"#f472b6",
	];
	const datasets = data.datasets.map((ds, i) => ({
		...ds,
		backgroundColor: palette[i % palette.length],
		borderRadius: 3,
		barPercentage: 0.7,
		categoryPercentage: 0.85,
	}));

	if (!embeddingCosineChartInstance) {
		embeddingCosineChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: { labels: data.labels, datasets },
			options: {
				...baseChartOptions({ xTitle: "cosine" }),
				plugins: {
					legend: {
						display: true,
						position: "top",
						labels: { color: CHART_COLORS.muted, boxWidth: 10, font: { size: 11 } },
					},
					tooltip: {
						backgroundColor: "#161b22",
						titleColor: CHART_COLORS.text,
						bodyColor: CHART_COLORS.text,
						borderColor: "#30363d",
						borderWidth: 1,
						callbacks: {
							label: (ctx) =>
								`${ctx.dataset.label}: ${ctx.parsed.x?.toFixed?.(3) ?? ctx.parsed.y?.toFixed?.(3) ?? ctx.formattedValue}`,
						},
					},
				},
			},
		});
	} else {
		embeddingCosineChartInstance.data.labels = data.labels;
		embeddingCosineChartInstance.data.datasets = datasets;
		embeddingCosineChartInstance.update();
	}
	sizeChartHost(canvas, data.labels.length);
}

let embeddingLatencyChartInstance = null;
let embeddingThroughputChartInstance = null;

function renderEmbeddingLatencyChart() {
	renderEmbeddingPerfChart({
		canvasId: "embedding-latency-chart",
		hostId: "embedding-latency-host",
		emptyId: "embedding-latency-empty",
		instanceRef: () => embeddingLatencyChartInstance,
		setInstance: (v) => {
			embeddingLatencyChartInstance = v;
		},
		buildData: buildEmbeddingLatencyChartData,
		xTitle: "median ms / text",
		color: CHART_COLORS.purple,
	});
}

function renderEmbeddingThroughputChart() {
	renderEmbeddingPerfChart({
		canvasId: "embedding-throughput-chart",
		hostId: "embedding-throughput-host",
		emptyId: "embedding-throughput-empty",
		instanceRef: () => embeddingThroughputChartInstance,
		setInstance: (v) => {
			embeddingThroughputChartInstance = v;
		},
		buildData: buildEmbeddingThroughputChartData,
		xTitle: "texts / sec",
		color: CHART_COLORS.green,
	});
}

function renderEmbeddingPerfChart({
	canvasId,
	hostId,
	emptyId,
	instanceRef,
	setInstance,
	buildData,
	xTitle,
	color,
}) {
	const canvas = document.getElementById(canvasId);
	const host = document.getElementById(hostId);
	const empty = document.getElementById(emptyId);
	if (!canvas) return;

	const evals = Array.from(state.evalsByEvalId.values());
	const data = buildData(evals);

	if (data.labels.length === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		const inst = instanceRef();
		if (inst) {
			inst.destroy();
			setInstance(null);
		}
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	const datasets = data.datasets.map((ds) => ({
		...ds,
		backgroundColor: color,
		borderRadius: 3,
		barPercentage: 0.6,
		categoryPercentage: 0.85,
	}));

	let inst = instanceRef();
	if (!inst) {
		inst = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: { labels: data.labels, datasets },
			options: baseChartOptions({
				xTitle,
				tooltipLabel: (ctx) =>
					`${ctx.dataset.label}: ${typeof ctx.parsed.x === "number" ? ctx.parsed.x.toFixed(2) : ctx.formattedValue}`,
			}),
		});
		setInstance(inst);
	} else {
		inst.data.labels = data.labels;
		inst.data.datasets = datasets;
		inst.update();
	}
	sizeChartHost(canvas, data.labels.length);
}

function renderTempSweepChart() {
	const canvas = document.getElementById("temp-sweep-chart");
	const host = document.getElementById("temp-sweep-host");
	const empty = document.getElementById("temp-sweep-empty");
	if (!canvas) return;

	const data = buildTempSweepChartData(state.evalsByEvalId.values());
	if (data.labels.length === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (tempSweepChartInstance) { tempSweepChartInstance.destroy(); tempSweepChartInstance = null; }
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	if (!tempSweepChartInstance) {
		tempSweepChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data,
			options: {
				...baseChartOptions({ xTitle: "% score" }),
				plugins: {
					legend: {
						display: true,
						position: "top",
						labels: { color: CHART_COLORS.muted, boxWidth: 10, font: { size: 11 } },
					},
					tooltip: {
						backgroundColor: "#161b22",
						titleColor: CHART_COLORS.text,
						bodyColor: CHART_COLORS.text,
						borderColor: "#30363d",
						borderWidth: 1,
					},
				},
			},
		});
	} else {
		tempSweepChartInstance.data.labels = data.labels;
		tempSweepChartInstance.data.datasets = data.datasets;
		tempSweepChartInstance.update();
	}
	sizeChartHost(canvas, data.labels.length);
}

// ── 16. Thinking ON vs OFF delta (Qwen) ─────────────────────────────

function renderThinkingDeltaChart() {
	const canvas = document.getElementById("thinking-delta-chart");
	const host = document.getElementById("thinking-delta-host");
	const empty = document.getElementById("thinking-delta-empty");
	if (!canvas) return;

	// Group Qwen evals by (modelId, temperature) and split thinking on/off.
	const pairs = new Map();
	for (const ev of state.evalsByEvalId.values()) {
		if (!ev.modelId.toLowerCase().includes("qwen")) continue;
		const t = ev.params?.temperature ?? 0;
		const roundedT = Math.round(t * 10) / 10;
		const thinkKey = ev.thinking === "on" ? "on" : "off";
		const k = `${ev.modelId}::${roundedT}`;
		if (!pairs.has(k)) pairs.set(k, { model: ev.modelId, temp: roundedT, dims: {} });
		const pair = pairs.get(k);
		for (const [dim, ds] of Object.entries(ev.dimensions ?? {})) {
			if (!pair.dims[dim]) pair.dims[dim] = {};
			const prev = pair.dims[dim][thinkKey];
			if (!prev || prev.ts < ev.timestamp) {
				pair.dims[dim][thinkKey] = { score: Math.round((ds.score ?? 0) * 100), ts: ev.timestamp };
			}
		}
	}

	// Only include pairs that have both on and off.
	const matched = Array.from(pairs.values()).filter((p) => {
		return Object.values(p.dims).some((d) => d.on && d.off);
	});
	if (matched.length === 0) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (thinkingDeltaChartInstance) { thinkingDeltaChartInstance.destroy(); thinkingDeltaChartInstance = null; }
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	// Flatten to one row per (model, temp, dimension).
	const rows = [];
	for (const p of matched) {
		for (const [dim, d] of Object.entries(p.dims)) {
			if (d.on && d.off) rows.push({ label: `${p.model} · t=${p.temp} · ${dim}`, off: d.off.score, on: d.on.score });
		}
	}
	rows.sort((a, b) => b.on - a.on);

	const labels = rows.map((r) => r.label);
	const datasets = [
		{
			label: "thinking OFF",
			data: rows.map((r) => r.off),
			backgroundColor: CHART_COLORS.blue,
			borderRadius: 3,
			barPercentage: 0.45,
			categoryPercentage: 0.9,
		},
		{
			label: "thinking ON",
			data: rows.map((r) => r.on),
			backgroundColor: CHART_COLORS.purple,
			borderRadius: 3,
			barPercentage: 0.45,
			categoryPercentage: 0.9,
		},
	];

	if (!thinkingDeltaChartInstance) {
		thinkingDeltaChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: { labels, datasets },
			options: {
				...baseChartOptions({ xTitle: "% score" }),
				plugins: {
					legend: {
						display: true,
						position: "top",
						labels: { color: CHART_COLORS.muted, boxWidth: 10, font: { size: 11 } },
					},
					tooltip: {
						backgroundColor: "#161b22",
						titleColor: CHART_COLORS.text,
						bodyColor: CHART_COLORS.text,
						borderColor: "#30363d",
						borderWidth: 1,
					},
				},
			},
		});
	} else {
		thinkingDeltaChartInstance.data.labels = labels;
		thinkingDeltaChartInstance.data.datasets = datasets;
		thinkingDeltaChartInstance.update();
	}
	sizeChartHost(canvas, labels.length);
}

// ── 17. Time-to-first-token (prefill) ───────────────────────────────

function renderTtftChart() {
	const canvas = document.getElementById("ttft-chart");
	const empty = document.getElementById("ttft-empty");
	if (!canvas) return;

	const runs = Array.from(state.runsByRunId.values());
	const latestByKey = new Map();
	for (const run of runs) {
		const key = run.profile ?? run.model;
		const prev = latestByKey.get(key);
		if (!prev || prev.timestamp < run.timestamp) latestByKey.set(key, run);
	}
	const rows = Array.from(latestByKey.entries())
		.map(([label, run]) => ({ label, ms: run.oneShot?.prefillMs }))
		.filter((r) => typeof r.ms === "number" && r.ms > 0);

	if (rows.length === 0) {
		if (empty) empty.hidden = false;
		canvas.hidden = true;
		if (ttftChartInstance) { ttftChartInstance.destroy(); ttftChartInstance = null; }
		return;
	}
	if (empty) empty.hidden = true;
	canvas.hidden = false;

	rows.sort((a, b) => a.ms - b.ms);
	const labels = rows.map((r) => r.label);
	const data = rows.map((r) => r.ms);

	if (!ttftChartInstance) {
		ttftChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: {
				labels,
				datasets: [{
					label: "prefill ms",
					data,
					backgroundColor: CHART_COLORS.blue,
					borderRadius: 3,
					barPercentage: 0.6,
					categoryPercentage: 0.9,
				}],
			},
			options: baseChartOptions({
				xTitle: "ms",
				tooltipLabel: (ctx) => `${ctx.parsed.x.toFixed(0)} ms prefill`,
			}),
		});
	} else {
		ttftChartInstance.data.labels = labels;
		ttftChartInstance.data.datasets[0].data = data;
		ttftChartInstance.update();
	}
	sizeChartHost(canvas, labels.length);
}

// ── 18. Finish reason breakdown ─────────────────────────────────────

function renderFinishChart() {
	const canvas = document.getElementById("finish-chart");
	const empty = document.getElementById("finish-empty");
	if (!canvas) return;

	const runs = Array.from(state.runsByRunId.values());
	const latestByKey = new Map();
	for (const run of runs) {
		const key = run.profile ?? run.model;
		const prev = latestByKey.get(key);
		if (!prev || prev.timestamp < run.timestamp) latestByKey.set(key, run);
	}

	// For each profile, count finish reasons across all runs (not just latest).
	const profileReasons = new Map();
	for (const run of runs) {
		const key = run.profile ?? run.model;
		if (!profileReasons.has(key)) profileReasons.set(key, {});
		const reasons = profileReasons.get(key);
		const reason = run.oneShot?.finishReason ?? "unknown";
		reasons[reason] = (reasons[reason] ?? 0) + 1;
	}

	const rows = Array.from(profileReasons.entries())
		.filter(([, reasons]) => Object.keys(reasons).length > 0);

	if (rows.length === 0) {
		if (empty) empty.hidden = false;
		canvas.hidden = true;
		if (finishChartInstance) { finishChartInstance.destroy(); finishChartInstance = null; }
		return;
	}
	if (empty) empty.hidden = true;
	canvas.hidden = false;

	// Collect all unique reason labels.
	const allReasons = new Set();
	for (const [, reasons] of rows) {
		for (const r of Object.keys(reasons)) allReasons.add(r);
	}
	const reasonList = Array.from(allReasons).sort();

	const reasonColors = {
		eos: CHART_COLORS.green,
		"stop-token": CHART_COLORS.blue,
		"max-tokens": CHART_COLORS.yellow,
		error: CHART_COLORS.red,
		unknown: "#6e7681",
	};

	const labels = rows.map(([key]) => key);
	const datasets = reasonList.map((reason) => ({
		label: reason,
		data: rows.map(([, reasons]) => reasons[reason] ?? 0),
		backgroundColor: reasonColors[reason] ?? CHART_COLORS.muted,
		borderRadius: 2,
		barPercentage: 0.9,
	}));

	if (!finishChartInstance) {
		finishChartInstance = new Chart(canvas.getContext("2d"), {
			type: "bar",
			data: { labels, datasets },
			options: {
				...baseChartOptions({ xTitle: "count" }),
				indexAxis: "y",
				scales: {
					x: {
						stacked: true,
						beginAtZero: true,
						ticks: { color: CHART_COLORS.muted, stepSize: 1 },
						grid: { color: CHART_COLORS.grid },
						title: { display: true, text: "count", color: CHART_COLORS.muted },
					},
					y: {
						stacked: true,
						ticks: { color: CHART_COLORS.muted },
						grid: { display: false },
					},
				},
				plugins: {
					legend: {
						display: true,
						position: "top",
						labels: { color: CHART_COLORS.muted, boxWidth: 10, font: { size: 11 } },
					},
					tooltip: {
						backgroundColor: "#161b22",
						titleColor: CHART_COLORS.text,
						bodyColor: CHART_COLORS.text,
						borderColor: "#30363d",
						borderWidth: 1,
					},
				},
			},
		});
	} else {
		finishChartInstance.data.labels = labels;
		finishChartInstance.data.datasets = datasets;
		finishChartInstance.update();
	}
	sizeChartHost(canvas, labels.length);
}

// ── 19. Score over time (regression detection) ──────────────────────

let seriesLoaded = false;

function renderSeriesChart() {
	const canvas = document.getElementById("series-chart");
	const host = document.getElementById("series-host");
	const empty = document.getElementById("series-empty");
	if (!canvas) return;

	if (!seriesLoaded) {
		seriesLoaded = true;
		fetch("/evals/series")
			.then((res) => res.ok ? res.json() : { series: [] })
			.then((json) => {
				state.evalSeries = json.series ?? [];
				renderSeriesChartImpl(canvas, host, empty);
			})
			.catch(() => {
				state.evalSeries = [];
				renderSeriesChartImpl(canvas, host, empty);
			});
	} else {
		renderSeriesChartImpl(canvas, host, empty);
	}
}

function renderSeriesChartImpl(canvas, host, empty) {
	const series = state.evalSeries ?? [];
	if (series.length < 2) {
		if (host) host.hidden = true;
		if (empty) empty.hidden = false;
		if (seriesChartInstance) { seriesChartInstance.destroy(); seriesChartInstance = null; }
		return;
	}
	if (host) host.hidden = false;
	if (empty) empty.hidden = true;

	// Group by (profile || modelId). For the category x-axis to plot the
	// points correctly we need (a) a labels array shared by every dataset
	// and (b) each dataset's `data[i]` aligned to `labels[i]` (null for a
	// profile that has no eval at that timestamp). Without this Chart.js
	// has no way to position the points and the canvas renders blank.
	const byProfile = new Map();
	for (const pt of series) {
		const key = pt.profile ?? pt.modelId;
		if (!byProfile.has(key)) byProfile.set(key, new Map());
		byProfile.get(key).set(pt.timestamp, Math.round(pt.overall * 100));
	}

	const labels = Array.from(
		new Set(series.map((pt) => pt.timestamp)),
	).sort();
	// Render compact human-readable labels (MM-DD HH:mm) for the axis but
	// keep the full ISO string available for the tooltip via parsing.
	const fmt = (iso) => {
		const d = new Date(iso);
		if (Number.isNaN(d.getTime())) return iso;
		const pad = (n) => String(n).padStart(2, "0");
		return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
	};
	const tickLabels = labels.map(fmt);

	const palette = [
		CHART_COLORS.blue, CHART_COLORS.purple, CHART_COLORS.green,
		CHART_COLORS.yellow, "#f778ba", "#79c0ff", "#56d364",
	];
	const profiles = Array.from(byProfile.keys()).sort();

	const datasets = profiles.map((profile, i) => {
		const pointsByTs = byProfile.get(profile);
		return {
			label: profile,
			data: labels.map((ts) => pointsByTs.get(ts) ?? null),
			borderColor: palette[i % palette.length],
			backgroundColor: "transparent",
			pointRadius: 3,
			pointHoverRadius: 5,
			borderWidth: 2,
			tension: 0.15,
			// Don't break a profile's line when it skips a timestamp another
			// profile produced — connect across the null gap.
			spanGaps: true,
		};
	});

	if (!seriesChartInstance) {
		seriesChartInstance = new Chart(canvas.getContext("2d"), {
			type: "line",
			data: { labels: tickLabels, datasets },
			options: {
				responsive: true,
				maintainAspectRatio: false,
				animation: { duration: 400 },
				scales: {
					x: {
						type: "category",
						ticks: { color: CHART_COLORS.muted, maxTicksLimit: 8, font: { size: 10 } },
						grid: { color: CHART_COLORS.grid },
					},
					y: {
						min: 0, max: 100,
						ticks: { color: CHART_COLORS.muted },
						grid: { color: CHART_COLORS.grid },
						title: { display: true, text: "overall %", color: CHART_COLORS.muted },
					},
				},
				plugins: {
					legend: {
						display: true,
						position: "top",
						labels: { color: CHART_COLORS.muted, boxWidth: 10, font: { size: 11 } },
					},
					tooltip: {
						backgroundColor: "#161b22",
						titleColor: CHART_COLORS.text,
						bodyColor: CHART_COLORS.text,
						borderColor: "#30363d",
						borderWidth: 1,
					},
				},
			},
		});
	} else {
		seriesChartInstance.data.labels = tickLabels;
		seriesChartInstance.data.datasets = datasets;
		seriesChartInstance.update();
	}
	sizeChartHost(canvas, profiles.length * 2);
}

function passesFilter(run) {
	if (state.thinkingFilter !== "all" && run.thinking !== state.thinkingFilter) {
		return false;
	}
	if (state.textFilter) {
		const needle = state.textFilter.toLowerCase();
		const haystack = `${run.profile ?? ""} ${run.model}`.toLowerCase();
		if (!haystack.includes(needle)) return false;
	}
	return true;
}

function comparator(a, b) {
	const k = state.sortKey;
	const dir = state.sortDir === "asc" ? 1 : -1;
	const av = valueFor(a, k);
	const bv = valueFor(b, k);
	if (av === bv) return 0;
	if (av === undefined || av === null) return 1;
	if (bv === undefined || bv === null) return -1;
	return av < bv ? -1 * dir : 1 * dir;
}

function valueFor(run, key) {
	switch (key) {
		case "timestamp": return run.timestamp;
		case "profile": return run.profile ?? "";
		case "model": return run.model;
		case "thinking": return run.thinking;
		case "oneShotTokensPerSec": return run.oneShot?.tokensPerSecond ?? 0;
		case "oneShotPrefillMs": return run.oneShot?.prefillMs ?? 0;
		case "oneShotTotalMs": return run.oneShot?.totalMs ?? 0;
		case "oneShotFinishReason": return run.oneShot?.finishReason ?? "";
		case "oneShotGenTokens": return run.oneShot?.genTokens ?? 0;
		default: return "";
	}
}

function renderTable() {
	const tbody = document.getElementById("runs-tbody");
	const countEl = document.getElementById("run-count");
	const runs = Array.from(state.runsByRunId.values()).filter(passesFilter);
	runs.sort(comparator);
	countEl.textContent = String(runs.length);
	tbody.innerHTML = "";
	for (const run of runs) {
		const tr = document.createElement("tr");
		tr.dataset.runId = run.runId;
		if (state.selected.has(run.runId)) tr.classList.add("selected");
		if (state.freshRunIds.has(run.runId)) tr.classList.add("fresh");
		const thinkPill = run.thinking === "on" ? "think-on" : "";
		const profileCell = run.profile
			? escapeHtml(run.profile)
			: `<span class="dim">—</span>`;
		tr.innerHTML = `
			<td class="col-pick"><input type="checkbox" ${state.selected.has(run.runId) ? "checked" : ""} data-pick="${run.runId}"></td>
			<td>${formatTime(run.timestamp)}</td>
			<td>${profileCell}</td>
			<td>${escapeHtml(run.model)}</td>
			<td><span class="pill ${thinkPill}">${run.thinking}</span></td>
			<td class="num">${formatNum(run.oneShot?.tokensPerSecond, 1)}</td>
			<td class="num">${formatNum(run.oneShot?.prefillMs, 0)}</td>
			<td class="num">${formatNum(run.oneShot?.totalMs, 0)}</td>
			<td>${escapeHtml(run.oneShot?.finishReason ?? "—")}</td>
			<td class="num">${run.oneShot?.genTokens ?? "—"}</td>
			<td>${systemPill(run.systemId)}</td>
		`;
		tr.addEventListener("click", (e) => {
			if (e.target.dataset?.pick) return;
			openDetail([run]);
		});
		tbody.appendChild(tr);
	}

	// failed rows: show at bottom with red stripe
	for (const failed of state.failedByRunId.values()) {
		const tr = document.createElement("tr");
		tr.className = "failed";
		tr.innerHTML = `
			<td class="col-pick"></td>
			<td>${formatTime(new Date().toISOString())}</td>
			<td>${escapeHtml(failed.profile ?? "—")}</td>
			<td>${escapeHtml(failed.model)}</td>
			<td></td>
			<td colspan="5"><span class="pill fail">FAILED</span> ${escapeHtml(failed.error)}</td>
		`;
		tbody.appendChild(tr);
	}
	updateSortHeaders();
}

function updateSortHeaders() {
	document.querySelectorAll("thead th.sortable").forEach((th) => {
		th.classList.remove("sort-asc", "sort-desc");
		if (th.dataset.sort === state.sortKey) {
			th.classList.add(state.sortDir === "asc" ? "sort-asc" : "sort-desc");
		}
	});
}

function renderCompareButton() {
	const btn = document.getElementById("compare-btn");
	btn.textContent = `Compare selected (${state.selected.size})`;
	btn.disabled = state.selected.size < 2;
}

// ── evals ────────────────────────────────────────────────────────

function renderEvals() {
	const evals = Array.from(state.evalsByEvalId.values());
	const running = Array.from(state.runningEvalsByEvalId.values());
	const countEl = document.getElementById("eval-count");
	const emptyEl = document.getElementById("evals-empty");
	const bodyEl = document.getElementById("evals-body");
	const progressWrap = document.getElementById("eval-progress-wrap");
	const progressLabel = document.getElementById("eval-progress-label");

	countEl.textContent = String(evals.length);

	if (running.length > 0) {
		const r = running[0];
		progressWrap.hidden = false;
		progressLabel.textContent = `${r.modelId} · ${r.label ?? "all"} · ${r.completedTasks}/${r.totalTasks} tasks (${r.passedTasks} passing)`;
	} else {
		progressWrap.hidden = true;
	}

	if (evals.length === 0 && running.length === 0) {
		emptyEl.hidden = false;
		bodyEl.hidden = true;
		return;
	}
	emptyEl.hidden = true;
	bodyEl.hidden = false;

	renderEvalDimensions(evals);
	renderEvalsTable(evals);
}

function renderEvalDimensions(evals) {
	const host = document.getElementById("evals-dimensions");
	// Latest per model for dimension aggregation.
	const latestByModel = new Map();
	for (const rep of evals) {
		const prev = latestByModel.get(rep.modelId);
		if (!prev || prev.timestamp < rep.timestamp) latestByModel.set(rep.modelId, rep);
	}
	if (latestByModel.size === 0) {
		host.innerHTML = `<div class="bar-empty">no completed evals yet</div>`;
		state.renderedDimModels.clear();
		return;
	}

	// Incremental update: create cards for new models (animate bars),
	// refresh existing cards in place (no bar re-animation), remove obsolete.
	const existing = new Map();
	for (const el of host.querySelectorAll(".dim-card[data-model]")) {
		existing.set(el.dataset.model, el);
	}
	// Drop empty-state placeholder if present.
	const empty = host.querySelector(".bar-empty");
	if (empty) empty.remove();

	const newlyAdded = [];
	for (const [model, rep] of latestByModel) {
		const overallPct = Math.round((rep.overall ?? 0) * 100);
		const thinkLabel = rep.thinking === "on" ? "ON" : "OFF";
		const thinkClass = rep.thinking === "on" ? "think-on" : "off";
		const dimEntries = Object.entries(rep.dimensions ?? {});
		const isNew = !existing.has(model);

		// Freshly-added cards get width:0 + data-width so animateBars() can
		// transition them to target. Existing cards get their target width
		// set directly so they do not replay the 0→value animation.
		const dimRows = dimEntries
			.map(([dim, score]) => {
				const pct = Math.round((score.score ?? 0) * 100);
				const widthAttr = isNew
					? `style="width:0%" data-width="${pct}"`
					: `style="width:${pct}%"`;
				return `
					<div class="dim-row">
						<div class="dim-label">${escapeHtml(dim)}</div>
						<div class="dim-bar"><div class="dim-fill" ${widthAttr}></div></div>
						<div class="dim-value" title="${pct}% = mean score across all ${score.total} tasks (partial-credit scorers allowed). ${score.passed}/${score.total} = tasks scoring ≥ 0.5.">${pct}% <span class="dim-sub">(${score.passed}/${score.total})</span></div>
					</div>
				`;
			})
			.join("");
		const headHTML = `
			<div class="dim-card-head">
				<strong>${escapeHtml(model)}</strong>
				<span class="pill ${thinkClass}" title="thinking mode for the latest run">think ${thinkLabel}</span>
				<span class="overall-pill" data-strength="${overallStrength(rep.overall)}">${overallPct}%</span>
			</div>
			${dimRows}
		`;

		let card = existing.get(model);
		if (card) {
			card.innerHTML = headHTML;
			existing.delete(model);
		} else {
			card = document.createElement("div");
			card.className = "dim-card fresh";
			card.dataset.model = model;
			card.innerHTML = headHTML;
			host.appendChild(card);
			newlyAdded.push(card);
			state.renderedDimModels.add(model);
		}
	}
	// Remove cards for models that no longer appear.
	for (const [model, el] of existing) {
		el.remove();
		state.renderedDimModels.delete(model);
	}

	// Animate only the bars inside freshly-added cards.
	for (const card of newlyAdded) {
		animateBars(card);
		// Clear the fade-in class after the flash animation ends so a later
		// re-layout does not re-trigger it.
		setTimeout(() => card.classList.remove("fresh"), 1800);
	}
}

function overallStrength(overall) {
	if (overall >= 0.85) return "strong";
	if (overall >= 0.6) return "medium";
	return "weak";
}

function renderEvalsTable(evals) {
	const tbody = document.getElementById("evals-tbody");
	const sorted = [...evals].sort(evalComparator);
	tbody.innerHTML = "";
	for (const rep of sorted) {
		const tr = document.createElement("tr");
		tr.dataset.evalId = rep.evalId;
		if (state.freshEvalIds.has(rep.evalId)) tr.classList.add("fresh");
		const dimensionChips = Object.entries(rep.dimensions ?? {})
			.map(([dim, s]) => {
				const pct = Math.round((s.score ?? 0) * 100);
				return `<span class="dim-chip" data-strength="${overallStrength(s.score)}" title="${escapeHtml(dim)}: ${s.passed}/${s.total} tasks scored ≥ 0.5; ${pct}% is the mean score across all ${s.total} (partial-credit scorers allowed).">${escapeHtml(dim)}: ${s.passed}/${s.total} · ${pct}%</span>`;
			})
			.join(" ");
		const thinkLabel = rep.thinking === "on" ? "on" : "off";
		const thinkClass = rep.thinking === "on" ? "think-on" : "";
		const profileCell = rep.profile
			? escapeHtml(rep.profile)
			: `<span class="dim">—</span>`;
		const tempCell = renderTempCell(rep);
		tr.innerHTML = `
			<td>${formatTime(rep.timestamp)}</td>
			<td>${escapeHtml(rep.modelId)}</td>
			<td>${profileCell}</td>
			<td><span class="pill ${thinkClass}">${thinkLabel}</span></td>
			<td>${tempCell}</td>
			<td class="num">${rep.totalTasks}</td>
			<td class="num"><strong>${Math.round((rep.overall ?? 0) * 100)}%</strong></td>
			<td>${dimensionChips}</td>
			<td>${systemPill(rep.systemId)}</td>
		`;
		tr.addEventListener("click", () => openEvalDetail(rep));
		tbody.appendChild(tr);
	}
	updateEvalSortHeaders();
}

function evalValueFor(rep, key) {
	if (key === "temperature") return rep.params?.temperature;
	return rep[key];
}

function evalComparator(a, b) {
	const k = state.evalSortKey;
	const dir = state.evalSortDir === "asc" ? 1 : -1;
	const av = evalValueFor(a, k);
	const bv = evalValueFor(b, k);
	if (av === bv) return 0;
	if (av === undefined || av === null) return 1;
	if (bv === undefined || bv === null) return -1;
	return av < bv ? -1 * dir : 1 * dir;
}

function renderTempCell(rep) {
	const t = rep.params?.temperature;
	const bucket = tempBucket(t);
	if (!bucket) return `<span class="dim">—</span>`;
	return `<span class="pill temp-${bucket}" title="temperature = ${t}">${bucket} · ${t}</span>`;
}

function updateEvalSortHeaders() {
	document.querySelectorAll("#evals-table thead th.sortable").forEach((th) => {
		th.classList.remove("sort-asc", "sort-desc");
		if (th.dataset.sort === state.evalSortKey) {
			th.classList.add(
				state.evalSortDir === "asc" ? "sort-asc" : "sort-desc",
			);
		}
	});
}

function openEvalDetail(rep) {
	const backdrop = document.getElementById("modal-backdrop");
	const body = document.getElementById("modal-body");
	body.innerHTML = renderEvalModal(rep);
	backdrop.hidden = false;
}

function renderEvalModal(rep) {
	const dimSummary = Object.entries(rep.dimensions ?? {})
		.map(
			([dim, s]) =>
				`<div class="k">${escapeHtml(dim)}</div><div class="v" title="${s.passed}/${s.total} tasks scored ≥ 0.5. ${Math.round((s.score ?? 0) * 100)}% is the mean score across all ${s.total} tasks (partial-credit scorers allowed).">${s.passed}/${s.total} · ${Math.round((s.score ?? 0) * 100)}% · avg ${Math.round(s.avgLatencyMs ?? 0)}ms</div>`,
		)
		.join("");
	const taskRows = (rep.results ?? [])
		.map((r) => {
			const pct = Math.round((r.score ?? 0) * 100);
			const strength = overallStrength(r.score);
			const toolCallLabel = r.toolCalls?.length
				? `${r.toolCalls.length}× ${escapeHtml(r.toolCalls[0].name)}${r.toolCalls.length > 1 ? " +" + (r.toolCalls.length - 1) : ""}`
				: "—";
			return `
				<tr>
					<td>${escapeHtml(r.taskId)}</td>
					<td><span class="dim-chip">${escapeHtml(r.dimension)}</span></td>
					<td>${escapeHtml(r.difficulty)}</td>
					<td class="num" data-strength="${strength}"><strong>${pct}%</strong></td>
					<td class="num">${Math.round(r.latencyMs)}ms</td>
					<td class="num">${formatNum(r.tokensPerSecond, 1)}</td>
					<td>${toolCallLabel}</td>
					<td class="small">${escapeHtml(r.error ?? "")}</td>
				</tr>`;
		})
		.join("");
	const p = rep.params ?? {};
	const paramRows = [
		["context length", p.contextLength],
		["max tokens", p.maxTokens],
		["temperature", p.temperature],
		["top-k", p.topK],
		["top-p", p.topP],
		["rep. penalty", p.repetitionPenalty],
		["seed", p.seed],
	]
		.filter(([, v]) => v !== undefined && v !== null)
		.map(
			([k, v]) =>
				`<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div>`,
		)
		.join("");

	return `
		<h3>Eval · ${escapeHtml(rep.modelId)}</h3>
		<div class="modal-kv">
			<div class="k">timestamp</div><div class="v">${escapeHtml(rep.timestamp)}</div>
			${rep.profile ? `<div class="k">profile</div><div class="v">${escapeHtml(rep.profile)}</div>` : ""}
			<div class="k">thinking</div><div class="v">${escapeHtml(rep.thinking ?? "off")}</div>
			${paramRows}
			<div class="k">tasks</div><div class="v">${rep.totalTasks}</div>
			<div class="k">overall</div><div class="v">${Math.round((rep.overall ?? 0) * 100)}%</div>
			${dimSummary}
		</div>
		<div class="modal-section">
			<h4>Tasks</h4>
			<div class="table-wrap">
				<table class="eval-detail-table">
					<thead>
						<tr>
							<th>task</th>
							<th>dim</th>
							<th>diff</th>
							<th class="num">score</th>
							<th class="num">latency</th>
							<th class="num">tok/s</th>
							<th>tools</th>
							<th>error</th>
						</tr>
					</thead>
					<tbody>${taskRows}</tbody>
				</table>
			</div>
		</div>
	`;
}

// ── modal ────────────────────────────────────────────────────────

function openDetail(runs) {
	const backdrop = document.getElementById("modal-backdrop");
	const body = document.getElementById("modal-body");
	body.innerHTML = "";
	const cols = document.createElement("div");
	cols.className = "modal-cols";
	cols.style.gridTemplateColumns = `repeat(${runs.length}, minmax(0, 1fr))`;
	for (const run of runs) {
		const col = document.createElement("div");
		col.className = "modal-col";
		col.innerHTML = renderRunDetail(run);
		cols.appendChild(col);
	}
	body.appendChild(cols);
	backdrop.hidden = false;
}

function renderRunDetail(run) {
	const kv = [
		["profile", run.profile ?? "—"],
		["model", run.model],
		["thinking", run.thinking],
		["page", run.page],
		["context", run.params?.contextLength ?? "—"],
		["max tokens", run.params?.maxTokens ?? "—"],
		["temperature", run.params?.temperature ?? "—"],
		["top-k", run.params?.topK ?? "—"],
		["top-p", run.params?.topP ?? "—"],
		["rep. penalty", run.params?.repetitionPenalty ?? "—"],
		["timestamp", run.timestamp],
	];
	const kvRows = kv
		.map(
			([k, v]) =>
				`<div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div>`,
		)
		.join("");

	const oneShot = run.oneShot
		? `
			<div class="modal-section">
				<h4>One-shot — ${escapeHtml(run.oneShot.finishReason ?? "?")} · ${formatNum(run.oneShot.tokensPerSecond, 1)} tok/s · ${run.oneShot.genTokens} tokens · prefill ${formatNum(run.oneShot.prefillMs, 0)}ms / total ${formatNum(run.oneShot.totalMs, 0)}ms</h4>
				<div class="modal-text">${escapeHtml(run.oneShot.assistantText)}</div>
			</div>`
		: "";

	const interactive = run.interactive
		? `
			<div class="modal-section">
				<h4>Interactive — ${escapeHtml(run.interactive.finishReason)}${
					run.interactive.tokensPerSecond
						? ` · ${formatNum(run.interactive.tokensPerSecond, 1)} tok/s · ${run.interactive.genTokens} tokens · ${formatNum(run.interactive.totalMs, 0)}ms`
						: ""
				}</h4>
				<div class="modal-text">${escapeHtml(run.interactive.assistantText)}</div>
			</div>`
		: "";

	const prompt = `
		<div class="modal-section">
			<h4>Prompt</h4>
			<div class="modal-text">${escapeHtml(run.prompt)}</div>
		</div>`;

	return `
		<h3>${escapeHtml(run.profile ?? run.model)}</h3>
		<div class="modal-kv">${kvRows}</div>
		${prompt}
		${oneShot}
		${interactive}
	`;
}

// ── utilities ────────────────────────────────────────────────────

function escapeHtml(v) {
	if (v === null || v === undefined) return "";
	return String(v)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}

function formatTime(iso) {
	try {
		const d = new Date(iso);
		return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
	} catch {
		return iso;
	}
}

function formatNum(v, digits) {
	if (typeof v !== "number" || !Number.isFinite(v)) return "—";
	return v.toFixed(digits);
}

// ── wiring ───────────────────────────────────────────────────────

document.querySelectorAll("#runs-table thead th.sortable").forEach((th) => {
	th.addEventListener("click", () => {
		const key = th.dataset.sort;
		if (state.sortKey === key) {
			state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
		} else {
			state.sortKey = key;
			state.sortDir = key === "timestamp" ? "desc" : "desc";
		}
		renderTable();
	});
});
document.querySelectorAll("#evals-table thead th.sortable").forEach((th) => {
	th.addEventListener("click", () => {
		const key = th.dataset.sort;
		if (state.evalSortKey === key) {
			state.evalSortDir = state.evalSortDir === "asc" ? "desc" : "asc";
		} else {
			state.evalSortKey = key;
			state.evalSortDir = "desc";
		}
		renderEvals();
	});
});

document.getElementById("filter-thinking").addEventListener("change", (e) => {
	state.thinkingFilter = e.target.value;
	renderTable();
});
document.getElementById("filter-text").addEventListener("input", (e) => {
	state.textFilter = e.target.value.trim();
	renderTable();
});

document.getElementById("runs-tbody").addEventListener("change", (e) => {
	const runId = e.target?.dataset?.pick;
	if (!runId) return;
	if (e.target.checked) state.selected.add(runId);
	else state.selected.delete(runId);
	renderTable();
	renderCompareButton();
});

document.getElementById("compare-btn").addEventListener("click", () => {
	const runs = Array.from(state.selected)
		.map((id) => state.runsByRunId.get(id))
		.filter(Boolean);
	if (runs.length >= 2) openDetail(runs);
});

document.getElementById("modal-close").addEventListener("click", () => {
	document.getElementById("modal-backdrop").hidden = true;
});
document.getElementById("modal-backdrop").addEventListener("click", (e) => {
	if (e.target === e.currentTarget) {
		document.getElementById("modal-backdrop").hidden = true;
	}
});
document.addEventListener("keydown", (e) => {
	if (e.key === "Escape") {
		document.getElementById("modal-backdrop").hidden = true;
	}
});

connect();
