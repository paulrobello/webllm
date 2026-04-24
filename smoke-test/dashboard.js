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
	// Evals
	evalsByEvalId: new Map(),
	runningEvalsByEvalId: new Map(),
	evalSortKey: "timestamp",
	evalSortDir: "desc",
	// System profiles
	systemProfilesById: new Map(),
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

// ── rendering ────────────────────────────────────────────────────

function render() {
	renderRunning();
	renderChart();
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
	host.className = "running-list";
	host.innerHTML = "";
	for (const item of speedRuns) {
		const el = document.createElement("div");
		el.className = "running-item";
		const label = item.profile ?? item.model;
		el.innerHTML = `
			<span class="spinner"></span>
			<span class="pill phase">speed</span>
			<strong>${escapeHtml(label)}</strong>
			<span class="dim">${escapeHtml(item.model)}</span>
			<span class="pill ${item.thinking === "on" ? "think-on" : ""}">${item.thinking}</span>
			<span class="dim">${escapeHtml(item.prompt)}</span>
		`;
		host.appendChild(el);
	}
	for (const item of evalRuns) {
		const el = document.createElement("div");
		el.className = "running-item";
		const label = item.label ?? item.modelId;
		const completed = item.completedTasks ?? 0;
		const total = item.totalTasks ?? 0;
		const passed = item.passedTasks ?? 0;
		el.innerHTML = `
			<span class="spinner"></span>
			<span class="pill phase accuracy">accuracy</span>
			<strong>${escapeHtml(label)}</strong>
			<span class="dim">${escapeHtml(item.modelId)}</span>
			<span class="dim">${completed}/${total} tasks (${passed} passing)</span>
		`;
		host.appendChild(el);
	}
}

function renderChart() {
	const host = document.getElementById("chart");
	const runs = Array.from(state.runsByRunId.values());
	if (runs.length === 0) {
		host.innerHTML = `<div class="bar-empty">no runs yet</div>`;
		return;
	}
	// latest per (profile || model) for oneShot and interactive
	const latestByKey = new Map();
	for (const run of runs) {
		const key = run.profile ?? run.model;
		const prev = latestByKey.get(key);
		if (!prev || prev.timestamp < run.timestamp) latestByKey.set(key, run);
	}
	const rows = Array.from(latestByKey.entries());
	const maxVal = Math.max(
		1,
		...rows.flatMap(([, r]) => [
			r.oneShot?.tokensPerSecond ?? 0,
			r.interactive?.tokensPerSecond ?? 0,
		]),
	);
	rows.sort(
		(a, b) =>
			(b[1].oneShot?.tokensPerSecond ?? 0) -
			(a[1].oneShot?.tokensPerSecond ?? 0),
	);
	host.innerHTML = "";
	for (const [label, run] of rows) {
		const row = document.createElement("div");
		row.className = "bar-row";
		const oneShot = run.oneShot?.tokensPerSecond ?? 0;
		const interactive = run.interactive?.tokensPerSecond ?? 0;
		const oneShotPct = (oneShot / maxVal) * 100;
		const interactivePct = (interactive / maxVal) * 100;
		row.innerHTML = `
			<div class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
			<div class="bar-stack">
				<div class="bar"><div class="bar-fill" style="width:0%" data-width="${oneShotPct}"></div></div>
				<div class="bar interactive"><div class="bar-fill" style="width:0%" data-width="${interactivePct}"></div></div>
			</div>
			<div class="bar-value">${oneShot.toFixed(1)}/${interactive.toFixed(1)}</div>
		`;
		host.appendChild(row);
	}
	animateBars(host);
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
	host.innerHTML = "";
	// Latest per model for dimension aggregation.
	const latestByModel = new Map();
	for (const rep of evals) {
		const prev = latestByModel.get(rep.modelId);
		if (!prev || prev.timestamp < rep.timestamp) latestByModel.set(rep.modelId, rep);
	}
	if (latestByModel.size === 0) {
		host.innerHTML = `<div class="bar-empty">no completed evals yet</div>`;
		return;
	}
	for (const [model, rep] of latestByModel) {
		const card = document.createElement("div");
		card.className = "dim-card";
		const dimRows = Object.entries(rep.dimensions ?? {})
			.map(([dim, score]) => {
				const pct = Math.round((score.score ?? 0) * 100);
				return `
					<div class="dim-row">
						<div class="dim-label">${escapeHtml(dim)}</div>
						<div class="dim-bar"><div class="dim-fill" style="width:0%" data-width="${pct}"></div></div>
						<div class="dim-value">${pct}% <span class="dim-sub">(${score.passed}/${score.total})</span></div>
					</div>
				`;
			})
			.join("");
		const overallPct = Math.round((rep.overall ?? 0) * 100);
		const thinkLabel = rep.thinking === "on" ? "ON" : "OFF";
		const thinkClass = rep.thinking === "on" ? "think-on" : "off";
		card.innerHTML = `
			<div class="dim-card-head">
				<strong>${escapeHtml(model)}</strong>
				<span class="pill ${thinkClass}" title="thinking mode for the latest run">think ${thinkLabel}</span>
				<span class="overall-pill" data-strength="${overallStrength(rep.overall)}">${overallPct}%</span>
			</div>
			${dimRows}
		`;
		host.appendChild(card);
	}
	// After bars are in the DOM at width:0, animate them to their value.
	animateBars(host);
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
		const dimensionChips = Object.entries(rep.dimensions ?? {})
			.map(([dim, s]) => {
				const pct = Math.round((s.score ?? 0) * 100);
				return `<span class="dim-chip" data-strength="${overallStrength(s.score)}">${escapeHtml(dim)}: ${s.passed}/${s.total} · ${pct}%</span>`;
			})
			.join(" ");
		const thinkLabel = rep.thinking === "on" ? "on" : "off";
		const thinkClass = rep.thinking === "on" ? "think-on" : "";
		const profileCell = rep.profile
			? escapeHtml(rep.profile)
			: `<span class="dim">—</span>`;
		tr.innerHTML = `
			<td>${formatTime(rep.timestamp)}</td>
			<td>${escapeHtml(rep.modelId)}</td>
			<td>${profileCell}</td>
			<td><span class="pill ${thinkClass}">${thinkLabel}</span></td>
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

function evalComparator(a, b) {
	const k = state.evalSortKey;
	const dir = state.evalSortDir === "asc" ? 1 : -1;
	const av = a[k];
	const bv = b[k];
	if (av === bv) return 0;
	if (av === undefined || av === null) return 1;
	if (bv === undefined || bv === null) return -1;
	return av < bv ? -1 * dir : 1 * dir;
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
				`<div class="k">${escapeHtml(dim)}</div><div class="v">${s.passed}/${s.total} · ${Math.round((s.score ?? 0) * 100)}% · avg ${Math.round(s.avgLatencyMs ?? 0)}ms</div>`,
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
	return `
		<h3>Eval · ${escapeHtml(rep.modelId)}</h3>
		<div class="modal-kv">
			<div class="k">timestamp</div><div class="v">${escapeHtml(rep.timestamp)}</div>
			${rep.profile ? `<div class="k">profile</div><div class="v">${escapeHtml(rep.profile)}</div>` : ""}
			<div class="k">thinking</div><div class="v">${escapeHtml(rep.thinking ?? "off")}</div>
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
