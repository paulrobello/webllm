// Frame-timing probe — first-class smoke-page mode.
//
// Gated by ?frameProbe=1 in real-model-page.js. Spins a small WebGL2 cube via
// rAF in parallel with [7/8] chat generation and reports per-phase frame
// deltas (baseline / prefill / decode / post). The stats answer the agent +
// Three.js coexistence question: does main-thread inference disrupt a 60fps
// render loop?
//
// Phase segmentation is timestamp-based (no engine monkey-patching): the
// caller passes `tStart` (immediately before chatCompletion), `prefillMs`
// (from `smokeResult.prefillMs`), and `tEnd` (immediately after). We split
// captured rAF timestamps into the four windows post-hoc.

const VERTS = new Float32Array([
	-1, -1, -1, 1, 0, 0, 1, -1, -1, 0, 1, 0, 1, 1, -1, 0, 0, 1, -1, 1, -1, 1, 1, 0,
	-1, -1, 1, 1, 0, 1, 1, -1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 0.4, 0.4,
	0.4,
]);
const IDX = new Uint16Array([
	0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 2, 6, 3, 6, 7, 0, 3,
	7, 0, 7, 4, 1, 2, 6, 1, 6, 5,
]);

const VS = `#version 300 es
in vec3 p; in vec3 c; uniform mat4 mvp; out vec3 vc;
void main() { gl_Position = mvp * vec4(p, 1.0); vc = c; }`;
const FS = `#version 300 es
precision mediump float;
in vec3 vc; out vec4 frag;
void main() { frag = vec4(vc, 1.0); }`;

function mat4Mul(a, b) {
	const r = new Float32Array(16);
	for (let i = 0; i < 4; i++)
		for (let j = 0; j < 4; j++)
			r[i * 4 + j] =
				a[i * 4] * b[j] +
				a[i * 4 + 1] * b[j + 4] +
				a[i * 4 + 2] * b[j + 8] +
				a[i * 4 + 3] * b[j + 12];
	return r;
}
function persp(fov, aspect, n, f) {
	const t = Math.tan(fov / 2);
	return new Float32Array([
		1 / (aspect * t), 0, 0, 0, 0, 1 / t, 0, 0, 0, 0, (n + f) / (n - f), -1, 0,
		0, (2 * n * f) / (n - f), 0,
	]);
}
function rotY(a) {
	const c = Math.cos(a), s = Math.sin(a);
	return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
}
function rotX(a) {
	const c = Math.cos(a), s = Math.sin(a);
	return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}
function trans(x, y, z) {
	return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

function statsOf(arr) {
	if (!arr.length) return null;
	const sorted = [...arr].sort((a, b) => a - b);
	const med = sorted[Math.floor(sorted.length / 2)];
	const p95 = sorted[Math.floor(sorted.length * 0.95)];
	const max = sorted[sorted.length - 1];
	const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
	return {
		n: arr.length,
		mean: +mean.toFixed(2),
		median: +med.toFixed(2),
		p95: +p95.toFixed(2),
		max: +max.toFixed(2),
		drops60: arr.filter((x) => x > 16.67).length,
		drops30: arr.filter((x) => x > 33.33).length,
		drops50: arr.filter((x) => x > 50).length,
	};
}

function buildOverlay({ width, height, labelText }) {
	const root = document.createElement("div");
	root.id = "frame-probe-overlay";
	root.style.cssText =
		"position:fixed;top:12px;right:12px;z-index:9999;background:#161b22;border:1px solid #30363d;border-radius:6px;padding:6px;font:11px ui-monospace,monospace;color:#8b949e;text-align:center;";
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	canvas.style.cssText = "display:block;border-radius:4px;background:#0d1117;";
	const label = document.createElement("div");
	label.textContent = labelText;
	label.style.marginTop = "4px";
	root.appendChild(canvas);
	root.appendChild(label);
	document.body.appendChild(root);
	return { root, canvas, label };
}

function runCubeProbe({ root, canvas }) {
	const gl = canvas.getContext("webgl2", { antialias: true });
	if (!gl) {
		root.remove();
		return { stop: () => {}, samples: [], unsupported: true };
	}
	function compile(t, src) {
		const s = gl.createShader(t);
		gl.shaderSource(s, src);
		gl.compileShader(s);
		return s;
	}
	const prog = gl.createProgram();
	gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
	gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
	gl.linkProgram(prog);
	gl.useProgram(prog);
	const vao = gl.createVertexArray();
	gl.bindVertexArray(vao);
	const vb = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, vb);
	gl.bufferData(gl.ARRAY_BUFFER, VERTS, gl.STATIC_DRAW);
	const ib = gl.createBuffer();
	gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
	gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, IDX, gl.STATIC_DRAW);
	const pLoc = gl.getAttribLocation(prog, "p");
	gl.enableVertexAttribArray(pLoc);
	gl.vertexAttribPointer(pLoc, 3, gl.FLOAT, false, 24, 0);
	const cLoc = gl.getAttribLocation(prog, "c");
	gl.enableVertexAttribArray(cLoc);
	gl.vertexAttribPointer(cLoc, 3, gl.FLOAT, false, 24, 12);
	const mvpLoc = gl.getUniformLocation(prog, "mvp");
	gl.enable(gl.DEPTH_TEST);
	gl.viewport(0, 0, canvas.width, canvas.height);
	const P = persp(Math.PI / 3, canvas.width / canvas.height, 0.1, 100);

	const samples = [];
	let lastT = performance.now();
	let running = true;
	function tick(now) {
		const dt = now - lastT;
		lastT = now;
		samples.push({ t: now, dt });
		gl.clearColor(0.06, 0.07, 0.09, 1);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		const m = mat4Mul(
			mat4Mul(rotX(now * 0.0007), rotY(now * 0.001)),
			trans(0, 0, -4),
		);
		gl.uniformMatrix4fv(mvpLoc, false, mat4Mul(m, P));
		gl.bindVertexArray(vao);
		gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
		if (running) requestAnimationFrame(tick);
	}
	requestAnimationFrame(tick);

	return {
		samples,
		sceneInfo: { kind: "cube", triangles: 12 },
		stop({ removeOverlay = true } = {}) {
			running = false;
			if (removeOverlay) root.remove();
		},
	};
}

/**
 * Loads Three.js + a real GLTF scene at `sceneUrl` to stress-test the GPU
 * queue alongside main-thread inference. Three.js is loaded via esm.sh so
 * the dependency stays opt-in and out of the production bundle.
 *
 * Returns the same controller shape as `runCubeProbe` so callers don't
 * branch on mode.
 */
async function runSceneProbe({ root, canvas, label }, sceneUrl) {
	label.textContent = "loading three.js…";
	const THREE = await import("https://esm.sh/three@0.160.0");
	const { GLTFLoader } = await import(
		"https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js"
	);

	const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
	renderer.setPixelRatio(window.devicePixelRatio || 1);
	renderer.setSize(canvas.width, canvas.height, false);
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.outputColorSpace = THREE.SRGBColorSpace;

	const scene = new THREE.Scene();
	scene.background = new THREE.Color(0x0d1117);
	scene.add(new THREE.AmbientLight(0xffffff, 0.6));
	const sun = new THREE.DirectionalLight(0xffffff, 1.5);
	sun.position.set(80, 120, 60);
	scene.add(sun);
	const camera = new THREE.PerspectiveCamera(
		50,
		canvas.width / canvas.height,
		0.1,
		8000,
	);

	label.textContent = `loading ${sceneUrl.split("/").pop()}…`;
	const tFetch = performance.now();
	const gltf = await new Promise((resolve, reject) => {
		new GLTFLoader().load(sceneUrl, resolve, undefined, reject);
	});
	const fetchMs = Math.round(performance.now() - tFetch);
	scene.add(gltf.scene);

	let triangles = 0;
	gltf.scene.traverse((obj) => {
		if (!obj.isMesh) return;
		const g = obj.geometry;
		if (g.index) triangles += g.index.count / 3;
		else if (g.attributes.position) triangles += g.attributes.position.count / 3;
	});

	const box = new THREE.Box3().setFromObject(gltf.scene);
	const size = box.getSize(new THREE.Vector3());
	const center = box.getCenter(new THREE.Vector3());
	// Pick orbit radius that makes the *entire* bounding box fit within the
	// camera frustum at the given FOV/aspect, then add a 1.4× margin so
	// rotating views never clip the scene's diagonal extent.
	const maxExtent = Math.max(size.x, size.y, size.z) || 10;
	const aspect = canvas.width / canvas.height;
	const vFov = (camera.fov * Math.PI) / 180;
	const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
	const fitDist =
		maxExtent / 2 / Math.min(Math.tan(vFov / 2), Math.tan(hFov / 2));
	const orbitRadius = fitDist * 1.4;
	camera.near = Math.max(0.01, orbitRadius * 0.001);
	camera.far = orbitRadius * 10;
	camera.updateProjectionMatrix();

	const triLabel = `${Math.round(triangles).toLocaleString()} tri`;
	const extLabel = `${size.x.toFixed(0)}×${size.y.toFixed(0)}×${size.z.toFixed(0)}`;
	label.textContent = `scene · ${triLabel} · ${extLabel} · loaded ${fetchMs}ms`;

	const samples = [];
	let lastT = performance.now();
	let running = true;
	function tick(now) {
		const dt = now - lastT;
		lastT = now;
		samples.push({ t: now, dt });
		const angle = now * 0.0003;
		camera.position.x = center.x + Math.cos(angle) * orbitRadius;
		camera.position.z = center.z + Math.sin(angle) * orbitRadius;
		camera.position.y = center.y + orbitRadius * 0.5;
		camera.lookAt(center);
		renderer.render(scene, camera);
		if (running) requestAnimationFrame(tick);
	}
	requestAnimationFrame(tick);

	return {
		samples,
		sceneInfo: {
			kind: "gltf",
			url: sceneUrl,
			triangles: Math.round(triangles),
			loadMs: fetchMs,
		},
		stop({ removeOverlay = true } = {}) {
			running = false;
			renderer.dispose();
			if (removeOverlay) root.remove();
		},
	};
}

/**
 * Mount a render loop in the top-right of the viewport and start an rAF
 * loop that records callback timestamps. Returns a controller with
 * `stop()` and `samples` (the captured `{t, dt}` records).
 *
 * Modes:
 *   default      — trivial WebGL2 spinning cube, 200×200 (low GPU contention,
 *                  isolates main-thread scheduling effects).
 *   sceneUrl=... — Three.js + GLTFLoader, loads a real scene at the given
 *                  URL (relative or absolute), 480×320 (real GPU contention
 *                  alongside WebGPU LLM dispatches).
 */
export async function startFrameProbe(options = {}) {
	const sceneUrl = options.sceneUrl || null;
	if (sceneUrl) {
		const overlay = buildOverlay({
			width: 480,
			height: 320,
			labelText: "frame-probe (scene)",
		});
		try {
			return await runSceneProbe(overlay, sceneUrl);
		} catch (e) {
			overlay.label.textContent = `scene load failed: ${e.message}`;
			console.error("[frame-probe] scene load failed", e);
			// Fall back to cube so the run isn't lost.
			overlay.canvas.width = 200;
			overlay.canvas.height = 200;
			return runCubeProbe(overlay);
		}
	}
	const overlay = buildOverlay({
		width: 200,
		height: 200,
		labelText: "frame-probe (rAF)",
	});
	return runCubeProbe(overlay);
}

/**
 * Segment captured rAF samples by wall-clock against the chat completion
 * timeline and return per-phase stats + a verdict.
 *
 * Inputs:
 *   samples   — array of {t, dt} produced by startFrameProbe
 *   tStart    — performance.now() captured immediately before chatCompletion
 *   prefillMs — smokeResult.prefillMs (time from tStart to first decoded token)
 *   tEnd      — performance.now() captured immediately after chatCompletion
 *
 * The first 2 baseline samples are dropped (rAF warmup spike).
 */
export function summarizeFrameProbe({ samples, tStart, prefillMs, tEnd }) {
	const tFirstToken = tStart + (prefillMs || 0);
	const baseline = [];
	const prefill = [];
	const decode = [];
	const post = [];
	for (const s of samples) {
		if (s.t < tStart) baseline.push(s.dt);
		else if (s.t < tFirstToken) prefill.push(s.dt);
		else if (s.t <= tEnd) decode.push(s.dt);
		else post.push(s.dt);
	}
	const baseTrim = baseline.slice(2);
	const sBase = statsOf(baseTrim);
	const sPrefill = statsOf(prefill);
	const sDecode = statsOf(decode);
	const sPost = statsOf(post);
	const decodeP95 = sDecode?.p95 ?? 0;
	let verdict;
	if (decodeP95 < 18) verdict = "clean";
	else if (decodeP95 < 35) verdict = "marginal";
	else verdict = "drops";
	return {
		baseline: sBase,
		prefill: sPrefill,
		decode: sDecode,
		post: sPost,
		verdict,
		windowMs: { baseline: tStart, firstToken: tFirstToken, end: tEnd },
	};
}

/**
 * Multi-call variant: bucket samples per chat completion call so we can
 * see whether a per-call hitch is deterministic (lands at the same point
 * every call) or stochastic (scattered/absent). `calls` is an array of
 * `{tStart, prefillMs, tEnd}` records — one per chatCompletion invocation.
 *
 * `tBaselineStart` defines the start of the baseline window. Samples
 * before it are discarded as warmup.
 */
export function summarizeFrameProbeMulti({
	samples,
	tBaselineStart,
	calls,
}) {
	const baseline = [];
	const perCall = calls.map(() => ({ prefill: [], decode: [] }));
	const post = [];
	const finalEnd = calls[calls.length - 1]?.tEnd ?? Number.POSITIVE_INFINITY;
	const firstStart = calls[0]?.tStart ?? Number.POSITIVE_INFINITY;
	for (const s of samples) {
		if (s.t < tBaselineStart) continue;
		if (s.t < firstStart) {
			baseline.push(s.dt);
			continue;
		}
		if (s.t > finalEnd) {
			post.push(s.dt);
			continue;
		}
		for (let i = 0; i < calls.length; i++) {
			const c = calls[i];
			const tFirstToken = c.tStart + (c.prefillMs || 0);
			if (s.t < c.tStart) break;
			if (s.t < tFirstToken) {
				perCall[i].prefill.push(s.dt);
				break;
			}
			if (s.t <= c.tEnd) {
				perCall[i].decode.push(s.dt);
				break;
			}
		}
	}
	return {
		baseline: statsOf(baseline.slice(2)),
		post: statsOf(post),
		calls: perCall.map((c, i) => ({
			index: i,
			prefill: statsOf(c.prefill),
			decode: statsOf(c.decode),
		})),
	};
}

/**
 * Format multi-call summary as a per-call table plus an aggregate read.
 */
export function formatFrameProbeMultiReport(summary) {
	const lines = [];
	lines.push("=== FRAME TIMING (rAF deltas, ms) — multi-call ===");
	const b = summary.baseline;
	if (b)
		lines.push(
			`  baseline: n=${b.n}  median=${b.median}ms  p95=${b.p95}ms  max=${b.max}ms  drops60=${b.drops60}`,
		);
	lines.push(
		"  call#   prefill_p95  prefill_max  decode_p95  decode_max  drops60  drops50",
	);
	const decodeMaxes = [];
	const decodeP95s = [];
	let totalDrops60 = 0;
	let totalDrops50 = 0;
	let totalDecodeFrames = 0;
	for (const c of summary.calls) {
		const p = c.prefill;
		const d = c.decode;
		if (!d) {
			lines.push(`  ${String(c.index).padStart(2)}      —`);
			continue;
		}
		decodeMaxes.push(d.max);
		decodeP95s.push(d.p95);
		totalDrops60 += d.drops60;
		totalDrops50 += d.drops50;
		totalDecodeFrames += d.n;
		const fmt = (n, w) => String(n).padStart(w);
		lines.push(
			`  ${fmt(c.index, 5)}   ${fmt((p?.p95 ?? "—").toString(), 11)}  ${fmt((p?.max ?? "—").toString(), 11)}  ${fmt(d.p95.toString(), 10)}  ${fmt(d.max.toString(), 10)}  ${fmt(`${d.drops60}/${d.n}`, 7)}  ${fmt(d.drops50.toString(), 7)}`,
		);
	}
	if (decodeMaxes.length) {
		const sortMax = [...decodeMaxes].sort((a, b) => a - b);
		const med = sortMax[Math.floor(sortMax.length / 2)];
		const min = sortMax[0];
		const max = sortMax[sortMax.length - 1];
		lines.push(
			`  decode_max across calls: min=${min.toFixed(1)}  median=${med.toFixed(1)}  max=${max.toFixed(1)}`,
		);
		lines.push(
			`  aggregate decode drops: >16.67ms ${totalDrops60}/${totalDecodeFrames} (${((totalDrops60 / totalDecodeFrames) * 100).toFixed(1)}%)  >50ms ${totalDrops50}/${totalDecodeFrames}`,
		);
		// Heuristic: if every call has a max ≥ 30ms AND the variance is
		// tight, the hitch looks deterministic. If maxes vary widely or
		// some calls have no drop, it's jitter.
		const allHitched = sortMax.every((x) => x >= 30);
		const tightSpread = max - min < 25;
		let pattern;
		if (allHitched && tightSpread) pattern = "DETERMINISTIC (every call hitches in a narrow band)";
		else if (totalDrops50 === 0) pattern = "NO HITCH (no >50ms drops across any call)";
		else pattern = "STOCHASTIC (drops scattered, not every call hitches the same way)";
		lines.push(`  pattern: ${pattern}`);
	}
	return lines;
}

/**
 * Format a multi-line summary suitable for the smoke-page log. Returned as
 * an array of lines (caller controls log() invocation cadence).
 */
export function formatFrameProbeReport(summary) {
	const lines = [];
	lines.push("=== FRAME TIMING (rAF deltas, ms) ===");
	for (const [label, s] of [
		["baseline", summary.baseline],
		["prefill ", summary.prefill],
		["decode  ", summary.decode],
		["post    ", summary.post],
	]) {
		if (!s) {
			lines.push(`  ${label}: no samples`);
			continue;
		}
		lines.push(
			`  ${label}: n=${s.n}  median=${s.median}ms  p95=${s.p95}ms  max=${s.max}ms`,
		);
		const pct = ((s.drops60 / s.n) * 100).toFixed(1);
		lines.push(
			`    drops >16.67ms (60fps): ${s.drops60} (${pct}%)  >33ms: ${s.drops30}  >50ms: ${s.drops50}`,
		);
	}
	const dp95 = summary.decode?.p95 ?? 0;
	const verdictLine =
		summary.verdict === "clean"
			? `CLEAN: decode p95 ${dp95}ms — 60fps maintained.`
			: summary.verdict === "marginal"
				? `MARGINAL: decode p95 ${dp95}ms — 60fps drops, 30fps held.`
				: `DROPS: decode p95 ${dp95}ms — visible stutter.`;
	lines.push(`=== VERDICT: ${verdictLine} ===`);
	const prefMax = summary.prefill?.max ?? 0;
	if (prefMax > 100)
		lines.push(`  PREFILL HITCH: ${prefMax.toFixed(0)}ms longest gap.`);
	else lines.push(`  PREFILL CLEAN: longest gap ${prefMax.toFixed(0)}ms.`);
	return lines;
}
