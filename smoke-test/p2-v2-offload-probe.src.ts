// P2-v2 Phase 2 follow-on Task 13 — synthetic offload probe.
//
// Validates the Task 10 patch (`supports_buft` + `offload_op`) in its
// native habitat: a tiny ggml graph whose MUL_MAT inputs live on the
// CPU host buffer type. Under OUTCOME-D the production chat path is
// structurally dormant for JSEP (weights+KV in webgpu_buf, host-only
// gate in `ggml-backend.cpp:921` never fires). This probe sidesteps
// libllama and asks: does the scheduler route a MUL_MAT to JSEP when
// its src tensors really do live on host_buf?
//
// PASS = status === 0 AND module.__jsep.counters.runOp delta ≥ 1.
//
// Bundled to smoke-test/p2-v2-offload-probe.js via:
//   bun build smoke-test/p2-v2-offload-probe.src.ts \
//     --outfile smoke-test/p2-v2-offload-probe.js --target browser

import { installJsepCallbacks } from "../src/inference/jsep/index.js";

function log(msg: string, cls = ""): void {
	const el = document.getElementById("log");
	if (!el) return;
	const line = document.createElement("div");
	if (cls) line.className = cls;
	line.textContent = msg;
	el.appendChild(line);
	console.log(msg);
}

async function runProbe(): Promise<void> {
	try {
		log("[1/6] Initializing JSEP WASM module...");
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import("./webllm-wasm-jsep.js")).default;
		(window as any).__stderrLines = [];
		const mod: any = await createModule({
			printErr: (s: string) => {
				(window as any).__stderrLines.push(s);
				console.error(s);
			},
		});

		log("[2/6] Acquiring WebGPU device...");
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) {
			log("no WebGPU adapter", "fail");
			return;
		}
		const device = await adapter.requestDevice();

		log("[3/6] Installing JSEP callbacks (must precede webgpu_init)...");
		installJsepCallbacks(mod, device);

		log("[4/6] Initializing ggml-webgpu backend...");
		const initStatus = await mod._webgpu_init();
		if (initStatus !== 0) {
			log(`webgpu_init returned ${initStatus}`, "fail");
			return;
		}

		log("[5/6] Snapshotting JSEP counters before probe...");
		const counter0: Record<string, number> = {
			...(mod.__jsep?.counters ?? {}),
		};
		log(`     counters@pre = ${JSON.stringify(counter0)}`);

		log("[6/6] Running webllm_synthetic_offload_probe...");
		const status: number = mod._webllm_synthetic_offload_probe();
		const logPtr = mod._webllm_synthetic_probe_log();
		const probeLog = mod.UTF8ToString(logPtr);

		const counter1: Record<string, number> = {
			...(mod.__jsep?.counters ?? {}),
		};
		const deltas: Record<string, number> = {};
		for (const k of Object.keys(counter1)) {
			deltas[k] = (counter1[k] ?? 0) - (counter0[k] ?? 0);
		}
		const runOpDelta = deltas.runOp ?? 0;

		log(`PROBE_STATUS = ${status}`);
		log(`PROBE_LOG =\n${probeLog}`);
		log(`COUNTER_DELTAS = ${JSON.stringify(deltas)}`);
		log(`RUN_OP_DELTA = ${runOpDelta}`);

		const pass = status === 0 && runOpDelta >= 1;
		if (pass) {
			log("VERDICT: PASS — JSEP fired ≥1 MUL_MAT via offload_op.", "pass");
		} else if (status === 2) {
			log(
				"VERDICT: NOT-APPLICABLE — JSEP not registered in this build.",
				"fail",
			);
		} else if (status !== 0) {
			log(`VERDICT: FAIL — probe returned non-zero status ${status}.`, "fail");
		} else {
			log(
				`VERDICT: FAIL — scheduler ran but JSEP runOp delta = ${runOpDelta} (expected ≥ 1).`,
				"fail",
			);
		}

		(window as any).__probeResult = {
			status,
			log: probeLog,
			runOpDelta,
			deltas,
			counter0,
			counter1,
			pass,
		};
	} catch (err: unknown) {
		const e = err as Error;
		log(`FAIL — ${e.message}\n${e.stack ?? ""}`, "fail");
	}
}

void runProbe();
