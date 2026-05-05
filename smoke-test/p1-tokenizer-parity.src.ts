// P1 tokenizer parity smoke harness. For each vocab in the fixture:
// 1. fetch its GGUF
// 2. webllm_load_model
// 3. construct LlamaTokenizer
// 4. for each fixture prompt, encode via LlamaTokenizer and assert
//    Int32Array equality with the fixture's expected ids
// 5. log per-vocab PASS/FAIL with the first 3 mismatches if any
//
// Bundled to smoke-test/p1-tokenizer-parity.js via:
//   bun build smoke-test/p1-tokenizer-parity.src.ts \
//     --outfile smoke-test/p1-tokenizer-parity.js --target browser

import { createLlamaBridge } from "../src/inference/llama-bridge.js";
import { LlamaTokenizer } from "../src/inference/llama-tokenizer.js";

const FIXTURE_URL = "/parity-fixture.json";

interface FixtureEntry {
	vocab: string;
	ggufUrl: string;
	expected: { prompt: string; ids: number[] }[];
}
interface Fixture {
	prompts: string[];
	fixture: FixtureEntry[];
}

function log(msg: string, cls = ""): void {
	const el = document.getElementById("log");
	if (!el) return;
	const line = document.createElement("div");
	if (cls) line.className = cls;
	line.textContent = msg;
	el.appendChild(line);
	// Mirror to console so agentchrome console-follow captures it.
	console.log(msg);
}

function arraysEqual(a: number[], b: Int32Array): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
	return true;
}

async function runParity(): Promise<void> {
	try {
		log("[setup] Fetching fixture…", "info");
		const fixtureResp = await fetch(FIXTURE_URL);
		if (!fixtureResp.ok) {
			log(
				`fetch fixture failed: ${fixtureResp.status} ${fixtureResp.statusText}`,
				"fail",
			);
			return;
		}
		const f = (await fixtureResp.json()) as Fixture;
		log(
			`[setup] ${f.fixture.length} vocab(s), ${f.prompts.length} prompts each`,
			"info",
		);

		log("[setup] Initializing WASM module…", "info");
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import("./webllm-wasm.js")).default;
		// biome-ignore lint/suspicious/noExplicitAny: emscripten module shape
		const mod: any = await createModule();
		log("[setup] Initializing WebGPU backend…", "info");
		const initStatus = await mod._webgpu_init();
		if (initStatus !== 0) {
			log(`webgpu_init returned ${initStatus}`, "fail");
			return;
		}
		const bridge = createLlamaBridge(mod);

		let totalFail = 0;
		let totalPass = 0;
		for (const entry of f.fixture) {
			log(`\n[${entry.vocab}] fetching ${entry.ggufUrl}…`, "info");
			const ggufResp = await fetch(entry.ggufUrl);
			if (!ggufResp.ok) {
				log(
					`[${entry.vocab}] fetch failed: ${ggufResp.status} ${ggufResp.statusText}`,
					"fail",
				);
				totalFail += entry.expected.length;
				continue;
			}
			const buf = new Uint8Array(await ggufResp.arrayBuffer());
			log(
				`[${entry.vocab}] loading model (${(buf.byteLength / 1024 / 1024).toFixed(0)} MiB)…`,
				"info",
			);
			const model = await bridge.loadModel(buf);
			const tk = new LlamaTokenizer(bridge, model);

			let mismatches = 0;
			const samples: string[] = [];
			for (let i = 0; i < entry.expected.length; i++) {
				const { prompt, ids: expected } = entry.expected[i];
				const got = tk.encode(prompt);
				const gotArr = new Int32Array(got);
				if (!arraysEqual(expected, gotArr)) {
					mismatches++;
					if (samples.length < 3) {
						samples.push(
							`  [${i}] ${JSON.stringify(prompt).slice(0, 60)}\n` +
								`    expected: [${expected.slice(0, 12).join(",")}${expected.length > 12 ? ",…" : ""}]\n` +
								`    got     : [${Array.from(gotArr.slice(0, 12)).join(",")}${gotArr.length > 12 ? ",…" : ""}]`,
						);
					}
				}
			}

			if (mismatches === 0) {
				log(
					`[${entry.vocab}] PASS — ${entry.expected.length}/${entry.expected.length} byte-exact`,
					"pass",
				);
				totalPass += entry.expected.length;
			} else {
				totalFail += mismatches;
				totalPass += entry.expected.length - mismatches;
				log(
					`[${entry.vocab}] FAIL — ${mismatches}/${entry.expected.length} mismatches`,
					"fail",
				);
				for (const s of samples) log(s, "fail");
			}

			bridge.freeModel(model);
		}

		if (totalFail === 0) {
			log(
				`\nALL VOCABS PASS — parity gate green (${totalPass} prompts byte-exact)`,
				"pass",
			);
		} else {
			log(
				`\nFAIL — ${totalFail} mismatches out of ${totalPass + totalFail} total prompts`,
				"fail",
			);
		}
	} catch (err: unknown) {
		const e = err as Error;
		log(`FAIL — ${e.message}\n${e.stack ?? ""}`, "fail");
	}
}

void runParity();
