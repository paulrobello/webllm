// P1 fixture regenerator — reads the existing parity-fixture.json
// (for the canonical 200-prompt corpus and per-vocab GGUF URL list),
// loads each model via webllm_load_model, encodes every prompt via
// LlamaTokenizer, and POSTs the regenerated JSON back to the
// smoke-serve /save-parity-fixture endpoint. Output replaces the
// legacy-encoder-derived `expected` arrays with canonical
// llama_tokenize output.
//
// Per-vocab addBos:
//   - causal-LM vocabs (spm-llama, llama-bpe, qwen2, qwen3): false.
//     Engine.ts adds BOS via chat template; matching that here.
//   - encoder-only (wordpiece-bert): true. For BERT-family the BOS
//     IS [CLS]; addBos=true makes llama_tokenize prepend [CLS] and
//     append [SEP] which is the canonical BERT input layout.
//
// Bundled to smoke-test/p1-fixture-regen.js via:
//   bun build smoke-test/p1-fixture-regen.src.ts \
//     --outfile smoke-test/p1-fixture-regen.js --target browser

import { createLlamaBridge } from "../src/inference/llama-bridge.js";
import { LlamaTokenizer } from "../src/inference/llama-tokenizer.js";

const FIXTURE_URL = "/parity-fixture.json";
const SAVE_URL = "/save-parity-fixture";

interface FixtureEntry {
	vocab: string;
	ggufUrl: string;
	expected: { prompt: string; ids: number[] }[];
}
interface Fixture {
	prompts: string[];
	fixture: FixtureEntry[];
}

const ENCODER_ONLY_VOCABS = new Set<string>(["wordpiece-bert"]);

function log(msg: string, cls = ""): void {
	const el = document.getElementById("log");
	if (!el) return;
	const line = document.createElement("div");
	if (cls) line.className = cls;
	line.textContent = msg;
	el.appendChild(line);
	console.log(msg);
}

async function run(): Promise<void> {
	try {
		log("[setup] Fetching existing fixture (for prompts + URLs)…", "info");
		const fixtureResp = await fetch(FIXTURE_URL);
		if (!fixtureResp.ok) {
			log(`fetch fixture failed: ${fixtureResp.status}`, "fail");
			return;
		}
		const f = (await fixtureResp.json()) as Fixture;
		log(
			`[setup] ${f.fixture.length} vocab(s), ${f.prompts.length} prompts each`,
			"info",
		);

		// ?only=<vocab> filter — regenerate just one vocab's entry and
		// merge into the existing fixture before saving. Lets us avoid
		// the 4 GiB-wasm32-cap WebGPU buffer leak that fires on
		// sequential cross-vocab loads. Run once per vocab in fresh
		// page loads to fully regenerate.
		const params = new URLSearchParams(window.location.search);
		const only = params.get("only");
		const fixtureToRun = only
			? f.fixture.filter((e) => e.vocab === only)
			: f.fixture;
		if (only && fixtureToRun.length === 0) {
			log(`[setup] no vocab matches ?only=${only}`, "fail");
			return;
		}
		log(
			only
				? `[setup] filtering to vocab="${only}" — will merge into existing fixture`
				: `[setup] running all ${fixtureToRun.length} vocabs (will replace fixture wholesale)`,
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

		const regen: FixtureEntry[] = [];
		for (const entry of fixtureToRun) {
			const encoderOnly = ENCODER_ONLY_VOCABS.has(entry.vocab);
			log(
				`\n[${entry.vocab}] encoderOnly=${encoderOnly} — fetching ${entry.ggufUrl}…`,
				"info",
			);
			const ggufResp = await fetch(entry.ggufUrl);
			if (!ggufResp.ok) {
				log(
					`[${entry.vocab}] fetch failed: ${ggufResp.status} ${ggufResp.statusText}`,
					"fail",
				);
				return;
			}
			const buf = new Uint8Array(await ggufResp.arrayBuffer());
			log(
				`[${entry.vocab}] loading model (${(buf.byteLength / 1024 / 1024).toFixed(0)} MiB)…`,
				"info",
			);
			const model = await bridge.loadModel(buf);
			const tk = new LlamaTokenizer(bridge, model, { encoderOnly });

			const expected = f.prompts.map((prompt) => ({
				prompt,
				ids: tk.encode(prompt),
			}));
			regen.push({
				vocab: entry.vocab,
				ggufUrl: entry.ggufUrl,
				expected,
			});
			log(
				`[${entry.vocab}] encoded ${f.prompts.length} prompts (first ids=${expected[0].ids.slice(0, 8).join(",")}…)`,
				"pass",
			);

			bridge.freeModel(model);
		}

		// Merge with existing fixture when filtering: replace only the
		// regenerated vocab(s), preserve the others. Without merge, a
		// single-vocab run would clobber the other vocabs' entries.
		let mergedFixture: FixtureEntry[];
		if (only) {
			mergedFixture = f.fixture.map((e) => {
				const replaced = regen.find((r) => r.vocab === e.vocab);
				return replaced ?? e;
			});
		} else {
			mergedFixture = regen;
		}
		const out = JSON.stringify(
			{ prompts: f.prompts, fixture: mergedFixture },
			null,
			2,
		);
		log(
			`\n[save] POSTing ${out.length} bytes to ${SAVE_URL}…`,
			"info",
		);
		const saveResp = await fetch(SAVE_URL, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: out,
		});
		if (!saveResp.ok) {
			log(
				`[save] HTTP ${saveResp.status}: ${await saveResp.text()}`,
				"fail",
			);
			return;
		}
		log(
			`[save] OK — ${await saveResp.text()}`.trimEnd(),
			"pass",
		);
		log("\nDONE — re-run /p1-tokenizer-parity.html to verify byte-exact green.", "pass");
	} catch (err: unknown) {
		const repr =
			err instanceof Error
				? `${err.name}: ${err.message}\n${err.stack ?? ""}`
				: `(non-Error throw, typeof=${typeof err}) ${String(err)}`;
		log(`FAIL — ${repr}`, "fail");
		console.error("regen-harness throw:", err);
	}
}

void run();
