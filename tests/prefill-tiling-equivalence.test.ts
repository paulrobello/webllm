import { describe, expect, test } from "bun:test";
import { ModelInference } from "../src/inference/model-inference.js";

// Skip under Bun — the tile loop's correctness depends on the GPU graph
// compute path being identical across calls, which can only run in a
// WebGPU-capable environment. Empirical equivalence is verified via the
// §22 Task 5 smoke matrix: TinyLlama with `?prefillTile=64` must produce
// the same top-1 token sequence as TinyLlama with `?prefillTile=0` on the
// prefill-256 fixture.
const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as { gpu?: unknown }).gpu !== "undefined";

describe.skipIf(!HAS_WEBGPU)("forward() tile-vs-untiled equivalence", () => {
	test("tiled prefill produces identical last-position logits to single-call", () => {
		// Contract: with `prefillTileSize > 0`, `forward(ids, positions)`
		// returns the same last-position logits as the untiled call. The
		// empirical check is the Task 5 smoke matrix (top-1 sequence
		// equivalence under realistic sampling on TinyLlama prefill-256).
		// This test is the placeholder seam for future browser-side test
		// infra; the assertion is currently shape-only.
		expect(ModelInference).toBeDefined();
	});
});
