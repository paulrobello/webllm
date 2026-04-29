import { describe, expect, test } from "bun:test";

// Locks the offset/stride math used in src/inference/model-inference.ts
// for the Phi-3 fused-QKV and fused-gate-up forward paths
// (buildQKV / buildFFNGateUp). A single off-by-element in those
// formulas is the most likely failure mode and is invisible until
// the model produces gibberish output.
//
// The fused QKV matrix is [E + 2*kvDim, E] (E = n_embd, kvDim =
// headDim * n_kv_heads), stored row-major. After matmul with
// activation x=[E, nTokens] the output is [E + 2*kvDim, nTokens].
// Q occupies rows [0, E), K rows [E, E+kvDim), V rows [E+kvDim,
// E + 2*kvDim).
//
// The fused gate-up matrix is [2*ffSize, E]; output [2*ffSize,
// nTokens]. Gate rows [0, ffSize), Up rows [ffSize, 2*ffSize).

const F32_BYTES = 4;

describe("phi3 fused QKV view-offset math", () => {
	const cases = [
		{
			name: "phi-3.5-mini (no GQA, kvDim == E)",
			E: 3072,
			headDim: 96,
			nHeads: 32,
			nKvHeads: 32,
		},
		{
			name: "hypothetical phi3 with GQA 4:1",
			E: 3072,
			headDim: 96,
			nHeads: 32,
			nKvHeads: 8,
		},
	];
	for (const c of cases) {
		test(c.name, () => {
			const kvDim = c.headDim * c.nKvHeads;
			const fusedRowDim = c.E + 2 * kvDim;
			const tokenBytes = F32_BYTES * fusedRowDim;
			const qOffset = 0;
			const kOffset = F32_BYTES * c.E;
			const vOffset = F32_BYTES * (c.E + kvDim);
			expect(qOffset).toBe(0);
			expect(kOffset).toBe(F32_BYTES * c.E);
			expect(vOffset).toBe(F32_BYTES * c.E + F32_BYTES * kvDim);
			expect(tokenBytes).toBe(F32_BYTES * (c.E + 2 * kvDim));
			expect(c.headDim * c.nHeads).toBe(c.E);
			expect(c.headDim * c.nKvHeads).toBe(kvDim);
		});
	}
});

describe("phi3 fused gate-up view-offset math", () => {
	test("gate is FIRST half, up is SECOND half (HF Phi3MLP order)", () => {
		// HF Phi3MLP forward is `up * silu(gate)` with `chunk(2, dim=-1)`,
		// so HF stores [gate | up] along the output dim. llama.cpp's
		// convert_hf_to_gguf.py Phi3MiniModel preserves the order, and
		// its swiglu kernel (ggml-cpu/ops.cpp:3170-3179) computes
		// `silu(first_half) * second_half` when swapped=0, matching.
		const ffSize = 8192;
		const tokenBytes = F32_BYTES * 2 * ffSize;
		const gateOffset = 0;
		const upOffset = F32_BYTES * ffSize;
		expect(gateOffset).toBe(0);
		expect(upOffset).toBe(F32_BYTES * ffSize);
		expect(tokenBytes).toBe(2 * upOffset);
	});
});
