/**
 * Locks `assertContiguousF32` — the runtime guard wired into
 * `buildQKV` / `buildFFNGateUp`'s fused branches in
 * `src/inference/model-inference.ts`.
 *
 * Defense-in-depth against the bug class fixed in commit `7c85a2a`
 * (Phi-3 closure, 2026-04-29): a strided `opView*` slice fed into the
 * rope/permute chain without an `opCont` materialization produced
 * silent gibberish at decode. The integration call sites are exercised
 * by the Phi-3 smoke run; this test locks the helper itself so a
 * future refactor of the assertion is caught without needing the full
 * WebGPU smoke harness.
 */

import { describe, expect, test } from "bun:test";
import {
	F32_BYTES,
	GgmlType,
	type GgmlWasm,
	type TensorPtr,
} from "../src/inference/ggml-wasm.js";
import { assertContiguousF32 } from "../src/inference/model-inference.js";

interface FakeTensor {
	type: number;
	ne: [number, number, number, number];
	nb: [number, number, number, number];
}

function makeWasm(
	tensor: FakeTensor,
): Pick<GgmlWasm, "tensorType" | "tensorNe" | "tensorNb"> {
	return {
		tensorType: (_t: TensorPtr) => tensor.type,
		tensorNe: (_t: TensorPtr, dim: number) => tensor.ne[dim],
		tensorNb: (_t: TensorPtr, dim: number) => tensor.nb[dim],
	};
}

function contiguous3d(ne0: number, ne1: number, ne2: number): FakeTensor {
	const nb0 = F32_BYTES;
	const nb1 = nb0 * ne0;
	const nb2 = nb1 * ne1;
	const nb3 = nb2 * ne2;
	return {
		type: GgmlType.F32,
		ne: [ne0, ne1, ne2, 1],
		nb: [nb0, nb1, nb2, nb3],
	};
}

describe("assertContiguousF32", () => {
	test("passes for a tightly packed F32 tensor matching buildQKV.q3 shape", () => {
		// phi-3.5-mini fused-QKV q3 shape: [headDim=96, nHeads=32, nTokens]
		const t = contiguous3d(96, 32, 7);
		expect(() =>
			assertContiguousF32(makeWasm(t), 1 as TensorPtr, "q3"),
		).not.toThrow();
	});

	test("passes for a tightly packed F32 2D tensor (gate-up output)", () => {
		// fused-gate-up gate shape is 2D [ffSize, nTokens] with ne[2]=1.
		const t = contiguous3d(8192, 7, 1);
		expect(() =>
			assertContiguousF32(makeWasm(t), 2 as TensorPtr, "gate"),
		).not.toThrow();
	});

	test("rejects non-F32 dtype with the expected error", () => {
		const t = contiguous3d(96, 32, 7);
		t.type = GgmlType.F16;
		expect(() =>
			assertContiguousF32(makeWasm(t), 3 as TensorPtr, "q3"),
		).toThrow(/q3: expected F32 \(type 0\), got type 1/);
	});

	test("rejects nb[0] != element_size (the strided-leading-dim case)", () => {
		const t = contiguous3d(96, 32, 7);
		t.nb[0] = 8;
		expect(() =>
			assertContiguousF32(makeWasm(t), 4 as TensorPtr, "q3"),
		).toThrow(/not F32-contiguous/);
	});

	test("rejects nb[1] != nb[0]*ne[0] (the row-stride-gap case)", () => {
		const t = contiguous3d(96, 32, 7);
		// Simulate a strided view that opCont would normally fix: nb[1] is
		// larger than the contiguous packing requires.
		t.nb[1] = t.nb[1] + F32_BYTES;
		expect(() =>
			assertContiguousF32(makeWasm(t), 5 as TensorPtr, "q3"),
		).toThrow(/not F32-contiguous/);
	});

	test("rejects nb[2] != nb[1]*ne[1] (the page-stride-gap case)", () => {
		const t = contiguous3d(96, 32, 7);
		t.nb[2] = t.nb[2] * 2;
		t.nb[3] = t.nb[2] * t.ne[2];
		expect(() =>
			assertContiguousF32(makeWasm(t), 6 as TensorPtr, "q3"),
		).toThrow(/not F32-contiguous/);
	});

	test("rejects nb[3] != nb[2]*ne[2] (the slab-stride-gap case)", () => {
		const t = contiguous3d(96, 32, 7);
		t.nb[3] = t.nb[3] + F32_BYTES;
		expect(() =>
			assertContiguousF32(makeWasm(t), 7 as TensorPtr, "q3"),
		).toThrow(/not F32-contiguous/);
	});

	test("error message includes ne[] and nb[] for diagnosis", () => {
		const t = contiguous3d(96, 32, 7);
		t.nb[1] = t.nb[1] + F32_BYTES;
		expect(() =>
			assertContiguousF32(makeWasm(t), 8 as TensorPtr, "buildQKV.fused.k3"),
		).toThrow(/buildQKV\.fused\.k3:.+ne=\[96,32,7\].+nb=\[/);
	});
});
