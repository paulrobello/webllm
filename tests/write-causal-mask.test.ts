import { describe, expect, test } from "bun:test";
import { writeCausalMaskF16 } from "../src/inference/model-inference.js";

const VISIBLE = 0x0000;
const MASKED = 0xfc00;

function asGrid(
	view: Uint16Array,
	totalLen: number,
	maskPaddedCols: number,
): number[][] {
	const grid: number[][] = [];
	for (let q = 0; q < maskPaddedCols; q++) {
		const row: number[] = [];
		for (let k = 0; k < totalLen; k++) row.push(view[q * totalLen + k]);
		grid.push(row);
	}
	return grid;
}

describe("writeCausalMaskF16 — full causal (no swaWindow)", () => {
	test("prefill nTokens=4, pastLen=0 → standard lower-triangular causal mask", () => {
		const totalLen = 4;
		const nTokens = 4;
		const maskPaddedCols = 32;
		const view = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(view, totalLen, nTokens, 0, maskPaddedCols);

		const grid = asGrid(view, totalLen, maskPaddedCols);
		// q=0 sees only k=0; q=1 sees k=0,1; etc.
		expect(grid[0]).toEqual([VISIBLE, MASKED, MASKED, MASKED]);
		expect(grid[1]).toEqual([VISIBLE, VISIBLE, MASKED, MASKED]);
		expect(grid[2]).toEqual([VISIBLE, VISIBLE, VISIBLE, MASKED]);
		expect(grid[3]).toEqual([VISIBLE, VISIBLE, VISIBLE, VISIBLE]);
	});

	test("prefill with pastLen=2, nTokens=3, totalLen=5", () => {
		const totalLen = 5;
		const nTokens = 3;
		const pastLen = 2;
		const maskPaddedCols = 32;
		const view = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(view, totalLen, nTokens, pastLen, maskPaddedCols);

		const grid = asGrid(view, totalLen, maskPaddedCols);
		// q=0 sits at absolute position 2, sees k=0..2
		expect(grid[0]).toEqual([VISIBLE, VISIBLE, VISIBLE, MASKED, MASKED]);
		expect(grid[1]).toEqual([VISIBLE, VISIBLE, VISIBLE, VISIBLE, MASKED]);
		expect(grid[2]).toEqual([VISIBLE, VISIBLE, VISIBLE, VISIBLE, VISIBLE]);
	});

	test("padding rows [nTokens, maskPaddedCols) are filled with 0", () => {
		const totalLen = 3;
		const nTokens = 2;
		const maskPaddedCols = 4;
		const view = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(view, totalLen, nTokens, 0, maskPaddedCols);

		const grid = asGrid(view, totalLen, maskPaddedCols);
		expect(grid[2]).toEqual([VISIBLE, VISIBLE, VISIBLE]);
		expect(grid[3]).toEqual([VISIBLE, VISIBLE, VISIBLE]);
	});
});

describe("writeCausalMaskF16 — banded SWA mask", () => {
	test("swaWindow=2, prefill nTokens=4, pastLen=0 → 2-wide band along the diagonal", () => {
		const totalLen = 4;
		const nTokens = 4;
		const maskPaddedCols = 32;
		const view = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(view, totalLen, nTokens, 0, maskPaddedCols, 2);

		const grid = asGrid(view, totalLen, maskPaddedCols);
		// Window of size 2: visible iff (q - 2 < k <= q), i.e. {q-1, q}.
		expect(grid[0]).toEqual([VISIBLE, MASKED, MASKED, MASKED]);
		expect(grid[1]).toEqual([VISIBLE, VISIBLE, MASKED, MASKED]);
		expect(grid[2]).toEqual([MASKED, VISIBLE, VISIBLE, MASKED]);
		expect(grid[3]).toEqual([MASKED, MASKED, VISIBLE, VISIBLE]);
	});

	test("swaWindow=3, decode-step (nTokens=1) at pastLen=5 → look-back covers 3 keys", () => {
		const totalLen = 6;
		const nTokens = 1;
		const pastLen = 5;
		const maskPaddedCols = 32;
		const view = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(view, totalLen, nTokens, pastLen, maskPaddedCols, 3);

		// Single query at absolute position 5; window of 3 → visible {3,4,5}.
		const grid = asGrid(view, totalLen, maskPaddedCols);
		expect(grid[0]).toEqual([
			MASKED,
			MASKED,
			MASKED,
			VISIBLE,
			VISIBLE,
			VISIBLE,
		]);
	});

	test("swaWindow=4, pastLen=2, nTokens=3 → band shifts with each query position", () => {
		const totalLen = 5;
		const nTokens = 3;
		const pastLen = 2;
		const maskPaddedCols = 32;
		const view = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(view, totalLen, nTokens, pastLen, maskPaddedCols, 4);

		const grid = asGrid(view, totalLen, maskPaddedCols);
		// q=0 at abs=2, window 4 → visible [-1, 2] ∩ [0, totalLen) = {0,1,2}
		expect(grid[0]).toEqual([VISIBLE, VISIBLE, VISIBLE, MASKED, MASKED]);
		// q=1 at abs=3, window 4 → visible [0, 3] = {0,1,2,3}
		expect(grid[1]).toEqual([VISIBLE, VISIBLE, VISIBLE, VISIBLE, MASKED]);
		// q=2 at abs=4, window 4 → visible [1, 4] = {1,2,3,4}
		expect(grid[2]).toEqual([MASKED, VISIBLE, VISIBLE, VISIBLE, VISIBLE]);
	});

	test("swaWindow=Infinity collapses to full causal (no-op window)", () => {
		const totalLen = 4;
		const nTokens = 3;
		const maskPaddedCols = 32;
		const causalView = new Uint16Array(totalLen * maskPaddedCols);
		const swaInfView = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(causalView, totalLen, nTokens, 0, maskPaddedCols);
		writeCausalMaskF16(
			swaInfView,
			totalLen,
			nTokens,
			0,
			maskPaddedCols,
			Number.POSITIVE_INFINITY,
		);
		expect(swaInfView).toEqual(causalView);
	});

	test("swaWindow=0 collapses to full causal (defensive guard)", () => {
		const totalLen = 4;
		const nTokens = 3;
		const maskPaddedCols = 32;
		const causalView = new Uint16Array(totalLen * maskPaddedCols);
		const swaZeroView = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(causalView, totalLen, nTokens, 0, maskPaddedCols);
		writeCausalMaskF16(swaZeroView, totalLen, nTokens, 0, maskPaddedCols, 0);
		expect(swaZeroView).toEqual(causalView);
	});
});

describe("writeCausalMaskF16 — global+SWA parity at large window", () => {
	test("when swaWindow >= totalLen the SWA mask is bit-identical to the global mask", () => {
		const totalLen = 16;
		const nTokens = 8;
		const pastLen = 4;
		const maskPaddedCols = 32;
		const causalView = new Uint16Array(totalLen * maskPaddedCols);
		const swaView = new Uint16Array(totalLen * maskPaddedCols);
		writeCausalMaskF16(causalView, totalLen, nTokens, pastLen, maskPaddedCols);
		writeCausalMaskF16(
			swaView,
			totalLen,
			nTokens,
			pastLen,
			maskPaddedCols,
			totalLen + 8, // window wider than the sequence
		);
		expect(swaView).toEqual(causalView);
	});
});
