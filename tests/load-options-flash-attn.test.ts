import { expect, test } from "bun:test";
import type { ModelLoadOptions } from "../src/core/types.js";

test("ModelLoadOptions accepts flashAttn", () => {
	// Type-level smoke: this file failing to typecheck means the field
	// regressed off ModelLoadOptions and consumers like smoke-test/chat-models.js
	// (which pass `{ flashAttn: true }` to loadModelFromBuffer) silently drop it.
	const opts: Partial<ModelLoadOptions> = { flashAttn: true };
	expect(opts.flashAttn).toBe(true);
});
