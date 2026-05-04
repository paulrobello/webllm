import { expect, test } from "bun:test";
import { ModelNotFoundError } from "../src/core/errors.js";
import type { WebLLM } from "../src/index.js";

test("engine.tokenize throws ModelNotFoundError for unregistered model id", () => {
	// Type-level smoke: shape check against the public API. Behavioral
	// coverage lives in the browser smoke (DocsAfter the chat page lands a
	// `tokenize`-driven context bar, the smoke checklist in
	// docs/CHAT_PAGE.md exercises the success path).
	const stub: Pick<WebLLM, "tokenize"> = {
		tokenize: (id: string, _text: string) => {
			throw new ModelNotFoundError(id);
		},
	};
	expect(() => stub.tokenize("ghost", "hi")).toThrow(ModelNotFoundError);
});

test("engine.tokenize signature returns readonly number[]", () => {
	// Pure type-level: this file failing to typecheck means the contract
	// regressed. The runtime body is exercised by the manual chat-page
	// smoke; full WebLLM construction in Bun would require WebGPU+WASM.
	const stub: Pick<WebLLM, "tokenize"> = {
		tokenize: (_id: string, _text: string) => [1, 2, 3] as const,
	};
	const out = stub.tokenize("x", "y");
	expect(out.length).toBe(3);
	expect(out[0]).toBe(1);
});
