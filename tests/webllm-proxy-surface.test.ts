import { describe, expect, test } from "bun:test";
import { WebLLM } from "../src/core/engine.js";
import { WebLLMProxy } from "../src/core/webllm-proxy.js";

// The proxy must mirror the public methods on WebLLM. New methods on
// WebLLM that aren't mirrored here cause this test to fail loudly.
//
// `chat` is intentionally proxied even though most consumers use
// `chatCompletion` — keeps API parity for the smaller "drop-in" set.
const PROXIED_METHODS: ReadonlyArray<keyof WebLLMProxy & string> = [
	"loadModelFromBuffer",
	"unloadModel",
	"embed",
	"chat",
	"chatCompletion",
	"generateStream",
	"createConversation",
	"disposeConversation",
	"forkConversation",
	"dispose",
];

describe("WebLLMProxy — surface mirror sentinel", () => {
	test("every proxied method exists on WebLLM", () => {
		for (const name of PROXIED_METHODS) {
			expect(
				typeof (WebLLM.prototype as unknown as Record<string, unknown>)[name],
			).toBe("function");
		}
	});

	test("WebLLMProxy.prototype carries dispose (declared methods land on the prototype)", () => {
		// The proxy uses arrow-class-field syntax for most methods, so they
		// live on instances rather than the prototype. `dispose` is declared
		// as a regular `async` method; that one IS on the prototype.
		expect(
			typeof (WebLLMProxy.prototype as unknown as Record<string, unknown>)
				.dispose,
		).toBe("function");
	});
});
