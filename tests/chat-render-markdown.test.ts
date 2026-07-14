// Stub the marked + DOMPurify globals so renderMarkdown's main (render +
// sanitize) path runs in Bun. DOMPurify is a passthrough here — this unit
// test checks rendering, not sanitization (the XSS guarantee is verified in
// the browser workflow).
// @ts-expect-error — extending globalThis for the test.
globalThis.marked = (s) =>
	`<pre><code>${s.replace(/^```\n?|\n?```$/g, "")}</code></pre>`;
// @ts-expect-error — extending globalThis for the test.
globalThis.DOMPurify = { sanitize: (html) => html };

import { expect, test } from "bun:test";
import { renderMarkdown, splitThinking } from "../smoke-test/chat-render.js";

test("splitThinking separates <think>...</think> from the visible answer", () => {
	const raw = "<think>internal monologue</think>The answer is 42.";
	const { thinking, answer } = splitThinking(raw);
	expect(thinking).toBe("internal monologue");
	expect(answer).toBe("The answer is 42.");
});

test("splitThinking returns empty thinking when absent", () => {
	const { thinking, answer } = splitThinking("just an answer");
	expect(thinking).toBe("");
	expect(answer).toBe("just an answer");
});

test("splitThinking handles unclosed <think> as still-thinking (no answer)", () => {
	const { thinking, answer } = splitThinking("<think>still working");
	expect(thinking).toBe("still working");
	expect(answer).toBe("");
});

test("renderMarkdown wraps fenced code in <pre><code>", () => {
	const html = renderMarkdown("```\nlet x = 1;\n```");
	expect(html).toContain("<pre>");
	expect(html).toContain("<code");
	expect(html).toContain("let x = 1;");
});
