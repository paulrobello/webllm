import { expect, test } from "bun:test";
import { contextBarState, formatContext } from "../smoke-test/chat-metrics.js";

test("contextBarState classifies neutral / amber / red", () => {
	expect(contextBarState(0, 4096)).toBe("neutral");
	expect(contextBarState(3000, 4096)).toBe("neutral");
	expect(contextBarState(3277, 4096)).toBe("amber"); // 80%
	expect(contextBarState(3891, 4096)).toBe("red"); // 95%
	expect(contextBarState(4096, 4096)).toBe("red");
});

test("formatContext renders 'used / max — pct%'", () => {
	expect(formatContext(1842, 4096)).toBe("1,842 / 4,096 — 45%");
	expect(formatContext(0, 4096)).toBe("0 / 4,096 — 0%");
});
