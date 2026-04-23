import { expect, test } from "bun:test";
import { buildSmokeTestUrl } from "../eval/browser-smoke.js";

test("buildSmokeTestUrl can target the debug smoke page", () => {
	expect(
		buildSmokeTestUrl("qwen3-0.6b-q4f16", 4096, {
			page: "debug",
			extraParams: {
				chatSmoke: 123,
			},
		}),
	).toBe(
		"http://localhost:8031/real-model-debug.html?model=qwen3-0.6b-q4f16&ctx=4096&chatSmoke=123",
	);
});
