// Automated browser regression lane. Run via `make test-browser` (local-only;
// requires `make smoke-test` artifacts + a real GPU). NOT part of checkall/CI.
//
// Asserts the existing smoke page (smoke-test/real-model.html) generates text
// end-to-end with the smallest registered chat model and surfaces any uncaught
// errors or unexpected console errors. Automates the manual agentchrome workflow
// described in CLAUDE.md without replacing agentchrome for interactive debugging.
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

const MODEL_ID = "smollm2-360m-q4f16";
const MODEL_PATH = resolve("smoke-test/models", `${MODEL_ID}.gguf`);
const BASE_URL = "http://127.0.0.1:8034/real-model.html";

// `ctx` is intentionally omitted: smoke-test/real-model-page.js falls back to
// DEFAULT_CONTEXT_LENGTH when the query param is absent (verified in the page
// source). `ingest=off` prevents the page from posting run_complete to the
// dashboard at :8033 (per CLAUDE.md). `v` cache-busts the page + its imports.
const buildUrl = (): string =>
	`${BASE_URL}?model=${MODEL_ID}&ingest=off&v=${Date.now()}`;

// Benign console-error allowlist: error-type console messages that do NOT fail
// the lane. Derived from `grep -rh "console.error(" src smoke-test/webllm-bundle.js`
// — every console.error in the codebase is an internal-dispatch invariant check
// (dispatchMatmul / dispatchRmsNorm / dispatchSetRows) that must NEVER fire in a
// healthy run, so none belong here. One allowlisted substring:
//   - "adapter_info:" — WASM backend informational line. Actually emitted via
//     console.info (src/inference/ggml-wasm.ts:243), so it never reaches the
//     error-type filter; allowlisted defensively. CLAUDE.md calls adapter_info
//     benign informational output.
//
// NOTE: the favicon 404 ("Failed to load resource: the server responded with a
// status of 404") is NOT allowlisted here. Chrome auto-requests /favicon.ico for
// real-model.html (which declares no <link rel="icon">); rather than allowlist
// Chrome's generic 404 text (which would mask ANY future post-success 404), we
// intercept the request below with a 204 route handler so no 404 reaches the
// console. This keeps the allowlist strict — any 404 that does surface is real.
const BENIGN_ERROR_SUBSTRINGS = ["adapter_info:"];

// Markers that indicate a fetch/load failure in the early [1/8]/[2/8] steps —
// if any appear in #log during the model-present guard, the lane fast-fails
// with an actionable message instead of waiting the full 180s timeout.
//   - "[2/8] Fetch failed:" — the page's explicit fetch-fail log line.
//   - "Failed to fetch" — the browser's network-error text; covers a [1/8] WASM
//     or model fetch that fails before the page can prefix it with [2/8].
const FETCH_FAIL_MARKERS = ["[2/8] Fetch failed:", "Failed to fetch"];

test("smoke page generates text end-to-end", async ({ page }) => {
	test.info().annotations.push({
		type: "model-weights",
		description: `${MODEL_PATH} present=${existsSync(MODEL_PATH)}`,
	});

	const consoleMessages: { type: string; text: string }[] = [];
	const pageErrors: string[] = [];
	page.on("console", (msg) => {
		consoleMessages.push({ type: msg.type(), text: msg.text() });
	});
	page.on("pageerror", (err) => {
		// Include the stack (not just name+message) so future failures are diagnosable.
		pageErrors.push(err.stack ?? `${err.name}: ${err.message}`);
	});

	// Intercept the browser's automatic /favicon.ico request (real-model.html
	// declares no <link rel="icon">) so it never 404s into the console — see
	// BENIGN_ERROR_SUBSTRINGS note above. Must be set before navigation.
	await page.route("**/favicon.ico", (route) =>
		route.fulfill({ status: 204, body: "" }),
	);

	await page.goto(buildUrl(), {
		waitUntil: "domcontentloaded",
		timeout: 30_000,
	});

	// Model-present guard: scan #log for ~20s. If the [2/8] fetch step failed
	// (missing weights / 404), throw an actionable message instead of waiting
	// the full 180s for a success marker that will never arrive. The page logs
	// failures via log("fail", ...) -> `<div class="step fail">`.
	const guardDeadline = Date.now() + 20_000;
	while (Date.now() < guardDeadline) {
		const logText =
			(await page
				.locator("#log")
				.textContent({ timeout: 500 })
				.catch(() => "")) ?? "";
		if (FETCH_FAIL_MARKERS.some((m) => logText.includes(m))) {
			throw new Error(
				"Model weights missing — run 'make smoke-test' (and ensure " +
					`smoke-test/models/${MODEL_ID}.gguf is present) before 'make test-browser'.`,
			);
		}
		if (logText.includes(" tok/s, finish=")) {
			break; // generation already succeeded; skip to final assertions
		}
		await page.waitForTimeout(500);
	}

	// Success: the [7/8] generation summary line. This substring uniquely
	// identifies completed generation (no other step logs "tok/s, finish=").
	// (The [8/8] step is a SEPARATE reference-encoder check that loads a second
	// engine — not the generation-success signal.)
	await expect(page.locator("#log")).toContainText(" tok/s, finish=", {
		timeout: 180_000,
	});

	// No step failed (any `<div class="...fail...">` in #log is a regression).
	await expect(page.locator("#log .fail")).toHaveCount(0);

	// No uncaught page errors — these always fail the lane.
	expect(
		pageErrors,
		`uncaught page errors: ${JSON.stringify(pageErrors)}`,
	).toEqual([]);

	// No unexpected console.error messages (allowlist documented above).
	const offenders = consoleMessages
		.filter((m) => m.type === "error")
		.filter((m) => !BENIGN_ERROR_SUBSTRINGS.some((p) => m.text.includes(p)));
	expect(
		offenders,
		`unexpected console.error messages: ${JSON.stringify(offenders)}`,
	).toEqual([]);
});
