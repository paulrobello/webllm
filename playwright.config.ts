// Playwright config for `make test-browser` — the automated browser regression
// lane. Local-only (needs a real GPU + built smoke artifacts, `make smoke-test`
// first). NOT part of checkall/CI.
//
// Headless mode: probed 2026-07-14. Headless DOES acquire a WebGPU adapter, but
// it is google/swiftshader (software/CPU renderer) — it would not exercise the
// real GPU inference path this gate protects and risks the 180s timeout on
// model load. Headed acquires apple/metal-3 (the real M4 Max GPU), the same
// adapter the manual agentchrome lane uses. So headed is the default. Flip to
// headless (SwiftShader) for re-probing via WEBLLM_TEST_BROWSER_HEADLESS=1.
import { defineConfig, devices } from "@playwright/test";

const headless = process.env.WEBLLM_TEST_BROWSER_HEADLESS
	? ["1", "true", "yes", "on"].includes(process.env.WEBLLM_TEST_BROWSER_HEADLESS.toLowerCase())
	: false;

export default defineConfig({
	testDir: "./tests-browser",
	testMatch: /.*\.spec\.ts/,
	timeout: 180_000,
	fullyParallel: false,
	workers: 1,
	retries: 0,
	reporter: [["list"]],
	use: {
		headless,
		launchOptions: {
			args: ["--enable-unsafe-webgpu"],
		},
	},
	projects: [
		{
			name: "chromium",
			use: { ...devices["Desktop Chrome"] },
		},
	],
	webServer: {
		command: "bun run eval/smoke-serve.ts --port 8034",
		url: "http://127.0.0.1:8034",
		reuseExistingServer: !process.env.CI,
		timeout: 60_000,
	},
});
