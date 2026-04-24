import { expect, test } from "bun:test";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	buildSmokeRunRecord,
	DEFAULT_SMOKE_RUNS_DIR,
	resolveSmokeRunsDir,
	SMOKE_RUN_SCHEMA_VERSION,
	SMOKE_RUNS_DIR_ENV,
	smokeRunFileName,
	writeSmokeRunRecord,
} from "../eval/smoke-runs.js";

test("buildSmokeRunRecord pins schemaVersion and includes both sub-records when given", () => {
	const timestamp = new Date("2026-04-23T22:34:19.000Z");
	const record = buildSmokeRunRecord({
		profile: {
			name: "qwen3-0.6b-thinking-warm",
			model: "qwen3-0.6b-q4f16",
			thinking: "on",
			maxTokens: 1024,
		},
		modelId: "qwen3-0.6b-q4f16",
		page: "smoke",
		thinking: "on",
		prompt: "Tell one short joke.",
		contextLength: 4096,
		params: { maxTokens: 1024, temperature: 0.6 },
		oneShotResult: {
			tokensGenerated: 267,
			prefillMs: 128,
			decodeMs: 15396,
			totalMs: 15524,
			tokensPerSecond: 17.3,
			completionPageMs: 99999,
			finishReason: "eos",
		},
		oneShotAssistantText: "A joke.",
		oneShotFinishReason: "eos",
		interactiveResult: {
			assistantText: "Hi.",
			finishReason: "eos",
			metrics: { genTokens: 11, tokensPerSecond: 15.5, totalMs: 830 },
		},
		timestamp,
	});
	expect(record.schemaVersion).toBe(SMOKE_RUN_SCHEMA_VERSION);
	expect(record.timestamp).toBe("2026-04-23T22:34:19.000Z");
	expect(record.profile).toBe("qwen3-0.6b-thinking-warm");
	expect(record.model).toBe("qwen3-0.6b-q4f16");
	expect(record.thinking).toBe("on");
	expect(record.params).toEqual({
		contextLength: 4096,
		maxTokens: 1024,
		temperature: 0.6,
	});
	expect(record.oneShot).toEqual({
		assistantText: "A joke.",
		finishReason: "eos",
		genTokens: 267,
		prefillMs: 128,
		decodeMs: 15396,
		totalMs: 15524,
		tokensPerSecond: 17.3,
	});
	expect(record.interactive).toEqual({
		assistantText: "Hi.",
		finishReason: "eos",
		genTokens: 11,
		tokensPerSecond: 15.5,
		totalMs: 830,
	});
});

test("buildSmokeRunRecord omits profile field when no profile passed", () => {
	const record = buildSmokeRunRecord({
		profile: null,
		modelId: "llama-3.2-1b-q4f16",
		page: "debug",
		thinking: "off",
		prompt: "hi",
		contextLength: 2048,
		params: {},
	});
	expect(record.profile).toBeUndefined();
	expect(record.model).toBe("llama-3.2-1b-q4f16");
	expect(record.oneShot).toBeUndefined();
	expect(record.interactive).toBeUndefined();
});

test("smokeRunFileName prefixes with epoch ms for chronological sorting", () => {
	const a = smokeRunFileName({
		timestamp: "2026-04-23T22:34:19.000Z",
		profile: "qwen3-0.6b-thinking-warm",
		model: "qwen3-0.6b-q4f16",
	});
	const b = smokeRunFileName({
		timestamp: "2026-04-23T22:34:20.000Z",
		profile: "qwen3-0.6b-thinking-warm",
		model: "qwen3-0.6b-q4f16",
	});
	expect(a.endsWith("-qwen3-0.6b-thinking-warm.json")).toBe(true);
	expect(b.endsWith("-qwen3-0.6b-thinking-warm.json")).toBe(true);
	expect(a < b).toBe(true);
});

test("smokeRunFileName falls back to model id when no profile", () => {
	const name = smokeRunFileName({
		timestamp: "2026-04-23T22:34:19.000Z",
		profile: undefined,
		model: "llama-3.2-1b-q4f16",
	});
	expect(name.endsWith("-llama-3.2-1b-q4f16.json")).toBe(true);
});

test("resolveSmokeRunsDir honors arg > env > default", () => {
	const prev = process.env[SMOKE_RUNS_DIR_ENV];
	try {
		delete process.env[SMOKE_RUNS_DIR_ENV];
		expect(resolveSmokeRunsDir()).toBe(DEFAULT_SMOKE_RUNS_DIR);
		process.env[SMOKE_RUNS_DIR_ENV] = "/tmp/from-env";
		expect(resolveSmokeRunsDir()).toBe("/tmp/from-env");
		expect(resolveSmokeRunsDir("/tmp/explicit")).toBe("/tmp/explicit");
	} finally {
		if (prev === undefined) delete process.env[SMOKE_RUNS_DIR_ENV];
		else process.env[SMOKE_RUNS_DIR_ENV] = prev;
	}
});

test("writeSmokeRunRecord persists a JSON file matching the record", () => {
	const dir = join(tmpdir(), `webllm-smoke-runs-${Date.now()}`);
	try {
		const record = buildSmokeRunRecord({
			profile: null,
			modelId: "qwen3-0.6b-q4f16",
			page: "smoke",
			thinking: "off",
			prompt: "hi",
			contextLength: 4096,
			params: {},
			timestamp: new Date("2026-04-23T22:34:19.000Z"),
		});
		const path = writeSmokeRunRecord(record, dir);
		expect(path.startsWith(dir)).toBe(true);
		const onDisk = JSON.parse(readFileSync(path, "utf-8"));
		expect(onDisk.schemaVersion).toBe(SMOKE_RUN_SCHEMA_VERSION);
		expect(onDisk.model).toBe("qwen3-0.6b-q4f16");
		expect(onDisk.timestamp).toBe("2026-04-23T22:34:19.000Z");
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
