import { expect, test } from "bun:test";
import {
	LIVE_EVENT_SCHEMA_VERSION,
	LiveEventStore,
} from "../eval/live-events.js";
import type { SmokeRunRecord } from "../eval/smoke-runs.js";

function makeRecord(overrides: Partial<SmokeRunRecord> = {}): SmokeRunRecord {
	return {
		schemaVersion: 1,
		timestamp: "2026-04-23T22:34:19.000Z",
		model: "qwen3-0.6b-q4f16",
		page: "smoke",
		thinking: "off",
		prompt: "hi",
		params: { contextLength: 4096 },
		...overrides,
	};
}

test("stampEvent assigns monotonic seq and timestamps each event", () => {
	const store = new LiveEventStore();
	const a = store.stampEvent({
		kind: "run_started",
		payload: {
			runId: "r1",
			model: "m",
			page: "smoke",
			thinking: "off",
			prompt: "hi",
		},
	});
	const b = store.stampEvent({
		kind: "run_failed",
		payload: { runId: "r1", model: "m", error: "boom" },
	});
	expect(a.seq).toBe(1);
	expect(b.seq).toBe(2);
	expect(typeof a.timestamp).toBe("string");
	expect(store.currentSeq()).toBe(2);
});

test("run_complete events populate the completed-runs snapshot", () => {
	const store = new LiveEventStore();
	store.stampEvent({
		kind: "run_complete",
		payload: { ...makeRecord(), runId: "r1" },
	});
	store.stampEvent({
		kind: "run_complete",
		payload: {
			...makeRecord({
				profile: "qwen3-0.6b-thinking-warm",
				thinking: "on",
				timestamp: "2026-04-23T22:34:20.000Z",
			}),
			runId: "r2",
		},
	});
	expect(store.completedRuns()).toHaveLength(2);
	const snap = store.snapshot();
	expect(snap.kind).toBe("snapshot");
	expect(snap.payload.schemaVersion).toBe(LIVE_EVENT_SCHEMA_VERSION);
	expect(snap.payload.runs.map((r) => r.runId)).toEqual(["r1", "r2"]);
});

test("reset clears completed runs but keeps the seq counter advancing", () => {
	const store = new LiveEventStore();
	store.stampEvent({
		kind: "run_complete",
		payload: { ...makeRecord(), runId: "r1" },
	});
	expect(store.completedRuns()).toHaveLength(1);
	store.stampEvent({ kind: "reset", payload: { reason: "rotating" } });
	expect(store.completedRuns()).toHaveLength(0);
	expect(store.currentSeq()).toBe(2);
});

test("eventsSince returns only events after the given seq", () => {
	const store = new LiveEventStore();
	for (let i = 0; i < 5; i++) {
		store.stampEvent({
			kind: "run_complete",
			payload: { ...makeRecord(), runId: `r${i}` },
		});
	}
	expect(store.eventsSince(2).map((e) => e.seq)).toEqual([3, 4, 5]);
	expect(store.eventsSince(99)).toHaveLength(0);
});

test("event ring buffer drops oldest entries past maxEventBuffer", () => {
	const store = new LiveEventStore(3);
	for (let i = 0; i < 10; i++) {
		store.stampEvent({
			kind: "run_complete",
			payload: { ...makeRecord(), runId: `r${i}` },
		});
	}
	const since0 = store.eventsSince(0);
	expect(since0).toHaveLength(3);
	expect(since0.map((e) => e.seq)).toEqual([8, 9, 10]);
});
