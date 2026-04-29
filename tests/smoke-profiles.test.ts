import { expect, test } from "bun:test";
import { getModelById } from "../eval/models.js";
import {
	getSmokeProfile,
	getSmokeProfileSet,
	listSmokeProfileSets,
	listSmokeProfiles,
	profileToUrlParams,
	resolveProfileModel,
	SMOKE_PROFILE_SETS,
	SMOKE_PROFILES,
} from "../eval/smoke-profiles.js";

test("every profile declares a known model id; non-embedding profiles also carry a prompt", () => {
	expect(SMOKE_PROFILES.length).toBeGreaterThan(0);
	for (const profile of SMOKE_PROFILES) {
		expect(profile.name).toMatch(/^[a-z0-9][a-z0-9-_.]*$/);
		expect(profile.model.length).toBeGreaterThan(0);
		expect(resolveProfileModel(profile)).toBeDefined();
		if (profile.embedding) {
			// Embedding profiles drive the encoder path, which takes per-task
			// inputs from the eval suite — there is no single global prompt.
			expect(profile.prompt).toBeUndefined();
		} else {
			expect(profile.prompt?.length ?? 0).toBeGreaterThan(0);
		}
	}
});

test("embedding profiles point at embedding-capable models", () => {
	const embeddingProfiles = SMOKE_PROFILES.filter((p) => p.embedding === true);
	expect(embeddingProfiles.length).toBeGreaterThan(0);
	for (const profile of embeddingProfiles) {
		const model = resolveProfileModel(profile);
		expect(model?.capabilities?.embedding).toBe(true);
	}
});

test("the `embeddings` profile set lists every embedding profile and only those", () => {
	const set = getSmokeProfileSet("embeddings");
	expect(set).toBeDefined();
	const fromFlag = SMOKE_PROFILES.filter((p) => p.embedding === true).map(
		(p) => p.name,
	);
	expect([...(set ?? [])].sort()).toEqual(fromFlag.sort());
});

test("the `full` profile set includes the embedding profiles", () => {
	const full = getSmokeProfileSet("full");
	expect(full).toBeDefined();
	expect(full).toContain("arctic-embed-s");
	expect(full).toContain("arctic-embed-m");
});

test("profile names are unique", () => {
	const names = SMOKE_PROFILES.map((p) => p.name);
	expect(new Set(names).size).toBe(names.length);
});

test("getSmokeProfile resolves by name and returns undefined for unknown", () => {
	const profile = getSmokeProfile("qwen3-0.6b-thinking-warm");
	expect(profile?.model).toBe("qwen3-0.6b-q4f16");
	expect(profile?.thinking).toBe("on");
	expect(getSmokeProfile("does-not-exist")).toBeUndefined();
	expect(listSmokeProfiles()).toContain("qwen3-0.6b-off-warm");
});

test("profileToUrlParams only emits fields that were actually set", () => {
	expect(
		profileToUrlParams({
			name: "x",
			model: "qwen3-0.6b-q4f16",
		}),
	).toEqual({});

	expect(
		profileToUrlParams({
			name: "x",
			model: "qwen3-0.6b-q4f16",
			thinking: "on",
			temperature: 0.6,
			topK: 20,
			topP: 0.95,
			repetitionPenalty: 1.05,
			maxTokens: 512,
			contextLength: 4096,
			seed: 7,
			prompt: "hi",
		}),
	).toEqual({
		thinking: 1,
		temp: 0.6,
		topK: 20,
		topP: 0.95,
		rep: 1.05,
		max: 512,
		ctx: 4096,
		seed: 7,
		prompt: "hi",
	});

	// thinking=off is the default and should NOT be emitted
	expect(
		profileToUrlParams({
			name: "x",
			model: "qwen3-0.6b-q4f16",
			thinking: "off",
		}),
	).toEqual({});
});

test("profileToUrlParams auto-routes >3.5 GiB models to wasm64", () => {
	// Phase 6 dual-binary routing: profileToUrlParams must inject
	// wasm=mem64 for any profile whose resolved model has vramMB > 3500,
	// matching pickWasmUrl in src/core/engine.ts. Otherwise
	// bench-browser-eval silently falls back to wasm32 and OOMs on
	// >4 GiB models. Asserts via real registered models so the boundary
	// is checked end-to-end through resolveProfileModel().
	const ensure = (id: string, predicate: (m: number) => boolean) => {
		const m = getModelById(id);
		expect(m).toBeDefined();
		expect(predicate(m?.vramMB ?? -1)).toBe(true);
		return m as { vramMB: number };
	};

	// Below threshold (~1.4 GiB).
	ensure("qwen3-0.6b-q4f16", (v) => v <= 3500);
	expect(profileToUrlParams({ name: "x", model: "qwen3-0.6b-q4f16" })).toEqual(
		{},
	);

	// Exactly at the 3500 boundary — the rule is `>` not `>=`, so this
	// should NOT trigger wasm64.
	ensure("mistral-7b-instruct-v0.3-q3km", (v) => v === 3500);
	expect(
		profileToUrlParams({
			name: "x",
			model: "mistral-7b-instruct-v0.3-q3km",
		}),
	).toEqual({});

	// Just above the 3500 boundary (4400 MB → 3501 covered).
	ensure("mistral-7b-instruct-v0.3-q4ks", (v) => v > 3500);
	expect(
		profileToUrlParams({
			name: "x",
			model: "mistral-7b-instruct-v0.3-q4ks",
		}),
	).toEqual({ wasm: "mem64" });

	// Well above boundary (5400 MB).
	ensure("mistral-7b-instruct-v0.3-q5km", (v) => v >= 5400);
	expect(
		profileToUrlParams({
			name: "x",
			model: "mistral-7b-instruct-v0.3-q5km",
		}),
	).toEqual({ wasm: "mem64" });
});

test("profile sets reference only known profile names", () => {
	expect(listSmokeProfileSets()).toContain("llama-vs-qwen");
	expect(listSmokeProfileSets()).toContain("full");
	for (const [setName, members] of Object.entries(SMOKE_PROFILE_SETS)) {
		expect(members.length).toBeGreaterThan(0);
		for (const memberName of members) {
			expect(getSmokeProfile(memberName)).toBeDefined();
		}
		expect(getSmokeProfileSet(setName)).toEqual(members);
	}
});
