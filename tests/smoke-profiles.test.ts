import { expect, test } from "bun:test";
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

test("every profile declares a known model id and a non-empty prompt", () => {
	expect(SMOKE_PROFILES.length).toBeGreaterThan(0);
	for (const profile of SMOKE_PROFILES) {
		expect(profile.name).toMatch(/^[a-z0-9][a-z0-9-_.]*$/);
		expect(profile.model.length).toBeGreaterThan(0);
		expect(profile.prompt?.length ?? 0).toBeGreaterThan(0);
		expect(resolveProfileModel(profile)).toBeDefined();
	}
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
