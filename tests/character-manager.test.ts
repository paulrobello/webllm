import { describe, expect, test } from "bun:test";
import { CharacterManager } from "../src/characters/character-manager.js";

describe("CharacterManager", () => {
	test("create and get character", () => {
		const mgr = new CharacterManager();
		const char = mgr.create({
			modelId: "test-model",
			systemPrompt: "You are helpful.",
		});
		expect(char.modelId).toBe("test-model");
		expect(mgr.get(char.id)).toBe(char);
	});

	test("getAll returns all characters", () => {
		const mgr = new CharacterManager();
		const c1 = mgr.create({ modelId: "m1", systemPrompt: "P1" });
		const c2 = mgr.create({ modelId: "m2", systemPrompt: "P2" });
		const all = mgr.getAll();
		expect(all.length).toBe(2);
		expect(all).toContain(c1);
		expect(all).toContain(c2);
	});

	test("remove deletes character", async () => {
		const mgr = new CharacterManager();
		const char = mgr.create({ modelId: "m1", systemPrompt: "P1" });
		await mgr.remove(char.id);
		expect(mgr.get(char.id)).toBeUndefined();
		expect(mgr.count).toBe(0);
	});

	test("remove unknown id does nothing", async () => {
		const mgr = new CharacterManager();
		await mgr.remove("nonexistent");
		expect(mgr.count).toBe(0);
	});

	test("stopAll stops all active generations", () => {
		const mgr = new CharacterManager();
		mgr.create({ modelId: "m1", systemPrompt: "P1" });
		mgr.create({ modelId: "m2", systemPrompt: "P2" });
		mgr.stopAll();
		expect(mgr.activeCount).toBe(0);
	});

	test("clear removes all", async () => {
		const mgr = new CharacterManager();
		mgr.create({ modelId: "m1", systemPrompt: "P1" });
		mgr.create({ modelId: "m2", systemPrompt: "P2" });
		await mgr.clear();
		expect(mgr.count).toBe(0);
		expect(mgr.getAll().length).toBe(0);
	});

	test("count and activeCount", () => {
		const mgr = new CharacterManager();
		expect(mgr.count).toBe(0);
		expect(mgr.activeCount).toBe(0);
		mgr.create({ modelId: "m1", systemPrompt: "P1" });
		expect(mgr.count).toBe(1);
		expect(mgr.activeCount).toBe(0);
	});

	test("get returns undefined for unknown ID", () => {
		const mgr = new CharacterManager();
		expect(mgr.get("nope")).toBeUndefined();
	});
});
