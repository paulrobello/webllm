import { describe, expect, test } from "bun:test";
import { GameLoop } from "../src/core/game-loop.js";

describe("GameLoop", () => {
	test("start calls callback", async () => {
		const loop = new GameLoop({ frameBudgetMs: 8, targetFps: 60 });
		let called = false;
		loop.start(() => {
			called = true;
		});
		await new Promise((r) => setTimeout(r, 50));
		loop.stop();
		expect(called).toBe(true);
	});

	test("stop prevents further callbacks", async () => {
		const loop = new GameLoop({ frameBudgetMs: 8, targetFps: 60 });
		let count = 0;
		loop.start(() => {
			count++;
		});
		await new Promise((r) => setTimeout(r, 80));
		loop.stop();
		const countAtStop = count;
		await new Promise((r) => setTimeout(r, 50));
		expect(count).toBe(countAtStop);
	});

	test("pause stops callbacks, resume restarts", async () => {
		const loop = new GameLoop({ frameBudgetMs: 8, targetFps: 60 });
		let count = 0;
		loop.start(() => {
			count++;
		});
		await new Promise((r) => setTimeout(r, 50));
		loop.pause();
		const pausedCount = count;
		await new Promise((r) => setTimeout(r, 50));
		expect(count).toBe(pausedCount);
		loop.resume();
		await new Promise((r) => setTimeout(r, 50));
		expect(count).toBeGreaterThan(pausedCount);
		loop.stop();
	});

	test("setFrameBudget updates config", async () => {
		const loop = new GameLoop({ frameBudgetMs: 8, targetFps: 60 });
		let receivedBudget = 0;
		loop.start((_d, budget) => {
			receivedBudget = budget;
		});
		await new Promise((r) => setTimeout(r, 30));
		loop.setFrameBudget(16);
		await new Promise((r) => setTimeout(r, 30));
		expect(receivedBudget).toBe(16);
		loop.stop();
	});

	test("isRunning and isPaused reflect state", () => {
		const loop = new GameLoop({
			frameBudgetMs: 8,
			targetFps: 60,
			paused: true,
		});
		expect(loop.isRunning).toBe(false);
		expect(loop.isPaused).toBe(true);
		loop.start(() => {});
		expect(loop.isRunning).toBe(true);
		expect(loop.isPaused).toBe(true);
		loop.resume();
		expect(loop.isPaused).toBe(false);
		loop.stop();
		expect(loop.isRunning).toBe(false);
	});

	test("frameCount increments", async () => {
		const loop = new GameLoop({ frameBudgetMs: 8, targetFps: 60 });
		loop.start(() => {});
		await new Promise((r) => setTimeout(r, 80));
		expect(loop.frameCount).toBeGreaterThan(0);
		loop.stop();
	});
});
