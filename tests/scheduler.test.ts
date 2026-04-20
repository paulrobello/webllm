import { describe, expect, test } from "bun:test";
import { type ScheduledTask, Scheduler } from "../src/core/scheduler.js";

function makeTask(
	id: string,
	priority: number,
	modelId = "model-a",
	execute?: (signal: AbortSignal) => Promise<void>,
): ScheduledTask {
	return {
		id,
		priority,
		modelId,
		execute:
			execute ??
			(async () => {
				/* no-op */
			}),
	};
}

describe("Scheduler", () => {
	test("executes highest priority task first", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		const executed: number[] = [];
		scheduler.enqueue(
			makeTask("low", 2, "model-a", async () => {
				executed.push(2);
			}),
		);
		scheduler.enqueue(
			makeTask("high", 0, "model-a", async () => {
				executed.push(0);
			}),
		);
		scheduler.enqueue(
			makeTask("mid", 1, "model-a", async () => {
				executed.push(1);
			}),
		);
		await scheduler.runFrame();
		expect(executed).toEqual([0, 1, 2]);
	});

	test("respects frame budget and stops when exceeded", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 1 });
		let callCount = 0;
		for (let i = 0; i < 10; i++) {
			scheduler.enqueue(
				makeTask(`task-${i}`, 0, "model-a", async () => {
					callCount++;
					const start = performance.now();
					while (performance.now() - start < 0.5) {
						/* spin */
					}
				}),
			);
		}
		await scheduler.runFrame();
		expect(callCount).toBeLessThan(10);
	});

	test("preempts lower priority when higher arrives", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		const executed: string[] = [];
		scheduler.enqueue(
			makeTask("background", 2, "model-a", async () => {
				executed.push("background");
			}),
		);
		scheduler.enqueue(
			makeTask("urgent", 0, "model-a", async () => {
				executed.push("urgent");
			}),
		);
		await scheduler.runFrame();
		expect(executed[0]).toBe("urgent");
	});

	test("removes task by id", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		const executed: string[] = [];
		scheduler.enqueue(
			makeTask("task-a", 0, "model-a", async () => {
				executed.push("a");
			}),
		);
		scheduler.enqueue(
			makeTask("task-b", 0, "model-a", async () => {
				executed.push("b");
			}),
		);
		scheduler.dequeue("task-a");
		await scheduler.runFrame();
		expect(executed).toEqual(["b"]);
	});

	test("reports pending task count", () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		expect(scheduler.pendingCount).toBe(0);
		scheduler.enqueue(makeTask("a", 0));
		scheduler.enqueue(makeTask("b", 1));
		expect(scheduler.pendingCount).toBe(2);
	});

	test("runs async task within frame budget", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 500 });
		let resolved = false;
		scheduler.enqueue(
			makeTask("async-task", 0, "model-a", async () => {
				await new Promise((r) => setTimeout(r, 10));
				resolved = true;
			}),
		);
		await scheduler.runFrame();
		expect(resolved).toBe(true);
	});

	test("preemptModel aborts running tasks for model", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 5000 });
		let aborted = false;

		scheduler.enqueue(
			makeTask("long-task", 0, "model-x", async (signal) => {
				await new Promise<void>((resolve, reject) => {
					const timer = setTimeout(resolve, 10000);
					signal.addEventListener("abort", () => {
						clearTimeout(timer);
						aborted = true;
						reject(new DOMException("Aborted", "AbortError"));
					});
				});
			}),
		);

		// Start the frame in background, then preempt
		const framePromise = scheduler.runFrame();
		// Small delay to let the task start
		await new Promise((r) => setTimeout(r, 5));
		scheduler.preemptModel("model-x");
		await framePromise;

		expect(aborted).toBe(true);
		expect(scheduler.pendingCount).toBe(0);
		expect(scheduler.runningCount).toBe(0);
	});

	test("emits taskComplete event", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		const completed: string[] = [];
		scheduler.on("taskComplete", (...args: unknown[]) => {
			completed.push(args[0] as string);
		});
		scheduler.enqueue(makeTask("t1", 0, "model-a"));
		scheduler.enqueue(makeTask("t2", 1, "model-a"));
		await scheduler.runFrame();
		expect(completed).toEqual(["t1", "t2"]);
	});

	test("emits taskStart event", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		const started: string[] = [];
		scheduler.on("taskStart", (...args: unknown[]) => {
			started.push(args[0] as string);
		});
		scheduler.enqueue(makeTask("t1", 0, "model-a"));
		await scheduler.runFrame();
		expect(started).toEqual(["t1"]);
	});

	test("pendingCount and runningCount track state", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 5000 });
		scheduler.enqueue(makeTask("t1", 0, "model-a"));
		scheduler.enqueue(makeTask("t2", 1, "model-a"));
		expect(scheduler.pendingCount).toBe(2);
		expect(scheduler.runningCount).toBe(0);

		// Run a frame with a task that takes a moment
		let taskStarted = false;
		scheduler.enqueue(
			makeTask("t3", 0, "model-a", async () => {
				taskStarted = true;
				await new Promise((r) => setTimeout(r, 20));
			}),
		);

		const framePromise = scheduler.runFrame();
		// Wait just enough for the task to start
		await new Promise((r) => setTimeout(r, 5));
		expect(taskStarted).toBe(true);

		await framePromise;
		expect(scheduler.pendingCount).toBe(0);
		expect(scheduler.runningCount).toBe(0);
	});

	test("emits budgetExceeded when deadline passes", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 1 });
		let budgetExceeded = false;
		scheduler.on("budgetExceeded", () => {
			budgetExceeded = true;
		});
		for (let i = 0; i < 100; i++) {
			scheduler.enqueue(
				makeTask(`slow-${i}`, 0, "model-a", async () => {
					const start = performance.now();
					while (performance.now() - start < 1) {
						/* spin */
					}
				}),
			);
		}
		await scheduler.runFrame();
		expect(budgetExceeded).toBe(true);
	});

	test("off removes event listener", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		const events: string[] = [];
		const handler = (...args: unknown[]) => events.push(args[0] as string);
		scheduler.on("taskComplete", handler);
		scheduler.off("taskComplete", handler);
		scheduler.enqueue(makeTask("t1", 0, "model-a"));
		await scheduler.runFrame();
		expect(events).toEqual([]);
	});

	test("clear aborts running tasks and empties queue", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 5000 });
		scheduler.enqueue(makeTask("t1", 0, "model-a"));
		scheduler.enqueue(makeTask("t2", 1, "model-a"));
		expect(scheduler.pendingCount).toBe(2);
		scheduler.clear();
		expect(scheduler.pendingCount).toBe(0);
		expect(scheduler.runningCount).toBe(0);
	});

	test("dequeue returns false for unknown task", () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		expect(scheduler.dequeue("nonexistent")).toBe(false);
	});

	test("getPending returns queued tasks", () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		scheduler.enqueue(makeTask("a", 2));
		scheduler.enqueue(makeTask("b", 1));
		const pending = scheduler.getPending();
		expect(pending.length).toBe(2);
		// Sorted by priority
		expect(pending[0].id).toBe("b");
		expect(pending[1].id).toBe("a");
	});

	test("maxConcurrent limits parallel execution", async () => {
		const scheduler = new Scheduler({ frameBudgetMs: 5000, maxConcurrent: 1 });
		const order: string[] = [];
		scheduler.enqueue(
			makeTask("first", 0, "model-a", async () => {
				order.push("first-start");
				await new Promise((r) => setTimeout(r, 10));
				order.push("first-end");
			}),
		);
		scheduler.enqueue(
			makeTask("second", 1, "model-a", async () => {
				order.push("second-start");
				await new Promise((r) => setTimeout(r, 10));
				order.push("second-end");
			}),
		);
		await scheduler.runFrame();
		// With maxConcurrent=1, first must fully complete before second starts
		expect(order.indexOf("first-end")).toBeLessThan(
			order.indexOf("second-start"),
		);
	});

	test("preemptModel removes pending tasks for model", () => {
		const scheduler = new Scheduler({ frameBudgetMs: 100 });
		scheduler.enqueue(makeTask("t1", 0, "model-a"));
		scheduler.enqueue(makeTask("t2", 0, "model-b"));
		scheduler.enqueue(makeTask("t3", 0, "model-a"));
		scheduler.preemptModel("model-a");
		expect(scheduler.pendingCount).toBe(1);
		expect(scheduler.getPending()[0].id).toBe("t2");
	});
});
