import { describe, expect, test } from "bun:test";
import { Scheduler } from "../src/core/scheduler.js";

describe("Scheduler", () => {
  test("executes highest priority task first", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 16 });
    const executed: number[] = [];
    scheduler.enqueue({
      id: "low",
      priority: 2,
      execute: async () => {
        executed.push(2);
      },
    });
    scheduler.enqueue({
      id: "high",
      priority: 0,
      execute: async () => {
        executed.push(0);
      },
    });
    scheduler.enqueue({
      id: "mid",
      priority: 1,
      execute: async () => {
        executed.push(1);
      },
    });
    scheduler.runCycle();
    expect(executed).toEqual([0, 1, 2]);
  });

  test("respects frame budget and stops when exceeded", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 1 });
    let callCount = 0;
    for (let i = 0; i < 10; i++) {
      scheduler.enqueue({
        id: `task-${i}`,
        priority: 0,
        execute: async () => {
          callCount++;
          const start = performance.now();
          while (performance.now() - start < 0.5) {}
        },
      });
    }
    scheduler.runCycle();
    expect(callCount).toBeLessThan(10);
  });

  test("preempts lower priority when higher arrives", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    const executed: string[] = [];
    scheduler.enqueue({
      id: "background",
      priority: 2,
      execute: async () => {
        executed.push("background");
      },
    });
    scheduler.enqueue({
      id: "urgent",
      priority: 0,
      execute: async () => {
        executed.push("urgent");
      },
    });
    scheduler.runCycle();
    expect(executed[0]).toBe("urgent");
  });

  test("removes task by id", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    const executed: string[] = [];
    scheduler.enqueue({
      id: "task-a",
      priority: 0,
      execute: async () => {
        executed.push("a");
      },
    });
    scheduler.enqueue({
      id: "task-b",
      priority: 0,
      execute: async () => {
        executed.push("b");
      },
    });
    scheduler.dequeue("task-a");
    scheduler.runCycle();
    expect(executed).toEqual(["b"]);
  });

  test("reports pending task count", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    expect(scheduler.pendingCount).toBe(0);
    scheduler.enqueue({ id: "a", priority: 0, execute: async () => {} });
    scheduler.enqueue({ id: "b", priority: 1, execute: async () => {} });
    expect(scheduler.pendingCount).toBe(2);
  });
});
