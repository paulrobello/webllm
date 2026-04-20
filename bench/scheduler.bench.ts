import { bench, group } from "mitata";
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

group("enqueue", () => {
  bench("10 tasks", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    for (let i = 0; i < 10; i++) {
      scheduler.enqueue(makeTask(`task-${i}`, i));
    }
  });

  bench("100 tasks", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    for (let i = 0; i < 100; i++) {
      scheduler.enqueue(makeTask(`task-${i}`, i));
    }
  });
});

group("dequeue", () => {
  bench("from 100 tasks", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    for (let i = 0; i < 100; i++) {
      scheduler.enqueue(makeTask(`task-${i}`, 0));
    }
    scheduler.dequeue("task-50");
  });
});

group("runFrame", () => {
  bench("10 no-op tasks", async () => {
    const scheduler = new Scheduler({ frameBudgetMs: 1000 });
    for (let i = 0; i < 10; i++) {
      scheduler.enqueue(makeTask(`task-${i}`, i));
    }
    await scheduler.runFrame();
  });
});

group("preemptModel", () => {
  bench("50 tasks across 5 models", () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    for (let i = 0; i < 50; i++) {
      const modelId = `model-${i % 5}`;
      scheduler.enqueue(makeTask(`task-${i}`, 0, modelId));
    }
    scheduler.preemptModel("model-2");
  });
});
