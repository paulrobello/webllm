export interface ScheduledTask {
  id: string;
  priority: number; // lower = higher priority
  execute: () => Promise<void>;
}

export interface SchedulerConfig {
  frameBudgetMs: number;
}

export class Scheduler {
  private queue: ScheduledTask[] = [];
  private frameBudgetMs: number;

  constructor(config: SchedulerConfig) {
    this.frameBudgetMs = config.frameBudgetMs;
  }

  get pendingCount(): number {
    return this.queue.length;
  }

  enqueue(task: ScheduledTask): void {
    this.queue.push(task);
    this.queue.sort((a, b) => a.priority - b.priority);
  }

  dequeue(id: string): boolean {
    const idx = this.queue.findIndex((t) => t.id === id);
    if (idx === -1) return false;
    this.queue.splice(idx, 1);
    return true;
  }

  runCycle(): void {
    const deadline = performance.now() + this.frameBudgetMs;
    while (this.queue.length > 0 && performance.now() < deadline) {
      const task = this.queue.shift();
      if (task) task.execute();
    }
  }

  clear(): void {
    this.queue.length = 0;
  }
}
