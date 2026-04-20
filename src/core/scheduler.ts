/** A unit of work submitted to the scheduler for cooperative execution. */
export interface ScheduledTask {
	id: string;
	priority: number; // 0 = highest
	modelId: string;
	execute: (signal: AbortSignal) => Promise<void>;
	yieldAfterMs?: number;
}

/** Configuration for the scheduler's frame budget and concurrency limits. */
export interface SchedulerConfig {
	frameBudgetMs: number;
	maxConcurrent?: number;
}

/** Event types emitted during the scheduler lifecycle. */
export type SchedulerEvent =
	| "taskStart"
	| "taskYield"
	| "taskComplete"
	| "taskPreempted"
	| "budgetExceeded";

/**
 * Priority-based task scheduler with cooperative yielding and AbortController preemption.
 */
export class Scheduler {
	private queue: ScheduledTask[] = [];
	private running: Map<
		string,
		{ task: ScheduledTask; controller: AbortController }
	> = new Map();
	private frameBudgetMs: number;
	private maxConcurrent: number;
	private listeners: Map<string, Set<(...args: unknown[]) => void>> = new Map();

	constructor(config: SchedulerConfig) {
		this.frameBudgetMs = config.frameBudgetMs;
		this.maxConcurrent = config.maxConcurrent ?? 1;
	}

	/**
	 * Add a task to the priority queue, sorted by ascending priority.
	 *
	 * @param task - The task to schedule.
	 */
	enqueue(task: ScheduledTask): void {
		this.queue.push(task);
		this.queue.sort((a, b) => a.priority - b.priority);
	}

	/**
	 * Remove a task by ID from the queue or abort it if currently running.
	 *
	 * @param id - Task identifier.
	 * @returns True if the task was found and removed or aborted.
	 */
	dequeue(id: string): boolean {
		const idx = this.queue.findIndex((t) => t.id === id);
		if (idx !== -1) {
			this.queue.splice(idx, 1);
			return true;
		}
		const running = this.running.get(id);
		if (running) {
			running.controller.abort();
			this.running.delete(id);
			return true;
		}
		return false;
	}

	/**
	 * Execute as many queued tasks as possible within the configured frame budget.
	 *
	 * Emits budgetExceeded if tasks remain after the deadline.
	 */
	async runFrame(): Promise<void> {
		const deadline = performance.now() + this.frameBudgetMs;
		while (
			this.queue.length > 0 &&
			this.running.size < this.maxConcurrent &&
			performance.now() < deadline
		) {
			const task = this.queue.shift();
			if (!task) break;
			const controller = new AbortController();
			this.running.set(task.id, { task, controller });
			this.emit("taskStart", task.id);

			try {
				await task.execute(controller.signal);
				if (!controller.signal.aborted) {
					this.emit("taskComplete", task.id);
				}
			} catch {
				if (controller.signal.aborted) {
					this.emit("taskPreempted", task.id);
				}
			}
			this.running.delete(task.id);
		}
		if (performance.now() >= deadline && this.queue.length > 0) {
			this.emit("budgetExceeded");
		}
	}

	/**
	 * Abort all running and queued tasks belonging to the given model.
	 *
	 * @param modelId - Model identifier whose tasks should be preempted.
	 */
	preemptModel(modelId: string): void {
		for (const [id, { task, controller }] of this.running) {
			if (task.modelId === modelId) {
				controller.abort();
				this.running.delete(id);
				this.emit("taskPreempted", id);
			}
		}
		this.queue = this.queue.filter((t) => t.modelId !== modelId);
	}

	/** @returns Snapshot of tasks waiting in the queue. */
	getPending(): readonly ScheduledTask[] {
		return this.queue;
	}

	/** @returns Snapshot of currently executing tasks. */
	getRunning(): readonly ScheduledTask[] {
		return [...this.running.values()].map((r) => r.task);
	}

	/**
	 * Subscribe to a scheduler event.
	 *
	 * @param event - Event type to listen for.
	 * @param handler - Callback invoked with event-specific arguments.
	 */
	on(event: SchedulerEvent, handler: (...args: unknown[]) => void): void {
		if (!this.listeners.has(event)) this.listeners.set(event, new Set());
		this.listeners.get(event)?.add(handler);
	}

	/**
	 * Unsubscribe from a scheduler event.
	 *
	 * @param event - Event type.
	 * @param handler - Previously registered callback.
	 */
	off(event: SchedulerEvent, handler: (...args: unknown[]) => void): void {
		this.listeners.get(event)?.delete(handler);
	}

	private emit(event: SchedulerEvent, ...args: unknown[]): void {
		for (const handler of this.listeners.get(event) ?? []) {
			handler(...args);
		}
	}

	/** Abort all running tasks and clear the pending queue. */
	clear(): void {
		for (const [, { controller }] of this.running) controller.abort();
		this.running.clear();
		this.queue.length = 0;
	}

	/** Number of tasks waiting in the queue. */
	get pendingCount(): number {
		return this.queue.length;
	}

	/** Number of tasks currently executing. */
	get runningCount(): number {
		return this.running.size;
	}
}
