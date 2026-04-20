/** Configuration for the game loop's timing and initial state. */
export interface GameLoopConfig {
	frameBudgetMs: number;
	targetFps?: number;
	paused?: boolean;
}

/** Callback invoked each frame with timing information. */
export type GameLoopCallback = (deltaMs: number, frameBudgetMs: number) => void;

/**
 * setTimeout-based game loop with FPS tracking for frame-budget-aware inference.
 */
export class GameLoop {
	private config: GameLoopConfig;
	private callback: GameLoopCallback | null = null;
	private _running = false;
	private _paused = false;
	private lastFrameTime = 0;
	private timerId: ReturnType<typeof setTimeout> | null = null;
	private _frameCount = 0;
	private fpsSamples: number[] = [];
	private readonly maxFpsSamples = 10;

	constructor(config: GameLoopConfig) {
		this.config = { targetFps: 60, paused: false, ...config };
		this._paused = this.config.paused ?? false;
	}

	/**
	 * Start the game loop, invoking the callback each frame.
	 *
	 * @param callback - Function called with delta time and frame budget each tick.
	 */
	start(callback: GameLoopCallback): void {
		this.callback = callback;
		this._running = true;
		this.lastFrameTime = performance.now();
		this.scheduleNext();
	}

	/** Stop the game loop and clear the pending timer. */
	stop(): void {
		this._running = false;
		if (this.timerId !== null) {
			clearTimeout(this.timerId);
			this.timerId = null;
		}
		this.callback = null;
	}

	/** Pause frame dispatch while keeping the loop in a running state. */
	pause(): void {
		this._paused = true;
	}

	/** Resume frame dispatch after a pause, resetting the frame timer baseline. */
	resume(): void {
		this._paused = false;
		this.lastFrameTime = performance.now();
		if (this._running) this.scheduleNext();
	}

	/**
	 * Dynamically adjust the per-frame time budget.
	 *
	 * @param ms - New frame budget in milliseconds.
	 */
	setFrameBudget(ms: number): void {
		this.config.frameBudgetMs = ms;
	}

	private tick = (): void => {
		if (!this._running || this._paused) return;

		const now = performance.now();
		const deltaMs = now - this.lastFrameTime;
		this.lastFrameTime = now;

		if (deltaMs > 0) {
			this.fpsSamples.push(1000 / deltaMs);
			if (this.fpsSamples.length > this.maxFpsSamples) this.fpsSamples.shift();
		}

		this._frameCount++;
		this.callback?.(deltaMs, this.config.frameBudgetMs);
		this.scheduleNext();
	};

	private scheduleNext(): void {
		if (!this._running || this._paused) return;
		const interval = 1000 / (this.config.targetFps ?? 60);
		this.timerId = setTimeout(this.tick, Math.max(0, interval));
	}

	/** Whether the loop has been started and not yet stopped. */
	get isRunning(): boolean {
		return this._running;
	}

	/** Whether the loop is currently paused. */
	get isPaused(): boolean {
		return this._paused;
	}

	/** Total frames dispatched since the loop started. */
	get frameCount(): number {
		return this._frameCount;
	}

	/** Smoothed FPS averaged over the last 10 frames. */
	get currentFps(): number {
		if (this.fpsSamples.length === 0) return 0;
		return this.fpsSamples.reduce((a, b) => a + b, 0) / this.fpsSamples.length;
	}
}
