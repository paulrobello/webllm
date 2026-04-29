/** Configuration for {@link StreamRouter}. */
export interface StreamRouterOptions {
	/**
	 * Maximum number of pending items queued for a single consumer before
	 * the consumer is force-interrupted as a backpressure safety net. A
	 * single permanently-stalled consumer cannot grow memory beyond this
	 * bound. Default: 4096. Set to `Infinity` to disable the cap.
	 */
	maxQueueDepth?: number;
}

/**
 * Fan-out async generator router with backpressure for streaming tokens to
 * multiple consumers.
 *
 * Each consumer gets its own AsyncGenerator that yields values in real time.
 * Backpressure is best-effort: if a consumer is slow, values queue in memory
 * up to {@link StreamRouterOptions.maxQueueDepth}. When the cap is exceeded,
 * the offending consumer is force-interrupted (subsequent emits to that id
 * are dropped) so a single stalled consumer cannot starve the producer.
 */
export class StreamRouter<T> {
	private streams = new Map<
		string,
		{
			push: (value: T) => void;
			close: () => void;
			interrupt: () => void;
			depth: () => number;
		}
	>();
	private readonly maxQueueDepth: number;

	constructor(options?: StreamRouterOptions) {
		this.maxQueueDepth = options?.maxQueueDepth ?? 4096;
	}

	/**
	 * Create a new consumer stream that yields values as they are emitted.
	 *
	 * @param id - Unique consumer identifier.
	 * @returns AsyncGenerator that yields values until closed or interrupted.
	 */
	createConsumer(id: string): AsyncGenerator<T> {
		let resolve: ((result: IteratorResult<T>) => void) | null = null;
		const queue: T[] = [];
		let closed = false;
		let interrupted = false;
		const cap = this.maxQueueDepth;

		const close = (): void => {
			closed = true;
			if (resolve) {
				resolve({ value: undefined, done: true });
				resolve = null;
			}
		};
		const interrupt = (): void => {
			interrupted = true;
			close();
		};
		const push = (value: T): void => {
			if (interrupted || closed) return;
			if (resolve) {
				resolve({ value, done: false });
				resolve = null;
				return;
			}
			if (queue.length >= cap) {
				// Backpressure safety net: a permanently-stalled consumer
				// cannot grow memory past `cap`. Drop the consumer rather
				// than back-propagating to the producer.
				interrupt();
				return;
			}
			queue.push(value);
		};
		const depth = (): number => queue.length;

		this.streams.set(id, { push, close, interrupt, depth });

		async function* generator(): AsyncGenerator<T> {
			while (true) {
				if (queue.length > 0) {
					yield queue.shift() as T;
					continue;
				}
				if (closed || interrupted) return;
				const result = await new Promise<IteratorResult<T>>((r) => {
					resolve = r;
				});
				if (result.done) return;
				yield result.value;
			}
		}
		return generator();
	}

	/**
	 * Push a value to a specific consumer's stream.
	 *
	 * @param id - Consumer identifier.
	 * @param value - Value to emit.
	 */
	emit(id: string, value: T): void {
		this.streams.get(id)?.push(value);
	}
	/**
	 * Gracefully close a consumer stream, allowing it to drain queued values.
	 *
	 * @param id - Consumer identifier.
	 */
	close(id: string): void {
		const s = this.streams.get(id);
		if (s) {
			s.close();
			this.streams.delete(id);
		}
	}
	/**
	 * Immediately terminate a consumer stream, discarding queued values.
	 *
	 * @param id - Consumer identifier.
	 */
	interrupt(id: string): void {
		const s = this.streams.get(id);
		if (s) {
			s.interrupt();
			this.streams.delete(id);
		}
	}
	/**
	 * Alias for close — removes and gracefully shuts down a consumer.
	 *
	 * @param id - Consumer identifier.
	 */
	removeConsumer(id: string): void {
		this.close(id);
	}
	/**
	 * Check whether a consumer with the given ID exists.
	 *
	 * @param id - Consumer identifier.
	 * @returns True if the consumer is registered.
	 */
	hasConsumer(id: string): boolean {
		return this.streams.has(id);
	}
	/**
	 * Current pending-item count for a consumer (post-resolve, pre-yield).
	 * Returns 0 for unknown ids. Useful for telemetry and slow-consumer
	 * detection ahead of the {@link StreamRouterOptions.maxQueueDepth} cap.
	 */
	getQueueDepth(id: string): number {
		return this.streams.get(id)?.depth() ?? 0;
	}
}
