/**
 * Fan-out async generator router with backpressure for streaming tokens to multiple consumers.
 *
 * Each consumer gets its own AsyncGenerator that yields values in real time.
 * Backpressure is implicit: if a consumer is slow, values queue in memory until consumed.
 */
export class StreamRouter<T> {
	private streams = new Map<
		string,
		{ push: (value: T) => void; close: () => void; interrupt: () => void }
	>();

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

		const push = (value: T): void => {
			if (resolve) {
				resolve({ value, done: false });
				resolve = null;
			} else {
				queue.push(value);
			}
		};
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

		this.streams.set(id, { push, close, interrupt });

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
}
