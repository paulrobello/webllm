export class StreamRouter<T> {
	private streams = new Map<
		string,
		{ push: (value: T) => void; close: () => void; interrupt: () => void }
	>();

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

	emit(id: string, value: T): void {
		this.streams.get(id)?.push(value);
	}
	close(id: string): void {
		const s = this.streams.get(id);
		if (s) {
			s.close();
			this.streams.delete(id);
		}
	}
	interrupt(id: string): void {
		const s = this.streams.get(id);
		if (s) {
			s.interrupt();
			this.streams.delete(id);
		}
	}
	removeConsumer(id: string): void {
		this.close(id);
	}
	hasConsumer(id: string): boolean {
		return this.streams.has(id);
	}
}
