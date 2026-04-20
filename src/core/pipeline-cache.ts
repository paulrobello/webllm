/**
 * IndexedDB-backed cache for WebGPU compute pipelines enabling reuse across sessions.
 */
export class PipelineCache {
	private dbName: string;
	private storeName = "pipelines";
	private db: IDBDatabase | null = null;

	constructor(dbName: string) {
		this.dbName = dbName;
	}

	private async getDb(): Promise<IDBDatabase> {
		if (this.db) return this.db;
		return new Promise((resolve, reject) => {
			const request = indexedDB.open(this.dbName, 1);
			request.onupgradeneeded = () => {
				const db = request.result;
				if (!db.objectStoreNames.contains(this.storeName)) {
					db.createObjectStore(this.storeName);
				}
			};
			request.onsuccess = () => {
				this.db = request.result;
				resolve(this.db);
			};
			request.onerror = () => reject(request.error);
		});
	}

	/**
	 * Retrieve a cached pipeline by key.
	 *
	 * @param key - Cache key (typically a shader hash).
	 * @returns The cached pipeline data, or undefined if not found.
	 */
	async get(key: string): Promise<Uint8Array | undefined> {
		const db = await this.getDb();
		return new Promise((resolve, reject) => {
			const request = db
				.transaction(this.storeName, "readonly")
				.objectStore(this.storeName)
				.get(key);
			request.onsuccess = () => resolve(request.result ?? undefined);
			request.onerror = () => reject(request.error);
		});
	}

	/**
	 * Store a pipeline in the cache.
	 *
	 * @param key - Cache key.
	 * @param data - Serialized pipeline binary data.
	 */
	async put(key: string, data: Uint8Array): Promise<void> {
		const db = await this.getDb();
		return new Promise((resolve, reject) => {
			const request = db
				.transaction(this.storeName, "readwrite")
				.objectStore(this.storeName)
				.put(data, key);
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	/**
	 * Remove a single entry from the cache.
	 *
	 * @param key - Cache key to delete.
	 */
	async delete(key: string): Promise<void> {
		const db = await this.getDb();
		return new Promise((resolve, reject) => {
			const request = db
				.transaction(this.storeName, "readwrite")
				.objectStore(this.storeName)
				.delete(key);
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}

	/**
	 * List all cache keys.
	 *
	 * @returns Array of stored cache keys.
	 */
	async keys(): Promise<string[]> {
		const db = await this.getDb();
		return new Promise((resolve, reject) => {
			const request = db
				.transaction(this.storeName, "readonly")
				.objectStore(this.storeName)
				.getAllKeys();
			request.onsuccess = () => resolve(request.result as string[]);
			request.onerror = () => reject(request.error);
		});
	}

	/** Remove all entries from the cache. */
	async clear(): Promise<void> {
		const db = await this.getDb();
		return new Promise((resolve, reject) => {
			const request = db
				.transaction(this.storeName, "readwrite")
				.objectStore(this.storeName)
				.clear();
			request.onsuccess = () => resolve();
			request.onerror = () => reject(request.error);
		});
	}
}
