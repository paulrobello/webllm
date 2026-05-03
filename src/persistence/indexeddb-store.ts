import {
	PersistenceIOError,
	PersistenceQuotaError,
	PersistenceUnavailableError,
} from "../core/errors.js";

/**
 * IndexedDB-backed store for persisted-conversation blobs. Companion
 * to the engine's `exportConversation` / `importConversation`
 * primitives. Optional: apps that want OPFS or server-side storage
 * implement their own store against the same Uint8Array contract.
 *
 * Spec: 2026-05-03-prefix-cache-persistence-design.md
 */
export interface ConversationStoreEntry {
	key: string;
	byteLength: number;
	savedAtMs: number;
}

interface ConversationMetaRecord {
	byteLength: number;
	savedAtMs: number;
}

const PAYLOAD_STORE = "conversations";
const META_STORE = "conversation-meta";

export class IndexedDBConversationStore {
	private dbName: string;
	private db: IDBDatabase | null = null;

	constructor(dbName: string) {
		this.dbName = dbName;
	}

	async open(): Promise<void> {
		if (this.db) return;
		if (typeof indexedDB === "undefined") {
			throw new PersistenceUnavailableError("indexeddb-missing");
		}
		return new Promise<void>((resolve, reject) => {
			let req: IDBOpenDBRequest;
			try {
				req = indexedDB.open(this.dbName, 1);
			} catch (e) {
				reject(new PersistenceUnavailableError("indexeddb-blocked", e));
				return;
			}
			req.onupgradeneeded = () => {
				const db = req.result;
				if (!db.objectStoreNames.contains(PAYLOAD_STORE)) {
					db.createObjectStore(PAYLOAD_STORE);
				}
				if (!db.objectStoreNames.contains(META_STORE)) {
					db.createObjectStore(META_STORE);
				}
			};
			req.onsuccess = () => {
				this.db = req.result;
				resolve();
			};
			req.onerror = () =>
				reject(new PersistenceUnavailableError("open-failed", req.error));
			req.onblocked = () =>
				reject(new PersistenceUnavailableError("indexeddb-blocked"));
		});
	}

	async put(key: string, blob: Uint8Array): Promise<void> {
		await this.open();
		const db = this.db as IDBDatabase;
		const meta: ConversationMetaRecord = {
			byteLength: blob.byteLength,
			savedAtMs: Date.now(),
		};
		return new Promise<void>((resolve, reject) => {
			const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
			tx.oncomplete = () => resolve();
			tx.onabort = () => {
				const err = tx.error;
				if (err && err.name === "QuotaExceededError") {
					reject(new PersistenceQuotaError(blob.byteLength));
				} else {
					reject(new PersistenceIOError("transaction-aborted", err));
				}
			};
			tx.onerror = () => {
				const err = tx.error;
				if (err && err.name === "QuotaExceededError") {
					reject(new PersistenceQuotaError(blob.byteLength));
				} else {
					reject(new PersistenceIOError("io-failure", err));
				}
			};
			try {
				const payloadReq = tx.objectStore(PAYLOAD_STORE).put(blob, key);
				payloadReq.onerror = () => {
					if (payloadReq.error?.name === "QuotaExceededError") {
						reject(new PersistenceQuotaError(blob.byteLength));
						tx.abort();
					}
				};
				tx.objectStore(META_STORE).put(meta, key);
			} catch (e) {
				reject(new PersistenceIOError("io-failure", e));
			}
		});
	}

	async get(key: string): Promise<Uint8Array | undefined> {
		await this.open();
		const db = this.db as IDBDatabase;
		return new Promise<Uint8Array | undefined>((resolve, reject) => {
			const req = db
				.transaction(PAYLOAD_STORE, "readonly")
				.objectStore(PAYLOAD_STORE)
				.get(key);
			req.onsuccess = () => resolve(req.result ?? undefined);
			req.onerror = () =>
				reject(new PersistenceIOError("io-failure", req.error));
		});
	}

	async delete(key: string): Promise<void> {
		await this.open();
		const db = this.db as IDBDatabase;
		return new Promise<void>((resolve, reject) => {
			const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
			tx.oncomplete = () => resolve();
			tx.onerror = () => reject(new PersistenceIOError("io-failure", tx.error));
			tx.onabort = () =>
				reject(new PersistenceIOError("transaction-aborted", tx.error));
			tx.objectStore(PAYLOAD_STORE).delete(key);
			tx.objectStore(META_STORE).delete(key);
		});
	}

	async list(): Promise<ConversationStoreEntry[]> {
		await this.open();
		const db = this.db as IDBDatabase;
		return new Promise<ConversationStoreEntry[]>((resolve, reject) => {
			const tx = db.transaction(META_STORE, "readonly");
			const store = tx.objectStore(META_STORE);
			const out: ConversationStoreEntry[] = [];
			const req = store.openCursor();
			req.onsuccess = () => {
				const cursor = req.result;
				if (cursor) {
					const key = String(cursor.key);
					const meta = cursor.value as ConversationMetaRecord;
					out.push({
						key,
						byteLength: meta.byteLength,
						savedAtMs: meta.savedAtMs,
					});
					cursor.continue();
				} else {
					resolve(out);
				}
			};
			req.onerror = () =>
				reject(new PersistenceIOError("io-failure", req.error));
		});
	}

	async clear(): Promise<void> {
		await this.open();
		const db = this.db as IDBDatabase;
		return new Promise<void>((resolve, reject) => {
			const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
			tx.oncomplete = () => resolve();
			tx.onerror = () => reject(new PersistenceIOError("io-failure", tx.error));
			tx.objectStore(PAYLOAD_STORE).clear();
			tx.objectStore(META_STORE).clear();
		});
	}

	async close(): Promise<void> {
		if (this.db) {
			this.db.close();
			this.db = null;
		}
	}
}
