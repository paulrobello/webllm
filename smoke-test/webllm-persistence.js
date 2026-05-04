// src/core/errors.ts
class WebLLMError extends Error {
  code;
  constructor(message, code, options) {
    super(message, options);
    this.name = "WebLLMError";
    this.code = code;
  }
}
class PersistenceUnavailableError extends WebLLMError {
  reason;
  constructor(reason, cause) {
    super(`persistence unavailable: ${reason}`, "PERSISTENCE_UNAVAILABLE", cause !== undefined ? { cause } : undefined);
    this.name = "PersistenceUnavailableError";
    this.reason = reason;
  }
}

class PersistenceQuotaError extends WebLLMError {
  attemptedBytes;
  constructor(attemptedBytes) {
    super(`persistence quota exceeded (attempted ${attemptedBytes} bytes)`, "PERSISTENCE_QUOTA");
    this.name = "PersistenceQuotaError";
    this.attemptedBytes = attemptedBytes;
  }
}

class PersistenceIOError extends WebLLMError {
  reason;
  constructor(reason, cause) {
    super(`persistence IO error: ${reason}`, "PERSISTENCE_IO", { cause });
    this.name = "PersistenceIOError";
    this.reason = reason;
  }
}

// src/persistence/indexeddb-store.ts
var PAYLOAD_STORE = "conversations";
var META_STORE = "conversation-meta";

class IndexedDBConversationStore {
  dbName;
  db = null;
  constructor(dbName) {
    this.dbName = dbName;
  }
  async open() {
    if (this.db)
      return;
    if (typeof indexedDB === "undefined") {
      throw new PersistenceUnavailableError("indexeddb-missing");
    }
    return new Promise((resolve, reject) => {
      let req;
      try {
        req = indexedDB.open(this.dbName, 1);
      } catch (e) {
        reject(new PersistenceUnavailableError("open-failed", e));
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
      req.onerror = () => reject(new PersistenceUnavailableError("open-failed", req.error));
      req.onblocked = () => reject(new PersistenceUnavailableError("indexeddb-blocked"));
    });
  }
  async put(key, blob) {
    await this.open();
    const db = this.db;
    const meta = {
      byteLength: blob.byteLength,
      savedAtMs: Date.now()
    };
    return new Promise((resolve, reject) => {
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
  async get(key) {
    await this.open();
    const db = this.db;
    return new Promise((resolve, reject) => {
      const req = db.transaction(PAYLOAD_STORE, "readonly").objectStore(PAYLOAD_STORE).get(key);
      req.onsuccess = () => resolve(req.result ?? undefined);
      req.onerror = () => reject(new PersistenceIOError("io-failure", req.error));
    });
  }
  async delete(key) {
    await this.open();
    const db = this.db;
    return new Promise((resolve, reject) => {
      const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(new PersistenceIOError("io-failure", tx.error));
      tx.onabort = () => reject(new PersistenceIOError("transaction-aborted", tx.error));
      tx.objectStore(PAYLOAD_STORE).delete(key);
      tx.objectStore(META_STORE).delete(key);
    });
  }
  async list() {
    await this.open();
    const db = this.db;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(META_STORE, "readonly");
      const store = tx.objectStore(META_STORE);
      const out = [];
      const req = store.openCursor();
      req.onsuccess = () => {
        const cursor = req.result;
        if (cursor) {
          const key = String(cursor.key);
          const meta = cursor.value;
          out.push({
            key,
            byteLength: meta.byteLength,
            savedAtMs: meta.savedAtMs
          });
          cursor.continue();
        } else {
          resolve(out);
        }
      };
      req.onerror = () => reject(new PersistenceIOError("io-failure", req.error));
    });
  }
  async clear() {
    await this.open();
    const db = this.db;
    return new Promise((resolve, reject) => {
      const tx = db.transaction([PAYLOAD_STORE, META_STORE], "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(new PersistenceIOError("io-failure", tx.error));
      tx.objectStore(PAYLOAD_STORE).clear();
      tx.objectStore(META_STORE).clear();
    });
  }
  async close() {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}
export {
  IndexedDBConversationStore
};
