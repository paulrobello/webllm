export class PipelineCache {
  private dbName: string;
  private storeName = 'pipelines';
  private db: IDBDatabase | null = null;

  constructor(dbName: string) { this.dbName = dbName; }

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
      request.onsuccess = () => { this.db = request.result; resolve(this.db); };
      request.onerror = () => reject(request.error);
    });
  }

  async get(key: string): Promise<Uint8Array | undefined> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const request = db.transaction(this.storeName, 'readonly').objectStore(this.storeName).get(key);
      request.onsuccess = () => resolve(request.result ?? undefined);
      request.onerror = () => reject(request.error);
    });
  }

  async put(key: string, data: Uint8Array): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const request = db.transaction(this.storeName, 'readwrite').objectStore(this.storeName).put(data, key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async delete(key: string): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const request = db.transaction(this.storeName, 'readwrite').objectStore(this.storeName).delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async keys(): Promise<string[]> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const request = db.transaction(this.storeName, 'readonly').objectStore(this.storeName).getAllKeys();
      request.onsuccess = () => resolve(request.result as string[]);
      request.onerror = () => reject(request.error);
    });
  }

  async clear(): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const request = db.transaction(this.storeName, 'readwrite').objectStore(this.storeName).clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
}
