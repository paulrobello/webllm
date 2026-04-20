export interface BufferAllocation {
  readonly id: number;
  size: number;
  priority: number;
  freed: boolean;
}

export class MemoryPool {
  private allocations = new Map<number, BufferAllocation>();
  private nextId = 0;
  private _budget: number;
  private _usedBytes = 0;

  constructor(budget: number) {
    this._budget = budget;
  }

  get budget(): number { return this._budget; }
  get usedBytes(): number { return this._usedBytes; }
  get remainingBytes(): number { return this._budget - this._usedBytes; }
  get pressureRatio(): number { return this._usedBytes / this._budget; }
  get isUnderPressure(): boolean { return this.pressureRatio > 0.75; }
  get allocationCount(): number { return this.allocations.size; }

  allocate(size: number, priority = 0): number {
    if (size > this.remainingBytes) {
      throw new Error(`Allocation of ${size} bytes exceeds memory budget (remaining: ${this.remainingBytes})`);
    }
    const id = this.nextId++;
    this.allocations.set(id, { id, size, priority, freed: false });
    this._usedBytes += size;
    return id;
  }

  free(id: number): void {
    const alloc = this.allocations.get(id);
    if (!alloc || alloc.freed) return;
    alloc.freed = true;
    this._usedBytes -= alloc.size;
    this.allocations.delete(id);
  }

  canAllocate(size: number): boolean { return size <= this.remainingBytes; }

  evictForAllocation(neededSize: number): number | null {
    const candidates = [...this.allocations.values()]
      .filter((a) => !a.freed)
      .sort((a, b) => b.priority - a.priority);
    for (const candidate of candidates) {
      if (this.remainingBytes + candidate.size >= neededSize) {
        this.free(candidate.id);
        return candidate.id;
      }
    }
    return null;
  }

  reset(): void {
    this.allocations.clear();
    this._usedBytes = 0;
  }
}
