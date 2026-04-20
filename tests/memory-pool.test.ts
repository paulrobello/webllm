import { describe, expect, test } from 'bun:test';
import { MemoryPool } from '../src/core/memory-pool.js';

describe('MemoryPool', () => {
  test('allocates a buffer within budget', () => {
    const pool = new MemoryPool(1024);
    const id = pool.allocate(256);
    expect(id).toBe(0);
    expect(pool.usedBytes).toBe(256);
    expect(pool.remainingBytes).toBe(768);
  });

  test('throws when allocation exceeds budget', () => {
    const pool = new MemoryPool(100);
    expect(() => pool.allocate(200)).toThrow('exceeds memory budget');
  });

  test('frees a buffer and reclaims memory', () => {
    const pool = new MemoryPool(1024);
    const id = pool.allocate(256);
    pool.free(id);
    expect(pool.usedBytes).toBe(0);
    expect(pool.remainingBytes).toBe(1024);
  });

  test('tracks multiple allocations', () => {
    const pool = new MemoryPool(1024);
    const id0 = pool.allocate(256);
    const id1 = pool.allocate(512);
    expect(pool.usedBytes).toBe(768);
    pool.free(id0);
    expect(pool.usedBytes).toBe(512);
    pool.free(id1);
    expect(pool.usedBytes).toBe(0);
  });

  test('reports memory pressure correctly', () => {
    const pool = new MemoryPool(1000);
    pool.allocate(800);
    expect(pool.pressureRatio).toBe(0.8);
    expect(pool.isUnderPressure).toBe(true);
  });

  test('evicts lowest priority allocation on pressure', () => {
    const pool = new MemoryPool(1024);
    const low = pool.allocate(512, 2);
    const high = pool.allocate(512, 0);
    expect(pool.canAllocate(256)).toBe(false);
    const evicted = pool.evictForAllocation(256);
    expect(evicted).toBe(low);
    expect(pool.canAllocate(256)).toBe(true);
  });

  test('reset clears all allocations', () => {
    const pool = new MemoryPool(1024);
    pool.allocate(256);
    pool.allocate(512);
    pool.reset();
    expect(pool.usedBytes).toBe(0);
    expect(pool.allocationCount).toBe(0);
  });
});
