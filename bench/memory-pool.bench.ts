import { bench, group } from "mitata";
import { MemoryPool } from "../src/core/memory-pool.js";

group("allocate", () => {
  bench("1000 allocations (256 bytes each)", () => {
    const pool = new MemoryPool(1024 * 1024);
    for (let i = 0; i < 1000; i++) {
      pool.allocate(256, 0, "model-a");
    }
  });
});

group("free", () => {
  bench("1000 allocations", () => {
    const pool = new MemoryPool(1024 * 1024);
    const ids: number[] = [];
    for (let i = 0; i < 1000; i++) {
      ids.push(pool.allocate(256, 0, "model-a"));
    }
    for (const id of ids) {
      pool.free(id);
    }
  });
});

group("evictForAllocation", () => {
  bench("evict lowest priority", () => {
    const pool = new MemoryPool(4096);
    pool.allocate(2048, 5, "model-a");
    pool.allocate(2048, 0, "model-b");
    pool.evictForAllocation(1024, "model-b");
  });
});

group("evictModel", () => {
  bench("3 models, 30 allocations each", () => {
    const pool = new MemoryPool(1024 * 1024);
    for (const model of ["model-a", "model-b", "model-c"]) {
      for (let i = 0; i < 30; i++) {
        pool.allocate(1024, 0, model);
      }
    }
    pool.evictModel("model-b");
  });
});

group("getModelUsage", () => {
  bench("3 models, 30 allocations each", () => {
    const pool = new MemoryPool(1024 * 1024);
    for (const model of ["model-a", "model-b", "model-c"]) {
      for (let i = 0; i < 30; i++) {
        pool.allocate(1024, 0, model);
      }
    }
    pool.getModelUsage("model-b");
  });
});
