import { bench, group } from "mitata";
import { KVCache } from "../src/models/kv-cache.js";
import { makeKVCacheConfig, makePopulatedKVCache } from "./helpers.js";

group("findSlots", () => {
  bench("1 token (empty cache)", () => {
    const cache = new KVCache(makeKVCacheConfig(4, 1024));
    cache.findSlots(1, 0);
  });

  bench("128 tokens (empty cache)", () => {
    const cache = new KVCache(makeKVCacheConfig(4, 1024));
    cache.findSlots(128, 0);
  });
});

group("updateSlots", () => {
  bench("128 slots", () => {
    const cache = new KVCache(makeKVCacheConfig(4, 1024));
    const slots = cache.findSlots(128, 0);
    cache.updateSlots(
      slots,
      Array.from({ length: 128 }, (_, i) => i),
      0,
    );
  });
});

group("evictSequence", () => {
  bench("populated cache (2 sequences, 256 cells)", () => {
    const cache = makePopulatedKVCache(256, 2);
    cache.evictSequence(0);
  });
});

group("sharePromptCells", () => {
  bench("64 cells between 2 sequences", () => {
    const cache = new KVCache(makeKVCacheConfig(4, 256));
    const slots = cache.findSlots(64, 0);
    cache.updateSlots(
      slots,
      Array.from({ length: 64 }, (_, i) => i),
      0,
    );
    cache.sharePromptCells(0, 1, 64);
  });
});
