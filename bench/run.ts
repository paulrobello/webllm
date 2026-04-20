import { run } from "mitata";

import "./tokenizer.bench.ts";
import "./sampler.bench.ts";
import "./kv-cache.bench.ts";
import "./scheduler.bench.ts";
import "./memory-pool.bench.ts";
import "./gguf-parser.bench.ts";
import "./stream-router.bench.ts";
import "./generation.bench.ts";
import "./tool-system.bench.ts";
import "./character.bench.ts";

await run();
