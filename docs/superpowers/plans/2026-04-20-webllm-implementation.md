# WebLLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TypeScript library for high-performance LLM inference in the browser using llama.cpp's ggml-webgpu WASM backend with a custom scheduling layer for hierarchical multi-model inference.

**Architecture:** Three-layer system — WASM compute core (ggml-webgpu compiled via Emscripten with JSPI), TypeScript orchestration (GGUF parser, scheduler, KV cache, streaming), and a lightweight pure-WGSL path for small models. The TypeScript layer owns memory and scheduling; the WASM core is a stateless tensor engine.

**Tech Stack:** TypeScript, Bun, WebGPU, Emscripten, WGSL compute shaders, llama.cpp ggml-webgpu backend

**Design Spec:** `/Users/probello/.claude/plans/i-want-to-create-mutable-toucan.md`

---

## File Structure

```
webllm/
├── src/
│   ├── core/
│   │   ├── engine.ts              # Main WebLLM class — public entry point
│   │   ├── scheduler.ts           # Multi-model priority scheduler
│   │   ├── memory-pool.ts         # GPU buffer pool manager
│   │   ├── pipeline-cache.ts      # WGSL pipeline cache (IndexedDB)
│   │   └── types.ts               # Shared type definitions
│   ├── models/
│   │   ├── gguf-parser.ts         # GGUF binary format parser
│   │   ├── gguf-types.ts          # GGUF format type definitions
│   │   ├── model-loader.ts        # Model loading + weight streaming to GPU
│   │   ├── inference-session.ts   # Per-session inference context
│   │   └── kv-cache.ts            # Paged KV cache manager
│   ├── inference/
│   │   ├── ggml-wasm.ts           # WASM bindings for ggml-webgpu
│   │   ├── lightweight.ts         # Pure WGSL inference path
│   │   ├── tokenizer.ts           # Tokenizer (SentencePiece, BPE)
│   │   ├── sampler.ts             # Token sampling (top-k, top-p, temp)
│   │   └── stream-router.ts       # Token stream routing via async generators
│   ├── characters/
│   │   ├── character.ts           # Character class
│   │   ├── character-manager.ts   # Character lifecycle
│   │   └── tool-system.ts         # Function/tool calling
│   ├── wasm/
│   │   ├── CMakeLists.txt         # Emscripten build for ggml-webgpu
│   │   ├── webgpu-bridge.cpp      # C bridge API exported to WASM
│   │   └── exports.def            # Emscripten export definitions
│   ├── shaders/
│   │   ├── matmul.wgsl            # Lightweight matmul shaders
│   │   ├── norms.wgsl             # Layer norm, RMS norm
│   │   ├── activations.wgsl       # GELU, SiLU
│   │   └── embedding.wgsl         # Embedding lookup
│   └── index.ts                   # Public API re-exports
├── tests/
│   ├── gguf-parser.test.ts
│   ├── scheduler.test.ts
│   ├── inference.test.ts
│   └── lightweight.test.ts
├── examples/
│   ├── chat-demo/
│   ├── game-integration/
│   └── multi-model/
├── package.json
├── tsconfig.json
├── Makefile
├── .gitignore
└── README.md
```

---

## Phase 1: Foundation (WASM Core + GGUF Parser)

### Task 1: Project Scaffolding

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `Makefile`
- Create: `.gitignore`
- Create: `src/index.ts`
- Create: `src/core/types.ts`

- [ ] **Step 1: Initialize git repo and create package.json**

```bash
cd /Users/probello/Repos/webllm
git init
```

Create `package.json`:
```json
{
  "name": "@paulrobello/webllm",
  "version": "0.1.0",
  "description": "High-performance LLM inference in the browser via WebGPU, backed by llama.cpp",
  "type": "module",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "files": ["dist", "src"],
  "scripts": {
    "build": "bun build src/index.ts --outdir dist --target browser --minify",
    "dev": "bun build src/index.ts --outdir dist --target browser --watch",
    "test": "bun test",
    "lint": "biome check src tests",
    "lint:fix": "biome check --fix src tests",
    "fmt": "biome format --write src tests",
    "typecheck": "tsc --noEmit",
    "checkall": "bun run fmt && bun run lint && bun run typecheck && bun run test"
  },
  "keywords": ["llm", "webgpu", "inference", "browser", "llama-cpp", "gguf"],
  "license": "MIT",
  "author": "Paul Robello <probello@gmail.com>",
  "repository": {
    "type": "git",
    "url": "https://github.com/paulrobello/webllm"
  },
  "devDependencies": {
    "@biomejs/biome": "^2.0.0",
    "typescript": "^5.8.0",
    "@types/bun": "^1.2.0"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

Create `tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "exactOptionalPropertyTypes": false
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
```

- [ ] **Step 3: Create Makefile**

Create `Makefile`:
```makefile
.PHONY: build test lint fmt typecheck checkall clean install deps wasm-build

install:
	bun install

deps: install

build:
	bun run build

test:
	bun run test

lint:
	bun run lint

lint-fix:
	bun run lint:fix

fmt:
	bun run fmt

typecheck:
	bun run typecheck

checkall:
	bun run checkall

wasm-build:
	cd src/wasm && mkdir -p build && cd build && \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=ON \
		-DCMAKE_BUILD_TYPE=Release && \
	cmake --build . --config Release -j

clean:
	rm -rf dist node_modules src/wasm/build
```

- [ ] **Step 4: Create .gitignore**

Create `.gitignore`:
```
node_modules/
dist/
*.local
*.local.*
*-mcp.json
.gemini-clipboard
.cc2cc-session-id
claude_scratch/
.idea
settings.local.json
CLAUDE.local.md
.DS_Store
*.wasm
src/wasm/build/
```

- [ ] **Step 5: Create src/index.ts**

Create `src/index.ts`:
```typescript
export { WebLLM } from './core/engine.js';
export type { WebLLMConfig, ModelLoadOptions, ModelHandle } from './core/types.js';
```

- [ ] **Step 6: Create src/core/types.ts with shared type definitions**

Create `src/core/types.ts`:
```typescript
export interface WebLLMConfig {
  device: GPUDevice;
  cacheDir?: string;
  memoryBudget: number;
  frameBudgetMs?: number;
}

export interface ModelLoadOptions {
  priority: number;
  contextLength?: number;
  gpuLayers?: number;
  lightweight?: boolean;
}

export interface ModelHandle {
  readonly id: string;
  readonly name: string;
  readonly priority: number;
  readonly lightweight: boolean;
}

export type GgmlType =
  | 'f32' | 'f16' | 'q4_0' | 'q4_1' | 'q5_0' | 'q5_1' | 'q8_0'
  | 'q2_k' | 'q3_k' | 'q4_k' | 'q5_k' | 'q6_k'
  | 'iq2_xxs' | 'iq2_xs' | 'iq2_s' | 'iq3_xxs' | 'iq3_s'
  | 'iq1_s' | 'iq1_m' | 'iq4_nl' | 'iq4_xs';

export type ModelArchitecture =
  | 'llama' | 'mistral' | 'phi' | 'gemma' | 'qwen' | 'mixtral' | 'deepseek';

export interface TensorInfo {
  name: string;
  nDimensions: number;
  dimensions: number[];
  type: GgmlType;
  offset: number;
  size: number;
}

export interface ModelMetadata {
  architecture: ModelArchitecture;
  contextLength: number;
  embeddingLength: number;
  headCount: number;
  layerCount: number;
  vocabularySize: number;
  ropeFreqBase: number;
  ropeScale: number;
}

export type EventHandler<T = void> = (event: T) => void;

export interface MemoryPressureEvent {
  used: number;
  total: number;
  modelId: string;
}
```

- [ ] **Step 7: Install dependencies and verify**

```bash
cd /Users/probello/Repos/webllm
bun install
bun run typecheck
```

Expected: typecheck passes with no errors (empty exports are valid).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: initial project scaffolding with bun, typescript, biome"
```

---

### Task 2: GGUF Binary Format Types

**Files:**
- Create: `src/models/gguf-types.ts`
- Test: `tests/gguf-types.test.ts`

- [ ] **Step 1: Write tests for GGUF format constants and type definitions**

Create `tests/gguf-types.test.ts`:
```typescript
import { describe, expect, test } from 'bun:test';
import {
  GGUF_MAGIC,
  GGUF_VERSION,
  GGUF_DEFAULT_ALIGNMENT,
  GgufValueType,
  type GgufHeader,
  type GgufTensorInfo,
} from '../src/models/gguf-types.js';

describe('GGUF Format Constants', () => {
  test('GGUF_MAGIC is "GGUF" as uint32 little-endian', () => {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, GGUF_MAGIC, true);
    const decoded = new TextDecoder().decode(new Uint8Array(view.buffer));
    expect(decoded).toBe('GGUF');
  });

  test('GGUF_VERSION is 3', () => {
    expect(GGUF_VERSION).toBe(3);
  });

  test('GGUF_DEFAULT_ALIGNMENT is 32', () => {
    expect(GGUF_DEFAULT_ALIGNMENT).toBe(32);
  });
});

describe('GgufValueType enum', () => {
  test('covers all 13 GGUF value types', () => {
    const values = Object.values(GgufValueType).filter((v) => typeof v === 'number');
    expect(values).toHaveLength(13);
  });

  test('matches llama.cpp gguf_type enum values', () => {
    expect(GgufValueType.UINT8).toBe(0);
    expect(GgufValueType.INT8).toBe(1);
    expect(GgufValueType.UINT16).toBe(2);
    expect(GgufValueType.INT16).toBe(3);
    expect(GgufValueType.UINT32).toBe(4);
    expect(GgufValueType.INT32).toBe(5);
    expect(GgufValueType.FLOAT32).toBe(6);
    expect(GgufValueType.BOOL).toBe(7);
    expect(GgufValueType.STRING).toBe(8);
    expect(GgufValueType.ARRAY).toBe(9);
    expect(GgufValueType.UINT64).toBe(10);
    expect(GgufValueType.INT64).toBe(11);
    expect(GgufValueType.FLOAT64).toBe(12);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/gguf-types.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement gguf-types.ts**

Create `src/models/gguf-types.ts`:
```typescript
/** Magic number: "GGUF" encoded as uint32 little-endian = 0x46554747 */
export const GGUF_MAGIC = 0x46475547;

export const GGUF_VERSION = 3;

export const GGUF_DEFAULT_ALIGNMENT = 32;

export enum GgufValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export interface GgufHeader {
  magic: number;
  version: number;
  tensorCount: number;
  metadataKvCount: number;
}

export interface GgufKv {
  key: string;
  type: GgufValueType;
  isArray: boolean;
  value: unknown;
}

export interface GgufTensorInfo {
  name: string;
  nDimensions: number;
  dimensions: number[];
  type: number;
  offset: number;
}

export interface GgufContext {
  header: GgufHeader;
  metadata: Map<string, GgufKv>;
  tensors: GgufTensorInfo[];
  alignment: number;
  dataOffset: number;
  totalDataSize: number;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/gguf-types.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/gguf-types.ts tests/gguf-types.test.ts
git commit -m "feat: add GGUF binary format type definitions"
```

---

### Task 3: GGUF Binary Parser

**Files:**
- Create: `src/models/gguf-parser.ts`
- Test: `tests/gguf-parser.test.ts`

This is a pure TypeScript parser that reads GGUF binary format from an ArrayBuffer. Reference: `/Users/probello/Repos/llama.cpp/ggml/src/gguf.cpp`.

GGUF file layout:
```
[header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)] = 24 bytes
[metadata_kv: repeated (key_string, value_type, value)]
[tensor_infos: repeated (name_string, n_dims(4), dims(n*8), type(4), offset(8))]
[padding to alignment]
[tensor_data: contiguous block]
```

String encoding: `length(uint64) + chars(length bytes)`.

- [ ] **Step 1: Write tests for GGUF parser**

Create `tests/gguf-parser.test.ts`:
```typescript
import { describe, expect, test } from 'bun:test';
import { GgufParser } from '../src/models/gguf-parser.js';
import { GGUF_MAGIC, GGUF_VERSION } from '../src/models/gguf-types.js';
import type { GgufValueType } from '../src/models/gguf-types.js';

function writeString(buf: DataView, offset: number, str: string): number {
  buf.setBigUint64(offset, BigInt(str.length), true);
  offset += 8;
  for (let i = 0; i < str.length; i++) {
    buf.setUint8(offset++, str.charCodeAt(i));
  }
  return offset;
}

function buildMinimalGguf(): ArrayBuffer {
  const headerSize = 24;
  const kv: Array<{ key: string; type: GgufValueType; value: unknown }> = [
    { key: 'general.architecture', type: 8 as GgufValueType, value: 'llama' },
    { key: 'llama.context_length', type: 4 as GgufValueType, value: 4096 },
  ];
  const tensorCount = 1;
  const tensorName = 'token_embd.weight';

  // Calculate sizes
  let kvSize = 0;
  for (const entry of kv) {
    kvSize += 8 + entry.key.length; // key string
    kvSize += 4; // type
    if ((entry.type as GgufValueType) === 8) {
      // STRING
      kvSize += 8 + (entry.value as string).length;
    } else if ((entry.type as GgufValueType) === 4) {
      // UINT32
      kvSize += 4;
    }
  }

  let tensorInfoSize = 0;
  tensorInfoSize += 8 + tensorName.length; // name string
  tensorInfoSize += 4; // n_dimensions
  tensorInfoSize += 8 * 1; // 1 dimension
  tensorInfoSize += 4; // type
  tensorInfoSize += 8; // offset

  const totalSize = headerSize + kvSize + tensorInfoSize + 32 + 64; // +padding + data
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);

  let offset = 0;
  // Header
  view.setUint32(offset, GGUF_MAGIC, true);
  offset += 4;
  view.setUint32(offset, GGUF_VERSION, true);
  offset += 4;
  view.setBigUint64(offset, BigInt(tensorCount), true);
  offset += 8;
  view.setBigUint64(offset, BigInt(kv.length), true);
  offset += 8;

  // KV pairs
  for (const entry of kv) {
    offset = writeString(view, offset, entry.key);
    view.setUint32(offset, entry.type as number, true);
    offset += 4;
    if ((entry.type as GgufValueType) === 8) {
      offset = writeString(view, offset, entry.value as string);
    } else if ((entry.type as GgufValueType) === 4) {
      view.setUint32(offset, entry.value as number, true);
      offset += 4;
    }
  }

  // Tensor info
  offset = writeString(view, offset, tensorName);
  view.setUint32(offset, 1, true); // n_dimensions
  offset += 4;
  view.setBigUint64(offset, BigInt(10), true); // dim[0] = 10
  offset += 8;
  view.setUint32(offset, 0, true); // type = f32
  offset += 4;
  view.setBigUint64(offset, BigInt(0), true); // offset = 0
  offset += 8;

  return buf.slice(0, offset);
}

describe('GgufParser', () => {
  test('parses header correctly', () => {
    const buf = buildMinimalGguf();
    const ctx = GgufParser.parse(buf);

    expect(ctx.header.magic).toBe(GGUF_MAGIC);
    expect(ctx.header.version).toBe(GGUF_VERSION);
    expect(ctx.header.tensorCount).toBe(1);
    expect(ctx.header.metadataKvCount).toBe(2);
  });

  test('parses metadata KV pairs', () => {
    const buf = buildMinimalGguf();
    const ctx = GgufParser.parse(buf);

    expect(ctx.metadata.size).toBe(2);
    expect(ctx.metadata.get('general.architecture')?.value).toBe('llama');
    expect(ctx.metadata.get('llama.context_length')?.value).toBe(4096);
  });

  test('parses tensor info', () => {
    const buf = buildMinimalGguf();
    const ctx = GgufParser.parse(buf);

    expect(ctx.tensors).toHaveLength(1);
    expect(ctx.tensors[0].name).toBe('token_embd.weight');
    expect(ctx.tensors[0].nDimensions).toBe(1);
    expect(ctx.tensors[0].dimensions[0]).toBe(10);
    expect(ctx.tensors[0].type).toBe(0); // f32
  });

  test('throws on invalid magic', () => {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, 0xDEADBEEF, true); // wrong magic
    expect(() => GgufParser.parse(buf)).toThrow('Invalid GGUF magic');
  });

  test('throws on unsupported version', () => {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, GGUF_MAGIC, true);
    view.setUint32(4, 99, true); // wrong version
    expect(() => GgufParser.parse(buf)).toThrow('Unsupported GGUF version');
  });

  test('getMetadataString returns string value', () => {
    const buf = buildMinimalGguf();
    const ctx = GgufParser.parse(buf);

    expect(GgufParser.getMetadataString(ctx, 'general.architecture')).toBe('llama');
  });

  test('getMetadataNumber returns numeric value', () => {
    const buf = buildMinimalGguf();
    const ctx = GgufParser.parse(buf);

    expect(GgufParser.getMetadataNumber(ctx, 'llama.context_length')).toBe(4096);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/gguf-parser.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement GgufParser**

Create `src/models/gguf-parser.ts`:
```typescript
import {
  GGUF_MAGIC,
  GGUF_VERSION,
  GGUF_DEFAULT_ALIGNMENT,
  GgufValueType,
  type GgufContext,
  type GgufHeader,
  type GgufKv,
  type GgufTensorInfo,
} from './gguf-types.js';

export class GgufParser {
  static parse(buffer: ArrayBuffer): GgufContext {
    const view = new DataView(buffer);
    let offset = 0;

    const header = readHeader(view, offset);
    offset = 24;

    if (header.magic !== GGUF_MAGIC) {
      throw new Error(`Invalid GGUF magic: expected 0x${GGUF_MAGIC.toString(16)}, got 0x${header.magic.toString(16)}`);
    }
    if (header.version !== GGUF_VERSION) {
      throw new Error(`Unsupported GGUF version: ${header.version}, expected ${GGUF_VERSION}`);
    }

    const metadata = new Map<string, GgufKv>();
    for (let i = 0; i < header.metadataKvCount; i++) {
      const { kv, newOffset } = readKv(view, offset);
      metadata.set(kv.key, kv);
      offset = newOffset;
    }

    const tensors: GgufTensorInfo[] = [];
    for (let i = 0; i < header.tensorCount; i++) {
      const { info, newOffset } = readTensorInfo(view, offset);
      tensors.push(info);
      offset = newOffset;
    }

    const alignment = getAlignment(metadata);
    const dataOffset = alignTo(offset, alignment);
    const totalDataSize = calculateTotalDataSize(tensors);

    return { header, metadata, tensors, alignment, dataOffset, totalDataSize };
  }

  static getMetadataString(ctx: GgufContext, key: string): string | undefined {
    const kv = ctx.metadata.get(key);
    if (!kv || (kv.type !== GgufValueType.STRING)) return undefined;
    return kv.value as string;
  }

  static getMetadataNumber(ctx: GgufContext, key: string): number | undefined {
    const kv = ctx.metadata.get(key);
    if (!kv) return undefined;
    if (typeof kv.value === 'number') return kv.value;
    return undefined;
  }
}

function readHeader(view: DataView, offset: number): GgufHeader {
  const magic = view.getUint32(offset, true);
  const version = view.getUint32(offset + 4, true);
  const tensorCount = Number(view.getBigUint64(offset + 8, true));
  const metadataKvCount = Number(view.getBigUint64(offset + 16, true));
  return { magic, version, tensorCount, metadataKvCount };
}

function readString(view: DataView, offset: number): { value: string; newOffset: number } {
  const length = Number(view.getBigUint64(offset, true));
  offset += 8;
  const bytes = new Uint8Array(view.buffer, view.byteOffset + offset, length);
  const value = new TextDecoder().decode(bytes);
  return { value, newOffset: offset + length };
}

function readKv(view: DataView, offset: number): { kv: GgufKv; newOffset: number } {
  const { value: key, newOffset: afterKey } = readString(view, offset);
  offset = afterKey;

  const type = view.getUint32(offset, true) as GgufValueType;
  offset += 4;

  const { value, newOffset } = readValue(view, offset, type, false);
  return { kv: { key, type, isArray: false, value }, newOffset };
}

function readValue(view: DataView, offset: number, type: GgufValueType, _isArray: boolean): { value: unknown; newOffset: number } {
  switch (type) {
    case GgufValueType.UINT8:
      return { value: view.getUint8(offset), newOffset: offset + 1 };
    case GgufValueType.INT8:
      return { value: view.getInt8(offset), newOffset: offset + 1 };
    case GgufValueType.UINT16:
      return { value: view.getUint16(offset, true), newOffset: offset + 2 };
    case GgufValueType.INT16:
      return { value: view.getInt16(offset, true), newOffset: offset + 2 };
    case GgufValueType.UINT32:
      return { value: view.getUint32(offset, true), newOffset: offset + 4 };
    case GgufValueType.INT32:
      return { value: view.getInt32(offset, true), newOffset: offset + 4 };
    case GgufValueType.FLOAT32:
      return { value: view.getFloat32(offset, true), newOffset: offset + 4 };
    case GgufValueType.BOOL:
      return { value: view.getUint8(offset) !== 0, newOffset: offset + 1 };
    case GgufValueType.STRING: {
      return readString(view, offset);
    }
    case GgufValueType.ARRAY: {
      const elemType = view.getUint32(offset, true) as GgufValueType;
      offset += 4;
      const count = Number(view.getBigUint64(offset, true));
      offset += 8;
      const arr: unknown[] = [];
      for (let i = 0; i < count; i++) {
        const { value, newOffset } = readValue(view, offset, elemType, true);
        arr.push(value);
        offset = newOffset;
      }
      return { value: arr, newOffset: offset };
    }
    case GgufValueType.UINT64:
      return { value: Number(view.getBigUint64(offset, true)), newOffset: offset + 8 };
    case GgufValueType.INT64:
      return { value: Number(view.getBigInt64(offset, true)), newOffset: offset + 8 };
    case GgufValueType.FLOAT64:
      return { value: view.getFloat64(offset, true), newOffset: offset + 8 };
    default:
      throw new Error(`Unknown GGUF value type: ${type}`);
  }
}

function readTensorInfo(view: DataView, offset: number): { info: GgufTensorInfo; newOffset: number } {
  const { value: name, newOffset: afterName } = readString(view, offset);
  offset = afterName;

  const nDimensions = view.getUint32(offset, true);
  offset += 4;

  const dimensions: number[] = [];
  for (let i = 0; i < nDimensions; i++) {
    dimensions.push(Number(view.getBigUint64(offset, true)));
    offset += 8;
  }

  const type = view.getUint32(offset, true);
  offset += 4;

  const tensorOffset = Number(view.getBigUint64(offset, true));
  offset += 8;

  return {
    info: { name, nDimensions, dimensions, type, offset: tensorOffset },
    newOffset: offset,
  };
}

function getAlignment(metadata: Map<string, GgufKv>): number {
  const kv = metadata.get('general.alignment');
  if (kv && typeof kv.value === 'number') return kv.value;
  return GGUF_DEFAULT_ALIGNMENT;
}

function alignTo(offset: number, alignment: number): number {
  return Math.ceil(offset / alignment) * alignment;
}

function calculateTotalDataSize(tensors: GgufTensorInfo[]): number {
  if (tensors.length === 0) return 0;
  let size = 0;
  for (const t of tensors) {
    const elemCount = t.dimensions.reduce((a, b) => a * b, 1);
    const typeSize = ggmlTypeSize(t.type);
    size = Math.max(size, t.offset + elemCount * typeSize);
  }
  return size;
}

function ggmlTypeSize(type: number): number {
  // Quantized types store multiple elements per block
  const sizes: Record<number, number> = {
    0: 4,   // f32
    1: 2,   // f16
    2: 0.5, // q4_0 (18 bytes per 32 elements = 0.5625 bytes/elem)
    3: 0.5, // q4_1
    6: 1,   // q8_0
    7: 1,   // q8_1
  };
  return sizes[type] ?? 4;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/gguf-parser.test.ts
```

Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/models/gguf-parser.ts tests/gguf-parser.test.ts
git commit -m "feat: implement GGUF binary format parser"
```

---

### Task 4: WebGPU Buffer Pool Manager

**Files:**
- Create: `src/core/memory-pool.ts`
- Test: `tests/memory-pool.test.ts`

- [ ] **Step 1: Write tests for MemoryPool**

Create `tests/memory-pool.test.ts`:
```typescript
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
    const low = pool.allocate(512, 2);  // priority 2
    const high = pool.allocate(512, 0); // priority 0
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/memory-pool.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement MemoryPool**

Create `src/core/memory-pool.ts`:
```typescript
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

  get budget(): number {
    return this._budget;
  }

  get usedBytes(): number {
    return this._usedBytes;
  }

  get remainingBytes(): number {
    return this._budget - this._usedBytes;
  }

  get pressureRatio(): number {
    return this._usedBytes / this._budget;
  }

  get isUnderPressure(): boolean {
    return this.pressureRatio > 0.75;
  }

  get allocationCount(): number {
    return this.allocations.size;
  }

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

  canAllocate(size: number): boolean {
    return size <= this.remainingBytes;
  }

  evictForAllocation(neededSize: number): number | null {
    const candidates = [...this.allocations.values()]
      .filter((a) => !a.freed)
      .sort((a, b) => b.priority - a.priority); // highest priority number = lowest priority

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/memory-pool.test.ts
```

Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/memory-pool.ts tests/memory-pool.test.ts
git commit -m "feat: implement GPU buffer memory pool manager"
```

---

### Task 5: Pipeline Cache (IndexedDB-backed)

**Files:**
- Create: `src/core/pipeline-cache.ts`
- Test: `tests/pipeline-cache.test.ts`

- [ ] **Step 1: Write tests for PipelineCache**

Create `tests/pipeline-cache.test.ts`:
```typescript
import { describe, expect, test } from 'bun:test';
import { PipelineCache } from '../src/core/pipeline-cache.js';

describe('PipelineCache', () => {
  test('stores and retrieves pipeline data', async () => {
    const cache = new PipelineCache('test-cache');
    const data = new Uint8Array([1, 2, 3, 4]);
    await cache.put('pipeline-matmul-q4_0', data);

    const retrieved = await cache.get('pipeline-matmul-q4_0');
    expect(retrieved).toEqual(data);
  });

  test('returns undefined for missing key', async () => {
    const cache = new PipelineCache('test-cache-missing');
    const result = await cache.get('nonexistent');
    expect(result).toBeUndefined();
  });

  test('deletes a cached pipeline', async () => {
    const cache = new PipelineCache('test-cache-delete');
    await cache.put('to-delete', new Uint8Array([5, 6]));
    await cache.delete('to-delete');
    const result = await cache.get('to-delete');
    expect(result).toBeUndefined();
  });

  test('lists all cached pipeline keys', async () => {
    const cache = new PipelineCache('test-cache-list');
    await cache.put('a', new Uint8Array([1]));
    await cache.put('b', new Uint8Array([2]));
    const keys = await cache.keys();
    expect(keys).toContain('a');
    expect(keys).toContain('b');
  });

  test('clears all cached data', async () => {
    const cache = new PipelineCache('test-cache-clear');
    await cache.put('x', new Uint8Array([1]));
    await cache.put('y', new Uint8Array([2]));
    await cache.clear();
    const keys = await cache.keys();
    expect(keys).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/pipeline-cache.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement PipelineCache**

Create `src/core/pipeline-cache.ts`:
```typescript
export class PipelineCache {
  private dbName: string;
  private storeName = 'pipelines';
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

  async get(key: string): Promise<Uint8Array | undefined> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      const request = store.get(key);
      request.onsuccess = () => {
        resolve(request.result ?? undefined);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async put(key: string, data: Uint8Array): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.put(data, key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async delete(key: string): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async keys(): Promise<string[]> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readonly');
      const store = tx.objectStore(this.storeName);
      const request = store.getAllKeys();
      request.onsuccess = () => resolve(request.result as string[]);
      request.onerror = () => reject(request.error);
    });
  }

  async clear(): Promise<void> {
    const db = await this.getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(this.storeName, 'readwrite');
      const store = tx.objectStore(this.storeName);
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/pipeline-cache.test.ts
```

Expected: PASS (all 5 tests). Note: IndexedDB tests may need a browser-like environment; if Bun doesn't support IndexedDB natively, we'll need to mock it or use a test harness.

- [ ] **Step 5: Commit**

```bash
git add src/core/pipeline-cache.ts tests/pipeline-cache.test.ts
git commit -m "feat: implement IndexedDB-backed pipeline cache"
```

---

### Task 6: Token Sampler

**Files:**
- Create: `src/inference/sampler.ts`
- Test: `tests/sampler.test.ts`

Reference: `/Users/probello/Repos/llama.cpp/src/llama-sampler.cpp`

- [ ] **Step 1: Write tests for Sampler**

Create `tests/sampler.test.ts`:
```typescript
import { describe, expect, test } from 'bun:test';
import { Sampler } from '../src/inference/sampler.js';

describe('Sampler', () => {
  test('greedy sampling picks highest logit', () => {
    const sampler = new Sampler({ temperature: 0 });
    const logits = new Float32Array([0.1, 0.5, 0.9, 0.3]);
    const token = sampler.sample(logits);
    expect(token).toBe(2); // index of 0.9
  });

  test('temperature scaling works', () => {
    const sampler = new Sampler({ temperature: 2.0 });
    const logits = new Float32Array([1.0, 2.0]);
    const scaled = sampler.applyTemperature(logits);
    expect(scaled[0]).toBeCloseTo(0.5, 5);
    expect(scaled[1]).toBeCloseTo(1.0, 5);
  });

  test('top-k filters to k highest logits', () => {
    const sampler = new Sampler({ temperature: 1.0, topK: 2 });
    const logits = new Float32Array([0.1, 0.5, 0.9, 0.3]);
    const filtered = sampler.applyTopK(logits);
    // After top-k, only indices 1 and 2 should remain
    expect(filtered[0]).toBe(-Infinity);
    expect(filtered[1]).toBe(0.5);
    expect(filtered[2]).toBe(0.9);
    expect(filtered[3]).toBe(-Infinity);
  });

  test('top-p filters by cumulative probability', () => {
    const sampler = new Sampler({ temperature: 1.0, topP: 0.5 });
    const logits = new Float32Array([0.1, 0.2, 3.0, 0.1]); // index 2 dominates
    const filtered = sampler.applyTopP(logits);
    expect(filtered[2]).toBe(3.0); // dominant token survives
    // Others should be filtered
  });

  test('repetition penalty penalizes repeated tokens', () => {
    const sampler = new Sampler({ temperature: 1.0, repetitionPenalty: 1.5 });
    const logits = new Float32Array([1.0, 2.0, 3.0]);
    const recentTokens = [2]; // token 2 was just generated
    sampler.applyRepetitionPenalty(logits, recentTokens);
    // Token 2 (positive logit) should be divided by penalty
    expect(logits[2]).toBeCloseTo(2.0, 5); // 3.0 / 1.5
  });

  test('deterministic with fixed seed', () => {
    const sampler1 = new Sampler({ temperature: 1.0, seed: 42 });
    const sampler2 = new Sampler({ temperature: 1.0, seed: 42 });
    const logits = new Float32Array([0.1, 0.5, 0.9, 0.3, 0.7]);

    const results1: number[] = [];
    const results2: number[] = [];
    for (let i = 0; i < 100; i++) {
      results1.push(sampler1.sample(new Float32Array(logits)));
      results2.push(sampler2.sample(new Float32Array(logits)));
    }
    expect(results1).toEqual(results2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/sampler.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement Sampler**

Create `src/inference/sampler.ts`:
```typescript
export interface SamplerConfig {
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  seed?: number;
}

export class Sampler {
  private temperature: number;
  private topK: number;
  private topP: number;
  private repetitionPenalty: number;
  private rng: () => number;

  constructor(config: SamplerConfig = {}) {
    this.temperature = config.temperature ?? 1.0;
    this.topK = config.topK ?? 0; // 0 = disabled
    this.topP = config.topP ?? 1.0;
    this.repetitionPenalty = config.repetitionPenalty ?? 1.0;

    if (config.seed !== undefined) {
      // Simple seeded PRNG (xoshiro128**)
      let s0 = config.seed;
      let s1 = config.seed ^ 0xDEADBEEF;
      let s2 = config.seed ^ 0xCAFEBABE;
      let s3 = config.seed ^ 0x12345678;
      this.rng = () => {
        const result = Math.imul(s0, 5) ^ (s0 << 7) ^ (s0 << 13);
        const t = s1 << 9;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = (s3 << 11) | (s3 >>> 21);
        return (result >>> 0) / 4294967296;
      };
    } else {
      this.rng = Math.random;
    }
  }

  sample(logits: Float32Array): number {
    if (this.temperature === 0) {
      return argmax(logits);
    }

    const scaled = this.applyTemperature(logits);
    const filtered = this.applyTopK(scaled);
    const topPFiltered = this.applyTopP(filtered);

    // Softmax to get probabilities
    const probs = softmax(topPFiltered);

    // Weighted random sampling
    let r = this.rng();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  applyTemperature(logits: Float32Array): Float32Array {
    if (this.temperature === 1.0) return new Float32Array(logits);
    const result = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      result[i] = logits[i] / this.temperature;
    }
    return result;
  }

  applyTopK(logits: Float32Array): Float32Array {
    if (this.topK === 0 || this.topK >= logits.length) return new Float32Array(logits);

    const indices = Array.from({ length: logits.length }, (_, i) => i);
    indices.sort((a, b) => logits[b] - logits[a]);

    const result = new Float32Array(logits.length).fill(-Infinity);
    for (let i = 0; i < this.topK; i++) {
      result[indices[i]] = logits[indices[i]];
    }
    return result;
  }

  applyTopP(logits: Float32Array): Float32Array {
    if (this.topP >= 1.0) return new Float32Array(logits);

    const probs = softmax(logits);
    const indices = Array.from({ length: probs.length }, (_, i) => i);
    indices.sort((a, b) => probs[b] - probs[a]);

    let cumulative = 0;
    const result = new Float32Array(logits.length).fill(-Infinity);
    for (const idx of indices) {
      cumulative += probs[idx];
      result[idx] = logits[idx];
      if (cumulative >= this.topP) break;
    }
    return result;
  }

  applyRepetitionPenalty(logits: Float32Array, recentTokens: number[]): void {
    if (this.repetitionPenalty === 1.0) return;
    const seen = new Set(recentTokens);
    for (const idx of seen) {
      if (idx < 0 || idx >= logits.length) continue;
      if (logits[idx] > 0) {
        logits[idx] /= this.repetitionPenalty;
      } else {
        logits[idx] *= this.repetitionPenalty;
      }
    }
  }
}

function argmax(arr: Float32Array): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits);
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = exps[i] / sum;
  }
  return probs;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/sampler.test.ts
```

Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/inference/sampler.ts tests/sampler.test.ts
git commit -m "feat: implement token sampler with top-k, top-p, temperature, repetition penalty"
```

---

### Task 7: Stream Router (Async Generator Token Streaming)

**Files:**
- Create: `src/inference/stream-router.ts`
- Test: `tests/stream-router.test.ts`

- [ ] **Step 1: Write tests for StreamRouter**

Create `tests/stream-router.test.ts`:
```typescript
import { describe, expect, test } from 'bun:test';
import { StreamRouter } from '../src/inference/stream-router.js';

describe('StreamRouter', () => {
  test('emits tokens to registered consumer', async () => {
    const router = new StreamRouter<string>();
    const consumer = router.createConsumer('test');

    router.emit('test', 'Hello');
    router.emit('test', ' ');
    router.emit('test', 'world');
    router.close('test');

    const tokens: string[] = [];
    for await (const token of consumer) {
      tokens.push(token);
    }
    expect(tokens).toEqual(['Hello', ' ', 'world']);
  });

  test('supports multiple consumers', async () => {
    const router = new StreamRouter<string>();
    const consumer1 = router.createConsumer('c1');
    const consumer2 = router.createConsumer('c2');

    router.emit('c1', 'a');
    router.emit('c2', 'b');
    router.close('c1');
    router.close('c2');

    const tokens1: string[] = [];
    for await (const token of consumer1) { tokens1.push(token); }

    const tokens2: string[] = [];
    for await (const token of consumer2) { tokens2.push(token); }

    expect(tokens1).toEqual(['a']);
    expect(tokens2).toEqual(['b']);
  });

  test('interrupt cancels stream', async () => {
    const router = new StreamRouter<string>();
    const consumer = router.createConsumer('test');

    router.emit('test', 'start');
    router.interrupt('test');

    const tokens: string[] = [];
    for await (const token of consumer) { tokens.push(token); }
    expect(tokens).toEqual(['start']);
  });

  test('removeConsumer cleans up', () => {
    const router = new StreamRouter<string>();
    router.createConsumer('test');
    router.removeConsumer('test');
    expect(router.hasConsumer('test')).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/stream-router.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement StreamRouter**

Create `src/inference/stream-router.ts`:
```typescript
export class StreamRouter<T> {
  private streams = new Map<string, { push: (value: T) => void; close: () => void; interrupt: () => void }>();

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

    const self = this;
    async function* generator(): AsyncGenerator<T> {
      while (!closed && !interrupted) {
        if (queue.length > 0) {
          yield queue.shift()!;
          continue;
        }
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
    const stream = this.streams.get(id);
    if (stream) stream.push(value);
  }

  close(id: string): void {
    const stream = this.streams.get(id);
    if (stream) {
      stream.close();
      this.streams.delete(id);
    }
  }

  interrupt(id: string): void {
    const stream = this.streams.get(id);
    if (stream) {
      stream.interrupt();
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/stream-router.test.ts
```

Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/inference/stream-router.ts tests/stream-router.test.ts
git commit -m "feat: implement stream router with async generator token streaming"
```

---

### Task 8: Multi-Model Priority Scheduler

**Files:**
- Create: `src/core/scheduler.ts`
- Test: `tests/scheduler.test.ts`

- [ ] **Step 1: Write tests for Scheduler**

Create `tests/scheduler.test.ts`:
```typescript
import { describe, expect, test } from 'bun:test';
import { Scheduler, type ScheduledTask } from '../src/core/scheduler.js';

describe('Scheduler', () => {
  test('executes highest priority task first', () => {
    const scheduler = new Scheduler({ frameBudgetMs: 16 });
    const executed: number[] = [];

    scheduler.enqueue({
      id: 'low',
      priority: 2,
      execute: async () => { executed.push(2); },
    });
    scheduler.enqueue({
      id: 'high',
      priority: 0,
      execute: async () => { executed.push(0); },
    });
    scheduler.enqueue({
      id: 'mid',
      priority: 1,
      execute: async () => { executed.push(1); },
    });

    scheduler.runCycle();
    expect(executed).toEqual([0, 1, 2]);
  });

  test('respects frame budget and stops when exceeded', () => {
    const scheduler = new Scheduler({ frameBudgetMs: 1 });
    let callCount = 0;

    for (let i = 0; i < 10; i++) {
      scheduler.enqueue({
        id: `task-${i}`,
        priority: 0,
        execute: async () => {
          callCount++;
          // Simulate work by spinning
          const start = performance.now();
          while (performance.now() - start < 0.5) { /* spin */ }
        },
      });
    }

    scheduler.runCycle();
    // Should not have executed all 10 tasks within 1ms budget
    expect(callCount).toBeLessThan(10);
  });

  test('preempts lower priority when higher arrives', () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    const executed: string[] = [];

    scheduler.enqueue({
      id: 'background',
      priority: 2,
      execute: async () => { executed.push('background'); },
    });

    // High priority arrives mid-cycle
    scheduler.enqueue({
      id: 'urgent',
      priority: 0,
      execute: async () => { executed.push('urgent'); },
    });

    scheduler.runCycle();
    expect(executed[0]).toBe('urgent');
  });

  test('removes task by id', () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    const executed: string[] = [];

    scheduler.enqueue({
      id: 'task-a',
      priority: 0,
      execute: async () => { executed.push('a'); },
    });
    scheduler.enqueue({
      id: 'task-b',
      priority: 0,
      execute: async () => { executed.push('b'); },
    });

    scheduler.dequeue('task-a');
    scheduler.runCycle();

    expect(executed).toEqual(['b']);
  });

  test('reports pending task count', () => {
    const scheduler = new Scheduler({ frameBudgetMs: 100 });
    expect(scheduler.pendingCount).toBe(0);

    scheduler.enqueue({ id: 'a', priority: 0, execute: async () => {} });
    scheduler.enqueue({ id: 'b', priority: 1, execute: async () => {} });

    expect(scheduler.pendingCount).toBe(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
bun test tests/scheduler.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement Scheduler**

Create `src/core/scheduler.ts`:
```typescript
export interface ScheduledTask {
  id: string;
  priority: number; // lower = higher priority
  execute: () => Promise<void>;
}

export interface SchedulerConfig {
  frameBudgetMs: number;
}

export class Scheduler {
  private queue: ScheduledTask[] = [];
  private frameBudgetMs: number;

  constructor(config: SchedulerConfig) {
    this.frameBudgetMs = config.frameBudgetMs;
  }

  get pendingCount(): number {
    return this.queue.length;
  }

  enqueue(task: ScheduledTask): void {
    this.queue.push(task);
    this.queue.sort((a, b) => a.priority - b.priority); // lower number = higher priority
  }

  dequeue(id: string): boolean {
    const idx = this.queue.findIndex((t) => t.id === id);
    if (idx === -1) return false;
    this.queue.splice(idx, 1);
    return true;
  }

  runCycle(): void {
    const deadline = performance.now() + this.frameBudgetMs;

    while (this.queue.length > 0 && performance.now() < deadline) {
      const task = this.queue.shift();
      if (task) {
        task.execute();
      }
    }
  }

  clear(): void {
    this.queue.length = 0;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
bun test tests/scheduler.test.ts
```

Expected: PASS (all 5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/scheduler.ts tests/scheduler.test.ts
git commit -m "feat: implement priority-based multi-model scheduler"
```

---

### Task 9: WASM Bridge — C Bridge API

**Files:**
- Create: `src/wasm/CMakeLists.txt`
- Create: `src/wasm/webgpu-bridge.cpp`
- Create: `src/wasm/exports.def`

This is the C bridge that exposes ggml-webgpu operations to JavaScript via Emscripten. It links against llama.cpp's ggml-webgpu backend.

- [ ] **Step 1: Create the Emscripten export definitions**

Create `src/wasm/exports.def`:
```
_exports: [
  "webgpu_init",
  "webgpu_shutdown",
  "webgpu_create_buffer",
  "webgpu_write_buffer",
  "webgpu_read_buffer",
  "webgpu_destroy_buffer",
  "webgpu_mul_mat",
  "webgpu_flash_attn",
  "webgpu_rope",
  "webgpu_rms_norm",
  "webgpu_soft_max",
  "webgpu_cpy",
  "webgpu_get_rows",
  "webgpu_set_rows",
  "webgpu_get_pipeline_cache",
  "webgpu_load_pipeline_cache",
]
```

- [ ] **Step 2: Create CMakeLists.txt for WASM build**

Create `src/wasm/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.13)

# Point to llama.cpp source
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../llama.cpp" CACHE PATH "Path to llama.cpp repo")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}")
endif()

project(webllm-wasm C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Include llama.cpp's ggml CMake as subdirectory
set(GGML_WEBGPU ON CACHE BOOL "" FORCE)
set(GGML_WEBGPU_JSPI ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(LLAMA_WASM_SINGLE_FILE OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

# Our bridge library
add_library(webllm-bridge webgpu-bridge.cpp)
target_link_libraries(webllm-bridge PRIVATE ggml-webgpu)
target_include_directories(webllm-bridge PRIVATE ${LLAMA_CPP_DIR}/ggml/include)
```

- [ ] **Step 3: Create webgpu-bridge.cpp**

Create `src/wasm/webgpu-bridge.cpp`:
```cpp
#include <cstdint>
#include <cstring>
#include <webgpu/webgpu_cpp.h>

extern "C" {

// --- Lifecycle ---

int32_t webgpu_init() {
    // Device is provided by the browser/emdawnwebgpu runtime.
    // The ggml_webgpu backend creates its own device via the registration API.
    // For now, return success. Full init will call ggml_backend_webgpu_reg().
    return 0;
}

void webgpu_shutdown() {
    // Cleanup handled by static destructors in ggml-webgpu
}

// --- Buffer Management ---
// Buffers are identified by integer IDs. The bridge maintains a simple vector
// of GPU buffers. The TypeScript side tracks which ID maps to which tensor.

static constexpr int32_t MAX_BUFFERS = 256;
static wgpu::Buffer buffers[MAX_BUFFERS];
static int32_t next_buffer_id = 0;

int32_t webgpu_create_buffer(uint64_t size, uint32_t usage) {
    if (next_buffer_id >= MAX_BUFFERS) return -1;
    // Buffer creation requires a device, which is obtained from the
    // ggml_webgpu global context. This is a placeholder — the full
    // implementation will use the device from ggml_backend_webgpu_reg().
    int32_t id = next_buffer_id++;
    // buffers[id] = device.CreateBuffer(...);
    return id;
}

void webgpu_write_buffer(int32_t id, const void* data, uint64_t size) {
    // device.GetQueue().WriteBuffer(buffers[id], 0, data, size);
}

void webgpu_read_buffer(int32_t id, void* out, uint64_t size) {
    // Readback requires a staging buffer + map async
}

void webgpu_destroy_buffer(int32_t id) {
    if (id >= 0 && id < MAX_BUFFERS && buffers[id]) {
        buffers[id].Destroy();
        buffers[id] = nullptr;
    }
}

// --- Tensor Operations ---
// These are stubs that will be implemented by wiring into ggml_webgpu's
// graph compute pipeline. The full implementation constructs a ggml_cgraph,
// populates tensor descriptors, and calls ggml_backend_webgpu_graph_compute().

void webgpu_mul_mat(int32_t a, int32_t b, int32_t out,
                    int32_t m, int32_t n, int32_t k,
                    int32_t type_a, int32_t type_b) {
    // TODO: Wire into ggml_webgpu mul_mat shader
}

void webgpu_flash_attn(int32_t q, int32_t k, int32_t v, int32_t out,
                       int32_t head_dim, int32_t n_heads, int32_t seq_len,
                       float scale) {
    // TODO: Wire into ggml_webgpu flash_attn shader
}

void webgpu_rope(int32_t tensor, int32_t freqs, int32_t out,
                 int32_t dim, float freq_base, float freq_scale) {
    // TODO: Wire into ggml_webgpu rope shader
}

void webgpu_rms_norm(int32_t x, int32_t weight, int32_t out,
                     int32_t rows, int32_t cols, float eps) {
    // TODO: Wire into ggml_webgpu rms_norm shader
}

void webgpu_soft_max(int32_t x, int32_t out, int32_t rows, int32_t cols, float scale) {
    // TODO: Wire into ggml_webgpu soft_max shader
}

void webgpu_cpy(int32_t src, int32_t dst, int32_t size) {
    // TODO: Wire into ggml_webgpu cpy shader
}

void webgpu_get_rows(int32_t x, int32_t indices, int32_t out,
                     int32_t n_rows, int32_t row_size) {
    // TODO: Wire into ggml_webgpu get_rows shader
}

void webgpu_set_rows(int32_t x, int32_t indices, int32_t out,
                     int32_t n_rows, int32_t row_size) {
    // TODO: Wire into ggml_webgpu set_rows shader
}

} // extern "C"
```

- [ ] **Step 4: Commit**

```bash
git add src/wasm/
git commit -m "feat: add WASM C bridge stubs for ggml-webgpu tensor operations"
```

---

### Task 10: WASM TypeScript Bindings

**Files:**
- Create: `src/inference/ggml-wasm.ts`

- [ ] **Step 1: Create TypeScript wrapper for WASM module**

Create `src/inference/ggml-wasm.ts`:
```typescript
export interface GgmlWasmConfig {
  wasmUrl: string;
  device: GPUDevice;
}

export class GgmlWasm {
  private wasm: WebAssembly.Exports | null = null;
  private device: GPUDevice | null = null;
  private bufferMap = new Map<number, GPUBuffer>();
  private nextBufferId = 0;

  async init(config: GgmlWasmConfig): Promise<void> {
    this.device = config.device;

    const { instance } = await WebAssembly.instantiateStreaming(
      fetch(config.wasmUrl),
      {
        env: {
          // WebGPU imports — bridge between WASM and browser WebGPU API
          // These are filled by emdawnwebgpu port
        },
      },
    );

    this.wasm = instance.exports;

    const initResult = (this.wasm.webgpu_init as () => number)();
    if (initResult !== 0) {
      throw new Error(`WASM init failed with code ${initResult}`);
    }
  }

  async shutdown(): Promise<void> {
    if (!this.wasm) return;
    (this.wasm.webgpu_shutdown as () => void)();
    for (const [id, buffer] of this.bufferMap) {
      buffer.destroy();
    }
    this.bufferMap.clear();
    this.wasm = null;
    this.device = null;
  }

  createBuffer(size: number, usage: GPUBufferUsageFlags): number {
    if (!this.device) throw new Error('Not initialized');
    const buffer = this.device.createBuffer({ size, usage });
    const id = this.nextBufferId++;
    this.bufferMap.set(id, buffer);
    return id;
  }

  writeBuffer(id: number, data: ArrayBuffer | ArrayBufferView, offset = 0): void {
    if (!this.device) throw new Error('Not initialized');
    const buffer = this.bufferMap.get(id);
    if (!buffer) throw new Error(`Buffer ${id} not found`);
    this.device.queue.writeBuffer(buffer, offset, data);
  }

  async readBuffer(id: number, size: number): Promise<ArrayBuffer> {
    if (!this.device) throw new Error('Not initialized');
    const buffer = this.bufferMap.get(id);
    if (!buffer) throw new Error(`Buffer ${id} not found`);

    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return data;
  }

  destroyBuffer(id: number): void {
    const buffer = this.bufferMap.get(id);
    if (buffer) {
      buffer.destroy();
      this.bufferMap.delete(id);
    }
  }

  mulMat(a: number, b: number, out: number, m: number, n: number, k: number, typeA: number, typeB: number): void {
    if (!this.wasm) throw new Error('Not initialized');
    (this.wasm.webgpu_mul_mat as (...args: number[]) => void)(a, b, out, m, n, k, typeA, typeB);
  }

  flashAttn(q: number, k: number, v: number, out: number, headDim: number, nHeads: number, seqLen: number, scale: number): void {
    if (!this.wasm) throw new Error('Not initialized');
    (this.wasm.webgpu_flash_attn as (...args: number[]) => void)(q, k, v, out, headDim, nHeads, seqLen, scale);
  }

  rope(tensor: number, freqs: number, out: number, dim: number, freqBase: number, freqScale: number): void {
    if (!this.wasm) throw new Error('Not initialized');
    (this.wasm.webgpu_rope as (...args: number[]) => void)(tensor, freqs, out, dim, freqBase, freqScale);
  }

  rmsNorm(x: number, weight: number, out: number, rows: number, cols: number, eps: number): void {
    if (!this.wasm) throw new Error('Not initialized');
    (this.wasm.webgpu_rms_norm as (...args: number[]) => void)(x, weight, out, rows, cols, eps);
  }

  softmax(x: number, out: number, rows: number, cols: number, scale: number): void {
    if (!this.wasm) throw new Error('Not initialized');
    (this.wasm.webgpu_soft_max as (...args: number[]) => void)(x, out, rows, cols, scale);
  }
}
```

- [ ] **Step 2: Run typecheck**

```bash
bun run typecheck
```

Expected: PASS (no type errors in new file).

- [ ] **Step 3: Commit**

```bash
git add src/inference/ggml-wasm.ts
git commit -m "feat: add TypeScript WASM bindings for ggml-webgpu tensor operations"
```

---

### Task 11: Main Engine Entry Point

**Files:**
- Create: `src/core/engine.ts`
- Update: `src/index.ts`

- [ ] **Step 1: Implement WebLLM engine class**

Create `src/core/engine.ts`:
```typescript
import type { WebLLMConfig, ModelLoadOptions, ModelHandle, MemoryPressureEvent, EventHandler } from './types.js';
import { MemoryPool } from './memory-pool.js';
import { Scheduler } from './scheduler.js';
import { PipelineCache } from './pipeline-cache.js';

export class WebLLM {
  private config: WebLLMConfig;
  private memoryPool: MemoryPool;
  private scheduler: Scheduler;
  private pipelineCache: PipelineCache;
  private models = new Map<string, ModelHandle>();
  private eventHandlers = new Map<string, Set<EventHandler>>();

  private constructor(config: WebLLMConfig) {
    this.config = config;
    this.memoryPool = new MemoryPool(config.memoryBudget);
    this.scheduler = new Scheduler({
      frameBudgetMs: config.frameBudgetMs ?? 8,
    });
    this.pipelineCache = new PipelineCache(config.cacheDir ?? 'webllm-cache');
  }

  static async init(config: WebLLMConfig): Promise<WebLLM> {
    const engine = new WebLLM(config);
    return engine;
  }

  async loadModel(name: string, options: ModelLoadOptions): Promise<ModelHandle> {
    const id = `model-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const handle: ModelHandle = {
      id,
      name,
      priority: options.priority,
      lightweight: options.lightweight ?? false,
    };

    this.models.set(id, handle);
    return handle;
  }

  async unloadModel(id: string): Promise<void> {
    this.models.delete(id);
  }

  getMemoryPool(): MemoryPool {
    return this.memoryPool;
  }

  getScheduler(): Scheduler {
    return this.scheduler;
  }

  on(event: string, handler: EventHandler): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  off(event: string, handler: EventHandler): void {
    this.eventHandlers.get(event)?.delete(handler);
  }

  async shutdown(): Promise<void> {
    this.models.clear();
    this.memoryPool.reset();
    this.scheduler.clear();
    this.eventHandlers.clear();
  }
}
```

- [ ] **Step 2: Update src/index.ts with all exports**

Update `src/index.ts`:
```typescript
export { WebLLM } from './core/engine.js';
export { MemoryPool } from './core/memory-pool.js';
export { Scheduler } from './core/scheduler.js';
export { PipelineCache } from './core/pipeline-cache.js';
export { GgufParser } from './models/gguf-parser.js';
export { Sampler } from './inference/sampler.js';
export { StreamRouter } from './inference/stream-router.js';
export { GgmlWasm } from './inference/ggml-wasm.js';

export type { WebLLMConfig, ModelLoadOptions, ModelHandle, TensorInfo, ModelMetadata, ModelArchitecture, GgmlType, EventHandler, MemoryPressureEvent } from './core/types.js';
export type { GgufContext, GgufHeader, GgufKv, GgufTensorInfo } from './models/gguf-types.js';
export type { SamplerConfig } from './inference/sampler.js';
```

- [ ] **Step 3: Run typecheck and all tests**

```bash
bun run typecheck && bun test
```

Expected: typecheck passes, all tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: implement WebLLM engine entry point with full public API"
```

---

### Task 12: Final Verification and Checkall

- [ ] **Step 1: Run full check suite**

```bash
bun run checkall
```

Expected: All formatting, linting, type checking, and tests pass.

- [ ] **Step 2: Fix any issues found by checkall**

If any linter or type errors are found, fix them and re-run.

- [ ] **Step 3: Verify project structure matches spec**

```bash
find src tests -type f -name '*.ts' | sort
```

Expected output should include all files listed in the File Structure section above.

- [ ] **Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: resolve checkall issues from Phase 1"
```

---

## Phase 2-6 Outline

These phases will be detailed in subsequent plans after Phase 1 is verified end-to-end.

### Phase 2: Inference Pipeline
- Tokenizer (SentencePiece + BPE) — port from `llama.cpp/src/llama-vocab.cpp`
- KV Cache Manager (paged allocation) — port from `llama.cpp/src/llama-kv-cache.cpp`
- Autoregressive generation loop wiring
- Wire ggml-webgpu WASM tensor ops into actual graph computation

### Phase 3: Multi-Model Scheduling
- Frame-budget-aware scheduling with `requestAnimationFrame`
- Memory pressure eviction
- KV cache sharing for system prompts

### Phase 4: Lightweight WGSL Path
- Pure WGSL shaders for matmul, norms, activations, embeddings
- Direct WebGPU buffer management (no WASM)

### Phase 5: Character System + Developer API
- Character class, tool/function calling, memory
- Polish TypeScript types and JSDoc

### Phase 6: Reference Apps + Polish
- Chat demo, game-loop demo, multi-model demo
- Performance benchmarking suite
- Documentation
