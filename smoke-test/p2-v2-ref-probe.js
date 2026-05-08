// src/inference/llama-bridge.ts
function createLlamaBridge(mod) {
  let is64 = false;
  try {
    const probe = mod._bridge_malloc(0);
    is64 = typeof probe === "bigint";
    mod._bridge_free(probe);
  } catch {
    const probe = mod._bridge_malloc(0n);
    is64 = true;
    mod._bridge_free(probe);
  }
  const to64 = is64 ? (n) => BigInt(n) : (n) => n;
  const from64 = is64 ? (v) => Number(v) : (v) => v >>> 0;
  const malloc = (size) => from64(mod._bridge_malloc(to64(size)));
  const free = (ptr) => {
    mod._bridge_free(to64(ptr));
  };
  return {
    async loadModel(buf) {
      const ptr = malloc(buf.byteLength);
      if (ptr === 0) {
        throw new Error("webllm: bridge_malloc failed for GGUF buffer");
      }
      try {
        mod.HEAPU8.set(buf, ptr);
        const handle = from64(await mod._webllm_load_model(to64(ptr), to64(buf.byteLength)));
        if (handle === 0) {
          throw new Error("webllm: webllm_load_model returned null");
        }
        return handle;
      } finally {
        free(ptr);
      }
    },
    freeModel(handle) {
      mod._webllm_free_model(to64(handle));
    },
    async createContext(model, params) {
      const handle = from64(await mod._webllm_create_context(to64(model), params.nCtx, params.embeddings ? 1 : 0, params.poolingType ?? 0, params.flashAttn ? 1 : 0));
      if (handle === 0) {
        throw new Error("webllm: webllm_create_context returned null");
      }
      return handle;
    },
    freeContext(ctx) {
      mod._webllm_free_context(to64(ctx));
    },
    async decode(ctx, tokens, pastLen) {
      const ptr = malloc(tokens.byteLength);
      if (ptr === 0) {
        throw new Error("webllm: bridge_malloc failed for decode tokens");
      }
      try {
        new Int32Array(mod.HEAPU8.buffer, ptr, tokens.length).set(tokens);
        return await mod._webllm_decode(to64(ctx), to64(ptr), tokens.length, pastLen);
      } finally {
        free(ptr);
      }
    },
    async getLogits(ctx, model, ith = -1) {
      const ptr = from64(await mod._webllm_get_logits(to64(ctx), ith));
      if (ptr === 0) {
        throw new Error("webllm: webllm_get_logits returned null");
      }
      const nVocab = mod._webllm_n_vocab(to64(model));
      return new Float32Array(mod.HEAPU8.buffer, ptr, nVocab);
    },
    nVocab(model) {
      return mod._webllm_n_vocab(to64(model));
    },
    tokenize(model, text, options) {
      const addBos = options?.addBos ? 1 : 0;
      const parseSpecial = options?.parseSpecial !== false ? 1 : 0;
      const utf8 = new TextEncoder().encode(text);
      const textPtr = malloc(utf8.byteLength);
      if (textPtr === 0) {
        throw new Error("webllm: bridge_malloc failed for tokenize text");
      }
      try {
        mod.HEAPU8.set(utf8, textPtr);
        let cap = Math.max(16, utf8.byteLength + 8);
        let tokensPtr = malloc(cap * 4);
        if (tokensPtr === 0) {
          throw new Error("webllm: bridge_malloc failed for tokenize tokens");
        }
        try {
          let n = mod._webllm_tokenize(to64(model), to64(textPtr), utf8.byteLength, to64(tokensPtr), cap, addBos, parseSpecial);
          if (n < 0) {
            const required = -n;
            free(tokensPtr);
            cap = required;
            tokensPtr = malloc(cap * 4);
            if (tokensPtr === 0) {
              throw new Error("webllm: bridge_malloc failed for tokenize retry");
            }
            n = mod._webllm_tokenize(to64(model), to64(textPtr), utf8.byteLength, to64(tokensPtr), cap, addBos, parseSpecial);
            if (n < 0) {
              throw new Error(`webllm: tokenize returned ${n} after retry (required ${required})`);
            }
          }
          return new Int32Array(mod.HEAPU8.buffer.slice(tokensPtr, tokensPtr + n * 4));
        } finally {
          free(tokensPtr);
        }
      } finally {
        free(textPtr);
      }
    },
    detokenize(model, tokens) {
      const tokensPtr = malloc(tokens.byteLength);
      if (tokensPtr === 0) {
        throw new Error("webllm: bridge_malloc failed for detokenize tokens");
      }
      try {
        new Int32Array(mod.HEAPU8.buffer, tokensPtr, tokens.length).set(tokens);
        let cap = Math.max(64, tokens.length * 4 + 8);
        let textPtr = malloc(cap);
        if (textPtr === 0) {
          throw new Error("webllm: bridge_malloc failed for detokenize text");
        }
        try {
          let n = mod._webllm_detokenize(to64(model), to64(tokensPtr), tokens.length, to64(textPtr), cap);
          if (n < 0) {
            const required = -n;
            free(textPtr);
            cap = required;
            textPtr = malloc(cap);
            if (textPtr === 0) {
              throw new Error("webllm: bridge_malloc failed for detokenize retry");
            }
            n = mod._webllm_detokenize(to64(model), to64(tokensPtr), tokens.length, to64(textPtr), cap);
            if (n < 0) {
              throw new Error(`webllm: detokenize returned ${n} after retry (required ${required})`);
            }
          }
          const bytes = new Uint8Array(mod.HEAPU8.buffer.slice(textPtr, textPtr + n));
          return new TextDecoder().decode(bytes);
        } finally {
          free(textPtr);
        }
      } finally {
        free(tokensPtr);
      }
    },
    tokenBos(model) {
      return mod._webllm_token_bos(to64(model));
    },
    tokenEos(model) {
      return mod._webllm_token_eos(to64(model));
    },
    getMetadata(model, key) {
      const utf8 = new TextEncoder().encode(`${key}\x00`);
      const keyPtr = malloc(utf8.byteLength);
      if (keyPtr === 0) {
        throw new Error("webllm: bridge_malloc failed for metadata key");
      }
      try {
        mod.HEAPU8.set(utf8, keyPtr);
        const required = mod._webllm_get_metadata(to64(model), to64(keyPtr), to64(0), 0);
        if (required < 0)
          return null;
        const cap = required + 1;
        const bufPtr = malloc(cap);
        if (bufPtr === 0) {
          throw new Error("webllm: bridge_malloc failed for metadata buf");
        }
        try {
          const n = mod._webllm_get_metadata(to64(model), to64(keyPtr), to64(bufPtr), cap);
          if (n < 0)
            return null;
          return new TextDecoder().decode(new Uint8Array(mod.HEAPU8.buffer.slice(bufPtr, bufPtr + n)));
        } finally {
          free(bufPtr);
        }
      } finally {
        free(keyPtr);
      }
    },
    nCtxTrain(model) {
      return mod._webllm_n_ctx_train(to64(model));
    },
    nEmbd(model) {
      return mod._webllm_n_embd(to64(model));
    },
    nLayer(model) {
      return mod._webllm_n_layer(to64(model));
    },
    nHead(model) {
      return mod._webllm_n_head(to64(model));
    },
    nHeadKv(model) {
      return mod._webllm_n_head_kv(to64(model));
    },
    nCtx(ctx) {
      return mod._webllm_n_ctx(to64(ctx));
    },
    kvSeqRm(ctx, seqId, p0, p1) {
      mod._webllm_kv_seq_rm(to64(ctx), seqId, p0, p1);
    },
    kvClear(ctx) {
      mod._webllm_kv_clear(to64(ctx));
    },
    stateSeqGetSize(ctx, seqId) {
      return mod._webllm_state_seq_get_size(to64(ctx), seqId);
    },
    stateSeqGetData(ctx, seqId) {
      const size = mod._webllm_state_seq_get_size(to64(ctx), seqId);
      if (size === 0)
        return new Uint8Array(0);
      const ptr = malloc(size);
      if (ptr === 0) {
        throw new Error("webllm: bridge_malloc failed for state-seq blob");
      }
      try {
        const n = mod._webllm_state_seq_get_data(to64(ctx), to64(ptr), size, seqId);
        if (n === 0) {
          throw new Error("webllm: state_seq_get_data returned 0 bytes");
        }
        return new Uint8Array(mod.HEAPU8.buffer.slice(ptr, ptr + n));
      } finally {
        free(ptr);
      }
    },
    stateSeqSetData(ctx, blob, destSeqId) {
      if (blob.byteLength === 0)
        return true;
      const ptr = malloc(blob.byteLength);
      if (ptr === 0) {
        throw new Error("webllm: bridge_malloc failed for state-seq restore");
      }
      try {
        mod.HEAPU8.set(blob, ptr);
        const n = mod._webllm_state_seq_set_data(to64(ctx), to64(ptr), blob.byteLength, destSeqId);
        return n > 0;
      } finally {
        free(ptr);
      }
    },
    async getEmbeddings(ctx, model, ith = -1) {
      const ptr = from64(await mod._webllm_get_embeddings(to64(ctx), ith));
      if (ptr === 0) {
        throw new Error("webllm: webllm_get_embeddings returned null");
      }
      const nEmbd = mod._webllm_n_embd(to64(model));
      return new Float32Array(mod.HEAPU8.buffer, ptr, nEmbd);
    }
  };
}

// smoke-test/p2-v2-ref-probe.src.ts
var MODEL_REGISTRY = {
  tinyllama: {
    ggufUrl: "/models/tinyllama-1.1b-chat-q4_0.gguf",
    promptText: "The capital of France is"
  },
  "qwen3-0.6b": {
    ggufUrl: "/models/qwen3-0.6b-q4f16.gguf",
    promptText: "The capital of France is"
  },
  "qwen3-1.7b": {
    ggufUrl: "/models/qwen3-1.7b-q4f16.gguf",
    promptText: "The capital of France is"
  }
};
function resolveModelKey() {
  const params = new URLSearchParams(window.location.search);
  return params.get("model") ?? "tinyllama";
}
var MODEL_KEY = resolveModelKey();
var MODEL_ENTRY = MODEL_REGISTRY[MODEL_KEY];
if (!MODEL_ENTRY) {
  throw new Error(`Unknown model key '${MODEL_KEY}'. Known keys: ${Object.keys(MODEL_REGISTRY).join(", ")}`);
}
var N_GENERATE = 5;
var GGUF_URL = MODEL_ENTRY.ggufUrl;
function log(msg, cls = "") {
  const el = document.getElementById("log");
  if (!el)
    return;
  const line = document.createElement("div");
  if (cls)
    line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  console.log(msg);
}
async function runRefProbe() {
  try {
    log("[1/7] Initializing non-JSEP WASM module...");
    const cacheBust = window.location.search || "";
    const createModule = (await import(`./webllm-wasm.js${cacheBust}`)).default;
    window.__stderrLines = [];
    const mod = await createModule({
      printErr: (s) => {
        window.__stderrLines.push(s);
        console.error(s);
      }
    });
    log("[2/7] Initializing WebGPU backend...");
    const initStatus = await mod._webgpu_init();
    if (initStatus !== 0) {
      log(`webgpu_init returned ${initStatus}`, "fail");
      return;
    }
    log(`[3/7] Fetching ${GGUF_URL}...`);
    const resp = await fetch(GGUF_URL);
    if (!resp.ok) {
      log(`fetch failed: ${resp.status}`, "fail");
      return;
    }
    const buf = new Uint8Array(await resp.arrayBuffer());
    log(`     loaded ${(buf.byteLength / 1024 / 1024).toFixed(1)} MiB`);
    log("[4/7] Loading model + creating context...");
    const bridge = createLlamaBridge(mod);
    const tLoadStart = performance.now();
    const model = await bridge.loadModel(buf);
    const tLoadMs = performance.now() - tLoadStart;
    const vocab = bridge.nVocab(model);
    const ctx = await bridge.createContext(model, { nCtx: 512 });
    log(`     vocab=${vocab}, load=${tLoadMs.toFixed(0)} ms`);
    mod._webllm_enable_node_dump(200);
    const tokenizedPrompt = bridge.tokenize(model, MODEL_ENTRY.promptText, {
      addBos: true,
      parseSpecial: true
    });
    const promptTokenIds = Array.from(tokenizedPrompt);
    log(`     [stage4.36] model=${MODEL_KEY} promptText="${MODEL_ENTRY.promptText}" promptIds=${JSON.stringify(promptTokenIds)}`);
    log(`[5/7] Prefill (${promptTokenIds.length} tokens)...`);
    const promptTokens = new Int32Array(promptTokenIds);
    const tPrefillStart = performance.now();
    const status = await bridge.decode(ctx, promptTokens, 0);
    const tPrefillMs = performance.now() - tPrefillStart;
    if (status !== 0) {
      log(`     prefill failed status=${status}`, "fail");
      bridge.freeContext(ctx);
      bridge.freeModel(model);
      return;
    }
    log(`     prefill ${tPrefillMs.toFixed(0)} ms`);
    log(`[6/7] Greedy ${N_GENERATE}-token decode...`);
    const generatedIds = [];
    let nPast = promptTokenIds.length;
    const tDecodeStart = performance.now();
    const logitsStep0Stats = {
      topId: -1,
      topVal: -Infinity
    };
    for (let step = 0;step < N_GENERATE; ++step) {
      const logits = await bridge.getLogits(ctx, model);
      let topId = 0;
      let topVal = -Infinity;
      for (let i = 0;i < logits.length; i++) {
        if (logits[i] > topVal) {
          topVal = logits[i];
          topId = i;
        }
      }
      if (step === 0) {
        logitsStep0Stats.topId = topId;
        logitsStep0Stats.topVal = topVal;
      }
      generatedIds.push(topId);
      const single = new Int32Array([topId]);
      const dStatus = await bridge.decode(ctx, single, nPast);
      if (dStatus !== 0) {
        log(`     decode step ${step} failed status=${dStatus}`, "fail");
        break;
      }
      nPast++;
    }
    const tDecodeMs = performance.now() - tDecodeStart;
    const perTokenMs = tDecodeMs / N_GENERATE;
    log(`     decode ${tDecodeMs.toFixed(0)} ms (${perTokenMs.toFixed(2)} ms/tok)`);
    log("[7/7] Capturing checkpoints + summary...");
    const checkpointLines = window.__stderrLines.filter((s) => s.includes("[CHECKPOINT"));
    window.__refCheckpoints = checkpointLines;
    log(`CHECKPOINT_COUNT = ${checkpointLines.length}`);
    for (const line of checkpointLines)
      log(line);
    const stage431Pat = /\[CHECKPOINT-FULL idx=(\d+) name=(\S+) n_elements=(\d+) finite=(\d+) mean=(\S+) abs_max=(\S+) abs_min=(\S+) nan=(\d+) inf=(\d+)\]/;
    const stage431Stats = [];
    for (const line of window.__stderrLines) {
      const m = line.match(stage431Pat);
      if (!m)
        continue;
      stage431Stats.push({
        idx: +m[1],
        name: m[2],
        n_elements: +m[3],
        finite: +m[4],
        mean: Number(m[5]),
        abs_max: Number(m[6]),
        abs_min: Number(m[7]),
        nan: +m[8],
        inf: +m[9]
      });
    }
    window.__stage431Stats = stage431Stats;
    log(`STAGE431_STATS_COUNT = ${stage431Stats.length}`);
    for (const s of stage431Stats) {
      log(`[STAGE-4.31] idx=${s.idx} name=${s.name} n=${s.n_elements} finite=${s.finite} mean=${s.mean} abs_max=${s.abs_max} abs_min=${s.abs_min} nan=${s.nan} inf=${s.inf}`);
    }
    log(`MODEL_KEY = ${MODEL_KEY}`);
    log(`PROMPT_TEXT = ${JSON.stringify(MODEL_ENTRY.promptText)}`);
    log(`PROMPT_IDS = ${JSON.stringify(promptTokenIds)}`);
    log(`LOGIT_STATS_STEP0 = ${JSON.stringify(logitsStep0Stats)}`);
    log(`GENERATED_TOKENS = ${JSON.stringify(generatedIds)}`);
    log(`PER_TOKEN_MS = ${perTokenMs.toFixed(2)}`);
    log(`TOTAL_PREFILL_MS = ${tPrefillMs.toFixed(0)}`);
    log(`MODEL_LOAD_MS = ${tLoadMs.toFixed(0)}`);
    log("DONE", "pass");
    try {
      await fetch("http://localhost:8032/STAGE-4.33-ref.txt", {
        method: "POST",
        body: window.__stderrLines.join(`
`)
      });
    } catch (e) {
      console.error("Failed to POST logs:", e);
    }
    window.__refResult = {
      modelKey: MODEL_KEY,
      promptText: MODEL_ENTRY.promptText,
      promptIds: promptTokenIds,
      generatedIds,
      perTokenMs,
      totalPrefillMs: tPrefillMs,
      modelLoadMs: tLoadMs,
      checkpointCount: checkpointLines.length,
      logitStats: logitsStep0Stats
    };
    bridge.freeContext(ctx);
    bridge.freeModel(model);
  } catch (err) {
    const e = err;
    log(`FAIL — ${e.message}
${e.stack ?? ""}`, "fail");
  }
}
runRefProbe();
