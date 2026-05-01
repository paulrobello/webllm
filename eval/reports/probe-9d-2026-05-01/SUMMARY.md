# Probe 9d — Worker-resident hitch

**Model:** qwen3-0.6b-q4f16 (worker-resident vs main-thread control)
**Date:** 2026-05-01
**Calls:** 5
**Build:** post-§27 rebase, tip `e29753286`.

## TL;DR

Moving the inference engine into a `DedicatedWorker` **completely
absorbs the per-call decode hitch** from the main-thread render
loop's perspective. Same-day same-model control:

| Scenario          | call-0 | call-1 | call-2 | call-3 | call-4 | median |
|---|---:|---:|---:|---:|---:|---:|
| Main-thread       | 41.0   | 33.6   | 58.3   | 49.8   | 58.2   | **49.8 ms** |
| Worker-resident   |  9.1   |  9.4   |  9.0   |  9.1   |  9.2   | **9.1 ms**  |

Worker median is **5.5× lower** and sits squarely in the rAF
baseline band. **PASS.** Item 10 (dual-mode main + worker
deployment) is the load-bearing path forward for the agent +
Three.js coexistence case.

## Method

Standalone main page `smoke-test/probe-9d.html` boots a module
Worker (`probe-9d-worker.js`) that imports `./webllm-bundle.js`,
fetches the GGUF as an ArrayBuffer, and instantiates the engine
via `WebLLM.loadModelFromBuffer`. The main thread runs the
frame-probe rAF tracker (idle cube, 120 Hz baseline target) and
dispatches 5 sequential `chatCompletion` calls to the worker via
`postMessage`. The Worker collects all stream chunks internally
and posts a single `chat-done` reply per call (so the rAF loop
measures postMessage round-trip + engine work without per-chunk
message traffic distorting the timing).

Control: a separate `real-model.html?model=qwen3-0.6b-q4f16&
frameProbe=1&frameProbeCalls=5` run for apples-to-apples baseline
on the same model with the engine on the main thread.

## Worker-resident — per-call rAF stats (main-thread perspective)

```
  call 0: prefill_max=9.4ms, decode_p95=8.8ms, decode_max=9.1ms
  call 1: prefill_max=9.4ms, decode_p95=9.3ms, decode_max=9.4ms
  call 2: prefill_max=9.2ms, decode_p95=8.6ms, decode_max=9.0ms
  call 3: prefill_max=8.5ms, decode_p95=8.7ms, decode_max=9.1ms
  call 4: prefill_max=8.9ms, decode_p95=9.2ms, decode_max=9.2ms
```

Aggregate: decode_max min=9.0 / median=9.1 / max=9.4 ms. Zero
`drops50` in any call.

## Main-thread control — per-call rAF stats (same model, same day)

```
  call 0: prefill_max=9.1ms, decode_p95=9.2ms, decode_max=41.0ms (drops50=0)
  call 1: prefill_max=8.4ms, decode_p95=9.3ms, decode_max=33.6ms (drops50=0)
  call 2: prefill_max=8.6ms, decode_p95=9.3ms, decode_max=58.3ms (drops50=1)
  call 3: prefill_max=8.5ms, decode_p95=9.3ms, decode_max=49.8ms (drops50=0)
  call 4: prefill_max=9.3ms, decode_p95=9.2ms, decode_max=58.2ms (drops50=1)
```

Aggregate: decode_max min=33.6 / median=49.8 / max=58.3 ms. The
hitch is deterministic per-call as previously characterized
(probes 9c on qwen3-8b-iq3m and the original frame-probe
multi-call results in TODO § Next session pickup item 8).

## What this tells us about the hitch source

The hitch survives at ~50 ms on every call when the engine runs
on the main thread, and disappears when the engine is moved to
a Worker — even though both threads ultimately submit the same
WebGPU command buffers to the same physical GPU. That rules out
several candidates for the hitch source:

- **NOT GPU contention.** The Worker uses the same GPU as the
  main thread; if the hitch were coming from GPU-queue scheduling
  collision with the rAF cube render, moving the engine to a
  Worker wouldn't help.
- **NOT first-call shape JIT** (already ruled out by probe 9c
  with the warmup throwaway, which didn't move the needle).
- **NOT model-bandwidth-bound decode.** Per-token decode is
  deterministic and would land in the rAF baseline band; the
  spike is a discrete per-call event.

What's *consistent* with the data:

- **Main-thread JS scheduling jitter.** The engine's chatCompletion
  loop runs synchronous JS work (token sampling, KV-cache
  housekeeping, graph building between prefill and decode)
  interleaved with awaits. When this work runs on the same thread
  as rAF, it monopolizes 1-2 frame slots once per call. The fixed
  ~42-58 ms shape suggests a single discrete blocking operation
  per call rather than smeared compute.
- **Likely candidate: prefill→decode graph rebuild + first decode
  step.** The graph topology changes from prefill (multi-token
  forward pass) to decode (single-token forward pass). The
  rebuild + first decode step is ASYNCIFY-driven but its
  awaitable boundaries don't yield to the browser's rAF
  scheduler cleanly enough to keep render frames flowing.

## Verdict

**PASS** — the hitch is **fully absorbed** by moving the engine
to a Worker (5.5× decode_max reduction, hitch goes from 50 ms ×
every call to nothing). The agent + Three.js coexistence target
is **achievable on a 16 GB-floor box for ≥1 Hz NPC tick rates
with smooth 120 Hz rendering** when the engine is worker-resident.

## Downstream decision

**Item 10 (dual-mode main + worker deployment) is now load-bearing
work**, gated only on prefix-cache spec (probe 9a's promotion).
The worker spike here covers about 30% of the dual-mode delivery
work — the remaining work is:

1. Engine init path that auto-detects worker context (the spike's
   worker is a one-off; production needs `WebLLM.initInWorker()`
   or a unified `WebLLM.init()` that DTRT in both contexts).
2. **Postmessage bridge for streaming**: the spike collects all
   chunks worker-side and posts once. Production needs
   `AsyncIterable<ChunkEvent>` reconstruction main-side from
   per-chunk worker postMessages — without re-introducing the
   per-chunk message traffic that would defeat the hitch fix.
3. Heap-streaming loader port: 7B+ models won't fit a single
   ArrayBuffer fetch. The smoke page's HEAPU8 streaming loader
   needs to live in the worker too. Significant LOC.
4. Embedder parity: `engine.embed` paths (encoder, causal-LM
   embedder, bucket D self-embed) all need the worker dispatch
   shim.
5. Smoke + bench parity: `?worker=1` page-level flag on
   `real-model.html`; `--worker` flag on `eval/bench.ts` and
   `eval/perf.ts`.

The spike's results justify investing in (1)-(5). Without this
probe, prioritizing dual-mode work would have been speculative;
with it, the win is measured (5.5× hitch reduction, full hitch
elimination from the render-loop perspective) and the cost is a
known engineering chunk.

## Caveats

- **Smaller model than originally specified.** Probe 9c originally
  observed the hitch on `qwen3-8b-iq3m`. This probe used
  `qwen3-0.6b-q4f16` because the 8B model requires the smoke page's
  heap-streaming loader inside the worker (its 3.6 GB GGUF can't
  be fetched as a single ArrayBuffer). The same-day same-model
  main-thread control above confirms the hitch is identical at
  0.6B (median 49.8 ms vs 8B's 41.7 ms in probe 9c) — the
  structural conclusion generalizes. Production dual-mode work
  needs to verify the result holds on the 8B model, but the
  qualitative answer is clear.
- **Single-run probe.** No averaging across multiple page loads.
  N=5 calls per scenario; verdict robust to noise (worker-resident
  decode_max range 9.0-9.4 vs control 33.6-58.3 — the bands don't
  overlap by 24 ms).
- **Engine init outside the timed window.** Worker model load +
  WebGPU adapter acquisition happens during the page load, not
  during the timed probe. The frame-probe baseline only starts
  after `init-done` fires.
- **Pure spike, not production code.** `probe-9d-worker.js` is
  ~80 LOC and does not handle: `loadModel` cache, multi-model
  registration, drafter / spec-decode wiring, KV-cache
  multiplexing (probe 9a's queued spec), error recovery, or the
  full async-iterable streaming protocol. It exists to answer
  the structural question; production dual-mode work starts from
  scratch in the engine.

## Public API change shipped with this probe

`WebLLM.loadModelFromBuffer` previously ignored
`options.contextLength` — `initKVCache` always used the GGUF's
full max context, which OOMs on `qwen3-0.6b-q4f16` in the worker
(32 K context KV-cache exceeds the 8 GB memory budget). Fixed in
this cycle: `loadModelFromBuffer` now clamps initKVCache to
`min(options.contextLength, parsed.kvCacheConfig.maxContextLength)`
when the option is set. Backwards-compatible — omitting the option
preserves prior behavior. Tests pass (513/12/0).
