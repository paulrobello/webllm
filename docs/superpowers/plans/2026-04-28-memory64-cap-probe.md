# MEMORY64 Cap Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate `webllm-wasm-mem64` build target + standalone harness that answers a single decision-grade question about MEMORY64 viability on the webllm browser stack and produces a closure report citing Chrome's actual heap ceiling on the dev box.

**Architecture:** Mirror the existing `webllm-wasm` CMake target with three flag deltas (`-sMEMORY64=1 -sWASM_BIGINT=1 -sMAXIMUM_MEMORY=16GB`); add a `make mem64-probe` orchestration target; write a self-contained `smoke-test/mem64-probe.html` harness that runs four sequential phases (ASYNCIFY round-trip, BigInt ABI smoke, sequential 1-GiB cap probe, post-probe shutdown health) and emits both a `#log` div and a structured `window.__memory64ProbeResult` JSON blob; execute via the existing agentchrome session; write a closure report and TODO §31 entry citing the decision-rule branch taken.

**Tech Stack:** Emscripten (`-sMEMORY64=1` + `-sWASM_BIGINT=1`), CMake (existing pattern in `src/wasm/CMakeLists.txt`), GNU make (existing pattern in `Makefile`), vanilla HTML + ES module JS (no SDK import, no bundler), agentchrome CLI for browser control.

**Spec:** [`docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md`](../specs/2026-04-28-memory64-cap-probe-design.md) (commit `f479ce5`).

**Project policies in effect** (committed `e23ac03`):
- 30B model ceiling — out-of-scope justifications must not lean on 70B+ targets.
- Probe-first default — this plan implements that policy.
- Complexity ≠ implementation time — score on surface area / risk / reversibility, not duration.
- Always commit before work — every phase below ends with its own commit before the next phase starts.

---

## File structure

| Path | Disposition | Responsibility |
|---|---|---|
| `src/wasm/CMakeLists.txt` | **Modify** | Add `webllm-wasm-mem64` `add_executable` block + flag deltas. Existing `webllm-wasm` target unchanged. |
| `Makefile` | **Modify** | Add `mem64-probe` orchestration target. Add it to `.PHONY`. |
| `smoke-test/mem64-probe.html` | **Create** | Self-contained HTML harness, ~120 LOC (HTML + inline ES module). Loads `webllm-wasm-mem64.js`, runs four probe phases, populates `#log` and `window.__memory64ProbeResult`. |
| `eval/reports/memory64-probe-2026-04-28/SUMMARY.md` | **Create** (Phase 5) | Closure report citing phase outcomes, JSON blob, environment versions, decision-rule branch. |
| `TODO.md` | **Modify** (Phase 5) | Add §31 closure entry referencing the report. |
| `src/wasm/webgpu-bridge.cpp` | **Untouched** | Already wasm64-clean by inspection (`size_t` casts at all pointer arithmetic sites). |
| `src/inference/*.ts` | **Untouched** | Bridge migration is out of scope for this probe. |

---

## Phase 0 — Baseline verification (no commit; preflight only)

**Purpose:** Confirm the existing pipeline is healthy before introducing any change, so a Phase 1 build break is unambiguously caused by Phase 1 work.

### Task 0.1: Confirm checkall green

- [ ] **Step 1: Run `make checkall`**

```bash
make checkall
```

Expected: final lines

```
 428 pass
 11 skip
 0 fail
```

- [ ] **Step 2: If checkall is not 428/11/0, halt the plan**

Investigate before continuing — Phase 0 must be clean. Do not start Phase 1 against a broken baseline.

### Task 0.2: Confirm WASM build is healthy

- [ ] **Step 1: Confirm the patched llama.cpp branch is checked out**

```bash
git -C ~/Repos/llama.cpp rev-parse --abbrev-ref HEAD
```

Expected: `webllm-browser-patches`

- [ ] **Step 2: Confirm the patch tip matches the post-§27 rebase**

```bash
git -C ~/Repos/llama.cpp log --oneline -1 webllm-browser-patches
```

Expected: tip is `981859864 ggml-webgpu: fix UB shift-by-32 in load_u32_at_src{,0} for aligned offsets` (patch 11 / bug #28 fix). If the tip differs, do not proceed — investigate.

- [ ] **Step 3: Confirm the live WASM artifacts exist**

```bash
ls -la smoke-test/webllm-wasm.js smoke-test/webllm-wasm.wasm src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm
```

All four files must exist. If `src/wasm/build/` is missing, run `make wasm-build` once to populate it.

### Task 0.3: Confirm agentchrome session is reachable

Phase 4 needs a live agentchrome session. Confirm it's running now so a Phase 4 surprise doesn't cost the cap-probe execution.

- [ ] **Step 1: Check session status**

```bash
agentchrome connect --status
```

Expected: JSON with `"active": true` and a CDP `port`. If `"active": false`, run `agentchrome connect --launch --headed` (or per project conventions in `~/.claude/CLAUDE.md` — agentchrome SKILL.md). The session must be reachable before Phase 4.

**Phase 0 gate:** Tasks 0.1 / 0.2 / 0.3 all pass. **No commit** for Phase 0 — nothing changed.

---

## Phase 1 — Add `webllm-wasm-mem64` CMake target

**Purpose:** Produce a separate build target that compiles `webgpu-bridge.cpp` + `ggml-webgpu` against `-sMEMORY64=1`. The existing `webllm-wasm` target is unchanged.

### Task 1.1: Add the new target to `src/wasm/CMakeLists.txt`

**Files:**
- Modify: `src/wasm/CMakeLists.txt:46-92` (add a sibling `add_executable(webllm-wasm-mem64 ...)` block in the `if(EMSCRIPTEN)` branch).

- [ ] **Step 1: Read the current `if(EMSCRIPTEN)` block**

```bash
sed -n '46,92p' src/wasm/CMakeLists.txt
```

Confirm the existing structure: `EXPORTED_FUNCTIONS` string built up first, then `add_executable(webllm-wasm webgpu-bridge.cpp)`, then `target_link_libraries`, `target_include_directories`, `target_link_options`, `set_target_properties`.

- [ ] **Step 2: Add the mem64 target inside `if(EMSCRIPTEN)` after the existing `set_target_properties(webllm-wasm ...)` block**

Edit `src/wasm/CMakeLists.txt`. Locate the block:

```cmake
    set_target_properties(webllm-wasm PROPERTIES
        PREFIX ""
    )
else()
```

Replace with:

```cmake
    set_target_properties(webllm-wasm PROPERTIES
        PREFIX ""
    )

    # ── MEMORY64 cap probe target (separate build, dead code on `main`
    # until promoted; spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md).
    # Reuses the same source + same exports list as `webllm-wasm`; deltas:
    #   -sMEMORY64=1            64-bit pointers on the WASM side
    #   -sWASM_BIGINT=1         BigInt at the JS↔WASM boundary
    #   -sMAXIMUM_MEMORY=16GB   probes the upper bound of Chrome's heap grant
    add_executable(webllm-wasm-mem64 webgpu-bridge.cpp)
    target_link_libraries(webllm-wasm-mem64 PRIVATE ggml-base ggml-webgpu)
    target_include_directories(webllm-wasm-mem64 PRIVATE
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/ggml/src
    )
    target_link_options(webllm-wasm-mem64 PRIVATE
        "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}"
        "-sEXPORTED_RUNTIME_METHODS=['stackAlloc','stackSave','stackRestore','stringToUTF8','lengthBytesUTF8','HEAPU8','HEAPF32','cwrap','Asyncify']"
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sUSE_ES6_IMPORT_META=1"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sMEMORY64=1"
        "-sWASM_BIGINT=1"
        "-sMAXIMUM_MEMORY=16GB"
        "-sSTACK_SIZE=8388608"
        "-sENVIRONMENT='web'"
        "-sASYNCIFY_STACK_SIZE=1048576"
        "-O3"
        "-sASSERTIONS=$<IF:$<BOOL:${WEBLLM_ASSERTIONS}>,1,0>"
    )
    set_target_properties(webllm-wasm-mem64 PROPERTIES
        PREFIX ""
    )
else()
```

**Note:** `MAXIMUM_MEMORY=16GB` replaces the `4GB` from the wasm32 target. `MEMORY64=1` and `WASM_BIGINT=1` are the only flag additions; everything else is identical.

- [ ] **Step 3: Run the build to confirm both targets compile**

```bash
source ~/emsdk/emsdk_env.sh && cd src/wasm/build && cmake --build . --config Release -j 2>&1 | tail -40
```

Expected: both `webllm-wasm.{js,wasm}` (untouched) and new `webllm-wasm-mem64.{js,wasm}` produced. No new errors from the mem64 target.

If cmake configure didn't see the new target (it was added after configure), force a re-configure:

```bash
cd src/wasm/build && cmake .. 2>&1 | tail -10 && cmake --build . --config Release -j 2>&1 | tail -40
```

- [ ] **Step 4: Verify both binaries exist**

```bash
ls -la src/wasm/build/webllm-wasm.{js,wasm} src/wasm/build/webllm-wasm-mem64.{js,wasm}
```

All four files must exist. The mem64 wasm will likely be slightly larger than the wasm32 wasm (extra runtime support for 64-bit pointers + BigInt).

**Phase 1 gate (no-regression check):**

- [ ] **Step 5: Confirm `make checkall` still passes 428/11/0**

```bash
make checkall
```

The existing `webllm-wasm` build is the input to this; if checkall regresses, the change to `CMakeLists.txt` somehow affected the wasm32 path. Investigate before committing.

- [ ] **Step 6: Commit**

```bash
git add src/wasm/CMakeLists.txt
git commit -m "$(cat <<'EOF'
build(wasm): add webllm-wasm-mem64 CMake target for MEMORY64 cap probe

Mirrors the existing webllm-wasm target with three flag deltas:
  -sMEMORY64=1        64-bit pointers on the WASM side
  -sWASM_BIGINT=1     BigInt at the JS↔WASM boundary
  -sMAXIMUM_MEMORY=16GB  probes Chrome's heap-grant ceiling

Same source file (webgpu-bridge.cpp), same exports list. Existing
webllm-wasm target unchanged; checkall still 428/11/0; binary is
dead code on main until promoted.

Spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md
EOF
)"
```

**Phase 1 gate satisfied** if both binaries built and checkall stayed 428/11/0.

---

## Phase 2 — Add `make mem64-probe` orchestration target

**Purpose:** One command that builds the mem64 binary, copies it next to the smoke server, restarts the server, and emits a cache-busted URL the user opens via agentchrome.

### Task 2.1: Add the target to `Makefile`

**Files:**
- Modify: `Makefile:1-7` (add `mem64-probe` to `.PHONY`).
- Modify: `Makefile:114-116` (add the `mem64-probe` target body after the existing `smoke-test:` target so it shares the wasm build dependency naturally).

- [ ] **Step 1: Add `mem64-probe` to the `.PHONY` declaration**

Edit `Makefile` line 5. The current line reads:

```makefile
        smoke-test smoke-serve smoke-stop smoke-restart smoke-open smoke-run smoke-bench \
```

Change to:

```makefile
        smoke-test smoke-serve smoke-stop smoke-restart smoke-open smoke-run smoke-bench mem64-probe \
```

- [ ] **Step 2: Add the target body after the `smoke-test:` target**

Locate this block (around line 114):

```makefile
smoke-test: wasm-build ## Bundle + copy WASM artifacts into smoke-test/
	bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
	cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/

smoke-serve: smoke-test ## Serve smoke-test/ on http://localhost:$(SMOKE_PORT)
```

Insert between `smoke-test:` body and `smoke-serve:`:

```makefile
mem64-probe: ## Build mem64 WASM, copy to smoke-test/, restart smoke server, print probe URL
	@source ~/emsdk/emsdk_env.sh 2>/dev/null; \
	cd src/wasm/build && cmake --build . --target webllm-wasm-mem64 --config Release -j
	cp src/wasm/build/webllm-wasm-mem64.js src/wasm/build/webllm-wasm-mem64.wasm smoke-test/
	@lsof -ti:$(SMOKE_PORT) | xargs kill -9 2>/dev/null || true
	@bun run eval/smoke-serve.ts --port $(SMOKE_PORT) >/dev/null 2>&1 &
	@sleep 1
	@echo ""
	@echo "MEMORY64 cap probe ready. Open in your existing agentchrome tab:"
	@echo "  http://localhost:$(SMOKE_PORT)/mem64-probe.html?v=$$(date +%s)"
	@echo ""
	@echo "Result blob will be at window.__memory64ProbeResult after the harness completes."
```

**Note:** This target deliberately does **not** depend on `wasm-build` or `smoke-test` — those would rebuild the wasm32 target which is unrelated. We invoke the mem64-only build via `cmake --build . --target webllm-wasm-mem64`. This requires Phase 1's `cmake` configure to have already run (i.e. `src/wasm/build/CMakeCache.txt` exists). If a future user runs `make mem64-probe` from a clean tree, the configure step needs to happen first; for the probe execution we'll run it after Phase 1 has already configured the build dir.

- [ ] **Step 3: Test the target end-to-end**

The harness file doesn't exist yet (Phase 3) — but the build + copy + server restart + URL emission should work. The 404 on the URL is expected:

```bash
make mem64-probe
```

Expected output (last few lines):

```
MEMORY64 cap probe ready. Open in your existing agentchrome tab:
  http://localhost:8031/mem64-probe.html?v=<timestamp>
```

And:

```bash
ls -la smoke-test/webllm-wasm-mem64.{js,wasm}
```

Both must exist.

- [ ] **Step 4: Confirm `make checkall` still passes**

```bash
make checkall
```

Expected: 428/11/0. Makefile change does not touch any test or `src/` file.

- [ ] **Step 5: Stop the smoke server before committing (so subsequent phases don't fight a stale server)**

```bash
make smoke-stop
```

- [ ] **Step 6: Commit**

```bash
git add Makefile
git commit -m "$(cat <<'EOF'
build(make): add mem64-probe target for MEMORY64 cap probe

One command that builds the mem64 wasm via cmake's named target,
copies the binary to smoke-test/, restarts the smoke server, and
echoes a cache-busted URL for the agentchrome harness page.

Does not depend on smoke-test:/wasm-build: targets — those would
rebuild the unrelated wasm32 binary. Assumes Phase 1 has already
configured src/wasm/build/.

Spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md
EOF
)"
```

**Phase 2 gate satisfied** if `make mem64-probe` produces both binaries in `smoke-test/`, prints the URL, checkall stays 428/11/0, and the change is committed.

---

## Phase 3 — Standalone harness HTML

**Purpose:** Implement the four-phase probe (spec §4) as a self-contained page.

### Task 3.1: Write `smoke-test/mem64-probe.html`

**Files:**
- Create: `smoke-test/mem64-probe.html`

The harness has two responsibilities:
1. Load `webllm-wasm-mem64.js` and exercise four sequential phases per spec §4.1–§4.4.
2. Populate `#log` (human-readable) and `window.__memory64ProbeResult` (machine-readable) with phase outcomes.

- [ ] **Step 1: Create the file with the full harness body**

Write `smoke-test/mem64-probe.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MEMORY64 cap probe</title>
<style>
  body { font-family: ui-monospace, monospace; padding: 1em; background: #1e1e1e; color: #e6e6e6; line-height: 1.4; }
  h1 { color: #2196f3; margin-top: 0; }
  #log { white-space: pre-wrap; font-size: 12px; }
  .ok { color: #4caf50; }
  .fail { color: #f44336; }
  .info { color: #2196f3; }
  .header { color: #ffc107; font-weight: bold; margin-top: 0.6em; }
</style>
</head>
<body>
<h1>MEMORY64 cap probe</h1>
<div id="log"></div>
<script type="module">
const logEl = document.getElementById("log");
const append = (msg, cls = "") => {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  logEl.appendChild(line);
  console.log(msg);
};

const result = {
  phase1: null,
  phase2: null,
  phase3_cap_bytes: null,
  phase3_iterations: null,
  phase4: null,
  user_agent: navigator.userAgent,
  module_bytes: null,
  init_wall_ms: null,
  started_at: new Date().toISOString(),
  finished_at: null,
};
window.__memory64ProbeResult = result;

const ONE_GIB = 1n << 30n;
const benignConsoleErrorPattern = /^adapter_info:/;

// Phase 1 calls webgpu_init/shutdown. Phase 4 re-runs init+shutdown
// after the cap probe, validating the runtime survived heap pressure.
let m;

try {
  append("Loading webllm-wasm-mem64.js…", "info");
  const moduleFactory = (await import(`./webllm-wasm-mem64.js?v=${Date.now()}`)).default;
  m = await moduleFactory();
  result.module_bytes = m.HEAPU8?.buffer?.byteLength ?? null;
  append(`Module loaded. Initial heap: ${result.module_bytes} bytes`, "info");
} catch (e) {
  append(`MODULE LOAD FAILED: ${e?.message || e}`, "fail");
  result.phase1 = `fail: module load — ${e?.message || e}`;
  result.finished_at = new Date().toISOString();
  throw e;
}

// ── Phase 1: ASYNCIFY × MEMORY64 round-trip ──
try {
  append("--- Phase 1: ASYNCIFY × MEMORY64 round-trip ---", "header");
  const t0 = performance.now();
  const rc = await m._webgpu_init();
  const dt = performance.now() - t0;
  result.init_wall_ms = dt;
  if (rc !== 0) throw new Error(`_webgpu_init returned ${rc}`);
  append(`_webgpu_init OK in ${dt.toFixed(1)}ms`, "ok");
  await m._webgpu_shutdown();
  append("_webgpu_shutdown OK", "ok");
  result.phase1 = "ok";
  append("PHASE 1 PASS", "ok");
} catch (e) {
  result.phase1 = `fail: ${e?.message || e}`;
  append(`PHASE 1 FAIL: ${e?.message || e}`, "fail");
}

// ── Phase 2: BigInt ABI smoke ──
// Skipped if Phase 1 failed (per spec §5.1: phase 1 fail closes the lever).
if (result.phase1 === "ok") {
  try {
    append("--- Phase 2: BigInt ABI smoke ---", "header");

    const ctxIdx = m._ctx_create(65536);
    if (ctxIdx < 0) throw new Error(`_ctx_create returned ${ctxIdx}`);
    append(`_ctx_create returned idx=${ctxIdx}`, "info");

    // _tensor_new_1d returns void* — must come back as bigint under wasm64.
    const tensorPtr = m._tensor_new_1d(0 /* GGML_TYPE_F32 */, 4);
    append(`_tensor_new_1d → typeof=${typeof tensorPtr} value=0x${tensorPtr.toString(16)}`, "info");
    if (typeof tensorPtr !== "bigint") {
      throw new Error(`tensor ptr typeof=${typeof tensorPtr}, expected bigint`);
    }
    if (tensorPtr <= 0n) throw new Error(`tensor ptr <= 0 (${tensorPtr})`);

    // _malloc takes size_t — pass BigInt; returned pointer must be BigInt.
    const dataPtr = m._malloc(16n);
    append(`_malloc(16n) → typeof=${typeof dataPtr} value=0x${dataPtr.toString(16)}`, "info");
    if (typeof dataPtr !== "bigint") {
      throw new Error(`data ptr typeof=${typeof dataPtr}, expected bigint`);
    }
    if (dataPtr <= 0n) throw new Error(`data ptr <= 0`);

    // Heap round-trip: write 4 F32, read back, verify byte-equal.
    const writeView = new Float32Array(m.HEAPU8.buffer, Number(dataPtr), 4);
    writeView.set([1, 2, 3, 4]);
    const readView = new Float32Array(m.HEAPU8.buffer, Number(dataPtr), 4);
    const expected = [1, 2, 3, 4];
    for (let i = 0; i < 4; i++) {
      if (readView[i] !== expected[i]) {
        throw new Error(`round-trip mismatch at i=${i}: got ${readView[i]}, expected ${expected[i]}`);
      }
    }
    append("Heap round-trip OK (4×F32 byte-equal)", "ok");

    m._free(dataPtr);
    m._ctx_free();
    result.phase2 = "ok";
    append("PHASE 2 PASS", "ok");
  } catch (e) {
    result.phase2 = `fail: ${e?.message || e}`;
    append(`PHASE 2 FAIL: ${e?.message || e}`, "fail");
  }
} else {
  append("Phase 2 SKIPPED (Phase 1 failed)", "info");
}

// ── Phase 3: Cap probe (sequential 1 GiB malloc with page commit) ──
// Runs even if Phase 2 failed — the cap measurement is independent of
// the BigInt ABI quirk that might have caused Phase 2 to fail. Only
// skipped if Phase 1 failed (no point exercising heap if init broke).
if (result.phase1 === "ok") {
  try {
    append("--- Phase 3: Cap probe (sequential 1 GiB malloc) ---", "header");
    const heldPtrs = [];
    let totalCommitted = 0n;
    let iterations = 0;
    for (let i = 0; i < 16; i++) {
      let ptr;
      try {
        ptr = m._malloc(ONE_GIB);
      } catch (mallocErr) {
        append(`Iter ${i}: _malloc threw — ${mallocErr?.message || mallocErr}`, "info");
        break;
      }
      if (typeof ptr !== "bigint" || ptr === 0n) {
        append(`Iter ${i}: _malloc returned 0 / wrong type (typeof=${typeof ptr})`, "info");
        break;
      }
      // Commit the first 64 KiB to force actual page allocation, not lazy reservation.
      try {
        const commitView = new Uint8Array(m.HEAPU8.buffer, Number(ptr), 64 * 1024);
        commitView.fill(0xab);
      } catch (commitErr) {
        append(`Iter ${i}: page commit threw — ${commitErr?.message || commitErr}`, "info");
        // Pointer was allocated but page commit failed — count it as a fail and stop.
        break;
      }
      heldPtrs.push(ptr);
      totalCommitted += ONE_GIB;
      iterations = i + 1;
      const gibSoFar = (Number(totalCommitted) / (1 << 30)).toFixed(2);
      append(`Iter ${i}: malloc(1 GiB) OK, total committed=${gibSoFar} GiB`, "info");
    }
    result.phase3_cap_bytes = Number(totalCommitted);
    result.phase3_iterations = iterations;
    const gibFinal = (Number(totalCommitted) / (1 << 30)).toFixed(2);
    append(`PHASE 3 cap = ${gibFinal} GiB across ${iterations} iters`, iterations > 0 ? "ok" : "fail");
    // Free all probed allocations so Phase 4 starts with a clean heap.
    for (const ptr of heldPtrs) m._free(ptr);
    append(`Freed ${heldPtrs.length} cap-probe allocations`, "info");
  } catch (e) {
    append(`PHASE 3 unexpected error: ${e?.message || e}`, "fail");
  }
} else {
  append("Phase 3 SKIPPED (Phase 1 failed)", "info");
}

// ── Phase 4: Post-cap-probe shutdown health ──
// Re-runs init+shutdown after the heap pressure of Phase 3 to verify
// the runtime is still healthy.
if (result.phase1 === "ok") {
  try {
    append("--- Phase 4: Post-cap-probe shutdown health ---", "header");
    const rc = await m._webgpu_init();
    if (rc !== 0) throw new Error(`_webgpu_init (post-probe) returned ${rc}`);
    append("_webgpu_init (post-probe) OK", "ok");
    await m._webgpu_shutdown();
    append("_webgpu_shutdown (post-probe) OK", "ok");
    result.phase4 = "ok";
    append("PHASE 4 PASS", "ok");
  } catch (e) {
    result.phase4 = `fail: ${e?.message || e}`;
    append(`PHASE 4 FAIL: ${e?.message || e}`, "fail");
  }
} else {
  append("Phase 4 SKIPPED (Phase 1 failed)", "info");
}

result.finished_at = new Date().toISOString();
window.__memory64ProbeResult = result;

append("--- Result blob (window.__memory64ProbeResult) ---", "header");
append(JSON.stringify(result, (_k, v) => (typeof v === "bigint" ? Number(v) : v), 2));
append("=== HARNESS COMPLETE ===", "ok");
</script>
</body>
</html>
```

- [ ] **Step 2: Run the build + serve loop and verify the page loads**

```bash
make mem64-probe
```

Confirm the printed URL responds with the page HTML (curl, then full browser open in Phase 4):

```bash
curl -sf -o /dev/null -w '%{http_code}\n' http://localhost:8031/mem64-probe.html
```

Expected: `200`. If 404, the file wasn't copied or the smoke server was started before the file existed; restart with `make mem64-probe` again.

- [ ] **Step 3: Confirm `make checkall` still passes**

```bash
make checkall
```

Expected: 428/11/0. Static HTML in `smoke-test/` is excluded from typecheck/lint by the existing project config (sibling files like `real-model.html`, `mem64-probe.html` follow the same shape).

- [ ] **Step 4: Stop the smoke server before committing**

```bash
make smoke-stop
```

- [ ] **Step 5: Commit**

```bash
git add smoke-test/mem64-probe.html
git commit -m "$(cat <<'EOF'
feat(probe): standalone MEMORY64 cap probe harness

Self-contained smoke-test/mem64-probe.html implementing spec §4's
four phases against webllm-wasm-mem64.js:

  Phase 1 — ASYNCIFY × MEMORY64 round-trip via _webgpu_init/_shutdown.
  Phase 2 — BigInt ABI smoke: _tensor_new_1d return-type test +
            _malloc/HEAPU8 round-trip (no _tensor_set_data — that
            requires backend_alloc_ctx_tensors which is out of scope).
  Phase 3 — Sequential 1 GiB malloc with first-64-KiB page commit;
            stops on first failure; reports total committed bytes.
  Phase 4 — Re-run init/shutdown post-cap-probe to verify runtime
            survived heap pressure.

Logs to #log div + window.__memory64ProbeResult JSON blob (BigInt
narrowed to Number at serialization). Phase 1 failure halts subsequent
phases per spec §5.1; Phase 2 failure does not skip Phase 3 (cap
measurement is ABI-quirk-independent).

Spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md
EOF
)"
```

**Phase 3 gate satisfied** when the file is committed, the URL responds 200, and checkall remains 428/11/0.

---

## Phase 4 — Execute the probe via agentchrome

**Purpose:** Drive the harness to completion in the existing agentchrome session/tab and capture both the screen log and the JSON result blob.

**No commit at the end of Phase 4.** This phase produces transient measurement artifacts that get incorporated into the Phase 5 closure report; the result blob and log are pasted into the report file as part of Phase 5's commit.

### Task 4.1: Restart the smoke server with the latest binaries

- [ ] **Step 1: Run `make mem64-probe` once more so the build is current and the server is running**

```bash
make mem64-probe
```

Save the printed URL (with the `?v=<timestamp>` cache-buster) for the next step.

### Task 4.2: Discover the agentchrome session + tab

- [ ] **Step 1: Confirm an active session and capture its CDP port**

```bash
agentchrome connect --status
```

Expected: JSON with `"active": true` and `"port": <N>`. Save `<N>` as `PORT` for subsequent commands.

- [ ] **Step 2: List tabs and pick one to reuse**

```bash
agentchrome --port <PORT> tabs list
```

Pick an existing tab to reuse (per `~/Repos/webllm/CLAUDE.md` — do not open a new tab). Save its ID as `TAB_ID`. If only the default tab exists, that's the one to use.

### Task 4.3: Navigate the chosen tab to the probe URL

- [ ] **Step 1: Navigate**

```bash
agentchrome --port <PORT> --tab <TAB_ID> navigate '<URL_FROM_TASK_4.1>'
```

Expected: navigation OK, no error.

- [ ] **Step 2: Wait for the harness to complete**

The harness pressure-allocates up to 16 GiB and exercises asyncify; total wall time should be 5–30 s on the dev box. Poll `window.__memory64ProbeResult.finished_at` until it is non-null:

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  status=$(agentchrome --port <PORT> --tab <TAB_ID> page evaluate '(window.__memory64ProbeResult || {}).finished_at')
  echo "iter $i: finished_at=$status"
  case "$status" in null|''|undefined) sleep 5 ;; *) break ;; esac
done
```

Stop polling once `finished_at` is a date string. If after 50 s the harness hasn't completed, capture the page state for debugging (`agentchrome page snapshot`) and treat it as a Phase 1 failure — note in Phase 5 closure.

### Task 4.4: Capture the result blob

- [ ] **Step 1: Read `window.__memory64ProbeResult` as JSON**

```bash
agentchrome --port <PORT> --tab <TAB_ID> page evaluate 'JSON.stringify(window.__memory64ProbeResult, null, 2)' > /tmp/mem64-probe-result.json
cat /tmp/mem64-probe-result.json
```

Expected: a JSON object matching the spec §3.2 shape. Verify the fields are populated:

- `phase1`, `phase2`, `phase4`: each is `"ok"` or `"fail: <reason>"`.
- `phase3_cap_bytes`: a Number (could be `0` if Phase 1 failed, otherwise the headline ceiling).
- `phase3_iterations`: a Number (matches the `cap_bytes / (1<<30)` division if all 1-GiB allocs succeeded).
- `init_wall_ms`: positive Number if Phase 1 reached the `_webgpu_init` call.
- `module_bytes`: initial heap size, informational.
- `user_agent`: Chrome's UA string (used to record the Chrome version in Phase 5).

### Task 4.5: Capture the visible log

- [ ] **Step 1: Read the `#log` div text**

```bash
agentchrome --port <PORT> --tab <TAB_ID> page evaluate 'document.getElementById("log").textContent' > /tmp/mem64-probe-log.txt
head -100 /tmp/mem64-probe-log.txt
```

Expected: the human-readable phase-by-phase log produced by the harness's `append()` calls.

### Task 4.6: Capture environment versions

- [ ] **Step 1: Emscripten version**

```bash
source ~/emsdk/emsdk_env.sh && emcc --version | head -1
```

Save the first line.

- [ ] **Step 2: Chrome version (from the captured UA string in `/tmp/mem64-probe-result.json`)**

```bash
python3 -c "import json; d=json.load(open('/tmp/mem64-probe-result.json')); print(d.get('user_agent',''))"
```

Save the UA string and extract the Chrome version.

- [ ] **Step 3: Hardware specs (macOS)**

```bash
sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu | tr '\n' ' '
echo
```

Save the result for the report.

**Phase 4 gate satisfied** when `/tmp/mem64-probe-result.json` contains valid JSON with `finished_at` populated, the visible log was captured, and Emscripten + Chrome + hardware versions are recorded. **No commit yet** — Phase 5 incorporates all of this into the closure report and commits it as one atomic artifact.

---

## Phase 5 — Closure report + TODO §31 entry

**Purpose:** Translate the measurement into a decision per spec §5.1, document the evidence, and update the TODO so future sessions can resume from the result.

### Task 5.1: Apply the decision rule to the captured result

The decision rule (spec §5.1) keys on **`result.phase1`** plus the cap value **`result.phase3_cap_bytes`** in GiB.

Compute `cap_gib = result.phase3_cap_bytes / (1 << 30)`.

| Branch | Trigger | Closure rendering |
|---|---|---|
| **A: Lever closed** | `result.phase1 !== "ok"` | "MEMORY64 lever CLOSED — ASYNCIFY × MEMORY64 incompatibility on current Emscripten + Chrome." Cite Emscripten + Chrome versions; no follow-up until external state changes. |
| **B: Narrow follow-up needed** | `phase1 === "ok"` AND `phase2 !== "ok"` | "MEMORY64 ABI partially viable — Phase 2 fail mode `<reason>` blocks the canonical bridge migration; needs a narrower follow-up." Cite the specific failure mode. |
| **C: Cap below 13B Q4_K_S** | All earlier phases pass AND `cap_gib < 6` | "MEMORY64 infrastructure functional but doesn't pay for itself at the ≤30B ceiling — Chrome cap of `<cap_gib>` GiB doesn't unlock 13B Q4_K_S. Lever closed pending Chrome moving its cap." |
| **D: Cap is ambiguous** | All earlier phases pass AND `6 <= cap_gib < 8` | "MEMORY64 functional with `<cap_gib>` GiB cap — viable for some 8B Q4_K_S targets, marginal for 13B. Recommend a narrower follow-up that picks one 8B Q4_K_S model and measures load + first-token under wasm64." |
| **E: Lever viable** | All earlier phases pass AND `cap_gib >= 8` | "MEMORY64 lever VIABLE — `<cap_gib>` GiB cap unlocks 13B Q4_K_S. Promote to a P2-class follow-up spec covering full TS bridge migration to BigInt + GGUF streaming validation + one real 13B Q4_K_S load." |

The Phase 4 outcome (`result.phase4`) is informational — a `fail` here is documented as a caveat in the report but does not change the decision branch.

### Task 5.2: Write the closure report

**Files:**
- Create: `eval/reports/memory64-probe-2026-04-28/SUMMARY.md`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p eval/reports/memory64-probe-2026-04-28
```

- [ ] **Step 2: Write the report**

Replace `<…>` placeholders with values captured in Phase 4:

````markdown
# MEMORY64 Cap Probe — Closure Report

**Date:** 2026-04-28
**Spec:** [`docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md`](../../../docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md) (commit `f479ce5`)
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-cap-probe.md`](../../../docs/superpowers/plans/2026-04-28-memory64-cap-probe.md)
**Branch (decision rule §5.1):** **<A | B | C | D | E>**

## Decision

**<one-line summary from Task 5.1's rendering for the matched branch>**

Headline: Chrome's practical heap ceiling on the dev box was **<cap_gib> GiB**
across **<iterations>** sequential 1-GiB allocations with first-page commit.

## Phase outcomes

| Phase | Outcome |
|---|---|
| 1 — ASYNCIFY × MEMORY64 round-trip | `<result.phase1>` |
| 2 — BigInt ABI smoke | `<result.phase2>` |
| 3 — Cap probe (1 GiB sequential) | `<cap_gib> GiB / <iterations> iters` |
| 4 — Post-probe shutdown health | `<result.phase4>` |

`_webgpu_init` wall time (Phase 1): **<result.init_wall_ms> ms**
Initial heap size at module load: **<result.module_bytes>** bytes

## Result blob (`window.__memory64ProbeResult`)

```json
<paste /tmp/mem64-probe-result.json verbatim>
```

## Visible log

```
<paste /tmp/mem64-probe-log.txt verbatim>
```

## Environment

- **Emscripten:** `<emcc --version line>`
- **Chrome:** `<UA-derived version>` (full UA: `<user_agent>`)
- **Hardware:** `<sysctl line>`
- **llama.cpp:** branch `webllm-browser-patches`, tip `981859864` (post-§27 rebase, patch 11 / bug #28 fix).
- **Build flags (mem64 target):** `-sMEMORY64=1 -sWASM_BIGINT=1 -sMAXIMUM_MEMORY=16GB` plus all flags inherited from the wasm32 target (full list in `src/wasm/CMakeLists.txt`).

## Reproduction

```bash
# 1. Build mem64 target + start smoke server + emit URL
make mem64-probe
# 2. Open the printed URL in the existing agentchrome tab
# 3. Read window.__memory64ProbeResult after the harness completes
```

## Follow-up

<For branch E>: Promote to a P2-class follow-up spec — full TS bridge
migration (`src/inference/ggml-wasm.ts`, `src/inference/model-inference.ts`)
to BigInt pointers + GGUF streaming validation under wasm64 + one
real 13B Q4_K_S load. Brainstorm before planning.

<For branches A/B/C>: Lever closed. No follow-up until external state
changes (Emscripten/Chrome MEMORY64 maturity / ABI bug fix / Chrome
cap raise).

<For branch D>: Narrower follow-up — pick one 8B Q4_K_S model
(candidate: `qwen3-8b-q4ks` or `llama-3.1-8b-q4ks` if available),
brainstorm a smaller spec that measures load + first-token under
wasm64 without committing the full bridge migration.
````

### Task 5.3: Add the §31 closure entry to TODO.md

**Files:**
- Modify: `TODO.md` (add `§31` entry to the resumption checklist).

- [ ] **Step 1: Locate the resumption-checklist closures section**

The pattern is established by `§30`, `§29`, `§28`, etc. Find the line in `TODO.md` that closes `§30` (around the line starting `§30. ~~Heuristic-based prefill-tile default in ModelInference.~~`) and append the new `§31` entry **after the §30 line**.

- [ ] **Step 2: Insert the §31 entry**

Replace the `§30` line plus the next blank line with `§30` (unchanged) followed by the appropriate `§31` rendering for the matched branch:

````markdown
§31. ~~MEMORY64 cap probe.~~ **CLOSED 2026-04-28 — branch <X>.**
<one-line decision rendering from Task 5.1>. Chrome practical cap
on dev box: **<cap_gib> GiB / <iterations> iters**. Emscripten
`<version>`, Chrome `<version>`. Build target `webllm-wasm-mem64`
+ harness `smoke-test/mem64-probe.html` retained as side infra
(dead code on `main` until promoted; zero impact on existing
`webllm-wasm` build, checkall remains 428/11/0). Spec `f479ce5`,
plan `<this commit's parent SHA after closure commit lands>`,
report `eval/reports/memory64-probe-2026-04-28/SUMMARY.md`.
<For branch E>: Promote candidate documented in report's
"Follow-up" section. <For branches A/B/C>: No follow-up — see
report's "Follow-up" section for the deferral rationale.
<For branch D>: Narrower follow-up scoped in report.
````

(Pick exactly one of the three trailing sentences based on the matched branch.)

### Task 5.4: Add the §31 entry to the "Recommended first move" / candidate list

- [ ] **Step 1: Locate the "Candidate next levers" / "3 candidates remain open" block**

Around the lines starting `**3 candidates remain open**` and the numbered candidates list (`1. ~~§C-v2-A resurrection…~~ CLOSED…`, `2. **MEMORY64 for the 8-30B fleet.**`, etc.).

- [ ] **Step 2: Strike through candidate #2 (MEMORY64) per the matched branch**

For branches A/B/C: replace the existing `2. **MEMORY64 for the 8-30B fleet.**` block with:

````markdown
2. ~~**MEMORY64 for the 8-30B fleet.**~~ **CLOSED 2026-04-28 — §31
   branch <X>.** <one-line decision summary>. Probe artifact retained
   as `webllm-wasm-mem64` build target (dead code) + harness; no
   `src/inference/*.ts` change.
````

For branch D: same strike-through, but the trailing sentence reads "Narrower follow-up scoped in report — pick one 8B Q4_K_S model and measure load + first-token under wasm64."

For branch E: do **not** strike through — replace the existing block with:

````markdown
2. **MEMORY64 for the 8-30B fleet — PROBE PASSED 2026-04-28 — §31
   branch E.** Cap probe at <cap_gib> GiB unlocks 13B Q4_K_S. Next
   move is a P2-class follow-up spec covering full TS bridge
   migration to BigInt + GGUF streaming validation under wasm64 +
   one real 13B Q4_K_S load. Brainstorm before planning.
````

### Task 5.5: Run `make checkall` and commit

- [ ] **Step 1: Confirm no test regression**

```bash
make checkall
```

Expected: 428/11/0. Phase 5 only touches docs.

- [ ] **Step 2: Stage all Phase 5 artifacts**

```bash
git add -f docs/superpowers/plans/2026-04-28-memory64-cap-probe.md
git add eval/reports/memory64-probe-2026-04-28/SUMMARY.md
git add TODO.md
```

(The plan file is force-added because `docs/superpowers/` is gitignored — same convention as the spec.)

- [ ] **Step 3: Commit the plan first, then the closure**

Per the commit-before-work policy, the plan should have been committed before Phase 1 started. If it wasn't, commit it now ahead of the closure commit:

```bash
git diff --cached --stat | grep -q 'docs/superpowers/plans/2026-04-28-memory64-cap-probe.md' && \
  git commit -- docs/superpowers/plans/2026-04-28-memory64-cap-probe.md -m "$(cat <<'EOF'
docs(plan): MEMORY64 cap probe — phased implementation plan

Plan implementing the spec at docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md.
Six phases: baseline verification, CMake target, Makefile target,
harness HTML, agentchrome execution, closure report. Each phase
commits before the next starts (per the always-commit-before-work
policy).
EOF
)" || true
```

- [ ] **Step 4: Commit the closure report + TODO entry**

```bash
git add eval/reports/memory64-probe-2026-04-28/SUMMARY.md TODO.md
git commit -m "$(cat <<'EOF'
docs(perf): §31 — MEMORY64 cap probe CLOSED, branch <X>

Probe executed 2026-04-28. <one-line decision>. Chrome practical
cap on dev box: <cap_gib> GiB. Phase 1 (ASYNCIFY × MEMORY64):
<status>. Phase 2 (BigInt ABI): <status>. Phase 4 (post-probe
shutdown): <status>.

Report: eval/reports/memory64-probe-2026-04-28/SUMMARY.md
Spec: f479ce5, plan: <plan-commit-sha>.

Build target webllm-wasm-mem64 + harness smoke-test/mem64-probe.html
retained as side infrastructure; no src/inference/*.ts change;
checkall remains 428/11/0; existing webllm-wasm build bit-identical.
EOF
)"
```

(Substitute `<X>`, `<one-line decision>`, `<cap_gib>`, `<status>`, and `<plan-commit-sha>` with the actual values.)

**Phase 5 gate satisfied** when:
- The closure report exists at the expected path with all `<…>` placeholders filled.
- The TODO §31 entry is in place.
- `make checkall` is 428/11/0.
- Two commits landed on `main`: the plan commit and the closure commit.

---

## Self-review checklist (writing-plans skill)

- [x] **Spec coverage:** every spec section maps to a phase. §1 context → plan header. §2 goal → Phase 5 decision rule. §3 architecture → Phase 1 + Phase 2 + Phase 3. §4 phases → Phase 3 harness implementation. §5 thresholds → Phase 5 Task 5.1 decision matrix. §6 out-of-scope → enforced by Phase 1 only touching `src/wasm/CMakeLists.txt` and Phase 3 harness being self-contained. §7 risks → addressed in Phase 1 Step 5 (no-regression) and Phase 4 timeout fallback. §8 files touched → matches plan File Structure section. §9 review checklist → Phase 5 closure report fields.
- [x] **No placeholders:** every code block contains real, executable content. Phase 4 + Phase 5 use `<…>` placeholders only for values that must come from the captured measurement (deliberate — they cannot be predetermined).
- [x] **Type consistency:** `_tensor_new_1d`, `_malloc`, `_ctx_create`, `_ctx_free`, `_free`, `_webgpu_init`, `_webgpu_shutdown` referenced consistently. `result.phase1` / `phase2` / `phase3_cap_bytes` / `phase3_iterations` / `phase4` / `init_wall_ms` / `module_bytes` referenced consistently across spec §3.2, harness JS, Phase 4 capture, and Phase 5 decision matrix.
- [x] **Scope check:** single subsystem, single decision-grade question. No decomposition needed.
