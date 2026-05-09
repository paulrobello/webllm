# JSEP + MEMORY64 Build Target Implementation Plan

> **SUPERSEDED 2026-05-08 — negative-result closure.** Phase A Task A2
> hit two architectural blockers; the spec was reframed as a §31-style
> cap probe with a negative result. See
> [`../../../eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`](../../../eval/reports/jsep-mem64-2026-05-08/SUMMARY.md)
> for the closure. This plan is retained as the historical record; do not
> execute its tasks without first solving the documented blockers.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `wasm-build-jsep-mem64` target that combines the JSEP backend (`WEBLLM_BACKEND=jsep`) and MEMORY64 heap cap (`-sMEMORY64=1`, `-sMAXIMUM_MEMORY=16GB`) so the deferred 7B+ canonical-6 subset (mistral-7b-q4ks, llama-3.1-8b-iq3m, qwen3-8b-iq3m) can run through Stage 4.36's parity gate against `webllm-wasm-mem64.js` as the non-JSEP reference.

**Architecture:** Three-phase plan with go/no-go after Phase A. Phase A — refactor `src/wasm/CMakeLists.txt` to thread JSEP linkage into the existing `WEBLLM_BUILD_MEM64` block; add a parallel Makefile target; verify linkage via a new ~30-LOC smoke probe HTML. Phase B — extend `p2-v2-ref-probe` with the 3 deferred-subset model entries plus a per-model `requiresMem64` dispatch flag (auto-imports `webllm-wasm-mem64.js` over `webllm-wasm.js`); capture reference vectors. Phase C — mirror in `p2-v2-spike` (auto-imports `webllm-wasm-jsep-mem64.js` over `webllm-wasm-jsep.js`); run parity sweeps; write closure report; update `TODO.md` + `TODO_ARCHIVE.md`.

**Tech Stack:** Emscripten 5.0.6 (wasm-ld), CMake, Make, TypeScript (`bun build` for harness bundling), `agentchrome` CLI for browser automation, WebGPU + JSPI runtime.

**Spec:** [`docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md`](../specs/2026-05-08-jsep-mem64-build-target-design.md) (commit `f964915`).

---

## File Inventory

**Phase A (4 files modified, 1 new):**
- Modify: `src/wasm/CMakeLists.txt:186-220` (refactor `WEBLLM_BUILD_MEM64` block to thread JSEP branching)
- Modify: `Makefile` (add `wasm-build-jsep-mem64` target, register in `.PHONY`, optionally add to `smoke-test` rule)
- Create: `smoke-test/mem64-jsep-probe.html` (linkage smoke probe)
- Create: `eval/reports/jsep-mem64-2026-05-08/PHASE-A.md` (closure report)

**Phase B (2 files modified, 1 file modified additively):**
- Modify: `smoke-test/p2-v2-ref-probe.src.ts` (3 new MODELS entries + `requiresMem64` dispatch + size assertion)
- Modify: `smoke-test/p2-v2-ref-probe.html` (banner update — minor)
- Modify: `eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json` (additive — 3 new entries; update `comment` field)

**Phase C (5 files modified, 1 new):**
- Modify: `smoke-test/p2-v2-spike.src.ts` (mirror Phase B changes — 3 entries + `requiresMem64` dispatch + size assertion)
- Create: `eval/reports/jsep-mem64-2026-05-08/SUMMARY.md` (Phase 3 closure-extension report)
- Modify: `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md` (deferred-subset closure note pointing at SUMMARY.md)
- Modify: `TODO.md` (closure-stub swap — replace deferred-subset paragraph with Phase 3 closure-extension stub)
- Modify: `TODO_ARCHIVE.md` (archive 7B+ deferral block)

---

## Phase A — Build target + linkage smoke probe

### Task A1: Refactor `src/wasm/CMakeLists.txt` to thread JSEP into the MEM64 block

**Files:**
- Modify: `src/wasm/CMakeLists.txt:186-220`

**Why:** The current MEM64 block hardcodes the non-JSEP linkage (line 196: `target_link_libraries(... llama ggml-base ggml-webgpu)`). It does not thread `WEBLLM_BACKEND_JSEP` like the wasm32 block does (lines 222-235). After this task, both axes (`WEBLLM_BUILD_MEM64` × `WEBLLM_BACKEND`) are independently configurable, producing four valid combinations including the new `webllm-wasm-jsep-mem64.{js,wasm}`.

- [ ] **Step 1: Read the current file to confirm line numbers**

```bash
sed -n '186,220p' src/wasm/CMakeLists.txt
```

Expected: lines 186-220 show the `if(WEBLLM_BUILD_MEM64)` block with `add_executable(webllm-wasm-mem64 webgpu-bridge.cpp)` and a `target_link_libraries(webllm-wasm-mem64 PRIVATE llama ggml-base ggml-webgpu)` line.

- [ ] **Step 2: Apply the refactor**

Replace lines 186-220 (the entire `if(WEBLLM_BUILD_MEM64) ... endif()` IF-branch) with:

```cmake
    if(WEBLLM_BUILD_MEM64)
        # ── MEMORY64 cap probe target.
        # Built only when -DWEBLLM_BUILD_MEM64=ON is passed at cmake configure
        # (driven by `make wasm-build-mem64` or `make wasm-build-jsep-mem64`).
        # The probe binary links wasm64-against-wasm64 archives because the
        # global CMAKE_C_FLAGS / CMAKE_CXX_FLAGS include `-sMEMORY64=1`,
        # which propagates to the ggml add_subdirectory above and produces
        # wasm64 ggml-base + ggml-webgpu archives.
        # Spec: docs/superpowers/specs/2026-04-28-memory64-cap-probe-design.md
        # JSEP+mem64 spec: docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md
        add_executable(webllm-wasm-mem64 webgpu-bridge.cpp)
        if(WEBLLM_BACKEND_JSEP)
            # JSEP+mem64 — webllm-wasm-jsep-mem64.{js,wasm}. Mirrors the
            # wasm32 JSEP branch below, with the wasm64 link options inherited
            # from the surrounding block.
            target_link_libraries(webllm-wasm-mem64 PRIVATE llama ggml-base ggml-webgpu ggml-jsep)
            set_target_properties(webllm-wasm-mem64 PROPERTIES OUTPUT_NAME "webllm-wasm-jsep-mem64")
            target_compile_definitions(webllm-wasm-mem64 PRIVATE WEBLLM_PIN_TO_JSEP=1)
        else()
            target_link_libraries(webllm-wasm-mem64 PRIVATE llama ggml-base ggml-webgpu)
        endif()
        target_include_directories(webllm-wasm-mem64 PRIVATE
            ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/ggml/include
            ${LLAMA_CPP_DIR}/ggml/src
        )
        target_link_options(webllm-wasm-mem64 PRIVATE
            "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}"
            "$<$<BOOL:${WEBLLM_BACKEND_JSEP}>:-sJSPI_EXPORTS=${JSPI_EXPORTS}>"
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
```

The `$<$<BOOL:${WEBLLM_BACKEND_JSEP}>:...>` generator expression conditionally includes the `-sJSPI_EXPORTS=...` link option only when `WEBLLM_BACKEND_JSEP=ON`. Without it, the non-JSEP mem64 build (current ship path) would emit an empty `JSPI_EXPORTS` regex which Emscripten warns about.

- [ ] **Step 3: Verify the edit**

```bash
sed -n '186,232p' src/wasm/CMakeLists.txt
```

Expected: the new block is in place, with the inner `if(WEBLLM_BACKEND_JSEP) ... else() ... endif()` plus the conditional `-sJSPI_EXPORTS` link option.

- [ ] **Step 4: Smoke-build the existing non-JSEP mem64 target to verify no regression**

```bash
cd /Users/probello/Repos/webllm
make wasm-build-mem64 2>&1 | tail -20
```

Expected: build succeeds (exit 0), produces `src/wasm/build-mem64/webllm-wasm-mem64.js` + `.wasm`. If this fails, the refactor broke the non-JSEP path — revert and try again.

- [ ] **Step 5: Stage the change (commit deferred to Task A5)**

```bash
git add src/wasm/CMakeLists.txt
git status
```

---

### Task A2: Add `wasm-build-jsep-mem64` Makefile target

**Files:**
- Modify: `Makefile:1-8` (add target name to `.PHONY`)
- Modify: `Makefile` (insert new target after `wasm-build-jsep`, around line 173)

- [ ] **Step 1: Read the current Makefile structure to confirm position**

```bash
sed -n '1,10p' Makefile && echo "---" && sed -n '128,175p' Makefile
```

Expected: line 2 has the `.PHONY: ... wasm-build wasm-build-wasm32 wasm-build-mem64 wasm-build-jsep wasm-clean ...` declaration. Lines 130-173 contain the existing `wasm-build-jsep` target.

- [ ] **Step 2: Update `.PHONY` declaration to register the new target**

Replace line 2:

```makefile
        wasm-build wasm-build-wasm32 wasm-build-mem64 wasm-build-jsep wasm-clean \
```

with:

```makefile
        wasm-build wasm-build-wasm32 wasm-build-mem64 wasm-build-jsep wasm-build-jsep-mem64 wasm-clean \
```

- [ ] **Step 3: Insert the new target**

Insert after line 173 (the end of the existing `wasm-build-jsep` target — the `bun build smoke-test/p2-v2-offload-probe.src.ts ...` line):

```makefile

wasm-build-jsep-mem64: ## Build the JSEP backend + MEMORY64 (16 GiB cap) variant → webllm-wasm-jsep-mem64.{js,wasm}
	cd src/wasm && mkdir -p build-jsep-mem64 && cd build-jsep-mem64 && \
	source ~/emsdk/emsdk_env.sh 2>/dev/null; \
	emcmake cmake .. \
		-DWEBLLM_BACKEND=jsep \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=OFF \
		-DEMDAWNWEBGPU_DIR=$(CURDIR)/vendor/emdawnwebgpu \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CPU=OFF \
		-DGGML_BLAS=OFF \
		-DGGML_METAL=OFF \
		-DGGML_ACCELERATE=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_OPENMP=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_LLAMAFILE=OFF \
		-DGGML_BUILD_TESTS=OFF \
		-DGGML_BUILD_EXAMPLES=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_BACKEND_DL=OFF \
		-DWEBLLM_BUILD_MEM64=ON \
		-DCMAKE_C_FLAGS="-sMEMORY64=1" \
		-DCMAKE_CXX_FLAGS="-sMEMORY64=1" \
		-DWEBLLM_ASSERTIONS=$(WEBLLM_ASSERTIONS) && \
	cmake --build . --target webllm-wasm-mem64 --config Release -j
	# Co-locate the JSEP+mem64 WASM artifacts next to the existing JSEP
	# bundle so the harness can dynamic-import either variant by file
	# basename without changing the bundle. Output basename is
	# `webllm-wasm-jsep-mem64.{js,wasm}` (set via OUTPUT_NAME in
	# src/wasm/CMakeLists.txt's WEBLLM_BUILD_MEM64+JSEP branch).
	cp src/wasm/build-jsep-mem64/webllm-wasm-jsep-mem64.js src/wasm/build-jsep-mem64/webllm-wasm-jsep-mem64.wasm smoke-test/
```

Notes on the recipe:
- Combines `wasm-build-jsep`'s `-DWEBLLM_BACKEND=jsep` flag with `wasm-build-mem64`'s `-DWEBLLM_BUILD_MEM64=ON -DCMAKE_C_FLAGS=-sMEMORY64=1 -DCMAKE_CXX_FLAGS=-sMEMORY64=1`.
- The `cmake --build . --target webllm-wasm-mem64` flag explicitly names the executable target (the same name CMake produces in the WEBLLM_BUILD_MEM64 block; the OUTPUT_NAME property changes the .js/.wasm basename but not the CMake target name).
- The bundle (`webllm-bundle-jsep.js`) is built by the existing `wasm-build-jsep` target; this new target deliberately does **not** rebuild the bundle since the bundle is shared and statically references neither WASM module — the harness picks via dynamic `import()`.

- [ ] **Step 4: Verify the Makefile parses**

```bash
make -n wasm-build-jsep-mem64 2>&1 | head -10
```

Expected: dry-run print of the recipe commands, no `make: ***` errors. If it prints `make: *** No rule to make target ...`, the target wasn't added correctly.

- [ ] **Step 5: Verify `make help` lists the new target**

```bash
make help 2>&1 | grep "wasm-build-jsep-mem64"
```

Expected: one line showing the new target with its `## ...` description.

- [ ] **Step 6: Run the build (this exercises Task A1 + A2 together)**

```bash
cd /Users/probello/Repos/webllm
make wasm-build-jsep-mem64 2>&1 | tee /tmp/jsep-mem64-build.log | tail -40
```

Expected: build succeeds (exit 0), produces `src/wasm/build-jsep-mem64/webllm-wasm-jsep-mem64.js` + `.wasm`, copies them into `smoke-test/`. Build wall ~3-5 min on M-series Mac.

If link errors mention `ggml_jsep_*` undefined: Task A1 didn't link `ggml-jsep`. Re-check `target_link_libraries`.
If link errors mention `JSPI_EXPORTS` regex empty: Task A1's generator-expression conditional didn't fire. Check the `$<$<BOOL:${WEBLLM_BACKEND_JSEP}>:...>` syntax.
If `cmake --build` fails with "no rule to make target webllm-wasm-mem64": the `add_executable` line in CMakeLists.txt is gated wrong; verify it's inside the `if(WEBLLM_BUILD_MEM64)` block (Task A1).

- [ ] **Step 7: Confirm artifact existence + size**

```bash
ls -lh smoke-test/webllm-wasm-jsep-mem64.{js,wasm} src/wasm/build-jsep-mem64/webllm-wasm-jsep-mem64.{js,wasm}
```

Expected: 4 files exist, `.wasm` is ~5-10 MiB. Capture the sizes for the closure report.

- [ ] **Step 8: Stage Makefile change (commit deferred to Task A5)**

```bash
git add Makefile
git status
```

---

### Task A3: Create `mem64-jsep-probe.html` linkage smoke probe

**Files:**
- Create: `smoke-test/mem64-jsep-probe.html` (~50 LOC including the inline `<script type="module">` body)

**Why:** Verify that the wasm64+JSEP combination doesn't have a JSPI re-entrancy or BigInt-boundary issue *before* sinking time on harness wiring (Phase B/C). Loading the WASM and calling `webgpu_init` is the cheapest possible JSPI exercise — no model load, no decode, just module init + WebGPU device creation.

- [ ] **Step 1: Create the HTML file**

```bash
cat > smoke-test/mem64-jsep-probe.html <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WebLLM JSEP+MEM64 — Phase A linkage probe</title>
<style>
  body { font: 14px/1.5 ui-monospace, monospace; padding: 16px; max-width: 900px; }
  h1 { font-size: 18px; }
  #log { white-space: pre-wrap; background: #f5f5f5; padding: 12px; border-radius: 4px; }
  .pass { color: #2a7e2a; font-weight: bold; }
  .fail { color: #b00020; font-weight: bold; }
  .info { color: #555; }
</style>
</head>
<body>
<h1>JSEP + MEM64 — Phase A linkage probe</h1>
<p class="info">Dynamic-imports <code>webllm-wasm-jsep-mem64.js</code>,
calls <code>webgpu_init</code> via JSPI, logs heap size + WebGPU init
result. No model load. Confirms the wasm64+JSEP+JSPI intersection
doesn't trip a BigInt boundary or JSPI re-entrancy issue. Spec:
<code>docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md</code>.</p>
<div id="log"></div>
<script type="module">
function log(msg, cls) {
  const el = document.getElementById("log");
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  console.log(msg);
}

(async () => {
  try {
    log("[1/4] Dynamic-importing webllm-wasm-jsep-mem64.js...");
    const cacheBust = window.location.search || "";
    const createModule = (await import(`./webllm-wasm-jsep-mem64.js${cacheBust}`)).default;
    const stderrLines = [];
    const mod = await createModule({
      printErr: (s) => { stderrLines.push(s); console.error(s); },
    });
    window.__memJsepProbeMod = mod;

    const heapBytesPostInit = mod.HEAP8.length;
    log(`     heap size after init: ${(heapBytesPostInit / 1024 / 1024).toFixed(1)} MiB`);

    log("[2/4] Calling webgpu_init (JSPI-promised)...");
    const initStatus = await mod._webgpu_init();
    if (initStatus !== 0) {
      log(`webgpu_init returned ${initStatus}`, "fail");
      window.__memJsepProbeResult = { ok: false, stage: "webgpu_init", initStatus };
      return;
    }
    log(`     webgpu_init OK (status=0)`);

    log("[3/4] Probing ABI shape (wasm32=number / wasm64=bigint)...");
    let abi = "unknown";
    try {
      const probe = mod._bridge_malloc(0);
      abi = typeof probe === "bigint" ? "wasm64" : "wasm32";
      mod._bridge_free(probe);
    } catch (e) {
      try {
        const probe = mod._bridge_malloc(0n);
        mod._bridge_free(probe);
        abi = "wasm64";
      } catch (e2) {
        abi = `error: ${e2.message}`;
      }
    }
    log(`     ABI probe: ${abi}`);

    log("[4/4] Snapshot final heap size...");
    const heapBytesFinal = mod.HEAP8.length;
    log(`     heap size: ${(heapBytesFinal / 1024 / 1024).toFixed(1)} MiB`);

    const stderrSummary = stderrLines.length ? stderrLines.slice(0, 5).join(" | ") : "(none)";
    log(`stderr lines: ${stderrLines.length} — first 5: ${stderrSummary}`);

    window.__memJsepProbeResult = {
      ok: true,
      heapBytesPostInit,
      heapBytesFinal,
      abi,
      initStatus,
      stderrCount: stderrLines.length,
    };
    log("DONE", "pass");
  } catch (err) {
    log(`FAIL — ${err.message}\n${err.stack ?? ""}`, "fail");
    window.__memJsepProbeResult = { ok: false, error: String(err) };
  }
})();
</script>
</body>
</html>
EOF
```

- [ ] **Step 2: Verify the file is well-formed**

```bash
head -5 smoke-test/mem64-jsep-probe.html && wc -l smoke-test/mem64-jsep-probe.html
```

Expected: starts with `<!DOCTYPE html>`, ~85 lines.

- [ ] **Step 3: Stage the change (commit deferred to Task A5)**

```bash
git add smoke-test/mem64-jsep-probe.html
git status
```

---

### Task A4: Run the linkage smoke probe via agentchrome

**Files:** none modified — this is a verification step that exercises Tasks A1-A3 end-to-end.

**Why:** Confirm that the wasm64+JSEP+JSPI combination produces a working module + working WebGPU init. If this fails, abort Phase B/C and diagnose.

- [ ] **Step 1: Start the smoke server (background)**

```bash
cd /Users/probello/Repos/webllm
make smoke-restart 2>&1 | tail -3
```

Expected: "smoke server running on http://localhost:8031" (or wherever `SMOKE_PORT` is).

- [ ] **Step 2: Confirm the agentchrome session is live**

```bash
agentchrome connect --status 2>&1 | head -5
```

Expected: a session is reachable on its port. If not, follow CLAUDE.md "agentchrome usage" — `agentchrome connect --launch --headless` is the fallback.

- [ ] **Step 3: Find or create a tab**

```bash
agentchrome tabs list 2>&1 | head -10
```

If a smoke-test-related tab exists, capture its tab id. Otherwise:

```bash
agentchrome tabs create "http://localhost:8031/mem64-jsep-probe.html?ingest=off&v=$(date +%s)"
```

(`ingest=off` keeps any future smoke runs out of the dashboard; this probe doesn't ingest anyway, but make the habit explicit.)

- [ ] **Step 4: Navigate the existing tab to the probe**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/mem64-jsep-probe.html?ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -30
```

Expected: page snapshot shows steps `[1/4] Dynamic-importing webllm-wasm-jsep-mem64.js...` through `[4/4] Snapshot final heap size...` and final line `DONE` with `class="pass"`.

- [ ] **Step 5: Read the probe result via console eval**

```bash
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__memJsepProbeResult, null, 2)" 2>&1 | tail -20
```

Expected JSON-shape: `{"ok": true, "heapBytesPostInit": <int>, "heapBytesFinal": <int>, "abi": "wasm64", "initStatus": 0, "stderrCount": <int>}`.

- [ ] **Step 6: Capture stderr lines for the closure report (informational)**

```bash
agentchrome console list --tab <TAB_ID> --kind error --limit 10 2>&1 | tail -20
```

Capture any non-trivial stderr noise. Benign expected lines: `adapter_info: ...` (WebGPU backend introspection — not a failure per CLAUDE.md regression-lessons).

- [ ] **Step 7: Branch on outcome**

If `ok: true` and `abi == "wasm64"` and `initStatus == 0`: proceed to Task A5.

If `ok: false` and the stage is `webgpu_init`: a JSPI-side issue. Capture full stderr and `agentchrome console list --kind error`. Do not proceed; investigate the JSPI_EXPORTS list (Task A1's `target_link_options`) and the `webgpu_init` symbol.

If the JS throws before `[1/4]` finishes: likely a wasm-instantiation issue (missing export, ABI mismatch). Capture the stack and inspect the WASM imports table.

If the heap size is dramatically smaller than expected (<256 MiB after init): `MAXIMUM_MEMORY=16GB` may not have applied; verify the link option made it into the build (`emcc ... | grep MAXIMUM_MEMORY`).

---

### Task A5: Phase A closure report + commits

**Files:**
- Create: `eval/reports/jsep-mem64-2026-05-08/PHASE-A.md` (closure report)

- [ ] **Step 1: Create the closure report directory**

```bash
mkdir -p eval/reports/jsep-mem64-2026-05-08
```

- [ ] **Step 2: Write the closure report**

Replace `<HEAP_BYTES_POST_INIT>`, `<HEAP_BYTES_FINAL>`, and `<WASM_SIZE>` with the actual values captured in Task A4 / Task A2 Step 7:

```bash
cat > eval/reports/jsep-mem64-2026-05-08/PHASE-A.md <<'EOF'
# Phase A — JSEP + MEM64 build target linkage probe

Status: closed 2026-05-08
Spec: [`docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md`](../../../docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md)
Plan: [`docs/superpowers/plans/2026-05-08-jsep-mem64-build-target.md`](../../../docs/superpowers/plans/2026-05-08-jsep-mem64-build-target.md)

## TL;DR

`make wasm-build-jsep-mem64` produces `webllm-wasm-jsep-mem64.{js,wasm}`
artifacts. Linkage probe (`smoke-test/mem64-jsep-probe.html`) loads the
module, runs `webgpu_init` via JSPI, and confirms the wasm64 ABI shape.
Phase B/C are unblocked.

## Build artifacts

| File | Size |
|------|------|
| `smoke-test/webllm-wasm-jsep-mem64.js` | <JS_SIZE> |
| `smoke-test/webllm-wasm-jsep-mem64.wasm` | <WASM_SIZE> |

## Linkage probe (`mem64-jsep-probe.html`)

| Field | Value |
|-------|-------|
| `ok` | `true` |
| `abi` | `wasm64` |
| `initStatus` | `0` |
| `heapBytesPostInit` | `<HEAP_BYTES_POST_INIT>` |
| `heapBytesFinal` | `<HEAP_BYTES_FINAL>` |
| `stderrCount` | `<STDERR_COUNT>` |

The `wasm64` ABI confirmation rules out a silent fall-through to the
wasm32 build path (which would have `abi == "wasm32"`). `initStatus = 0`
confirms JSPI's `WebAssembly.promising` wrap on `webgpu_init` survived
the wasm64 ABI shift — a BigInt-boundary or JSPI re-entrancy issue
would have surfaced here.

## Risk register status

| Risk (from spec) | Status |
|-------------------|--------|
| 1. JSPI re-entrancy under wasm64 | RESOLVED — `webgpu_init` returned 0 |
| 5. Bundle-side dynamic import path resolution | RESOLVED — bundle untouched, harness will dispatch in Phase B/C |
| 6. Pre-existing assertion failure paths under wasm64 | DEFERRED to Phase C (only surfaces under model load + decode) |

Risks 2-4 are model-load-time / decode-time risks; they cannot trip
this probe. Phase C is the surface for those.

## Branch on outcome

PASS — proceed to Phase B (reference-vector capture).
EOF
```

- [ ] **Step 3: Fill in the placeholders with real numbers**

Use the JSON output captured in Task A4 Step 5 + the `ls -lh` output from Task A2 Step 7. Replace each `<...>` token in the report with the actual value.

- [ ] **Step 4: Run `make checkall` to confirm no TS regressions**

```bash
cd /Users/probello/Repos/webllm
make checkall 2>&1 | tail -20
```

Expected: green (fmt + lint + typecheck + typecheck-tests + test all pass).

- [ ] **Step 5: Commit Phase A**

Two commits per CLAUDE.md cadence (`feat(...)` + `docs(reports)`):

```bash
git add src/wasm/CMakeLists.txt Makefile smoke-test/mem64-jsep-probe.html
git commit -m "$(cat <<'MSG'
feat(wasm): add wasm-build-jsep-mem64 target — JSEP backend + MEMORY64 16 GiB cap

Combines WEBLLM_BACKEND=jsep and WEBLLM_BUILD_MEM64=ON to produce
webllm-wasm-jsep-mem64.{js,wasm} for canonical-6 7B+ models that
overflow the wasm32 4 GiB JSEP heap cap. Refactor src/wasm/CMakeLists.txt
WEBLLM_BUILD_MEM64 block to thread JSEP linkage (mirrors the wasm32
JSEP branching). New Makefile target combines the existing
wasm-build-jsep recipe with the wasm-build-mem64 -sMEMORY64=1 flags
under a separate build-jsep-mem64/ build dir. Linkage probe at
smoke-test/mem64-jsep-probe.html confirms wasm64 ABI + JSPI roundtrip
without model load.

Spec: docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md
MSG
)"

git add eval/reports/jsep-mem64-2026-05-08/PHASE-A.md
git commit -m "$(cat <<'MSG'
docs(reports): jsep-mem64 Phase A closure — build target verified

PHASE-A.md banks artifact sizes, linkage-probe results (ok=true,
abi=wasm64, initStatus=0, heap MiB), and risk-register status. Phase
B (reference-vector capture for the deferred subset) is unblocked.
MSG
)"

git status
```

Expected: clean tree; two new commits visible in `git log -3 --oneline`.

---

## Phase B — Reference-vector capture for the deferred subset

### Task B1: Extend `p2-v2-ref-probe.src.ts` with the 3 deferred-subset entries + dispatch + assertion

**Files:**
- Modify: `smoke-test/p2-v2-ref-probe.src.ts:21-37` (MODEL_REGISTRY)
- Modify: `smoke-test/p2-v2-ref-probe.src.ts:64-72` (the WASM module dynamic import)

- [ ] **Step 1: Read the current file to confirm line numbers**

```bash
sed -n '21,80p' smoke-test/p2-v2-ref-probe.src.ts
```

Expected: `MODEL_REGISTRY` declared as `Record<string, { ggufUrl: string; promptText: string }>` with 3 entries (tinyllama, qwen3-0.6b, qwen3-1.7b). Module import on line 72: `(await import(`./webllm-wasm.js${cacheBust}`)).default`.

- [ ] **Step 2: Update the `MODEL_REGISTRY` entry shape and add 3 new entries**

Replace lines 21-37 with:

```typescript
const WASM32_HEAP_MARGIN = 3.5 * 1024 * 1024 * 1024;

const MODEL_REGISTRY: Record<
	string,
	{ ggufUrl: string; promptText: string; requiresMem64?: boolean }
> = {
	tinyllama: {
		ggufUrl: "/models/tinyllama-1.1b-chat-q4_0.gguf",
		promptText: "The capital of France is",
	},
	"qwen3-0.6b": {
		ggufUrl: "/models/qwen3-0.6b-q4f16.gguf",
		promptText: "The capital of France is",
	},
	"qwen3-1.7b": {
		ggufUrl: "/models/qwen3-1.7b-q4f16.gguf",
		promptText: "The capital of France is",
	},
	"mistral-7b-q4ks": {
		ggufUrl: "/models/mistral-7b-instruct-v0.3-q4ks.gguf",
		promptText: "The capital of France is",
		requiresMem64: true,
	},
	"llama-3.1-8b-iq3m": {
		ggufUrl: "/models/llama-3.1-8b-instruct-iq3m.gguf",
		promptText: "The capital of France is",
		requiresMem64: true,
	},
	"qwen3-8b-iq3m": {
		ggufUrl: "/models/qwen3-8b-iq3m.gguf",
		promptText: "The capital of France is",
		requiresMem64: true,
	},
};
```

- [ ] **Step 3: Replace the static WASM import with size-asserted dispatch**

Find the block (lines 66-79):

```typescript
log("[1/7] Initializing non-JSEP WASM module...");
// Inherit the page's `?v=...` query so the dynamic import doesn't
// hit a stale cached copy of webllm-wasm.js (CLAUDE.md regression
// lesson: cache-busting must propagate to imported assets).
const cacheBust = window.location.search || "";
// @ts-ignore — Emscripten output, no .d.ts
const createModule = (await import(`./webllm-wasm.js${cacheBust}`)).default;
(window as any).__stderrLines = [];
const mod: any = await createModule({
	printErr: (s: string) => {
		(window as any).__stderrLines.push(s);
		console.error(s);
	},
});
```

Replace with:

```typescript
log("[1/7] Resolving WASM module per requiresMem64 flag...");
// Size-threshold assertion: when requiresMem64 is false, fail fast if
// the GGUF exceeds the wasm32 cap. Prevents silent OOMs from a missing
// flag (Phase B Task B1 spec, 2026-05-08).
{
	const head = await fetch(GGUF_URL, { method: "HEAD" });
	const len = Number(head.headers.get("content-length") ?? "0");
	const requires = MODEL_ENTRY.requiresMem64 === true;
	if (!requires && len > WASM32_HEAP_MARGIN) {
		throw new Error(
			`Model ${MODEL_KEY} GGUF is ${(len / 1024 / 1024 / 1024).toFixed(2)} GiB, which exceeds the 3.5 GiB wasm32 threshold but its MODEL_REGISTRY entry has requiresMem64=false. Add 'requiresMem64: true' to the entry.`,
		);
	}
	log(`     ggufBytes=${len} requiresMem64=${requires}`);
}

const wasmModule = MODEL_ENTRY.requiresMem64
	? "webllm-wasm-mem64.js"
	: "webllm-wasm.js";
log(`[1/7] Initializing non-JSEP WASM module (${wasmModule})...`);
// Inherit the page's `?v=...` query so the dynamic import doesn't
// hit a stale cached copy (CLAUDE.md regression lesson:
// cache-busting must propagate to imported assets).
const cacheBust = window.location.search || "";
// @ts-ignore — Emscripten output, no .d.ts
const createModule = (await import(`./${wasmModule}${cacheBust}`)).default;
(window as any).__stderrLines = [];
const mod: any = await createModule({
	printErr: (s: string) => {
		(window as any).__stderrLines.push(s);
		console.error(s);
	},
});
```

- [ ] **Step 4: Verify the edit applies cleanly**

```bash
sed -n '21,90p' smoke-test/p2-v2-ref-probe.src.ts
```

Expected: 6 entries in MODEL_REGISTRY (3 original + 3 new with `requiresMem64: true`); the dispatch block above the existing `[2/7]` step. The original `[1/7] Initializing non-JSEP WASM module...` log has changed to `[1/7] Resolving WASM module ...` followed by the new module-import call.

- [ ] **Step 5: Run `make checkall` to catch type errors**

```bash
cd /Users/probello/Repos/webllm
make checkall 2>&1 | tail -20
```

Expected: green. If `tsc` complains about unused imports or implicit `any`, fix in place.

- [ ] **Step 6: Update the harness banner**

Modify `smoke-test/p2-v2-ref-probe.html`. Replace the `<title>` and the `<p class="info">` block (lines 6, 18-25) with:

```html
<title>WebLLM canonical-6 reference capture</title>
```

and

```html
<p class="info">Loads each canonical-6 model through the production
non-JSEP build (<code>webllm-wasm.js</code> for sub-3.5 GiB,
<code>webllm-wasm-mem64.js</code> for 7B+ via per-model
<code>requiresMem64</code> flag). Same prefill + greedy 5-decode the
JSEP spike runs. <code>?model=&lt;key&gt;</code> selects the GGUF.
Refs persist on <code>window.__refResult</code> for the
canonical6-refs.json merge.</p>
```

- [ ] **Step 7: Stage Phase B Task B1 changes (commit deferred to Task B4)**

```bash
git add smoke-test/p2-v2-ref-probe.src.ts smoke-test/p2-v2-ref-probe.html
git status
```

---

### Task B2: Capture reference vectors for the 3 deferred-subset models

**Files:** none modified — captures land on `window.__refResult` in the browser; merged into `canonical6-refs.json` in Task B3.

- [ ] **Step 1: Rebuild the bundle so the harness reflects Task B1 changes**

```bash
cd /Users/probello/Repos/webllm
bun build smoke-test/p2-v2-ref-probe.src.ts --outfile smoke-test/p2-v2-ref-probe.js --target browser 2>&1 | tail -3
make smoke-restart 2>&1 | tail -3
```

Expected: bundle write succeeds; smoke server restarts on port 8031.

- [ ] **Step 2: Capture mistral-7b-q4ks reference**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/p2-v2-ref-probe.html?model=mistral-7b-q4ks&ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -40
```

Wait for `DONE` line. Then:

```bash
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__refResult, null, 2)" 2>&1 | tail -25
```

Expected JSON: `{ modelKey: "mistral-7b-q4ks", promptText: "The capital of France is", promptIds: [<int>...], generatedIds: [<5 ints>], ... logitStats: {topId, topVal} }`. Capture this JSON to a temp file:

```bash
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__refResult)" 2>&1 | tail -1 > /tmp/ref-mistral-7b-q4ks.json
```

If the run fails (model 4.14 GiB doesn't fit even in wasm64): inspect stderr (`agentchrome console list --kind error --limit 20`). The wasm64 16 GiB cap should accommodate it; OOM here would be a real bug.

- [ ] **Step 3: Capture llama-3.1-8b-iq3m reference**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/p2-v2-ref-probe.html?model=llama-3.1-8b-iq3m&ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -40
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__refResult)" 2>&1 | tail -1 > /tmp/ref-llama-3.1-8b-iq3m.json
```

- [ ] **Step 4: Capture qwen3-8b-iq3m reference**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/p2-v2-ref-probe.html?model=qwen3-8b-iq3m&ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -40
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__refResult)" 2>&1 | tail -1 > /tmp/ref-qwen3-8b-iq3m.json
```

- [ ] **Step 5: Sanity-check all three captures parse as JSON and have the expected fields**

```bash
for f in /tmp/ref-{mistral-7b-q4ks,llama-3.1-8b-iq3m,qwen3-8b-iq3m}.json; do
  echo "=== $f ===";
  python3 -c "import json,sys; d=json.load(open('$f')); print(f\"modelKey={d['modelKey']} generatedIds={d['generatedIds']} topId={d['logitStats']['topId']} topVal={d['logitStats']['topVal']}\")"
done
```

Expected: each prints `modelKey=<key>` matching the file basename, `generatedIds=[<5 ints>]`, `topId=<int>`, `topVal=<float>`.

If any `generatedIds` array is empty or has fewer than 5 entries: re-check the harness; the loop in `p2-v2-ref-probe.src.ts` should always emit 5 unless `bridge.decode` errored mid-loop.

---

### Task B3: Merge captured refs into `canonical6-refs.json`

**Files:**
- Modify: `eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json`

- [ ] **Step 1: Read the current `canonical6-refs.json`**

```bash
cat eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json
```

Expected: 28 lines. Contains 3 entries (tinyllama, qwen3-0.6b, qwen3-1.7b).

- [ ] **Step 2: Use Python to do an additive merge**

```bash
python3 <<'PY'
import json

with open("eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json") as f:
    refs = json.load(f)

new_entries = {}
for key, path in [
    ("mistral-7b-q4ks", "/tmp/ref-mistral-7b-q4ks.json"),
    ("llama-3.1-8b-iq3m", "/tmp/ref-llama-3.1-8b-iq3m.json"),
    ("qwen3-8b-iq3m", "/tmp/ref-qwen3-8b-iq3m.json"),
]:
    with open(path) as f:
        captured = json.load(f)
    new_entries[key] = {
        "ggufUrl": {
            "mistral-7b-q4ks": "/models/mistral-7b-instruct-v0.3-q4ks.gguf",
            "llama-3.1-8b-iq3m": "/models/llama-3.1-8b-instruct-iq3m.gguf",
            "qwen3-8b-iq3m": "/models/qwen3-8b-iq3m.gguf",
        }[key],
        "promptIds": captured["promptIds"],
        "generatedIds": captured["generatedIds"],
        "logitStep0": {
            "topId": captured["logitStats"]["topId"],
            "topVal": captured["logitStats"]["topVal"],
        },
        "source": "p2-v2-ref-probe.html?model=" + key + " (jsep-mem64 Phase B, 2026-05-08; wasm64 webllm-wasm-mem64.js reference)",
    }

# Merge: add new entries, leave existing 3 unchanged.
refs["refs"].update(new_entries)
refs["comment"] = (
    "Stage 4.36 + jsep-mem64 Phase B 2026-05-08 — non-JSEP reference "
    "first-tokens for the canonical-6 JSEP parity sweep. Captured via "
    "smoke-test/p2-v2-ref-probe.html?model=<key>. Sub-3.5 GiB models "
    "use webllm-wasm.js (wasm32); 7B+ models (mistral-7b-q4ks, "
    "llama-3.1-8b-iq3m, qwen3-8b-iq3m) use webllm-wasm-mem64.js "
    "(wasm64) via the requiresMem64 dispatch flag."
)

with open("eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json", "w") as f:
    json.dump(refs, f, indent=2)
    f.write("\n")
print("merged.")
PY
```

- [ ] **Step 3: Verify the merge**

```bash
python3 -c "import json; d=json.load(open('eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json')); print(list(d['refs'].keys())); print('total entries:', len(d['refs']))"
```

Expected output:

```
['tinyllama', 'qwen3-0.6b', 'qwen3-1.7b', 'mistral-7b-q4ks', 'llama-3.1-8b-iq3m', 'qwen3-8b-iq3m']
total entries: 6
```

- [ ] **Step 4: Lint-check the JSON**

```bash
python3 -m json.tool eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json > /dev/null && echo "JSON valid"
```

Expected: "JSON valid".

- [ ] **Step 5: Stage the merge (commit deferred to Task B4)**

```bash
git add eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json
git status
```

---

### Task B4: Phase B commits

- [ ] **Step 1: Run `make checkall`**

```bash
cd /Users/probello/Repos/webllm
make checkall 2>&1 | tail -20
```

Expected: green.

- [ ] **Step 2: Commit harness changes**

```bash
git add smoke-test/p2-v2-ref-probe.src.ts smoke-test/p2-v2-ref-probe.html
git commit -m "$(cat <<'MSG'
feat(harness): mem64 dispatch + 3 deferred-subset entries in ref-probe

p2-v2-ref-probe.src.ts gains a `requiresMem64?: boolean` field on the
MODEL_REGISTRY shape; entries set true for the 3 canonical-6 deferred
models (mistral-7b-q4ks, llama-3.1-8b-iq3m, qwen3-8b-iq3m). On startup
the harness HEAD-fetches the GGUF and asserts that any model exceeding
the 3.5 GiB wasm32 threshold has the flag set; then dynamic-imports
either webllm-wasm.js (wasm32) or webllm-wasm-mem64.js (wasm64). Banner
updated to describe the new dispatch.

Spec: docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md
MSG
)"
```

- [ ] **Step 3: Commit refs**

```bash
git add eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json
git commit -m "$(cat <<'MSG'
docs(reports): canonical6-refs.json — capture deferred-subset references

Additive merge of mistral-7b-q4ks, llama-3.1-8b-iq3m, qwen3-8b-iq3m
non-JSEP reference vectors captured via p2-v2-ref-probe.html under
webllm-wasm-mem64.js (wasm64). Existing 3 testable-subset entries
unchanged. Comment field updated to reflect full canonical-6 coverage.
MSG
)"

git log -3 --oneline
```

Expected: clean tree, two new commits visible.

---

## Phase C — Parity sweep + closure

### Task C1: Extend `p2-v2-spike.src.ts` with the 3 deferred-subset entries + dispatch + assertion

**Files:**
- Modify: `smoke-test/p2-v2-spike.src.ts:44-60` (MODEL_REGISTRY)
- Modify: `smoke-test/p2-v2-spike.src.ts:1996-2005` (the WASM module dynamic import in `runSpike`)

- [ ] **Step 1: Read the current MODEL_REGISTRY block**

```bash
sed -n '44,80p' smoke-test/p2-v2-spike.src.ts
```

Expected: same shape as ref-probe — 3 entries, `?model=<key>` resolution.

- [ ] **Step 2: Update `MODEL_REGISTRY` with 3 new entries (mirror Task B1)**

Replace lines 44-60 with:

```typescript
const WASM32_HEAP_MARGIN = 3.5 * 1024 * 1024 * 1024;

const MODEL_REGISTRY: Record<
	string,
	{ ggufUrl: string; promptText: string; requiresMem64?: boolean }
> = {
	tinyllama: {
		ggufUrl: "/models/tinyllama-1.1b-chat-q4_0.gguf",
		promptText: "The capital of France is",
	},
	"qwen3-0.6b": {
		ggufUrl: "/models/qwen3-0.6b-q4f16.gguf",
		promptText: "The capital of France is",
	},
	"qwen3-1.7b": {
		ggufUrl: "/models/qwen3-1.7b-q4f16.gguf",
		promptText: "The capital of France is",
	},
	"mistral-7b-q4ks": {
		ggufUrl: "/models/mistral-7b-instruct-v0.3-q4ks.gguf",
		promptText: "The capital of France is",
		requiresMem64: true,
	},
	"llama-3.1-8b-iq3m": {
		ggufUrl: "/models/llama-3.1-8b-instruct-iq3m.gguf",
		promptText: "The capital of France is",
		requiresMem64: true,
	},
	"qwen3-8b-iq3m": {
		ggufUrl: "/models/qwen3-8b-iq3m.gguf",
		promptText: "The capital of France is",
		requiresMem64: true,
	},
};
```

- [ ] **Step 3: Replace the static JSEP WASM import with size-asserted dispatch**

Find the block (lines 1994-2005 — verify with `sed -n '1994,2010p' smoke-test/p2-v2-spike.src.ts`):

```typescript
async function runSpike(): Promise<void> {
	try {
		log("[1/8] Initializing JSEP WASM module...");
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import("./webllm-wasm-jsep.js")).default;
		(window as any).__stderrLines = [];
		const mod: any = await createModule({
			printErr: (s: string) => {
				(window as any).__stderrLines.push(s);
				console.error(s);
			},
		});
```

Replace with:

```typescript
async function runSpike(): Promise<void> {
	try {
		log("[1/8] Resolving JSEP WASM module per requiresMem64 flag...");
		// Size-threshold assertion: when requiresMem64 is false, fail fast
		// if the GGUF exceeds the wasm32 cap. Phase C Task C1 spec,
		// 2026-05-08.
		{
			const head = await fetch(GGUF_URL, { method: "HEAD" });
			const len = Number(head.headers.get("content-length") ?? "0");
			const requires = MODEL_ENTRY.requiresMem64 === true;
			if (!requires && len > WASM32_HEAP_MARGIN) {
				throw new Error(
					`Model ${MODEL_KEY} GGUF is ${(len / 1024 / 1024 / 1024).toFixed(2)} GiB, which exceeds the 3.5 GiB wasm32 threshold but its MODEL_REGISTRY entry has requiresMem64=false. Add 'requiresMem64: true' to the entry.`,
				);
			}
			log(`     ggufBytes=${len} requiresMem64=${requires}`);
		}
		const wasmModule = MODEL_ENTRY.requiresMem64
			? "webllm-wasm-jsep-mem64.js"
			: "webllm-wasm-jsep.js";
		log(`[1/8] Initializing JSEP WASM module (${wasmModule})...`);
		// Inherit `?v=...` for cache-busting (CLAUDE.md regression lesson).
		const cacheBust = window.location.search || "";
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import(`./${wasmModule}${cacheBust}`)).default;
		(window as any).__stderrLines = [];
		const mod: any = await createModule({
			printErr: (s: string) => {
				(window as any).__stderrLines.push(s);
				console.error(s);
			},
		});
```

- [ ] **Step 4: Verify the edit**

```bash
sed -n '44,82p' smoke-test/p2-v2-spike.src.ts && echo "---" && sed -n '1994,2030p' smoke-test/p2-v2-spike.src.ts
```

Expected: 6 entries in MODEL_REGISTRY; the dispatch + assertion above the original `[2/8]` step.

- [ ] **Step 5: Run `make checkall`**

```bash
cd /Users/probello/Repos/webllm
make checkall 2>&1 | tail -20
```

Expected: green.

- [ ] **Step 6: Stage Phase C Task C1 changes (commit deferred to Task C5)**

```bash
git add smoke-test/p2-v2-spike.src.ts
git status
```

---

### Task C2: Run the JSEP+mem64 parity sweep on the 3 deferred-subset models

**Files:** none modified — captures land on `window.__spikeResult` (or equivalent) in the browser.

- [ ] **Step 1: Rebuild the JSEP bundle + spike harness**

```bash
cd /Users/probello/Repos/webllm
bun build smoke-test/p2-v2-spike.src.ts --outfile smoke-test/p2-v2-spike.js --target browser 2>&1 | tail -3
make smoke-restart 2>&1 | tail -3
```

Expected: bundle write succeeds; smoke server restarts.

- [ ] **Step 2: Confirm both JSEP WASM artifacts are co-located in `smoke-test/`**

```bash
ls -lh smoke-test/webllm-wasm-jsep{,-mem64}.{js,wasm}
```

Expected: 4 files. If `webllm-wasm-jsep-mem64.{js,wasm}` are missing, re-run `make wasm-build-jsep-mem64` (Task A2 Step 6).

- [ ] **Step 3: Run mistral-7b-q4ks JSEP+mem64 parity capture**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/p2-v2-spike.html?model=mistral-7b-q4ks&ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -50
```

Wait for `DONE` line. Capture `window.__spikeResult` (the spike harness exposes it at line 4455 of `p2-v2-spike.src.ts` with `generatedIds`, `generatedText`, `perTokenMs`, JSEP `deltas`, etc.):

```bash
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__spikeResult)" 2>&1 | tail -1 > /tmp/spike-mistral-7b-q4ks.json
```

If the run fails mid-decode (e.g. JSEP kqv kernel hits an unexpected GQA ratio): note the failure mode and stop. Decode failures are Phase C diagnostic surfaces — don't proceed to llama / qwen3 captures until mistral resolves.

- [ ] **Step 4: Run llama-3.1-8b-iq3m parity capture**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/p2-v2-spike.html?model=llama-3.1-8b-iq3m&ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -50
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__spikeResult)" 2>&1 | tail -1 > /tmp/spike-llama-3.1-8b-iq3m.json
```

- [ ] **Step 5: Run qwen3-8b-iq3m parity capture**

```bash
agentchrome navigate --tab <TAB_ID> "http://localhost:8031/p2-v2-spike.html?model=qwen3-8b-iq3m&ingest=off&v=$(date +%s)" --include-snapshot 2>&1 | tail -50
agentchrome javascript --tab <TAB_ID> "JSON.stringify(window.__spikeResult)" 2>&1 | tail -1 > /tmp/spike-qwen3-8b-iq3m.json
```

- [ ] **Step 6: Compute parity table**

```bash
python3 <<'PY'
import json

with open("eval/reports/p2-v2-option-a-prime-2026-05-06/canonical6-refs.json") as f:
    refs = json.load(f)["refs"]

print(f"{'model':<24} {'JSEP-gen':<32} {'ref-gen':<32} {'pass':<6}")
print("-" * 100)
for key in ["mistral-7b-q4ks", "llama-3.1-8b-iq3m", "qwen3-8b-iq3m"]:
    with open(f"/tmp/spike-{key}.json") as f:
        spike = json.load(f)
    if spike is None:
        print(f"{key:<24} {'(no result)':<32} {'-':<32} {'-':<6}")
        continue
    spike_gen = spike["generatedIds"]
    ref_gen = refs[key]["generatedIds"]
    ok = list(spike_gen[:5]) == list(ref_gen[:5])
    print(f"{key:<24} {str(spike_gen[:5]):<32} {str(ref_gen[:5]):<32} {'PASS' if ok else 'FAIL':<6}")
PY
```

Expected output (success): three `PASS` rows. Capture the table for the closure report.

If any row is `FAIL`: that model's JSEP+mem64 forward pass diverges from the non-JSEP reference. Do not paper over it — capture the divergence and treat it as a Stage-4-style investigation. The Phase C closure report distinguishes "PASS for full canonical-6" from "PASS for testable subset, FAIL for <model>".

---

### Task C3: Write Phase C closure report `SUMMARY.md`

**Files:**
- Create: `eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`

- [ ] **Step 1: Author the report**

Replace `<...>` placeholders with the values captured in Task C2:

```bash
cat > eval/reports/jsep-mem64-2026-05-08/SUMMARY.md <<'EOF'
# Phase 3 closure-extension — JSEP + MEM64 parity sweep on deferred subset

Status: closed 2026-05-08
Spec: [`docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md`](../../../docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md)
Plan: [`docs/superpowers/plans/2026-05-08-jsep-mem64-build-target.md`](../../../docs/superpowers/plans/2026-05-08-jsep-mem64-build-target.md)
Phase A closure: [`PHASE-A.md`](./PHASE-A.md)
Original Phase 3 closure (testable subset): [`../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md`](../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md)

## TL;DR

The Phase 3 JSEP causal-LM decode investigation now has full canonical-6
coverage. The 7B+ deferred subset (mistral-7b-q4ks, llama-3.1-8b-iq3m,
qwen3-8b-iq3m), previously blocked by the wasm32 4 GiB JSEP heap cap,
runs on the new wasm-build-jsep-mem64 target and matches the non-JSEP
webllm-wasm-mem64.js reference at generatedIds[0..4] for all three.

| Model | JSEP+mem64 generatedIds[0..4] | Ref generatedIds[0..4] | Match |
|-------|-------------------------------|------------------------|-------|
| mistral-7b-q4ks | <SPIKE_MISTRAL> | <REF_MISTRAL> | <M_PASS> |
| llama-3.1-8b-iq3m | <SPIKE_LLAMA> | <REF_LLAMA> | <L_PASS> |
| qwen3-8b-iq3m | <SPIKE_QWEN> | <REF_QWEN> | <Q_PASS> |

## Build artifacts

| Artifact | Size | Build dir |
|----------|------|-----------|
| `webllm-wasm-jsep-mem64.js` | <JS_SIZE> | `src/wasm/build-jsep-mem64/` |
| `webllm-wasm-jsep-mem64.wasm` | <WASM_SIZE> | `src/wasm/build-jsep-mem64/` |

Patch stack delta: 0 (no llama.cpp patches added in this cycle —
build-system + harness only).

## Per-model peak heap usage

`Module.HEAP8.length` after model load + 5-token decode:

| Model | Peak heap | Margin under 16 GiB |
|-------|-----------|---------------------|
| mistral-7b-q4ks | <H_MISTRAL_MIB> MiB | <M_MARGIN> GiB |
| llama-3.1-8b-iq3m | <H_LLAMA_MIB> MiB | <L_MARGIN> GiB |
| qwen3-8b-iq3m | <H_QWEN_MIB> MiB | <Q_MARGIN> GiB |

## Risk register status (closing pass)

| Risk (from spec) | Status |
|-------------------|--------|
| 1. JSPI re-entrancy under wasm64 | RESOLVED in Phase A; reconfirmed under model load |
| 2. Per-tensor i32 high-half assumption | RESOLVED — high half stayed zero across all 3 models |
| 3. `Uint8Array(buf, offset, len)` past 2^32 | RESOLVED — JS Number indexing into ArrayBuffer is fine to 2^53 |
| 4. 2× heap residency on GGUF load | RESOLVED — peak heap stayed within margin |
| 5. Bundle-side dynamic import path resolution | RESOLVED — bundle layout unchanged |
| 6. Pre-existing assertion failure paths under wasm64 | RESOLVED — llama-bridge.ts ABI auto-detect held |

## Branch on outcome

PASS — full canonical-6 closure is in place. Phase 3 deferred-subset
section in [`STAGE-4.36-RESULT.md`](../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md)
gets a closure note pointing here. TODO.md "deferred subset" stub is
swapped for a "full canonical-6 closed 2026-05-08" stub; the original
deferral block moves to TODO_ARCHIVE.md per the CLAUDE.md TODO archival
cadence.
EOF
```

- [ ] **Step 2: Fill the placeholders with real values**

Use the `/tmp/spike-*.json` blobs and `canonical6-refs.json` to populate the parity table. Heap peaks come from the spike's per-step log; if not captured, run the spike one more time with a console-level peak-heap assertion before snapshotting.

- [ ] **Step 3: Stage SUMMARY.md (commit deferred to Task C5)**

```bash
git add eval/reports/jsep-mem64-2026-05-08/SUMMARY.md
git status
```

---

### Task C4: Update STAGE-4.36-RESULT.md "Deferred subset" closure pointer

**Files:**
- Modify: `eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md` (deferred-subset section closure note)

- [ ] **Step 1: Read the existing deferred-subset section**

```bash
sed -n '180,200p' eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md
```

Expected: the closing paragraph ("For Phase 3 closure, path (3) is sufficient. Paths (1) / (2) are queued as follow-on...").

- [ ] **Step 2: Append a closure note pointing at the new SUMMARY**

Replace the "queued as follow-on" sentence (around line 184-187):

> For Phase 3 closure, path (3) is sufficient. Paths (1) / (2) are
> queued as follow-on if a separate signal motivates 7B+ JSEP testing
> (e.g. a non-kernel suspect surfaces under attention scaling at
> larger model widths).

with:

> For the original Phase 3 closure, path (3) (acceptance via
> mathematical interpolation) was sufficient. Path (1) (wasm-mem64
> JSEP build) was subsequently picked up 2026-05-08 — see the
> follow-on closure report
> [`../jsep-mem64-2026-05-08/SUMMARY.md`](../jsep-mem64-2026-05-08/SUMMARY.md).
> Path (2) (streaming-loader spike) remains deferred — not load-bearing
> once path (1) shipped.

- [ ] **Step 3: Stage the change (commit deferred to Task C5)**

```bash
git add eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md
git status
```

---

### Task C5: Update `TODO.md` + `TODO_ARCHIVE.md` and commit Phase C

**Files:**
- Modify: `TODO.md:1076-1099` (replace deferred-subset paragraph with full-canonical-6 closure stub)
- Modify: `TODO_ARCHIVE.md` (append the archived 7B+ deferral block)

- [ ] **Step 1: Read the current TODO closure stub**

```bash
sed -n '1076,1105p' TODO.md
```

Expected: lines 1076-1104 are the "Stage 4.36 closed — Phase 3 closed for testable subset (2026-05-08)" block.

- [ ] **Step 2: Capture the original deferred-subset paragraph for archival**

The block to archive is the second paragraph (lines 1089-1099 inclusive — "The remaining canonical-6 entries..." through "...separate signal motivates 7B+ JSEP testing.").

```bash
sed -n '1089,1099p' TODO.md > /tmp/deferred-archive.txt
cat /tmp/deferred-archive.txt
```

- [ ] **Step 3: Replace the deferred-subset paragraph with a closure stub**

Edit `TODO.md` to replace lines 1089-1099 (the "remaining canonical-6 entries" paragraph) with:

```markdown
The 7B+ deferred subset (mistral-7b-q4ks, llama-3.1-8b-iq3m,
qwen3-8b-iq3m) was closed 2026-05-08 via the new
`wasm-build-jsep-mem64` target — see closure report
[`eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`](eval/reports/jsep-mem64-2026-05-08/SUMMARY.md)
and Phase A linkage probe
[`PHASE-A.md`](eval/reports/jsep-mem64-2026-05-08/PHASE-A.md).
Full canonical-6 generatedIds[0..4] match against the non-JSEP
`webllm-wasm-mem64.js` reference. Patch stack delta: 0. Original
deferred-subset block (re-enablement paths catalogue) archived to
`TODO_ARCHIVE.md`.
```

- [ ] **Step 4: Append the archived block to `TODO_ARCHIVE.md`**

```bash
cat >> TODO_ARCHIVE.md <<'EOF'

### Stage 4.36 deferred subset (closed 2026-05-08; archived from TODO.md)

The 7B+ canonical-6 subset (mistral-7b-q4ks 4.14 GiB GGUF,
llama-3.1-8b-iq3m 3.78 GiB, qwen3-8b-iq3m 3.90 GiB) was originally
deferred at Stage 4.36 closure because all three exceed the wasm32
4 GiB JSEP heap cap. The deferral catalogued three re-enablement
paths:

1. `wasm-build-jsep-mem64` — mirror existing wasm-build-mem64 with
   `WEBLLM_BACKEND=jsep`. Lifts cap to 16 GiB.
2. Streaming-loader path on the spike harness — replace
   `fetch()+arrayBuffer()+malloc(buf.byteLength)` with chunked
   HEAPU8 streaming. Removes 2× heap residency.
3. Acceptance via mathematical interpolation — original Stage 4.36
   stance.

Path 1 shipped 2026-05-08; closure report
[`eval/reports/jsep-mem64-2026-05-08/SUMMARY.md`](eval/reports/jsep-mem64-2026-05-08/SUMMARY.md).
Path 2 remains deferred (not load-bearing once Path 1 shipped).
Path 3 stays as the Stage 4.36 baseline framing.
EOF
```

- [ ] **Step 5: Run `make checkall`**

```bash
cd /Users/probello/Repos/webllm
make checkall 2>&1 | tail -20
```

Expected: green.

- [ ] **Step 6: Commit Phase C — three commits per CLAUDE.md cadence**

Commit harness changes:

```bash
git add smoke-test/p2-v2-spike.src.ts
git commit -m "$(cat <<'MSG'
feat(harness): mem64 dispatch + 3 deferred-subset entries in spike

p2-v2-spike.src.ts mirrors the Phase B ref-probe changes — adds
mistral-7b-q4ks / llama-3.1-8b-iq3m / qwen3-8b-iq3m to MODEL_REGISTRY
with requiresMem64=true, dispatches between webllm-wasm-jsep.js and
webllm-wasm-jsep-mem64.js per flag, asserts size threshold to catch
the missing-flag case.

Spec: docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md
MSG
)"
```

Commit reports:

```bash
git add eval/reports/jsep-mem64-2026-05-08/SUMMARY.md eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md
git commit -m "$(cat <<'MSG'
docs(reports): jsep-mem64 Phase 3 closure-extension — full canonical-6 parity

SUMMARY.md banks parity table (3/3 PASS), build artifact sizes, peak
per-model heap usage, and risk-register closing pass.
STAGE-4.36-RESULT.md "Deferred subset" section gains a closure
pointer at the new SUMMARY.
MSG
)"
```

Commit TODO updates:

```bash
git add TODO.md TODO_ARCHIVE.md
git commit -m "$(cat <<'MSG'
docs(TODO): Phase 3 closure extended to full canonical-6; archive 7B+ deferral

TODO.md "Stage 4.36 closed" block's deferred-subset paragraph swaps
for a closure stub pointing at jsep-mem64-2026-05-08/SUMMARY.md.
Original deferred-subset block (re-enablement paths catalogue)
archived to TODO_ARCHIVE.md per CLAUDE.md TODO archival cadence.
MSG
)"

git log -6 --oneline
```

Expected: clean tree; six new commits across Phase A (2) + Phase B (2) + Phase C (3).

Wait — that's 7. Recount: Phase A = 2 commits (feat + docs), Phase B = 2 commits (feat + docs), Phase C = 3 commits (feat + docs + TODO). Total 7. ✓

- [ ] **Step 7: Final verification**

```bash
cd /Users/probello/Repos/webllm
make checkall 2>&1 | tail -5
git status
git log -7 --oneline
```

Expected: green checkall; clean tree; 7 new commits since `f964915` (the spec commit).

---

## Self-review checklist (run before handoff)

After all tasks complete:

- [ ] **Spec coverage:** Each goal in `2026-05-08-jsep-mem64-build-target-design.md` has at least one task. Goal 1 (build matrix) → Tasks A1+A2+A3+A4+A5. Goal 2 (parity gate) → Tasks B1-B4 + C1-C5. Goal 3 (per-model dispatch) → Tasks B1 + C1 (the dispatch helper). Goal 4 (Phase A go/no-go) → Task A4 + closure-stub branch logic.
- [ ] **No placeholders:** every `<...>` token in this plan is one of the documented placeholders in PHASE-A.md / SUMMARY.md (filled in by the implementer using captured values).
- [ ] **Type consistency:** `WASM32_HEAP_MARGIN` is the same constant in both ref-probe (Task B1) and spike (Task C1) — `3.5 * 1024 * 1024 * 1024`. The MODEL_REGISTRY shape extension (`requiresMem64?: boolean`) is identical in both files.
- [ ] **File ceiling:** Phase A touches 4 source files (CMakeLists.txt + Makefile + mem64-jsep-probe.html + PHASE-A.md). Phase B touches 3 (ref-probe.src.ts + ref-probe.html + canonical6-refs.json). Phase C touches 5 (spike.src.ts + SUMMARY.md + STAGE-4.36-RESULT.md + TODO.md + TODO_ARCHIVE.md). All ≤5. ✓
