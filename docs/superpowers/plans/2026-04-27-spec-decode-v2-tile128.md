# §26 §C-v2-A re-measurement under §22 tile=128 — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the §C-v2-A speculative-decode gate matrix (4 cells × 3-trial median) on the `feat/spec-decode-v2-greedy` side branch with §22's `--prefill-tile 128` plumbing cherry-picked in. Produce one decisive empirical datapoint that closes (or, less likely, reopens) §C-v2-A.

**Architecture:** Cherry-pick the 5 §22 implementation commits onto the side branch (skipping §23's registry auto-default to keep the variable under test unambiguous). No `src/` change. Drive 4 perf.ts cells via a `run-matrix.sh` script. Apply the §C-v2-A spec gates to the medians. Write SUMMARY + TODO §26 closure entry on `main`.

**Tech Stack:** Bun (test + bundler), Emscripten/WASM (llama.cpp build), `eval/perf.ts` (measurement harness), bash (matrix driver), git (cherry-pick + dual-branch commits).

**Spec reference:** `docs/superpowers/specs/2026-04-27-spec-decode-v2-tile128-design.md` (`b23ccc9` on `main`).

**Predecessor closures:** §C v1 (§19), §C v2-A (`646320c` on side branch), §22 (`a73ad88` on `main`).

---

## File Structure

| File                                                                                       | Status         | Purpose                                                                       |
|--------------------------------------------------------------------------------------------|----------------|-------------------------------------------------------------------------------|
| `eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh`                             | Create (side)  | 4-cell × 3-trial driver script. Mirrors `eval/reports/prefill-tiling-2026-04-27/run-matrix.sh` shape. |
| `eval/reports/spec-decode-v2-tile128-2026-04-27/cell-{1,2,3,4}.log`                        | Create (side)  | Raw `perf.ts` output per cell, all 3 trials concatenated.                     |
| `eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md`                                | Create (side)  | Cell table, gate evaluation, decision-rule outcome.                           |
| `TODO.md`                                                                                  | Modify (main)  | Append §26 closure entry under "### Completed on 2026-04-27"; refresh resumption checklist. |
| `eval/perf.ts`                                                                             | Cherry-picked  | Acquires `--prefill-tile <n>` flag from `18e1677`. Side-branch `--drafter`/`--draft-length` flags retained. Possible 5-line conflict. |
| `src/inference/model-inference.ts`                                                         | Cherry-picked  | Acquires `prefillTileSize` ctor option + tile dispatcher from `c38fb8f`. Possible conflict around the forward path the v2-A driver wires through. |
| `tests/prefill-tiling-config.test.ts` + `tests/prefill-tiling-equivalence.test.ts`         | Cherry-picked  | §22 ctor-option + equivalence-stub tests. Should pass on side branch unchanged. |
| `smoke-test/real-model-page.js`                                                            | Cherry-picked  | `?prefillTile=N` URL param plumbing (not measured in this cycle but rides along). |
| `smoke-test/webllm-bundle.js`, `smoke-test/webllm-wasm.{js,wasm}`                          | Rebuild        | Bundle artifacts must contain the cherry-picked ctor option / dispatcher.     |

**No file lives on both branches inconsistently.** All cherry-picked changes go to the side branch. Only the TODO §26 closure entry lands on `main`.

---

## Task 0: Pre-flight verification

**Files:** none (read-only checks).

- [ ] **Step 1: Confirm working tree is clean and on `main`**

```bash
git status -s
git rev-parse --abbrev-ref HEAD
```

Expected: empty status output; current branch `main`.

- [ ] **Step 2: Confirm main tip and ship gate**

```bash
git log --oneline -1
make checkall 2>&1 | tail -5
```

Expected: tip is `b23ccc9 docs(spec): §26 §C-v2-A re-measurement under §22 tile=128` (or later if the closure cycle has progressed). `make checkall` returns 427 pass / 11 skip / 0 fail (or current ship-gate equivalent — confirm zero fail).

- [ ] **Step 3: Confirm side branch state**

```bash
git log feat/spec-decode-v2-greedy --oneline -5
git merge-base feat/spec-decode-v2-greedy main
```

Expected: side branch tip is `646320c docs(TODO): §24 — §C v2-A measured + CLOSED on §22.5 ship gates`. Merge-base is `77a5e118…` (pre-§22).

- [ ] **Step 4: Confirm low-α fixture is on side branch and high-α fixture is `prefill-256`**

```bash
git show feat/spec-decode-v2-greedy:eval/fixtures/long-prompts.ts | grep -E "creative-low-alpha|prefill-256"
```

Expected: both keys present in `LONG_PROMPTS`.

- [ ] **Step 5: Confirm llama.cpp branch state**

```bash
git -C ~/Repos/llama.cpp log --oneline -3 webllm-browser-patches
```

Expected: tip is `a536df4f4 ggml-webgpu: fix UB shift-by-32 in load_u32_at_src{,0}` (patch 11). The cherry-pick does not touch llama.cpp; this is a pre-flight safety check only.

---

## Task 1: Cherry-pick §22 implementation commits onto side branch

**Files:** `eval/perf.ts`, `src/inference/model-inference.ts`, `smoke-test/real-model-page.js`, `tests/prefill-tiling-config.test.ts`, `tests/prefill-tiling-equivalence.test.ts`. Modifications inherited from `c38fb8f`, `f281ac3`, `2fcc334`, `18e1677`. (Skip `8e21036` — Phase-0 diagnostic, evidence-only, drop on conflict.)

- [ ] **Step 1: Check out the side branch**

```bash
git checkout feat/spec-decode-v2-greedy
git status -s
```

Expected: empty status; HEAD at `646320c`.

- [ ] **Step 2: Cherry-pick §22 Task 1 (ctor option + dispatcher)**

```bash
git cherry-pick c38fb8f
```

Expected outcomes:

- **Clean apply:** proceed to step 3.
- **Conflict in `src/inference/model-inference.ts`:** resolve by keeping the v2-A `forwardVerifyArgmax` path intact AND introducing the §22 `prefillTileSize` ctor field + tile dispatcher in `forwardSingle()`. The v2-A driver does not call into the prefill path; the two changes are orthogonal at code level. Resolution recipe:
  - In the constructor, add `private readonly prefillTileSize: number;` field and assign from `opts?.prefillTileSize ?? 0` exactly as `c38fb8f` does.
  - In the prefill-graph dispatcher (the function `c38fb8f` modifies), keep the v2-A signature unchanged and add the `if (this.prefillTileSize > 0 && nTokens > this.prefillTileSize)` tile loop.
  - Run `git diff` and confirm `forwardVerifyArgmax` is untouched.
  - `git add src/inference/model-inference.ts && git cherry-pick --continue`.

- [ ] **Step 3: Cherry-pick §22 Task 2 (equivalence stub test)**

```bash
git cherry-pick f281ac3
```

Expected: clean apply (new test file, no conflict). If conflict, accept incoming version.

- [ ] **Step 4: Cherry-pick §22 Task 3 (smoke `?prefillTile=N`)**

```bash
git cherry-pick 2fcc334
```

Expected: clean apply (smoke-page-only).

- [ ] **Step 5: Cherry-pick §22 Task 4 (`--prefill-tile` flag on perf.ts)**

```bash
git cherry-pick 18e1677
```

Expected outcomes:

- **Clean apply:** proceed to step 6.
- **Conflict in `eval/perf.ts`:** the side branch added `--drafter` and `--draft-length` to the same flag-parsing block. Resolve by accepting both: keep `--drafter`, `--draft-length` from HEAD and add `--prefill-tile <n>` from the incoming commit. Both flags pass through to the `ModelInference` constructor; the `ctor` opts object accepts `{ drafter, prefillTileSize }` simultaneously.

- [ ] **Step 6: Verify cherry-pick log**

```bash
git log --oneline -7
```

Expected: 5 new commits at the tip authored by Paul Robello, with subjects:

```
<sha> chore(perf): §22 Task 5 — prefill-tile measurement matrix     ← if the matrix commit was also picked; SKIP, not in our list
<sha> feat(perf): §22 Task 4 — add --prefill-tile <n> flag to eval/perf.ts
<sha> feat(smoke): §22 Task 3 — wire ?prefillTile=N URL param through smoke page
<sha> test(inference): §22 Task 2 — prefill-tile equivalence test stub
<sha> feat(inference): §22 Task 1 — add prefillTileSize ctor option + tile dispatcher
646320c docs(TODO): §24 — §C v2-A measured + CLOSED on §22.5 ship gates
```

- [ ] **Step 7: Confirm Phase-0 diagnostic is NOT picked**

`8e21036` was deliberately skipped. If it accidentally landed (e.g., a prior cherry-pick range pulled it), drop it via `git rebase -i HEAD~6` and remove the line; force-update is unnecessary because the side branch isn't pushed.

---

## Task 2: Verify side-branch build + tests after cherry-pick

**Files:** none (read-only). The side-branch ship gate validates that the cherry-pick is structurally compatible with the v2-A driver.

- [ ] **Step 1: Run side-branch checkall**

```bash
make checkall 2>&1 | tail -10
```

Expected: pass count = (pre-pick count) + 5-7 from §22 tests (`prefill-tiling-config` adds 5 unit tests; `prefill-tiling-equivalence` adds 1 Bun-skipped stub). Side-branch pre-pick count was 426 + 19 v2-A tests = 445-ish; the new total should be ~450 pass / 12 skip / 0 fail. Exact numbers don't matter — **0 fail is the load-bearing assertion**.

- [ ] **Step 2: If checkall fails, diagnose**

If failure is in `tests/eval-models.test.ts` (registry shape) — that test was added by §23 (`0c50e03`), which we deliberately did NOT cherry-pick. The side branch has no `recommendedPrefillTile` field in its registry, so the test cannot exist on side branch. If it landed via accidental cherry-pick range, remove it: `git rm tests/eval-models.test.ts && git commit -m "test: drop §23 registry-shape tests (deferred to ship cycle)"`.

If failure is elsewhere, surface the failure to the user — it indicates a real conflict-resolution bug from Task 1 step 2.

- [ ] **Step 3: Rebuild WASM bundle**

```bash
source ~/emsdk/emsdk_env.sh && make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/
```

Expected: WASM build succeeds; bundle artifacts updated. `ls -la smoke-test/webllm-bundle.js smoke-test/webllm-wasm.{js,wasm}` shows mtimes within the last few minutes.

- [ ] **Step 4: Smoke-verify tile=128 takes effect**

```bash
make smoke-restart
```

Then in another terminal:

```bash
agentchrome connect --status
# If a session exists, reuse it; otherwise launch fresh:
# agentchrome connect --launch --headless
agentchrome --port <PORT> tabs list  # grab existing smoke-test tab id
agentchrome --port <PORT> --tab <TAB_ID> navigate "http://localhost:8031/?v=$(date +%s)&model=tinyllama-1.1b-chat-q4_0&prefillTile=128"
```

Expected: page loads, model downloads, [7/8] decode produces coherent English ("Why don't scientists trust atoms…" or similar). Console shows no errors. The `prefillTile` URL param is plumbed but for TinyLlama it has no behavioural effect on the gate matrix — this is a smoke check that the cherry-pick didn't break the bundle.

If the page errors at boot or generates gibberish, the cherry-pick conflict resolution is suspect. Roll back: `git reset --hard 646320c` and surface the issue to the user.

---

## Task 3: Author run-matrix.sh

**Files:**
- Create: `eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh`

- [ ] **Step 1: Create the report directory**

```bash
mkdir -p eval/reports/spec-decode-v2-tile128-2026-04-27
```

- [ ] **Step 2: Write the matrix driver script**

Save to `eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh`:

```bash
#!/usr/bin/env bash
# §26 §C-v2-A re-measurement matrix under §22 tile=128.
# 4 cells × 3-trial median. Side branch: feat/spec-decode-v2-greedy.
# Plan: docs/superpowers/plans/2026-04-27-spec-decode-v2-tile128.md
# Spec: docs/superpowers/specs/2026-04-27-spec-decode-v2-tile128-design.md
set -euo pipefail

REPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git rev-parse --show-toplevel)"

TARGET="qwen3-8b-iq3m"
DRAFTER="qwen3-0.6b-q4f16"
DRAFT_LEN=4
TILE=128
TRIALS=3
DECODE_TOKENS=64

run_cell () {
  local cell_id="$1"; local fixture="$2"; local with_drafter="$3"; local label="$4"
  local out="$REPORT_DIR/cell-${cell_id}.log"
  echo "============================================================" | tee -a "$out"
  echo "CELL $cell_id: $label"                                          | tee -a "$out"
  echo "  fixture=$fixture, with_drafter=$with_drafter, tile=$TILE"     | tee -a "$out"
  echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"                                | tee -a "$out"
  echo "============================================================" | tee -a "$out"

  for trial in $(seq 1 "$TRIALS"); do
    echo "" | tee -a "$out"
    echo "--- trial $trial/$TRIALS ---" | tee -a "$out"
    local cmd=(bun run eval/perf.ts \
      --model "$TARGET" \
      --prompt-fixture "$fixture" \
      --decode-tokens "$DECODE_TOKENS" \
      --temperature 0 \
      --prefill-tile "$TILE")
    if [[ "$with_drafter" == "yes" ]]; then
      cmd+=(--drafter "$DRAFTER" --draft-length "$DRAFT_LEN")
    fi
    echo "+ ${cmd[*]}" | tee -a "$out"
    "${cmd[@]}" 2>&1 | tee -a "$out"
  done
}

run_cell 1 prefill-256        no  "target alone, high-α templated, tile=128"
run_cell 2 creative-low-alpha no  "target alone, low-α creative,   tile=128"
run_cell 3 prefill-256        yes "target + drafter K=4, high-α templated, tile=128"
run_cell 4 creative-low-alpha yes "target + drafter K=4, low-α creative,   tile=128"

echo ""
echo "All cells complete. Logs at $REPORT_DIR/cell-{1,2,3,4}.log"
```

- [ ] **Step 3: Make it executable and commit on side branch**

```bash
chmod +x eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh
git add eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh
git commit -m "chore(perf): §26 add run-matrix.sh for §C-v2-A tile=128 re-measurement"
```

Expected: single commit on side branch tip.

---

## Task 4: Run the matrix

**Files:**
- Create: `eval/reports/spec-decode-v2-tile128-2026-04-27/cell-1.log`
- Create: `eval/reports/spec-decode-v2-tile128-2026-04-27/cell-2.log`
- Create: `eval/reports/spec-decode-v2-tile128-2026-04-27/cell-3.log`
- Create: `eval/reports/spec-decode-v2-tile128-2026-04-27/cell-4.log`

- [ ] **Step 1: Confirm system is quiet**

Close other heavy processes. The §C-v2-A close commit `4ff14a2` documented that 2026-04-27 saw a system slowdown that had to be noted; we want to avoid repeating that. Check Activity Monitor / `top` and confirm no other GPU-bound app is running.

- [ ] **Step 2: Execute the matrix**

```bash
./eval/reports/spec-decode-v2-tile128-2026-04-27/run-matrix.sh
```

Expected runtime: ~25-30 minutes total. Each cell runs ~6-8 minutes (cold-load ~2 min + 3 trials × ~30 s decode + warmup overhead).

Watch for in the log output:

- **Cells 1, 2:** lines like `decode tok/s p50 = 16.2` (expected ~16 tok/s for qwen3-8b-iq3m without drafter).
- **Cells 3, 4:** lines like `decode tok/s p50 = 5.8` (expected ~5-7 tok/s if §C-v2-A close numbers reproduce; >25 tok/s would be the surprise outcome that flips the gate). Also expect lines indicating drafter engagement and per-step α.
- **Adaptive gate fire on cell 4:** look for `adaptive gate fired at step N` or equivalent. The v2-A driver logs this; the spec requires fire within first 16 steps.

- [ ] **Step 3: Sanity-check coherence on cells 3 + 4**

After the matrix completes, eyeball the generated text in each cell-3 and cell-4 trial. Output should be coherent English (continuation of "It was the morning after the storm when…" for cell 4; a software-team explanation for cell 3). Any gibberish indicates a contract gate or vocab-mismatch bug, not a perf number — flag and stop.

- [ ] **Step 4: Commit logs on side branch**

```bash
git add eval/reports/spec-decode-v2-tile128-2026-04-27/cell-*.log
git commit -m "chore(perf): §26 raw matrix logs (4 cells × 3 trials, tile=128)"
```

---

## Task 5: Compute medians + write SUMMARY.md

**Files:**
- Create: `eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md`

- [ ] **Step 1: Extract the 3-trial p50 median per cell**

```bash
for c in 1 2 3 4; do
  echo "=== cell $c ==="
  grep -E "decode tok/s|prefill ms|alpha|adaptive gate" \
    eval/reports/spec-decode-v2-tile128-2026-04-27/cell-${c}.log
done
```

Compute per-cell median manually from the 3 trial outputs (sort 3 numbers, take middle one).

- [ ] **Step 2: Compute gate ratios**

- Gate 1 ratio = cell 3 median / cell 1 median. Pass if ≥ 1.5.
- Gate 2 ratio = cell 4 median / cell 2 median. Pass if ≥ 0.95 AND adaptive gate fired within first 16 steps on cell 4.

- [ ] **Step 3: Author SUMMARY.md**

Template — fill in the bracketed values from steps 1-2:

```markdown
# §26 §C-v2-A re-measurement under §22 tile=128 — SUMMARY

**Date:** 2026-04-27
**Side branch tip:** [git rev-parse HEAD on feat/spec-decode-v2-greedy]
**Plan:** ../../../docs/superpowers/plans/2026-04-27-spec-decode-v2-tile128.md
**Spec:** ../../../docs/superpowers/specs/2026-04-27-spec-decode-v2-tile128-design.md
**Predecessor:** §C-v2-A close at `646320c` (gate 1 0.36×, gate 2 0.78×).

## Matrix (3-trial median per cell)

| Cell | Target          | Drafter | Workload         | Tile | Decode tok/s p50 | Prefill ms p50 | α (mean) | Gate fired? |
|------|-----------------|---------|------------------|------|-----------------:|---------------:|---------:|-------------|
| 1    | qwen3-8b-iq3m   | —       | prefill-256      | 128  | [X.X]            | [XXXX]         | n/a      | n/a         |
| 2    | qwen3-8b-iq3m   | —       | creative-low-α   | 128  | [X.X]            | [XXXX]         | n/a      | n/a         |
| 3    | qwen3-8b-iq3m   | qwen3-0.6b-q4f16 K=4 | prefill-256 | 128 | [X.X] | [XXXX]      | [0.XX]  | [Yes/No, step N] |
| 4    | qwen3-8b-iq3m   | qwen3-0.6b-q4f16 K=4 | creative-low-α | 128 | [X.X] | [XXXX] | [0.XX] | [Yes/No, step N] |

## Gate evaluation

- **Gate 1 (speedup ≥1.5×):** cell 3 / cell 1 = [X.X] / [X.X] = **[Y.YY]×** — [PASS / FAIL]
- **Gate 2 (safety ≥0.95×):** cell 4 / cell 2 = [X.X] / [X.X] = **[Y.YY]×** — [PASS / FAIL]
- **Gate 2 adaptive fire:** [fired at step N / did not fire / n/a]

## Decision

[ONE OF:
 - "Both gates fail. §C-v2-A definitively closed under all known levers; resurrection requires architectural change (faster K+1 verify, MEMORY64 + 70B target). Side branch retained as archived infra."
 - "Gate 1 passes, gate 2 fails. Open §C-v2-B cycle (greedy-only contract + tightened adaptive disengage)."
 - "Gate 1 fails, gate 2 passes. Same as both-fail — gate 1 is load-bearing."
 - "Both gates pass. Open ship cycle: rebase side branch onto main, run full coherence + bench-full, follow §22 gated-ship template."
]

## Cross-cycle comparison

| Cycle           | Gate 1 ratio | Gate 2 ratio | Conditions |
|-----------------|-------------:|-------------:|------------|
| §C-v2-A (`646320c`) | 0.36×    | 0.78×        | tile=0 (no §22) |
| §26 (this)      | [Y.YY]×      | [Y.YY]×      | tile=128 (§22) |

Cell 1 baseline drift: [X.X] tok/s vs §C-v2-A's 16.0 tok/s = [+/−P.P]%. [Within / outside ±10% drift threshold; if outside, document the source.]

## Coherence

[Eyeball verdict: "All 12 trials produced coherent English output." or list any anomaly.]
```

- [ ] **Step 4: Commit SUMMARY on side branch**

```bash
git add eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md
git commit -m "docs(perf): §26 SUMMARY.md — gate evaluation under tile=128"
```

---

## Task 6: Write TODO §26 closure entry on `main`

**Files:**
- Modify: `TODO.md` on `main`

- [ ] **Step 1: Switch to main**

```bash
git checkout main
git status -s
```

Expected: empty status; current branch `main`.

- [ ] **Step 2: Append §26 entry under "### Completed on 2026-04-27"**

Open `TODO.md`, locate the line `### Completed on 2026-04-27` (around line 2257). The most recent §-numbered entry there is §24 (or §25, depending on whether a previous closure already extended the section). Insert §26 after the highest existing §-number for that date, mirroring the §24 prose shape.

The entry must contain:

1. One-line headline mentioning the gate verdict and "tile=128" so future grep finds it.
2. Branch name (`feat/spec-decode-v2-greedy` — side branch retained).
3. Reference to spec + plan paths.
4. Headline finding (the gate-1 and gate-2 ratios with absolute tok/s).
5. Cross-cycle comparison vs `646320c` (§C-v2-A).
6. Decision rule outcome (close / re-ship / etc.).
7. Reference to `eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md`.
8. Future-resurrection paths (architectural change required).
9. Ship-gate stamp (zero `src/` change on `main`; checkall unchanged).

Template — fill in numerical values from the SUMMARY:

```markdown
26. **§26 §C-v2-A re-measurement under §22 tile=128 — [CLOSED / GATE-1-PASS / SHIP].**
    Cherry-picked the §22 implementation commits onto the
    `feat/spec-decode-v2-greedy` side branch (skipping §23's registry
    auto-default for variable isolation) and re-ran the §C-v2-A
    4-cell gate matrix under explicit `--prefill-tile 128`. The verify
    graph at K+1=5 tokens is three orders of magnitude below the
    128-token tile threshold, so the cycle was a deliberate empirical
    test of the resumption checklist's "candidate, not a conclusion"
    framing.

    **Headline:**
    - Gate 1 (speedup ≥1.5×): [Y.YY]× — [cell 3 tok/s] vs
      [cell 1 tok/s] non-drafted baseline. **[PASS / FAIL]**.
    - Gate 2 (safety ≥0.95×): [Y.YY]× — [cell 4 tok/s] vs
      [cell 2 tok/s] non-drafted baseline. Adaptive gate fired
      [at step N / did not fire]. **[PASS / FAIL]**.

    **Cross-cycle:** vs `646320c` (§C-v2-A under tile=0): gate 1
    moved [0.36 → Y.YY] ([±]P.P%), gate 2 moved [0.78 → Y.YY]
    ([±]P.P%). [Within noise band — confirms tile=128 has no
    measurable effect on K+1=5 verify cost. / Surprise — see
    SUMMARY.md.]

    **Decision:** [§C-v2-A definitively closed under all known
    levers. Resurrection requires architectural change (faster K+1
    verify, MEMORY64 + 70B target). Side branch `feat/spec-decode-
    v2-greedy` retained as archived infra; do not merge to `main`. /
    Open §C-v2-B cycle. / Open ship cycle.]

    **Spec:** `docs/superpowers/specs/2026-04-27-spec-decode-v2-tile128-design.md`.
    **Plan:** `docs/superpowers/plans/2026-04-27-spec-decode-v2-tile128.md`.
    **Raw matrix:** `eval/reports/spec-decode-v2-tile128-2026-04-27/SUMMARY.md`
    (on side branch tip).

    **Ship-gate stamp:** zero `src/` change on `main`; checkall
    unchanged at the pre-§26 ship-gate count.
```

- [ ] **Step 3: Refresh resumption checklist (around line 2561)**

Find the candidate-next-levers section (currently lists §C-v2-A resurrection as item 1). Update item 1 to reflect §26's outcome:

- If §C-v2-A closed: replace "**§C-v2-A resurrection (conditional).**" with "~~§C-v2-A resurrection.~~ **CLOSED 2026-04-27 — §26.**" and shift remaining items up.
- If gate 1 passed: leave §C-v2-A in the list but add a §26 reference and the next-cycle scope.

- [ ] **Step 4: Commit TODO update on main**

```bash
git add TODO.md
git commit -m "docs(TODO): §26 — §C-v2-A re-measurement under tile=128 [CLOSED|SHIP]"
```

(Pick the bracketed verb based on the actual gate outcome.)

- [ ] **Step 5: Verify main checkall**

```bash
make checkall 2>&1 | tail -5
```

Expected: pass count unchanged from Task 0 step 2 (no `src/` change on `main` means no test count delta).

---

## Task 7: Branch hygiene + final state report

**Files:** none (read-only verification + branch label).

- [ ] **Step 1: Confirm side branch ends on a `docs(TODO)` commit if §26 also touched it**

`§26` should NOT touch the side branch's `TODO.md` — the TODO closure lives on `main` only. The side branch's tip after Task 5 should be the SUMMARY commit. Verify:

```bash
git log feat/spec-decode-v2-greedy --oneline -8
```

Expected: side-branch tip is the SUMMARY commit (Task 5). The cell-logs commit (Task 4) is below it; the run-matrix.sh commit (Task 3) is below that; the 5 cherry-picked §22 commits are below that; `646320c` is the floor.

- [ ] **Step 2: Confirm main tip + ship gate one final time**

```bash
git checkout main
git log --oneline -3
make checkall 2>&1 | tail -5
```

Expected: top commit is the §26 TODO update. `make checkall` returns the same pass count as pre-§26 with 0 fail.

- [ ] **Step 3: Report results to user**

Summarize in <200 words:

- Gate 1 verdict + ratio
- Gate 2 verdict + ratio
- Decision rule outcome
- Whether the §C-v2-A close is now definitive or a follow-on cycle is needed
- Side branch state + what (if anything) needs to happen next

---

## Self-Review (run after writing the plan)

**1. Spec coverage check:**
- Spec §1 Goal — covered by Tasks 4 + 5 + 6.
- Spec §3.1 Integration — covered by Tasks 1 + 2.
- Spec §3.2 Matrix — covered by Tasks 3 + 4.
- Spec §3.3 Decision rule — covered by Tasks 5 + 6 (TODO closure with the four outcome branches).
- Spec §4 Failure modes — Task 4 step 1 (system load), Task 4 step 3 (coherence check), Task 1 step 2 (cherry-pick conflict), Task 2 step 4 (smoke verify).
- Spec §5 Output artifacts — Tasks 3 (run-matrix.sh), 4 (cell logs), 5 (SUMMARY), 6 (TODO §26 on main).
- Spec §6 What this is NOT — enforced by Task 7 (no main `src/` change verified by checkall pass-count invariance).

**2. Placeholder scan:** the SUMMARY and TODO §26 templates contain `[X.X]` / `[Y.YY]` / `[CLOSED / SHIP]` brackets. These are intentional fill-in-the-blank slots for the operator running the plan, not plan-level placeholders. Each is paired with explicit instructions in the same task ("Fill in the bracketed values from steps 1-2"). No silent TBDs anywhere in the plan itself.

**3. Type/name consistency:**
- `prefillTileSize` (ctor option) used consistently in Task 1 step 2 + Task 2 step 4 + the §22 cherry-pick subjects.
- `--prefill-tile` (CLI flag) and `?prefillTile=N` (URL param) match the §22 commit subjects.
- `feat/spec-decode-v2-greedy` (branch name) used uniformly.
- `qwen3-8b-iq3m` (target), `qwen3-0.6b-q4f16` (drafter), `K=4`, `prefill-256` (high-α fixture), `creative-low-alpha` (low-α fixture) used uniformly across spec → plan → run-matrix.sh template → SUMMARY template → TODO closure template.

No issues found.

---

## Execution Handoff

Per global preference (CLAUDE.md): **Subagent-Driven** is the chosen execution mode. Proceeding to `superpowers:subagent-driven-development` to dispatch tasks.
