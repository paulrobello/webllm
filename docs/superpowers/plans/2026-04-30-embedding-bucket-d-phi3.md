# Bucket D extension — Phi-3.5-mini self-embedding implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend bucket D coverage to **Phi-3.5-mini-q4km** (the only fused-QKV / fused-gate-up arch in the fleet). API surface is fixed at v1; this is a registration + ref capture + parity + bench + closure cycle. First end-to-end exercise of `forwardForEmbedding`'s fused-path branches.

**Architecture:** No new code; reuses every artifact bucket D shipped (`ModelInference.embed`, `engine.embed` dispatch, `embeddingCapable` flag, parity harness, embed-perf bench, gate-by-quant-tier scheme). Adds a `q4km` tier to `QuantFormat` and calibrates its gate empirically (predicted band: 0.97-0.99, between `hyb` 0.995 and `iq3m` 0.90).

**Tech Stack:** TypeScript (Bun), `ModelInference.forwardForEmbedding` (already shipping), `eval/causal-embedder-parity.ts` browser harness, Python `transformers` for ref capture, **`hfdownloader` for HF fetches per CLAUDE.md policy**.

**Predecessor:** [bucket D plan](2026-04-29-embedding-bucket-d.md), [bucket D closure](../../../eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md).

---

## Task 1: Add `q4km` tier to QuantFormat; fix Phi-3.5-mini tag

**Files:** Modify: `eval/models.ts`.

- [ ] **Step 1: Add `q4km` to `QuantFormat`** (line 4 of `eval/models.ts`):

```ts
export type QuantFormat = "q4f16_1" | "q4f32_1" | "q0f16" | "q0f32" | "hyb" | "iq3m" | "q4km";
```

- [ ] **Step 2: Correct the `phi-3.5-mini-q4km` `defaultQuant`** (around line 312):

```ts
defaultQuant: "q4km",
availableQuants: ["q4km"],
```

(Current value is `"q4f16_1"` which mistags the actual Q4_K_M quantization — same fleet-wide tag drift bucket D corrected for `qwen3-8b-iq3m`.)

- [ ] **Step 3: Verify typecheck**: `make typecheck` clean.

- [ ] **Step 4: Commit**:

```
feat(embed): add q4km QuantFormat tier; fix phi-3.5-mini tag

Adds Q4_K_M to the QuantFormat union and corrects phi-3.5-mini-q4km
defaultQuant from the misleading "q4f16_1" to "q4km". Same fleet-
wide tag drift bucket D corrected for qwen3-8b-iq3m. Gate value for
the new tier will be calibrated empirically in the parity run.
```

---

## Task 2: Set `embeddingCapable: true` on phi-3.5-mini-q4km

**Files:** Modify: `eval/models.ts`.

- [ ] **Step 1:** Add `embeddingCapable: true,` to the `phi-3.5-mini-q4km` registration block (around line 320, near other functional flags).

- [ ] **Step 2:** `make typecheck` clean.

- [ ] **Step 3: Commit**:

```
feat(embed): enable bucket D on phi-3.5-mini-q4km

First fused-projection arch in bucket D — exercises forwardForEmbedding's
buildQKV / buildFFNGateUp fused-path branches end-to-end. Parity
validation follows in a subsequent commit.
```

---

## Task 3: Capture PyTorch reference fixtures

**Files:**
- Create: `eval/reports/bucket-d-phi3-probe-2026-04-30/inputs.json`
- Create: `eval/reports/bucket-d-phi3-probe-2026-04-30/capture-refs-requirements.txt`
- Create: `eval/reports/bucket-d-phi3-probe-2026-04-30/capture-refs.py`
- Create (script output): `eval/reports/bucket-d-phi3-probe-2026-04-30/phi-3.5-mini-ref.json`

- [ ] **Step 1: Create inputs.json** — reuse the same 10 sentences as bucket D for direct parity-harness comparability:

```bash
mkdir -p eval/reports/bucket-d-phi3-probe-2026-04-30
cp eval/reports/bucket-d-probe-2026-04-29/inputs.json \
   eval/reports/bucket-d-phi3-probe-2026-04-30/inputs.json
```

- [ ] **Step 2: Create requirements file** — same as bucket D (`torch`, `transformers`, `safetensors`, `accelerate`).

```bash
cp eval/reports/bucket-d-probe-2026-04-29/capture-refs-requirements.txt \
   eval/reports/bucket-d-phi3-probe-2026-04-30/capture-refs-requirements.txt
```

- [ ] **Step 3: Pre-warm HF cache via `hfdownloader`** (per the CLAUDE.md "HuggingFace downloads" policy):

```bash
hfdownloader download microsoft/Phi-3.5-mini-instruct
```

This populates `~/.cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/`. Script in Step 4 will load from this cache instantly.

- [ ] **Step 4: Write `capture-refs.py`** — adapted from bucket D's script with three changes:
  1. `MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"`
  2. Different `EOS_ID` (Phi-3 has `<|endoftext|>` = 32000 or `<|end|>` = 32007 depending on tokenizer config; the script reads `tokenizer.eos_token_id` automatically — log the value for verification).
  3. Output filename: `phi-3.5-mini-ref.json`.

The bucket D script structure is otherwise unchanged. Copy it as a template:

```bash
cp eval/reports/bucket-d-probe-2026-04-29/capture-refs.py \
   eval/reports/bucket-d-phi3-probe-2026-04-30/capture-refs.py
```

Then edit:
- Line containing `Bucket D Phase 0` → `Bucket D Phi-3.5 Phase 0`.
- `MODEL_NAME = "Qwen/Qwen3-8B"` → `MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"`.
- `out_path = HERE / "qwen3-8b-ref.json"` → `out_path = HERE / "phi-3.5-mini-ref.json"`.

The `# pyright: reportMissingImports=false` pragma at the top stays.

- [ ] **Step 5: Run the script**:

```bash
cd eval/reports/bucket-d-phi3-probe-2026-04-30 && \
uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
```

**Wall time** with warm cache: ~1-2 min (~7.6 GB load + 10 forwards). Phi-3.5-mini is smaller than Qwen3-8B; forward pass should be fast.

- [ ] **Step 6: Sanity check**:

```bash
python3 -c "
import json
d = json.load(open('eval/reports/bucket-d-phi3-probe-2026-04-30/phi-3.5-mini-ref.json'))
print('rows:', len(d['fixtures']))
print('dim:', len(d['fixtures'][0]['vec']))
print('eos_id:', d['eos_id'])
"
```

Expected: `rows: 10`, `dim: 3072` (Phi-3.5-mini hidden dim, NOT 4096), `eos_id`: whatever `tokenizer.eos_token_id` returns (verify against webllm's `tokenizer.eosId` for phi-3.5-mini-q4km in Task 4).

- [ ] **Step 7: Commit ref artifacts (force-add; eval/reports is gitignored)**:

```bash
git add -f eval/reports/bucket-d-phi3-probe-2026-04-30/
git commit -m "test(embed): pin Phi-3.5-mini parity refs for bucket D extension

10 fixtures captured against microsoft/Phi-3.5-mini-instruct
(torch.float16 cpu) with add_special_tokens=False + manual EOS append,
post-final-norm last-token pool + L2 normalize. Mirrors bucket D's
qwen3-8b ref capture exactly; only model name and output path differ.

Cache pre-warmed via hfdownloader per CLAUDE.md HuggingFace downloads
policy."
```

---

## Task 4: Run parity gate + 4-pair distinguishability

**Files:**
- Modify: `eval/causal-embedder-parity.ts` (add `q4km` gate tier; calibrate empirically)
- Output: `eval/reports/bucket-d-phi3-parity-2026-04-30/run.txt`

- [ ] **Step 1: Add `q4km` gate tier to the harness.** Find the gate selection block (introduced in commit `261e0b2`) and add a `q4km` branch. **First-attempt gate value: 0.97** (predicted between hybrid 0.995 and IQ3_M 0.90; Q4_K_M is more aggressive than hybrid but less than IQ3_M). Plan to calibrate empirically based on observed cosines.

```ts
const COSINE_GATE_Q4KM = 0.97;
// ...inside the gate-selection switch:
if (model.defaultQuant === "q4km") return COSINE_GATE_Q4KM;
```

- [ ] **Step 2: Verify webllm's tokenizer.eosId for phi-3.5-mini matches the PyTorch reference's `eos_id`.**

Run a one-off probe via the harness or a small TS script:

```ts
import { GgufParser } from "./src/models/gguf-parser.js";
import { ModelLoader } from "./src/models/model-loader.js";
import { Tokenizer } from "./src/inference/tokenizer.js";
import { readFileSync } from "node:fs";

const data = readFileSync("smoke-test/models/phi-3.5-mini-q4km.gguf");
const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
const parsed = ModelLoader.parseModel(view);
const tok = new Tokenizer(parsed.tokenizerConfig);
console.log("eosId:", tok.eosId);
```

Compare to the `eos_id` field in `phi-3.5-mini-ref.json`. **If they differ**, the parity run will fail at the EOS-append step. Fix by adjusting the ref-capture script (Task 3) to use the matching id, or by patching the chat-template / eos detection in webllm — diagnose case-by-case.

- [ ] **Step 3: Run the parity harness against `phi-3.5-mini-q4km`**:

```bash
mkdir -p eval/reports/bucket-d-phi3-parity-2026-04-30
bun run eval/causal-embedder-parity.ts phi-3.5-mini-q4km \
    eval/reports/bucket-d-phi3-probe-2026-04-30/phi-3.5-mini-ref.json \
    2>&1 | tee eval/reports/bucket-d-phi3-parity-2026-04-30/run.txt
```

Required: smoke server on port 8031 (`make smoke-serve` in another terminal). Use existing agentchrome session per CLAUDE.md.

Expected outcome:
- Gate selection logs `cos >= 0.97` (q4km tier).
- 10/10 fixtures PASS at the calibrated gate. Magnitudes 1.000 ± 1e-6.
- 4-pair distinguishability passes — every paraphrase cosine > every unrelated cosine.
- No console errors.

**If the predicted 0.97 gate fails**: this is the calibration question. Inspect the actual cosines:
- All 10 cosines clustered tightly above some value X → set the gate to X - 0.01 (small safety margin) and re-run.
- Cosines wildly variable (e.g., one fixture at 0.65, others at 0.97) → forward-graph bug, not quant noise. Diagnose using bucket D's bug ladder. Likely failure modes for the fused-arch first-exercise:
  - `assertContiguousF32` firing on `buildQKV` / `buildFFNGateUp` outputs (would log a clear error; check for `assertContiguousF32 failed` in console)
  - `lw.qkvFused` or `lw.gateUpFused` null when expected (would throw "split-QKV path requires qProj/kProj/vProj for phi3" or similar — phi3 should NOT throw because it's fused; if it does, the registration's arch field is off)
  - normBias path: Phi-3 has no `output_norm.bias`, so the conditional in `forwardForEmbedding` is a no-op. Verify by inspection.

**If it's a real forward-graph bug**: this is the kind of finding the cycle exists to surface. Document the diagnosis, fix in `model-inference.ts` or wherever the bug lives, re-run. Don't paper over.

- [ ] **Step 4: Update the gate value in the harness** if calibration moved it. Commit harness changes:

```
test(embed): add q4km parity gate tier (calibrated to <VALUE>)

Phi-3.5-mini Q4_K_M parity against f16 PyTorch reference at
cos >= <VALUE> (parallel to hyb 0.995 / iq3m 0.90 / default 0.999).
First fused-projection arch through forwardForEmbedding's
buildQKV / buildFFNGateUp branches.
```

(`run.txt` stays uncommitted — captured by the closure report in Task 6.)

---

## Task 5: embed-perf bench coverage

**Files:** Modify: `eval/embed-perf.ts`. Output: `eval/reports/embed-perf-phi-3.5-mini-2026-04-30/`.

- [ ] **Step 1:** Add `"phi-3.5-mini-q4km"` to the `EMBEDDER_MODELS` list (mirror the qwen3-8b-iq3m row added in commit `48c30b6`).

- [ ] **Step 2: Run the bench** with `--mode single --reps 3` (matching bucket D's reduced-reps decision; Phi-3.5 is smaller so a few more reps are also fine — try 5):

```bash
bun run eval/embed-perf.ts phi-3.5-mini-q4km --mode single --reps 5
```

Expected: short p50 ~300-500 ms, long p50 ~600-900 ms (Phi-3.5-mini is ~2x faster than Qwen3-8B per token; 32 layers vs 36 layers, and 3072 vs 4096 hidden dim).

- [ ] **Step 3:** Save bench output under `eval/reports/embed-perf-phi-3.5-mini-2026-04-30/`. Mirror bucket D's directory structure.

- [ ] **Step 4: Commit**:

```
feat(embed-perf): bench coverage for Phi-3.5-mini bucket D

Adds phi-3.5-mini-q4km row to the embed-perf bench. Second bucket D
row in the dashboard's Embeddings section; first fused-projection
arch (buildQKV + buildFFNGateUp fused branches exercised end-to-end
through forwardForEmbedding for the first time). Informational, not
gated.
```

---

## Task 6: Closure report

**Files:** Create: `eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md`.

- [ ] **Step 1:** Mirror bucket D's closure report (`eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`) with substitutions:
  - Title: "Bucket D extension — Phi-3.5-mini parity (2026-04-30)"
  - Outcome table: 10 rows × cosine + magnitude.
  - Gate selection: q4km tier at the calibrated value; cite the three-tier-now-four-tier scheme (`hyb` 0.995, `iq3m` 0.90, `q4km` <new>, default 0.999).
  - 4-pair distinguishability table.
  - Bench numbers from Task 5.
  - Bugs / discoveries (likely candidates: defaultQuant tag fix, EOS id verification, fused-path first-exercise findings).
  - Files touched + commit chronology for this 6-7-commit cycle.
  - Notes for follow-on: cross-arch coverage status (Llama 3.x and Mistral 7B remain queued).

- [ ] **Step 2: Force-add and commit** (`eval/reports/` is gitignored):

```bash
git add -f eval/reports/bucket-d-phi3-parity-2026-04-30/
git commit -m "docs(report): Phi-3.5-mini bucket D extension closure

10/10 PASS at cos >= <VALUE> against microsoft/Phi-3.5-mini-instruct
reference vectors (q4km-calibrated gate; new fourth tier in the gate-
by-quant-tier scheme alongside hyb 0.995, iq3m 0.90, default 0.999).
4-pair distinguishability sanity passes. First fused-projection arch
in bucket D — forwardForEmbedding's buildQKV / buildFFNGateUp fused
branches exercised end-to-end through to parity for the first time."
```

---

## Task 7: TODO update + archive

**Files:** Modify: `TODO.md` (append a one-line note under the bucket D closure stub at item 6); `TODO_ARCHIVE.md` (extend the bucket D archive entry with the Phi-3.5-mini cross-arch follow-up).

- [ ] **Step 1:** Append to TODO.md item 6:

```markdown
   **Cross-arch follow-up (2026-04-30):** Phi-3.5-mini-q4km shipped
   as the second bucket D model — first fused-projection arch through
   `forwardForEmbedding`. Closure report
   [`eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md`](eval/reports/bucket-d-phi3-parity-2026-04-30/SUMMARY.md);
   plan [`docs/superpowers/plans/2026-04-30-embedding-bucket-d-phi3.md`](docs/superpowers/plans/2026-04-30-embedding-bucket-d-phi3.md).
   Llama 3.x and Mistral 7B remain queued as separate cycles.
```

- [ ] **Step 2:** Append a sub-section to the bucket D archive entry in `TODO_ARCHIVE.md`:

```markdown
**Phi-3.5-mini cross-arch extension (2026-04-30):** Second bucket D
model. q4km gate tier added (cos >= <VALUE>). First fused-projection
arch through forwardForEmbedding's buildQKV / buildFFNGateUp fused
branches. Per-task commit list: <fill from git log>.
```

- [ ] **Step 3:** Verify `make checkall` clean.

- [ ] **Step 4: Commit**:

```
docs(TODO): note Phi-3.5-mini bucket D extension shipped

Second bucket D model; first fused-projection arch through
forwardForEmbedding. Updates the inline closure stub at TODO.md
item 6 and extends the bucket D archive entry in TODO_ARCHIVE.md.
Cross-arch follow-up (Llama 3.x, Mistral 7B) remains queued as
separate cycles.
```

---

## Self-review checklist (planner — before handoff)

**Spec coverage:** every step of bucket D's "Out of scope: Other chat archs" follow-up cycle is mapped to a task.

**Placeholder scan:** `<VALUE>` placeholders in commit messages and report text mark spots that resolve from Task 4's empirical calibration. The implementer fills them in at commit time. Not a TBD — they're parameter slots.

**Type consistency:** `q4km`, `phi-3.5-mini-q4km`, `forwardForEmbedding`, `buildQKV`/`buildFFNGateUp` used consistently across tasks.

**Frequent commits:** 7 commits across 7 tasks. Each independently revertable. Quant-format change (Task 1) is type-only and intentionally separate from the registration flag flip (Task 2).
