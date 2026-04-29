# Bucket C Phase 0 Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a two-stage probe characterizing Qwen3-Embedding-0.6B (causal-LM-derived embedder) for bucket C support, producing a metadata report (Stage 1) and reference vectors (Stage 2). No production code lands.

**Architecture:** Stage 1 reuses bucket B's `probe-gguf.ts` pattern (download GGUF + parse via `GgufParser` + dump metadata) plus a manual read-only walk of `ModelInference.forward()` for tap-point analysis. Stage 2 adapts bucket B's `capture-refs.py` to two-mode reference capture (document + instruction-prefixed query) via sentence-transformers. User checkpoint between stages.

**Tech Stack:** Bun (probe-gguf), `GgufParser` (`src/models/gguf-parser.ts`), Python via `uv run --no-project` (capture-refs), sentence-transformers (reference vectors).

**Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md` (committed `5ea108b`).

**Output directory:** `eval/reports/bucket-c-probe-2026-04-29/` (created in Task 1).

---

## Stage 1 — Metadata + embed-surface analysis

### Task 1: Scaffold probe directory and embed-surface analysis

**Files:**
- Create: `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md` (initial scaffold; tap-point section populated, metadata table left as a stub completed in Task 3)
- Read: `src/inference/model-inference.ts`
- Read: `src/inference/encoder-inference.ts:520-560` (poolAndNormalize for context)

- [ ] **Step 1: Create probe directory**

```bash
mkdir -p eval/reports/bucket-c-probe-2026-04-29
```

- [ ] **Step 2: Walk `ModelInference.forward()` to identify tap-points**

Open `src/inference/model-inference.ts` and locate the forward graph builder (look for the function that emits the residual-stream loop ending in the `output_norm` + `lm_head` matmul). Identify candidate hidden-state tap-points. Expected candidates (verify against actual code):

  1. **Pre-output-norm (last layer residual stream output).** Tensor shape `[E, N]`, no final RMSNorm applied. Typical for raw hidden-state extraction; closest to what `encoder-inference.ts` produces for BERT-family.
  2. **Post-output-norm.** Tensor shape `[E, N]`, after final RMSNorm. Common Qwen embedding convention; reference implementations frequently apply final norm before pooling.
  3. **Skip output-norm AND skip `lm_head`.** Same as candidate 1 but framed as "build forward graph but stop before the final norm + projection".

Cross-reference `~/Repos/llama.cpp/src/models/qwen3.cpp` (or nearest `qwen*.cpp` if `qwen3.cpp` is absent — find via `ls ~/Repos/llama.cpp/src/models/qwen*.cpp`). Look for the section where `LLM_ARCH_QWEN3` builds its causal output and identify which residual-stream point is conventionally pooled for embedding variants. **The architecture-truth-source is decisive here** — don't guess from naming.

- [ ] **Step 3: Write the embed-surface analysis section of STAGE-1-METADATA.md**

Create `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md` with this initial content (fill in real line numbers + recommendation from your Step 2 walk; do not copy this verbatim if the actual code differs):

```markdown
# Bucket C Phase 0 — Stage 1 metadata report

**Date:** 2026-04-29
**Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md` (`5ea108b`)
**Plan:** `docs/superpowers/plans/2026-04-29-embedding-bucket-c-phase-0-probe.md`

## Embed-surface analysis

Walked `src/inference/model-inference.ts` `forward()` graph builder
(approx. lines TBD-TBD). Cross-referenced
`~/Repos/llama.cpp/src/models/<qwen3-or-equivalent>.cpp` per the
`llama-cpp-as-arch-truth-source.md` doctrine.

| # | Tap-point | Insertion graph node (`model-inference.ts` line) | Tensor shape | Output-norm applied | Notes |
|---|---|---|---|---|---|
| 1 | Pre-output-norm | TBD | `[E, N]` | no | Raw last-layer residual; matches encoder-inference.ts convention |
| 2 | Post-output-norm | TBD | `[E, N]` | yes | Common Qwen embed convention; aligned with reference implementations |
| 3 | Skip norm + lm_head | TBD | `[E, N]` | no | Same as #1 but framed as graph-build halt |

**Recommendation:** Tap-point #TBD. Reasoning: TBD.

**Architecture truth source consulted:** `~/Repos/llama.cpp/src/models/<file>.cpp:<lines>`.

## Qwen3-Embedding-0.6B metadata

(Populated in Task 3; placeholder until probe-gguf runs.)

## Stage 2 plan refinement

(Populated in Task 4; placeholder until metadata is read.)
```

- [ ] **Step 4: Verify file exists and reads correctly**

```bash
cat eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md | head -20
```

Expected: The header lines render and tap-point table is present.

- [ ] **Step 5: Commit**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md
git commit -m "docs(probe): bucket-c phase-0 stage 1 — embed-surface analysis scaffold"
```

---

### Task 2: Adapt probe-gguf.ts for Qwen3-Embedding-0.6B

**Files:**
- Create: `eval/reports/bucket-c-probe-2026-04-29/probe-gguf.ts` (copy + edit from bucket B)

- [ ] **Step 1: Confirm GGUF mirror availability**

Inspect Hugging Face for a Qwen3-Embedding-0.6B GGUF mirror with f16 or Q8_0 weights. Preferred candidates (try in order):

```bash
# Check each URL via curl -sI for a 200 status. Use the first that returns 200.
curl -sI "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/qwen3-embedding-0.6b-f16.gguf" | head -1
curl -sI "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-f16.gguf" | head -1
```

If neither returns `HTTP/2 200`, search via:

```bash
# Open https://huggingface.co/models?search=qwen3-embedding-0.6b-gguf manually,
# or use HF API:
curl -s "https://huggingface.co/api/models?search=Qwen3-Embedding-0.6B&filter=gguf" | head -50
```

**If no f16 or Q8_0 mirror is available**, halt and surface the abort condition per spec ("No GGUF mirror with f16/Q8_0 weights exists"). Document the alternatives that were checked, then stop and ask the user before proceeding to Stage 1's later steps.

- [ ] **Step 2: Create `probe-gguf.ts` adapted for Qwen3-Embedding**

```typescript
#!/usr/bin/env bun
/**
 * Bucket C Phase 0 probe — download and inspect Qwen3-Embedding-0.6B GGUF.
 * Writes 00-gguf-discovery.txt summarizing metadata keys + tensor list.
 * Run: bun eval/reports/bucket-c-probe-2026-04-29/probe-gguf.ts
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { GgufParser } from "../../../src/models/gguf-parser.js";

const OUT_DIR = "eval/reports/bucket-c-probe-2026-04-29";
const CACHE_DIR = join(OUT_DIR, "cache");
mkdirSync(CACHE_DIR, { recursive: true });

interface Candidate {
	label: string;
	url: string;
}

// Replace url with the verified mirror from Task 2 Step 1.
const candidates: Candidate[] = [
	{
		label: "qwen3-embedding-0.6b",
		url: "TBD-replace-with-verified-mirror-from-step-1",
	},
];

const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	console.log(s);
};

for (const c of candidates) {
	log(`\n=== ${c.label} ===`);
	log(`URL: ${c.url}`);
	const localPath = join(CACHE_DIR, `${c.label}.gguf`);
	if (!existsSync(localPath)) {
		log(`Downloading…`);
		const res = await fetch(c.url);
		if (!res.ok) {
			log(`FAILED: HTTP ${res.status} ${res.statusText}`);
			continue;
		}
		const buf = await res.arrayBuffer();
		writeFileSync(localPath, new Uint8Array(buf));
		log(`Saved ${buf.byteLength} bytes to ${localPath}`);
	} else {
		log(`Reusing cached ${localPath}`);
	}

	const bytes = new Uint8Array(await Bun.file(localPath).arrayBuffer());
	const ctx = GgufParser.parse(bytes);

	log(
		`general.architecture = ${JSON.stringify(ctx.metadata.get("general.architecture")?.value)}`,
	);

	// Filter to embedding-relevant keys + qwen-arch + tokenizer.
	const archKeys: string[] = [];
	for (const k of ctx.metadata.keys()) {
		if (
			k.startsWith("qwen") ||
			k.startsWith("general.") ||
			k.startsWith("tokenizer.") ||
			k.includes("pool") ||
			k.includes("embed") ||
			k.includes("norm")
		) {
			archKeys.push(k);
		}
	}
	archKeys.sort();
	log(`Metadata keys (${archKeys.length}):`);
	for (const k of archKeys) {
		const v = ctx.metadata.get(k)?.value;
		const repr =
			typeof v === "string" || typeof v === "number" || typeof v === "boolean"
				? String(v)
				: `<${typeof v}>`;
		log(`  ${k} = ${repr}`);
	}

	const tensorNames = ctx.tensors.map((t) => t.name).sort();
	log(`Tensors (${tensorNames.length}):`);
	for (const n of tensorNames.slice(0, 80)) log(`  ${n}`);
	if (tensorNames.length > 80) log(`  … (${tensorNames.length - 80} more)`);

	// Highlight projection-head + pooling-related tensors.
	const projTensors = tensorNames.filter(
		(n) =>
			n.includes("proj") ||
			n.includes("pool") ||
			n.includes("output_norm") ||
			n.includes("output.weight"),
	);
	log(`Projection / pooling / output tensors (${projTensors.length}):`);
	for (const n of projTensors) log(`  ${n}`);
}

writeFileSync(join(OUT_DIR, "00-gguf-discovery.txt"), `${lines.join("\n")}\n`);
log(`\nWrote ${OUT_DIR}/00-gguf-discovery.txt`);
```

Replace the `TBD-replace-with-verified-mirror-from-step-1` placeholder with the URL verified in Step 1 before proceeding.

- [ ] **Step 3: Verify TypeScript compiles in isolation**

```bash
bun run --bun --print "console.log('ts ok')" eval/reports/bucket-c-probe-2026-04-29/probe-gguf.ts 2>&1 | head -5
```

(This won't actually execute the script — it just verifies the import resolution. The actual run is Task 3.) If it errors on `GgufParser` import, verify the relative path `../../../src/models/gguf-parser.js` matches `src/models/gguf-parser.ts`'s emitted location.

- [ ] **Step 4: Commit**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/probe-gguf.ts
git commit -m "docs(probe): bucket-c phase-0 stage 1 — probe-gguf script"
```

---

### Task 3: Run the GGUF probe and assemble the metadata table

**Files:**
- Create (run-output): `eval/reports/bucket-c-probe-2026-04-29/00-gguf-discovery.txt` (committed; binary GGUF in `cache/` is .gitignored implicitly via repo `.gitignore` — check before commit)
- Modify: `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md` (fill metadata table)
- Read: HF config for `Qwen/Qwen3-Embedding-0.6B` for the "intended" column

- [ ] **Step 1: Run the probe**

```bash
bun eval/reports/bucket-c-probe-2026-04-29/probe-gguf.ts
```

Expected: Script downloads the GGUF (~300-1000 MB depending on quant), parses it, and writes `00-gguf-discovery.txt`. If the download fails, surface the HTTP error and re-verify the mirror URL in Task 2 Step 1.

- [ ] **Step 2: Inspect the discovery output**

```bash
cat eval/reports/bucket-c-probe-2026-04-29/00-gguf-discovery.txt
```

Confirm the output contains: `general.architecture`, the qwen-arch metadata keys, tokenizer keys, tensor names. If `general.architecture` is absent or unexpected (not `qwen3` / `qwen2` / similar), halt per spec abort condition ("HF config reveals an architecture genuinely outside the qwen family").

- [ ] **Step 3: Read HF config for the "intended" column**

```bash
curl -s "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json" | head -100
curl -s "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/tokenizer_config.json" | head -50
curl -s "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/README.md" | head -200
```

Pull out: pooling type (last-token typical for Qwen3-Embedding), normalization (L2), projection-head presence + dim, RoPE config (mode/base/dim), `eos_token_id`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, FFN type (SwiGLU typical for Qwen), instruction-prefix conventions documented for query/document modes.

- [ ] **Step 4: Update STAGE-1-METADATA.md with the metadata table**

Edit `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md` to replace the "Qwen3-Embedding-0.6B metadata" placeholder with this populated table (fill in real values from Steps 2 + 3; mark divergences in **bold** with a 🚩 emoji prefix):

```markdown
## Qwen3-Embedding-0.6B metadata

| Property | Intended (HF config) | As-shipped (GGUF) | Divergence |
|---|---|---|---|
| Architecture | TBD | TBD | |
| Pooling type | TBD (last-token expected) | TBD | |
| Normalization | TBD (L2 expected) | TBD | |
| Projection head present | TBD | TBD | |
| Projection-head output dim | TBD | TBD | |
| `hidden_size` | TBD | TBD | |
| `num_hidden_layers` | TBD | TBD | |
| `num_attention_heads` | TBD | TBD | |
| `num_key_value_heads` | TBD | TBD | |
| RoPE mode | TBD (NEOX expected) | TBD | |
| RoPE freq_base | TBD | TBD | |
| RoPE freq_dim | TBD | TBD | |
| FFN type | TBD (SwiGLU expected) | TBD | |
| `eos_token_id` | TBD | TBD | |
| `bos_token_id` | TBD | TBD | |
| Vocab size | TBD | TBD | |
| Instruction-prefix (query) | TBD (exact string from README) | n/a | |
| Instruction-prefix (document) | TBD (or "raw") | n/a | |

**Divergences flagged:** TBD (or "none").

**GGUF mirror used:** `TBD-url`.

**Raw discovery:** `00-gguf-discovery.txt` (committed in this directory).
```

Replace every `TBD` with the real value or `n/a`. **Do not commit the table with TBDs.**

- [ ] **Step 5: Verify `cache/` is gitignored**

```bash
git check-ignore eval/reports/bucket-c-probe-2026-04-29/cache/qwen3-embedding-0.6b.gguf
```

Expected: the path prints (= ignored). If it returns nothing (= not ignored), the GGUF would be staged on commit. Handle by:

```bash
echo "eval/reports/bucket-c-probe-2026-04-29/cache/" >> .gitignore
git add .gitignore
```

(Verify `.gitignore` doesn't already cover this via a broader rule before adding.)

- [ ] **Step 6: Commit Stage 1 deliverables**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md \
           eval/reports/bucket-c-probe-2026-04-29/00-gguf-discovery.txt
git commit -m "docs(probe): bucket-c phase-0 stage 1 — metadata"
```

---

### Task 4: Stage 1 finalization — Stage 2 plan refinement

**Files:**
- Modify: `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md` (fill Stage 2 plan refinement section)

- [ ] **Step 1: Decide fixture-mode count from metadata**

Default per spec: 2 modes (document + query). Inspect the README content captured in Task 3 Step 3. If Qwen3-Embedding documents more pooling modes than expected (e.g., 3+ task-specific instruction templates), expand to N modes for the capture step. If fewer (e.g., the model accepts only one form), reduce to 1 mode and document why.

- [ ] **Step 2: Pin the exact instruction-prefix string for query mode**

From the Qwen3-Embedding README, copy the exact prefix string used in the canonical `model.encode` example. Common form (verify against actual README — do not guess):

> `Instruct: <task description>\nQuery: <text>`

Where `<task description>` is the task-specific instruction (often `Given a web search query, retrieve relevant passages that answer the query`). Record the exact string verbatim, including whitespace.

- [ ] **Step 3: Update STAGE-1-METADATA.md Stage 2 plan refinement section**

Replace the placeholder with:

```markdown
## Stage 2 plan refinement

**Fixture-mode count:** 2 (document + query). Default from spec retained.
[OR document expansion/reduction with reasoning if metadata revealed more/fewer modes.]

**Exact instruction-prefix string (query mode):**

```
Instruct: <exact task description from README>\nQuery: 
```

(With the trailing colon-space; the fixture text appends directly after.)

**Document mode:** raw fixture text passed unmodified to `model.encode`.

**Expected output dim:** TBD (from metadata table — projection-head output dim if present, else `hidden_size`).

**Stage 2 ready-to-execute:** yes / no (with reasoning).
```

- [ ] **Step 4: Verify Stage 1 exit criteria**

Confirm against the spec's Stage-1 exit criteria:
- [x] Report committed (in Task 3 Step 6).
- [ ] Metadata table populated for all rows (verify no TBDs remain).
- [ ] Embed-surface analysis lists ≥2 tap-point options with a recommendation (verify Task 1 Step 3 was filled in Task 1 — if not, fill it now using the metadata findings).
- [ ] No open question blocks Stage 2.

If any criterion fails, address before committing.

- [ ] **Step 5: Commit Stage 1 finalization**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md
git commit -m "docs(probe): bucket-c phase-0 stage 1 — final (stage 2 ready)"
```

- [ ] **Step 6: Stage 1 user checkpoint**

Stop here. Surface the Stage 1 report to the user for review. Do not proceed to Stage 2 until the user explicitly approves Stage 1 findings and gives go-ahead.

Suggested message:

> Stage 1 closed. Metadata report at `eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md`. Key findings: [tap-point recommendation], [pooling type], [projection head presence], [any divergences flagged]. Ready to proceed to Stage 2 (reference-vector capture)?

---

## Stage 2 — Reference-vector capture

### Task 5: Set up capture-refs.py and inputs.json

**Files:**
- Create: `eval/reports/bucket-c-probe-2026-04-29/inputs.json` (copied from bucket B verbatim)
- Create: `eval/reports/bucket-c-probe-2026-04-29/capture-refs.py`
- Create: `eval/reports/bucket-c-probe-2026-04-29/capture-refs-requirements.txt`

- [ ] **Step 1: Copy the 5-fixture inputs.json verbatim from bucket B**

```bash
cp eval/reports/encoder-parity-2026-04-28/inputs.json \
   eval/reports/bucket-c-probe-2026-04-29/inputs.json
```

Verify content:

```bash
cat eval/reports/bucket-c-probe-2026-04-29/inputs.json
```

Expected: `["Hello world.", "The quick brown fox…", "Embedding models map…", "Café — naïve façade…", "."]` (the 5 bucket-B fixtures).

- [ ] **Step 2: Create capture-refs-requirements.txt**

```bash
cp eval/reports/encoder-parity-2026-04-28/capture-refs-requirements.txt \
   eval/reports/bucket-c-probe-2026-04-29/capture-refs-requirements.txt
```

Verify the file contains `sentence-transformers` (and any other deps bucket B used). If Qwen3-Embedding requires additional packages (e.g., `transformers>=4.51` for Qwen3 support), append them to this file.

- [ ] **Step 3: Create capture-refs.py adapted for two-mode capture**

```python
"""
Bucket C Phase 0 — reference-embedding capture for Qwen3-Embedding-0.6B.
Two-mode: document (raw) + query (instruction-prefixed).
Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
Writes: qwen3-embedding-0.6b-ref.json — {model, captured_with, pooling, instruction_prefix, fixtures}.
"""
import json
import sys
from pathlib import Path

import sentence_transformers
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
POOLING = "TBD-from-stage-1"  # e.g., "last-token"

# Exact prefix string pinned from Stage 1 README read.
# Verify this matches the Stage-1 metadata report's "Stage 2 plan refinement"
# section before running.
INSTRUCTION_PREFIX = (
    "Instruct: TBD-task-description-from-readme\n"
    "Query: "
)

print(f"Loading {MODEL_NAME}…", file=sys.stderr)
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

# Document mode: raw fixtures.
print("Encoding document mode…", file=sys.stderr)
doc_vecs = model.encode(inputs, normalize_embeddings=True)

# Query mode: instruction-prefixed fixtures.
print("Encoding query mode…", file=sys.stderr)
query_inputs = [INSTRUCTION_PREFIX + s for s in inputs]
query_vecs = model.encode(query_inputs, normalize_embeddings=True)

# Magnitude + dim sanity check.
EXPECTED_DIM = doc_vecs.shape[1]  # Locked from first document mode encoding.
print(f"Expected output dim (from doc-mode shape): {EXPECTED_DIM}", file=sys.stderr)

import math
TOLERANCE = 1e-3
def _check(label, vec, idx):
    mag = math.sqrt(sum(float(x) * float(x) for x in vec.tolist()))
    if abs(mag - 1.0) > TOLERANCE:
        raise SystemExit(
            f"Magnitude assertion failed for {label} row {idx}: |v|_2 = {mag:.6f} (expected 1.0 ± {TOLERANCE})"
        )
    if len(vec) != EXPECTED_DIM:
        raise SystemExit(
            f"Dim assertion failed for {label} row {idx}: dim = {len(vec)} (expected {EXPECTED_DIM})"
        )

for i, v in enumerate(doc_vecs):
    _check("document", v, i)
for i, v in enumerate(query_vecs):
    _check("query", v, i)
print(f"All {len(doc_vecs) + len(query_vecs)} vectors passed magnitude + dim assertions.", file=sys.stderr)

# Build interleaved-by-row output.
fixtures = []
for i, raw in enumerate(inputs):
    fixtures.append({
        "row": i,
        "input": raw,
        "mode": "document",
        "vec": [float(x) for x in doc_vecs[i].tolist()],
    })
    fixtures.append({
        "row": i,
        "input": raw,
        "mode": "query",
        "vec": [float(x) for x in query_vecs[i].tolist()],
    })

out = {
    "model": MODEL_NAME,
    "captured_with": f"sentence-transformers {sentence_transformers.__version__}",
    "pooling": POOLING,
    "instruction_prefix": INSTRUCTION_PREFIX,
    "fixtures": fixtures,
}

out_path = HERE / "qwen3-embedding-0.6b-ref.json"
out_path.write_text(json.dumps(out))
print(
    f"Wrote {out_path} ({len(fixtures)} vectors, dim={EXPECTED_DIM})",
    file=sys.stderr,
)
```

**Before saving:** replace `TBD-from-stage-1` (POOLING) and `TBD-task-description-from-readme` (INSTRUCTION_PREFIX) with the values pinned in `STAGE-1-METADATA.md`'s Stage 2 plan refinement section. Do not commit with TBDs.

- [ ] **Step 4: Commit the harness**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/inputs.json \
           eval/reports/bucket-c-probe-2026-04-29/capture-refs.py \
           eval/reports/bucket-c-probe-2026-04-29/capture-refs-requirements.txt
git commit -m "docs(probe): bucket-c phase-0 stage 2 — capture-refs harness"
```

---

### Task 6: Run the capture and verify magnitudes

**Files:**
- Create (run-output): `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`

- [ ] **Step 1: Run capture-refs.py**

```bash
cd eval/reports/bucket-c-probe-2026-04-29
uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
cd ../../..
```

Expected stderr:
```
Loading Qwen/Qwen3-Embedding-0.6B…
Encoding document mode…
Encoding query mode…
Expected output dim (from doc-mode shape): <integer matching Stage 1 expected_output_dim>
All 10 vectors passed magnitude + dim assertions.
Wrote /…/qwen3-embedding-0.6b-ref.json (10 vectors, dim=<integer>)
```

If the magnitude or dim assertion fails: **halt**. Diagnose via the Stage-2 abort conditions (see spec). Common causes:
- HF model fetch returned a stale/incomplete weight set → re-run.
- sentence-transformers default for Qwen3-Embedding doesn't normalize → check the README; the `normalize_embeddings=True` flag may not be enough if the model wraps a custom forward path. Fall back to manual L2-normalization in the script and document.

- [ ] **Step 2: Verify the JSON shape**

```bash
python3 -c "
import json
d = json.load(open('eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json'))
print('model:', d['model'])
print('pooling:', d['pooling'])
print('fixtures:', len(d['fixtures']))
print('modes:', sorted(set(f['mode'] for f in d['fixtures'])))
print('dim:', len(d['fixtures'][0]['vec']))
print('first vec mag:', sum(x*x for x in d['fixtures'][0]['vec']) ** 0.5)
"
```

Expected:
```
model: Qwen/Qwen3-Embedding-0.6B
pooling: <last-token or whatever was pinned>
fixtures: 10
modes: ['document', 'query']
dim: <expected dim>
first vec mag: 1.0  (within 1e-3)
```

- [ ] **Step 3: Commit the reference vectors**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json
git commit -m "docs(probe): bucket-c phase-0 stage 2 — reference vectors captured"
```

---

### Task 7: Stage 2 final report and probe-conclusion

**Files:**
- Create: `eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md`

- [ ] **Step 1: Compute pairwise cosine table (optional informational sub-step from spec)**

```bash
python3 -c "
import json, math
d = json.load(open('eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json'))
fx = d['fixtures']
def cos(a, b):
    return sum(x*y for x,y in zip(a,b))
print('Doc-vs-doc 5x5:')
docs = [f for f in fx if f['mode']=='document']
for i,a in enumerate(docs):
    print('  ' + ' '.join(f'{cos(a[\"vec\"], b[\"vec\"]):.3f}' for b in docs))
print('Query-vs-query 5x5:')
qs = [f for f in fx if f['mode']=='query']
for i,a in enumerate(qs):
    print('  ' + ' '.join(f'{cos(a[\"vec\"], b[\"vec\"]):.3f}' for b in qs))
print('Doc-vs-query (same row):')
for i in range(5):
    a = docs[i]['vec']; b = qs[i]['vec']
    print(f'  row {i}: {cos(a,b):.3f}')
"
```

Save the output for inclusion in the report.

- [ ] **Step 2: Write STAGE-2-REFERENCE-VECTORS.md**

```markdown
# Bucket C Phase 0 — Stage 2 reference vectors

**Date:** 2026-04-29
**Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md` (`5ea108b`)
**Plan:** `docs/superpowers/plans/2026-04-29-embedding-bucket-c-phase-0-probe.md`
**Stage 1:** `STAGE-1-METADATA.md` (this directory)

## Inputs

5 fixed strings in `inputs.json` (verbatim copy from
`eval/reports/encoder-parity-2026-04-28/inputs.json`).

## Capture configuration

- **Model:** `Qwen/Qwen3-Embedding-0.6B`
- **sentence-transformers version:** `<from JSON metadata>`
- **Pooling:** `<from Stage 1>`
- **Instruction prefix (query mode):** `<exact string>`
- **Document mode:** raw fixture passed unmodified.
- **Output dim:** `<expected_output_dim>` (matches Stage 1 metadata table).

## Document mode (5 vectors)

| Row | Input (truncated) | Magnitude | First 3 dims |
|----:|-------------------|----------:|--------------|
|   0 | Hello world. | 1.000000 | [TBD, TBD, TBD] |
|   1 | The quick brown fox jumps … | 1.000000 | [TBD, TBD, TBD] |
|   2 | Embedding models map text … | 1.000000 | [TBD, TBD, TBD] |
|   3 | Café — naïve façade résumé … | 1.000000 | [TBD, TBD, TBD] |
|   4 | . | 1.000000 | [TBD, TBD, TBD] |

## Query mode (5 vectors)

| Row | Input (truncated, prefix elided) | Magnitude | First 3 dims |
|----:|----------------------------------|----------:|--------------|
|   0 | Hello world. | 1.000000 | [TBD, TBD, TBD] |
|   1 | The quick brown fox jumps … | 1.000000 | [TBD, TBD, TBD] |
|   2 | Embedding models map text … | 1.000000 | [TBD, TBD, TBD] |
|   3 | Café — naïve façade résumé … | 1.000000 | [TBD, TBD, TBD] |
|   4 | . | 1.000000 | [TBD, TBD, TBD] |

## Pairwise cosine (informational)

(From Step 1's python output. Doc-vs-query same-row should be high
but <1.0; mode prefix shifts the embedding subspace. If same-row
doc-vs-query is very low, that suggests the prefix is dominating
the representation — flag for Phase 3 to validate.)

```
Doc-vs-doc 5x5:
  TBD
Query-vs-query 5x5:
  TBD
Doc-vs-query (same row):
  TBD
```

## Probe conclusion

**Recommendation:** [proceed to Phase 1 / pause for additional probing / surface a scope concern]

**Risks resolved:**
1. ✅ GGUF mirror keys present (Stage 1 metadata diff).
2. ✅ Pooling type confirmed: <from Stage 1>.
3. ✅ Instruction-prefix convention pinned: see capture configuration above.
4. ✅ Hidden-state tap-point recommendation landed: see Stage 1 embed-surface analysis.
5. ✅ Reference vectors L2-normalized; magnitude assertion passed.
6. ✅ Projection-head presence + dim characterized: <from Stage 1>.

**Risks deferred (per spec):**
- a. WebGPU graph-build cost in embed mode → Phase 3 surfaces, Phase 4 measures.
- b. Causal-mask semantics under last-token pooling → Phase 2 validation against parity gate.
- c. 4B/8B variant feasibility → Phase 5.

**Open questions for Phase 1 plan:**
- TBD (anything that came up during execution and should be addressed in the Phase 1 spec).

**Phase 1 entry posture:** [ready / needs decision].
```

Replace every TBD with the actual values from the JSON + Stage 1 + your pairwise cosine output.

- [ ] **Step 3: Verify Stage 2 exit criteria**

- [x] 10 reference vectors committed (in Task 6 Step 3).
- [x] Reference vectors verified at unit magnitude (in Task 6 Step 1).
- [ ] Report committed (after this task's Step 4).
- [ ] Probe-conclusion section recommends a clear Phase 1 entry posture.

- [ ] **Step 4: Commit Stage 2 final report**

```bash
git add -f eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md
git commit -m "docs(probe): bucket-c phase-0 stage 2 — final report + probe conclusion"
```

---

## Stage 3 — Housekeeping

### Task 8: Close TODO.md item 5 with closure stub

**Files:**
- Modify: `TODO.md` (item 5 "Embedding bucket C" block)

- [ ] **Step 1: Read the current TODO item 5 block**

```bash
grep -n "Embedding bucket C\|next-session focus" TODO.md
```

Locate the lines spanning the item-5 block (currently around lines 934-989; verify with `wc -l TODO.md` and Read tool against the actual range before editing).

- [ ] **Step 2: Replace the block with a closure stub**

The closure stub follows the project's TODO archival convention: 4-8 lines linking the canonical closure report and noting whether full archival is appropriate. Since this is a Phase 0 probe (not a full feature closure), the right move is to **update the block in place** to reflect Phase 0 closure and queue Phase 1 — not full archival, because Phase 1-5 sub-items remain open underneath.

Replace the block content with:

```markdown
5. **Embedding bucket C — causal-LM-derived embedders.** Phase 0
   probe **CLOSED YYYY-MM-DD** — see
   [`eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md`](eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md).
   Probe artifacts: Stage 1 metadata + embed-surface analysis;
   Stage 2 reference vectors (5 fixtures × 2 modes = 10 refs at
   cosine ≥unit magnitude). Probe conclusion:
   [proceed to Phase 1 / paused on <concern>].

   **Phase 1-5 plan:** [queued / deferred pending <decision>].
   See spec
   [`docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md`](docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md)
   for the original phase plan template; Phase 1+ specs to be
   written when Phase 1 is queued for execution.
```

Replace `YYYY-MM-DD` with the actual closure date and bracketed placeholders with the actual probe-conclusion outcome.

- [ ] **Step 3: Commit TODO.md update**

```bash
git add TODO.md
git commit -m "docs(TODO): close bucket-c phase-0 probe"
```

- [ ] **Step 4: Optional — vault-save a pattern note**

If Stage 1 surfaced a generalizable lesson worth saving for future bucket-C-shaped probes (e.g., "causal-LM-derived embedder probe must dump GGUF metadata before fixture capture"), spawn the `research-agent` to save the note to `~/ClaudeVault/Patterns/` and rebuild the index. Skip this step if no novel lesson surfaced — most of the doctrine is already covered by `encoder-architecture-probe-saved-spec-rewrite.md` and `encoder-parity-gate-via-sentence-transformers.md`.

If you do save: rebuild the index after:

```bash
uv run --no-project ~/.claude/skills/parsidion/scripts/update_index.py
```

---

## Self-Review Checklist

After plan execution completes, verify:
- [ ] All 6 risks from the spec's risk register are addressed in the Stage 2 probe-conclusion section.
- [ ] Both Stage 1 and Stage 2 exit criteria from the spec are met.
- [ ] All commits are separate (1 per logical unit; no bundling).
- [ ] No production code (`src/`, `eval/*.ts` outside the new probe directory) was modified.
- [ ] The probe-conclusion section gives a clear Phase 1 entry posture (no ambiguity for the next session).

## References

- **Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md` (`5ea108b`).
- **Bucket B template (capture-refs):** `eval/reports/encoder-parity-2026-04-28/`.
- **Bucket B template (probe-gguf):** `eval/reports/encoder-parity-2026-04-28/probe-gguf.ts`.
- **Vault doctrines applied:**
  - `~/ClaudeVault/Patterns/encoder-architecture-probe-saved-spec-rewrite.md`
  - `~/ClaudeVault/Patterns/encoder-parity-gate-via-sentence-transformers.md`
  - `~/ClaudeVault/Patterns/llama-cpp-as-arch-truth-source.md`
  - `~/ClaudeVault/Patterns/cap-probe-bump-first-doctrine.md`
  - `~/ClaudeVault/Knowledge/encoder-cosine-degradation-signatures.md` (Phase 3 diagnostic ladder if parity fails downstream).
