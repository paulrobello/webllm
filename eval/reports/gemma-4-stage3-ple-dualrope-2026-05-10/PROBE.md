# Stage 3 pre-implementation probe — PLE table sizing

## Tensor list (per_layer / embd hits)

```
=== per_layer / embd tensors ===
per_layer_model_proj.weight: shape=[1536, 8960] type=BF16 size_bytes=27525120
per_layer_proj_norm.weight:  shape=[256]        type=F32  size_bytes=1024
per_layer_token_embd.weight: shape=[8960, 262144] type=Q5_K size_bytes=1614807040
token_embd.weight:           shape=[1536, 262144] type=Q4_K size_bytes=226492416
```

## Metadata

Raw field dump:

```
gemma4.block_count:                        [35]
gemma4.embedding_length:                   [1536]
gemma4.embedding_length_per_layer_input:   [256]
```

- `gemma4.embedding_length`: 1536 (main hidden dim)
- `gemma4.embedding_length_per_layer_input`: 256 (PLE dim per layer — the "short" residual dimension injected at each block input)
- `gemma4.block_count`: 35 layers
- Vocab size (`tokenizer.ggml.tokens` length): 262144 (verified in Phase 1)

Note: the PLE tensor's first dimension (8960) equals `256 × 35` — the table is
stored monolithically as `[embd_per_layer × block_count, vocab_size]` = `[8960, 262144]`.
One per-layer slice is a `[256, 262144]` sub-matrix.

## Per-layer slice analysis

Tensor name: `per_layer_token_embd.weight`
Stored dtype: `Q5_K`
Full tensor shape: `[8960, 262144]` (rows = 256 dim × 35 layers, cols = 262144 vocab)
Full tensor size in bytes: 1,614,807,040 bytes (1,540.00 MiB = 1.5039 GiB)

### Per-layer slice (monolithic tensor split across layers)

```
Per-layer slice shape:     [256, 262144]   (256 rows × 262144 cols)
Per-layer slice bytes:     1,614,807,040 / 35 = 46,137,344 bytes
Per-layer slice MiB:       44.00 MiB
```

### Binding-cap assessment

```
Per-binding WebGPU cap:    128 MiB = 134,217,728 bytes
Per-layer slice size:       46,137,344 bytes (44.00 MiB)
Single per-layer slice exceeds cap?  NO  (fits with 84 MiB headroom)
```

> **The PLE slice fits comfortably.** Stage 3 can bind one per-layer slice per
> forward pass with no chunking. The existing `token_embd.weight` path (Q4_K,
> 216 MiB) already exceeds the cap and requires chunked dispatch; PLE does not
> inherit that requirement.

### Companion tensors (also per-layer, binding-safe)

| Tensor | Shape | Dtype | Bytes | MiB | Exceeds 128 MiB? |
|--------|-------|-------|-------|-----|------------------|
| `per_layer_model_proj.weight` | [1536, 8960] | BF16 | 27,525,120 | 26.25 | NO |
| `per_layer_proj_norm.weight`  | [256]        | F32  | 1,024      | ~0   | NO |

Both companion tensors fit without chunking.

## Decision recommendation

- **Single slice fits 128 MiB → no chunking needed for PLE.**
- Stage 3 Task 3.2 can load `per_layer_token_embd.weight` as a single buffer and
  bind `[layer_idx * 256, (layer_idx+1) * 256)` rows per forward call using an
  offset + size view (or a pre-split upload strategy).
- The chunked dispatch path in `token_embd` machinery is **not** required for
  PLE; do not inherit it. Keeping PLE dispatch simple (one binding per layer)
  reduces shader complexity and avoids a second source of chunking bugs.

## Total memory cost (GPU-resident, Stage 3 baseline)

| Tensor | Bytes | MiB | GiB |
|--------|-------|-----|-----|
| `per_layer_token_embd.weight` | 1,614,807,040 | 1,540.00 | 1.504 |
| `per_layer_model_proj.weight` | 27,525,120    |    26.25 | 0.026 |
| `per_layer_proj_norm.weight`  | 1,024          |    ~0    | ~0   |
| **PLE subtotal**              | **1,642,333,184** | **1,566.25** | **1.530** |

Additive on top of the 3.11 GB Q4_K_M weight footprint (existing token_embd + all
transformer layers). Combined ceiling: **~4.64 GB GPU-resident** before KV cache.

On the 16 GB minimum hardware floor (WebGPU sees ~10–11 GB), this is within
budget. On 32 GB recommended hardware, comfortable.

## Followups

- Stage 3 Task 3.2 loads this tensor; use a single binding per layer (no
  chunking required — binding-cap assessment confirmed 44 MiB << 128 MiB).
- Stage 3 Task 3.3 injects per-layer lookups into the residual; the 256-dim PLE
  output is added to the 1536-dim hidden state — this requires a zero-padded or
  separately projected add (the PLE dim 256 ≠ hidden dim 1536). The
  `per_layer_model_proj.weight` tensor [1536, 8960] — structured as
  [hidden_dim, ple_dim × block_count] — is the learned projection that maps the
  256-dim PLE slice up to 1536 before residual addition.
- Confirm the injection point (pre-norm vs. post-norm per block) by inspecting
  the llama.cpp Gemma 4 compute graph or the reference HuggingFace implementation
  before Task 3.3.

---

## Tensor list addendum — Task 3.2 broad probe (2026-05-10)

Broader re-probe (`per_layer OR embd OR norm` filter) confirmed the full
PLE tensor set. Three tensors, not two, are required:

| Tensor name | Shape | Dtype | Role |
|---|---|---|---|
| `per_layer_token_embd.weight` | [8960, 262144] | Q5_K | Lookup table; row-index by token ID, column slices by layer |
| `per_layer_model_proj.weight` | [1536, 8960] | BF16 | Linear projection: [pleDim × layerCount, n_tokens] → [hiddenDim, n_tokens] |
| `per_layer_proj_norm.weight`  | [256]         | F32  | RMSNorm scale applied to the projection output before ADD |

All three are present in `gemma-4-e2b-it-q4km.gguf`. No `per_layer_token_embd_norm.weight`
or `per_layer_token_embd.norm.weight` variant exists — the norm key is
`per_layer_proj_norm.weight` (confirmed by llama.cpp `src/llama-arch.cpp:428`
mapping `LLM_TENSOR_PER_LAYER_PROJ_NORM → "per_layer_proj_norm"`).

### Op sequence (confirmed from `~/Repos/llama.cpp/src/models/gemma3n.cpp`)

**`build_inp_per_layer()`** — runs once before the layer loop:
1. `GET_ROWS(per_layer_token_embd.weight, token_ids)` → `[pleDim×layerCount, n_tokens]`
   scaled by `sqrt(pleDim)` (= sqrt(256) = 16)

**`project_per_layer_inputs(inp_batch, inp_per_layer)`** — also pre-loop:
2. `MUL_MAT(per_layer_model_proj.weight, inp_batch)` → `[pleDim×layerCount, n_tokens]`
   scaled by `1/sqrt(hiddenDim)` (= 1/sqrt(1536))
3. `RMS_NORM(per_layer_proj_norm.weight)` applied to step-2 result
4. `ADD(step-3, step-1)` → scaled by `1/sqrt(2)` → `inp_per_layer [pleDim, n_tokens, layerCount]`
   (permuted to `[pleDim, n_tokens, layerCount]` for fast per-layer slicing)

**Per-block** (inside the layer loop, layer `L`):
5. `SLICE(inp_per_layer, L)` → `[pleDim, n_tokens]`  ← the PLE residual for this block
6. `MUL_MAT(blk.L.per_layer_inp_gate.weight, active_prediction)` + GELU
7. `MUL(step-6, step-5)` → gated PLE contribution `[pleDim, n_tokens]`
8. `MUL_MAT(blk.L.per_layer_proj.weight, step-7)` → `[hiddenDim, n_tokens]`
9. `RMS_NORM(blk.L.per_layer_post_norm.weight)`
10. ADD into `corrected[1:]` (all AltUp streams except the active one)

**Injection point:** PLE does NOT sit between `attn_norm` and the attention op.
It is applied **after the AltUp correct step** (`altup_correct(predictions, attn_ffw_laurel_gated)`),
modifying the non-active AltUp streams. It is therefore a correction residual,
not a main-stream residual addition.

Note: `blk.L.per_layer_inp_gate.weight` and `blk.L.per_layer_proj.weight` are
**per-block** weights (not global PLE tensors). Task 3.3 must also load these
from the GGUF tensor list (`blk.%d.inp_gate`, `blk.%d.proj` via
`LLM_TENSOR_PER_LAYER_INP_GATE` / `LLM_TENSOR_PER_LAYER_PROJ` in llama-arch.cpp).
They are not part of the Task 3.2 surface (global PLE tensors only).
