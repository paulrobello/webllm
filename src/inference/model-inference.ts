import type { ModelHyperparams } from "../core/types.js";
import type { GgufContext, GgufTensorInfo } from "../models/gguf-types.js";
import {
	type BufferPtr,
	F32_BYTES,
	GgmlType,
	type GgmlWasm,
	type GraphComputeProfile,
	RopeMode,
	type TensorDownloadRequest,
	type TensorPtr,
} from "./ggml-wasm.js";

interface LayerWeights {
	attnNorm: TensorPtr;
	// Optional Phi-3 attn-norm bias. RMSNorm + bias add. Null on llama /
	// qwen / mistral GGUFs (which lack norm biases) and on Phi-3.5-mini
	// (which also has no norm biases) — the field stays in the interface
	// so future Phi-3 variants that ship norm biases can light up the
	// path without an interface change.
	attnNormBias: TensorPtr | null;
	// Phi-3 fused QKV: single [3*n_embd, n_embd] matrix (or [E + 2*kvDim,
	// n_embd] under GQA). When non-null, qProj/kProj/vProj are null and
	// the forward graph takes the fused matmul + view-slice path
	// (mirrors encoder-inference.ts:263-296).
	qkvFused: TensorPtr | null;
	qProj: TensorPtr | null;
	kProj: TensorPtr | null;
	vProj: TensorPtr | null;
	// Qwen2 / Qwen2.5 use biased Q/K/V projections. Llama, Qwen3, Mistral,
	// and Phi-3 (whose biases — if any — are baked into qkvFused) don't —
	// these stay null and the forward graph skips the add. Without this,
	// qwen2 GGUFs produce garbage (random-token) output because Q/K/V are
	// off by the bias shift.
	qBias: TensorPtr | null;
	kBias: TensorPtr | null;
	vBias: TensorPtr | null;
	qNorm: TensorPtr | null;
	kNorm: TensorPtr | null;
	oProj: TensorPtr;
	ffnNorm: TensorPtr;
	ffnNormBias: TensorPtr | null;
	// Phi-3 fused gate-up: single [2*n_ff, n_embd] matrix. When non-null,
	// gateProj/upProj are null and the forward graph splits the matmul
	// output into gate/up halves before SwiGLU.
	gateUpFused: TensorPtr | null;
	gateProj: TensorPtr | null;
	upProj: TensorPtr | null;
	downProj: TensorPtr;
	// Gemma family post-norm pattern: extra RMSNorm applied to attention
	// output AND FFN output, BEFORE the residual add. Null for every other
	// architecture (Llama/Qwen/Mistral/Phi family does pre-norm only).
	postAttentionNorm: TensorPtr | null;
	postFfwNorm: TensorPtr | null;
	// Gemma 4 per-block Per-Layer Embedding tensors. Null for every other arch.
	pleInpGate: TensorPtr | null; // blk.L.inp_gate.weight   [n_embd, pleDim]
	plePerBlockProj: TensorPtr | null; // blk.L.proj.weight    [pleDim, n_embd]
	plePostNorm: TensorPtr | null; // blk.L.post_norm.weight  [n_embd]
	// Gemma 4 per-layer output scale: a learned [1]-shape scalar multiplied
	// into the layer's residual after PLE injection and before the next layer.
	// Null on all other architectures (absent from GGUF).
	layerOutputScale: TensorPtr | null;
	// Per-layer RoPE freq_factors weight (Gemma 4 global-attention layers,
	// Llama 3.1, etc.) — shape [n_embd_head/2]. Applied as per-dim divisor
	// to theta inside ggml_rope_ext. Null on Gemma 4 SWA layers, all
	// pre-Llama-3.1 GGUFs, Qwen, Mistral, Phi (absent from GGUF).
	ropeFreqs: TensorPtr | null;
}

interface WeightTensors {
	tokEmb: TensorPtr;
	norm: TensorPtr;
	// Optional Phi-3 final-norm bias.
	normBias: TensorPtr | null;
	output: TensorPtr | null;
	// Phi-3 lm_head bias. Null for all other architectures and for
	// Phi-3.5-mini (its lm_head is biasless).
	outputBias: TensorPtr | null;
	// Gemma 4 / Gemma 3N Per-Layer Embedding (PLE) global tensors.
	// Null for all non-Gemma-4 architectures (tensors absent from GGUF).
	perLayerEmbed: TensorPtr | null;
	perLayerProj: TensorPtr | null;
	perLayerProjNorm: TensorPtr | null;
	layers: LayerWeights[];
}

interface LayerKVCache {
	k: TensorPtr;
	v: TensorPtr;
}

/**
 * Per-phase wall-clock timing for one `forward()` call, in milliseconds.
 * Captured only when `traceEnabled` is true — the default is off so the
 * hot path pays nothing. Used by the profile harness in `eval/perf.ts`.
 */
export type DecodeMode = "full" | "greedy" | "topk" | "verify";

export interface DecodeResult {
	/** Full logits (mode='full'). */
	logits?: Float32Array;
	/** Greedy argmax token ID (mode='greedy'). */
	tokenId?: number;
	/** Top-K indices (mode='topk'). */
	topKIndices?: Int32Array;
	/** Top-K logit values (mode='topk'). */
	topKValues?: Float32Array;
}

export interface ForwardTrace {
	mode: DecodeMode;
	nTokens: number;
	pastLen: number;
	ctxCreateMs: number;
	buildGraphMs: number;
	backendAllocMs: number;
	uploadLeavesMs: number;
	graphComputeMs: number;
	downloadResultMs: number;
	teardownMs: number;
	totalMs: number;
	backendProfileTotalMs?: number | undefined;
	backendMatmulMs?: number | null | undefined;
	backendAttentionMs?: number | null | undefined;
	backendEncodeOverheadMs?: number | undefined;
	backendDispatchCount?: number | undefined;
	backendBreakdownAvailable?: boolean | undefined;
}

/**
 * Per-architecture softmax pre-scale for attention. Most models use the
 * standard `1 / sqrt(head_dim)` so QK^T sits in unit variance before the
 * softmax. Gemma 3 / 3n / 4 are the exception — `gemma4.cpp:11` sets
 * `hparams.f_attention_scale = 1.0` ("Gemma4 uses self.scaling = 1.0,
 * no pre-attn scaling"), passed verbatim as `kq_scale` to
 * `ggml_soft_max_ext` at `llama-graph.cpp:2033`. The q/k norm gains in
 * Gemma 4 are trained to compensate for the missing 1/√d_k factor;
 * applying the default scale produces VASTLY softer (more uniform)
 * attention weights and the residual stream drifts non-uniformly across
 * layers (parity-capture diagnostic 2026-05-11: cosine 0.97 at L0/L1,
 * 0.66 catastrophic drop at L2, end-of-stack 0.14).
 */
export function attnSoftmaxScale(
	hp: { architecture: ModelHyperparams["architecture"] },
	headDim: number,
): number {
	if (hp.architecture === "gemma4") return 1.0;
	return 1.0 / Math.sqrt(headDim);
}

/**
 * Gemma family architectures (gemma, gemma2, gemma3, gemma4) share
 * input-embed scaling by sqrt(embeddingLength) and GELU-parallel FFN
 * activation per llama.cpp/src/models/gemma{,2,3,4}.cpp. Gemma 4 adds
 * QK-norm + attention scale 1.0 + V bare-RMS-norm on top; those stay
 * gated to gemma4 specifically (see attnSoftmaxScale + V branch).
 */
function isGemmaFamily(arch: ModelHyperparams["architecture"]): boolean {
	return (
		arch === "gemma" ||
		arch === "gemma2" ||
		arch === "gemma3" ||
		arch === "gemma4"
	);
}

export function getRopeModeForArchitecture(
	architecture: ModelHyperparams["architecture"],
): number {
	// nomic-bert uses GPT-NeoX-style RoPE (split-halves), per
	// llama.cpp/src/llama-model.cpp:9266 (LLM_ARCH_NOMIC_BERT →
	// LLAMA_ROPE_TYPE_NEOX). The HF config also surfaces this as
	// `rotary_emb_interleaved: false`.
	if (architecture === "nomic-bert") return RopeMode.NEOX;
	// Phi-3 / Phi-3.5 / PhiMoE all use NEOX-style RoPE (split-halves)
	// per llama.cpp's llama-model.cpp:9282 — phi3 sits in the same
	// case-list as PLAMO, GEMMA, STARCODER2, GPTNEOX, etc., all
	// returning LLAMA_ROPE_TYPE_NEOX. Without this, Phi-3 forward
	// produces fluent-but-meaningless output (interleaved RoPE
	// rotates the wrong feature pairs and the model decodes to
	// nonsense like "IMDbSidenoteSidenotepisode...").
	if (architecture === "phi3") return RopeMode.NEOX;
	// Gemma family (gemma, gemma2, gemma3, gemma3n, gemma4) uses
	// NEOX-style RoPE — see llama-model.cpp:2275-2310 where every
	// LLM_ARCH_GEMMA* case falls through to LLAMA_ROPE_TYPE_NEOX.
	// Without this, the per-block forward path diverges
	// catastrophically as a function of sequence length — short
	// completion prompts squeak past the parity threshold but
	// chat-formatted eval prompts (≥ ~20 tokens) collapse the
	// residual stream to near-orthogonal vs the HF reference and
	// the model emits `<eos>` (id 1) as its first response token.
	// Closure data: Task 3.5 Phase B bisection (length is the
	// dominant variable) at
	// `eval/reports/parity-gemma-4-e2b-phaseB-bisect-2026-05-11/`.
	// gemma2/gemma3 are included pre-emptively: the demote of
	// `gemma-2-2b-warm` from the canonical fleet on 2026-05-01
	// attributed failure to multiple Gemma 2 quirks (post-norms,
	// SWA, soft-capping, tied embeddings); the NEOX-RoPE bug was
	// likely the dominant cause. Re-probing gemma2 is a separate
	// follow-up filed under "post-NEOX-fix audit candidates".
	if (
		architecture === "gemma" ||
		architecture === "gemma2" ||
		architecture === "gemma3" ||
		architecture === "gemma4"
	) {
		return RopeMode.NEOX;
	}
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}

/**
 * Largest prefill tile that keeps each `forwardSingle` call inside the
 * upstream `ggml-webgpu` FA VEC path. The VEC selector at
 * `ggml-webgpu-shader-lib.hpp:734` requires `src0.ne[1] < 20` (Q tokens);
 * VEC is the only FA shader path that fits Gemma family head_dim ∈ {256, 512}
 * in WebGPU's 16 KiB LDS budget. At `q_tile=4` TILE/SUBGROUP_MATRIX consume
 * too much shared memory and `max_kv_tile` rounds to 0 after the subgroup-size
 * granularity quantizer → path = NONE → `GGML_ASSERT` fires at
 * `ggml-webgpu-shader-lib.hpp:2560`. Repro + bisection at
 * `eval/reports/gemma-4-stage4-swa-mask-2026-05-12/SUMMARY.md`. 16 chosen
 * over 19 (just under the bound) for a margin of safety and to keep the tile
 * count obvious in profile traces.
 */
const FA_LARGE_HEAD_DIM_PREFILL_TILE = 16;

/**
 * Largest per-layer head_dim before the FA VEC clamp kicks in. Llama / Qwen /
 * Mistral / Phi-3 family head_dims are all ≤ 128 and route through TILE
 * comfortably; Gemma 2/3/4 use 256 (SWA) or 512 (global) and blow the LDS
 * budget at every path except VEC.
 */
const FA_VEC_HEAD_DIM_THRESHOLD = 128;

/**
 * Compute the default `prefillTileSize` for a model based on hyperparameters.
 *
 * Two layered rules, applied in order:
 *
 * 1. **FA VEC clamp** (set 2026-05-12). When `flashAttn === true` AND any
 *    layer's head_dim exceeds {@link FA_VEC_HEAD_DIM_THRESHOLD}, clamp the
 *    tile size to {@link FA_LARGE_HEAD_DIM_PREFILL_TILE} so each
 *    `forwardSingle` call sees `nTokens < 20` and FA VEC engages. Without
 *    this clamp Gemma 4 + FA prefill at chat-template length traps with
 *    `RuntimeError: unreachable` on the upstream `decisions.path != NONE`
 *    assert. See the doc comment on `FA_LARGE_HEAD_DIM_PREFILL_TILE` for
 *    the upstream constraint chain.
 *
 * 2. **§22 graph-allocator clamp.** `layerCount >= 32` → 128, else 0.
 *    Maps to the §22 abort signature observed in
 *    `eval/reports/prefill-tiling-2026-04-27/00-phase0-diagnostic.txt`:
 *    "ggml_tallocr_alloc: not enough space in the buffer" at ggml-alloc.c:82
 *    during `backendAllocCtxTensors`. F32 intermediates scale with
 *    layers × seq × emb; layer count is the dominant predictor of per-tile
 *    graph allocator pressure, so the gate keys off layers alone.
 *
 *    Originally an AND gate (`layerCount >= 32 AND embeddingLength >= 4096`),
 *    but qwen3-4B (36 layers × 2560 emb) reproducibly aborted on tc-005's
 *    3-tool prompt despite being below the emb gate. The 36-layer count is
 *    exactly the §22 abort regime — at seq≈800 (tc-005's tool-heavy
 *    prefill) the F32 intermediates total 295 MB, larger than the
 *    Mistral-7B-at-seq-512 case the heuristic was originally tuned for.
 *    Dropping the emb gate keeps qwen3-4B inside the tiling envelope while
 *    still leaving sub-32-layer models (qwen3-1.7B, tinyllama, smollm2)
 *    untiled — they stay well within the graph budget.
 *
 * Override surface (ctor opt / `?prefillTile=` / `--prefill-tile`) wins
 * unconditionally — including the explicit-zero force-disable path.
 *
 * Spec: `docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md`.
 */
export function computeDefaultPrefillTileSize(
	hp: ModelHyperparams,
	flashAttn = false,
): number {
	const sec22 = hp.layerCount >= 32 ? 128 : 0;
	if (!flashAttn) return sec22;
	const maxHeadDim = hp.embeddingHeadLengthPerLayer
		? Math.max(...hp.embeddingHeadLengthPerLayer)
		: hp.embeddingHeadLength;
	if (maxHeadDim <= FA_VEC_HEAD_DIM_THRESHOLD) return sec22;
	// Large head_dim + FA: clamp to FA VEC ceiling. When §22 is already
	// non-zero, take the min so we don't accidentally raise it.
	return sec22 === 0
		? FA_LARGE_HEAD_DIM_PREFILL_TILE
		: Math.min(sec22, FA_LARGE_HEAD_DIM_PREFILL_TILE);
}

/**
 * Verify a tensor is laid out contiguously as F32. The fused-projection
 * helpers `buildQKV` / `buildFFNGateUp` produce strided `opView*` slices
 * that they immediately wrap in `opCont()` to materialize a contiguous
 * f32 copy — required because downstream rope/permute ops assume
 * contiguity. If a future refactor inserts an op between the helper and
 * the rope/permute chain that re-introduces strided derivatives, this
 * assertion fires loud at graph build time instead of producing silent
 * gibberish at decode (the failure mode of bug #2 in the Phi-3 closure,
 * commit `7c85a2a`).
 *
 * Implements the same predicate as ggml's `ggml_is_contiguous`, scoped
 * to F32 (the dtype of every fused-projection output here): tightly
 * packed strides nb[0]=4, nb[i]=nb[i-1]*ne[i-1] for i=1..3.
 */
export function assertContiguousF32(
	wasm: Pick<GgmlWasm, "tensorType" | "tensorNe" | "tensorNb">,
	tensor: TensorPtr,
	label: string,
): void {
	const type = wasm.tensorType(tensor);
	if (type !== GgmlType.F32) {
		throw new Error(
			`${label}: expected F32 (type ${GgmlType.F32}), got type ${type}`,
		);
	}
	const ne0 = wasm.tensorNe(tensor, 0);
	const ne1 = wasm.tensorNe(tensor, 1);
	const ne2 = wasm.tensorNe(tensor, 2);
	const nb0 = wasm.tensorNb(tensor, 0);
	const nb1 = wasm.tensorNb(tensor, 1);
	const nb2 = wasm.tensorNb(tensor, 2);
	const nb3 = wasm.tensorNb(tensor, 3);
	if (
		nb0 !== F32_BYTES ||
		nb1 !== nb0 * ne0 ||
		nb2 !== nb1 * ne1 ||
		nb3 !== nb2 * ne2
	) {
		throw new Error(
			`${label}: tensor not F32-contiguous (ne=[${ne0},${ne1},${ne2}], nb=[${nb0},${nb1},${nb2},${nb3}])`,
		);
	}
}

/**
 * Convert raw BF16 bytes to F32 bytes.
 *
 * BF16 is the high 16 bits of an equivalent F32 value (identical sign +
 * exponent layout; mantissa truncated to 7 bits). Conversion to F32 is
 * therefore lossless: shift each u16 into the high half of a u32 with the
 * low 16 bits zeroed.
 *
 * Used when WebGPU lacks BF16 support: the GGUF source is BF16 but the
 * GPU-side tensor is allocated as F32 so the well-supported
 * `mul_mat_f32_f32` shader fires instead of the unsupported
 * `mul_mat_f32_bf16` (which fails compile and cascades through
 * CommandBuffer invalidation).
 */
export function bf16BytesToF32Bytes(bf16: Uint8Array): Uint8Array {
	if (bf16.byteLength % 2 !== 0) {
		throw new Error(
			`bf16BytesToF32Bytes: expected even byte count, got ${bf16.byteLength}`,
		);
	}
	const u16 = new Uint16Array(
		bf16.buffer,
		bf16.byteOffset,
		bf16.byteLength / 2,
	);
	const u32 = new Uint32Array(u16.length);
	for (let i = 0; i < u16.length; i++) {
		u32[i] = u16[i] << 16;
	}
	return new Uint8Array(u32.buffer);
}

/**
 * Fill an F16 causal mask buffer in ggml-webgpu layout for `opSoftMaxExt` /
 * `opFlashAttn`. The mask is a `[totalLen, maskPaddedCols]` tensor (ne0=totalLen,
 * ne1=maskPaddedCols), so element `(row q, col k)` lives at `view[q*totalLen+k]`.
 * Cells are F16 bit patterns: `0x0000` = visible (0.0), `0xFC00` = masked (-Inf).
 *
 * `swaWindow` undefined → full causal mask (the existing behavior; visible iff
 * `k ≤ pastLen + q`). Finite → banded causal / sliding-window mask
 * (visible iff `pastLen + q - swaWindow < k ≤ pastLen + q`), used by Gemma
 * SWA layers.
 *
 * Padding rows `[nTokens, maskPaddedCols)` are filled with 0; they're outside
 * the query range but still read by the shader.
 *
 * Verified Stage 4.0 — both `soft_max.wgsl` and `flash_attn.wgsl` apply the
 * mask as a purely additive per-element term, so a banded mask is admissible
 * without shader changes. See `eval/reports/gemma-4-stage4-probe-2026-05-11/`.
 */
export function writeCausalMaskF16(
	view: Uint16Array,
	totalLen: number,
	nTokens: number,
	pastLen: number,
	maskPaddedCols: number,
	swaWindow?: number,
): void {
	const F16_NEG_INF = 0xfc00;
	const useWindow =
		swaWindow !== undefined &&
		Number.isFinite(swaWindow) &&
		(swaWindow as number) > 0;
	for (let q = 0; q < nTokens; q++) {
		const rowBase = q * totalLen;
		const visibleUpTo = pastLen + q;
		const visibleFrom = useWindow ? visibleUpTo - (swaWindow as number) + 1 : 0;
		for (let k = 0; k < totalLen; k++) {
			view[rowBase + k] =
				k <= visibleUpTo && k >= visibleFrom ? 0 : F16_NEG_INF;
		}
	}
	for (let q = nTokens; q < maskPaddedCols; q++) {
		const rowBase = q * totalLen;
		for (let k = 0; k < totalLen; k++) view[rowBase + k] = 0;
	}
}

/** Round `v` up to the next multiple of `mult` (mask row/column padding). */
function padTo(v: number, mult: number): number {
	return Math.ceil(v / mult) * mult;
}

/**
 * Per-mode graph sizing (ctx bytes + node count) for the forward-pass builders.
 * Each mode maps to the EXACT expression the inline code used before ARC-001 —
 * encoded as parameters, never collapsed to one: embedding uses a smaller
 * per-element ctx multiplier (128) and fewer nodes; taps adds per-layer +
 * embedding-output tap nodes; standard (single/allPositions/decode) carries the
 * PLE-adjusted node count (buildPreLoopPle + injectPerBlockPle).
 */
type ForwardGraphMode = "standard" | "embedding" | "taps";
function computeGraphSizing(
	hp: ModelHyperparams,
	totalLen: number,
	mode: ForwardGraphMode,
): { ctxBytes: number; nodeCount: number } {
	const perElemBytes = mode === "embedding" ? 128 : 256;
	const ctxBytes =
		hp.layerCount * 32768 + totalLen * hp.embeddingLength * perElemBytes;
	const nodeCount =
		mode === "embedding"
			? hp.layerCount * 32 + 128
			: mode === "taps"
				? hp.layerCount * 80 + 200
				: hp.layerCount * 72 + 160; // standard (PLE-adjusted)
	return { ctxBytes, nodeCount };
}

/**
 * Terminal output kind for {@link ModelInference.buildForwardGraph}.
 *
 * - `"logits"`: full lm_head output (forwardSingle / forwardAllPositions).
 * - `"decode"`: logits followed by argmax or topk reduction (forwardDecode).
 * - `"embedding"`: post-`output_norm` hidden state, no lm_head (forwardForEmbedding).
 * - `"taps"`: logits + per-layer hidden taps (forwardWithLayerTaps).
 *
 * B1 implements `"logits"` (forwardSingle); B2 adds `"decode"` (forwardDecode);
 * B3 delegates `forwardAllPositions` onto the same `"logits"` path; B4 adds
 * `"embedding"` (forwardForEmbedding); B5 will add `"taps"`.
 */
type ForwardOutputKind = "logits" | "decode" | "embedding" | "taps";

/**
 * Options for {@link ModelInference.buildForwardGraph}.
 *
 * Encodes the inputs and sizing/output discriminators for a forward-pass
 * graph. Designed to generalize across the 5 forward variants
 * (forwardSingle / forwardAllPositions / forwardDecode / forwardForEmbedding /
 * forwardWithLayerTaps); B1 implemented `mode: "standard"` +
 * `output: "logits"` (forwardSingle); B2 adds `output: "decode"` (forwardDecode);
 * B3 delegates forwardAllPositions onto the same `output: "logits"` path.
 */
interface BuildForwardGraphOpts {
	/** Sizing mode — drives {@link computeGraphSizing} (ctx bytes + node count). */
	mode: ForwardGraphMode;
	/** Terminal output kind — determines the graph's final tensor(s). */
	output: ForwardOutputKind;
	/**
	 * Decode mode — required when `output === "decode"`. Selects the
	 * argmax/topk/full tail that {@link buildForwardGraph} appends after the
	 * shared logits body.
	 */
	decodeMode?: DecodeMode;
	/**
	 * Top-K value — required when `output === "decode" && decodeMode === "topk"`.
	 */
	topK?: number;
	/** Input token IDs (length {@link nTokens}). */
	tokenIds: Int32Array;
	/** Input positions (length {@link nTokens}). */
	positions: Int32Array;
	/** Number of tokens in this forward call. */
	nTokens: number;
	/** Number of cached KV positions before this call. */
	pastLen: number;
	/** `pastLen + nTokens` — total sequence length after this call. */
	totalLen: number;
	/** RoPE mode for the architecture (from `getRopeModeForArchitecture`). */
	ropeMode: number;
	/** Whether a causal attention mask tensor is needed (`nTokens > 1`). */
	needsMask: boolean;
	/** Mask column count padded to the 32-wide FA requirement. */
	maskPaddedCols: number;
	/** Sliding-window size (`hp.slidingWindowSize ?? 0`); 0 = no SWA. */
	swaWindow: number;
	/** Whether an SWA mask tensor is needed. */
	needsSwaMask: boolean;
	/** When true, capture `t1`/`t2` trace timestamps. */
	trace: boolean;
	/**
	 * Taps-output controls — required when `output === "taps"`, ignored
	 * otherwise. Mirrors the {@link forwardWithLayerTaps} opts of the same
	 * names; see that method for the per-field rationale (128 MiB per-binding
	 * cap management, incremental long-context parity probe support).
	 */
	taps?: {
		/** Skip all per-layer residual taps (long-context probe). */
		skipLayerTaps: boolean;
		/** Pin only this layer's residual tap; undefined = capture every layer. */
		captureLayer: number | undefined;
		/** Slice lm_head input to the last-token column (parity probe). */
		lastTokenLogitsOnly: boolean;
	};
}

/**
 * Result of {@link ModelInference.buildForwardGraph}: the built graph handle,
 * the terminal output tensor, and the leaf input tensors that the caller must
 * upload data into (via {@link ModelInference.uploadLeaves}) before compute.
 */
interface BuiltForwardGraph {
	/** Graph handle (for `graphCompute` / `backendAllocCtxTensors`). */
	graph: number;
	/**
	 * Terminal output tensor — `logits` for `output: "logits"`; argmax result
	 * / top-K values / logits depending on `decodeMode` for `output: "decode"`;
	 * final hidden state for `output: "embedding"`; last-token logits for
	 * `output: "taps"`.
	 */
	outputTensor: TensorPtr;
	/**
	 * Secondary output tensor — top-K indices for `output: "decode"` +
	 * `decodeMode: "topk"`; 0 for all other output kinds/modes.
	 */
	secondaryTensor: TensorPtr;
	/** Position leaf tensor. */
	posTensor: TensorPtr;
	/** Token-ID leaf tensor. */
	tokenIdsTensor: TensorPtr;
	/** Causal mask leaf tensor (0 if not created). */
	maskTensor: TensorPtr;
	/** SWA mask leaf tensor (0 if not created). */
	swaMaskTensor: TensorPtr;
	/** Trace: timestamp after `ctxCreate`; 0 when `!trace`. */
	t1: number;
	/** Trace: timestamp after `graphBuildForwardExpand`; 0 when `!trace`. */
	t2: number;
	/**
	 * Taps-output (B5) fields — populated only for `output: "taps"`;
	 * 0 / empty for all other output kinds.
	 * `embeddingTapTensor` is the pre-block-0 hidden (HF `hidden_states[0]`
	 * equivalent); `layerTapsTensors` is the per-layer residual-tap array
	 * (sparse in single-layer-capture mode); `finalHiddenTensor` is the
	 * post-`output_norm` hidden (last-token slice when `lastTokenLogitsOnly`).
	 */
	embeddingTapTensor: TensorPtr;
	layerTapsTensors: TensorPtr[];
	finalHiddenTensor: TensorPtr;
}

/**
 * Manages model weights loaded into WASM/ggml tensors and runs forward passes.
 *
 * Loads GGUF weight data into ggml tensors allocated on the WebGPU backend,
 * then builds and executes computation graphs for transformer forward passes.
 * Supports incremental decoding via KV cache for O(n) autoregressive generation.
 *
 * V cache is stored as [maxCtx, headDim, nKvHeads] so that mul_mat(V, attn)
 * produces [headDim, nTokens, nHeads] with built-in GQA broadcast.
 */
export class ModelInference {
	private wasm: GgmlWasm;
	private hp: ModelHyperparams;
	private weights: WeightTensors | null = null;
	private weightBuf: BufferPtr = 0;
	private nameToTensor = new Map<string, TensorPtr>();
	// Tensor names whose GGUF source type was BF16 and have been allocated
	// as F32 on the GPU. Bytes need BF16 → F32 conversion at upload.
	// WebGPU has no native bf16 support; mul_mat_f32_bf16 fails to compile
	// and silently corrupts every dependent kernel. Casting to F32 lifts the
	// load-bearing matmul onto the supported mul_mat_f32_f32 path. Lossless
	// in the BF16 → F32 direction.
	private bf16OverriddenNames = new Set<string>();

	private kvLayers: LayerKVCache[] | null = null;
	private kvBuf: BufferPtr = 0;
	private nCached = 0;
	private kvMaxContextLength = 0;
	/** When true, `forward()` populates `lastTrace` on every call. */
	traceEnabled = false;
	/** Timing of the most recent `forward()` call, or null if never traced. */
	lastTrace: ForwardTrace | null = null;
	/**
	 * When true, attention call sites use ggml_flash_attn_ext + an FA-ready
	 * V cache layout ([headDim, maxCtx, nKvHeads]). When false, they use the
	 * manual opMulMat(K,Q) → opSoftMaxExt → opMulMat(V,attn) chain + the
	 * legacy V layout ([maxCtx, headDim, nKvHeads]).
	 *
	 * Pinned at construction. Switching at runtime would require either two
	 * cached V tensors (memory bloat) or a per-step transpose copy
	 * (regression). Set once and treat as immutable.
	 */
	readonly flashAttn: boolean;
	/**
	 * When > 0, `forward()` automatically chunks long inputs into tiles of
	 * this many tokens. Each tile is a separate ctxCreate → graph build →
	 * graph compute → ctxFree cycle; KV cache accumulates across tiles via
	 * the existing `nCached = totalLen` advance.
	 *
	 * Default 0 = disabled (single-call forward, bit-identical to pre-§22).
	 *
	 * Used to unblock long-prefill on 7B+ models that otherwise abort in
	 * the host-side ggml graph allocator at `backend_alloc_ctx_tensors`.
	 * See TODO §22 for the per-model recommended tile size and rationale.
	 */
	readonly prefillTileSize: number;
	/**
	 * When true (default), `buildQKV` / `buildFFNGateUp` verify that every
	 * fused-projection output (Q/K/V/gate/up) materialized via `opCont` is
	 * F32-contiguous. The check costs ~7 wasm-bridge round trips per output
	 * × 5 outputs/layer × layerCount per forward — measured at <1% of
	 * graph-build wall time on Phi-3.5-mini, paid only when
	 * `LayerWeights.qkvFused` / `gateUpFused` are non-null (Phi-3 family
	 * today). Set false to disable for benchmarking; the assertion is
	 * defense-in-depth against silent regression of the contiguous-view
	 * invariant fixed in commit `7c85a2a`.
	 */
	assertFusedContiguity: boolean;

	constructor(
		wasm: GgmlWasm,
		hyperparams: ModelHyperparams,
		opts: {
			flashAttn?: boolean;
			prefillTileSize?: number;
			assertFusedContiguity?: boolean;
		} = {},
	) {
		this.wasm = wasm;
		this.hp = hyperparams;
		this.flashAttn = opts.flashAttn ?? false;
		this.prefillTileSize =
			opts.prefillTileSize ??
			computeDefaultPrefillTileSize(hyperparams, this.flashAttn);
		if (this.prefillTileSize < 0) {
			throw new Error(
				`prefillTileSize must be >= 0; got ${this.prefillTileSize}`,
			);
		}
		this.assertFusedContiguity = opts.assertFusedContiguity ?? true;
	}

	loadWeights(
		ggufCtx: GgufContext,
		ggufData: Uint8Array | ((offset: number, byteLength: number) => Uint8Array),
	): void {
		// Callback form is required when the source bytes live in the WASM
		// heap: ctxCreate / backendAllocCtxTensors / per-chunk scratch
		// mallocs below may grow memory, which detaches any pre-existing
		// JS view of HEAPU8. uploadRangeChunked re-derives a fresh view
		// from the live heap once per chunk *after* its scratch malloc,
		// so the upload path stays valid across grow events.
		const isCallback = typeof ggufData === "function";
		const dataAt = isCallback
			? ggufData
			: (off: number, len: number) =>
					new Uint8Array(ggufData.buffer, ggufData.byteOffset + off, len);
		const { hp, wasm } = this;
		const tensorMap = new Map<string, GgufTensorInfo>();
		for (const t of ggufCtx.tensors) {
			tensorMap.set(t.name, t);
		}

		// ctxCreate uses no_alloc=true (see webgpu-bridge.cpp::ctx_create);
		// tensor data lives in GPU buffers assigned by backendAllocCtxTensors,
		// not in this mempool. We only need room for ggml_tensor + ggml_object
		// metadata per tensor — adding ggufCtx.totalDataSize here would
		// allocate a multi-GB unused buffer and push 4B-class models over
		// the WASM 4 GB cap.
		const memSize = ggufCtx.tensors.length * 16384 + (1 << 20);
		wasm.ctxCreate(memSize);

		const tokEmb = this.makeTensor(tensorMap, "token_embd.weight");
		const norm = this.makeTensor(tensorMap, "output_norm.weight");
		const output = tensorMap.has("output.weight")
			? this.makeTensor(tensorMap, "output.weight")
			: null;
		const outputBias = tensorMap.has("output.bias")
			? this.makeTensor(tensorMap, "output.bias")
			: null;
		const normBias = tensorMap.has("output_norm.bias")
			? this.makeTensor(tensorMap, "output_norm.bias")
			: null;

		// Gemma 4 / Gemma 3N PLE global tensors — absent on all other architectures.
		const perLayerEmbed = tensorMap.has("per_layer_token_embd.weight")
			? this.makeTensor(tensorMap, "per_layer_token_embd.weight")
			: null;
		const perLayerProj = tensorMap.has("per_layer_model_proj.weight")
			? this.makeTensor(tensorMap, "per_layer_model_proj.weight")
			: null;
		const perLayerProjNorm = tensorMap.has("per_layer_proj_norm.weight")
			? this.makeTensor(tensorMap, "per_layer_proj_norm.weight")
			: null;

		// Shared RoPE freq_factors weight: present in Gemma 4 (one tensor
		// TENSOR_DUPLICATED across global-attention layers per gemma4.cpp:86-88)
		// and in Llama 3.1 (YaRN scaling table). Per-layer assignment below
		// gates this tensor through `slidingWindowPattern` so Gemma 4 SWA
		// layers stay null (no freq_factors during their RoPE).
		const ropeFreqsGlobal = tensorMap.has("rope_freqs.weight")
			? this.makeTensor(tensorMap, "rope_freqs.weight")
			: null;

		const isPhi3 = hp.architecture === "phi3";
		const layers: LayerWeights[] = [];
		for (let i = 0; i < hp.layerCount; i++) {
			const p = (s: string) => `blk.${i}.${s}`;
			const opt = (s: string) =>
				tensorMap.has(p(s)) ? this.makeTensor(tensorMap, p(s)) : null;
			// Per-layer ropeFreqs assignment. Gemma 4 SWA layers must not
			// apply freq_factors (they weren't trained with it). For other
			// architectures with rope_freqs (Llama 3.1), slidingWindowPattern
			// is absent and every layer gets the shared tensor.
			const isSwaLayer = hp.slidingWindowPattern?.[i] === true;
			const layerRopeFreqs =
				ropeFreqsGlobal && !isSwaLayer ? ropeFreqsGlobal : null;
			if (isPhi3) {
				// Phi-3: fused QKV + fused gate-up. The forward graph slices
				// the fused outputs via opView3d / opView2d. Per-layer norms
				// may carry an optional bias (RMSNorm + bias) on some Phi-3
				// variants; Phi-3.5-mini does not have norm biases so the
				// opt() calls return null for this specific model.
				layers.push({
					attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
					attnNormBias: opt("attn_norm.bias"),
					qkvFused: this.makeTensor(tensorMap, p("attn_qkv.weight")),
					qProj: null,
					kProj: null,
					vProj: null,
					qBias: null,
					kBias: null,
					vBias: null,
					qNorm: null,
					kNorm: null,
					oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
					ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
					ffnNormBias: opt("ffn_norm.bias"),
					gateUpFused: this.makeTensor(tensorMap, p("ffn_up.weight")),
					gateProj: null,
					upProj: null,
					downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
					postAttentionNorm: null,
					postFfwNorm: null,
					pleInpGate: null,
					plePerBlockProj: null,
					plePostNorm: null,
					layerOutputScale: null,
					ropeFreqs: layerRopeFreqs,
				});
			} else {
				// Default split-QKV / split-gate-up path used by llama / qwen* /
				// mistral / etc. Existing behaviour preserved exactly; only
				// nulls were added for phi3-only fields.
				layers.push({
					attnNorm: this.makeTensor(tensorMap, p("attn_norm.weight")),
					attnNormBias: null,
					qkvFused: null,
					qProj: this.makeTensor(tensorMap, p("attn_q.weight")),
					kProj: this.makeTensor(tensorMap, p("attn_k.weight")),
					vProj: this.makeTensor(tensorMap, p("attn_v.weight")),
					qBias: opt("attn_q.bias"),
					kBias: opt("attn_k.bias"),
					vBias: opt("attn_v.bias"),
					qNorm: opt("attn_q_norm.weight"),
					kNorm: opt("attn_k_norm.weight"),
					oProj: this.makeTensor(tensorMap, p("attn_output.weight")),
					ffnNorm: this.makeTensor(tensorMap, p("ffn_norm.weight")),
					ffnNormBias: null,
					gateUpFused: null,
					gateProj: this.makeTensor(tensorMap, p("ffn_gate.weight")),
					upProj: this.makeTensor(tensorMap, p("ffn_up.weight")),
					downProj: this.makeTensor(tensorMap, p("ffn_down.weight")),
					postAttentionNorm: opt("post_attention_norm.weight"),
					postFfwNorm: opt("post_ffw_norm.weight"),
					pleInpGate: opt("inp_gate.weight"),
					plePerBlockProj: opt("proj.weight"),
					plePostNorm: opt("post_norm.weight"),
					layerOutputScale: opt("layer_output_scale.weight"),
					ropeFreqs: layerRopeFreqs,
				});
			}
		}

		this.weights = {
			tokEmb,
			norm,
			normBias,
			output,
			outputBias,
			perLayerEmbed,
			perLayerProj,
			perLayerProjNorm,
			layers,
		};
		this.weightBuf = wasm.backendAllocCtxTensors();

		for (const t of ggufCtx.tensors) {
			const tensor = this.nameToTensor.get(t.name);
			if (!tensor) continue;
			const srcOffset = ggufCtx.dataOffset + t.offset;
			const nbytes = wasm.tensorNbytes(tensor);
			const needsBf16Cast = this.bf16OverriddenNames.has(t.name);
			if (needsBf16Cast) {
				// One-shot BF16 → F32 conversion. The source `bf16Bytes` view
				// (potentially backed by WASM heap) is read entirely during
				// `bf16BytesToF32Bytes`, which only does JS-side ops — no WASM
				// calls between derivation and read, so the view can't detach
				// mid-conversion. The resulting `f32Bytes` lives on the JS
				// heap; the subsequent `uploadToTensorChunked` scratch malloc
				// only invalidates the now-unused BF16 view.
				const srcNbytes = nbytes / 2;
				const bf16Bytes = dataAt(srcOffset, srcNbytes);
				const f32Bytes = bf16BytesToF32Bytes(bf16Bytes);
				wasm.uploadToTensorChunked(tensor, f32Bytes);
				continue;
			}
			if (isCallback) {
				wasm.uploadRangeChunked(
					tensor,
					(off, len) => dataAt(srcOffset + off, len),
					nbytes,
				);
			} else {
				wasm.uploadToTensorChunked(tensor, dataAt(srcOffset, nbytes));
			}
		}
	}

	initKVCache(maxContextLength: number): void {
		if (this.kvLayers) return;

		const { hp, wasm } = this;
		// KV cache dtype:
		// - FA mode: F16 (2 bytes/elem). Required for ggml-webgpu's
		//   FLASH_ATTN_EXT VEC/TILE paths to engage — see
		//   flash_attn_get_decisions::kv_vec_type_supported in
		//   ggml-webgpu-shader-lib.hpp. opCpy handles F32→F16 on KV writes;
		//   FA reads F16 directly. The §6 -7.7% short-context measurement
		//   was made BEFORE FA could engage and no longer applies.
		// - Manual mode: F32 (4 bytes/elem). Legacy mul_mat-friendly default.
		const kvElemBytes = this.flashAttn ? 2 : 4;
		// For mixed-head-dim architectures (Gemma 4), KV cache slots are sized
		// for the worst-case (largest) head_dim so any layer can use the buffer.
		const maxHeadDim = hp.embeddingHeadLengthPerLayer
			? Math.max(...hp.embeddingHeadLengthPerLayer)
			: hp.embeddingHeadLength;
		const perLayerBytes = maxHeadDim * maxContextLength * kvElemBytes;
		// Shared-KV layers (Gemma 4) don't allocate their own K/V — they share
		// tensor handles with the source layer. Count only owning layers when
		// sizing the ctx pool.
		const owningLayers = hp.kvReuseFromLayer
			? hp.kvReuseFromLayer.filter((r) => r === null).length
			: hp.layerCount;
		const totalBytes = owningLayers * 2 * perLayerBytes;
		const memSize = hp.layerCount * 2 * 16384 + totalBytes + (1 << 20);

		wasm.ctxCreate(memSize);

		// FA mode requires F16 K + F16 V (per §18 commit baad612 +
		// flash_attn_get_decisions::kv_vec_type_supported). Manual mode
		// keeps the legacy F32 layout. opCpy converts F32→F16 on writes.
		const kvType = this.flashAttn ? GgmlType.F16 : GgmlType.F32;
		this.kvLayers = [];
		for (let i = 0; i < hp.layerCount; i++) {
			// Shared-KV layer (Gemma 4): point at the source layer's tensors.
			// The iSWA remap (model-loader.ts) guarantees the source layer has
			// matching head_dim/RoPE so the views constructed at the read site
			// are shape-compatible. No new tensors allocated — saves ~20 layers'
			// worth of K+V on Gemma 4 E2B (~480 MiB at maxCtx=4096, F16).
			const reuse = hp.kvReuseFromLayer?.[i] ?? null;
			if (reuse !== null) {
				const src = this.kvLayers[reuse];
				if (!src) {
					throw new Error(
						`initKVCache: layer ${i} reuses layer ${reuse} but source not yet pushed`,
					);
				}
				this.kvLayers.push({ k: src.k, v: src.v });
				continue;
			}
			// Per-layer head_dim for mixed-head-dim architectures (Gemma 4).
			// Non-uniform models (SWA vs global) allocate each layer's KV
			// tensors at the actual head_dim for that layer.
			const layerHeadDim = hp.embeddingHeadLengthPerLayer
				? hp.embeddingHeadLengthPerLayer[i]
				: hp.embeddingHeadLength;
			this.kvLayers.push({
				// K: [headDim, maxCtx, nKvHeads] — same layout in both modes;
				// dtype tracks FA mode.
				k: wasm.tensorNew3d(
					kvType,
					layerHeadDim,
					maxContextLength,
					hp.headCountKv,
				),
				// V layout depends on FA mode:
				// - FA mode:     [headDim, maxCtx, nKvHeads] (FA-ready)
				// - Manual mode: [maxCtx, headDim, nKvHeads] (mul_mat compat)
				v: this.flashAttn
					? wasm.tensorNew3d(
							kvType,
							layerHeadDim,
							maxContextLength,
							hp.headCountKv,
						)
					: wasm.tensorNew3d(
							kvType,
							maxContextLength,
							layerHeadDim,
							hp.headCountKv,
						),
			});
		}

		this.kvBuf = wasm.backendAllocCtxTensors();
		this.nCached = 0;
		this.kvMaxContextLength = maxContextLength;
	}

	resetKVCache(): void {
		this.nCached = 0;
	}

	/**
	 * Roll the logical KV cache size back to `n`. Used after partial accept
	 * in speculative decoding when only some of K drafted tokens are kept.
	 *
	 * The physical KV-buffer slots remain in place — they're overwritten on
	 * the next forward at the rolled-back position.
	 */
	truncateKVCache(n: number): void {
		if (n < 0) throw new Error(`truncateKVCache: n=${n} must be >= 0`);
		if (n > this.nCached) {
			throw new Error(
				`truncateKVCache: n=${n} > current nCached=${this.nCached}; cannot grow via truncate.`,
			);
		}
		this.nCached = n;
	}

	get cachedTokenCount(): number {
		return this.nCached;
	}

	/**
	 * Maximum tokens this KV cache was sized for at `initKVCache`. Returns 0
	 * if the cache hasn't been initialized.
	 */
	get maxContextLength(): number {
		return this.kvMaxContextLength;
	}

	/**
	 * Bytes of K + V per token, summed across all layers. Used by the
	 * conversation pool to size snapshot allocations. Requires FA mode.
	 */
	get kvBytesPerToken(): number {
		if (!this.flashAttn) {
			throw new Error("kvBytesPerToken requires FA mode (flashAttn=true)");
		}
		const { hp } = this;
		const elem = 2; // FA mode = F16
		// 2 = K + V per layer; FA layout is symmetric.
		// For mixed-head-dim architectures, sum the actual per-layer headDims.
		if (hp.embeddingHeadLengthPerLayer) {
			return (
				2 *
				hp.embeddingHeadLengthPerLayer.reduce((s, d) => s + d, 0) *
				hp.headCountKv *
				elem
			);
		}
		return hp.layerCount * 2 * hp.embeddingHeadLength * hp.headCountKv * elem;
	}

	/**
	 * Serialize positions [0, nTokens) of every layer's K and V into a flat
	 * Uint8Array. Layout: [layer0.K | layer0.V | layer1.K | layer1.V | ...]
	 * with each per-layer block dense at `nTokens` slots:
	 * `headDim × nTokens × nKvHeads × elemBytes`.
	 *
	 * Requires FA mode (`flashAttn === true`). Manual-mode V has
	 * `[maxCtx, headDim, nKvHeads]` layout so per-position bytes aren't
	 * contiguous; manual-mode snapshotting is not needed for any production
	 * path (FA is the 7B+ default per §24).
	 *
	 * Async because KV tensors are backend-allocated (WebGPU); their host
	 * memory at `tensor->data` is not readable. Reads route through the
	 * request-based async API (`beginDownloadFromTensor` → `wait` →
	 * `finish`). All `2 × layerCount` readbacks are queued via the sync
	 * `_backend_tensor_get_async_begin` upfront before any await — the
	 * GPU can pipeline them in parallel rather than per-tensor sequential
	 * round-trips. Each request's `finish()` awaits its own completion
	 * and returns the bytes; we do per-head slab compaction in between.
	 * Cuts cost by ~10× vs awaiting each tensor individually (72
	 * round-trips → 1 batched begin + 72 finishes).
	 *
	 * Per-head strided reads probed 2026-05-02: cut payload from full
	 * maxCtx-sized reads to only `[0, nTokens)` per head slab (576 MB
	 * → 195 MB on Qwen3-8B at 1325 shared tokens, 4096 maxCtx).
	 * Wall-time effect was a wash on the interleaved probe
	 * (2719 → 2736 ms); per-call overhead at 576 strided reads
	 * cancelled the bandwidth savings vs 72 full reads. The readback
	 * isn't actually the bandwidth bottleneck — Phase 1a already
	 * hides it behind the WebGPU command queue. Negative result
	 * documented in TODO §11.
	 */
	async serializeKVCache(nTokens: number): Promise<Uint8Array> {
		if (!this.flashAttn) {
			throw new Error("serializeKVCache requires FA mode (flashAttn=true)");
		}
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		if (nTokens < 0 || nTokens > this.nCached) {
			throw new Error(
				`serializeKVCache: nTokens=${nTokens} out of range [0, ${this.nCached}]`,
			);
		}
		const { hp, wasm } = this;
		const elem = 2; // FA mode = F16
		const maxCtx = this.kvMaxContextLength;
		// NOTE (Stage 3): for mixed-head-dim architectures (Gemma 4), this
		// assumes uniform head_dim across all layers — incorrect for SWA vs
		// global layers. Per-layer byte accounting is a Stage 6+ follow-up;
		// serializeKVCache/loadKVCache are not exercised by the Stage 3 smoke
		// probe (conversation pool is not activated for Gemma 4 yet).
		const perHeadFullBytes = hp.embeddingHeadLength * maxCtx * elem;
		const perHeadPopBytes = hp.embeddingHeadLength * nTokens * elem;
		const perLayerOutBytes = hp.headCountKv * perHeadPopBytes;
		const total = hp.layerCount * 2 * perLayerOutBytes;
		const out = new Uint8Array(total);
		const fullKBytes = wasm.tensorNbytes(this.kvLayers[0].k); // == fullVBytes in FA

		// Phase 1: pipeline-launch all readbacks. Each call queues a GPU
		// readback command. Under JSPI the queue dispatch is awaited per
		// call (returns a request with a resolved integer requestId).
		const requests: TensorDownloadRequest[] = [];
		for (let il = 0; il < hp.layerCount; il++) {
			const kv = this.kvLayers[il];
			requests.push(await wasm.beginDownloadFromTensor(kv.k, fullKBytes, 0));
			requests.push(await wasm.beginDownloadFromTensor(kv.v, fullKBytes, 0));
		}

		// Phase 2: finish each in order. The GPU is processing all of them
		// concurrently; we wait on the slowest one once. Per-head slab
		// compaction happens between awaits.
		let outOff = 0;
		try {
			for (const req of requests) {
				const fullBytes = await req.finish();
				for (let h = 0; h < hp.headCountKv; h++) {
					const slabSrc = h * perHeadFullBytes;
					out.set(
						fullBytes.subarray(slabSrc, slabSrc + perHeadPopBytes),
						outOff,
					);
					outOff += perHeadPopBytes;
				}
			}
		} catch (err) {
			// On any failure, cancel remaining outstanding requests so we
			// don't leak heap allocations / WebGPU staging buffers.
			for (const req of requests) {
				try {
					req.cancel();
				} catch {
					// best-effort
				}
			}
			throw err;
		}
		return out;
	}

	/**
	 * Inverse of `serializeKVCache`. Writes positions [0, nTokens) of every
	 * layer's K/V from the supplied buffer; uninitialized positions
	 * [nTokens, maxCtx) are not touched (they're stale, but unused —
	 * forwardSingle writes new positions before reading them).
	 *
	 * `snapshotLen` (default = `nTokens`) is the length the buffer was
	 * serialized at. Must be ≥ `nTokens`. When `snapshotLen > nTokens`,
	 * only the first `nTokens` slots of each per-head slab in `bytes` are
	 * loaded; the remaining `(snapshotLen - nTokens)` slots' bytes are
	 * skipped per slab. This supports loading a shared prefix from a
	 * longer-stored snapshot without re-serializing.
	 *
	 * Async signature retained for callers; internally this method does no
	 * awaits. Writes use `wasm.uploadToTensor`, which is sync but stages
	 * through a transient heap buffer internally. We deliberately avoid
	 * the ASYNCIFY readback that would otherwise dominate cost — see the
	 * comment on the upload loop below for rationale.
	 */
	async loadKVCache(
		bytes: Uint8Array,
		nTokens: number,
		snapshotLen?: number,
	): Promise<void> {
		if (!this.flashAttn) {
			throw new Error("loadKVCache requires FA mode (flashAttn=true)");
		}
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		if (nTokens < 0 || nTokens > this.kvMaxContextLength) {
			throw new Error(
				`loadKVCache: nTokens=${nTokens} out of range [0, ${this.kvMaxContextLength}]`,
			);
		}
		const sl = snapshotLen ?? nTokens;
		if (sl < nTokens) {
			throw new Error(`loadKVCache: snapshotLen=${sl} < nTokens=${nTokens}`);
		}
		const { hp, wasm } = this;
		const elem = 2; // FA mode = F16
		const maxCtx = this.kvMaxContextLength;
		// NOTE (Stage 3): for mixed-head-dim architectures (Gemma 4), this
		// assumes uniform head_dim — incorrect for SWA vs global layers.
		// Per-layer byte accounting deferred to Stage 6+.
		const perHeadFullBytes = hp.embeddingHeadLength * maxCtx * elem;
		const perHeadSnapBytes = hp.embeddingHeadLength * sl * elem;
		const perHeadLoadBytes = hp.embeddingHeadLength * nTokens * elem;
		const perLayerInBytes = hp.headCountKv * perHeadSnapBytes;
		const expected = hp.layerCount * 2 * perLayerInBytes;
		if (bytes.byteLength !== expected) {
			throw new Error(
				`loadKVCache: byte length ${bytes.byteLength} != expected ${expected} (nTokens=${nTokens}, snapshotLen=${sl})`,
			);
		}

		// Batch uploads via `_backend_tensor_set3` (3-at-once primitive).
		// Reduces 72 tensors → 24 ASYNCIFY round-trips on Qwen3-8B. Each
		// round-trip is ~10 ms of overhead independent of payload, so
		// fewer round-trips = proportionally less wall time.
		//
		// Stage buffer holds 3 full-sized tensor regions in the WASM heap
		// (so `_backend_tensor_set3` can read them directly). We populate
		// only the leading `nTokens` slots per head slab; the tail (slots
		// `[nTokens, maxCtx)`) is don't-care because the model only reads
		// up to `nCached` (set to `nTokens` after this method).
		// `inOff` advances by `perHeadSnapBytes` per slab — when
		// `snapshotLen > nTokens` we skip the unused tail of each slab in
		// the source buffer.
		const fullKBytes = wasm.tensorNbytes(this.kvLayers[0].k);
		const tensors: TensorPtr[] = [];
		for (let il = 0; il < hp.layerCount; il++) {
			tensors.push(this.kvLayers[il].k, this.kvLayers[il].v);
		}
		const triBytes = 3 * fullKBytes;
		const triPtr = wasm.malloc(triBytes);
		const fillRegion = (regionPtr: number, srcOff: number): number => {
			for (let h = 0; h < hp.headCountKv; h++) {
				const slabDst = regionPtr + h * perHeadFullBytes;
				wasm.heapU8.set(
					bytes.subarray(srcOff, srcOff + perHeadLoadBytes),
					slabDst,
				);
				srcOff += perHeadSnapBytes;
			}
			return srcOff;
		};
		try {
			let inOff = 0;
			for (let i = 0; i < tensors.length; i += 3) {
				const t1 = tensors[i];
				const t2 = i + 1 < tensors.length ? tensors[i + 1] : 0;
				const t3 = i + 2 < tensors.length ? tensors[i + 2] : 0;
				const p1 = triPtr;
				const p2 = triPtr + fullKBytes;
				const p3 = triPtr + 2 * fullKBytes;
				inOff = fillRegion(p1, inOff);
				if (t2) inOff = fillRegion(p2, inOff);
				if (t3) inOff = fillRegion(p3, inOff);
				wasm.backendTensorSet3(
					t1,
					p1,
					fullKBytes,
					t2,
					t2 ? p2 : 0,
					t2 ? fullKBytes : 0,
					t3,
					t3 ? p3 : 0,
					t3 ? fullKBytes : 0,
				);
			}
		} finally {
			wasm.free(triPtr);
		}
		this.nCached = nTokens;
	}

	/**
	 * Run a forward pass for the given tokens. When `prefillTileSize > 0`
	 * and `tokenIds.length > prefillTileSize`, the call is automatically
	 * chunked into tiles. Returns the **last** position's logits (matching
	 * the legacy single-call contract).
	 *
	 * KV cache accumulates across tiles via the existing
	 * `this.nCached = totalLen` advance at the end of `forwardSingle()`.
	 * Intermediate-tile readbacks are tiny (`vocabSize * 4 B` per tile,
	 * last position only) and are discarded.
	 */
	async forward(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		const tileSize = this.prefillTileSize;
		if (tileSize === 0 || tokenIds.length <= tileSize) {
			return await this.forwardSingle(tokenIds, positions);
		}
		if (tokenIds.length !== positions.length) {
			throw new Error(
				`forward: tokenIds.length (${tokenIds.length}) !== positions.length (${positions.length})`,
			);
		}
		let lastLogits: Float32Array | undefined;
		for (let off = 0; off < tokenIds.length; off += tileSize) {
			const end = Math.min(off + tileSize, tokenIds.length);
			const tileIds = tokenIds.subarray(off, end);
			const tilePos = positions.subarray(off, end);
			lastLogits = await this.forwardSingle(tileIds, tilePos);
		}
		// `lastLogits` is defined because the loop runs at least once
		// (tokenIds.length > tileSize >= 1).
		return lastLogits as Float32Array;
	}

	/**
	 * Build the per-layer QKV projection stage and return reshaped
	 * Q/K/V tensors ready for RoPE.
	 *
	 * Two paths share this helper:
	 * - **Split-QKV** (llama / qwen* / mistral / phi): three matmuls,
	 *   optional bias add, reshape to [headDim, nHeads_q_or_kv, nTokens].
	 * - **Fused-QKV** (phi3): one matmul on `lw.qkvFused`, then 3
	 *   `opView3d` slices over the [E + 2*kvDim, nTokens] output.
	 *   Mirrors the encoder fused path at
	 *   `src/inference/encoder-inference.ts:263-296` and llama.cpp's
	 *   `build_qkv` in `src/llama-graph.cpp`.
	 *
	 * After the split/fused branch the qNorm / kNorm gain is applied
	 * uniformly (Qwen3 family uses these; phi3 doesn't but the field
	 * is null and the branch skips).
	 *
	 * Used by `forwardSingle`, `forwardAllPositions`, `forwardDecode`.
	 * The `debugLayerOutput` path interleaves checkpoints between Q,
	 * K, and V so it cannot share this helper — it has its own inline
	 * split-QKV code that throws if the model is fused.
	 */
	private buildQKV(
		lw: LayerWeights,
		normed: TensorPtr,
		nTokens: number,
		headDim: number,
		nHeads: number,
		nHeadsKv: number = this.hp.headCountKv,
	): { qReady: TensorPtr; kReady: TensorPtr; v3: TensorPtr } {
		const { wasm, hp } = this;
		let q3: TensorPtr;
		let k3: TensorPtr;
		let v3: TensorPtr;
		if (lw.qkvFused) {
			// Phi-3 fused QKV: one matmul → 3 opView3d slices. No GQA on
			// Phi-3.5-mini (kvDim == E); the math handles GQA-shrunk K/V
			// correctly for any future Phi-3 variants that ship GQA.
			const E = hp.embeddingLength;
			const kvDim = headDim * nHeadsKv;
			const fusedRowDim = E + 2 * kvDim;
			const qkv = wasm.opMulMat(lw.qkvFused, normed); // [fusedRowDim, nTokens]
			const headBytes = F32_BYTES * headDim;
			const tokenBytes = F32_BYTES * fusedRowDim;
			q3 = wasm.opCont(
				wasm.opView3d(qkv, headDim, nHeads, nTokens, headBytes, tokenBytes, 0),
			);
			k3 = wasm.opCont(
				wasm.opView3d(
					qkv,
					headDim,
					nHeadsKv,
					nTokens,
					headBytes,
					tokenBytes,
					F32_BYTES * E,
				),
			);
			v3 = wasm.opCont(
				wasm.opView3d(
					qkv,
					headDim,
					nHeadsKv,
					nTokens,
					headBytes,
					tokenBytes,
					F32_BYTES * (E + kvDim),
				),
			);
			if (this.assertFusedContiguity) {
				assertContiguousF32(wasm, q3, "buildQKV.fused.q3");
				assertContiguousF32(wasm, k3, "buildQKV.fused.k3");
				assertContiguousF32(wasm, v3, "buildQKV.fused.v3");
			}
		} else {
			if (!lw.qProj || !lw.kProj || !lw.vProj) {
				throw new Error(
					`split-QKV path requires qProj/kProj/vProj for ${hp.architecture}`,
				);
			}
			const qRaw = wasm.opMulMat(lw.qProj, normed);
			const kRaw = wasm.opMulMat(lw.kProj, normed);
			const vRaw = wasm.opMulMat(lw.vProj, normed);
			const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
			const k = lw.kBias ? wasm.opAdd(kRaw, lw.kBias) : kRaw;
			const v = lw.vBias ? wasm.opAdd(vRaw, lw.vBias) : vRaw;
			q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
			k3 = wasm.opReshape3d(k, headDim, nHeadsKv, nTokens);
			v3 = wasm.opReshape3d(v, headDim, nHeadsKv, nTokens);
		}
		const qReady = lw.qNorm
			? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
			: q3;
		const kReady = lw.kNorm
			? wasm.opMul(wasm.opRmsNorm(k3, hp.normEpsilon), lw.kNorm)
			: k3;
		// Gemma 4 applies bare RMSNorm (no gain) to V before attention
		// (gemma4.cpp:221: ggml_rms_norm(Vcur, f_norm_rms_eps)). No other
		// architecture in the fleet does this — Q/K-only normed elsewhere.
		const vReady =
			hp.architecture === "gemma4" ? wasm.opRmsNorm(v3, hp.normEpsilon) : v3;
		return { qReady, kReady, v3: vReady };
	}

	/**
	 * Gemma soft-cap: `tanh(x / cap) * cap`. Used for attention logit
	 * soft-cap (Gemma 2 pre-softmax) and final-logit soft-cap (Gemma 2
	 * post-lm_head). The fused FA shader applies attention soft-cap
	 * natively via its `logit_softcap` arg — this helper is for the
	 * manual softmax path and the post-lm_head wrap.
	 */
	private softCap(x: TensorPtr, cap: number): TensorPtr {
		const wasm = this.wasm;
		return wasm.opScale(wasm.opTanh(wasm.opScale(x, 1 / cap)), cap);
	}

	/**
	 * Build only the Q projection (no K/V) for shared-KV layers (Gemma 4
	 * E2B layers 15-34). Mirrors the Q-only path of `buildQKV` for split-
	 * projection architectures. Fused-QKV (Phi-3) does not share KV so we
	 * throw rather than slicing out the unused K/V from a single matmul.
	 */
	private buildQOnly(
		lw: LayerWeights,
		normed: TensorPtr,
		nTokens: number,
		headDim: number,
		nHeads: number,
	): TensorPtr {
		const { wasm, hp } = this;
		if (lw.qkvFused) {
			throw new Error(
				"buildQOnly: fused QKV with shared-KV layer is not supported (no current model needs this)",
			);
		}
		if (!lw.qProj) {
			throw new Error(
				`buildQOnly: split-QKV path requires qProj for ${hp.architecture}`,
			);
		}
		const qRaw = wasm.opMulMat(lw.qProj, normed);
		const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
		const q3 = wasm.opReshape3d(q, headDim, nHeads, nTokens);
		return lw.qNorm
			? wasm.opMul(wasm.opRmsNorm(q3, hp.normEpsilon), lw.qNorm)
			: q3;
	}

	/**
	 * Apply RoPE with optional per-dim freq_factors weight. Gemma 4 globals
	 * and Llama 3.1 ship a `rope_freqs.weight` tensor that ggml_rope_ext
	 * applies as a per-dim divisor to theta (ops.cpp:5633: `ff =
	 * freq_factors ? freq_factors[i0/2] : 1.0f`). For Gemma 4, the loader
	 * assigns `lw.ropeFreqs` only on non-SWA layers (per
	 * `hp.slidingWindowPattern`); all other paths pass through to the
	 * plain `opRope`.
	 */
	private applyRope(
		x: TensorPtr,
		posTensor: TensorPtr,
		lw: LayerWeights,
		ropeDimCount: number,
		ropeMode: number,
		ropeFreqBase: number,
	): TensorPtr {
		const { wasm, hp } = this;
		if (lw.ropeFreqs !== null) {
			return wasm.opRopeWithFreqs(
				x,
				posTensor,
				lw.ropeFreqs,
				ropeDimCount,
				ropeMode,
				hp.contextLength,
				ropeFreqBase,
				hp.ropeScale,
				0.0,
				1.0,
				0.0,
				0.0,
			);
		}
		return wasm.opRope(
			x,
			posTensor,
			ropeDimCount,
			ropeMode,
			hp.contextLength,
			ropeFreqBase,
			hp.ropeScale,
			0.0,
			1.0,
			0.0,
			0.0,
		);
	}

	/**
	 * Build the per-layer FFN gate / up projections.
	 *
	 * - **Split** (llama / qwen* / mistral / phi): two matmuls.
	 * - **Fused gate-up** (phi3): one matmul on `lw.gateUpFused`,
	 *   then 2 `opView2d` slices over the [2*ffSize, nTokens] output.
	 *   Mirrors llama.cpp's `LLM_FFN_SWIGLU` mode where the up
	 *   tensor is [2*ffSize, n_embd] and gets split into halves
	 *   before the SwiGLU multiply.
	 */
	private buildFFNGateUp(
		lw: LayerWeights,
		ffnNormed: TensorPtr,
		nTokens: number,
		ffnDim?: number,
	): { gate: TensorPtr; up: TensorPtr } {
		const { wasm, hp } = this;
		if (lw.gateUpFused) {
			// llama.cpp's swiglu kernel (ggml-cpu/ops.cpp:3170-3179)
			// computes y = silu(first_half) * second_half when swapped=0.
			// HF Phi3MLP forward is `up * silu(gate)` with `chunk(2, dim=-1)`,
			// so HF puts gate first / up second along the output dim, and
			// llama.cpp's convert_hf_to_gguf.py Phi3MiniModel preserves
			// that order — gate is the FIRST half, up is the SECOND.
			const ffSize = ffnDim ?? hp.feedForwardLength;
			const fused = wasm.opMulMat(lw.gateUpFused, ffnNormed); // [2*ffSize, nTokens]
			const tokenBytes = F32_BYTES * 2 * ffSize;
			const gate = wasm.opCont(
				wasm.opView2d(fused, ffSize, nTokens, tokenBytes, 0),
			);
			const up = wasm.opCont(
				wasm.opView2d(fused, ffSize, nTokens, tokenBytes, F32_BYTES * ffSize),
			);
			if (this.assertFusedContiguity) {
				assertContiguousF32(wasm, gate, "buildFFNGateUp.fused.gate");
				assertContiguousF32(wasm, up, "buildFFNGateUp.fused.up");
			}
			return { gate, up };
		}
		if (!lw.gateProj || !lw.upProj) {
			throw new Error(
				`split-gate-up path requires gateProj/upProj for ${hp.architecture}`,
			);
		}
		return {
			gate: wasm.opMulMat(lw.gateProj, ffnNormed),
			up: wasm.opMulMat(lw.upProj, ffnNormed),
		};
	}

	/**
	 * Pre-loop Per-Layer Embedding (PLE) projection chain for Gemma 4 (Gemma 3N).
	 * Produces inp_per_layer with shape [pleDim, n_tokens, layerCount] which
	 * is consumed per-block via slice + gated injection inside the layer loop
	 * (separate concern from this helper).
	 *
	 * Returns null for non-Gemma-4 architectures (PLE tensors absent).
	 *
	 * Canonical reference: ~/Repos/llama.cpp/src/models/gemma3n.cpp
	 *   build_inp_per_layer() + project_per_layer_inputs()
	 */
	private buildPreLoopPle(
		tokenIdsTensor: TensorPtr,
		inpBatch: TensorPtr,
		nTokens: number,
	): TensorPtr | null {
		const { hp, wasm, weights } = this;
		if (!weights) return null;
		if (
			!weights.perLayerEmbed ||
			!weights.perLayerProj ||
			!weights.perLayerProjNorm ||
			!hp.pleDim
		) {
			return null;
		}
		const pleDim = hp.pleDim;
		const layerCount = hp.layerCount;
		const hiddenDim = hp.embeddingLength;
		const tokEmbdScale = Math.sqrt(pleDim);
		const perLayerProjectionScale = 1.0 / Math.sqrt(hiddenDim);
		const perLayerInputScale = 1.0 / Math.sqrt(2);

		// Steps 1-3: build_inp_per_layer()
		// GET_ROWS(per_layer_tok_embd, tokens) -> [pleDim*layerCount, n_tokens]
		let inpPerLayer = wasm.opGetRows(weights.perLayerEmbed, tokenIdsTensor);
		// reshape to [pleDim, layerCount, n_tokens]
		inpPerLayer = wasm.opReshape3d(inpPerLayer, pleDim, layerCount, nTokens);
		// scale by sqrt(pleDim)
		inpPerLayer = wasm.opScale(inpPerLayer, tokEmbdScale);

		// Steps 4-7: project_per_layer_inputs()
		// MUL_MAT(per_layer_model_proj, inp_batch) -> [pleDim*layerCount, n_tokens]
		let perLayerProjTensor = wasm.opMulMat(weights.perLayerProj, inpBatch);
		perLayerProjTensor = wasm.opScale(
			perLayerProjTensor,
			perLayerProjectionScale,
		);
		perLayerProjTensor = wasm.opReshape3d(
			perLayerProjTensor,
			pleDim,
			layerCount,
			nTokens,
		);
		// RMSNorm + per_layer_proj_norm gain
		perLayerProjTensor = wasm.opMul(
			wasm.opRmsNorm(perLayerProjTensor, hp.normEpsilon),
			weights.perLayerProjNorm,
		);

		// Steps 8-9: ADD + scale by 1/sqrt(2)
		inpPerLayer = wasm.opAdd(perLayerProjTensor, inpPerLayer);
		inpPerLayer = wasm.opScale(inpPerLayer, perLayerInputScale);

		// Step 10: permute to [pleDim, n_tokens, layerCount], materialize via cont
		inpPerLayer = wasm.opCont(wasm.opPermute(inpPerLayer, 0, 2, 1, 3));
		return inpPerLayer;
	}

	/**
	 * Per-block gated PLE injection for Gemma 4. Slices `inpPerLayer` at the
	 * current layer's slot, gates via per-block inp_gate + GELU, multiplies by
	 * the slice, projects back to hidden via per_layer_proj, RMSNorms with
	 * post_norm, and adds to the residual.
	 *
	 * Returns the new residual. Reference: gemma4.cpp:328-353
	 * (build_lora_mm(W,x) -> opMulMat; build_norm(x, gain, _, RMS, _) -> opMul(opRmsNorm(...), gain)).
	 *
	 * Callers should verify lw.pleInpGate / lw.plePerBlockProj / lw.plePostNorm
	 * are non-null AND inpPerLayer is non-null for performance; the helper
	 * returns `cur` unchanged if any precondition fails (defense in depth).
	 */
	private injectPerBlockPle(
		lw: LayerWeights,
		cur: TensorPtr,
		inpPerLayer: TensorPtr,
		layerIdx: number,
		nTokens: number,
	): TensorPtr {
		const { hp, wasm } = this;
		// Non-null contracts checked at call site; early-return if pleDim is absent.
		if (
			!lw.pleInpGate ||
			!lw.plePerBlockProj ||
			!lw.plePostNorm ||
			!hp.pleDim
		) {
			return cur;
		}
		const pleDim = hp.pleDim;

		// Save residual for the final add.
		const peIn = cur;

		// Gating: GELU(MUL_MAT(inp_gate, cur)) -> [pleDim, n_tokens]
		let gated = wasm.opMulMat(lw.pleInpGate, cur);
		gated = wasm.opGelu(gated);

		// Slice inpPerLayer at slot layerIdx along dim 2 -> [pleDim, n_tokens].
		// inpPerLayer is F32, shape [pleDim, n_tokens, layerCount] after buildPreLoopPle.
		const rowBytes = F32_BYTES * pleDim;
		const sliceBytes = F32_BYTES * pleDim * nTokens;
		const inpThisLayer = wasm.opView2d(
			inpPerLayer,
			pleDim,
			nTokens,
			rowBytes,
			layerIdx * sliceBytes,
		);

		// gated * slice -> [pleDim, n_tokens]
		let proj = wasm.opMul(gated, inpThisLayer);

		// Project back to hidden via per_layer_proj -> [n_embd, n_tokens]
		proj = wasm.opMulMat(lw.plePerBlockProj, proj);

		// post_norm: RMSNorm + gain -> [n_embd, n_tokens]
		proj = wasm.opMul(wasm.opRmsNorm(proj, hp.normEpsilon), lw.plePostNorm);

		// Residual add.
		return wasm.opAdd(peIn, proj);
	}

	/**
	 * Build the forward-pass computation graph (ctxCreate → op sequence →
	 * `graphBuildForwardExpand`) without executing it. The caller is
	 * responsible for `backendAllocCtxTensors`, `uploadLeaves`,
	 * `graphCompute`, readback, and teardown.
	 *
	 * This is the shared graph-construction entry point for the forward-pass
	 * variants. B1 implemented the `mode: "standard"` + `output: "logits"`
	 * path (forwardSingle); B2 adds `output: "decode"` (forwardDecode); B3
	 * delegates forwardAllPositions onto the same `output: "logits"` path
	 * (no new output kind — the pre-B3 body was a verbatim copy of
	 * forwardSingle's); B4 adds `mode: "embedding"` + `output: "embedding"`
	 * (forwardForEmbedding); B5 will add `taps`.
	 *
	 * The op sequence is byte-identical to the pre-B1 `forwardSingle` body —
	 * a mechanical extraction. Do not reorder, fuse, or "optimize" ops; the
	 * graph structure is a load-bearing parity invariant.
	 */
	private async buildForwardGraph(
		opts: BuildForwardGraphOpts,
	): Promise<BuiltForwardGraph> {
		if (!this.weights) throw new Error("Weights not loaded");
		// Embedding mode (B4) is stateless — it does not read or write the KV
		// cache. The guard below is load-bearing for TS narrowing in the
		// standard/taps layer loops; in practice kvLayers is always initialized
		// alongside weights, so the check is a no-op on the embedding path.
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm, weights } = this;
		const {
			mode,
			nTokens,
			pastLen,
			totalLen,
			ropeMode,
			needsMask,
			maskPaddedCols,
			needsSwaMask,
			trace,
		} = opts;

		// 64 bytes/elem covers long-prefill metadata on 7B+ (32x prior was
		// too tight at seq=512 on Mistral 7B).
		const { ctxBytes: graphMem, nodeCount } = computeGraphSizing(
			hp,
			totalLen,
			mode,
		);
		wasm.ctxCreate(graphMem);

		const t1 = trace ? performance.now() : 0;

		// Leaf input tensors — data is uploaded below, AFTER backendAllocCtxTensors
		// assigns real GPU buffers. ctxCreate uses no_alloc=true so tensor->data
		// is null until then; tensorSetData (memcpy to tensor->data) would be a no-op.
		const posTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);
		const tokenIdsTensor = wasm.tensorNew1d(GgmlType.I32, nTokens);

		// Causal attention mask tensor: [totalLen, nTokens]. Values are 0 for
		// visible positions and -Infinity for masked. ggml_soft_max_ext consumes
		// this alongside the pre-scale multiplier.
		//
		// For single-token decode (nTokens == 1) the mask is trivially all
		// zeros — the sole query sees every prior key — so we can skip creating
		// and uploading the tensor entirely and pass a null mask to
		// soft_max_ext. Saves one tensor alloc + one `backendTensorSet` per
		// decode step.
		//
		// Sliding-window (SWA) models (Gemma 2/3/4) carry a SECOND mask tensor
		// of the same shape that holds a banded windowed-causal mask. Per-layer
		// dispatch picks `swaMaskTensor` for layers with
		// `hp.slidingWindowPattern[il] === true`, otherwise `maskTensor`.
		// FA requires F16 mask (ggml.c:5330); opSoftMaxExt accepts F16 too,
		// so this works for both attention paths. Causal mask values are
		// written as F16 bit patterns: 0x0000 = 0.0, 0xFC00 = -Inf.
		const maskTensor = needsMask
			? wasm.tensorNew2d(GgmlType.F16, totalLen, maskPaddedCols)
			: 0;
		const swaMaskTensor = needsSwaMask
			? wasm.tensorNew2d(GgmlType.F16, totalLen, maskPaddedCols)
			: 0;

		// Embedding lookup: get_rows handles Q4_0→F32 dequant (opCpy does not)
		let x = wasm.opGetRows(weights.tokEmb, tokenIdsTensor);
		// Gemma family scales the residual stream by sqrt(embedding_length)
		// immediately after token-embedding lookup (gemma2.cpp:70, gemma4.cpp:149).
		// Llama/Qwen/Mistral/Phi do not — gated on the architecture predicate.
		if (isGemmaFamily(hp.architecture)) {
			x = wasm.opScale(x, Math.sqrt(hp.embeddingLength));
		}

		// Create the graph up front so each layer can expand its KV-cache writes
		// into it *before* attention ops that read kv.k / kv.v are added. Without
		// this ordering, the cpy (write) and the view (read) have no dependency
		// edge, so attention reads stale data (zeros).
		// +32 over the prior 128 covers the ~10 PLE projection nodes (buildPreLoopPle).
		// +8/layer over the prior 64 covers the 8 per-block PLE injection nodes (injectPerBlockPle).
		const graph = wasm.graphNew(nodeCount);

		// Pre-loop PLE projection chain (Gemma 4 only; null for all other models).
		// Keep the PLE chain in the graph; per-block slices consume this result in the layer loop below.
		const inpPerLayer = this.buildPreLoopPle(tokenIdsTensor, x, nTokens);
		if (inpPerLayer !== null) {
			wasm.graphBuildForwardExpand(graph, inpPerLayer);
		}

		let cur = x;
		if (mode === "embedding") {
			// B4: embedding layer loop + tail — byte-identical to the pre-B4
			// forwardForEmbedding body (mechanical move + N→nTokens rename).
			// Stateless manual attention (no KV cache writes, no FA), no PLE
			// injection, no SWA mask dispatch, no post-attention/post-FFW norms,
			// no layerOutputScale, 5-arg buildQKV (nHeadsKv default). Diverges
			// from the standard loop at 10 points; kept as a separate branch so
			// both paths preserve their exact pre-B4 op sequence. Architecture
			// rationale: qwen3.cpp res->t_embd = cur after final RMSNorm,
			// before lm_head. See forwardForEmbedding for the full contract.
			const nHeads = hp.headCount;
			for (let il = 0; il < hp.layerCount; il++) {
				const lw = weights.layers[il];

				// Per-layer scalars for mixed-head-dim architectures (Gemma 4).
				const headDim = hp.embeddingHeadLengthPerLayer
					? hp.embeddingHeadLengthPerLayer[il]
					: hp.embeddingHeadLength;
				const ropeFreqBase = hp.ropeFreqBasePerLayer
					? hp.ropeFreqBasePerLayer[il]
					: hp.ropeFreqBase;
				const ropeDimCount = hp.ropeDimensionCountPerLayer
					? hp.ropeDimensionCountPerLayer[il]
					: headDim;
				const ffnDim = hp.feedForwardLengthPerLayer
					? hp.feedForwardLengthPerLayer[il]
					: undefined;

				let normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);
				if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

				const { qReady, kReady, v3 } = this.buildQKV(
					lw,
					normed,
					nTokens,
					headDim,
					nHeads,
				);

				const qRope = this.applyRope(
					qReady,
					posTensor,
					lw,
					ropeDimCount,
					ropeMode,
					ropeFreqBase,
				);
				const kRope = this.applyRope(
					kReady,
					posTensor,
					lw,
					ropeDimCount,
					ropeMode,
					ropeFreqBase,
				);

				// Manual attention chain — mirrors CausalLMEmbedder.
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
				const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

				const qk = wasm.opMulMat(kp, qp);
				// Gemma 2 attention logit soft-cap: scale-first ordering
				// (gemma2.cpp:110 + ggml-cpu/ops.cpp:8232-8305). See
				// forwardSingle for the longer rationale.
				let qkProcessed = qk;
				let softmaxScale = attnSoftmaxScale(hp, headDim);
				if (hp.attnLogitSoftcap) {
					qkProcessed = this.softCap(
						wasm.opScale(qk, softmaxScale),
						hp.attnLogitSoftcap,
					);
					softmaxScale = 1.0;
				}
				const attnW = wasm.opSoftMaxExt(
					qkProcessed,
					maskTensor,
					softmaxScale,
					0.0,
				);
				const attnOut = wasm.opMulMat(vp, attnW);
				const merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					nTokens,
				);

				// This embedding tap path skips Gemma's post-attention / post-FFW
				// norms because no embedder GGUF in the registry carries them; if
				// a Gemma-family model is ever registered with embeddingCapable=true,
				// wire postAttentionNorm / postFfwNorm here to match forwardSingle.
				const oProj = wasm.opMulMat(lw.oProj, merged);
				const attnResidual = wasm.opAdd(oProj, cur);

				let ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);

				const { gate, up } = this.buildFFNGateUp(
					lw,
					ffnNormed,
					nTokens,
					ffnDim,
				);
				// Gemma family: gelu(gate) * up instead of silu(gate) * up
				// (gemma{,2,3,4}.cpp LLM_FFN_GELU + LLM_FFN_PAR).
				const ffnHidden = isGemmaFamily(hp.architecture)
					? wasm.opMul(wasm.opGelu(gate), up)
					: wasm.opSwigluSplit(gate, up);
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);

				cur = wasm.opAdd(ffnOut, attnResidual);
			}

			// Final output_norm — TAP POINT. No lm_head; no sampling.
			let finalHidden = wasm.opMul(
				wasm.opRmsNorm(cur, hp.normEpsilon),
				weights.norm,
			);
			if (weights.normBias)
				finalHidden = wasm.opAdd(finalHidden, weights.normBias);

			wasm.graphBuildForwardExpand(graph, finalHidden);

			const t2 = trace ? performance.now() : 0;

			return {
				graph,
				outputTensor: finalHidden,
				secondaryTensor: 0,
				posTensor,
				tokenIdsTensor,
				maskTensor,
				swaMaskTensor,
				t1,
				t2,
				embeddingTapTensor: 0,
				layerTapsTensors: [],
				finalHiddenTensor: 0,
			};
		}
		if (mode === "taps") {
			// B5: taps layer loop + tail — byte-identical to the pre-B5
			// forwardWithLayerTaps graph construction (mechanical move +
			// N→nTokens rename). Stateless manual attention (no KV cache
			// writes, no FA) to match the HF transformers reference numerical
			// path. Shared-KV layers (Gemma 4 E2B) reuse earlier-layer K/V via
			// per-layer arrays (no persistent cache). Per-layer tap capture
			// (skip / single-layer / full modes) + embedding-output tap +
			// final-norm tap + lm_head. lastTokenLogitsOnly slices the lm_head
			// input to the last-token column (128 MiB per-binding cap management).
			//
			// Diverges from the standard loop at: (1) K/V source — per-layer
			// arrays instead of persistent cache writes; (2) attention — manual
			// only (no FA branch); (3) per-layer tap capture. The divergence is
			// structural, not cosmetic — kept as a separate branch so both paths
			// preserve their exact pre-B5 op sequence. See forwardWithLayerTaps
			// for the full contract.
			const E = hp.embeddingLength;
			const skipLayerTaps = opts.taps?.skipLayerTaps === true;
			const captureLayer = opts.taps?.captureLayer;
			const singleLayerCapture =
				typeof captureLayer === "number" &&
				captureLayer >= 0 &&
				captureLayer < hp.layerCount;
			const lastTokenLogitsOnly = opts.taps?.lastTokenLogitsOnly === true;
			const captureEmbeddingTap = !lastTokenLogitsOnly;

			// Embedding-output tap (HF `hidden_states[0]` equivalent).
			// Read back here to compare against the reference's pre-block-0
			// state; pins down whether drift starts at embedding lookup +
			// scale or only inside the per-block forward path. Skipped in
			// `lastTokenLogitsOnly` (long-context parity) mode — saves
			// nTokens*E*F32 of pinned buffer that the allocator can otherwise
			// reclaim after block 0 consumes the embedding output.
			const embeddingTap = x;
			if (captureEmbeddingTap) {
				wasm.graphBuildForwardExpand(graph, embeddingTap);
			}

			const layerTaps: TensorPtr[] = [];
			// Shared-KV layers (Gemma 4 E2B layers 15-34) reuse the K/V
			// computed by an earlier layer in the SAME forward pass (this
			// path has no persistent cache). Save each owning layer's
			// post-RoPE kRope and v3 so shared layers can reference them.
			const kRopePerLayer: TensorPtr[] = new Array(hp.layerCount).fill(0);
			const v3PerLayer: TensorPtr[] = new Array(hp.layerCount).fill(0);
			for (let il = 0; il < hp.layerCount; il++) {
				const lw = weights.layers[il];

				const headDim = hp.embeddingHeadLengthPerLayer
					? hp.embeddingHeadLengthPerLayer[il]
					: hp.embeddingHeadLength;
				const nHeads = hp.headCountPerLayer
					? hp.headCountPerLayer[il]
					: hp.headCount;
				const nHeadsKv = hp.headCountKvPerLayer
					? hp.headCountKvPerLayer[il]
					: hp.headCountKv;
				const ropeFreqBase = hp.ropeFreqBasePerLayer
					? hp.ropeFreqBasePerLayer[il]
					: hp.ropeFreqBase;
				const ropeDimCount = hp.ropeDimensionCountPerLayer
					? hp.ropeDimensionCountPerLayer[il]
					: headDim;
				const ffnDim = hp.feedForwardLengthPerLayer
					? hp.feedForwardLengthPerLayer[il]
					: undefined;
				const kvReuse = hp.kvReuseFromLayer?.[il] ?? null;
				const isShared = kvReuse !== null;

				// Per-layer mask: SWA layers use the windowed mask, non-SWA use
				// the full-causal mask. Falls through to 0 when neither mask
				// was materialized (single-token paths; not reachable here at nTokens>1).
				const isSwaLayer = hp.slidingWindowPattern?.[il] === true;
				const layerMask =
					isSwaLayer && swaMaskTensor !== 0
						? swaMaskTensor
						: needsMask
							? maskTensor
							: 0;

				let normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);
				if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

				let qReady: TensorPtr;
				let kRope: TensorPtr;
				let v3: TensorPtr;
				if (isShared) {
					qReady = this.buildQOnly(lw, normed, nTokens, headDim, nHeads);
					// Reuse the source layer's post-RoPE K and pre-permute V.
					// Source layer ran earlier in this loop; tensors are still
					// graph nodes that will be evaluated together at compute.
					kRope = kRopePerLayer[kvReuse];
					v3 = v3PerLayer[kvReuse];
					if (!kRope || !v3) {
						throw new Error(
							`forwardWithLayerTaps: layer ${il} reuses layer ${kvReuse} but source K/V not yet materialized`,
						);
					}
				} else {
					const qkv = this.buildQKV(
						lw,
						normed,
						nTokens,
						headDim,
						nHeads,
						nHeadsKv,
					);
					qReady = qkv.qReady;
					v3 = qkv.v3;
					kRope = this.applyRope(
						qkv.kReady,
						posTensor,
						lw,
						ropeDimCount,
						ropeMode,
						ropeFreqBase,
					);
					kRopePerLayer[il] = kRope;
					v3PerLayer[il] = v3;
				}

				const qRope = this.applyRope(
					qReady,
					posTensor,
					lw,
					ropeDimCount,
					ropeMode,
					ropeFreqBase,
				);

				// Manual attention chain (no KV cache, no FA) — matches HF
				// transformers reference's numerical path for clean diff.
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				const kp = wasm.opPermute(kRope, 0, 2, 1, 3);
				const vp = wasm.opCont(wasm.opPermute(v3, 1, 2, 0, 3));

				const qk = wasm.opMulMat(kp, qp);
				// Gemma 2 attention logit soft-cap: scale-first ordering
				// (gemma2.cpp:110 + ggml-cpu/ops.cpp:8232-8305). See
				// forwardSingle for the longer rationale.
				let qkProcessed = qk;
				let softmaxScale = attnSoftmaxScale(hp, headDim);
				if (hp.attnLogitSoftcap) {
					qkProcessed = this.softCap(
						wasm.opScale(qk, softmaxScale),
						hp.attnLogitSoftcap,
					);
					softmaxScale = 1.0;
				}
				const attnW = wasm.opSoftMaxExt(
					qkProcessed,
					layerMask,
					softmaxScale,
					0.0,
				);
				const attnOut = wasm.opMulMat(vp, attnW);
				const merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					nTokens,
				);

				const oProjRaw = wasm.opMulMat(lw.oProj, merged);
				const oProj = lw.postAttentionNorm
					? wasm.opMul(
							wasm.opRmsNorm(oProjRaw, hp.normEpsilon),
							lw.postAttentionNorm,
						)
					: oProjRaw;
				const attnResidual = wasm.opAdd(oProj, cur);

				let ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);
				const { gate, up } = this.buildFFNGateUp(
					lw,
					ffnNormed,
					nTokens,
					ffnDim,
				);
				// Gemma family: gelu(gate) * up instead of silu(gate) * up
				// (gemma{,2,3,4}.cpp LLM_FFN_GELU + LLM_FFN_PAR).
				const ffnHidden = isGemmaFamily(hp.architecture)
					? wasm.opMul(wasm.opGelu(gate), up)
					: wasm.opSwigluSplit(gate, up);
				const ffnOutRaw = wasm.opMulMat(lw.downProj, ffnHidden);
				const ffnOut = lw.postFfwNorm
					? wasm.opMul(
							wasm.opRmsNorm(ffnOutRaw, hp.normEpsilon),
							lw.postFfwNorm,
						)
					: ffnOutRaw;

				cur = wasm.opAdd(ffnOut, attnResidual);
				if (
					inpPerLayer !== null &&
					lw.pleInpGate &&
					lw.plePerBlockProj &&
					lw.plePostNorm
				) {
					cur = this.injectPerBlockPle(lw, cur, inpPerLayer, il, nTokens);
				}
				if (lw.layerOutputScale) {
					cur = wasm.opMul(cur, lw.layerOutputScale);
				}

				// Tap point: ensure this tensor survives buffer alloc + compute
				// so we can read back its contents below. Skipped when
				// `skipLayerTaps` is set — used for long-context probes where
				// the 35 simultaneously-live tap buffers exceed the WebGPU
				// per-binding 128 MiB cap (each tap is nTokens*E*F32 bytes; at
				// nTokens=560 / E=1536 / L=35 that's 120 MB just for tap retention).
				// Final hidden + logits are still captured.
				//
				// `captureLayer` narrows this even further: pin ONLY one
				// specific layer's `cur`. Used by the incremental long-context
				// parity harness — call this function once per layer index
				// and the readback below returns a sparse `perLayerResidual`
				// with only that index populated. Other layers' `cur` tensors
				// stay intermediates and the allocator can pack them tightly.
				if (skipLayerTaps) {
					// no taps at all
				} else if (singleLayerCapture) {
					if (il === captureLayer) {
						// Slice last-token column of `cur` into a fresh tensor.
						// Pinning only the 1*E sliced copy lets the allocator
						// reuse the full nTokens*E*F32 `cur` (3.4 MB at nTokens=560) after
						// the next block consumes it. The opCont copies the
						// view's bytes into a fresh storage so the parent can
						// be lifetime-packed independently.
						const lastTokenView = wasm.opView2d(
							cur,
							E,
							1,
							E * F32_BYTES,
							(nTokens - 1) * E * F32_BYTES,
						);
						const lastTokenCur = wasm.opCont(lastTokenView);
						wasm.graphBuildForwardExpand(graph, lastTokenCur);
						layerTaps[il] = lastTokenCur;
					} else {
						layerTaps[il] = 0;
					}
				} else {
					wasm.graphBuildForwardExpand(graph, cur);
					layerTaps.push(cur);
				}
			}
			// Early termination was attempted but caused the GPU buffer to
			// be under-sized by the graph allocator. Disabled — single-
			// layer capture runs the full stack and pins only the target
			// layer's tap. The per-binding cap is managed via the other
			// shavings (sliced lm_head input, skipped embedding tap, etc.).
			const earlyBreak = false;

			// Final-norm + lm_head pinned only when we ran the full stack.
			// In `singleLayerCapture` we break out early after the target
			// layer's tap, so `cur` reflects that layer's output (not the
			// final block) and finalHidden/logits would be garbage. Set
			// them to 0 here and skip the readbacks below.
			let finalHidden: TensorPtr = 0;
			let logits: TensorPtr = 0;
			if (!earlyBreak) {
				let finalHiddenFull = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					weights.norm,
				);
				if (weights.normBias)
					finalHiddenFull = wasm.opAdd(finalHiddenFull, weights.normBias);

				// When the caller only needs the last-token row of the
				// final-norm hidden + logits (the parity probe), slice once
				// and feed both lm_head AND the readback pin from that slice.
				// Saves (nTokens-1)*E*F32 on the finalHidden pin AND (nTokens-1)*V*F32
				// on the logits pin (575 MB at nTokens=560/V=262144). The math is
				// unchanged: lm_head is a pointwise matmul over the residual
				// dim.
				const lmHeadInput = lastTokenLogitsOnly
					? wasm.opView2d(
							finalHiddenFull,
							E,
							1,
							E * F32_BYTES,
							(nTokens - 1) * E * F32_BYTES,
						)
					: finalHiddenFull;
				finalHidden = lastTokenLogitsOnly ? lmHeadInput : finalHiddenFull;
				logits = weights.output
					? wasm.opMulMat(weights.output, lmHeadInput)
					: wasm.opMulMat(weights.tokEmb, lmHeadInput);
				if (weights.outputBias) logits = wasm.opAdd(logits, weights.outputBias);
				// Gemma 2 / Gemma 4 post-lm_head soft-cap.
				if (hp.finalLogitSoftcap)
					logits = this.softCap(logits, hp.finalLogitSoftcap);

				wasm.graphBuildForwardExpand(graph, finalHidden);
				wasm.graphBuildForwardExpand(graph, logits);
			}

			const t2 = trace ? performance.now() : 0;

			return {
				graph,
				outputTensor: logits,
				secondaryTensor: 0,
				posTensor,
				tokenIdsTensor,
				maskTensor,
				swaMaskTensor,
				t1,
				t2,
				embeddingTapTensor: captureEmbeddingTap ? embeddingTap : 0,
				layerTapsTensors: layerTaps,
				finalHiddenTensor: finalHidden,
			};
		}
		for (let il = 0; il < hp.layerCount; il++) {
			const lw = weights.layers[il];
			const kv = this.kvLayers[il];

			// Per-layer head_dim, head_count, head_count_kv, RoPE dim, RoPE freq_base for mixed architectures
			// (Gemma 4: SWA layers use 256 head_dim / 8 heads / 1e4 freq; global use 512 / 16 heads / 1e6).
			// Fallback to scalar fields for uniform architectures (no behavioral delta).
			const headDim = hp.embeddingHeadLengthPerLayer
				? hp.embeddingHeadLengthPerLayer[il]
				: hp.embeddingHeadLength;
			const nHeads = hp.headCountPerLayer
				? hp.headCountPerLayer[il]
				: hp.headCount;
			const nHeadsKv = hp.headCountKvPerLayer
				? hp.headCountKvPerLayer[il]
				: hp.headCountKv;
			const ropeFreqBase = hp.ropeFreqBasePerLayer
				? hp.ropeFreqBasePerLayer[il]
				: hp.ropeFreqBase;
			const ropeDimCount = hp.ropeDimensionCountPerLayer
				? hp.ropeDimensionCountPerLayer[il]
				: headDim;
			const ffnDim = hp.feedForwardLengthPerLayer
				? hp.feedForwardLengthPerLayer[il]
				: undefined;
			// Shared-KV layer (Gemma 4 E2B layers 15-34): K/V come from an earlier
			// layer's cache slot. Skip own K/V projection + cache write; reads use
			// the same `kv.k`/`kv.v` handles which were aliased to the source
			// layer's tensors at `initKVCache` time.
			const isShared = (hp.kvReuseFromLayer?.[il] ?? null) !== null;

			// Per-layer attention mask: SWA layers use the banded windowed mask,
			// non-SWA layers use the full-causal mask. When `swaMaskTensor` is 0
			// (non-SWA model, or short-decode-step SWA case), fall through to the
			// global mask (which may itself be 0 at single-token decode).
			const isSwaLayer = hp.slidingWindowPattern?.[il] === true;
			const layerMask =
				isSwaLayer && swaMaskTensor !== 0
					? swaMaskTensor
					: needsMask
						? maskTensor
						: 0;

			// LLaMA RMSNorm: (x / rms(x)) * gamma. ggml_rms_norm only does the
			// normalize step — the per-dim gain `attn_norm.weight` must be applied
			// separately. Same for `ffn_norm.weight` and the final `output_norm.weight`.
			let normed = wasm.opMul(wasm.opRmsNorm(cur, hp.normEpsilon), lw.attnNorm);
			if (lw.attnNormBias) normed = wasm.opAdd(normed, lw.attnNormBias);

			let qReady: TensorPtr;
			let kReadyOwn: TensorPtr | null = null;
			let v3Own: TensorPtr | null = null;
			if (isShared) {
				qReady = this.buildQOnly(lw, normed, nTokens, headDim, nHeads);
			} else {
				const qkv = this.buildQKV(
					lw,
					normed,
					nTokens,
					headDim,
					nHeads,
					nHeadsKv,
				);
				qReady = qkv.qReady;
				kReadyOwn = qkv.kReady;
				v3Own = qkv.v3;
			}

			const qRope = this.applyRope(
				qReady,
				posTensor,
				lw,
				ropeDimCount,
				ropeMode,
				ropeFreqBase,
			);

			const kNb1 = wasm.tensorNb(kv.k, 1);
			const kNb2 = wasm.tensorNb(kv.k, 2);
			const vNb1 = wasm.tensorNb(kv.v, 1);
			const vNb2 = wasm.tensorNb(kv.v, 2);
			if (!isShared) {
				const kRope = this.applyRope(
					kReadyOwn as TensorPtr,
					posTensor,
					lw,
					ropeDimCount,
					ropeMode,
					ropeFreqBase,
				);
				const v3 = v3Own as TensorPtr;
				// Write K to cache: permute kRope(0,2,1,3) -> [headDim, nTokens, nKvHeads]
				const kWriteView = wasm.opView3d(
					kv.k,
					headDim,
					nTokens,
					nHeadsKv,
					kNb1,
					kNb2,
					pastLen * kNb1,
				);
				const kRopeP = wasm.opPermute(kRope, 0, 2, 1, 3);
				// ggml_cpy handles non-contiguous src via strides, so the previously
				// mandatory opCont is redundant — dropping it saves a GPU dispatch per
				// layer per forward.
				const kWrite = wasm.opCpy(kRopeP, kWriteView);
				// Expand into the graph NOW so the cpy node precedes attention reads.
				wasm.graphBuildForwardExpand(graph, kWrite);

				// V cache layout depends on this.flashAttn:
				// - FA mode:     [headDim, maxCtx, nKvHeads]  (FA-ready)
				// - Manual mode: [maxCtx, headDim, nKvHeads]  (mul_mat compat)
				let v3P: TensorPtr;
				let vWriteView: TensorPtr;
				if (this.flashAttn) {
					// src(headDim=0, nKvHeads=1, nTokens=2) -> dst(headDim=0, nTokens=1, nKvHeads=2): permute (0, 2, 1, 3).
					v3P = wasm.opPermute(v3, 0, 2, 1, 3);
					vWriteView = wasm.opView3d(
						kv.v,
						headDim,
						nTokens,
						nHeadsKv,
						vNb1,
						vNb2,
						pastLen * vNb1,
					);
				} else {
					// src(headDim=0, nKvHeads=1, nTokens=2) -> dst(nTokens=0, headDim=1, nKvHeads=2): permute (1, 2, 0, 3).
					const vNb0 = wasm.tensorNb(kv.v, 0);
					v3P = wasm.opPermute(v3, 1, 2, 0, 3);
					vWriteView = wasm.opView3d(
						kv.v,
						nTokens,
						headDim,
						nHeadsKv,
						vNb1,
						vNb2,
						pastLen * vNb0,
					);
				}
				// opCpy handles strided src — opCont skipped (see K write above).
				const vWrite = wasm.opCpy(v3P, vWriteView);
				wasm.graphBuildForwardExpand(graph, vWrite);
			}

			// Read K from cache: [headDim, totalLen, nKvHeads]. For shared
			// layers, kv.k aliases the source layer's K tensor; the source's
			// kWrite was expanded earlier in this same forward pass.
			const fullK = wasm.opView3d(
				kv.k,
				headDim,
				totalLen,
				nHeadsKv,
				kNb1,
				kNb2,
				0,
			);
			// Read V from cache. FA mode wants [headDim, totalLen, nKvHeads]
			// with stride_v1 = headDim (in elements). Manual mode wants
			// [totalLen, headDim, nKvHeads] for opMulMat(V, attn).
			const fullV = this.flashAttn
				? wasm.opView3d(kv.v, headDim, totalLen, nHeadsKv, vNb1, vNb2, 0)
				: wasm.opView3d(kv.v, totalLen, headDim, nHeadsKv, vNb1, vNb2, 0);

			// Permute Q: [headDim, nHeads, nTokens] -> [headDim, nTokens, nHeads]
			const qp = wasm.opPermute(qRope, 0, 2, 1, 3);

			let merged: TensorPtr;
			if (this.flashAttn) {
				// Fused FA: Q=[headDim, nTokens, nHeads], K/V=[headDim, totalLen, nKvHeads]
				// -> [headDim, nHeads, nTokens]. The ggml-webgpu backend picks
				// VEC/TILE shader if F16 K/V (KV is F32 in this plan; backend
				// supports F32 K/V too — see flash_attn_get_decisions).
				const attnOut = wasm.opFlashAttn(
					qp,
					fullK,
					fullV,
					layerMask,
					attnSoftmaxScale(hp, headDim),
					0.0, // max_bias (ALiBi disabled)
					// FA logit_softcap is the *attention* softcap (Gemma 2 only).
					// The fused FA shader implements `v = logit_softcap * tanh(v / logit_softcap)`
					// natively when logit_softcap != 0 (ggml-wgsl-shaders.hpp:2002, :2712).
					// Gemma 4 sets f_attention_scale = 1.0f with no attention softcap; its
					// f_final_logit_softcapping = 30.0 belongs AFTER lm_head, not here
					// (gemma4.cpp:11 vs :379). Gemma 2 sets attn_logit_softcapping = 50.0.
					hp.attnLogitSoftcap ?? 0.0,
				);
				// FA returns contiguous [headDim, nHeads, nTokens] — reshape
				// directly to [embDim, nTokens] for oProj. No permute or
				// opCont needed (this is one of FA's wins).
				merged = wasm.opReshape2d(attnOut, nHeads * headDim, nTokens);
			} else {
				// Manual chain: QK^T -> scaled+masked softmax -> V * attn.
				// QK^T: K=[headDim, totalLen, nKvHeads], Q=[headDim, nTokens, nHeads]
				//       -> [totalLen, nTokens, nHeads].
				const qk = wasm.opMulMat(fullK, qp);
				// Gemma 2 attention logit soft-cap. Reference order
				// (gemma2.cpp:110 + ggml-cpu/ops.cpp:8232-8305): scale qk
				// first, then softcap, then softmax with scale=1.0.
				// `opSoftMaxExt` applies its `scale` arg *inside* the
				// kernel, so passing softcap(qk) + scale would softmax
				// `softcap(qk) / sqrt(d_k)` — saturating tanh on the
				// sqrt(d_k)-larger raw qk magnitudes and collapsing the
				// attention distribution (gemma-2-2b smoke locked to id
				// 139 whitespace without this swap).
				let qkProcessed = qk;
				let softmaxScale = attnSoftmaxScale(hp, headDim);
				if (hp.attnLogitSoftcap) {
					qkProcessed = this.softCap(
						wasm.opScale(qk, softmaxScale),
						hp.attnLogitSoftcap,
					);
					softmaxScale = 1.0;
				}
				const attnW = wasm.opSoftMaxExt(
					qkProcessed,
					layerMask,
					softmaxScale,
					0.0,
				);
				// V * attn: V=[totalLen, headDim, nKvHeads], attn=[totalLen, nTokens, nHeads]
				//           -> [headDim, nTokens, nHeads].
				const attnOut = wasm.opMulMat(fullV, attnW);
				// Merge heads: [headDim, nTokens, nHeads] -> [headDim, nHeads, nTokens] -> [embDim, nTokens].
				merged = wasm.opReshape2d(
					wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
					nHeads * headDim,
					nTokens,
				);
			}

			const oProjRaw = wasm.opMulMat(lw.oProj, merged);
			// Gemma family post-attention norm: applied to attention output BEFORE the
			// residual add. Other archs leave postAttentionNorm null and skip.
			const oProj = lw.postAttentionNorm
				? wasm.opMul(
						wasm.opRmsNorm(oProjRaw, hp.normEpsilon),
						lw.postAttentionNorm,
					)
				: oProjRaw;
			const attnResidual = wasm.opAdd(oProj, cur);

			let ffnNormed = wasm.opMul(
				wasm.opRmsNorm(attnResidual, hp.normEpsilon),
				lw.ffnNorm,
			);
			if (lw.ffnNormBias) ffnNormed = wasm.opAdd(ffnNormed, lw.ffnNormBias);
			const { gate, up } = this.buildFFNGateUp(lw, ffnNormed, nTokens, ffnDim);
			// SwiGLU: fused silu(gate) * up. Gemma family uses gelu(gate) * up
			// instead (gemma{,2,3,4}.cpp LLM_FFN_GELU + LLM_FFN_PAR).
			const ffnHidden = isGemmaFamily(hp.architecture)
				? wasm.opMul(wasm.opGelu(gate), up)
				: wasm.opSwigluSplit(gate, up);
			const ffnOutRaw = wasm.opMulMat(lw.downProj, ffnHidden);
			// Gemma family post-FFW norm: applied to FFN output BEFORE the residual add.
			const ffnOut = lw.postFfwNorm
				? wasm.opMul(wasm.opRmsNorm(ffnOutRaw, hp.normEpsilon), lw.postFfwNorm)
				: ffnOutRaw;

			cur = wasm.opAdd(ffnOut, attnResidual);
			if (
				inpPerLayer !== null &&
				lw.pleInpGate &&
				lw.plePerBlockProj &&
				lw.plePostNorm
			) {
				cur = this.injectPerBlockPle(lw, cur, inpPerLayer, il, nTokens);
			}
			// Gemma 4: multiply residual by per-layer learned scalar before feeding
			// the next layer (broadcasts [1]-shape weight across all hidden dims).
			if (lw.layerOutputScale) {
				cur = wasm.opMul(cur, lw.layerOutputScale);
			}
		}

		let finalNorm = wasm.opMul(
			wasm.opRmsNorm(cur, hp.normEpsilon),
			weights.norm,
		);
		if (weights.normBias) finalNorm = wasm.opAdd(finalNorm, weights.normBias);
		let logits = weights.output
			? wasm.opMulMat(weights.output, finalNorm)
			: wasm.opMulMat(weights.tokEmb, finalNorm);
		if (weights.outputBias) logits = wasm.opAdd(logits, weights.outputBias);
		// Gemma 2 / Gemma 4 post-lm_head soft-cap: tanh(logits / s) * s.
		if (hp.finalLogitSoftcap)
			logits = this.softCap(logits, hp.finalLogitSoftcap);

		// Terminal output dispatch. The shared body above (ctxCreate → logits)
		// is byte-identical for every output kind; only the tensor expanded
		// into the graph and returned for readback differs.
		let outputTensor: TensorPtr;
		let secondaryTensor: TensorPtr = 0;
		if (opts.output === "logits") {
			wasm.graphBuildForwardExpand(graph, logits);
			outputTensor = logits;
		} else {
			// output === "decode": forwardDecode's argmax/topk/full tail.
			const decodeMode = opts.decodeMode;
			if (decodeMode === "greedy") {
				const argmaxResult = wasm.opArgmax(logits);
				wasm.graphBuildForwardExpand(graph, argmaxResult);
				outputTensor = argmaxResult;
			} else if (decodeMode === "topk" && opts.topK && opts.topK > 0) {
				const topKIndices = wasm.opTopK(logits, opts.topK);
				// `ggml_get_rows(a, b)` produces `[a.ne[0], b.ne[0], ...]` and
				// gathers along `a.ne[1]` (rows). `logits` is `[vocab, 1]` so
				// naively reshaping to `[vocab, 1]` leaves vocab on the inner
				// dim where get_rows can't reach it — the result would be
				// `[vocab, topK]` and the subsequent reshape to `[topK, 1]`
				// trips `GGML_ASSERT(nelements == ne0*ne1)`. Flip so vocab is
				// the row dim: each "row" holds one logit, and get_rows picks
				// the top-K rows.
				const logitsRow = wasm.opReshape2d(logits, 1, hp.vocabularySize);
				const topKValues2D = wasm.opGetRows(logitsRow, topKIndices);
				const topKValues = wasm.opReshape2d(topKValues2D, opts.topK, 1);
				wasm.graphBuildForwardExpand(graph, topKValues);
				outputTensor = topKValues;
				secondaryTensor = topKIndices;
			} else {
				// Fallback: full logits (also reached when mode === "verify"
				// or mode === "topk" with no valid topK).
				wasm.graphBuildForwardExpand(graph, logits);
				outputTensor = logits;
			}
		}

		const t2 = trace ? performance.now() : 0;

		return {
			graph,
			outputTensor,
			secondaryTensor,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			t1,
			t2,
			embeddingTapTensor: 0,
			layerTapsTensors: [],
			finalHiddenTensor: 0,
		};
	}

	/**
	 * Single-call forward pass: one ctxCreate → graph build → graph compute →
	 * ctxFree cycle for the given tokens. **Do not call directly** — public
	 * callers go through `forward()`, which dispatches to either this or the
	 * tiled loop based on `prefillTileSize`.
	 *
	 * Graph construction is delegated to {@link buildForwardGraph}; this method
	 * handles input marshalling, graph compute, readback, and teardown. The
	 * graph build invariants documented in {@link buildForwardGraph} are
	 * load-bearing — see comments around `no_alloc=true` rationale, V-cache
	 * permute axes, and last-position-only readback.
	 */
	private async forwardSingle(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const trace = this.traceEnabled;
		const t0 = trace ? performance.now() : 0;

		// Mask bits and ropeMode — pure computations passed to buildForwardGraph
		// via opts. buildForwardGraph creates the mask tensors after ctxCreate
		// (tensorNew needs the ctx; the bits are ctx-independent).
		const needsMask = nTokens > 1;
		const maskPaddedCols = padTo(nTokens, 32);
		const swaWindow = hp.slidingWindowSize ?? 0;
		const hasSwaLayers =
			(hp.slidingWindowPattern?.some((b) => b) ?? false) && swaWindow > 0;
		const needsSwaMask =
			hasSwaLayers && (nTokens > 1 || pastLen + nTokens > swaWindow);
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const {
			graph,
			outputTensor: logits,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			t1,
			t2,
		} = await this.buildForwardGraph({
			mode: "standard",
			output: "logits",
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			ropeMode,
			needsMask,
			maskPaddedCols,
			swaWindow,
			needsSwaMask,
			trace,
		});

		const graphBuf = wasm.backendAllocCtxTensors();

		const t3 = trace ? performance.now() : 0;

		// Upload leaf input data AFTER backend buffers are assigned. The helper
		// packs pos / tokenIds / mask into a single backendTensorSet3 call to
		// avoid 2-3 separate FFI hops per forward; when SWA layers are active a
		// follow-on backendTensorSet uploads `swaMaskTensor` from the same heap.
		this.uploadLeaves(
			wasm,
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			needsMask,
			maskPaddedCols,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			swaWindow,
		);

		const t4 = trace ? performance.now() : 0;

		const graphProfile = await this.computeGraphWithOptionalProfile(
			trace,
			graph,
		);

		const t5 = trace ? performance.now() : 0;

		const logitsBytes = hp.vocabularySize * 4;
		const offset = nTokens > 1 ? (nTokens - 1) * logitsBytes : 0;
		const resultBuf = await wasm.downloadFromTensor(
			logits,
			logitsBytes,
			offset,
		);
		const result = new Float32Array(
			resultBuf.buffer,
			resultBuf.byteOffset,
			hp.vocabularySize,
		).slice();

		const t6 = trace ? performance.now() : 0;

		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;

		const t7 = trace ? performance.now() : 0;

		if (trace) {
			this.recordTrace(
				"full",
				nTokens,
				pastLen,
				t0,
				t1,
				t2,
				t3,
				t4,
				t5,
				t6,
				t7,
				graphProfile,
			);
		}

		return result;
	}

	/**
	 * Build a single-pass causal forward over `tokenIds` and return the
	 * post-`output_norm` hidden state as a flat `[E * N]` Float32Array
	 * (row-major-reversed `[E, N]`). **Does not touch the KV cache.**
	 *
	 * Mirrors `CausalLMEmbedder.forwardEmbed` but handles every chat
	 * architecture in the fleet (split + fused QKV; split + fused
	 * gate-up). The chat session's `nCached`, `kvLayers`, and
	 * conversation transcript are unchanged after this call.
	 *
	 * Graph construction is delegated to {@link buildForwardGraph} (B4:
	 * `mode: "embedding"` + `output: "embedding"`); this method handles
	 * input marshalling, graph compute, hidden-state readback, and
	 * teardown. The pre-B4 op sequence (stateless manual attention, no
	 * FA, no PLE, no post-attention/post-FFW norms, no lm_head) is
	 * preserved byte-identical inside buildForwardGraph's embedding branch.
	 *
	 * Architecture truth source: `~/Repos/llama.cpp/src/models/qwen3.cpp`
	 * (`res->t_embd = cur` after the final RMSNorm, before `lm_head`).
	 *
	 * Concurrency: the caller must serialize this against any
	 * `forward()` / `forwardSingle()` on the same engine; both paths
	 * share the global WASM ctx-stack.
	 */
	private async forwardForEmbedding(
		tokenIds: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		const { hp, wasm } = this;
		// Shared-KV (Gemma 4 family) requires reusing earlier-layer K/V at
		// shared layers. The stateless embedding path doesn't write K/V to
		// a cache so it can't simply alias tensors — adding shared-KV here
		// follows the `forwardWithLayerTaps` per-layer KV passing pattern,
		// which isn't currently needed (Gemma 4 has `embedding: false`).
		if (hp.kvReuseFromLayer?.some((r) => r !== null)) {
			throw new Error(
				`forwardForEmbedding: shared-KV architecture (${hp.architecture}) is not registered as an embedder. Wire forwardWithLayerTaps-style K/V passing if this changes.`,
			);
		}
		const N = tokenIds.length;
		const E = hp.embeddingLength;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);
		const maskPaddedCols = padTo(N, 32);

		// Positions always start at 0 — embed() processes an isolated sequence
		// with no KV history; applying an offset would shift RoPE relative to
		// the training distribution.
		const positions = new Int32Array(N);
		for (let i = 0; i < N; i++) positions[i] = i;

		// Graph construction is delegated to buildForwardGraph (B4: mode
		// "embedding" + output "embedding"); this method handles input
		// marshalling, graph compute, hidden-state readback, and teardown.
		// The pre-B4 op sequence (manual attention, no FA, no PLE, no
		// post-norms, no lm_head) is preserved byte-identical inside
		// buildForwardGraph's embedding branch. Trace stays off to match the
		// pre-B4 behavior (the embedding path never recorded a trace).
		const {
			graph,
			outputTensor: finalHidden,
			posTensor,
			tokenIdsTensor,
			maskTensor,
		} = await this.buildForwardGraph({
			mode: "embedding",
			output: "embedding",
			tokenIds,
			positions,
			nTokens: N,
			pastLen: 0,
			totalLen: N,
			ropeMode,
			needsMask: true,
			maskPaddedCols,
			swaWindow: 0,
			needsSwaMask: false,
			trace: false,
		});

		const graphBuf = wasm.backendAllocCtxTensors();

		try {
			this.uploadLeaves(
				wasm,
				tokenIds,
				positions,
				N,
				0, // pastLen — embedding is stateless
				N, // totalLen
				true, // needsMask — embedding always creates the causal mask
				maskPaddedCols,
				posTensor,
				tokenIdsTensor,
				maskTensor,
			);

			await wasm.graphCompute(graph);

			const totalFloats = E * N;
			const bytes = await wasm.downloadFromTensor(finalHidden, totalFloats * 4);
			const hidden = new Float32Array(
				bytes.buffer,
				bytes.byteOffset,
				totalFloats,
			);
			return new Float32Array(hidden);
		} finally {
			wasm.backendBufferFree(graphBuf);
			wasm.ctxFree();
		}
	}

	/**
	 * Diagnostic tap-forward: run the full forwardSingle graph (PLE, post-
	 * norms, layerOutputScale, embedding scaling, Gemma GELU, rope_freqs —
	 * everything `forwardSingle` does) over `tokenIds` AS IF processing
	 * them as a fresh prefill (positions 0..N-1, no KV history). Read back
	 * the last-token row of every block's residual stream, the post-
	 * `output_norm` final hidden state, and the top-K logits.
	 *
	 * Used by `smoke-test/parity-capture.html` to compare layer-by-layer
	 * against a HuggingFace `transformers` reference capture (see
	 * `eval/tools/parity-capture/README.md`). Pinpoints the first block
	 * where webllm diverges from canonical fp32 output — the FIRST
	 * divergent block carries the bug.
	 *
	 * **Side-effect free**: does not touch `this.kvLayers` or
	 * `this.nCached`. Uses manual attention (no FA) to match the HF ref
	 * forward's numerical path.
	 *
	 * Output schema matches `eval/tools/parity-capture/README.md`:
	 *   - `perLayerResidual[i]` — `cur` AFTER block i, last-token row,
	 *     post-PLE-injection + post-`layerOutputScale` (i.e., the value
	 *     passed to block i+1).
	 *   - `finalNormHidden` — last-token row after `output_norm`.
	 *   - `logitsTop16` — top-K logits for the last token (K=16).
	 */
	async forwardWithLayerTaps(
		tokenIds: Int32Array,
		opts?: {
			topK?: number;
			skipLayerTaps?: boolean;
			captureLayer?: number;
			lastTokenLogitsOnly?: boolean;
		},
	): Promise<{
		embeddingOutput: Float32Array;
		perLayerResidual: Float32Array[];
		finalNormHidden: Float32Array;
		logitsTop16: { ids: Int32Array; values: Float32Array };
	}> {
		if (!this.weights) throw new Error("Weights not loaded");
		const { hp, wasm } = this;
		const N = tokenIds.length;
		const E = hp.embeddingLength;
		const V = hp.vocabularySize;
		const K = Math.max(1, opts?.topK ?? 16);
		const captureLayer = opts?.captureLayer;
		const singleLayerCapture =
			typeof captureLayer === "number" &&
			captureLayer >= 0 &&
			captureLayer < hp.layerCount;
		const skipLayerTaps = opts?.skipLayerTaps === true;
		const lastTokenLogitsOnly = opts?.lastTokenLogitsOnly === true;
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		// Mask bits and ropeMode — pure computations passed to buildForwardGraph
		// via opts. buildForwardGraph creates the mask tensors after ctxCreate.
		const needsMask = N > 1;
		const maskPaddedCols = padTo(N, 32);
		const swaWindow = hp.slidingWindowSize ?? 0;
		const hasSwaLayers =
			(hp.slidingWindowPattern?.some((b) => b) ?? false) && swaWindow > 0;
		const needsSwaMask = hasSwaLayers && (N > 1 || N > swaWindow);

		// Positions always start at 0 — this is a stateless diagnostic forward
		// (no KV history); applying an offset would shift RoPE relative to the
		// training distribution.
		const positions = new Int32Array(N);
		for (let i = 0; i < N; i++) positions[i] = i;

		// Graph construction is delegated to buildForwardGraph (B5: mode "taps"
		// + output "taps"); this method handles input marshalling, graph compute,
		// multi-tensor readback, and teardown. The pre-B5 op sequence (stateless
		// manual attention, per-layer tap capture, embedding tap, lastTokenLogitsOnly
		// slicing) is preserved byte-identical inside buildForwardGraph's taps
		// branch. Trace stays off to match the pre-B5 behavior (the taps path
		// never recorded a trace).
		const {
			graph,
			outputTensor: logits,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			embeddingTapTensor: embeddingTap,
			layerTapsTensors: layerTaps,
			finalHiddenTensor: finalHidden,
		} = await this.buildForwardGraph({
			mode: "taps",
			output: "taps",
			tokenIds,
			positions,
			nTokens: N,
			pastLen: 0,
			totalLen: N,
			ropeMode,
			needsMask,
			maskPaddedCols,
			swaWindow,
			needsSwaMask,
			trace: false,
			taps: {
				skipLayerTaps,
				captureLayer,
				lastTokenLogitsOnly,
			},
		});

		// earlyBreak mirrors the (disabled) early-termination guard inside
		// buildForwardGraph's taps branch. Always false — preserved for parity
		// with the pre-B5 readback contract.
		const earlyBreak = false;
		const captureEmbeddingTap = !lastTokenLogitsOnly;

		const graphBuf = wasm.backendAllocCtxTensors();

		try {
			this.uploadLeaves(
				wasm,
				tokenIds,
				positions,
				N,
				0, // pastLen — taps is stateless
				N, // totalLen
				needsMask,
				maskPaddedCols,
				posTensor,
				tokenIdsTensor,
				maskTensor,
				swaMaskTensor,
				swaWindow,
			);

			await wasm.graphCompute(graph);

			// Read back: embedding output (pre-block-0), every layer's
			// last-token row, then final-norm last row, then last-token
			// full-vocab logits -> top-K in JS.
			const rowBytes = E * 4;
			const lastTokenOffset = (N - 1) * rowBytes;
			let embeddingOutput: Float32Array;
			if (captureEmbeddingTap) {
				const embedBuf = await wasm.downloadFromTensor(
					embeddingTap,
					rowBytes,
					lastTokenOffset,
				);
				embeddingOutput = new Float32Array(
					embedBuf.buffer,
					embedBuf.byteOffset,
					E,
				).slice();
			} else {
				// embedding tap was not pinned (lastTokenLogitsOnly mode);
				// return an empty array. Callers in that mode don't compare
				// embedding output (per-layer SWA parity check is the focus).
				embeddingOutput = new Float32Array(0);
			}

			// In single-layer-capture mode the `layerTaps` array is sparse
			// (length=layerCount, only `captureLayer` populated). In
			// skipLayerTaps mode it's empty. Readback only the entries
			// the caller pinned; leave the rest as empty Float32Arrays.
			const perLayerResidual: Float32Array[] = [];
			if (skipLayerTaps) {
				// emit empty array; caller asked for no per-layer taps.
			} else if (singleLayerCapture) {
				// The pinned tap is the [E,1] last-token slice; readback
				// offset is 0 (no row stride). Size is one row.
				for (let il = 0; il < hp.layerCount; il++) {
					if (il === captureLayer && layerTaps[il]) {
						const buf = await wasm.downloadFromTensor(
							layerTaps[il],
							rowBytes,
							0,
						);
						perLayerResidual.push(
							new Float32Array(buf.buffer, buf.byteOffset, E).slice(),
						);
					} else {
						perLayerResidual.push(new Float32Array(0));
					}
				}
			} else {
				for (let il = 0; il < hp.layerCount; il++) {
					const buf = await wasm.downloadFromTensor(
						layerTaps[il],
						rowBytes,
						lastTokenOffset,
					);
					perLayerResidual.push(
						new Float32Array(buf.buffer, buf.byteOffset, E).slice(),
					);
				}
			}

			// In single-layer-capture mode we broke out of the per-layer
			// loop early - finalHidden + logits weren't built, so skip
			// their readbacks and return empty placeholders.
			let finalNormHidden = new Float32Array(0);
			let sortedIds = new Int32Array(0);
			let sortedVals = new Float32Array(0);
			if (!earlyBreak) {
				// When lastTokenLogitsOnly is set finalHidden is shape [E, 1]
				// (the last-token slice), so the readback offset is 0.
				const fnBuf = await wasm.downloadFromTensor(
					finalHidden,
					rowBytes,
					lastTokenLogitsOnly ? 0 : lastTokenOffset,
				);
				finalNormHidden = new Float32Array(
					fnBuf.buffer,
					fnBuf.byteOffset,
					E,
				).slice();

				// When lastTokenLogitsOnly is set the logits tensor is shape
				// [V, 1] (last-token sliced before lm_head), so the readback
				// offset is 0 instead of (N-1)*V*4.
				const logitsBytes = V * 4;
				const logitsOffset = lastTokenLogitsOnly ? 0 : (N - 1) * logitsBytes;
				const logitsBuf = await wasm.downloadFromTensor(
					logits,
					logitsBytes,
					logitsOffset,
				);
				const logitsFlat = new Float32Array(
					logitsBuf.buffer,
					logitsBuf.byteOffset,
					V,
				);

				// Top-K via maintained min-of-K. K is small (16); ~V*K compares.
				const topIds = new Int32Array(K).fill(-1);
				const topVals = new Float32Array(K).fill(-Infinity);
				let minIdx = 0;
				let minVal = -Infinity;
				for (let i = 0; i < V; i++) {
					const v = logitsFlat[i];
					if (v > minVal) {
						topIds[minIdx] = i;
						topVals[minIdx] = v;
						// Recompute min-of-K.
						minVal = topVals[0];
						minIdx = 0;
						for (let j = 1; j < K; j++) {
							if (topVals[j] < minVal) {
								minVal = topVals[j];
								minIdx = j;
							}
						}
					}
				}
				// Sort top-K descending by value.
				const order: number[] = [];
				for (let i = 0; i < K; i++) order.push(i);
				order.sort((a, b) => topVals[b] - topVals[a]);
				sortedIds = new Int32Array(K);
				sortedVals = new Float32Array(K);
				for (let i = 0; i < K; i++) {
					sortedIds[i] = topIds[order[i]];
					sortedVals[i] = topVals[order[i]];
				}
			}

			return {
				embeddingOutput,
				perLayerResidual,
				finalNormHidden,
				logitsTop16: { ids: sortedIds, values: sortedVals },
			};
		} finally {
			wasm.backendBufferFree(graphBuf);
			wasm.ctxFree();
		}
	}

	/**
	 * Compute an L2-normalized sentence embedding by running a single-
	 * pass causal forward over `tokenIds`, tapping the post-
	 * `output_norm` hidden state, pooling, and L2-normalizing.
	 * **Does not write to the KV cache** — the chat session's state is
	 * unchanged.
	 *
	 * Pooling (`opts.pooling`, default `"last-token"`):
	 *   - `"last-token"`: take the hidden state at column `N - 1`.
	 *     Canonical bucket D pool; matches qwen3-8b-iq3m ref capture.
	 *   - `"mean"`: average all `N` columns, then L2-normalize. Use on
	 *     models with high last-token anisotropy (e.g., Phi-3.5-mini).
	 *
	 * Concurrency: the caller (typically `engine.embed`) must serialize
	 * this against any concurrent `forward()` / `generate()` on the
	 * same engine. The two paths share the global WASM ctx-stack.
	 */
	async embed(
		tokenIds: Int32Array,
		opts?: { pooling?: "last-token" | "mean" },
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (tokenIds.length === 0) {
			throw new Error("embed() received empty input after tokenization");
		}
		const pooling = opts?.pooling ?? "last-token";
		const hidden = await this.forwardForEmbedding(tokenIds);
		const E = this.hp.embeddingLength;
		const N = tokenIds.length;
		const pooled = new Float32Array(E);
		if (pooling === "mean") {
			for (let col = 0; col < N; col++) {
				const base = col * E;
				for (let i = 0; i < E; i++) pooled[i] += hidden[base + i];
			}
			const invN = 1 / N;
			for (let i = 0; i < E; i++) pooled[i] *= invN;
		} else {
			const lastCol = (N - 1) * E;
			for (let i = 0; i < E; i++) pooled[i] = hidden[lastCol + i];
		}
		let sq = 0;
		for (let i = 0; i < E; i++) sq += pooled[i] * pooled[i];
		if (sq === 0) return pooled;
		const invNorm = 1 / Math.sqrt(sq);
		for (let i = 0; i < E; i++) pooled[i] *= invNorm;
		return pooled;
	}

	/**
	 * Multi-token forward pass that returns logits at **all** input positions.
	 *
	 * Same graph as `forward()` for `nTokens > 1`, but downloads the full logits
	 * tensor instead of slicing the last row. Output is row-major:
	 * `[k * vocabSize + i]` is the logit for the token-after-input-at-position-k,
	 * vocab index i.
	 *
	 * Used by speculative decoding to verify K drafted tokens in one forward.
	 * Caller must `truncateKVCache(pastLen + accepted)` after partial accept.
	 */
	async forwardVerify(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		if (tokenIds.length < 2) {
			throw new Error(
				"forwardVerify requires nTokens >= 2; use forward() for prefill or forwardDecode() for single-step decode.",
			);
		}
		if (tokenIds.length !== positions.length) {
			throw new Error(
				`forwardVerify: tokenIds.length (${tokenIds.length}) !== positions.length (${positions.length})`,
			);
		}
		return this.forwardAllPositions(tokenIds, positions);
	}

	/**
	 * Internal worker: same graph as `forwardSingle` (standard sizing, full
	 * `output: "logits"` path), but the logits readback returns all `nTokens`
	 * rows instead of only the last. Used by speculative decoding's
	 * `forwardVerify` to score K drafted tokens in one forward.
	 *
	 * Graph construction is delegated to {@link buildForwardGraph} (B3);
	 * this method handles input marshalling, graph compute, all-positions
	 * readback, and teardown. The graph build invariants documented in
	 * {@link buildForwardGraph} are load-bearing. The pre-B3 body was a
	 * verbatim copy of `forwardSingle`'s body (itself a verbatim copy of
	 * `forward()`), so the byte-identical parity carries over: the only
	 * deltas from `forwardSingle` are the readback (all rows vs last row)
	 * and the trace label (`"verify"` vs `"full"`).
	 */
	private async forwardAllPositions(
		tokenIds: Int32Array,
		positions: Int32Array,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const trace = this.traceEnabled;
		const t0 = trace ? performance.now() : 0;

		// Mask bits and ropeMode — pure computations passed to buildForwardGraph
		// via opts. buildForwardGraph creates the mask tensors after ctxCreate
		// (tensorNew needs the ctx; the bits are ctx-independent).
		const needsMask = nTokens > 1;
		const maskPaddedCols = padTo(nTokens, 32);
		const swaWindow = hp.slidingWindowSize ?? 0;
		const hasSwaLayers =
			(hp.slidingWindowPattern?.some((b) => b) ?? false) && swaWindow > 0;
		const needsSwaMask =
			hasSwaLayers && (nTokens > 1 || pastLen + nTokens > swaWindow);
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const {
			graph,
			outputTensor: logits,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			t1,
			t2,
		} = await this.buildForwardGraph({
			mode: "standard",
			output: "logits",
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			ropeMode,
			needsMask,
			maskPaddedCols,
			swaWindow,
			needsSwaMask,
			trace,
		});

		const graphBuf = wasm.backendAllocCtxTensors();

		const t3 = trace ? performance.now() : 0;

		// Upload leaf input data AFTER backend buffers are assigned. The helper
		// packs pos / tokenIds / mask into a single backendTensorSet3 call to
		// avoid 2-3 separate FFI hops per forward; when SWA layers are active a
		// follow-on backendTensorSet uploads `swaMaskTensor` from the same heap.
		this.uploadLeaves(
			wasm,
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			needsMask,
			maskPaddedCols,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			swaWindow,
		);

		const t4 = trace ? performance.now() : 0;

		const graphProfile = await this.computeGraphWithOptionalProfile(
			trace,
			graph,
		);

		const t5 = trace ? performance.now() : 0;

		// All-positions readback: download full logits tensor (nTokens rows of
		// vocabSize floats each) starting at offset 0, instead of slicing the
		// last row like forwardSingle does.
		const logitsBytes = hp.vocabularySize * 4;
		const totalBytes = nTokens * logitsBytes;
		const resultBuf = await wasm.downloadFromTensor(logits, totalBytes, 0);
		const result = new Float32Array(
			resultBuf.buffer,
			resultBuf.byteOffset,
			nTokens * hp.vocabularySize,
		).slice();

		const t6 = trace ? performance.now() : 0;

		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;

		const t7 = trace ? performance.now() : 0;

		if (trace) {
			this.recordTrace(
				"verify",
				nTokens,
				pastLen,
				t0,
				t1,
				t2,
				t3,
				t4,
				t5,
				t6,
				t7,
				graphProfile,
			);
		}

		return result;
	}

	/**
	 * Decode a single token using GPU-side reduction (ARGMAX or TOP_K).
	 * Avoids downloading the full vocabulary logits - returns either a single
	 * token ID (greedy) or a reduced set of indices + values (topk).
	 *
	 * @param tokenIds  - Input token IDs (must be length 1 for decode).
	 * @param positions - Corresponding positions.
	 * @param mode      - 'greedy' for ARGMAX, 'topk' for TOP_K + GET_ROWS.
	 * @param topK      - K value when mode is 'topk'.
	 */
	async forwardDecode(
		tokenIds: Int32Array,
		positions: Int32Array,
		mode: DecodeMode,
		topK?: number,
	): Promise<DecodeResult> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { hp, wasm } = this;
		const nTokens = tokenIds.length;
		const pastLen = this.nCached;
		const totalLen = pastLen + nTokens;

		const trace = this.traceEnabled;
		const t0 = trace ? performance.now() : 0;

		// Mask bits and ropeMode — pure computations passed to buildForwardGraph
		// via opts. buildForwardGraph creates the mask tensors after ctxCreate
		// (tensorNew needs the ctx; the bits are ctx-independent).
		const needsMask = nTokens > 1;
		const maskPaddedCols = padTo(nTokens, 32);
		const swaWindow = hp.slidingWindowSize ?? 0;
		const hasSwaLayers =
			(hp.slidingWindowPattern?.some((b) => b) ?? false) && swaWindow > 0;
		const needsSwaMask =
			hasSwaLayers && (nTokens > 1 || pastLen + nTokens > swaWindow);
		const ropeMode = getRopeModeForArchitecture(hp.architecture);

		const {
			graph,
			outputTensor,
			secondaryTensor,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			t1,
			t2,
		} = await this.buildForwardGraph({
			mode: "standard",
			output: "decode",
			decodeMode: mode,
			...(topK !== undefined ? { topK } : {}),
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			ropeMode,
			needsMask,
			maskPaddedCols,
			swaWindow,
			needsSwaMask,
			trace,
		});

		const graphBuf = wasm.backendAllocCtxTensors();

		const t3 = trace ? performance.now() : 0;

		// Upload leaf input data AFTER backend buffers are assigned. The helper
		// packs pos / tokenIds / mask into a single backendTensorSet3 call to
		// avoid 2-3 separate FFI hops per forward; when SWA layers are active a
		// follow-on backendTensorSet uploads `swaMaskTensor` from the same heap.
		this.uploadLeaves(
			wasm,
			tokenIds,
			positions,
			nTokens,
			pastLen,
			totalLen,
			needsMask,
			maskPaddedCols,
			posTensor,
			tokenIdsTensor,
			maskTensor,
			swaMaskTensor,
			swaWindow,
		);

		const t4 = trace ? performance.now() : 0;

		const graphProfile = await this.computeGraphWithOptionalProfile(
			trace,
			graph,
		);

		const t5 = trace ? performance.now() : 0;

		// Readback: which tensors to download depends on the decode mode. The
		// graph tail (argmax/topk/expand) was built inside buildForwardGraph;
		// here we only marshal the downloaded bytes into a DecodeResult.
		let result: DecodeResult;
		let traceMode: DecodeMode;
		if (mode === "greedy") {
			const buf = await wasm.downloadFromTensor(outputTensor, 4, 0);
			const tokenId = new Int32Array(buf.buffer, buf.byteOffset, 1)[0];
			result = { tokenId };
			traceMode = "greedy";
		} else if (mode === "topk" && topK && topK > 0) {
			const kBytes = topK * 4;
			const idxBuf = await wasm.downloadFromTensor(secondaryTensor, kBytes, 0);
			const valBuf = await wasm.downloadFromTensor(outputTensor, kBytes, 0);
			const indices = new Int32Array(
				idxBuf.buffer,
				idxBuf.byteOffset,
				topK,
			).slice();
			const values = new Float32Array(
				valBuf.buffer,
				valBuf.byteOffset,
				topK,
			).slice();
			result = { topKIndices: indices, topKValues: values };
			traceMode = "topk";
		} else {
			// Fallback: full logits (also reached when mode === "verify" or
			// mode === "topk" with no valid topK).
			const logitsBytes = hp.vocabularySize * 4;
			const offset = nTokens > 1 ? (nTokens - 1) * logitsBytes : 0;
			const resultBuf = await wasm.downloadFromTensor(
				outputTensor,
				logitsBytes,
				offset,
			);
			const fullLogits = new Float32Array(
				resultBuf.buffer,
				resultBuf.byteOffset,
				hp.vocabularySize,
			).slice();
			result = { logits: fullLogits };
			traceMode = "full";
		}

		const t6 = trace ? performance.now() : 0;

		wasm.backendBufferFree(graphBuf);
		wasm.ctxFree();
		this.nCached = totalLen;

		const t7 = trace ? performance.now() : 0;

		if (trace) {
			this.recordTrace(
				traceMode,
				nTokens,
				pastLen,
				t0,
				t1,
				t2,
				t3,
				t4,
				t5,
				t6,
				t7,
				graphProfile,
			);
		}

		return result;
	}

	private async computeGraphWithOptionalProfile(
		trace: boolean,
		graph: number,
	): Promise<GraphComputeProfile | null> {
		// Detailed backend profiling is gated behind trace mode so normal
		// inference and non-profile perf runs keep using the representative
		// graphCompute() path.
		if (!trace) {
			await this.wasm.graphCompute(graph);
			return null;
		}
		await this.wasm.graphComputeWithDetailedProfile(graph);
		return this.wasm.getLastGraphComputeProfile();
	}

	private recordTrace(
		mode: DecodeMode,
		nTokens: number,
		pastLen: number,
		t0: number,
		t1: number,
		t2: number,
		t3: number,
		t4: number,
		t5: number,
		t6: number,
		t7: number,
		graphProfile?: GraphComputeProfile | null,
	): void {
		this.lastTrace = {
			mode,
			nTokens,
			pastLen,
			ctxCreateMs: t1 - t0,
			buildGraphMs: t2 - t1,
			backendAllocMs: t3 - t2,
			uploadLeavesMs: t4 - t3,
			graphComputeMs: t5 - t4,
			downloadResultMs: t6 - t5,
			teardownMs: t7 - t6,
			totalMs: t7 - t0,
			backendProfileTotalMs: graphProfile?.totalMs,
			backendMatmulMs: graphProfile?.matmulMs,
			backendAttentionMs: graphProfile?.attentionMs,
			backendEncodeOverheadMs: graphProfile?.encodeOverheadMs,
			backendDispatchCount: graphProfile?.dispatchCount,
			backendBreakdownAvailable: graphProfile?.breakdownAvailable,
		};
	}

	/**
	 * Upload position, token ID, and mask data after backend alloc.
	 *
	 * When `swaMaskTensor !== 0`, a second mask tensor is also populated with
	 * a banded windowed-causal mask of width `swaWindow` for Gemma SWA layers.
	 * The global mask (`maskTensor`) carries the full causal mask consumed by
	 * non-SWA layers. Both masks share the same shape — the caller picks
	 * between them at attention dispatch time based on
	 * `hp.slidingWindowPattern[il]`.
	 */
	private uploadLeaves(
		wasm: GgmlWasm,
		tokenIds: Int32Array,
		positions: Int32Array,
		nTokens: number,
		pastLen: number,
		totalLen: number,
		needsMask: boolean,
		maskPaddedCols: number,
		posTensor: TensorPtr,
		tokenIdsTensor: TensorPtr,
		maskTensor: TensorPtr,
		swaMaskTensor: TensorPtr = 0,
		swaWindow = 0,
	): void {
		const posBytes = nTokens * 4;
		const idsBytes = nTokens * 4;
		const maskBytes = needsMask ? totalLen * maskPaddedCols * 2 : 0;
		const swaMaskBytes =
			swaMaskTensor !== 0 ? totalLen * maskPaddedCols * 2 : 0;
		const totalBytes = posBytes + idsBytes + maskBytes + swaMaskBytes;

		const heap = wasm.malloc(totalBytes);
		try {
			const posPtr = heap;
			const idsPtr = heap + posBytes;
			const maskPtr = heap + posBytes + idsBytes;
			const swaMaskPtr = heap + posBytes + idsBytes + maskBytes;

			const posView = new Int32Array(wasm.heapU8.buffer, posPtr, nTokens);
			const idsView = new Int32Array(wasm.heapU8.buffer, idsPtr, nTokens);
			for (let i = 0; i < nTokens; i++) {
				posView[i] = positions[i];
				idsView[i] = tokenIds[i];
			}

			if (needsMask) {
				const mask = new Uint16Array(
					wasm.heapU8.buffer,
					maskPtr,
					totalLen * maskPaddedCols,
				);
				writeCausalMaskF16(mask, totalLen, nTokens, pastLen, maskPaddedCols);
			}
			if (swaMaskTensor !== 0) {
				const swaMask = new Uint16Array(
					wasm.heapU8.buffer,
					swaMaskPtr,
					totalLen * maskPaddedCols,
				);
				writeCausalMaskF16(
					swaMask,
					totalLen,
					nTokens,
					pastLen,
					maskPaddedCols,
					swaWindow,
				);
			}

			wasm.backendTensorSet3(
				posTensor,
				posPtr,
				posBytes,
				tokenIdsTensor,
				idsPtr,
				idsBytes,
				needsMask ? maskTensor : 0,
				maskPtr,
				maskBytes,
			);
			if (swaMaskTensor !== 0) {
				wasm.backendTensorSet(swaMaskTensor, swaMaskPtr, 0, swaMaskBytes);
			}
		} finally {
			wasm.free(heap);
		}
	}

	// ── Debug tooling ────────────────────────────────────────────────────
	//
	// These helpers sidestep the normal forward() path to read intermediate
	// tensors directly from the GPU. They exist because the inference pipeline
	// has a long, subtle chain of ops where any one broken step poisons every
	// subsequent prediction, and it's very hard to tell from the final logits
	// alone which step broke. Keep them around — the next bug will need them too.
	//
	// Usage from the browser console (see `smoke-test/real-model.html` for the
	// scaffolding that exposes `window.inference` / `window.tokenizer`):
	//
	//   // Raw dequantized embedding row for a token
	//   const x = await window.inference.debugReadEmbeddingRow(1);  // BOS
	//
	//   // K / V cache contents after a forward pass (use a probe prefill first)
	//   window.inference.resetKVCache();
	//   await window.inference.forward(new Int32Array([22172]), new Int32Array([0]));
	//   const k = await window.inference.debugReadKCache(0, 64*4, 0);
	//   const v = await window.inference.debugReadVCache(0, 64*4, 0);
	//
	//   // First N floats of any F32 norm weight after loadWeights()
	//   const g = await window.inference.debugReadNormWeight("attn0", 8);
	//
	//   // Fully run the transformer up through layer L for a single token
	//   // and return the hidden state. Great for "where does it diverge?".
	//   const h = await window.inference.debugLayerOutput(1, /* layerIdx= */ 0);

	/** DEBUG: read back a slice of kv.k for a given layer. */
	async debugReadKCache(
		layerIdx: number,
		nBytes: number,
		offset = 0,
	): Promise<Float32Array> {
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const tensor = this.kvLayers[layerIdx].k;
		const bytes = await this.wasm.downloadFromTensor(tensor, nBytes, offset);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nBytes / 4);
	}

	/** DEBUG: read back a slice of kv.v for a given layer. */
	async debugReadVCache(
		layerIdx: number,
		nBytes: number,
		offset = 0,
	): Promise<Float32Array> {
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const tensor = this.kvLayers[layerIdx].v;
		const bytes = await this.wasm.downloadFromTensor(tensor, nBytes, offset);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nBytes / 4);
	}

	/**
	 * DEBUG: run the transformer stack for a single token and return the
	 * hidden state at a specific checkpoint inside a specific layer.
	 *
	 * checkpoint:
	 *   "layer_output" — cur after layer L (attn residual + ffn residual)
	 *   "post_attn"    — cur after layer L's attn residual add, before FFN
	 *   "pre_attn"     — cur entering layer L (= output of layer L-1)
	 *   "attn_out"     — just oProj(...) for layer L, before residual add
	 *   "ffn_out"      — just ffn_down(...) for layer L, before residual add
	 *   "ffn_gate"     — raw ffn_gate_proj output (pre-silu)
	 *   "ffn_up"       — raw ffn_up_proj output
	 *   "ffn_hidden"   — silu(gate) * up, i.e. the input to ffn_down_proj
	 *   "ffn_normed"   — input to FFN after RMSNorm + ffn_norm gain
	 *   "attn_normed"  — input to attention after RMSNorm + attn_norm gain
	 *   "attn_q"       — raw Q projection output (pre-reshape, pre-rope)
	 *   "attn_k"       — raw K projection output
	 *   "attn_v"       — raw V projection output
	 *
	 * The difference (post_attn - pre_attn) == attn_out, and
	 * (layer_output - post_attn) == ffn_out — so these together pinpoint
	 * which branch of a layer is under- or over-shooting.
	 */
	async debugLayerOutput(
		tokenId: number,
		layerIdx: number,
		checkpoint:
			| "layer_output"
			| "post_attn"
			| "pre_attn"
			| "attn_out"
			| "ffn_out"
			| "ffn_gate"
			| "ffn_up"
			| "ffn_hidden"
			| "ffn_normed"
			| "attn_normed"
			| "attn_q"
			| "attn_k"
			| "attn_v" = "layer_output",
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		if (!this.kvLayers) throw new Error("KV cache not initialized");
		const { wasm, hp, weights } = this;
		// debugLayerOutput interleaves checkpoint breaks between
		// individual Q / K / V matmuls and between gate / up matmuls.
		// Phi-3-class architectures fuse QKV and gate-up into single
		// matmuls so those checkpoints aren't addressable. The split
		// path below assumes per-layer split projections exist.
		if (
			weights.layers.length > 0 &&
			(weights.layers[0].qkvFused !== null ||
				weights.layers[0].gateUpFused !== null)
		) {
			throw new Error(
				`debugLayerOutput is split-QKV / split-gate-up only; ${hp.architecture} uses fused projections so per-Q/K/V or per-gate/up checkpoints are not addressable. Use forwardSingle for end-to-end output instead.`,
			);
		}
		// debugLayerOutput drives checkpoints into the Q/K/V split path
		// per-layer. Shared-KV layers don't own K/V projections, so
		// inspecting attn_k / attn_v there is meaningless. Reject loudly.
		if (hp.kvReuseFromLayer?.[layerIdx] != null) {
			throw new Error(
				`debugLayerOutput: layer ${layerIdx} is a shared-KV layer (reuses layer ${hp.kvReuseFromLayer[layerIdx]}); inspect the source layer instead.`,
			);
		}
		const isFfnIntermediate =
			checkpoint === "ffn_gate" ||
			checkpoint === "ffn_up" ||
			checkpoint === "ffn_hidden";
		const kvDim = hp.embeddingHeadLength * hp.headCountKv;
		const nbytes =
			(isFfnIntermediate
				? hp.feedForwardLength
				: checkpoint === "attn_k" || checkpoint === "attn_v"
					? kvDim
					: hp.embeddingLength) * 4;

		wasm.ctxCreate(16 * 1024 * 1024);
		try {
			const posTensor = wasm.tensorNew1d(GgmlType.I32, 1);
			const idsTensor = wasm.tensorNew1d(GgmlType.I32, 1);
			const maskTensor = wasm.tensorNew2d(GgmlType.F16, 1, 32);
			const ropeMode = getRopeModeForArchitecture(hp.architecture);

			let cur = wasm.opGetRows(weights.tokEmb, idsTensor);
			const graph = wasm.graphNew(2048);

			let returnTensor: TensorPtr = cur;

			for (let il = 0; il <= layerIdx; il++) {
				if (il === layerIdx && checkpoint === "pre_attn") {
					returnTensor = cur;
					break;
				}
				const lw = weights.layers[il];
				const kv = this.kvLayers[il];
				const normed = wasm.opMul(
					wasm.opRmsNorm(cur, hp.normEpsilon),
					lw.attnNorm,
				);
				if (il === layerIdx && checkpoint === "attn_normed") {
					returnTensor = normed;
					break;
				}
				const qRaw = wasm.opMulMat(lw.qProj as number, normed);
				const q = lw.qBias ? wasm.opAdd(qRaw, lw.qBias) : qRaw;
				if (il === layerIdx && checkpoint === "attn_q") {
					returnTensor = q;
					break;
				}
				const kRaw = wasm.opMulMat(lw.kProj as number, normed);
				const k = lw.kBias ? wasm.opAdd(kRaw, lw.kBias) : kRaw;
				if (il === layerIdx && checkpoint === "attn_k") {
					returnTensor = k;
					break;
				}
				const vRaw = wasm.opMulMat(lw.vProj as number, normed);
				const v = lw.vBias ? wasm.opAdd(vRaw, lw.vBias) : vRaw;
				if (il === layerIdx && checkpoint === "attn_v") {
					returnTensor = v;
					break;
				}
				const q3 = wasm.opReshape3d(q, hp.embeddingHeadLength, hp.headCount, 1);
				const k3 = wasm.opReshape3d(
					k,
					hp.embeddingHeadLength,
					hp.headCountKv,
					1,
				);
				const v3 = wasm.opReshape3d(
					v,
					hp.embeddingHeadLength,
					hp.headCountKv,
					1,
				);
				const qRope = wasm.opRope(
					q3,
					posTensor,
					hp.embeddingHeadLength,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0,
					1,
					0,
					0,
				);
				const kRope = wasm.opRope(
					k3,
					posTensor,
					hp.embeddingHeadLength,
					ropeMode,
					hp.contextLength,
					hp.ropeFreqBase,
					hp.ropeScale,
					0,
					1,
					0,
					0,
				);
				const kNb1 = wasm.tensorNb(kv.k, 1);
				const kNb2 = wasm.tensorNb(kv.k, 2);
				const kWriteView = wasm.opView3d(
					kv.k,
					hp.embeddingHeadLength,
					1,
					hp.headCountKv,
					kNb1,
					kNb2,
					0,
				);
				const kRopeP = wasm.opPermute(kRope, 0, 2, 1, 3);
				wasm.graphBuildForwardExpand(
					graph,
					wasm.opCpy(wasm.opCont(kRopeP), kWriteView),
				);
				// V cache write — same dual-layout pattern as forward(); see Task 3.
				// Single-token at offset 0 (no pastLen). Manual mode wraps in
				// opCont because debugLayerOutput's opCpy semantics here differ
				// from forward's (legacy: opCont preserved for shape safety).
				const vNb1 = wasm.tensorNb(kv.v, 1);
				const vNb2 = wasm.tensorNb(kv.v, 2);
				let v3P: TensorPtr;
				let vWriteView: TensorPtr;
				if (this.flashAttn) {
					v3P = wasm.opPermute(v3, 0, 2, 1, 3);
					vWriteView = wasm.opView3d(
						kv.v,
						hp.embeddingHeadLength,
						1,
						hp.headCountKv,
						vNb1,
						vNb2,
						0,
					);
				} else {
					v3P = wasm.opPermute(v3, 1, 2, 0, 3);
					vWriteView = wasm.opView3d(
						kv.v,
						1,
						hp.embeddingHeadLength,
						hp.headCountKv,
						vNb1,
						vNb2,
						0,
					);
				}
				wasm.graphBuildForwardExpand(
					graph,
					wasm.opCpy(wasm.opCont(v3P), vWriteView),
				);
				const fullK = wasm.opView3d(
					kv.k,
					hp.embeddingHeadLength,
					1,
					hp.headCountKv,
					kNb1,
					kNb2,
					0,
				);
				const fullV = this.flashAttn
					? wasm.opView3d(
							kv.v,
							hp.embeddingHeadLength,
							1,
							hp.headCountKv,
							vNb1,
							vNb2,
							0,
						)
					: wasm.opView3d(
							kv.v,
							1,
							hp.embeddingHeadLength,
							hp.headCountKv,
							vNb1,
							vNb2,
							0,
						);
				const qp = wasm.opPermute(qRope, 0, 2, 1, 3);
				let merged: TensorPtr;
				if (this.flashAttn) {
					const attnOut = wasm.opFlashAttn(
						qp,
						fullK,
						fullV,
						maskTensor,
						1 / Math.sqrt(hp.embeddingHeadLength),
						0,
						// Gemma 2 attention logit soft-cap (FA shader applies natively).
						hp.attnLogitSoftcap ?? 0,
					);
					merged = wasm.opReshape2d(
						attnOut,
						hp.headCount * hp.embeddingHeadLength,
						1,
					);
				} else {
					const qk = wasm.opMulMat(fullK, qp);
					// Gemma 2 attention logit soft-cap: scale-first ordering
					// (gemma2.cpp:110 + ggml-cpu/ops.cpp:8232-8305). See
					// forwardSingle for the longer rationale.
					let qkProcessed = qk;
					let softmaxScale = 1 / Math.sqrt(hp.embeddingHeadLength);
					if (hp.attnLogitSoftcap) {
						qkProcessed = this.softCap(
							wasm.opScale(qk, softmaxScale),
							hp.attnLogitSoftcap,
						);
						softmaxScale = 1.0;
					}
					const attnW = wasm.opSoftMaxExt(
						qkProcessed,
						maskTensor,
						softmaxScale,
						0,
					);
					const attnOut = wasm.opMulMat(fullV, attnW);
					merged = wasm.opReshape2d(
						wasm.opCont(wasm.opPermute(attnOut, 0, 2, 1, 3)),
						hp.headCount * hp.embeddingHeadLength,
						1,
					);
				}
				const oProj = wasm.opMulMat(lw.oProj, merged);
				if (il === layerIdx && checkpoint === "attn_out") {
					returnTensor = oProj;
					break;
				}
				const attnResidual = wasm.opAdd(oProj, cur);
				if (il === layerIdx && checkpoint === "post_attn") {
					returnTensor = attnResidual;
					break;
				}
				const ffnNormed = wasm.opMul(
					wasm.opRmsNorm(attnResidual, hp.normEpsilon),
					lw.ffnNorm,
				);
				if (il === layerIdx && checkpoint === "ffn_normed") {
					returnTensor = ffnNormed;
					break;
				}
				const gate = wasm.opMulMat(lw.gateProj as number, ffnNormed);
				if (il === layerIdx && checkpoint === "ffn_gate") {
					returnTensor = gate;
					break;
				}
				const up = wasm.opMulMat(lw.upProj as number, ffnNormed);
				if (il === layerIdx && checkpoint === "ffn_up") {
					returnTensor = up;
					break;
				}
				const ffnHidden = wasm.opSwigluSplit(gate, up);
				if (il === layerIdx && checkpoint === "ffn_hidden") {
					returnTensor = ffnHidden;
					break;
				}
				const ffnOut = wasm.opMulMat(lw.downProj, ffnHidden);
				if (il === layerIdx && checkpoint === "ffn_out") {
					returnTensor = ffnOut;
					break;
				}
				cur = wasm.opAdd(ffnOut, attnResidual);
				if (il === layerIdx && checkpoint === "layer_output") {
					returnTensor = cur;
				}
			}

			wasm.graphBuildForwardExpand(graph, returnTensor);
			const graphBuf = wasm.backendAllocCtxTensors();

			// Upload inputs
			const sp = wasm.stackSave();
			try {
				const posPtr = wasm.stackAlloc(4);
				new Int32Array(wasm.heapU8.buffer, posPtr, 1)[0] = 0;
				wasm.backendTensorSet(posTensor, posPtr, 0, 4);
				const idsPtr = wasm.stackAlloc(4);
				new Int32Array(wasm.heapU8.buffer, idsPtr, 1)[0] = tokenId;
				wasm.backendTensorSet(idsTensor, idsPtr, 0, 4);
			} finally {
				wasm.stackRestore(sp);
			}
			const maskHeap = wasm.malloc(32 * 2);
			try {
				new Uint16Array(wasm.heapU8.buffer, maskHeap, 32).fill(0);
				wasm.backendTensorSet(maskTensor, maskHeap, 0, 32 * 2);
			} finally {
				wasm.free(maskHeap);
			}
			await wasm.graphCompute(graph);

			const bytes = await wasm.downloadFromTensor(returnTensor, nbytes, 0);
			wasm.backendBufferFree(graphBuf);
			return new Float32Array(bytes.buffer, bytes.byteOffset, nbytes / 4);
		} finally {
			wasm.ctxFree();
		}
	}

	/**
	 * DEBUG: dequantize and return the embedding row for a single token by
	 * running a one-op graph: `opGetRows(tokEmb, [tokenId])`. Requires
	 * `initKVCache` to have been called first (needs a ctx on the stack).
	 */
	async debugReadEmbeddingRow(tokenId: number): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		const { wasm, hp, weights } = this;
		const nbytes = hp.embeddingLength * 4;

		wasm.ctxCreate(64 * 1024);
		try {
			const idsTensor = wasm.tensorNew1d(GgmlType.I32, 1);
			const out = wasm.opGetRows(weights.tokEmb, idsTensor);
			const graph = wasm.graphNew(8);
			wasm.graphBuildForwardExpand(graph, out);
			const graphBuf = wasm.backendAllocCtxTensors();

			// Upload the token id.
			const sp = wasm.stackSave();
			try {
				const ptr = wasm.stackAlloc(4);
				new Int32Array(wasm.heapU8.buffer, ptr, 1)[0] = tokenId;
				wasm.backendTensorSet(idsTensor, ptr, 0, 4);
			} finally {
				wasm.stackRestore(sp);
			}

			await wasm.graphCompute(graph);

			const bytes = await wasm.downloadFromTensor(out, nbytes, 0);
			wasm.backendBufferFree(graphBuf);
			return new Float32Array(bytes.buffer, bytes.byteOffset, nbytes / 4);
		} finally {
			wasm.ctxFree();
		}
	}

	/** DEBUG: read back first N floats of an F32 norm weight. */
	async debugReadNormWeight(
		which: "output" | "attn0" | "ffn0",
		nFloats = 8,
	): Promise<Float32Array> {
		if (!this.weights) throw new Error("Weights not loaded");
		let tensor: TensorPtr;
		if (which === "output") tensor = this.weights.norm;
		else if (which === "attn0") tensor = this.weights.layers[0].attnNorm;
		else tensor = this.weights.layers[0].ffnNorm;
		const bytes = await this.wasm.downloadFromTensor(tensor, nFloats * 4, 0);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nFloats);
	}

	async dispose(): Promise<void> {
		if (this.kvBuf) {
			this.wasm.backendBufferFree(this.kvBuf);
			this.kvBuf = 0;
		}
		if (this.weightBuf) {
			this.wasm.backendBufferFree(this.weightBuf);
			this.weightBuf = 0;
		}
		if (this.kvLayers) {
			this.wasm.ctxFree();
			this.kvLayers = null;
		}
		this.wasm.ctxFree();
		this.weights = null;
		this.nameToTensor.clear();
		this.nCached = 0;
	}

	private makeTensor(
		tensorMap: Map<string, GgufTensorInfo>,
		name: string,
	): TensorPtr {
		const info = tensorMap.get(name);
		if (!info) throw new Error(`Weight "${name}" not found in GGUF`);

		const d = info.dimensions;
		// WebGPU has no BF16 support; the mul_mat_f32_bf16 shader fails to
		// compile and silently corrupts dependent kernels. Override BF16 to
		// F32 here and convert the GGUF bytes at upload (see loadWeights).
		const isBf16 = info.type === GgmlType.BF16;
		if (isBf16) {
			this.bf16OverriddenNames.add(name);
		}
		const t: number = isBf16 ? GgmlType.F32 : info.type;
		let tensor: TensorPtr;

		if (d.length === 1) tensor = this.wasm.tensorNew1d(t, d[0]);
		else if (d.length === 2) tensor = this.wasm.tensorNew2d(t, d[0], d[1]);
		else if (d.length === 3)
			tensor = this.wasm.tensorNew3d(t, d[0], d[1], d[2]);
		else tensor = this.wasm.tensorNew4d(t, d[0], d[1], d[2], d[3]);

		this.wasm.tensorSetName(tensor, name);
		this.nameToTensor.set(name, tensor);
		return tensor;
	}
}
