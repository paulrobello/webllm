import type { ModelHyperparams } from "../core/types.js";
import { GgmlType, type GgmlWasm, type TensorPtr } from "./ggml-wasm.js";
import {
	getRopeModeForArchitecture,
	type LayerKVCache,
	type WeightTensors,
} from "./model-inference.js";

/**
 * Narrow read-only view of {@link ModelInference} used by the debug helpers
 * in {@link ModelInferenceDiagnostics}. Exposes only the state the debug
 * methods actually touch — not the full private surface of the inference
 * engine.
 */
export interface ModelInferenceDebugHandle {
	readonly hp: ModelHyperparams;
	readonly wasm: GgmlWasm;
	readonly weights: WeightTensors | null;
	readonly kvLayers: LayerKVCache[] | null;
	readonly flashAttn: boolean;
	softCap(x: TensorPtr, cap: number): TensorPtr;
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
//
// The methods on this class operate on a ModelInference indirectly, through
// the {@link ModelInferenceDebugHandle} interface; ModelInference exposes
// them externally via thin delegating methods so existing callers
// (smoke-test/real-model-page.js, browser-console `window.inference.*`)
// keep working unchanged.

/**
 * Holds the `debug*` diagnostic methods extracted from {@link ModelInference}.
 * Operating through {@link ModelInferenceDebugHandle} keeps the inference
 * engine's private state private — only the six things the debug path
 * actually needs (hp, wasm, weights, kvLayers, flashAttn, softCap) cross
 * the boundary.
 */
export class ModelInferenceDiagnostics {
	constructor(private readonly target: ModelInferenceDebugHandle) {}

	/** DEBUG: read back a slice of kv.k for a given layer. */
	async debugReadKCache(
		layerIdx: number,
		nBytes: number,
		offset = 0,
	): Promise<Float32Array> {
		if (!this.target.kvLayers) throw new Error("KV cache not initialized");
		const tensor = this.target.kvLayers[layerIdx].k;
		const bytes = await this.target.wasm.downloadFromTensor(
			tensor,
			nBytes,
			offset,
		);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nBytes / 4);
	}

	/** DEBUG: read back a slice of kv.v for a given layer. */
	async debugReadVCache(
		layerIdx: number,
		nBytes: number,
		offset = 0,
	): Promise<Float32Array> {
		if (!this.target.kvLayers) throw new Error("KV cache not initialized");
		const tensor = this.target.kvLayers[layerIdx].v;
		const bytes = await this.target.wasm.downloadFromTensor(
			tensor,
			nBytes,
			offset,
		);
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
		if (!this.target.weights) throw new Error("Weights not loaded");
		if (!this.target.kvLayers) throw new Error("KV cache not initialized");
		const { wasm, hp, weights } = this.target;
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
				const kv = this.target.kvLayers[il];
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
				if (this.target.flashAttn) {
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
				const fullV = this.target.flashAttn
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
				if (this.target.flashAttn) {
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
						qkProcessed = this.target.softCap(
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
		if (!this.target.weights) throw new Error("Weights not loaded");
		const { wasm, hp, weights } = this.target;
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
		if (!this.target.weights) throw new Error("Weights not loaded");
		let tensor: TensorPtr;
		if (which === "output") tensor = this.target.weights.norm;
		else if (which === "attn0") tensor = this.target.weights.layers[0].attnNorm;
		else tensor = this.target.weights.layers[0].ffnNorm;
		const bytes = await this.target.wasm.downloadFromTensor(
			tensor,
			nFloats * 4,
			0,
		);
		return new Float32Array(bytes.buffer, bytes.byteOffset, nFloats);
	}
}
