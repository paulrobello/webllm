import type { ModelHyperparams } from "../../core/types.js";
import { RopeMode } from "../ggml-wasm.js";

/**
 * Map a model architecture to its ggml rope mode. Read by
 * encoder-inference.ts and causal-embedder-inference.ts for
 * graph-builder dispatch. Will be deleted alongside those files
 * in P3 / P4.
 */
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
	return String(architecture).startsWith("qwen")
		? RopeMode.NEOX
		: RopeMode.NORMAL;
}
