# pyright: reportMissingImports=false
# torch / transformers load from the inline uv env via
# --with-requirements; they are not in any project venv that Pyright
# watches. Suppress at file level since this script is a self-
# contained side-run, not part of the TS project's typing surface.
"""
Generic HuggingFace `transformers` reference-capture for webllm parity.

Loads the named model, tokenizes each prompt from inputs.json, runs a
single forward pass with `output_hidden_states=True`, and writes a
standard-shape JSON dump for downstream `compare.py`. Generic across
causal-LM architectures supported by `transformers`.

The output schema (see eval/tools/parity-capture/README.md for the
canonical reference):

{
  "model": "<hf-id>",
  "captured_with": "transformers",
  "captured_at": "<UTC ISO8601>",
  "torch_dtype": "float32" | "bfloat16",
  "prompt": "<text>",
  "input_token_ids": [...],
  "n_layer": <int>,
  "n_embd": <int>,
  "per_layer_residual_last_token": [
    [<n_embd floats>],   # layer 0 output for last token
    [<n_embd floats>],   # layer 1
    ...                  # one entry per transformer block
  ],
  "final_norm_hidden_last_token": [<n_embd floats>],
  "logits_top16": {"ids": [...], "values": [...]}
}

Per CLAUDE.md, the model should be pre-fetched via `hfdownloader`
before running this script — `transformers.from_pretrained()` is
slower and easier to wedge mid-fetch on multi-GB downloads.

Run example:

    hfdownloader download unsloth/gemma-4-E2B-it
    RUN_DIR=eval/reports/parity-gemma-4-e2b-$(date +%Y-%m-%d)
    mkdir -p "$RUN_DIR"
    uv run --no-project --with-requirements \\
      eval/tools/parity-capture/requirements.txt \\
      python eval/tools/parity-capture/capture-hf-ref.py \\
      --model unsloth/gemma-4-E2B-it \\
      --inputs eval/tools/parity-capture/inputs.json \\
      --output "$RUN_DIR/hf-ref.json"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick_torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float32", "f32", "fp32"):
        return torch.float32
    if name in ("float16", "f16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype: {name}")


def _capture_single_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    add_bos: bool,
    top_k: int,
) -> dict:
    """
    Run a single forward pass and return the captured tensors in the
    canonical JSON-friendly shape (lists of plain floats, not tensors).
    """
    # Tokenize without specials, then optionally prepend BOS to match
    # webllm's chat-template encode path (engine.ts adds BOS via the
    # template for causal LMs; raw-text probes here mirror that).
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos and tokenizer.bos_token_id is not None:
        if not ids or ids[0] != tokenizer.bos_token_id:
            ids = [tokenizer.bos_token_id] + ids

    input_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    # `hidden_states` is a tuple of length (n_layer + 1):
    # index 0 is the embedding lookup output (BEFORE block 0).
    # index i in [1..n_layer] is the residual stream AFTER block i-1.
    # The reference shape we want is "per_layer_residual_last_token"
    # = [block_0_output, block_1_output, ..., block_{n_layer-1}_output]
    # i.e. hidden_states[1:].
    hidden = out.hidden_states
    n_layer = len(hidden) - 1
    per_layer = [
        hidden[i + 1][0, -1, :].to(torch.float32).cpu().tolist() for i in range(n_layer)
    ]

    # Final-norm output: for most causal LMs `output.hidden_states[-1]`
    # is already post-final-norm. Re-derive from `out.logits` only if a
    # specific architecture diverges from that convention.
    final_norm = hidden[-1][0, -1, :].to(torch.float32).cpu().tolist()

    # Top-k logits for the last token.
    logits = out.logits[0, -1, :].to(torch.float32).cpu()
    top_values, top_indices = torch.topk(logits, k=min(top_k, logits.numel()))
    logits_top = {
        "ids": top_indices.tolist(),
        "values": top_values.tolist(),
    }

    return {
        "prompt": prompt,
        "input_token_ids": ids,
        "n_layer": n_layer,
        "n_embd": len(final_norm),
        "per_layer_residual_last_token": per_layer,
        "final_norm_hidden_last_token": final_norm,
        "logits_top16": logits_top,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace model id (e.g. unsloth/gemma-4-E2B-it)",
    )
    p.add_argument(
        "--inputs",
        required=True,
        type=Path,
        help="Path to inputs.json (single object with `prompts` array)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the captured JSON (parent dir must exist)",
    )
    p.add_argument(
        "--torch-dtype",
        default="float32",
        help="Reference precision: float32 (default), bfloat16, or float16. fp32 is the cleanest baseline; switch to bf16 only if VRAM-constrained.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device_map (default: cpu). Use 'cuda' or 'mps' for speed.",
    )
    p.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend BOS to the input token ids if absent (matches webllm chat-template encode for causal LMs).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of top logits indices/values to capture per last-token (default 16)",
    )
    args = p.parse_args(argv)

    inputs_doc = json.loads(args.inputs.read_text())
    prompts: list[str] = inputs_doc.get("prompts", [])
    if not prompts:
        print(f"error: no `prompts` array in {args.inputs}", file=sys.stderr)
        return 2
    if len(prompts) > 1:
        # The output format is per-prompt; if multi-prompt support is
        # needed later, wrap captures in a list keyed by prompt. For
        # now keep the format flat to match the README schema.
        print(
            f"warning: {args.inputs} has {len(prompts)} prompts; only the first will be captured. Add multi-prompt support if needed.",
            file=sys.stderr,
        )

    torch_dtype = _pick_torch_dtype(args.torch_dtype)

    print(f"loading {args.model} (torch_dtype={torch_dtype}, device={args.device})…", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    model.eval()

    captured = _capture_single_prompt(
        model,
        tokenizer,
        prompts[0],
        add_bos=args.add_bos,
        top_k=args.top_k,
    )

    doc = {
        "model": args.model,
        "captured_with": "transformers",
        "captured_at": _utc_now_iso(),
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
        "add_bos": args.add_bos,
        **captured,
    }

    args.output.write_text(json.dumps(doc, indent=2))
    print(f"wrote {args.output} (n_layer={doc['n_layer']}, n_embd={doc['n_embd']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
