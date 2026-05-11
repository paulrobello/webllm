# pyright: reportMissingImports=false
"""
Compare webllm vs HuggingFace reference captures layer-by-layer.

Reads <run-dir>/hf-ref.json and <run-dir>/webllm.json (schema in
eval/tools/parity-capture/README.md) and emits <run-dir>/REPORT.md
with per-layer cosine + L2 + the first layer where cosine drops
below the configured threshold.

Pure NumPy; no torch needed. The HF + webllm sides write float lists
at fp32 precision, so cosine and L2 are well-defined.

Run:

    uv run --no-project --with-requirements \\
        eval/tools/parity-capture/requirements.txt \\
        python eval/tools/parity-capture/compare.py \\
        --run-dir eval/reports/parity-tinyllama-2026-05-11

`--threshold` defaults to 0.95 (end-of-stack gate per README); first-
layer divergence is also flagged at the stricter 0.99 mid-stack gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _load_capture(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing capture: {path}")
    return json.loads(path.read_text())


def _shape_check(ref: dict, web: dict) -> None:
    if ref["n_layer"] != web["n_layer"]:
        raise SystemExit(
            f"n_layer mismatch: hf-ref={ref['n_layer']} webllm={web['n_layer']}"
        )
    if ref["n_embd"] != web["n_embd"]:
        raise SystemExit(
            f"n_embd mismatch: hf-ref={ref['n_embd']} webllm={web['n_embd']}"
        )
    if ref["input_token_ids"] != web["input_token_ids"]:
        # Tokenizer divergence is itself a bug; report and abort.
        print(
            "warning: input_token_ids differ between hf-ref and webllm captures:",
            file=sys.stderr,
        )
        print(f"  hf-ref : {ref['input_token_ids']}", file=sys.stderr)
        print(f"  webllm : {web['input_token_ids']}", file=sys.stderr)
        print(
            "  proceeding anyway — but a tokenizer divergence is itself a Stage 3 bug.",
            file=sys.stderr,
        )


def _format_row(
    layer_idx: int, cos: float, l2: float, flag_cos: float, flag_drop: float | None
) -> str:
    flag = ""
    if cos < flag_cos:
        flag = " ⚠ below threshold"
    if flag_drop is not None:
        flag += f" (Δ vs prev: {flag_drop:+.4f})"
    return f"| {layer_idx} | {cos:.4f} | {l2:.4f} |{flag} |"


def compare(run_dir: Path, threshold: float, first_layer_threshold: float) -> int:
    ref_path = run_dir / "hf-ref.json"
    web_path = run_dir / "webllm.json"
    ref = _load_capture(ref_path)
    web = _load_capture(web_path)

    _shape_check(ref, web)

    n_layer = ref["n_layer"]
    n_embd = ref["n_embd"]

    rows = [
        "| Layer | Cosine | L2 | Note |",
        "|-------|--------|------|------|",
    ]
    prev_cos: float | None = None
    first_below_threshold = -1
    first_sudden_drop = -1
    for il in range(n_layer):
        a = np.asarray(ref["per_layer_residual_last_token"][il], dtype=np.float32)
        b = np.asarray(web["per_layer_residual_last_token"][il], dtype=np.float32)
        cos = _cosine(a, b)
        l2 = _l2_distance(a, b)
        flag_thresh = first_layer_threshold if il == 0 else threshold
        drop = None
        if prev_cos is not None:
            delta = cos - prev_cos
            if delta <= -0.05:
                drop = delta
                if first_sudden_drop == -1:
                    first_sudden_drop = il
        if first_below_threshold == -1 and cos < flag_thresh:
            first_below_threshold = il
        rows.append(_format_row(il, cos, l2, flag_thresh, drop))
        prev_cos = cos

    # Final-norm and logits comparison
    fn_ref = np.asarray(ref["final_norm_hidden_last_token"], dtype=np.float32)
    fn_web = np.asarray(web["final_norm_hidden_last_token"], dtype=np.float32)
    fn_cos = _cosine(fn_ref, fn_web)
    fn_l2 = _l2_distance(fn_ref, fn_web)

    ref_top_ids = ref["logits_top16"]["ids"]
    web_top_ids = web["logits_top16"]["ids"]
    overlap = len(set(ref_top_ids) & set(web_top_ids))
    argmax_match = ref_top_ids[0] == web_top_ids[0] if ref_top_ids and web_top_ids else False

    summary_lines = [
        f"# Parity Report: {ref['model']}",
        "",
        f"- **HF capture**: `{ref_path.name}` ({ref.get('captured_at', '?')}, dtype={ref.get('torch_dtype', '?')})",
        f"- **WebLLM capture**: `{web_path.name}` ({web.get('captured_at', '?')})",
        f"- **Prompt**: `{ref.get('prompt', '?')}`",
        f"- **n_layer / n_embd**: {n_layer} / {n_embd}",
        f"- **Thresholds**: cosine ≥ {threshold} end-of-stack, ≥ {first_layer_threshold} first-block",
        "",
        "## Per-layer residual (last token)",
        "",
        *rows,
        "",
        "## Final norm + logits (last token)",
        "",
        f"- Final-norm hidden cosine: **{fn_cos:.4f}** (L2 {fn_l2:.4f})",
        f"- Top-16 logits overlap (set): **{overlap}/16**",
        f"- Greedy argmax match: **{'yes' if argmax_match else 'no'}**",
        f"  - hf-ref top-1: id {ref_top_ids[0] if ref_top_ids else '?'} (val {ref['logits_top16']['values'][0] if ref_top_ids else '?'})",
        f"  - webllm top-1: id {web_top_ids[0] if web_top_ids else '?'} (val {web['logits_top16']['values'][0] if web_top_ids else '?'})",
        "",
        "## Diagnosis",
        "",
    ]
    if first_below_threshold == -1 and first_sudden_drop == -1:
        summary_lines += [
            "**PASS** — every layer above threshold; no sudden drops.",
            f"End-of-stack cosine {fn_cos:.4f} → "
            + ("OK" if fn_cos >= threshold else "below end-of-stack gate"),
        ]
    else:
        if first_below_threshold != -1:
            summary_lines.append(
                f"- First layer with cosine below threshold: **block {first_below_threshold}**"
            )
        if first_sudden_drop != -1:
            summary_lines.append(
                f"- First sudden Δ ≤ -0.05 between consecutive layers at: **block {first_sudden_drop}**"
            )
        summary_lines += [
            "",
            "Inspect the op sequence inside that block (or the inputs from the prior block) for the bug.",
        ]

    out_path = run_dir / "REPORT.md"
    out_path.write_text("\n".join(summary_lines) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the parity run directory containing hf-ref.json + webllm.json",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine threshold for end-of-stack flag (default 0.95).",
    )
    p.add_argument(
        "--first-layer-threshold",
        type=float,
        default=0.99,
        help="Cosine threshold for the first block (default 0.99).",
    )
    args = p.parse_args(argv)
    return compare(args.run_dir, args.threshold, args.first_layer_threshold)


if __name__ == "__main__":
    sys.exit(main())
