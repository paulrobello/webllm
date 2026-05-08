"""Stage 4.33 — spatial pattern analysis of kq-0 zero-regions.

kq-0 shape: [256 KV, 6 Q, 32 heads, 1] (innermost first).
For prefill of 6 tokens, only KV positions 0-5 should be non-zero
(K cache positions 6+ are zero-initialized).

Expected non-zero positions: 6 KV × 6 Q × 32 heads = 1152 elements.
JSEP zero count = 48648  →  active non-zeros = 504  →  648 missing.

This script localizes the missing non-zeros by head and Q position.
"""
from __future__ import annotations
import re
from collections import defaultdict


def parse_kq0_dump(path: str, target_idx: int = 9) -> list[float]:
    """Reconstruct full kq-0 tensor from IDX-DUMP rows."""
    n = 49152
    out = [0.0] * n
    pat = re.compile(r"^\[CHECKPOINT-IDX-DUMP idx=(\d+) name=kq-0 start=(\d+) count=(\d+) values=\[(.*)\]\]$")
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                continue
            if int(m.group(1)) != target_idx:
                continue
            start = int(m.group(2))
            vals = [float(x) for x in m.group(4).split(",") if x]
            for i, v in enumerate(vals):
                if start + i < n:
                    out[start + i] = v
    return out


def head_qpos_breakdown(data: list[float]) -> dict:
    """For shape [256,6,32], decompose into (head, q_pos) → (nonzero_count, max_abs).

    Linear index: idx = head * (256*6) + qpos * 256 + kpos
    """
    breakdown: dict[tuple[int, int], dict] = {}
    for h in range(32):
        for q in range(6):
            base = h * (256 * 6) + q * 256
            row = data[base : base + 256]
            nz = [v for v in row[:6] if v != 0.0]  # only first 6 K positions are "active"
            extra_nz = [v for v in row[6:] if v != 0.0]  # any non-zero past position 5 is anomalous
            breakdown[(h, q)] = {
                "nz_in_active": len(nz),
                "abs_max_active": max((abs(v) for v in nz), default=0.0),
                "nz_outside_active": len(extra_nz),
            }
    return breakdown


def summarize(label: str, breakdown: dict) -> None:
    total_nz_active = sum(b["nz_in_active"] for b in breakdown.values())
    total_outside = sum(b["nz_outside_active"] for b in breakdown.values())
    print(f"\n=== {label} ===")
    print(f"  Total non-zeros in active 6×6 region: {total_nz_active} / {32*6*6} = {total_nz_active / (32*6*6) * 100:.1f}%")
    print(f"  Total non-zeros outside active region (anomalous): {total_outside}")

    # Per-head completeness: 36 expected per head (6×6)
    by_head_active = defaultdict(int)
    for (h, _q), b in breakdown.items():
        by_head_active[h] += b["nz_in_active"]
    head_counts = sorted(by_head_active.items())
    print(f"  Per-head non-zero count (expected 36 each):")
    for h, n in head_counts:
        bar = "█" * (n // 2) + " " * (18 - n // 2)
        flag = "" if n == 36 else "  ← INCOMPLETE"
        print(f"    head {h:2d}: {n:2d} / 36  [{bar}]{flag}")


def per_qpos_breakdown(breakdown: dict) -> None:
    """Show non-zero count by Q-position summed across heads."""
    by_q = defaultdict(int)
    for (_h, q), b in breakdown.items():
        by_q[q] += b["nz_in_active"]
    print("\n  Per-Q-position non-zero count (expected 32×(q+1) per row, but kq is pre-mask so all 6 K should be non-zero):")
    print("    Note: kq is pre-mask; K cache positions 0-5 are populated, so ALL 6 K-positions should be non-zero per Q-position.")
    print("    Expected: 32 heads × 6 K-positions = 192 per Q-position.")
    for q in range(6):
        n = by_q[q]
        bar = "█" * (n // 8) + " " * (24 - n // 8)
        flag = "" if n == 192 else f"  ← MISSING {192 - n}"
        print(f"    q={q}: {n:3d} / 192  [{bar}]{flag}")


def main() -> None:
    base = "eval/reports/p2-v2-option-a-prime-2026-05-06"
    spike_path = f"{base}/STAGE-4.33-spike.txt"
    ref_path = f"{base}/STAGE-4.33-ref.txt"

    spike = parse_kq0_dump(spike_path, target_idx=9)
    ref = parse_kq0_dump(ref_path, target_idx=9)

    spike_b = head_qpos_breakdown(spike)
    ref_b = head_qpos_breakdown(ref)

    summarize("REF (CPU non-JSEP)", ref_b)
    per_qpos_breakdown(ref_b)
    summarize("JSEP (WGSL Q×K^T matmul)", spike_b)
    per_qpos_breakdown(spike_b)

    # Localize first head where JSEP differs from ref
    print("\n=== JSEP vs REF: heads with non-zero count mismatch ===")
    by_head_jsep = defaultdict(int)
    by_head_ref = defaultdict(int)
    for (h, _q), b in spike_b.items():
        by_head_jsep[h] += b["nz_in_active"]
    for (h, _q), b in ref_b.items():
        by_head_ref[h] += b["nz_in_active"]
    for h in range(32):
        if by_head_jsep[h] != by_head_ref[h]:
            print(f"  head {h:2d}: JSEP={by_head_jsep[h]:2d}  REF={by_head_ref[h]:2d}  Δ={by_head_ref[h] - by_head_jsep[h]:+d}")

    # Localize first (head, q) pair where they differ
    print("\n=== JSEP vs REF: first 20 (head, q) cells with non-zero count mismatch ===")
    diffs = [
        (h, q, ref_b[(h, q)]["nz_in_active"], spike_b[(h, q)]["nz_in_active"])
        for (h, q) in ref_b
        if ref_b[(h, q)]["nz_in_active"] != spike_b[(h, q)]["nz_in_active"]
    ]
    for h, q, r, j in diffs[:20]:
        print(f"  head={h:2d} q={q}: REF={r}  JSEP={j}  Δ={r - j:+d}")
    print(f"\n  Total mismatched (head,q) cells: {len(diffs)} / 192")


if __name__ == "__main__":
    main()
