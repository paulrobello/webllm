#!/usr/bin/env python3
"""Stage 4.18 diff — supports the new `backend=...` field added in
Probe 8b's instrumentation. Each input line:

  [CHECKPOINT idx=N name=X type=T backend=B ne=[a,b,c,d] contig=K first8=[...]]

Outputs first divergence point, per-name max-abs-delta, and the per-
name backend tally so the reader can immediately see which JSEP-side
nodes ran on `jsep_buf` vs `CPU`.
"""
import re
import sys
from pathlib import Path

PAT = re.compile(
    r"\[CHECKPOINT idx=(\d+) name=(\S+) type=(\d+) backend=(\S+) "
    r"ne=\[(\-?\d+),(\-?\d+),(\-?\d+),(\-?\d+)\] contig=(\d+) first8=\[(.*?)\]\]"
)


def parse(path):
    out = []
    for line in Path(path).read_text().splitlines():
        m = PAT.search(line)
        if not m:
            continue
        idx, name, typ, backend, n0, n1, n2, n3, contig, first8 = m.groups()
        vals = [float(x) for x in first8.split(",")]
        out.append(
            {
                "idx": int(idx),
                "name": name,
                "type": int(typ),
                "backend": backend,
                "ne": [int(n0), int(n1), int(n2), int(n3)],
                "contig": int(contig),
                "first8": vals,
            }
        )
    return out


def main(jsep_path, ref_path):
    j = parse(jsep_path)
    r = parse(ref_path)
    n = min(len(j), len(r))
    print(f"JSEP entries: {len(j)}  REF entries: {len(r)}  comparing first {n}")

    THRESHOLD = 1e-3
    first_div_idx = None
    print(f"\nThreshold for 'structural' divergence: max-abs-delta > {THRESHOLD}")
    print(
        f"\n{'idx':>4} {'name':<22} {'ne':<22} {'jsep_be':<10} max_abs_delta  jsep_first  ref_first"
    )
    print("-" * 110)
    for i in range(n):
        je, re_ = j[i], r[i]
        if je["name"] != re_["name"] or je["ne"] != re_["ne"]:
            print(
                f"!! mismatch idx={i}: jsep={je['name']}/{je['ne']} ref={re_['name']}/{re_['ne']}"
            )
            continue
        diffs = [abs(a - b) for a, b in zip(je["first8"], re_["first8"])]
        max_d = max(diffs)
        print(
            f"{i:>4} {je['name']:<22} {str(je['ne']):<22} {je['backend']:<10} "
            f"{max_d:>12.6f}  {je['first8'][0]:>10.6f}  {re_['first8'][0]:>10.6f}"
        )
        if first_div_idx is None and max_d > THRESHOLD:
            first_div_idx = i

    if first_div_idx is None:
        print(f"\nNo divergence > {THRESHOLD} found — checkpoints all close.")
    else:
        f = j[first_div_idx]
        rf = r[first_div_idx]
        print(
            f"\nFIRST DIVERGENCE > {THRESHOLD}: idx={first_div_idx} name={f['name']} "
            f"ne={f['ne']} jsep_backend={f['backend']} ref_backend={rf['backend']}"
        )
        print(f"  JSEP first8: {f['first8']}")
        print(f"  REF  first8: {rf['first8']}")

    print("\n=== Per-name max-abs-delta + JSEP-side backend tally ===")
    by_name = {}
    for i in range(n):
        if j[i]["name"] != r[i]["name"]:
            continue
        d = max(abs(a - b) for a, b in zip(j[i]["first8"], r[i]["first8"]))
        by_name.setdefault(j[i]["name"], {"deltas": [], "backends": []})
        by_name[j[i]["name"]]["deltas"].append(d)
        by_name[j[i]["name"]]["backends"].append(j[i]["backend"])
    for name in sorted(by_name, key=lambda k: -max(by_name[k]["deltas"])):
        deltas = by_name[name]["deltas"]
        backends = by_name[name]["backends"]
        be_set = sorted(set(backends))
        print(
            f"  {name:<22}  n={len(deltas):>3}  max={max(deltas):.6f}  "
            f"median={sorted(deltas)[len(deltas)//2]:.6f}  jsep_be={','.join(be_set)}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
