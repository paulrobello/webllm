import json
import sys
import re

def parse_checkpoints(file_path):
    full_stats = []
    idx_dumps = {} # name -> {idx -> values}
    
    full_pat = re.compile(r'^\[CHECKPOINT-FULL idx=(\d+) name=(\S+) n_elements=(\d+) finite=(\d+) mean=(\S+) abs_max=(\S+) abs_min=(\S+) nan=(\d+) inf=(\d+)\]$')
    dump_pat = re.compile(r'^\[CHECKPOINT-IDX-DUMP idx=(\d+) name=(\S+) start=(\d+) count=(\d+) values=\[(.*)\]\]$')

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = full_pat.search(line)
            if m:
                full_stats.append({
                    'idx': int(m.group(1)),
                    'name': m.group(2),
                    'n': int(m.group(3)),
                    'mean': float(m.group(5)),
                    'abs_max': float(m.group(6)),
                    'abs_min': float(m.group(7))
                })
                continue
            
            m = dump_pat.search(line)
            if m:
                idx = int(m.group(1))
                name = m.group(2)
                start = int(m.group(3))
                vals_str = m.group(5)
                if not vals_str:
                    vals = []
                else:
                    vals = [float(x) for x in vals_str.split(',')]
                
                if name not in idx_dumps: idx_dumps[name] = {}
                if idx not in idx_dumps[name]:
                    n = 49152 # default max
                    for s in full_stats:
                        if s['idx'] == idx and s['name'] == name:
                            n = s['n']
                            break
                    idx_dumps[name][idx] = [0.0] * n
                
                target = idx_dumps[name][idx]
                for i, v in enumerate(vals):
                    if start + i < len(target):
                        target[start + i] = v

    return full_stats, idx_dumps

def analyze_tensor(name, j_data, r_data):
    n = min(len(j_data), len(r_data))
    j_data = j_data[:n]
    r_data = r_data[:n]
    
    diff = [abs(j - r) for j, r in zip(j_data, r_data)]
    max_diff = max(diff) if diff else 0
    first_div = next((i for i, d in enumerate(diff) if d > 1e-5), None)
    
    print(f"\nElement-wise Analysis: {name} (n={n})")
    print(f"Max Abs Delta: {max_diff:.6g}")
    if first_div is not None:
        print(f"First Divergent Index: {first_div} (val_j={j_data[first_div]:.6g}, val_r={r_data[first_div]:.6g}, delta={diff[first_div]:.6g})")
        div_count = sum(1 for d in diff if d > 1e-5)
        print(f"Total Divergent Indices (>1e-5): {div_count} ({div_count/n*100:.2f}%)")
    else:
        print("No divergence > 1e-5 found.")

    # Zero run detection
    j_zeros = [v == 0 for v in j_data]
    r_zeros = [v == 0 for v in r_data]
    print(f"JSEP Zero Count: {sum(j_zeros)} ({sum(j_zeros)/n*100:.2f}%)")
    print(f"Ref Zero Count:  {sum(r_zeros)} ({sum(r_zeros)/n*100:.2f}%)")
    
    if any(j_zeros):
        runs = []
        current_run = 0
        for z in j_zeros:
            if z:
                current_run += 1
            else:
                if current_run > 0: runs.append(current_run)
                current_run = 0
        if current_run > 0: runs.append(current_run)
        if runs:
            print(f"Longest JSEP Zero Run: {max(runs)}")

def main():
    jsep_path = 'eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.33-spike.txt'
    ref_path = 'eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.33-ref.txt'
    
    try:
        jsep_stats, jsep_dumps = parse_checkpoints(jsep_path)
        ref_stats, ref_dumps = parse_checkpoints(ref_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"{ 'Name':<15} | { 'Idx':<3} | { 'JSEP AbsMax':<12} | { 'Ref AbsMax':<12} | { 'Delta':<12}")
    print("-" * 65)
    
    for js in jsep_stats:
        rs = next((s for s in ref_stats if s['idx'] == js['idx'] and s['name'] == js['name']), None)
        if rs:
            delta = abs(js['abs_max'] - rs['abs_max'])
            print(f"{js['name']:<15} | {js['idx']:<3} | {js['abs_max']:<12.6g} | {rs['abs_max']:<12.6g} | {delta:<12.6g}")

    # Analyze all dumped tensors at prefill (first occurrence)
    target_names = ['kq-0', 'kq_soft_max-0', 'kqv_out-0']
    for name in target_names:
        j_indices = jsep_dumps.get(name, {})
        r_indices = ref_dumps.get(name, {})
        if not j_indices or not r_indices: continue
        
        j_idx = sorted(j_indices.keys())[0]
        r_idx = sorted(r_indices.keys())[0]
        analyze_tensor(f"{name} (JSEP idx {j_idx}, Ref idx {r_idx})", j_indices[j_idx], r_indices[r_idx])

if __name__ == "__main__":
    main()
