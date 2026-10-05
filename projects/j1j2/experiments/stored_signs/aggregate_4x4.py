#!/usr/bin/env python3
"""Aggregate stored-sign 4x4 runs: table at chosen iterations vs the exact recursive loop."""
import json, glob, sys, os
import numpy as np

rdir = sys.argv[1]; tag = sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None
its = [5, 15, 30]
rows = []
if ref and os.path.exists(ref):
    h = {r['it']: r for r in json.load(open(ref))['history']}
    rows.append(('ref_plain(full_4x4_plain)', [(h[i]['w_s'], h[i]['eps'], 0.0) for i in its]))
groups = {}
for f in sorted(glob.glob(f'{rdir}/{tag}_*.json')):
    d = json.load(open(f))
    if 'history' not in d: continue
    m = d['method']; key = m['meth'] if m['meth'] == 'exact' else f"{m['meth']}{'+nbr' if m.get('nbr') else ''}_b{m.get('beta', 1.0)}_N{m['nsamp']}"
    h = {r['it']: r for r in d['history']}
    groups.setdefault(key, []).append([(h[i]['w_s'], h[i]['eps'], h[i].get('D_stored_vs_rec', 0.0)) if i in h else (np.nan,) * 3 for i in its])
out = {}
print(f"{'method':28s} " + ' '.join(f'| it{i}: w_s      eps      D   ' for i in its))
for name, vals in rows:
    print(f"{name:28s} " + ' '.join(f'| {a:.2e} {b:.2e} {c:.1e}' for a, b, c in vals)); out[name] = vals
for key in sorted(groups):
    arr = np.array(groups[key])  # seeds x its x 3
    for sd in range(arr.shape[0]):
        print(f"{key + f' s{sd}':28s} " + ' '.join(f'| {a:.2e} {b:.2e} {c:.1e}' for a, b, c in arr[sd]))
    out[key] = dict(per_seed=arr.tolist(), mean=np.nanmean(arr, axis=0).tolist())
json.dump(dict(iterations=its, columns=['w_s', 'eps', 'D_stored_vs_rec'], data=out), open(f'{rdir}/{tag}_summary.json', 'w'), indent=1)
