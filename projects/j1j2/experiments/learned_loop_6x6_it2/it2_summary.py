"""Collect it2 FN populations (fn_<tag>.json in RUNS/*/) per tag: mean, SE, median, SD, # below E0,
difference to the ViT VMC energy and to the ViT-guide FN (same protocol), and seed-paired differences
(common random numbers) when two tags share seeds.
Usage: python it2_summary.py RUNS_DIR [--ref vit,vitB] [--out summary.json]"""
import sys, json, glob, os, argparse
import numpy as np
E0 = -0.5038096538908783
EV = (-0.5036542608124052, 2.131044877124918e-05)
ap = argparse.ArgumentParser(); ap.add_argument('runs'); ap.add_argument('--ref', default='vit,vitB')
ap.add_argument('--out', default=None); ap.add_argument('--merge', default='')   # e.g. 'G2:G2a,G2b'
a = ap.parse_args()
groups = {}; meta = {}
for f in glob.glob(os.path.join(a.runs, '*', 'fn_*.json')):
    d = json.load(open(f)); key = (d['tag'], d['M'], d['beta_target'])
    for r in d['reps']:
        groups.setdefault(key, {})[r['seed']] = r['Emean_site']
    meta.setdefault(key, []).append(dict(file=f, gpu_h=d.get('gpu_h', {}).get('total'), n=len(d['reps'])))
merge = dict(m.split(':') for m in a.merge.split(';') if m)
merged = {}
for (tag, M, b), reps in groups.items():
    t = next((k for k, v in merge.items() if tag in v.split(',')), tag)
    merged.setdefault((t, M, b), {}).update(reps)
    meta[(t, M, b)] = meta.get((t, M, b), []) + (meta[(tag, M, b)] if t != tag else [])
out = {}
refs = [r for r in a.ref.split(',') if r]
ref_pool = {}
for (t, M, b), reps in merged.items():
    if t in refs and M == 128: ref_pool.update(reps)
ref_e = np.array(list(ref_pool.values())) if ref_pool else None
for (t, M, b), reps in sorted(merged.items()):
    e = np.array(list(reps.values())); n = len(e)
    se = float(e.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    row = dict(tag=t, M=M, beta=b, n=n, E_FN_site=float(e.mean()), SE=se, SD=float(e.std(ddof=1)) if n > 1 else None,
               median=float(np.median(e)), n_below_E0=int(np.sum(e < E0)),
               minus_vit_vmc=[float(e.mean() - EV[0]), float(np.hypot(se, EV[1]))],
               z_vs_vit_vmc=float((e.mean() - EV[0]) / np.hypot(se, EV[1])),
               gpu_h=float(sum((m['gpu_h'] or 0) for m in meta.get((t, M, b), []))))
    if ref_e is not None and t not in refs:
        sr = ref_e.std(ddof=1) / np.sqrt(len(ref_e))
        row['minus_vit_fn'] = [float(e.mean() - ref_e.mean()), float(np.hypot(se, sr))]
        common = sorted(set(reps) & set(ref_pool))
        if len(common) > 2:
            d = np.array([reps[s] - ref_pool[s] for s in common])
            x = np.array([reps[s] for s in common]); y = np.array([ref_pool[s] for s in common])
            row['paired_vs_vit_fn'] = dict(n=len(common), mean=float(d.mean()), SE=float(d.std(ddof=1) / np.sqrt(len(d))),
                                           corr=float(np.corrcoef(x, y)[0, 1]))
    out[f'{t}|M{M}|b{b}'] = row
print(json.dumps(out, indent=1))
if a.out: json.dump(out, open(a.out, 'w'), indent=1)
