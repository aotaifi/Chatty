#!/usr/bin/env python3
"""Aggregate capacity runs: plateau w_s (geometric mean over it 31-40), final eps, accuracies, honest D_true. usage: aggregate_capacity.py <results_dir> <tag>"""
import json, glob, sys, re
import numpy as np
rdir, tag = sys.argv[1], sys.argv[2]
ex = {r['it']: r for r in json.load(open(f'{rdir}/../runs_ref/main2_exact.json'))['history']} if False else None
groups = {}
for f in sorted(glob.glob(f'{rdir}/{tag}_*_s?.json')):
    d = json.load(open(f)); name = re.sub(r'_s\d$', '', f.split(f'{tag}_')[1][:-5])
    groups.setdefault(name, []).append(d)
def arr(ds, key): return np.array([[r[key] for r in d['history'][1:]] for d in ds])
out = {}
print(f"{'variant':14s} {'params':>7s} seeds | plateau w_s (s0, s1, it31-40 geomean) | eps it40 (s0,s1) | acc_hold_unseen/acc_q_full (it31-40 min) | D_true it40 | train_acc min")
for name, ds in groups.items():
    w = arr(ds, 'w_s'); e = arr(ds, 'eps')
    gm = np.exp(np.log(w[:, 30:40]).mean(axis=1))
    dt = arr(ds, 'D_true_stored_vs_rec')[:, -1]; acq = arr(ds, 'acc_q_full')[:, 30:].mean(axis=1); acu = arr(ds, 'acc_hold_unseen_w')[:, 30:].mean(axis=1)
    atr = arr(ds, 'acc_train_w')[:, 30:].mean(axis=1)
    print(f"{name:14s} {ds[0]['method']['params']:7d} {len(ds)} | " + ' '.join(f'{x:.1e}' for x in gm) + ' | ' + ' '.join(f'{x:.1e}' for x in e[:, -1]) +
          ' | ' + ' '.join(f'{x:.5f}/{y:.5f}' for x, y in zip(acu, acq)) + ' | ' + ' '.join(f'{x:.1e}' for x in dt) + ' | tr ' + ' '.join(f'{x:.5f}' for x in atr))
    out[name] = dict(params=ds[0]['method']['params'], plateau_w_s=gm.tolist(), eps_final=e[:, -1].tolist(), D_true_final=dt.tolist(), acc_q_full=acq.tolist(), acc_hold_unseen=acu.tolist(), acc_train=atr.tolist())
json.dump(out, open(f'{rdir}/{tag}_summary.json', 'w'), indent=1)
