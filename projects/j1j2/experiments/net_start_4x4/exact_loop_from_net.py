#!/usr/bin/env python3
"""Ideal (noise-free) FN/Krylov loop from a trained-network guide: a_{k+1} = exact ground state phi_FN of
H_FN[a_k, s_k], s_{k+1} = exact energy-optimal current-sign Krylov step.  Ceiling for what any amplitude
learner could deliver from this start.  (Functions imported from loop_from_net_4x4.py.)"""
import argparse, json
import numpy as np
from loop_from_net_4x4 import build_fn, krylov_update, build_H, ground, canonical, marshall

ap = argparse.ArgumentParser()
ap.add_argument('--start', required=True); ap.add_argument('--iters', type=int, default=40); ap.add_argument('--cpus', type=int, default=1); ap.add_argument('--out', required=True)
args = ap.parse_args()
H, diag, ei, ej, hij = build_H(0.5)
E0, psi0 = ground(H, tol=1e-12)
sM = canonical(marshall())
if np.dot(psi0, sM) < 0: psi0 = -psi0
strue = canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8)); ptrue = psi0 ** 2
st = np.load(args.start); a = np.asarray(st['a'], float); a /= np.linalg.norm(a); s = canonical(st['s'])
def score(a, s):
    p = a * s; E = float(p @ (H @ p) / (p @ p)); O = abs(float(np.sum(ptrue * s * strue)))
    return dict(E=E, eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2), F_amp=float(np.dot(a, np.abs(psi0)) ** 2 / (a @ a)))
hist = []
for k in range(args.iters + 1):
    F = build_fn(diag, ei, ej, hij, a, s)
    efn, phi = ground(F, v0=np.maximum(a, 1e-15), tol=2e-11); phi = np.abs(phi); phi /= np.linalg.norm(phi)
    rec = dict(it=k, **score(a, s), E_FN=efn, eps_FN=(efn - E0) / abs(E0)); hist.append(rec)
    print(json.dumps(rec), flush=True)
    snew, G = krylov_update(H, diag, ei, ej, hij, phi, s)       # sign step uses the refreshed amplitude phi
    a, s = phi, snew
json.dump(hist, open(args.out, 'w'), indent=1)
