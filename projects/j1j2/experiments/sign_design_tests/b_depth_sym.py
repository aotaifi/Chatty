#!/usr/bin/env python3
"""(b) Krylov depth from Marshall with the EXACT ground-state amplitude |psi0|, symmetric sector (k=0, A1, flip +).
d = 0..dmax label-free Krylov steps s_{k+1} = s_k sgn[T_k - r_k], r_k = (H |psi0| s_k)/(|psi0| s_k), T_k energy-optimal
(exact grouped threshold over all cuts of sorted r). Reuses krylov_sign_structure/experiments/closed_fn_krylov_sym6x6.py
(imported, unmodified). ED ground state only as the exact amplitude and for scoring.
Cluster: --L 4|6 (square torus) or --torus t1x,t1y,t2x,t2y (tilted torus, torus_cluster.py).
"""
import argparse, json, os, sys, time, types
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ.get("FN_CODE", "/home/a/A.Otaifi/chatty_fn6x6/code"))
sys.path.insert(0, os.environ.get("ED_COMMON_DIR", "/home/a/A.Otaifi/chatty_fn6x6/code"))
import closed_fn_krylov_sym6x6 as C

ap = argparse.ArgumentParser()
ap.add_argument('--L', type=int, default=4)
ap.add_argument('--torus', default=None)
ap.add_argument('--dmax', type=int, default=8)
ap.add_argument('--gs-vec', default=None); ap.add_argument('--gs-states', default=None)
ap.add_argument('--cache-dir', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)
cluster = None; ctag = f"L{args.L}"
if args.torus:
    import torus_cluster
    tv = [int(z) for z in args.torus.split(',')]
    cluster = torus_cluster.Cluster((tv[0], tv[1]), (tv[2], tv[3]))
    ctag = f"T{'_'.join(map(str, tv))}".replace('-', 'm')
os.makedirs(args.cache_dir, exist_ok=True)
S = C.Sym(args.L, 0.5, log=log, cache=os.path.join(args.cache_dir, f"orb_{ctag}.npz"), cluster=cluster)
E0, v0 = C.target_gs(S, types.SimpleNamespace(gs_vec=args.gs_vec, gs_states=args.gs_states, ed_tol=1e-12), log)
a = np.abs(v0); sg0 = np.where(v0 >= 0, 1, -1).astype(np.int8); p0 = v0 * v0
s = C.canonical(S.marshall.copy())
rows = []
for d in range(args.dmax + 1):
    psi = a * s; E = float(psi @ S.H(psi) / (psi @ psi))
    O = abs(float(np.sum(p0 * s * sg0)))
    rec = dict(d=d, w_s=max(0.0, (1 - O) / 2), eps=(E - E0) / abs(E0), E=E, n_wrong_orbits=int(np.sum(s * sg0 * (1 if np.sum(p0 * s * sg0) > 0 else -1) < 0)))
    rows.append(rec); log('DEPTH', json.dumps(rec))
    if rec['w_s'] < 1e-14 and d >= 1: break
    if d == args.dmax: break
    t0 = time.time()
    s, emin, G, r = S.krylov(a, s)
    log(f'  step {d+1}: groups {G} {time.time()-t0:.1f}s')
tag = ctag
out = dict(cluster=ctag, N=S.N, D=S.D, E0=E0, E0_per_site=E0 / S.N, rows=rows,
           note='exact |psi0| amplitude, Marshall start, energy-optimal grouped thresholds, sym sector k=0 A1 flip+')
Path(args.out).write_text(json.dumps(out, indent=1)); log('WROTE', args.out)
