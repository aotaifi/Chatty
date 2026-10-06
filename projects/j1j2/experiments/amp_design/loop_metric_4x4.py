#!/usr/bin/env python3
"""Loop-relevant score of the best sampled update (tempered edge-GN, rf model, beta=1/4, lam=1e-6, 4 steps x N):
next-iteration FN energy after the Krylov sign step, as a fraction of the ideal (exact phi_FN) amplitude gain,
and where the residual ratio error sits (kept edges of H_FN vs sign-violating edges of the NEXT guide)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0], '--state', os.environ.get('STATE', 'k0'), '--model', 'rf', '--set', 'scan4', '--Ns', '10']
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amp_design_variants_4x4 as V   # builds fz, model, H for the requested state (runs a tiny N=10 scan)
M = V.M; fz = V.fz; H, diag, ei, ej, hij = V.H, V.diag, V.ei, V.ej, V.hij; E0 = V.E0; s = V.s

def next_fn(u):
    b = fz.b(u); sn = M.krylov_update(H, diag, ei, ej, hij, b, s)
    e, _ = M.ground(M.build_fn(diag, ei, ej, hij, b, sn), v0=np.maximum(b, 1e-15), tol=1e-11)
    return (e - E0) / abs(E0), sn

def edge_err(u, sn):
    w = u - fz.ustar; phi = fz.phi
    viol = (sn[ei] * hij * sn[ej]) > 0
    t = hij * phi[ei] * phi[ej] * (w[ei] - w[ej]) ** 2
    return float(t[~viol].sum()), float(t[viol].sum())

res = {}
e_no, _ = next_fn(np.zeros(M.D)); e_id, sid = next_fn(fz.ustar)
res['eps_FN_next'] = dict(no_update=e_no, ideal=e_id)
for N in (10000, 100000):
    rows = []
    for sd in range(2):
        rng = np.random.default_rng([sd, N, 31])
        u = np.zeros(M.D)
        for st in range(4):
            U, w = V.draw(V.qdist('temp', u, 0.25), N, rng, u)
            g, S, A, _ = M.tab_stats(fz, V.oid, u, U, w)
            u = u + V.lsolve(A, g, 1e-6)[V.oid]
        e, sn = next_fn(u); k, vi = edge_err(u, sn); k0, v0 = edge_err(np.zeros(M.D), sn)
        rows.append(dict(frozen_gain=fz.gain(u), eps_FN_next=e, loop_frac=(e_no - e) / (e_no - e_id),
                         err_kept=k, err_viol=vi, err_kept_noupd=k0, err_viol_noupd=v0))
    res[f'N{N}'] = rows
    for r in rows: print(os.environ.get('STATE', 'k0'), N, {k: float('%.4g' % v) for k, v in r.items()}, flush=True)
print('eps_FN_next no-update %.4e ideal %.4e' % (e_no, e_id))
Path(M.ROOT / f"results/amp_design/loop_metric_{os.environ.get('STATE', 'k0')}.json").write_text(json.dumps(res, indent=1))
