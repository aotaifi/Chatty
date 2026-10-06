#!/usr/bin/env python3
"""Check that exact eigenstates (|phi_n|, sgn phi_n) are fixed points of the FN/Krylov loop:
r(x)=(H psi)(x)/psi(x)=E_n, E_FN=E_n, a_FN=|phi_n|, Krylov flips nothing.  4x4."""
import sys, json
import numpy as np
import run_bad_start as R, bad_start_lib as B
A = B.A
cache = sys.argv[1]
M, H, diag, ei, ej, hij, sym, vecs, info = R.setup("4x4", cache)
rows = {}
for k in ["A1_0", "A1_1", "A1_2", "S1_0", "kPiPi_0", "Brot_0"]:
    v = vecs[k]; E = info[k]["E"]
    a, s = B.guide_from_psi(v)
    nz = int((a < 1e-12).sum())
    act = a > 1e-12
    psi = a * s
    r = np.where(act, (H @ psi) / np.where(act, psi, 1.0), E)
    efn, afn = A.fixed_node_solve(H, diag, ei, ej, hij, a, s, 2e-11) if nz == 0 else (None, None)
    if nz == 0:
        sn, eth, G = A.projected_krylov_update(H, diag, ei, ej, hij, afn, s)
        flips = int(np.sum(sn != s)); da = float(np.linalg.norm(afn - a)); dE = efn - E
    else:
        # zero-amplitude configurations present: active-subspace solve (see run_loop)
        hist, term, _ = B.run_loop(M, H, diag, ei, ej, hij, a, s, info["A1_0"]["E"], dict(gs=vecs["A1_0"]), maxiter=1)
        flips = hist[0]["n_sign_flips"]; da = hist[0]["amp_delta_l2"]; dE = hist[0]["E_FN"] - E
    rows[k] = dict(E_n=E, n_zero_amp=nz, min_amp=float(a.min()), max_abs_r_minus_E=float(np.max(np.abs(r[act] - E))),
                   n_distinct_r_active=int(len(np.unique(np.round(r[act], 8)))), dE_FN_minus_En=float(dE),
                   amp_change_l2=da, n_sign_flips_krylov=flips)
    print(k, json.dumps(rows[k]))
json.dump(rows, open(sys.argv[2], "w"), indent=1)
