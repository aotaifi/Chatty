#!/usr/bin/env python3
"""Where does the loop started from the k=(pi,pi) state end up?  Residual / S^2 / symmetry of the end state."""
import sys, json
import numpy as np
import run_bad_start as R, bad_start_lib as B
A = B.A
cache = sys.argv[1]
M, H, diag, ei, ej, hij, sym, vecs, info = R.setup("4x4", cache)
S = R.make_starts(M, H, vecs, "4x4"); E0 = info["A1_0"]["E"]; S2 = sym.build_S2()
out = {}
for nm in sys.argv[3:]:
    a0, s0, _ = S[nm]
    hist, term, _ = B.run_loop(M, H, diag, ei, ej, hij, a0, s0, E0, dict(gs=vecs["A1_0"]), maxiter=int(sys.argv[2]),
                               stall_window=450)
    a, s = B.run_loop.last; psi = a * s; psi /= np.linalg.norm(psi)
    E = float(psi @ H @ psi); res = float(np.linalg.norm(H @ psi - E * psi))
    out[nm] = dict(term=term, n_iter=hist[-1]["it"], E_guide=E, eps=(E - E0) / abs(E0), eps_FN=hist[-1]["eps_FN"],
                   eigen_residual=res, S2=float(psi @ S2 @ psi), n_flips_last=hist[-2]["n_sign_flips"],
                   sym={o: sym.expect(o, psi) for o in ["Tx", "Ty", "rot", "mir", "flip"]},
                   min_amp=float(a.min()), frac_amp_small=float((a < 1e-6).mean()))
    print(nm, json.dumps(out[nm]), flush=True)
json.dump(out, open("../../results/bad_start/data/spurious_fixed_point_4x4.json", "w"), indent=1)
