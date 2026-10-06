#!/usr/bin/env python3
"""Can a cheap paired estimator certify a tiny amplitude refresh? (4x4, net-H frozen problem k0, exact).
New amplitude b1 = a exp(u*) (the exact refresh; frozen gain G = 3.3e-4 = 3.9e-5 per site x 16... per lattice).
Estimator: common samples x ~ q (q = a^{2 beta}), self-normalised IS for E_F[b] = <b|F|b>/<b|b>, b in {a, b1};
difference D_hat = E_F[b1] - E_F[a] on the SAME samples. Report mean, std over 200 replicas, vs true -G.
By majorisation E_FN[b1,s] <= <b1 s|H|b1 s> <= E_F[b1], so a resolved D_hat < 0 certifies the upper bound dropped."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amp_design_4x4 as M

H, diag, ei, ej, hij = M.build_H(0.5)
st = np.load(M.ROOT / 'results/net_start_4x4/startH.npz')
a = np.asarray(st['a'], float); a /= np.linalg.norm(a); s = M.canonical(st['s'])
fz = M.Frozen(M.build_fn(diag, ei, ej, hij, a, s), a)
b1 = fz.b(fz.ustar); b0 = fz.a
el0 = (fz.F @ b0) / b0; el1 = (fz.F @ b1) / b1
true = float(b1 @ (fz.F @ b1) - b0 @ (fz.F @ b0))
out = dict(true_diff=true)
print('true frozen difference %.3e' % true)
for beta in (1.0, 0.5, 0.25):
    q = b0 ** (2 * beta); q /= q.sum()
    for N in (1000, 10000, 100000):
        rng = np.random.default_rng([N, int(beta * 100)])
        D = []; U0 = []
        for r in range(200):
            x = rng.choice(M.D, size=N, p=q)
            w0 = b0[x] ** 2 / q[x]; w1 = b1[x] ** 2 / q[x]
            e0 = np.sum(w0 * el0[x]) / w0.sum(); e1 = np.sum(w1 * el1[x]) / w1.sum()
            D.append(e1 - e0); U0.append(e0)
        D = np.array(D); U0 = np.array(U0)
        out[f'beta{beta}_N{N}'] = dict(mean=float(D.mean()), std=float(D.std()), std_single=float(U0.std()))
        print(f'beta={beta:4} N={N:>6}  D_hat mean {D.mean():+.3e} std {D.std():.2e}  (unpaired E std {U0.std():.2e})  z={D.mean()/D.std():+.1f}', flush=True)
Path(M.ROOT / 'results/amp_design/paired_referee_4x4.json').write_text(json.dumps(out, indent=1))
