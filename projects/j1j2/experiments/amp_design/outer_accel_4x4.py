#!/usr/bin/env python3
"""Exact 4x4 test: can the OUTER FN/Krylov loop be accelerated (fewer amplitude refits)?
Ideal loop: log a_{k+1} = log phi_FN[a_k, s_k], s_{k+1} = Krylov(phi_FN, s_k) (as exact_loop_from_net.py).
Variants on the amplitude map only (signs always from the current refreshed amplitude):
  plain                : x_{k+1} = x_k + f_k,  f_k = log phi_FN[a_k,s_k] - log a_k
  over-relaxation w    : x_{k+1} = x_k + w f_k
  Anderson(m)          : type-II Anderson mixing on x = log a with phi_k^2-weighted residual norm
Start: net H (results/net_start_4x4/startH.npz). Scored against ED (eps of the FN energy E_FN[a_k, s_k])."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amp_design_4x4 import build_H, ground, canonical, build_fn, krylov_update, ROOT, D  # noqa

H, diag, ei, ej, hij = build_H(0.5)
E0, _ = ground(H, tol=1e-12)
st = np.load(ROOT / 'results/net_start_4x4/startH.npz')
a0 = np.asarray(st['a'], float); a0 /= np.linalg.norm(a0); s0 = canonical(st['s'])


def fn(a, s):
    F = build_fn(diag, ei, ej, hij, a, s)
    e, phi = ground(F, v0=np.maximum(a, 1e-15), tol=1e-11)
    phi = np.abs(phi); phi /= np.linalg.norm(phi)
    return e, phi


def run(mode, iters=12, w=1.0, m=3, clip=3.0):
    x = np.log(a0); s = s0.copy(); X = []; Fh = []; out = []
    for k in range(iters + 1):
        a = np.exp(x - x.max()); a /= np.linalg.norm(a); x = np.log(a)
        e, phi = fn(a, s)
        out.append(float((e - E0) / abs(E0)))
        f = np.log(np.maximum(phi, 1e-300)) - x
        wts = phi * phi; f = f - np.sum(wts * f)
        s = krylov_update(H, diag, ei, ej, hij, phi, s)     # sign step on the refreshed amplitude
        if mode == 'plain':
            xn = x + f
        elif mode == 'relax':
            xn = x + w * f
        else:  # Anderson type II, weighted least squares
            X.append(x.copy()); Fh.append(f.copy())
            X, Fh = X[-(m + 1):], Fh[-(m + 1):]
            if len(Fh) == 1:
                xn = x + f
            else:
                dF = np.array([Fh[i + 1] - Fh[i] for i in range(len(Fh) - 1)]).T
                dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
                sw = np.sqrt(wts)
                gam, *_ = np.linalg.lstsq(dF * sw[:, None], f * sw, rcond=None)
                xn = x + f - (dX + dF) @ gam
        step = np.clip(xn - x, -clip, clip)
        x = x + step
    return out


res = {}
res['plain'] = run('plain')
for w in (1.5, 2.0, 2.5, 3.0):
    res[f'relax{w}'] = run('relax', w=w)
for m in (2, 3, 5):
    res[f'anderson{m}'] = run('anderson', m=m)
for k, v in res.items():
    print(f'{k:12s}', ' '.join(f'{x:.2e}' for x in v), flush=True)
Path(ROOT / 'results/amp_design/outer_accel_4x4.json').write_text(json.dumps(res, indent=1))
