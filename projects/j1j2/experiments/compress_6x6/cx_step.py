"""Step-size test (README "step size"): can ONE loop iteration gain as much as one Lanczos step, and do iterations compound?
Exact 6x6 tables, no networks (stage 1). One-hop quantities of the current guide (la, s):
  V, W (violating / kept neighbour weights), local energy E_L = H_xx + V - W, E = <H>.
Variants of one iteration (new log-amplitude la + d, new sign s'):
  L      exact Lanczos:   f = 1 + alpha (E_L - E),  d = log|f|,               s' = s sgn f   (alpha energy-optimal)
  Lc_eps clipped Lanczos: d = log max(|f|, eps),    s' = s sgn f                             (alpha energy-optimal)
  SI_lam scaled semi-implicit FN target: d = lam T, T = log(1+W) - log(1+H_xx+V-E), s' = Krylov sign of (la+d, s)
         (lam = 1 is the current loop; lam energy-optimal over a grid)
Each variant is iterated k = 1..4 times on its own tables (step parameter re-optimised every iteration).
Recorded per iteration: <H> - E0 per site, chosen parameter, bond-rms roughness of d (how hard d is to learn; the
SI_1 value is the current loop's), fraction of sign flips.  Reference: psi_P + p Lanczos steps (cx_lanczos_ref).
  python cx_step.py OUT
"""
import os, sys
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); F = os.path.join(OUT, 'cx_step.json')
lab = Lab(); sec = lab.sec
K = 4
EPS = [0.3, 0.1, 0.03]
LAMS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def err(la, s):
    return lab.H_site(la, s)


def onehop(la, s):
    lan, V, W = lab.VW(la, s)
    v = sec.vec(la, s); E = float(v @ sec.H(v))
    return V, W, lab.D0 + V - W, E


def lanczos_state(la, s, EL, E, eps):
    def build(alpha):
        f = 1.0 + alpha * (EL - E)
        d = jnp.log(jnp.maximum(jnp.abs(f), max(eps, 1e-300)))
        return la + d, s * jnp.where(f >= 0, 1.0, -1.0).astype(s.dtype), d
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))     # energy sd of the guide
    r = minimize_scalar(lambda a: err(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded',
                        options=dict(xatol=1e-4 / sd))
    la2, s2, d = build(r.x)
    return la2, s2, d, float(r.x)


def si_state(la, s, V, W, E, lam):
    T = lab.T_of(V, W, E)
    la2 = la + lam * T
    s2, _, _ = sec.krylov(sec.vec(la2, jnp.ones_like(la2)), s)
    return la2, s2, lam * T


res = dict(start=err(lab.lP0, lab.sP0), lanczos_ref=[2.55e-5, 8.59e-6, 3.82e-6, 1.95e-6], runs={})
variants = ['L'] + [f'Lc_{e}' for e in EPS] + ['SI_1', 'SI_opt']
for name in variants:
    la, s = lab.lP0, lab.sP0; rows = []
    for k in range(1, K + 1):
        V, W, EL, E = onehop(la, s)
        if name == 'L':
            la2, s2, d, p = lanczos_state(la, s, EL, E, 0.0)
        elif name.startswith('Lc_'):
            la2, s2, d, p = lanczos_state(la, s, EL, E, float(name[3:]))
        elif name == 'SI_1':
            la2, s2, d = si_state(la, s, V, W, E, 1.0); p = 1.0
        else:
            best = None
            for lam in LAMS:
                cand = si_state(la, s, V, W, E, lam); e_ = err(*cand[:2])
                if best is None or e_ < best[0]: best = (e_, cand, lam)
            (la2, s2, d), p = best[1], best[2]
        dfin = jnp.where(jnp.isfinite(d), d, 0.0)
        row = dict(k=k, err=err(la2, s2), param=p, rough=bond_rms(lab, la, dfin),
                   flips=float(jnp.sum(sec.n * (s2 != s)) / jnp.sum(sec.n)),
                   flip_weight=float(jnp.sum(sec.vec(la, s) ** 2 * (s2 != s))))
        rows.append(row); log(name, row)
        la, s = la2 - jnp.max(la2), s2
        res['runs'][name] = rows; dump(F, res)
log('DONE')
