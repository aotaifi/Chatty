"""Amendment RA-2 (descriptive): where does the k = 1 re-anchor lose energy? Splices of the fitted table la_new with the
exact target la_t (and the start la_G) by probability decade of the target state; exact <H> of each splice with the
target sign.  python cx_reanchor_diag.py OUT SPEC.json   SPEC: {"runs": [run dirs of k = 1]}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_reanchor_diag.json'); EPS = 0.3
lab = Lab(); sec = lab.sec
la_G, s_G = lab.lP0, lab.sP0
_, V, W = lab.VW(la_G, s_G); _v = sec.vec(la_G, s_G); E = float(_v @ sec.H(_v)); del _v
EL = lab.D0 + V - W; del V, W


def build(alpha):
    f = 1.0 + alpha * (EL - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
    return la_G + d, s_G * jnp.where(f >= 0, 1.0, -1.0).astype(s_G.dtype), d
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_G, s_G) ** 2 * (EL - E) ** 2)))
r = minimize_scalar(lambda a: lab.H_site(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
la_t, s_t, d = build(float(r.x))
H_G, H_t = lab.H_site(la_G, s_G), lab.H_site(la_t, s_t)
PT = sec.vec(la_t, s_t) ** 2
dec, _ = lab.decade_of(la_t)
res = dict(H_G=H_G, H_t=H_t, H_G_newsign=lab.H_site(la_G, s_t), runs={})
sh = lambda h: (H_G - h) / (H_G - H_t)
for run in SPEC['runs']:
    la_n = jnp.asarray(np.load(os.path.join(run, 'la_table.npy')))
    la_n = la_n - jnp.sum(PT * (la_n - la_t))                   # align offsets in the target metric
    la_g = la_G - jnp.sum(PT * (la_G - la_t))
    out = dict(H=lab.H_site(la_n, s_t))
    for t in (-7, -8, -9, -10, -11, -12):
        bulk = dec >= t
        out[f'fit_bulk_target_tail_{t}'] = lab.H_site(jnp.where(bulk, la_n, la_t), s_t)
        out[f'target_bulk_fit_tail_{t}'] = lab.H_site(jnp.where(bulk, la_t, la_n), s_t)
        out[f'fit_bulk_start_tail_{t}'] = lab.H_site(jnp.where(bulk, la_n, la_g), s_t)
        out[f'weight_tail_{t}'] = float(jnp.sum(jnp.where(bulk, 0.0, PT)))
    out['shares'] = {k: sh(v) for k, v in out.items() if k.startswith(('fit', 'target', 'H'))}
    log(os.path.basename(run), json.dumps(out['shares']))
    res['runs'][os.path.basename(run)] = out; dump(F, res)
log('DONE')
