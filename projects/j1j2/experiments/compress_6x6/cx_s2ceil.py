"""Amendment V-5: equal-solve ceilings of the flipped signs s_2 (flip X_0.1 all / KKT) and one more KKT round.
  python cx_s2ceil.py OUT SPEC.json   SPEC: {"la_star": path, "solves": 40, "solves3": 30}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_s2ceil.json'); EPS = 0.3; T00 = time.time()
lab = Lab(); sec = lab.sec
z = np.load(TABLE); amp = np.asarray(z['amp'], np.float64); del z
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32)
la0 = jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)); del amp
la_P, s_P = lab.lP0, lab.sP0
_, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
EL = lab.D0 + V - W; del V, W
def sg(a): return s_P * jnp.where(1.0 + a * (EL - E) >= 0, 1.0, -1.0).astype(s_P.dtype)
def lclip(a): return la_P + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (EL - E)), EPS))
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
r_ = minimize_scalar(lambda a: lab.H_site(lclip(a), sg(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
s1 = sg(float(r_.x)); del EL
P0 = sec.vec(la0, s0) ** 2
s0 = s0 * (1.0 if float(jnp.sum(P0 * s0 * s1)) >= 0 else -1.0)
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
laPn = norm_la(la_P)
def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    return info['dE_FN_site'], jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n)
def ceiling(s, n):
    e, lp = efn(la_P, s); hist = [e]
    for k in range(n):
        e, lp = efn(lp, s); hist.append(e)
        if k % 10 == 9: log(f'    solve {k + 1}: {e:.6e}')
    return hist, norm_la(lp)
def kkt_flip(s, ls, tau=0.1):
    q = jnp.exp(ls - laPn); X = q < tau
    ph = jnp.exp(ls); mu = s * sec.H(s * ph * sec.sqrt_n) / sec.sqrt_n
    return X, X & (mu > 0)
res = dict(spec=SPEC, w_s1=float(jnp.sum(P0 * (s1 != s0))))
ls1 = norm_la(jnp.asarray(np.load(SPEC['la_star'])))
X, K = kkt_flip(s1, ls1)
res['arms'] = {}
best = None
for nm, fl in (('s2a_flip_all', X), ('s2b_kkt', K)):
    s2 = jnp.where(fl, -s1, s1)
    log(f'[{nm}] flipped frac {float(jnp.sum(sec.n * fl) / jnp.sum(sec.n)):.3e}')
    hist, ls2 = ceiling(s2, SPEC.get('solves', 40))
    res['arms'][nm] = dict(history=hist, final=hist[-1], w_s=float(jnp.sum(P0 * (s2 != s0))))
    log(f'== {nm}: 40 solves {hist[-1]:.6e}  (s_1: 2.509e-6)  w_s {res["arms"][nm]["w_s"]:.3e}'); dump(F, res)
    if best is None or hist[-1] < best[0]: best = (hist[-1], nm, s2, ls2)
_, nm, s2, ls2 = best
X3, K3 = kkt_flip(s2, ls2)
s3 = jnp.where(K3, -s2, s2)
log(f'[s3 from {nm}] flipped frac {float(jnp.sum(sec.n * K3) / jnp.sum(sec.n)):.3e}')
hist3, _ = ceiling(s3, SPEC.get('solves3', 30))
res['s3'] = dict(base=nm, history=hist3, final=hist3[-1], w_s=float(jnp.sum(P0 * (s3 != s0))))
log(f'== s3: {SPEC.get("solves3", 30)} solves {hist3[-1]:.6e}  w_s {res["s3"]["w_s"]:.3e}')
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
