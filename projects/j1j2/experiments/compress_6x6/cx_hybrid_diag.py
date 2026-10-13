"""Amendment HY-1: why does the hybrid drop at k = 2?  (a) decade splices of the hybrid k = 2 guide with the exact
target; (b) correction refit with + asinh E_L of the guide G_2 (information upper bound).
  python cx_hybrid_diag.py OUT SPEC.json   SPEC: {"k1": run dir hy_k1, "k2": run dir hy_k2}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_hybrid_diag.json'); EPS = 0.3
lab = Lab(); sec = lab.sec
G2 = jnp.asarray(np.load(os.path.join(SPEC['k1'], 'G_table.npy')))
s2 = jnp.asarray(np.load(os.path.join(SPEC['k1'], 's_next.npy')).astype(np.float32))
_, V, W = lab.VW(G2, s2); v = sec.vec(G2, s2); E = float(v @ sec.H(v)); del v
EL2 = lab.D0 + V - W; del V, W
sdE = float(jnp.sqrt(jnp.sum(sec.vec(G2, s2) ** 2 * (EL2 - E) ** 2)))


def build(a):
    f = 1.0 + a * (EL2 - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
    return G2 + d, s2 * jnp.where(f >= 0, 1.0, -1.0).astype(s2.dtype)
r = minimize_scalar(lambda a: lab.H_site(*build(a)), bounds=(-3.0 / sdE, 0.0), method='bounded', options=dict(xatol=1e-4 / sdE))
la_t, s_t = build(float(r.x))
H_G, H_t = lab.H_site(G2, s2), lab.H_site(la_t, s_t)
sh = lambda h: (H_G - h) / (H_G - H_t)
la_A = jnp.asarray(np.load(os.path.join(SPEC['k2'], 'vit', 'la_table.npy')))
G3 = jnp.asarray(np.load(os.path.join(SPEC['k2'], 'G_table.npy')))
PT = sec.vec(la_t, s_t) ** 2; dec, _ = lab.decade_of(la_t)
res = dict(H_G=H_G, H_t=H_t, share_hybrid=sh(lab.H_site(G3, s_t)), splices={})
G3a = G3 - jnp.sum(PT * (G3 - la_t))
for t in (-8, -9, -10, -11, -12):
    bulk = dec >= t
    res['splices'][t] = dict(tail_weight=float(jnp.sum(jnp.where(bulk, 0.0, PT))),
                             fit_bulk_target_tail=sh(lab.H_site(jnp.where(bulk, G3a, la_t), s_t)),
                             target_bulk_fit_tail=sh(lab.H_site(jnp.where(bulk, la_t, G3a), s_t)))
log('splices', res['splices']); dump(F, res)
# (b) correction with + asinh E_L of G2
resid = la_t - la_A
lanA, VA, WA = lab.VW(la_A, s_t)
vA = sec.vec(la_A, s_t); EA = float(vA @ sec.H(vA)); ELA = lab.D0 + VA - WA
sdA = float(jnp.sqrt(jnp.sum(vA ** 2 * (ELA - EA) ** 2))); del vA
_, _, meas = lab.proposal_q(resid, la_t, s_t)
cols = [lab.feat_table(lanA, VA, WA, meas)]
for u_ in (jnp.arcsinh((ELA - EA) / sdA), jnp.arcsinh((EL2 - E) / sdE)):
    m_ = float(jnp.sum(meas * u_)); s_ = float(jnp.sqrt(jnp.sum(meas * (u_ - m_) ** 2))) + 1e-30
    cols.append(((u_ - m_) / s_).astype(jnp.float32)[:, None])
FEAT = jnp.concatenate(cols, 1); del cols, lanA, VA, WA, ELA, meas
mc, flc, fit = robust_fit(lab, lambda sd_: FeatNet(FEAT, seed=sd_), la_t, s_t, resid, 30000, 1702, 3e-3, 'cEL',
                          heldout_curve=True)
c = mc.table(flc, sec.reps, lab.TIDX) if flc is not None else jnp.zeros_like(la_A)
Gb = la_A + c
res['upper_bound'] = dict(H=lab.H_site(Gb, s_t), share=sh(lab.H_site(Gb, s_t)), fit=fit)
_, u, info = sec.fn_solve(Gb, s_t); del u; res['upper_bound']['E_FN'] = info['dE_FN_site']
log('upper bound', {k: v for k, v in res['upper_bound'].items() if k != 'fit'})
dump(F, res); log('DONE')
