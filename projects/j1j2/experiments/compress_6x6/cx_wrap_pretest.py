"""Lanczos-wrapped network, exact pre-test W0 (README). No fits.
  python cx_wrap_pretest.py OUT SPEC.json   SPEC: {"runs": [re-anchor k = 1 run dirs]}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
from lanczos_lib import lanczos_ritz
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_wrap_pretest.json'); EPS = 0.3
lab = Lab(); sec = lab.sec
la_P, s_P = lab.lP0, lab.sP0
_, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
EL = lab.D0 + V - W; del V, W


def build(a):
    f = 1.0 + a * (EL - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
    return la_P + d, s_P * jnp.where(f >= 0, 1.0, -1.0).astype(s_P.dtype)
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
r = minimize_scalar(lambda a: lab.H_site(*build(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
la_s, s1 = build(float(r.x)); del EL
PT = sec.vec(la_s, s1) ** 2; dec, _ = lab.decade_of(la_s)


def wrap(la, s):
    rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(la, s), 1)
    return (rl[1]['E'] - sec.E0) / N, rl[1]['psi'], rl[1].get('alpha')


def efn_of_vec(psi):
    la = jnp.log(jnp.maximum(jnp.abs(psi), 1e-300) / sec.sqrt_n); s = jnp.where(psi >= 0, 1.0, -1.0).astype(jnp.float32)
    _, u, info = sec.fn_solve(la, s); del u
    return info['dE_FN_site'], float(lab.H_site(la, s))


res = dict(H_star=lab.H_site(la_s, s1), runs={})
Wstar, psi_star, beta = wrap(la_s, s1)
res['W_star'] = Wstar; res['beta_star'] = beta
res['W_star_EFN'], _ = efn_of_vec(psi_star)
# inheritance statistic for the wrapped N*
_, V, W = lab.VW(la_s, s1); vv = sec.vec(la_s, s1); Es = float(vv @ sec.H(vv)); del vv
ELs = lab.D0 + V - W; del V, W
own = jnp.abs(1.0 + beta * (lab.D0 - Es)); inh = jnp.abs(beta * (ELs - lab.D0))
pw = psi_star ** 2; decw, _ = lab.decade_of(jnp.log(jnp.maximum(jnp.abs(psi_star), 1e-300) / sec.sqrt_n))
res['inheritance'] = {int(q): dict(weight=float(jnp.sum(jnp.where(decw == q, pw, 0))),
                                   inherited_dominant_frac=float(jnp.sum(jnp.where((decw == q) & (inh > own), pw, 0)) /
                                                                 max(float(jnp.sum(jnp.where(decw == q, pw, 0))), 1e-300)))
                      for q in range(-4, -16, -1)}
del psi_star, ELs, own, inh, pw
log('W* ', Wstar, 'beta', beta, 'E_FN', res['W_star_EFN']); log('inheritance', res['inheritance']); dump(F, res)
for run in SPEC['runs']:
    la_f = jnp.asarray(np.load(os.path.join(run, 'la_table.npy'))); la_f = la_f - jnp.sum(PT * (la_f - la_s))
    Wf, psi_f, bf = wrap(la_f, s1)
    o = dict(H=lab.H_site(la_f, s1), W=Wf, beta=bf)
    o['W_EFN'], _ = efn_of_vec(psi_f); del psi_f
    shw = lambda w: (Wf - w) / (Wf - Wstar)
    for t in (-8, -9, -10):
        bulk = dec >= t
        wa = wrap(jnp.where(bulk, la_f, la_s), s1)[0]; wb = wrap(jnp.where(bulk, la_s, la_f), s1)[0]
        o[f'cut{t}'] = dict(tail_weight=float(jnp.sum(jnp.where(bulk, 0.0, PT))), fit_bulk_exact_tail=wa,
                            exact_bulk_fit_tail=wb, share_w_a=shw(wa), share_w_b=shw(wb))
    log(os.path.basename(run), json.dumps(o)); res['runs'][os.path.basename(run)] = o; dump(F, res)
log('DONE')
