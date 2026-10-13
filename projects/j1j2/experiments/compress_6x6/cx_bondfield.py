"""Bond-field fixed node (idea 1) and Delta_B (idea 2), exact 6x6 kill test (README, pre-registered before any run).
  python cx_bondfield.py OUT SPEC.json   SPEC: {"mode": "sign", "sign": "sP" | "s1"}  or  {"mode": "deltaB"}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
MODE = SPEC['mode']; EPS = 0.3; T00 = time.time()
F = os.path.join(OUT, f'cx_bondfield_{MODE}_{SPEC.get("sign", "")}.json')
lab = Lab(); sec = lab.sec
res = dict(spec=SPEC)
z = np.load(TABLE); amp = np.asarray(z['amp'], np.float64); del z
la0 = jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)); s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32)
del amp
la_P, s_P = lab.lP0, lab.sP0


def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    lphi = jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n); del u
    return info['dE_FN_site'], lphi


if MODE == 'deltaB':
    from scipy.sparse.linalg import eigsh, LinearOperator
    D0 = lab.D0
    def mv(x):
        xj = jnp.asarray(np.asarray(x).reshape(-1), jnp.float64)
        return np.asarray(2.0 * D0 * xj - sec.H(xj))
    t0 = time.time()
    ev, vec = eigsh(LinearOperator((sec.D, sec.D), matvec=mv, dtype=np.float64), k=1, which='SA', tol=1e-10)
    v = vec[:, 0]; pos = float(np.sum(v > 0)); neg = float(np.sum(v < 0))
    EB = float(ev[0]); res.update(E_B=EB, E0=float(sec.E0), deltaB_site=(sec.E0 - EB) / N, E_B_site=EB / N,
                                  E0_site=float(sec.E0) / N, perron_pos=pos, perron_neg=neg, sec=time.time() - t0)
    log('Delta_B', res); dump(F, res)
    # sanity on the exact sign
    Eb0, _ = efn(la0, s0); res['sanity_EFN_exact_amp_exact_sign'] = Eb0
    Ea0, _ = efn(la_P, s0); res['EFN_vitamp_exact_sign'] = Ea0
    log('sanity', Eb0, Ea0); dump(F, res); sys.exit(0)

# ---------------------------------------------------------------- the sign
if SPEC['sign'] == 'sP':
    s = s_P
else:
    _, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
    EL = lab.D0 + V - W; del V, W
    def build(a):
        f = 1.0 + a * (EL - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
        return la_P + d, s_P * jnp.where(f >= 0, 1.0, -1.0).astype(s_P.dtype)
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
    r_ = minimize_scalar(lambda a: lab.H_site(*build(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
    _, s = build(float(r_.x)); del EL
res['w_s'] = float(max(0.0, (1 - abs(float(jnp.sum(sec.p0 * s * sec.s0)))) / 2)) if sec.p0 is not None else None
Ea, lphi = efn(la_P, s); Eb, _ = efn(la0, s)
res['a_vit_ratios'] = Ea; res['b_exact_ratios'] = Eb
log(f'[{SPEC["sign"]}] (a) {Ea:.4e}  (b) {Eb:.4e}'); dump(F, res)
hist = [Ea]; Ep = Ea
for it in range(20):
    En, lphi_n = efn(lphi, s); hist.append(En)
    log(f'  (c) iter {it + 1}: {En:.6e}')
    lphi = lphi_n
    if abs(Ep - En) < 1e-8: break
    Ep = En
res['c_selfconsistent'] = hist[-1]; res['c_history'] = hist; dump(F, res)
la_star = lphi
# ---------------------------------------------------------------- (d) trained bond-ratio field on bad bonds only
target = la_star - la_P
_, _, meas = lab.proposal_q(target, la_star, s)
m_ = float(jnp.sum(meas * la_P)); s_ = float(jnp.sqrt(jnp.sum(meas * (la_P - m_) ** 2)))
FEATS = {'d1': ((la_P - m_) / s_).astype(jnp.float32)[:, None]}
lan0, V0, W0 = lab.VW(la_P, s_P); EL0 = lab.D0 + V0 - W0
v = sec.vec(la_P, s_P); E0P = float(v @ sec.H(v)); sd0 = float(jnp.sqrt(jnp.sum(v ** 2 * (EL0 - E0P) ** 2))); del v
u_ = jnp.arcsinh((EL0 - E0P) / sd0); mu = float(jnp.sum(meas * u_)); su = float(jnp.sqrt(jnp.sum(meas * (u_ - mu) ** 2))) + 1e-30
FEATS['d2'] = jnp.concatenate([lab.feat_table(lan0, V0, W0, meas), ((u_ - mu) / su).astype(jnp.float32)[:, None]], 1)
del lan0, V0, W0, EL0, u_
res['d'] = {}
for arm, FE in FEATS.items():
    t0 = time.time()
    mf, flf, fit = robust_fit(lab, lambda sd_: FeatNet(FE, seed=sd_), la_star, s, target, 30000, 811, 3e-3, arm,
                              heldout_curve=True, which='viol')
    f = mf.table(flf, sec.reps, lab.TIDX) if flf is not None else jnp.zeros_like(la_P)
    Ed, _ = efn(la_P + f, s)
    rec = dict(E_FN=Ed, recovered=(Ea - Ed) / (Ea - res['c_selfconsistent']),
               heldout=(fit['tries'][-1]['L0_heldout'], fit['tries'][-1]['L_heldout']), sec=time.time() - t0)
    log(f'== {arm}: E_FN {Ed:.4e}  recovered {rec["recovered"]:.3f}  held-out {rec["heldout"]}')
    res['d'][arm] = rec; dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
