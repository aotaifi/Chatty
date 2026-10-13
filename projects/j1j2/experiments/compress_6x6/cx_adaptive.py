"""Adaptive-depth guide test (README "Adaptive-depth guide", pre-registered before any run).
  python cx_adaptive.py OUT SPEC.json   SPEC: {"k1": hy_k1 run dir, "k2": hy_k2 run dir}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_adaptive.json'); EPS = 0.3; T00 = time.time()
lab = Lab(); sec = lab.sec
G2 = jnp.asarray(np.load(os.path.join(SPEC['k1'], 'G_table.npy')))
s2 = jnp.asarray(np.load(os.path.join(SPEC['k1'], 's_next.npy')).astype(np.float32))
V1 = jnp.asarray(np.load(os.path.join(SPEC['k1'], 'vit_table.npy')))
la_A = jnp.asarray(np.load(os.path.join(SPEC['k2'], 'vit', 'la_table.npy')))
_, V, W = lab.VW(G2, s2); v = sec.vec(G2, s2); E = float(v @ sec.H(v)); PG = v ** 2; del v
ELG = lab.D0 + V - W; del V, W
sdE = float(jnp.sqrt(jnp.sum(PG * (ELG - E) ** 2)))
def build(a):
    f = 1.0 + a * (ELG - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
    return G2 + d, s2 * jnp.where(f >= 0, 1.0, -1.0).astype(s2.dtype)
r_ = minimize_scalar(lambda a: lab.H_site(*build(a)), bounds=(-3.0 / sdE, 0.0), method='bounded', options=dict(xatol=1e-4 / sdE))
la_t, s_t = build(float(r_.x))
H_G, H_t = lab.H_site(G2, s2), lab.H_site(la_t, s_t)
sh = lambda h: (H_G - h) / (H_G - H_t)
_, V0, W0 = lab.VW(V1, s2); EL0 = lab.D0 + V0 - W0
resid = la_t - la_A
Q = lab.proposal_q(resid, la_t, s_t)[0]
c1 = G2 - V1; c1 = c1 - jnp.sum(PG * c1)
scores = {'F1': jnp.abs(jnp.arcsinh((EL0 - E) / sdE)), 'F2': V0 / (1.0 + W0), 'F3': jnp.abs(c1)}
del V0, W0
res = dict(H_G=H_G, H_t=H_t, criteria={})
u_ex = jnp.arcsinh((ELG - E) / sdE)
PT = sec.vec(la_t, s_t) ** 2


def quantile_threshold(sc, p):
    o = jnp.argsort(-sc); cw = jnp.cumsum(PG[o]); k = int(jnp.searchsorted(cw, p))
    return float(sc[o[min(k, sc.shape[0] - 1)]])


nbr_valid = []
rng = np.random.default_rng(9)
for cname, sc in scores.items():
    res['criteria'][cname] = {}
    for p in (0.01, 0.02, 0.05):
        t = quantile_threshold(sc, p); flag = sc >= t
        Emix = jnp.where(flag, ELG, EL0); u_mix = jnp.arcsinh((Emix - E) / sdE)
        rec = dict(threshold=t, p_flag_psi=float(jnp.sum(jnp.where(flag, PG, 0))), p_flag_Q=float(jnp.sum(jnp.where(flag, Q, 0))),
                   p_flag_target=float(jnp.sum(jnp.where(flag, PT, 0))),
                   feat_rms_err_target=float(jnp.sqrt(jnp.sum(PT * (u_mix - u_ex) ** 2))),
                   feat_rms_err_Q=float(jnp.sqrt(jnp.sum(Q * (u_mix - u_ex) ** 2))))
        # branching ratio: flagged samples ~ |psi(G2)|^2 restricted to the flag
        pf = np.array(jnp.where(flag, PG, 0.0), np.float64); pf /= pf.sum()
        idx = jnp.asarray(rng.choice(sec.D, size=2000, p=pf))
        x = sec.reps[idx]; valid = valid_bonds(x)
        y = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
        nflag = jnp.sum(valid & flag[iy], 1)
        rec['branching_r'] = float(jnp.mean(nflag)); rec['n_nb'] = float(jnp.mean(jnp.sum(valid, 1)))
        rr = rec['branching_r']
        rec['cost_hops_6x6'] = (1 + p * rec['n_nb'] / (1 - rr)) if rr < 1 else float('inf')
        rec['cost_hops_8x8_same_p'] = (1 + p * 145 / (1 - rr)) if rr < 1 else float('inf')
        res['criteria'][cname][str(p)] = rec
        log(cname, p, json.dumps(rec)); dump(F, res)

# ---------------- fits: c_2 on one hop of ViT_2 + E_mix column
lanA, VA, WA = lab.VW(la_A, s_t)
vA = sec.vec(la_A, s_t); EA = float(vA @ sec.H(vA)); ELA = lab.D0 + VA - WA
sdA = float(jnp.sqrt(jnp.sum(vA ** 2 * (ELA - EA) ** 2))); del vA
_, _, meas = lab.proposal_q(resid, la_t, s_t)
base = lab.feat_table(lanA, VA, WA, meas)
ua = jnp.arcsinh((ELA - EA) / sdA); m_ = float(jnp.sum(meas * ua)); s_ = float(jnp.sqrt(jnp.sum(meas * (ua - m_) ** 2))) + 1e-30
base = jnp.concatenate([base, ((ua - m_) / s_).astype(jnp.float32)[:, None]], 1); del lanA, VA, WA, ELA, ua
res['fits'] = {}
for arm, Emix in (('p0', EL0), ('F1', None), ('F2', None), ('F3', None)):
    t0 = time.time()
    if Emix is None:
        t = res['criteria'][arm]['0.02']['threshold']; Emix = jnp.where(scores[arm] >= t, ELG, EL0)
    u = jnp.arcsinh((Emix - E) / sdE); m_ = float(jnp.sum(meas * u)); s_ = float(jnp.sqrt(jnp.sum(meas * (u - m_) ** 2))) + 1e-30
    FEAT = jnp.concatenate([base, ((u - m_) / s_).astype(jnp.float32)[:, None]], 1)
    mc, flc, fit = robust_fit(lab, lambda sd_: FeatNet(FEAT, seed=sd_), la_t, s_t, resid, 30000, 1702, 3e-3, 'c' + arm,
                              heldout_curve=True)
    c = mc.table(flc, sec.reps, lab.TIDX) if flc is not None else jnp.zeros_like(la_A)
    Gb = la_A + c
    ex = Gb - la_t; ex = ex - jnp.sum(PT * ex)
    gw = float(jnp.sum(jnp.where(ex > 1.0, sec.vec(Gb, s_t) ** 2, 0.0)))
    if gw > 1e-3: Gb = la_A
    rec = dict(H=lab.H_site(Gb, s_t), guard_weight=gw, heldout=(fit['tries'][-1]['L0_heldout'], fit['tries'][-1]['L_heldout']))
    rec['share'] = sh(rec['H'])
    _, uu, info = sec.fn_solve(Gb, s_t); del uu; rec['E_FN'] = info['dE_FN_site']; rec['sec'] = time.time() - t0
    log(f'== {arm}: share {rec["share"]:.3f}  <H> {rec["H"]:.4e}  E_FN {rec["E_FN"]:.4e}  held-out {rec["heldout"]}')
    res['fits'][arm] = rec; dump(F, res)
    del FEAT, c, Gb, ex
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
