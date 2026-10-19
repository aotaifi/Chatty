"""Expulsion / veto test (README "Expulsion / veto test", pre-registered before any computation). Sign s_1.
  python cx_veto.py OUT SPEC.json   SPEC: {"la_star": path, "ceiling_json": path}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_veto.json'); EPS = 0.3; T00 = time.time()
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
res = dict(spec=SPEC)
# (1) ceiling + Aitken
cj = json.load(open(SPEC['ceiling_json'])); h = cj['c_history']
e0_, e1_, e2_ = h[-3], h[-2], h[-1]
res['ceiling'] = dict(final=e2_, solves=len(h) - 1, aitken=e2_ - (e2_ - e1_) ** 2 / (e2_ - 2 * e1_ + e0_),
                      ratio_last=(e2_ - e1_) / (e1_ - e0_))
log('ceiling', res['ceiling']); dump(F, res)


def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    return info['dE_FN_site'], jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n)


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


la_star = norm_la(jnp.asarray(np.load(SPEC['la_star'])))
laPn = norm_la(la_P)
q = jnp.exp(la_star - laPn)                                  # phi*/|psi_P|, per configuration
nW = sec.n / jnp.sum(sec.n)
P0 = sec.vec(la0, s0) ** 2; PP = sec.vec(la_P, s_P) ** 2; PS = sec.vec(la_star, s1) ** 2
wrong = s1 != s0
res['q_quantiles_phi'] = {str(p): float(np.quantile(np.asarray(q), p)) for p in (0.001, 0.01, 0.1, 0.5)}
CHW = 2048


@jax.jit
def gap_chunk(i0, tau):
    ii = jnp.minimum(i0 + jnp.arange(CHW), sec.D - 1); real = (i0 + jnp.arange(CHW)) < sec.D
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(CHW, NB)
    bd = vv & (s1[ii][:, None] * s1[jj] > 0) & real[:, None]
    phx = jnp.exp(la_star[ii])[:, None]; phy = jnp.exp(la_star[jj])
    r = jnp.exp(jnp.clip(laPn[jj] - laPn[ii][:, None], -60, 60))
    term = 0.5 * JB[None, :] * (jnp.sqrt(r) * phx - phy / jnp.sqrt(r)) ** 2 * sec.n[ii][:, None] * 0.5   # pair counted from both ends
    qx = q[ii][:, None]; qy = q[jj]
    exp_ = (qx < tau) | (qy < tau)
    near = ~exp_ & ((qx < 0.3) | (qy < 0.3))
    bulk = ~exp_ & ~near
    return jnp.array([jnp.sum(jnp.where(bd & bulk, term, 0)), jnp.sum(jnp.where(bd & near, term, 0)),
                      jnp.sum(jnp.where(bd & exp_, term, 0))])


Ea_s1, _ = efn(la_P, s1); res['EFN_s1_vit'] = Ea_s1
# Krylov flip set (rule K, ViT amplitudes, from s_1)
_, V, W = lab.VW(la_P, s1); v = sec.vec(la_P, s1); E1 = float(v @ sec.H(v)); EL1 = lab.D0 + V - W; del V, W
sd1 = float(jnp.sqrt(jnp.sum(v ** 2 * (EL1 - E1) ** 2))); del v
sk = lambda a: s1 * jnp.where(1.0 + a * (EL1 - E1) >= 0, 1.0, -1.0).astype(s1.dtype)
rk = minimize_scalar(lambda a: lab.H_site(la_P, sk(a)), bounds=(-3.0 / sd1, 0.0), method='bounded', options=dict(xatol=1e-4 / sd1))
kflip = sk(float(rk.x)) != s1; del EL1
res['krylov_flip_frac'] = float(jnp.sum(nW * kflip)); res['wrong_frac'] = float(jnp.sum(nW * wrong))
res['taus'] = {}
for tau in (1e-1, 1e-2, 1e-3):
    t0 = time.time()
    X = q < tau
    rec = dict(frac_count=float(jnp.sum(nW * X)), w_psi0=float(jnp.sum(P0 * X)), w_psiP=float(jnp.sum(PP * X)), w_phi=float(jnp.sum(PS * X)))
    tp = X & wrong
    rec['precision'] = dict(count=float(jnp.sum(nW * tp) / max(float(jnp.sum(nW * X)), 1e-300)),
                            psi0=float(jnp.sum(P0 * tp) / max(float(jnp.sum(P0 * X)), 1e-300)),
                            psiP=float(jnp.sum(PP * tp) / max(float(jnp.sum(PP * X)), 1e-300)))
    rec['recall'] = dict(count=float(jnp.sum(nW * tp) / max(float(jnp.sum(nW * wrong)), 1e-300)),
                         psi0=float(jnp.sum(P0 * tp) / max(float(jnp.sum(P0 * wrong)), 1e-300)),
                         psiP=float(jnp.sum(PP * tp) / max(float(jnp.sum(PP * wrong)), 1e-300)))
    g = sum(gap_chunk(i0, tau) for i0 in range(0, sec.D, CHW)) / N
    rec['gap_decomp_site'] = dict(bulk=float(g[0]), near_node=float(g[1]), into_expelled=float(g[2]))
    la_h = jnp.where(X, la_P - 40.0, la_P)
    rec['EFN_veto'], _ = efn(la_h, s1)
    flips = {'i_all': X, 'ii_krylov': X & kflip, 'iii_oracle': X & wrong}
    rec['flip'] = {}
    for nm, fl in flips.items():
        s2 = jnp.where(fl, -s1, s1)
        e2, lph = efn(la_P, s2)
        fr = dict(frac=float(jnp.sum(nW * fl)), EFN_vit=e2, w_s_change=float(jnp.sum(P0 * (s2 != s0))) - float(jnp.sum(P0 * wrong)))
        if tau == 1e-2:
            hist = [e2]
            for _ in range(6):
                e_, lph = efn(lph, s2); hist.append(e_)
            a_, b_, c_ = hist[-3:]
            fr['c_history'] = hist; fr['c_aitken'] = c_ - (c_ - b_) ** 2 / (c_ - 2 * b_ + a_)
        rec['flip'][nm] = fr
        log(f'tau {tau} flip {nm}: frac {fr["frac"]:.3e}  E_FN(ViT ratios) {e2:.4e}' + (f'  c6 {fr["c_history"][-1]:.4e} aitken {fr["c_aitken"]:.4e}' if 'c_history' in fr else ''))
    rec['sec'] = time.time() - t0
    res['taus'][str(tau)] = rec
    log(f'tau {tau}: X frac {rec["frac_count"]:.3e}  veto E_FN {rec["EFN_veto"]:.4e}  gap {rec["gap_decomp_site"]}  P {rec["precision"]}  R {rec["recall"]}')
    dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
