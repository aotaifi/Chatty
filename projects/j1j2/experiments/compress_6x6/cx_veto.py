"""Expulsion / veto / flip test (README "Expulsion / veto test" + amendment V-1, pre-registered before any computation).
  python cx_veto.py OUT SPEC.json   SPEC: {"fp_solves": 40, "lbfgs_evals": 30, "rounds": 2}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar, minimize
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
nW = sec.n / jnp.sum(sec.n)


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


la0n = norm_la(la0)


def S_viol(u, s):
    """per-configuration sum over violating neighbours of |h| u(y)."""
    U = u * sec.sqrt_n; h1, h2 = sec.H2(U, s * U)
    return 0.5 * (h1 + s * h2) / sec.sqrt_n - lab.D0 * u


def solve(la, s):
    """E_FN (per site), log phi (per configuration, normalised), f, D, V."""
    _, u, info = sec.fn_solve(la, s)
    lphi = jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n); del u
    lan = norm_la(la)
    _, Vg, _ = lab.VW(lan, s)                                # wall of the field a_hat (clamped as in fn_solve)
    phi = jnp.exp(lphi)
    D = float(jnp.sum(sec.n * phi ** 2 * Vg) - jnp.sum(sec.n * phi * S_viol(phi, s))) / N
    q0 = np.asarray(lphi - la0n)
    return dict(E=info['dE_FN_site'], D=D, f=info['dE_FN_site'] - D, minq0=float(q0.min()),
                q0_quant={str(p): float(np.quantile(q0, p)) for p in (1e-4, 1e-3, 1e-2)}), lphi, lan, Vg


def grad(lan, lphi, Vg, s):
    phi2 = jnp.exp(2 * lphi)
    t = 2 * lphi - lan; c = float(jnp.max(t))
    Sv = S_viol(jnp.exp(t - c), s)
    g = -phi2 * Vg + jnp.exp(jnp.clip(lan + c, -700, 700)) * Sv
    return sec.n * g


# ---------------------------------------------------------------- (1a) fixed point
r0, lphi, _, _ = solve(la_P, s1); res['EFN_s1_vit'] = r0['E']
fp = [r0]; la_fp = lphi
for k in range(SPEC.get('fp_solves', 40)):
    rk, lphi_k, _, _ = solve(la_fp, s1); fp.append(rk); la_fp = lphi_k
    if k % 5 == 4: log(f'  fp {k + 1}: E {rk["E"]:.6e}  D {rk["D"]:.3e}  min log(phi/psi0) {rk["minq0"]:.1f}'); res['fixed_point'] = fp; dump(F, res)
res['fixed_point'] = fp; dump(F, res)
# ---------------------------------------------------------------- (1b) L-BFGS on a_hat
lb = []; cache = {}
def fun(xv):
    la = jnp.asarray(xv)
    rk, lph, lan, Vg = solve(la, s1)
    g = grad(lan, lph, Vg, s1)
    lb.append(rk); cache['la'] = la; cache['lphi'] = lph
    if len(lb) % 5 == 0: log(f'  lbfgs eval {len(lb)}: E {rk["E"]:.6e}  D {rk["D"]:.3e}'); res['lbfgs'] = lb; dump(F, res)
    return rk['E'] * N, np.asarray(g, np.float64)
opt = minimize(fun, np.asarray(la_P, np.float64), jac=True, method='L-BFGS-B',
               options=dict(maxfun=SPEC.get('lbfgs_evals', 30), maxiter=SPEC.get('lbfgs_evals', 30), maxcor=10))
res['lbfgs'] = lb
best_lb = int(np.argmin([q['E'] for q in lb]))
# phi*: the lower endpoint; for L-BFGS phi* is the FN solution of the best field
if lb[best_lb]['E'] < fp[-1]['E']:
    _, la_star, _, _ = solve(jnp.asarray(opt.x), s1); res['phi_star_from'] = 'lbfgs'
else:
    la_star = la_fp; res['phi_star_from'] = 'fixed_point'
la_star = norm_la(la_star)
res['ceiling'] = min(fp[-1]['E'], lb[best_lb]['E'])
h_ = [q['E'] for q in fp]; a_, b_, c_ = h_[-3:]
res['fp_aitken'] = c_ - (c_ - b_) ** 2 / (c_ - 2 * b_ + a_); res['fp_last_ratio'] = (c_ - b_) / (b_ - a_)
phis = jnp.exp(la_star)
res['W_B_site'] = float(0.5 * jnp.sum(sec.n * phis * S_viol(phis, s1))) / N
log('ceiling', res['ceiling'], 'from', res['phi_star_from'], 'aitken(fp)', res['fp_aitken'], 'W_B', res['W_B_site']); dump(F, res)
np.save(os.path.join(OUT, 'la_star.npy'), np.asarray(la_star))

# ---------------------------------------------------------------- Z, mu, detectors, flips
laPn = norm_la(la_P)
q = jnp.exp(la_star - laPn)
P0 = sec.vec(la0, s0) ** 2; PP = sec.vec(la_P, s_P) ** 2; PS = sec.vec(la_star, s1) ** 2
wrong = s1 != s0
vS = s1 * phis * sec.sqrt_n; mu = s1 * sec.H(vS) / sec.sqrt_n          # (s H s phi*)_x
_, V1, W1 = lab.VW(la_P, s1); v = sec.vec(la_P, s1); E1 = float(v @ sec.H(v)); EL1 = lab.D0 + V1 - W1; del V1, W1
sd1 = float(jnp.sqrt(jnp.sum(v ** 2 * (EL1 - E1) ** 2))); del v
sk = lambda a: s1 * jnp.where(1.0 + a * (EL1 - E1) >= 0, 1.0, -1.0).astype(s1.dtype)
rk_ = minimize_scalar(lambda a: lab.H_site(la_P, sk(a)), bounds=(-3.0 / sd1, 0.0), method='bounded', options=dict(xatol=1e-4 / sd1))
kflip = sk(float(rk_.x)) != s1; del EL1
res['wrong_frac'] = dict(count=float(jnp.sum(nW * wrong)), psi0=float(jnp.sum(P0 * wrong)))


def own_c(s2, n=6):
    rr, lp, _, _ = solve(la_P, s2); hist = [rr['E']]
    for _ in range(n):
        rr, lp, _, _ = solve(lp, s2); hist.append(rr['E'])
    a_, b_, c_ = hist[-3:]
    return hist, c_ - (c_ - b_) ** 2 / (c_ - 2 * b_ + a_), lp


res['taus'] = {}
for tau in (1e-1, 1e-2, 1e-3):
    X = q < tau
    tp = X & wrong
    def pr(wt, a, b): return float(jnp.sum(wt * a) / max(float(jnp.sum(wt * b)), 1e-300))
    rec = dict(frac_count=float(jnp.sum(nW * X)), w_psi0=float(jnp.sum(P0 * X)), w_psiP=float(jnp.sum(PP * X)), w_phi=float(jnp.sum(PS * X)),
               precision=dict(count=pr(nW, tp, X), psi0=pr(P0, tp, X), psiP=pr(PP, tp, X)),
               recall=dict(count=pr(nW, tp, wrong), psi0=pr(P0, tp, wrong), psiP=pr(PP, tp, wrong)),
               mu_pos_frac_in_X=pr(nW, X & (mu > 0), X))
    la_h = jnp.where(X, la_P - 40.0, la_P)
    rec['EFN_veto_vit'] = solve(la_h, s1)[0]['E']
    flips = {'i_all': X, 'ii_krylov': X & kflip, 'iii_oracle': X & wrong, 'iv_kkt': X & (mu > 0)}
    rec['flip'] = {}
    for nm, fl in flips.items():
        s2 = jnp.where(fl, -s1, s1)
        fr = dict(frac=float(jnp.sum(nW * fl)), wrong_after_psi0=float(jnp.sum(P0 * (s2 != s0))))
        if tau == 1e-2:
            hist, ait, _ = own_c(s2); fr['EFN_vit'] = hist[0]; fr['c_history'] = hist; fr['c_aitken'] = ait
        else:
            fr['EFN_vit'] = solve(la_P, s2)[0]['E']
        rec['flip'][nm] = fr
        log(f'tau {tau} {nm}: frac {fr["frac"]:.3e}  wrong-sign weight after {fr["wrong_after_psi0"]:.3e}  E_FN(ViT) {fr["EFN_vit"]:.4e}'
            + (f'  own c(6) {fr["c_history"][-1]:.4e}  aitken {fr["c_aitken"]:.4e}' if 'c_history' in fr else ''))
    res['taus'][str(tau)] = rec
    log(f'tau {tau}: X {rec["frac_count"]:.3e}  veto E_FN {rec["EFN_veto_vit"]:.4e}  P {rec["precision"]}  R {rec["recall"]}'); dump(F, res)

# ---------------------------------------------------------------- iterate the best non-oracle arm (tau = 1e-2)
fl2 = res['taus']['0.01']['flip']
best_arm = min(('i_all', 'ii_krylov', 'iv_kkt'), key=lambda a: fl2[a]['c_history'][-1])
res['iterate'] = dict(arm=best_arm, rounds=[])
if fl2[best_arm]['c_history'][-1] < res['ceiling']:
    s_cur = s1; ls_cur = la_star
    for rnd in range(SPEC.get('rounds', 2) + 1):
        qq = jnp.exp(ls_cur - laPn); Xc = qq < 1e-2
        pc = jnp.exp(ls_cur); vv = s_cur * pc * sec.sqrt_n; muc = s_cur * sec.H(vv) / sec.sqrt_n
        flc = {'i_all': Xc, 'iv_kkt': Xc & (muc > 0)}.get(best_arm)
        if flc is None:                                     # ii_krylov: recompute the Krylov flip set for s_cur
            _, Vc, Wc = lab.VW(la_P, s_cur); v = sec.vec(la_P, s_cur); Ec = float(v @ sec.H(v)); ELc = lab.D0 + Vc - Wc
            sdc = float(jnp.sqrt(jnp.sum(v ** 2 * (ELc - Ec) ** 2))); del v, Vc, Wc
            skc = lambda a: s_cur * jnp.where(1.0 + a * (ELc - Ec) >= 0, 1.0, -1.0).astype(s_cur.dtype)
            rkc = minimize_scalar(lambda a: lab.H_site(la_P, skc(a)), bounds=(-3.0 / sdc, 0.0), method='bounded', options=dict(xatol=1e-4 / sdc))
            flc = Xc & (skc(float(rkc.x)) != s_cur); del ELc
        s_cur = jnp.where(flc, -s_cur, s_cur)
        hist, ait, lp = own_c(s_cur, n=10)
        ls_cur = norm_la(lp)
        rr = dict(round=rnd + 1, frac_flipped=float(jnp.sum(nW * flc)), wrong_psi0=float(jnp.sum(P0 * (s_cur != s0))),
                  EFN_vit=hist[0], c10=hist[-1], aitken=ait)
        res['iterate']['rounds'].append(rr); log('iterate', rr); dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
