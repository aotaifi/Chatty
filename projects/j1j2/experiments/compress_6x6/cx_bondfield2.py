"""Bond-field FN round 2 (README "Bond-field FN, round 2", pre-registered before any run). Sign s_1.
  mode 'ceiling': (c) up to 60 self-consistent solves; bad bonds per configuration under |phi|^2 and |psi_P|^2.
  mode 'refit'  : iterate-and-refit with the exact majoriser (cosh) loss on bad bonds; arm 'A' or 'B'.
  python cx_bondfield2.py OUT SPEC.json   SPEC: {"mode": ..., "arm": "A"|"B", "rounds": 6, "steps": 30000}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
MODE = SPEC['mode']; EPS = 0.3; T00 = time.time()
F = os.path.join(OUT, f'cx_bondfield2_{MODE}_{SPEC.get("arm", "")}.json')
lab = Lab(); sec = lab.sec
la_P, s_P = lab.lP0, lab.sP0
_, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
EL = lab.D0 + V - W; del V, W
def build(a):
    f = 1.0 + a * (EL - E)
    return s_P * jnp.where(f >= 0, 1.0, -1.0).astype(s_P.dtype)
def build_la(a):
    f = 1.0 + a * (EL - E); return la_P + jnp.log(jnp.maximum(jnp.abs(f), EPS))
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
r_ = minimize_scalar(lambda a: lab.H_site(build_la(a), build(a)), bounds=(-3.0 / sd, 0.0), method='bounded',
                     options=dict(xatol=1e-4 / sd))
s1 = build(float(r_.x)); del EL
res = dict(spec=SPEC)


def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    lphi = jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n); del u
    return info['dE_FN_site'], lphi


def bad_counts(la_meas, s, n=4096, seed=3):
    p = np.array(sec.n * jnp.exp(2 * (la_meas - jnp.max(la_meas))), np.float64); p /= p.sum()
    idx = jnp.asarray(np.random.default_rng(seed).choice(sec.D, size=n, p=p))
    x = sec.reps[idx]; valid = valid_bonds(x)
    y = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(n, NB)
    bad = valid & (s[idx][:, None] * s[iy] > 0)
    nb_ = np.asarray(jnp.sum(bad, 1)); nv = np.asarray(jnp.sum(valid, 1))
    return dict(mean_bad=float(nb_.mean()), median_bad=float(np.median(nb_)), p90_bad=float(np.percentile(nb_, 90)),
                mean_valid=float(nv.mean()), frac_bad=float(nb_.sum() / nv.sum()))


if MODE == 'ceiling':
    Ea, lphi = efn(la_P, s1); hist = [Ea]; Ep = Ea
    res['bad_psiP'] = bad_counts(la_P, s1)
    res['bad_phi_a'] = bad_counts(lphi, s1)
    log('bad bonds', res['bad_psiP'], res['bad_phi_a']); dump(F, res)
    for it in range(60):
        En, lphi_n = efn(lphi, s1); hist.append(En); lphi = lphi_n
        if it % 5 == 4: log(f'  (c) solve {it + 1}: {En:.6e}'); res['c_history'] = hist; dump(F, res)
        if abs(Ep - En) < 1e-9: break
        Ep = En
    res['c_history'] = hist; res['c_final'] = hist[-1]; res['c_solves'] = len(hist) - 1
    res['bad_phi_c'] = bad_counts(lphi, s1)
    np.save(os.path.join(OUT, 'la_star.npy'), np.asarray(lphi))
    log('ceiling', hist[-1], res['bad_phi_c']); dump(F, res)

if MODE == 'refit':
    ARM = SPEC['arm']
    lanP, VP, WP = lab.VW(la_P, s1)                        # psi_P one-hop, neighbours split by s_1
    v = sec.vec(la_P, s1); EP1 = float(v @ sec.H(v)); ELP = lab.D0 + VP - WP
    sdP = float(jnp.sqrt(jnp.sum(v ** 2 * (ELP - EP1) ** 2))); del v
    E0, lphi = efn(la_P, s1)
    la_hat = la_P; rounds = []; best = E0
    log(f'[{ARM}] start E_FN {E0:.4e}')
    for k in range(1, SPEC.get('rounds', 6) + 1):
        t0 = time.time()
        target = lphi - la_P
        _, _, meas = lab.proposal_q(target, lphi, s1)

        def stdcol(c):
            m_ = float(jnp.sum(meas * c)); s_ = float(jnp.sqrt(jnp.sum(meas * (c - m_) ** 2))) + 1e-30
            return ((c - m_) / s_).astype(jnp.float32)[:, None]
        cols = [lab.feat_table(lanP, VP, WP, meas), stdcol(jnp.arcsinh((ELP - EP1) / sdP))]
        if ARM == 'B':
            lanF, VF, WF = lab.VW(lphi, s1)
            v = sec.vec(lphi, s1); EF = float(v @ sec.H(v)); ELF = lab.D0 + VF - WF
            sdF = float(jnp.sqrt(jnp.sum(v ** 2 * (ELF - EF) ** 2))); del v
            cols += [lab.feat_table(lanF, VF, WF, meas), stdcol(jnp.arcsinh((ELF - EF) / sdF))]
            del lanF, VF, WF, ELF
        FEAT = jnp.concatenate(cols, 1); del cols
        mf, flf, fit = robust_fit(lab, lambda sd_: FeatNet(FEAT, seed=sd_), lphi, s1, target, SPEC.get('steps', 30000),
                                  900 + k, 1e-3, f'{ARM}{k}', heldout_curve=True, normalize=False, which='viol',
                                  loss_kind='cosh')
        if flf is None:
            rounds.append(dict(round=k, accepted=False, reason='fit rejected')); log(f'  round {k}: fit rejected'); break
        f = mf.table(flf, sec.reps, lab.TIDX)
        cand = la_P + f
        Ek, lphi_k = efn(cand, s1)
        acc = Ek < best
        rec = dict(round=k, E_FN=Ek, accepted=bool(acc), heldout=(fit['tries'][-1]['L0_heldout'], fit['tries'][-1]['L_heldout']),
                   sec=time.time() - t0)
        rounds.append(rec)
        log(f'  round {k}: E_FN {Ek:.4e} (best {best:.4e}) accepted {acc}  held-out {rec["heldout"]}  ({rec["sec"]:.0f}s)')
        res['rounds'] = rounds; res['best'] = min(best, Ek); dump(F, res)
        if not acc: break
        best = Ek; la_hat = cand; lphi = lphi_k
        np.save(os.path.join(OUT, 'f_best.npy'), np.asarray(f, np.float32))
    res['start'] = E0; res['best'] = best; res['bad_phi_best'] = bad_counts(lphi, s1)
    dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
