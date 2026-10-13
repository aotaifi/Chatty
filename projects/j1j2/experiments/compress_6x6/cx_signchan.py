"""Sign-channel check (README "Sign-channel check", pre-registered before any run).
For pairs (fitted guide la_f, exact reference la_ref, same current sign s): next sign step from each amplitude
(rule C: clipped-Lanczos sign, alpha energy-optimal for the clipped state; rule K: same form, alpha minimising <H> at
fixed amplitude), scored by E_FN[la_ref, s'] (common amplitude) and E_FN[la_f, s'_f] (own amplitude); iteration with
the exact FN amplitude of each chain's sign between steps (rule C).
  python cx_signchan.py OUT SPEC.json    SPEC: {"mode": "RA" | "S2", "cap_sec": 1680}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
MODE = SPEC['mode']; CAP = SPEC.get('cap_sec', 1680); EPS = 0.3
F = os.path.join(OUT, f'cx_signchan_{MODE}.json')
RUNS = '/project/theorie/a/A.Otaifi/chatty_compress6/runs'
lab = Lab(); sec = lab.sec
res = dict(spec=SPEC, pairs={})


def EL_of(la, s):
    _, V, W = lab.VW(la, s); v = sec.vec(la, s); E = float(v @ sec.H(v)); del v
    return lab.D0 + V - W, E


def rule_C(la, s):
    EL, E = EL_of(la, s)

    def build(a):
        f = 1.0 + a * (EL - E); d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
        return la + d, s * jnp.where(f >= 0, 1.0, -1.0).astype(s.dtype), d
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))
    r = minimize_scalar(lambda a: lab.H_site(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded',
                        options=dict(xatol=1e-4 / sd))
    la2, s2, d = build(float(r.x))
    return la2, s2, d, float(r.x)


def rule_K(la, s):
    EL, E = EL_of(la, s)
    sg = lambda a: s * jnp.where(1.0 + a * (EL - E) >= 0, 1.0, -1.0).astype(s.dtype)
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))
    r = minimize_scalar(lambda a: lab.H_site(la, sg(a)), bounds=(-3.0 / sd, 0.0), method='bounded',
                        options=dict(xatol=1e-4 / sd))
    return sg(float(r.x)), float(r.x)


def efn(la, s):
    _, u, info = sec.fn_solve(la, s)
    lphi = jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n); del u
    return info['dE_FN_site'], lphi


T0 = time.time()
def time_left():
    return CAP - (time.time() - T0)


# ------------------------------------------------------------------ pairs
la_P, s_P = lab.lP0, lab.sP0
la_t1, s_1, d_1, a_1 = rule_C(la_P, s_P)                        # exact k = 1 target (stage 2 / re-anchor)
pairs = []
if MODE == 'RA':
    for nm, run in (('RA-s1', 'ra_ms_s1_k1_17174477'), ('RA-adam', 'ra_ad_s0_k1_17174451'), ('RA-s0', 'ra_ms_s0_k1_17174476')):
        la_f = jnp.asarray(np.load(os.path.join(RUNS, run, 'la_table.npy')))
        s_chk = np.load(os.path.join(RUNS, run, 's_next.npy')).astype(np.float32)
        assert np.array_equal(s_chk, np.asarray(s_1)), 'sign mismatch'
        pairs.append((nm, la_f, la_t1, s_1))
else:
    run = os.path.join(RUNS, 'c3s0_17165794')
    lan0, V0, W0 = lab.VW(la_P, s_P); EL0 = lab.D0 + V0 - W0
    v = sec.vec(la_P, s_P); E0 = float(v @ sec.H(v)); sd0 = float(jnp.sqrt(jnp.sum(v ** 2 * (EL0 - E0) ** 2))); del v
    _, _, meas0 = lab.proposal_q(lab.T_of(V0, W0, FrozenOp(lab, la_P, s_P).energy(la_P)), la_P, s_P)
    FB = lab.feat_table(lan0, V0, W0, meas0)
    u_ = jnp.arcsinh((EL0 - E0) / sd0); m_ = float(jnp.sum(meas0 * u_)); s_ = float(jnp.sqrt(jnp.sum(meas0 * (u_ - m_) ** 2))) + 1e-30
    FB = jnp.concatenate([FB, ((u_ - m_) / s_).astype(jnp.float32)[:, None]], 1); del u_, lan0, V0, W0, EL0, meas0
    mdl = FeatNet(FB, 0)
    h1 = mdl.table(jnp.asarray(np.load(os.path.join(run, 'params_h1.npy')), jnp.float32), sec.reps, lab.TIDX)
    h2 = mdl.table(jnp.asarray(np.load(os.path.join(run, 'params_h2.npy')), jnp.float32), sec.reps, lab.TIDX)
    del FB, mdl
    G2 = la_P + h1; G3 = G2 + h2; del h1, h2
    s1f = jnp.asarray(np.load(os.path.join(run, 's_1.npy')).astype(np.float32)); assert np.array_equal(np.asarray(s1f), np.asarray(s_1))
    J = json.load(open(os.path.join(run, 'cx_chop3.json')))
    assert abs(lab.H_site(G2, s_1) - J['iters'][1]['guide_H']) < 1e-9
    la_t2, s_2, d_2, a_2 = rule_C(G2, s_1)
    s2f = np.load(os.path.join(run, 's_2.npy')).astype(np.float32); res['s2_match'] = bool(np.array_equal(s2f, np.asarray(s_2)))
    pairs.append(('S2-G3', G3, la_t2, s_2))
    pairs.append(('S2-G2', G2, la_t1, s_1))
log('pairs', [p[0] for p in pairs]); dump(F, res)

# ------------------------------------------------------------------ step 1, rule C then rule K
CH = {}
for rule in ('C', 'K'):
    for nm, la_f, la_r, s in pairs:
        if time_left() < 300: log('time cap: skip', nm, rule); continue
        t0 = time.time()
        P = res['pairs'].setdefault(nm, {})
        if 'base' not in P:
            E0r, _ = efn(la_r, s); E0f, _ = efn(la_f, s)
            P['base'] = dict(H_f=lab.H_site(la_f, s), H_ref=lab.H_site(la_r, s), EFN_ref_s=E0r, EFN_f_s=E0f)
        R = rule_C if rule == 'C' else rule_K
        out_f = R(la_f, s); out_r = R(la_r, s)
        sf, af = (out_f[1], out_f[3]) if rule == 'C' else out_f
        sr, ar = (out_r[1], out_r[3]) if rule == 'C' else out_r
        PTr = sec.vec(la_r, s) ** 2
        Ef, lphi_f = efn(la_r, sf); Er, lphi_r = efn(la_r, sr); Eown, _ = efn(la_f, sf)
        E0 = P['base']['EFN_ref_s']
        rec = dict(alpha_f=af, alpha_r=ar, flip_disagree_weight=float(jnp.sum(PTr * (sf != sr))),
                   flips_f=float(jnp.sum(PTr * (sf != s))), flips_r=float(jnp.sum(PTr * (sr != s))),
                   EFN_common_f=Ef, EFN_common_r=Er, EFN_own_f=Eown, gain_f=E0 - Ef, gain_r=E0 - Er,
                   ratio=(E0 - Ef) / (E0 - Er) if E0 != Er else None, H_newsign_f=lab.H_site(la_f, sf),
                   sec=time.time() - t0)
        P[f'step1_{rule}'] = rec
        log(f'[{nm} {rule}] E_FN common: start {E0:.4e}  fitted-sign {Ef:.4e}  exact-sign {Er:.4e}  ratio {rec["ratio"]:.3f}  '
            f'own {Eown:.4e}  disagree w {rec["flip_disagree_weight"]:.2e}  ({rec["sec"]:.0f}s)')
        dump(F, res)
        if rule == 'C' and nm == pairs[0][0]:
            CH[nm] = dict(f=(lphi_f, sf), r=(lphi_r, sr))
        del lphi_f, lphi_r

# ------------------------------------------------------------------ iteration (rule C), first pair only
nm = pairs[0][0]; P = res['pairs'][nm]
if nm in CH:
    ch = CH[nm]; P['iter_C'] = {'f': [], 'r': []}
    E0 = P['base']['EFN_ref_s']
    for step in (2, 3):
        for c in ('f', 'r'):
            if time_left() < 240: log('time cap: stop iteration'); break
            lphi, sc = ch[c]
            _, s_new, _, a_ = rule_C(lphi, sc)
            E_new, lphi_new = efn(lphi, s_new)
            ch[c] = (lphi_new, s_new)
            P['iter_C'][c].append(dict(step=step, alpha=a_, EFN=E_new, gain=E0 - E_new))
            log(f'[{nm} iter {step} {c}] E_FN {E_new:.4e}  gain {E0 - E_new:.4e}')
            dump(F, res)
res['sec'] = time.time() - T0; dump(F, res); log('DONE', res['sec'])
