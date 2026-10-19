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
    if SPEC.get('resume_f'):                                # amendment BF2-1: continue from a saved field
        f0 = jnp.asarray(np.load(SPEC['resume_f']), jnp.float64)
        E0, lphi = efn(la_P + f0, s1); la_hat = la_P + f0
    else:
        E0, lphi = efn(la_P, s1); la_hat = la_P
    rounds = []; best = E0
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
        if SPEC.get('save_rounds'): np.save(os.path.join(OUT, f'f_round{k}.npy'), np.asarray(f, np.float32))
    res['start'] = E0; res['best'] = best; res['bad_phi_best'] = bad_counts(lphi, s1)
    dump(F, res)
if MODE == 'errstruct':                                     # amendment BF2-1 add-on
    la_star = jnp.asarray(np.load(SPEC['la_star']))
    fields = [jnp.asarray(np.load(p_), jnp.float64) for p_ in SPEC['f_paths']]
    p = np.array(sec.n * jnp.exp(2 * (la_star - jnp.max(la_star))), np.float64); p /= p.sum()
    idx = jnp.asarray(np.random.default_rng(17).choice(sec.D, size=4096, p=p))
    x = sec.reps[idx]; valid = valid_bonds(x)
    y = jnp.where(valid, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    bad = valid & (s1[idx][:, None] * s1[iy] > 0)
    dstar = la_star[iy] - la_star[idx][:, None]
    w = jnp.where(bad, 0.5 * JB[None, :] * jnp.exp(jnp.clip(dstar, -60, 60)), 0.0); w = w / jnp.sum(w)
    deltas = []
    for fk in fields:
        ah = la_P + fk
        deltas.append(jnp.where(bad, (ah[iy] - ah[idx][:, None]) - dstar, 0.0))
    dl = deltas[-1]
    wrms = float(jnp.sqrt(jnp.sum(w * dl ** 2)))
    mu_d = jnp.sum(w * dl); mu_s = jnp.sum(w * dstar)
    corr = float(jnp.sum(w * (dl - mu_d) * (dstar - mu_s)) / jnp.sqrt(jnp.sum(w * (dl - mu_d) ** 2) * jnp.sum(w * (dstar - mu_s) ** 2)))
    nnmask = (jnp.arange(NB) < len(SS.NN))[None, :]
    bytype = {t: float(jnp.sqrt(jnp.sum(jnp.where(m, w * dl ** 2, 0)) / jnp.sum(jnp.where(m, w, 0)))) for t, m in (('NN', nnmask), ('NNN', ~nnmask))}
    # spatial correlation of delta between bad bonds of the same configuration
    BIn = np.asarray(SS.BI).astype(int); BJn = np.asarray(SS.BJ).astype(int)
    def sdist(p_, r_):
        dx = abs(p_ % 6 - r_ % 6); dy = abs(p_ // 6 - r_ // 6); dx = min(dx, 6 - dx); dy = min(dy, 6 - dy)
        return np.sqrt(dx * dx + dy * dy)
    DB = np.round(np.array([[min(sdist(a_, c_) for a_ in (BIn[a], BJn[a]) for c_ in (BIn[c], BJn[c])) for c in range(NB)] for a in range(NB)]), 3)
    dln = np.asarray(dl); badn = np.asarray(bad); wn = np.asarray(w)
    spatial = {}
    for k_ in sorted(set(DB.ravel().tolist())):
        m = (DB == k_) & ~np.eye(NB, dtype=bool)
        num = 0.0; den1 = 0.0; den2 = 0.0
        for i in range(dln.shape[0]):
            bb = np.nonzero(badn[i])[0]
            if len(bb) < 2: continue
            sub = m[np.ix_(bb, bb)]
            if not sub.any(): continue
            dv = dln[i, bb]; prod = np.outer(dv, dv)[sub]; num += prod.sum(); den1 += (dv ** 2).sum() * sub.sum() / len(bb); 
        spatial[str(k_)] = float(num / max(den1, 1e-30))
    cons = [float(jnp.sum(w * deltas[i] * deltas[i + 1]) / jnp.sqrt(jnp.sum(w * deltas[i] ** 2) * jnp.sum(w * deltas[i + 1] ** 2)))
            for i in range(len(deltas) - 1)]
    res['error'] = dict(wrms=wrms, corr_with_exact=corr, rms_by_type=bytype, spatial_corr_by_distance=spatial,
                        round_to_round_corr=cons)
    log('error structure', json.dumps(res['error'])); dump(F, res)

    # noise test: wall built directly from a per-bond field; custom FN solve with that diagonal
    def wall(la_f, sigma):
        out = []
        for i0 in range(0, sec.D, 16384):
            ii = jnp.arange(i0, min(i0 + 16384, sec.D)); xx = sec.reps[ii]; vv = valid_bonds(xx)
            yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
            jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(xx.shape[0], NB)
            bd = vv & (s1[ii][:, None] * s1[jj] > 0)
            lr = la_f[jj] - la_f[ii][:, None]
            if sigma > 0:
                a_ = jnp.minimum(ii[:, None], jj).astype(jnp.uint32); b_ = jnp.maximum(ii[:, None], jj).astype(jnp.uint32)
                h1 = (a_ * jnp.uint32(2654435761) ^ (b_ * jnp.uint32(40503) + jnp.uint32(12345))) * jnp.uint32(2246822519)
                h2 = (b_ * jnp.uint32(2654435761) ^ (a_ * jnp.uint32(97) + jnp.uint32(777))) * jnp.uint32(3266489917)
                u1 = (h1.astype(jnp.float64) + 0.5) / 4294967296.0; u2 = (h2.astype(jnp.float64) + 0.5) / 4294967296.0
                g = jnp.sqrt(-2 * jnp.log(u1)) * jnp.cos(2 * np.pi * u2)
                sgn = jnp.where(ii[:, None] < jj, 1.0, jnp.where(ii[:, None] > jj, -1.0, 0.0))
                lr = lr + sigma * g * sgn
            out.append(jnp.sum(jnp.where(bd, 0.5 * JB[None, :] * jnp.exp(jnp.clip(lr, -60, 60)), 0.0), 1))
        return lab.D0 + jnp.concatenate(out)

    def fn_D(Dg, s, tol=1e-10, maxit=400):
        import scipy.linalg as sla
        def A(x):
            h1, h2 = sec.H2(x, s * x); return Dg * x - 0.5 * (h1 - s * h2)
        x = sec.vec(la_star, s); x = x * s; x = jnp.abs(x); x = x / jnp.linalg.norm(x)
        Ax = A(x); th = float(x @ Ax); den = jnp.maximum(Dg - th, 1.0); P = None; AP = None
        for it in range(maxit):
            r = Ax - th * x
            if float(jnp.linalg.norm(r)) <= tol * abs(th): break
            z = r / den; z = z - (x @ z) * x
            if P is not None: z = z - (P @ z) * P / (P @ P)
            z = z / jnp.linalg.norm(z); Az = A(z)
            Sb = [x, z] + ([P] if P is not None else []); ASb = [Ax, Az] + ([AP] if P is not None else [])
            m = len(Sb)
            GA = np.array([[float(Sb[i] @ ASb[j]) for j in range(m)] for i in range(m)]); GA = 0.5 * (GA + GA.T)
            GB = np.array([[float(Sb[i] @ Sb[j]) for j in range(m)] for i in range(m)])
            ev, ec = sla.eigh(GA, GB); c = ec[:, 0]
            xn = sum(c[i] * Sb[i] for i in range(m)); Axn = sum(c[i] * ASb[i] for i in range(m))
            P = sum(c[i] * Sb[i] for i in range(1, m)); AP = sum(c[i] * ASb[i] for i in range(1, m))
            nr = float(jnp.linalg.norm(xn)); x = xn / nr; Ax = Axn / nr
            pn = float(jnp.linalg.norm(P))
            if pn > 0: P = P / pn; AP = AP / pn
            th = float(x @ Ax)
        return (th - sec.E0) / N, it
    ah = la_P + fields[-1]
    E_tr_check, _ = efn(ah, s1)
    E_tr_wall, it1 = fn_D(wall(ah, 0.0), s1)
    E_star, it2 = fn_D(wall(la_star, 0.0), s1)
    E_noise, it3 = fn_D(wall(la_star, wrms), s1)
    res['noise_test'] = dict(E_trained=E_tr_check, E_trained_via_wall=E_tr_wall, E_star=E_star, E_star_plus_iid_noise=E_noise,
                             sigma=wrms, iters=[it1, it2, it3])
    log('noise test', json.dumps(res['noise_test'])); dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
