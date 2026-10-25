"""Fold-back diagnostics D1 / D2 (README "Fold-back diagnostics D1 / D2", pre-registered before any run).
State: stage-2 seed-0 main arm (run c3s0_17165794), iteration k = 3: G_3 = log|psi_P| + h_1 + h_2, sign s_3 = s_2.npy,
exact clipped-Lanczos target d_3 (as cx_chop3).
  mode 'D1': re-base fits of the EXACT d_3 (metric of the target state), arms R0 (psi_P one-hop features + asinh E_L,P),
             RA (+ asinh E_L of G_3), RB (R0, target scaled to unit rms), RC (R0, 120k steps). Retention vs exact target.
  mode 'D2': exact light-cone analysis, no fits. For x ~ |psi_t|^2 and x ~ Q (fit proposal): E_L of G_3 at x rebuilt with
             G_3(x^b) = la_P(x^b) + h_1 + h_2 evaluated on psi_P's one-hop features of x^b, where psi_P at x^{bb'} is exact
             for bond distance d(b,b') <= rho and additive (la(x^b)+la(x^b')-la(x), s(x^b)s(x^b')s(x)) beyond.
             D2a: ratio / sign interaction of psi_P vs bond distance on x ~ |psi_P|^2.
  python cx_lightcone.py OUT SPEC.json      SPEC: {"mode": "D1"|"D2", "run": RUNDIR, "arms": [...], "n2": 2048, "n2a": 1024}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
MODE = SPEC['mode']
RUN = SPEC.get('run', '/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/c3s0_17165794')
K = 3; EPS = 0.3
F = os.path.join(OUT, f'cx_lightcone_{MODE}.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
res = dict(spec=SPEC, run=RUN, k=K)

# ------------------------------------------------------------ psi_P base features (exactly cx_chop3 with base_EL), constants kept
lan0, V0, W0 = lab.VW(lab.lP0, lab.sP0); EL0 = lab.D0 + V0 - W0
_v = sec.vec(lab.lP0, lab.sP0); E0 = float(_v @ sec.H(_v)); sd0 = float(jnp.sqrt(jnp.sum(_v ** 2 * (EL0 - E0) ** 2))); del _v
Eg0 = FrozenOp(lab, lab.lP0, lab.sP0).energy(lab.lP0)
_, _, meas0 = lab.proposal_q(lab.T_of(V0, W0, Eg0), lab.lP0, lab.sP0)
MAXLP = float(jnp.max(lab.lP0))
raw = [lan0, jnp.log(jnp.maximum(V0, 0) + 1e-6), jnp.log(jnp.maximum(W0, 0) + 1e-6), jnp.arcsinh((EL0 - E0) / sd0)]
CM, CS = [], []
for c_ in raw:
    m_ = float(jnp.sum(meas0 * c_)); s_ = float(jnp.sqrt(jnp.sum(meas0 * (c_ - m_) ** 2))) + 1e-30
    CM.append(m_); CS.append(s_)
FEAT_BASE = jnp.stack([((c_ - m_) / s_).astype(jnp.float32) for c_, m_, s_ in zip(raw, CM, CS)], 1)
_chk = float(jnp.max(jnp.abs(FEAT_BASE[:, :3] - lab.feat_table(lan0, V0, W0, meas0))))
log('feature table check (vs feat_table)', _chk); assert _chk < 1e-5
del raw, lan0, V0, W0, EL0

# ------------------------------------------------------------ the k = 3 state
model0 = FeatNet(FEAT_BASE, 0)
PH = [jnp.asarray(np.load(os.path.join(RUN, f'params_h{j}.npy')), jnp.float32) for j in (1, 2, 3)]
h1 = model0.table(PH[0], sec.reps, lab.TIDX); h2 = model0.table(PH[1], sec.reps, lab.TIDX)
h3 = model0.table(PH[2], sec.reps, lab.TIDX)
G3 = lab.lP0 + h1 + h2; del h1, h2
s3 = jnp.asarray(np.load(os.path.join(RUN, 's_2.npy')).astype(np.float32))
JR = json.load(open(os.path.join(RUN, 'cx_chop3.json')))
H_G3 = lab.H_site(G3, s3)
log(f'<H>(G_3) {H_G3:.10e}  json {JR["iters"][2]["guide_H"]:.10e}')
assert abs(H_G3 - JR['iters'][2]['guide_H']) < 1e-9


def clipped_lanczos(la, s, EL, E):
    def build(alpha):
        f = 1.0 + alpha * (EL - E)
        d = jnp.log(jnp.maximum(jnp.abs(f), EPS))
        return la + d, s * jnp.where(f >= 0, 1.0, -1.0).astype(s.dtype), d
    sd = float(jnp.sqrt(jnp.sum(sec.vec(la, s) ** 2 * (EL - E) ** 2)))
    r = minimize_scalar(lambda a: lab.H_site(*build(a)[:2]), bounds=(-3.0 / sd, 0.0), method='bounded',
                        options=dict(xatol=1e-4 / sd))
    la2, s2, d = build(float(r.x))
    return la2, s2, d, float(r.x), sd


lan3, V3, W3 = lab.VW(G3, s3); del lan3
_v = sec.vec(G3, s3); E3 = float(_v @ sec.H(_v)); del _v
EL3 = lab.D0 + V3 - W3; del V3, W3
la_t, s_t, d3, alpha, sdE = clipped_lanczos(G3, s3, EL3, E3)
H_t = lab.H_site(la_t, s_t)
log(f'alpha {alpha:.5f} (json {JR["iters"][2]["alpha"]:.5f})  <H> target {H_t:.10e}  json {JR["iters"][2]["H_target"]:.10e}')
assert abs(H_t - JR['iters'][2]['H_target']) < 1e-8
PT = sec.vec(la_t, s_t) ** 2
H_main = lab.H_site(G3 + h3, s_t)
res['state'] = dict(H_G3=H_G3, H_target=H_t, alpha=alpha, E=E3, sdE=sdE,
                    main_arm=dict(H=H_main, retention_vs_exact=(H_G3 - H_main) / (H_G3 - H_t),
                                  bond_rms=bond_rms(lab, la_t, G3 + h3 - la_t)))
log('state', res['state']); dump(F, res)


def ptwise_rms(e):
    e = e - jnp.sum(PT * e)
    return float(jnp.sqrt(jnp.sum(PT * e * e)))


# ============================================================================================ D1
if MODE == 'D1':
    res['arms'] = []
    for arm in SPEC.get('arms', ['R0', 'RA', 'RB', 'RC']):
        t0 = time.time(); FB = FEAT_BASE; target = d3; scale = 1.0; steps = 40000
        if arm == 'RA':
            u_ = jnp.arcsinh((EL3 - E3) / sdE)
            m_ = float(jnp.sum(meas0 * u_)); s_ = float(jnp.sqrt(jnp.sum(meas0 * (u_ - m_) ** 2))) + 1e-30
            FB = jnp.concatenate([FEAT_BASE, ((u_ - m_) / s_).astype(jnp.float32)[:, None]], 1); del u_
        if arm == 'RB':
            scale = ptwise_rms(d3); target = d3 / scale
        if arm == 'RC':
            steps = 120000
        mr, flr, fit = robust_fit(lab, lambda sd_: FeatNet(FB, seed=sd_), la_t, s_t, target, steps, 50 + K, 1e-3, arm,
                                  heldout_curve=True, normalize=False)      # as run (before the default changed)
        h = scale * mr.table(flr, sec.reps, lab.TIDX) if flr is not None else jnp.zeros_like(G3)
        if flr is not None: np.save(os.path.join(OUT, f'params_{arm}.npy'), np.asarray(flr))
        Hn = lab.H_site(G3 + h, s_t)
        tr = fit['tries'][-1]
        rec = dict(arm=arm, steps=steps, scale=scale, H=Hn, retention=(H_G3 - Hn) / (H_G3 - H_t),
                   bond_rms=bond_rms(lab, la_t, G3 + h - la_t), ptwise_rms=ptwise_rms(h - d3),
                   heldout_start=tr['L0_heldout'], heldout_final=tr['L_heldout'],
                   heldout_ratio=tr['L_heldout'] / tr['L0_heldout'], fit=fit, sec=time.time() - t0)
        log(f'== {arm}: retention {rec["retention"]:.3f}  <H> {Hn:.4e}  bond-rms {rec["bond_rms"]:.4f}  '
            f'held-out ratio {rec["heldout_ratio"]:.3g}  ({rec["sec"]:.0f}s)')
        res['arms'].append(rec); dump(F, res)
        del mr, h, FB

# ============================================================================================ D2
if MODE == 'D2':
    Qfit = lab.proposal_q(d3, la_t, s_t)[0]
    PP = sec.vec(lab.lP0, lab.sP0) ** 2
    sec.offload()
    canon_j = jax.jit(SS.canon)
    NBj = NB; JBn = np.asarray(SS.JB)
    # bond distances: minimum periodic site distance (0 = shared site)
    BIn = np.asarray(SS.BI).astype(int); BJn = np.asarray(SS.BJ).astype(int)

    def sdist(p, q):
        dx = abs(p % 6 - q % 6); dy = abs(p // 6 - q // 6)
        dx = min(dx, 6 - dx); dy = min(dy, 6 - dy)
        return np.sqrt(dx * dx + dy * dy)
    DB = np.array([[min(sdist(p, q) for p in (BIn[b], BJn[b]) for q in (BIn[c], BJn[c])) for c in range(NBj)]
                   for b in range(NBj)])
    res['bond_distances'] = sorted(set(np.round(DB.ravel(), 4).tolist()))
    LEVELS = [('0th', None), ('Fx', None), ('rho0', 0.0), ('rho1', 1.0), ('rho1.41', 1.4143), ('rho2', 2.0001),
              ('exact', 99.0)]
    DBj = jnp.asarray(DB)

    def lookup(conf):
        M = conf.shape[0]; B = 32768; out = []
        for i in range(0, M, B):
            c = conf[i:i + B]; m = c.shape[0]
            if m < B: c = jnp.concatenate([c, jnp.repeat(c[:1], B - m)])
            out.append(canon_j(sec.T, sec.reps, c)[:m])
        return jnp.concatenate(out)

    net, unravel = model0.net, model0.unravel

    @jax.jit
    def hnet(flat, Sx, ft):
        p = unravel(flat); X = bits(Sx)
        out = jax.lax.map(lambda k: net.apply({'params': p}, X[:, INV8[k // 2]] * SG2[k % 2], ft), jnp.arange(16))
        return out.mean(0).astype(f64)

    @jax.jit
    def feats_of(la_y, s_y, la_yy, s_yy, vy):
        """psi_P one-hop features of y (standardised exactly as FEAT_BASE). la_yy/s_yy: (..., NB) at y^{b'}."""
        r = jnp.exp(jnp.clip(la_yy - la_y[..., None], -60, 60))
        viol = s_y[..., None] * s_yy > 0
        Jb = JB
        V = jnp.sum(jnp.where(vy & viol, 0.5 * Jb * r, 0.0), -1)
        W = jnp.sum(jnp.where(vy & ~viol, 0.5 * Jb * r, 0.0), -1)
        D0y = 0.25 * jnp.sum(Jb) - 0.5 * jnp.sum(vy * Jb, -1)
        ELy = D0y + V - W
        cols = [la_y - MAXLP, jnp.log(jnp.maximum(V, 0) + 1e-6), jnp.log(jnp.maximum(W, 0) + 1e-6),
                jnp.arcsinh((ELy - E0) / sd0)]
        return jnp.stack([(c - m) / s for c, m, s in zip(cols, CM, CS)], -1).astype(jnp.float32)

    def dfun(E_):
        return jnp.log(jnp.maximum(jnp.abs(1.0 + alpha * (E_ - E3)), EPS))

    def sample(prob, n, seed):
        p = np.asarray(prob, np.float64); p = p / p.sum()
        return np.random.default_rng(seed).choice(sec.D, size=n, p=p)

    def chunk_data(idx):
        x = sec.reps[jnp.asarray(idx)]; C = x.shape[0]
        vx = valid_bonds(x)
        xb = x[:, None] ^ MASKS[None, :]
        ib = lookup(xb.reshape(-1)).reshape(C, NBj)
        xbb = xb[:, :, None] ^ MASKS[None, None, :]
        vy = valid_bonds(xb.reshape(-1)).reshape(C, NBj, NBj)
        ibb = lookup(xbb.reshape(-1)).reshape(C, NBj, NBj)
        ix = jnp.asarray(idx)
        return x, vx, xb, ib, vy, ibb, ix

    # ---------------- D2a: ratio / sign interaction vs bond distance, x ~ |psi_P|^2
    n2a = SPEC.get('n2a', 1024); CH = 64
    idxA = sample(PP, n2a, 11)
    dkeys = res['bond_distances']; acc = {k_: np.zeros(6) for k_ in dkeys}
    for c0 in range(0, n2a, CH):
        x, vx, xb, ib, vy, ibb, ix = chunk_data(idxA[c0:c0 + CH])
        la_x = lab.lP0[ix]; la_b = lab.lP0[ib]; la_bb = lab.lP0[ibb]
        s_x = lab.sP0[ix]; s_b = lab.sP0[ib]; s_bb = lab.sP0[ibb]
        I = la_bb + la_x[:, None, None] - la_b[:, :, None] - la_b[:, None, :]
        sig = s_bb * s_x[:, None, None] * s_b[:, :, None] * s_b[:, None, :]
        ok = vx[:, :, None] & vx[:, None, :] & (DBj[None] > 0)
        wgt = jnp.exp(jnp.clip(la_b[:, :, None] + la_b[:, None, :] - 2 * la_x[:, None, None], -60, 60))
        I = np.asarray(I); sig = np.asarray(sig); ok = np.asarray(ok); wgt = np.asarray(wgt)
        for k_ in dkeys:
            if k_ == 0: continue
            m = ok & (np.abs(DB - k_) < 1e-3)[None]
            if not m.any(): continue
            Ii = I[m]; si = sig[m]; wi = wgt[m]
            acc[k_] += np.array([m.sum(), np.sum(Ii * Ii), np.sum(np.abs(Ii)), np.sum(si < 0), np.sum(wi), np.sum(wi * Ii * Ii)])
    res['D2a'] = [dict(dist=k_, n=int(a[0]), rms_I=float(np.sqrt(a[1] / a[0])), mean_abs_I=float(a[2] / a[0]),
                       sign_defect_frac=float(a[3] / a[0]), wrms_I=float(np.sqrt(a[5] / a[4])))
                  for k_, a in acc.items() if a[0] > 0]
    for r_ in res['D2a']: log('D2a', r_)
    dump(F, res)

    # ---------------- D2b: light-cone E_L of G_3
    n2 = SPEC.get('n2', 2048)
    res['D2b'] = {}
    for dname, prob, seed in (('target', PT, 21), ('Q', Qfit, 22)):
        idxB = sample(prob, n2, seed)
        acc = {L: dict(EL=[], dd=[], flip=[], feat_err=[], feat_werr=[], nrho=[]) for L, _ in LEVELS}
        main_err = []; exact_chk = []
        for c0 in range(0, n2, CH):
            x, vx, xb, ib, vy, ibb, ix = chunk_data(idxB[c0:c0 + CH]); C = x.shape[0]
            la_x = lab.lP0[ix]; la_b = lab.lP0[ib]; la_bb = lab.lP0[ibb]
            s_x = lab.sP0[ix]; s_b = lab.sP0[ib]; s_bb = lab.sP0[ibb]
            la_add = la_b[:, :, None] + la_b[:, None, :] - la_x[:, None, None]
            s_add = s_b[:, :, None] * s_b[:, None, :] * s_x[:, None, None]
            ELx = EL3[ix]; G3x = G3[ix]
            hop = 0.5 * JB[None, :] * s3[ix][:, None] * s3[ib]          # (C, NB) sign/coupling of each hop of G_3
            D0x = lab.D0[ix]
            f_ex = feats_of(la_b, s_b, la_bb, s_bb, vy)                  # (C, NB, 4) exact features of x^b
            ew = jnp.where(vx, 0.5 * JB[None, :] * jnp.exp(jnp.clip(la_b - la_x[:, None], -60, 60)), 0.0)
            for L, rho in LEVELS:
                if L == '0th':
                    dG = la_b - la_x[:, None]                              # u(x^b) := u(x)
                    f_l = None; nr = jnp.zeros(())
                else:
                    if L == 'Fx':
                        f_l = jnp.broadcast_to(FEAT_BASE[ix][:, None, :], (C, NBj, 4)); nr = jnp.zeros(())
                    else:
                        mask = (DBj <= rho)[None] | (jnp.arange(NBj)[None, :, None] == jnp.arange(NBj)[None, None, :])
                        la_l = jnp.where(mask, la_bb, la_add); s_l = jnp.where(mask, s_bb, s_add)
                        f_l = feats_of(la_b, s_b, la_l, s_l, vy)
                        nr = jnp.sum(jnp.where(vx, jnp.sum(mask & vy & ~(jnp.eye(NBj, dtype=bool)[None]), -1), 0)) / jnp.sum(vx)
                    ff = f_l.reshape(-1, 4); Sy = xb.reshape(-1)
                    u = hnet(PH[0], Sy, ff) + hnet(PH[1], Sy, ff)
                    dG = la_b + u.reshape(C, NBj) - G3x[:, None]
                ELh = D0x + jnp.sum(jnp.where(vx, hop * jnp.exp(jnp.clip(dG, -60, 60)), 0.0), 1)
                acc[L]['EL'].append(np.asarray(ELh - ELx))
                acc[L]['dd'].append(np.asarray(dfun(ELh) - dfun(ELx)))
                acc[L]['flip'].append(np.asarray(((1 + alpha * (ELh - E3)) >= 0) != ((1 + alpha * (ELx - E3)) >= 0)))
                acc[L]['nrho'].append(float(nr))
                if f_l is not None and L != 'Fx':
                    fe = np.asarray(f_l - f_ex); m = np.asarray(vx)
                    acc[L]['feat_err'].append(np.sqrt(np.mean(fe[m] ** 2, 0)))
                    w_ = np.asarray(ew)[m]
                    acc[L]['feat_werr'].append(np.sqrt(np.sum(w_[:, None] * fe[m] ** 2, 0) / np.sum(w_)))
                if L == 'exact':
                    exact_chk.append(float(jnp.max(jnp.where(vx, jnp.abs(dG + G3x[:, None] - G3[ib]), 0.0))))
            main_err.append(np.asarray(h3[ix] - d3[ix]))
        me = np.concatenate(main_err); me = me - me.mean()
        out = dict(main_rebase_ptwise_rms=float(np.sqrt(np.mean(me ** 2))), exact_level_max_dG_err=float(max(exact_chk)),
                   levels={})
        for L, rho in LEVELS:
            e = np.concatenate(acc[L]['EL']); dd = np.concatenate(acc[L]['dd']); dd = dd - dd.mean()
            rec = dict(rho=rho, EL_rms=float(np.sqrt(np.mean(e ** 2))), d_rms=float(np.sqrt(np.mean(dd ** 2))),
                       flip_disagree=float(np.mean(np.concatenate(acc[L]['flip']))),
                       n_exact_per_bond=float(np.mean(acc[L]['nrho'])))
            if acc[L]['feat_err']:
                rec['feat_rms'] = np.mean(acc[L]['feat_err'], 0).tolist(); rec['feat_wrms'] = np.mean(acc[L]['feat_werr'], 0).tolist()
            out['levels'][L] = rec
            log(f'D2b[{dname}] {L}: EL rms {rec["EL_rms"]:.4e}  d rms {rec["d_rms"]:.4e}  flip {rec["flip_disagree"]:.2e}  '
                f'n_exact/bond {rec["n_exact_per_bond"]:.1f}')
        log(f'D2b[{dname}] main re-base ptwise rms {out["main_rebase_ptwise_rms"]:.4e}  exact-level check {out["exact_level_max_dG_err"]:.2e}')
        res['D2b'][dname] = out; dump(F, res)
    assert res['D2b']['target']['exact_level_max_dG_err'] < 1e-5, 'exact level does not reproduce G_3'

res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
