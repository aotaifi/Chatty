"""eps_m1.py: branch 2's information number (PLAN_pfaffian.md "Two branches for R1 after T-SIGN8").  CPU only.

eps^2 = |Delta*_M1 - Delta*_sG|^2 / |Delta*_sG|^2 in the Test W (ii) bond metric on the 6332 decades8 draws, Delta* = c f_J (c = 4, mode 'old', non-finite -> 0),
E' = -0.4987565 x 64, f_J[B; sign] with sign = sgn(M1) := sign Re[w psi_rb8_prod]  (w = M1 consts.json, w = -1) versus the stored direct s_G f_J.
B = psi_rb8_prod, so sgn(M1) at x, y, z comes from the lemma's own complex ratios: s(y) = sign Re[w e^{i Im lnB_x} r1], s(z) = sign Re[w e^{i Im lnB_x} r2].
Stored data: decades8 run8L + run8L2 (x, lnB, beta, dl, fx, fy at s_G-blocked larger-side bonds, sx, sy), analysed with the SAME estimator as decades8 final
(per-(beta, decade) self-normalised importance weights |B|^(2-2beta), MBAR decade mass, large-end convention, each pair in the decade of its smaller-|B| side y,
decade relative to the median 2 lnB of the beta = 1 draws; jackknife over process blocks = files).

sign providers of fj_core: 'm1' (above), 'sG' (direct, = decades8's route; the code control), 'marshall' (broken version), 'direct_m1' (sign from PL.logB, check).
CLI
  eps_m1.py task OUTDIR K [--kind m1]     one file (process block K of the 76) -> OUTDIR/eps_K.npz
  eps_m1.py analyze OUTDIR OUTJSON [--kind m1] [--kfiles ...]
  eps_m1.py control_sG OUTJSON [--nd 3] [--files 0,40]        sign = s_G through fj_core must reproduce the stored fx, fy, sx, sy; then eps^2 = 0
  eps_m1.py control_ratio OUTJSON                             ratio-derived sgn(M1) vs direct sign from PL.logB on sampled endpoints
"""
import os, sys, json, time, math, glob, hashlib, argparse
import numpy as np

CODE = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code'
FROZEN = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/decades8/final/code'      # decades8.py sha c46e8156 (the code that made the stored draws' analysis)
D8DIR = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/decades8'
sys.path.insert(0, FROZEN); sys.path.insert(0, CODE); sys.dont_write_bytecode = True
LN10 = math.log(10.0); C = 4.0
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def files_list():
    """the same list and sort as the decades8 final analysis (relative patterns, sorted as strings)."""
    os.chdir(D8DIR)
    return sorted(sum([glob.glob(f) for f in ('run8L/out_*.npz', 'run8L2/out_*.npz')], []))


def sign_of(v): return np.where(v >= 0, 1, -1).astype(np.int8)


# ============================================================================================ f_J with pluggable sign
def fj_core(p, x, kind, w=None, EP=None, sign_fn=None, eval_path='dedup', side='large', want=None):
    """f_J at x and at every one-swap y (m1 / marshall / direct_m1: all valid y; sG: s_G-blocked larger-side y only, as decades8.fj_raw) for the signs of `kind`.
    returns per-bond layout: lnB, fx, sx, fy (M, NB), dl (ln|B_y/B_x|), sy (M, NB), margin stats."""
    import jax.numpy as jnp
    import pp2_lemma as PL
    import decades8 as D8
    x = np.asarray(x); M, N = x.shape; lat = p.lat; NB = lat.NB
    d = PL.two_swap_ratios(p, x, eval_path=eval_path)
    am = np.asarray(d['am']); a = np.asarray(d['a']); dA = am.shape[1]; bm = np.asarray(d['bm']); dB = bm.shape[2]
    r1 = np.asarray(d['r1']); r2 = np.asarray(d['r2']); lB = np.asarray(d['logB'])
    info = {}
    if kind == 'm1':
        ph = w * np.exp(1j * lB.imag)                                             # (M,)  phase of w psi(x); |psi| > 0 does not change a sign
        vx = ph.real; vy = (ph[:, None] * r1).real; vz = (ph[:, None, None] * r2).real
        sx = sign_of(vx).astype(np.float64); sy = sign_of(vy).astype(np.float64); sz = sign_of(vz).astype(np.float64)
        sc = np.abs(ph)[:, None]
        info['min_rel_margin_y'] = float(np.min(np.where(am, np.abs(vy) / np.maximum(np.abs(ph[:, None] * r1), 1e-300), np.inf)))
        info['min_rel_margin_z'] = float(np.min(np.where(bm, np.abs(vz) / np.maximum(np.abs(ph[:, None, None] * r2), 1e-300), np.inf)))
        info['min_rel_margin_x'] = float(np.min(np.abs(vx) / np.abs(ph)))
        blk = None
    elif kind in ('marshall', 'direct_m1'):
        Y, Z = PL._configs(jnp.asarray(x, jnp.float64), {k: d[k] for k in ('upa', 'dna', 'am', 'ru', 'rd', 'k')}, with_z=True)
        Y = np.asarray(Y).reshape(M, dA, N); Z = np.asarray(Z).reshape(M, dA, dB, N)
        if kind == 'marshall':
            Ai = np.array([(s % lat.L + s // lat.L) % 2 == 0 for s in range(N)])    # sublattice A: (x + y) even, site = x + L y
            mar = lambda cfg: np.where((((cfg[..., Ai] > 0).sum(-1)) % 2) == 0, 1, -1).astype(np.float64)
            sx, sy, sz = mar(x), mar(Y), mar(Z)
        else:                                                                       # direct: sign Re[w psi] from PL.logB on every endpoint (check only, expensive)
            dirs = lambda cfg: sign_of((w * np.exp(1j * np.asarray(PL.logB(p, cfg.reshape(-1, N).astype(np.float64))).imag)).real).astype(np.float64).reshape(cfg.shape[:-1])
            sx = dirs(x); sy = dirs(Y); sz = np.zeros(Z.shape[:-1]); nz = bm
            sz[nz] = dirs(Z[nz])
        blk = None
    elif kind == 'sG':
        mask = np.asarray(PL._flip_masks(d['upa'], d['dna'], d['am'], d['ru'], d['rd'], d['k'], d['bm']))
        xkey = PL._cfg_keys(x)
        sxy, _ = D8._signs_keys(sign_fn, np.concatenate([xkey[:, None], xkey[:, None] ^ mask[:, :dA]], 1), N)
        sx = sxy[:, 0].astype(np.float64); sy = sxy[:, 1:].astype(np.float64)
        blk = am & (sx[:, None] * sy > 0) & (np.abs(r1) <= 1.0)
        if want is not None: blk = am & np.take_along_axis(np.asarray(want, bool), np.maximum(a, 0), 1)     # 'want' (M, NB): compute f_J(y) under s_G exactly on these bonds
        need = blk[:, :, None] & bm & am[:, :, None]
        zk = (xkey[:, None] ^ mask[:, dA:]).reshape(M, dA, dB)
        sz = np.repeat(sy[:, :, None], dB, 2).copy()
        if need.any():
            v, _ = D8._signs_keys(sign_fn, zk[need], N); sz[need] = v
    else: raise ValueError(kind)
    fx, py = PL._fj_parts(lat, jnp.asarray(x, jnp.float64), d, jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(sz), EP, True)
    SKy, VAy = py
    Y = np.asarray(PL._configs(jnp.asarray(x, jnp.float64), {k: d[k] for k in ('upa', 'dna', 'am', 'ru', 'rd', 'k')}, with_z=False)[0])
    Hy = 0.25 * (np.asarray(lat.JB)[None, None] * Y[..., lat.BI] * Y[..., lat.BJ]).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        fy = np.log(np.asarray(SKy)) - np.log(Hy + np.asarray(VAy) - EP)
    fy = np.where(am if blk is None else blk, fy, np.nan)
    FY = np.full((M, NB), np.nan); DL = np.full((M, NB), np.nan); SY = np.zeros((M, NB), np.int8)
    rr, kk = np.nonzero(am); b = a[rr, kk]
    FY[rr, b] = fy[rr, kk]; DL[rr, b] = np.log(np.abs(r1[rr, kk])); SY[rr, b] = sy[rr, kk].astype(np.int8)
    return dict(lnB=lB.real, fx=np.asarray(fx), sx=np.asarray(sx).astype(np.int8), fy=FY, dl=DL, sy=SY, info=info)


# ============================================================================================ task
def load_stored(fn):
    z = np.load(os.path.join(D8DIR, fn), allow_pickle=True)
    return {k: z[k] for k in ('beta', 'lnB', 'fx', 'sx', 'fy', 'dl', 'sy', 'x')}


def task(a):
    import pp2_lemma as PL
    import decades8 as D8
    fl = files_list(); fn = fl[a.k]; S = load_stored(fn); n = len(S['lnB'])
    assert D8.sha(D8.PROD8) == D8.PROD8_SHA
    p = PL.PP2.from_rbmpp_npz(D8.PROD8); al, E, w = D8.m1_consts(); EP = D8.EP8
    out = {k: [] for k in ('lnB', 'fx', 'sx', 'fy', 'dl', 'sy')}; t0 = time.time(); mins = []
    nmax = min(n, a.nmax) if a.nmax else n
    for i0 in range(0, nmax, a.chunk):
        xx = S['x'][i0:i0 + a.chunk].astype(np.float64)
        r = fj_core(p, xx, a.kind, w=w, EP=EP)
        for k in out: out[k].append(r[k])
        mins.append(r['info'])
        print(f'{time.strftime("%H:%M:%S")} [{time.time() - t0:7.0f}s] PROGRESS {fn} {i0 + len(xx)}/{nmax} {mins[-1]}', flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    # checks against the stored draws: same configs -> same lnB and ln|B_y/B_x| on the valid bonds
    m = nmax
    chk = dict(file=fn, n=int(m), lnB_maxdev=float(np.max(np.abs(out['lnB'] - S['lnB'][:m]))),
               dl_maxdev=float(np.nanmax(np.abs(out['dl'] - S['dl'][:m]))), dl_nan_agree=bool(np.array_equal(np.isfinite(out['dl']), np.isfinite(S['dl'][:m]))),
               min_rel_margin=min([min(d.get('min_rel_margin_x', 1), d.get('min_rel_margin_y', 1), d.get('min_rel_margin_z', 1)) for d in mins]), sec=time.time() - t0, kind=a.kind)
    os.makedirs(a.outdir, exist_ok=True)
    np.savez(f'{a.outdir}/eps_{a.k}.npz', **out, chk=json.dumps(chk), kind=a.kind)
    print('DONE', json.dumps(chk), flush=True)


def task_extra(a):
    """exact s_G f_J(y) on the bonds blocked under sgn(M1) but not under s_G whose smaller side y lies in relative decades -9..-2 (needed for eps^2 on sgn(M1)'s bonds)."""
    import pp2_lemma as PL
    import decades8 as D8
    fl = files_list(); fn = fl[a.k]; S = load_stored(fn); z = np.load(f'{a.outdir_m1}/eps_{a.k}.npz', allow_pickle=True)
    p = PL.PP2.from_rbmpp_npz(D8.PROD8); al, E, w = D8.m1_consts(); EP = D8.EP8; sfn = D8.make_sG8_np(al, E, w)
    valid = np.isfinite(S['dl']); dl = np.where(valid, S['dl'], 0.0); tie = valid & (S['dl'] <= 0)
    blkG = tie & (S['sx'][:, None].astype(float) * S['sy'] > 0); blkM = tie & (z['sx'][:, None].astype(float) * z['sy'] > 0)
    yd = np.floor((2.0 * (S['lnB'][:, None] + dl) - a.med) / LN10).astype(np.int64)
    want = blkM & ~blkG & (yd >= -9) & (yd <= -2)
    n = len(S['lnB']); FYX = np.full((n, want.shape[1]), np.nan); t0 = time.time(); idx = np.nonzero(want.any(1))[0]; nb = 0
    for i0 in range(0, len(idx), a.chunk):
        ii = idx[i0:i0 + a.chunk]
        r = fj_core(p, S['x'][ii].astype(np.float64), 'sG', EP=EP, sign_fn=sfn, want=want[ii])
        sel = want[ii]; FYX[ii] = np.where(sel, r['fy'], np.nan); nb += int(sel.sum())
        # consistency: sx under s_G is the stored one
        assert np.array_equal(r['sx'], S['sx'][ii])
        print(f'{time.strftime("%H:%M:%S")} [{time.time() - t0:7.0f}s] PROGRESS {fn} draws {i0 + len(ii)}/{len(idx)} bonds {nb}', flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    np.savez(f'{a.outdir}/extra_{a.k}.npz', fyG_extra=FYX, want=want, med=a.med)
    print('DONE', json.dumps(dict(file=fn, n_draws_with_want=int(len(idx)), n_bonds=int(want.sum()), sec=time.time() - t0)), flush=True)


# ============================================================================================ analysis
def mbar_logZ(lnb_list, betas, iters=5000, tol=1e-12):
    lnb = np.concatenate(lnb_list); Nk = np.array([len(l) for l in lnb_list], float); bk = np.asarray(betas)
    u = 2.0 * bk[:, None] * lnb[None, :]; f = np.zeros(len(bk))
    for it in range(iters):
        den = np.logaddexp.reduce(np.log(Nk)[:, None] + u - f[:, None], axis=0)
        fn = np.logaddexp.reduce(u - den[None, :], axis=1); fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol: f = fn; break
        f = fn
    den = np.logaddexp.reduce(np.log(Nk)[:, None] + u - f[:, None], axis=0)
    return f, 2.0 * lnb - den


DLO, DHI = -30, 6
QN = ['den_G', 'num_G', 'den_I', 'num_I', 'w_G', 'w_M', 'w_Gonly', 'w_Monly', 'dM2_Monly', 'dG2_Gonly', 'num_Gonly', 'nb_G', 'nb_flipG', 'nb_valid', 'nb_sy_ne']


def build(S, E, kind_label=''):
    """per-draw per-bond arrays and the per-draw decade vectors H[q] (n, NDEC).  S stored (s_G), E recomputed (sign kind)."""
    HW = np.concatenate([np.full(128, 0.5), np.full(128, 0.25)])
    dl0 = S['dl']; valid = np.isfinite(dl0); dl = np.where(valid, dl0, 0.0)
    tie = np.where(dl < 0, 1.0, np.where(dl == 0, 0.5, 0.0)) * valid
    wgt = HW[None, :] * np.exp(dl) * tie
    sxG = S['sx'].astype(float); syG = S['sy'].astype(float); sxM = E['sx'].astype(float); syM = E['sy'].astype(float)
    blkG = valid & (tie > 0) & (sxG[:, None] * syG > 0); blkM = valid & (tie > 0) & (sxM[:, None] * syM > 0)
    fin = lambda f: np.where(np.isfinite(f), f, 0.0)
    xG = C * fin(S['fx']); xM = C * fin(E['fx'])
    assert not np.isnan(S['fy'][blkG]).any()
    yG = C * fin(np.where(valid, S['fy'], 0.0)); yM = C * fin(np.where(valid, E['fy'], 0.0))
    dG = xG[:, None] - yG; dM = xM[:, None] - yM
    I = blkG & blkM; Go = blkG & ~blkM; Mo = blkM & ~blkG
    extra = {}
    if 'fyG_extra' in E:                                  # exact s_G f_J(y) on the M-only bonds of the fitted decades (task_extra); elsewhere nan
        ex = E['fyG_extra']; have = Mo & ~np.isnan(ex); dGx = np.where(have, xG[:, None] - C * fin(np.where(have, ex, 0.0)), 0.0)
        dGM = np.where(blkG, dG, dGx)
        extra = dict(den_M=wgt * (I | have) * dGM ** 2, num_M=wgt * (I | have) * (dM - dGM) ** 2, w_Mhave=wgt * have, w_Mmiss=wgt * (Mo & ~have))
    # tie-weighted counts for the fraction diagnostics (raw counts, valid bonds seen from the larger end)
    q = dict(den_G=wgt * blkG * dG ** 2, num_G=wgt * blkG * (dM - dG) ** 2, den_I=wgt * I * dG ** 2, num_I=wgt * I * (dM - dG) ** 2,
             w_G=wgt * blkG, w_M=wgt * blkM, w_Gonly=wgt * Go, w_Monly=wgt * Mo, dM2_Monly=wgt * Mo * dM ** 2, dG2_Gonly=wgt * Go * dG ** 2,
             num_Gonly=wgt * Go * (dM - dG) ** 2, nb_G=blkG.astype(float), nb_flipG=Go.astype(float), nb_valid=(valid & (tie > 0)).astype(float),
             nb_sy_ne=((valid & (tie > 0)) & (syG != syM)).astype(float)); q.update(extra)
    ydec_raw = None
    return q, dl, valid, tie


def decade_vectors(q, lnB, dl, med):
    n, NB = dl.shape; ref = med
    tdec = np.floor((2.0 * (lnB[:, None] + dl) - ref) / LN10).astype(np.int64)         # decade of y (smaller side)
    tdec = np.clip(tdec, DLO, DHI) - DLO
    ND = DHI - DLO + 1; rows = np.broadcast_to(np.arange(n)[:, None], tdec.shape)
    H = {}
    for k, v in q.items():
        h = np.zeros((n, ND)); np.add.at(h, (rows, tdec), v); H[k] = h
    return H


def estimator(H, lnB, beta, betas, med, idx):
    """per quantity: sum over target decades of the decades8 'cell' estimator, on the draw subset idx.  returns tot[q] (ND,), mass (ND,)."""
    lst = [lnB[idx][beta[idx] == b] for b in betas]
    f, lw_k = mbar_logZ(lst, betas)
    lw = np.empty(len(idx)); order = np.concatenate([np.nonzero(beta[idx] == b)[0] for b in betas]); lw[order] = lw_k
    decx = np.floor((2.0 * lnB[idx] - med) / LN10).astype(np.int64)
    ND = DHI - DLO + 1; D = np.arange(DLO, DHI + 1); wt = np.exp(lw - lw.max())
    mass = np.array([wt[decx == d].sum() for d in D]); mass /= mass.sum()
    b_ = beta[idx]; lnb = lnB[idx]
    tot = {k: np.zeros(ND) for k in H}
    for j, d in enumerate(D):
        md = decx == d; parts = {k: [] for k in H}; Eess = []
        for b in betas:
            m = md & (b_ == b)
            if not m.any(): continue
            l = (2.0 - 2.0 * b) * lnb[m]; ww = np.exp(l - l.max()); ww /= ww.sum(); Eess.append(1.0 / np.sum(ww * ww))
            for k in H: parts[k].append(ww @ H[k][idx][m])
        if Eess:
            Ee = np.array(Eess)
            for k in H: tot[k] += mass[j] * (Ee @ np.array(parts[k])) / Ee.sum()
    return tot, mass


def ratio_table(tot, dsel=range(-9, -1)):
    """eps^2 per decade and totals from tot[q]."""
    D = np.arange(DLO, DHI + 1); out = {}
    def r(nu, de, sel):
        s = np.array([(lo in sel) for lo in D]); n_ = nu[s].sum(); d_ = de[s].sum(); return n_ / d_ if d_ > 0 else np.nan
    allsel = set(D.tolist()); fit = set(dsel)
    res = {}
    for nm, nu, de in (('G', 'num_G', 'den_G'), ('I', 'num_I', 'den_I')) + ((('M', 'num_M', 'den_M'),) if 'num_M' in tot else ()):
        res[f'eps2_{nm}_dec'] = [(tot[nu][j] / tot[de][j] if tot[de][j] > 0 else np.nan) for j in range(len(D))]
        res[f'eps2_{nm}_fit'] = r(tot[nu], tot[de], fit); res[f'eps2_{nm}_all'] = r(tot[nu], tot[de], allsel)
    res['share_den_G'] = (tot['den_G'] / tot['den_G'].sum()).tolist()
    if 'num_M' in tot:
        fi = [j for j in range(len(D)) if D[j] in fit]; res['wfrac_fit_Mmissing'] = float(sum(tot['w_Mmiss'][j] for j in fi) / sum(tot['w_M'][j] for j in fi))
        res['wfrac_fit_Mhave_over_Monly'] = float(sum(tot['w_Mhave'][j] for j in fi) / sum(tot['w_Monly'][j] for j in fi))
    fl = lambda a_, b_: [(tot[a_][j] / tot[b_][j] if tot[b_][j] > 0 else np.nan) for j in range(len(D))]
    res['wfrac_Gbond_unblocked_by_M1'] = fl('w_Gonly', 'w_G'); res['wfrac_Mbond_not_G'] = fl('w_Monly', 'w_M')
    res['wfrac_fit_Gonly_over_G'] = float(sum(tot['w_Gonly'][j] for j in range(len(D)) if D[j] in fit) / sum(tot['w_G'][j] for j in range(len(D)) if D[j] in fit))
    res['wfrac_fit_Monly_over_M'] = float(sum(tot['w_Monly'][j] for j in range(len(D)) if D[j] in fit) / sum(tot['w_M'][j] for j in range(len(D)) if D[j] in fit))
    # share of eps^2 numerator from G-bonds that M1 unblocks
    res['num_G_frac_from_Gonly_fit'] = float(sum(tot['num_Gonly'][j] for j in range(len(D)) if D[j] in fit) / sum(tot['num_G'][j] for j in range(len(D)) if D[j] in fit))
    res['num_G_frac_from_Gonly_all'] = float(tot['num_Gonly'].sum() / tot['num_G'].sum())
    res['dM2_Monly_over_den_G_fit'] = float(sum(tot['dM2_Monly'][j] for j in range(len(D)) if D[j] in fit) / sum(tot['den_G'][j] for j in range(len(D)) if D[j] in fit))
    return res


def analyze(a):
    fl = files_list(); ks = [int(t) for t in a.kfiles.split(',')] if a.kfiles else list(range(len(fl))); fl = [fl[k] for k in ks]; nf = len(fl)
    Ss = [load_stored(f) for f in fl]
    Es = []; chks = []
    for k in ks:
        z = np.load(f'{a.outdir}/eps_{k}.npz', allow_pickle=True); Es.append({kk: z[kk] for kk in ('lnB', 'fx', 'sx', 'fy', 'dl', 'sy')}); chks.append(json.loads(str(z['chk'])))
        assert str(z['kind']) == a.kind
    cat = lambda L, ks: np.concatenate([L_[ks] for L_ in L]) if True else None
    def catd(L): return {k: np.concatenate([x[k] for x in L]) for k in L[0]}
    if a.extra:
        for k_, e_ in zip(ks, Es): e_['fyG_extra'] = np.load(f'{a.extra}/extra_{k_}.npz')['fyG_extra']
    S = catd(Ss); E = catd(Es); fid = np.concatenate([np.full(len(s['lnB']), k) for k, s in enumerate(Ss)])
    assert np.array_equal(np.isfinite(S['dl']), np.isfinite(E['dl'])) and np.nanmax(np.abs(S['dl'] - E['dl'])) < 1e-6 and np.abs(S['lnB'] - E['lnB']).max() < 1e-8
    betas = sorted({float(b) for b in np.unique(S['beta'])}, reverse=True); beta = S['beta']; lnB = S['lnB']
    med = float(np.median(2.0 * lnB[beta == 1.0]))
    q, dl, valid, tie = build(S, E); H = decade_vectors(q, lnB, np.where(valid, S['dl'], 0.0), med)
    D = np.arange(DLO, DHI + 1); decx = np.floor((2.0 * lnB - med) / LN10).astype(np.int64)
    res = dict(kind=a.kind, n_draws=int(len(lnB)), n_files=nf, betas=betas, med2lnB_beta1=med, c=C, mode='old', EP=-0.4987565 * 64, fit_decades=[-9, -2], files=fl,
               per_file_checks=dict(max_lnB_dev=max(c['lnB_maxdev'] for c in chks), max_dl_dev=max(c['dl_maxdev'] for c in chks), all_nan_agree=all(c['dl_nan_agree'] for c in chks),
                                    min_rel_margin=min(c['min_rel_margin'] for c in chks), sec_total=sum(c['sec'] for c in chks)), code_sha=dict(eps_m1=sha(os.path.abspath(__file__)),
                                    decades8_frozen=sha(f'{FROZEN}/decades8.py'), pp2_lemma=sha(f'{CODE}/pp2_lemma.py')))
    def block(sel_files, label):
        idx = np.nonzero(np.isin(fid, sel_files))[0]
        tot, mass = estimator(H, lnB, beta, betas, med, idx); base = ratio_table(tot)
        J = []
        for kf in sel_files:
            idj = np.nonzero(np.isin(fid, [f for f in sel_files if f != kf]))[0]
            tj, _ = estimator(H, lnB, beta, betas, med, idj); J.append(ratio_table(tj))
        n = len(sel_files); err = {}
        for key in (('eps2_M_fit', 'wfrac_fit_Mmissing') if 'num_M' in tot else ()) + ('eps2_G_fit', 'eps2_G_all', 'eps2_I_fit', 'eps2_I_all', 'wfrac_fit_Gonly_over_G', 'wfrac_fit_Monly_over_M', 'num_G_frac_from_Gonly_fit', 'num_G_frac_from_Gonly_all'):
            v = np.array([j[key] for j in J]); err[key] = float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))
        for key in (('eps2_M_dec',) if 'num_M' in tot else ()) + ('eps2_G_dec', 'eps2_I_dec', 'share_den_G', 'wfrac_Gbond_unblocked_by_M1', 'wfrac_Mbond_not_G'):
            v = np.array([j[key] for j in J], float); err[key] = np.sqrt((n - 1) / n * np.nansum((v - np.nanmean(v, 0)) ** 2, 0)).tolist()
        return dict(label=label, n_files=n, n_draws=int(len(idx)), base=base, err=err, mass=mass.tolist())
    res['full'] = block(list(range(nf)), 'full')
    res['half_A'] = block(list(range(0, nf, 2)), 'A (files 0,2,4,.. in the sorted list)'); res['half_B'] = block(list(range(1, nf, 2)), 'B (files 1,3,5,..)')
    # raw fractions per x-decade (draws) and per y-decade (bonds), unweighted counts
    ND = len(D); rawd = []
    ydec = np.floor((2.0 * (lnB[:, None] + np.where(valid, S['dl'], 0.0)) - med) / LN10).astype(np.int64)
    for j, d in enumerate(D):
        m = decx == d
        if m.sum() == 0 and not ((ydec == d) & valid & (tie > 0)).any(): continue
        mb = (ydec == d) & valid & (tie > 0)
        syne = (S['sy'] != E['sy'])
        rawd.append(dict(decade=int(d), n_draws=int(m.sum()), frac_draws_sx_ne=float((S['sx'][m] != E['sx'][m]).mean()) if m.any() else None,
                         n_bonds=int(mb.sum()), frac_bonds_sy_ne=float(syne[mb].mean()) if mb.any() else None,
                         n_Gblocked=int((mb & (S['sx'][:, None].astype(float) * S['sy'] > 0)).sum()),
                         frac_Gblocked_unblocked=float((mb & (S['sx'][:, None].astype(float) * S['sy'] > 0) & ~(E['sx'][:, None].astype(float) * E['sy'] > 0)).sum() / max(1, (mb & (S['sx'][:, None].astype(float) * S['sy'] > 0)).sum()))))
    res['raw_fractions_per_decade'] = rawd
    res['raw_frac_draws_sx_ne_total'] = float((S['sx'] != E['sx']).mean())
    # reproduction of decades8's own shares (control that the data + estimator are decades8's)
    ref = json.load(open(f'{D8DIR}/final/final_old.json'))['est']['cell_large']; Dr = np.array(ref['dec'])
    mine = np.array(res['full']['base']['share_den_G']); sr = np.array(ref['share'])
    sel = (Dr >= -12) & (Dr <= 3)
    res['share_reproduction_vs_final_old'] = dict(max_abs_diff=float(np.max(np.abs(mine[Dr - DLO][sel] - sr[sel]))), note='full-data cell_large shares of Delta*_sG on the stored draws vs decades8 final_old.json')
    json.dump(res, open(a.outjson, 'w'), indent=1, default=float)
    f = res['full']; print(f'kind {a.kind}  eps2 (s_G bonds)  fit -2..-9 {f["base"]["eps2_G_fit"]:.4f} +- {f["err"]["eps2_G_fit"]:.4f}  all {f["base"]["eps2_G_all"]:.4f}   (bonds in both sets: {f["base"]["eps2_I_fit"]:.4f} +- {f["err"]["eps2_I_fit"]:.4f})')
    for h in ('half_A', 'half_B'):
        g = res[h]; print(f'  {h}: {g["base"]["eps2_G_fit"]:.4f} +- {g["err"]["eps2_G_fit"]:.4f}')
    for j, d in enumerate(D):
        if -12 <= d <= 0:
            print(f'{d:4d} share {f["base"]["share_den_G"][j]:.4f}  eps2_G {f["base"]["eps2_G_dec"][j]:.4f} +- {f["err"]["eps2_G_dec"][j]:.4f}  eps2_I {f["base"]["eps2_I_dec"][j]:.4f}  Gbond unblocked wfrac {f["base"]["wfrac_Gbond_unblocked_by_M1"][j]:.4f}')
    print('share repro max abs diff', res['share_reproduction_vs_final_old']['max_abs_diff'])


# ============================================================================================ controls
def control_sG(a):
    """sign provider 's_G' (direct sign function, decades8's route) through fj_core on a few draws of a few files: stored fx, fy (blocked), sx, sy must be reproduced;
    then the eps^2 arithmetic with E = this recomputation must give 0.  Broken versions: s_G with alpha = 0; Marshall."""
    import pp2_lemma as PL
    import decades8 as D8
    p = PL.PP2.from_rbmpp_npz(D8.PROD8); al, E, w = D8.m1_consts(); EP = D8.EP8
    fl = files_list(); res = dict(rows=[], sec=0.0)
    sfn = D8.make_sG8_np(al, E, w); sfn_b = D8.make_sG8_np(0.0, E, w)
    t0 = time.time()
    for k in [int(t) for t in a.files.split(',')]:
        S = load_stored(fl[k]); sl = slice(0, a.nd); xx = S['x'][sl].astype(np.float64)
        r = fj_core(p, xx, 'sG', EP=EP, sign_fn=sfn)
        fyS = S['fy'][sl]; fin = np.isfinite(fyS) & np.isfinite(r['fy'])
        row = dict(file=fl[k], n=int(len(xx)), fx_maxdev=float(np.max(np.abs(r['fx'] - S['fx'][sl]))), sx_ne=int((r['sx'] != S['sx'][sl]).sum()),
                   fy_n_stored=int(np.isfinite(fyS).sum()), fy_n_mine=int(np.isfinite(r['fy']).sum()), fy_maxdev=float(np.max(np.abs(r['fy'][fin] - fyS[fin]))),
                   fy_support_equal=bool(np.array_equal(np.isfinite(fyS), np.isfinite(r['fy']))), sy_ne=int((r['sy'] != S['sy'][sl]).sum()))
        # eps^2 arithmetic with E = recomputation (must be exactly 0) on this subset; blocked bonds only
        Sx = {kk: v[sl] for kk, v in S.items()}
        q, dl, valid, tie = build(Sx, dict(sx=r['sx'], sy=r['sy'], fx=r['fx'], fy=np.where(np.isfinite(r['fy']), r['fy'], np.nan)))
        row['num_G_sum'] = float(q['num_G'].sum()); row['den_G_sum'] = float(q['den_G'].sum())
        # BROKEN 1: s_G with alpha = 0 (stored sign function replaced by the bare phase sign) -> differs
        rb = fj_core(p, xx, 'm1', w=w, EP=EP)
        q2, *_ = build(Sx, rb); row['eps2_numden_with_M1_this_subset'] = float(q2['num_G'].sum() / q2['den_G'].sum())
        row['M1_sx_ne_sG'] = int((rb['sx'] != S['sx'][sl]).sum())
        res['rows'].append(row); print(json.dumps(row), flush=True)
    res['sec'] = time.time() - t0
    res['PASS'] = bool(all(r_['fx_maxdev'] < 1e-9 and r_['sx_ne'] == 0 and r_['fy_support_equal'] and r_['fy_maxdev'] < 1e-9 and r_['sy_ne'] == 0 and r_['num_G_sum'] < 1e-20 * r_['den_G_sum'] for r_ in res['rows']))
    json.dump(res, open(a.outjson, 'w'), indent=1); print('PASS', res['PASS'])


def control_ratio(a):
    """ratio-derived sgn(M1) at x, y, z against the direct sign (PL.logB phase) on a few configs; and the Marshall sign must disagree on ~half."""
    import pp2_lemma as PL
    import decades8 as D8
    p = PL.PP2.from_rbmpp_npz(D8.PROD8); al, E, w = D8.m1_consts(); EP = D8.EP8
    fl = files_list(); S = load_stored(fl[0]); xx = S['x'][:a.nd].astype(np.float64)
    r1_ = fj_core(p, xx, 'm1', w=w, EP=EP); r2_ = fj_core(p, xx, 'direct_m1', w=w, EP=EP); r3_ = fj_core(p, xx, 'marshall', w=w, EP=EP)
    fin = lambda f: np.isfinite(f)
    res = dict(n=int(len(xx)), sx_ne_direct=int((r1_['sx'] != r2_['sx']).sum()), sy_ne_direct=int((r1_['sy'] != r2_['sy']).sum()), n_y=int((r1_['dl'] == r1_['dl']).sum()),
               fx_maxdev=float(np.max(np.abs(r1_['fx'] - r2_['fx']))), fy_finite_equal=bool(np.array_equal(fin(r1_['fy']), fin(r2_['fy']))),
               fy_maxdev=float(np.nanmax(np.abs(np.where(fin(r1_['fy']) & fin(r2_['fy']), r1_['fy'] - r2_['fy'], np.nan)))),
               marshall_sx_ne_M1=int((r3_['sx'] != r1_['sx']).sum()), marshall_sy_ne_M1_frac=float((r3_['sy'] != r1_['sy'])[np.isfinite(r1_['dl'])].mean()),
               marshall_fx_maxdev_vs_M1=float(np.max(np.abs(r3_['fx'] - r1_['fx']))))
    res['PASS'] = bool(res['sx_ne_direct'] == 0 and res['sy_ne_direct'] == 0 and res['fx_maxdev'] < 1e-9 and res['fy_finite_equal'] and res['fy_maxdev'] < 1e-9 and res['marshall_fx_maxdev_vs_M1'] > 1e-3)
    json.dump(res, open(a.outjson, 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd')
    t = sp.add_parser('task'); t.add_argument('outdir'); t.add_argument('k', type=int); t.add_argument('--kind', default='m1'); t.add_argument('--chunk', type=int, default=10); t.add_argument('--nmax', type=int, default=0)
    z = sp.add_parser('analyze'); z.add_argument('outdir'); z.add_argument('outjson'); z.add_argument('--kind', default='m1'); z.add_argument('--kfiles', default=''); z.add_argument('--extra', default='')
    c = sp.add_parser('control_sG'); c.add_argument('outjson'); c.add_argument('--nd', type=int, default=3); c.add_argument('--files', default='0,40')
    x = sp.add_parser('task_extra'); x.add_argument('outdir'); x.add_argument('k', type=int); x.add_argument('--outdir_m1', required=True); x.add_argument('--med', type=float, required=True); x.add_argument('--chunk', type=int, default=4)
    r = sp.add_parser('control_ratio'); r.add_argument('outjson'); r.add_argument('--nd', type=int, default=3)
    a = ap.parse_args()
    {'task_extra': task_extra, 'task': task, 'analyze': analyze, 'control_sG': control_sG, 'control_ratio': control_ratio}[a.cmd](a)
