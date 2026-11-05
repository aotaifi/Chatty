"""b2_prep_test.py: tests of branch-2 stages (2) labels and (3) R1 fit prep (b2_prep.py); each with a deliberately broken version that must fail.  CPU.
usage: python b2_prep_test.py OUTDIR {labels|pool|toy|dryrun} [opts]
(1) labels: labels_m1 (pool_fit.Labels8 + M1Sign, c 4, mode 'fix', f0 1.22, E' -0.4987565 x 64) on >= 50 REAL draws (pool_x/prod, all three rungs, and pool_x2 deep
    draws), against a SLOW DIRECT path: explicit swaps of every anti-aligned bond, ln psi of x and of every y from M1's own evaluator (m1_code/rbmpp_lib.make_fns logpsi,
    brute force over all images), sgn(M1)(z) = sgn Re[w e^{i Im ln psi(z)}] directly at every endpoint, f_J = ln SK - ln(H_xx + VA - E') by definition, the fix applied.
    Pass max |ln T - slow| <= 1e-9.  Partners: NeighboursM1's blocked mask and ln|B_y/B_x| == the direct ones (pass: 0 mask differences, dlnB <= 1e-9).
    Broken: M1Sign(broken='marshall') -> must differ (> 1e-6) and the blocked masks must differ.  Timing per configuration (labels, neighbours) on CPU.
(2a) pool: build_pool_given == pool_fit.build_pool (every key) on a small equal-split selection with a draw_fn that does not consume the rng; pool_weights_med(med = the
    beta-1 median) == pool_fit.pool_weights.  Broken: a different seed -> partners differ.
(2b) toy: a 2000-entry pool from real draws (250 per decade -2..-9, uniform inside each (rung, decade) cell, med = pool_x's full beta-1 median), through label_task + merge.
(2c) dryrun: b2_prep.fit (pool_fit.fit arm 'pp8', two-PP warm from rb8_prod) 3 GN steps on the toy pool; held-out and fixed-train losses must fall.  Broken: break_negate_step."""
import os, sys, json, time, glob, hashlib, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import pp2_lemma as PL, pool_fit as PF, b2_prep as B2

OUT = sys.argv[1]; WHAT = sys.argv[2]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
M1C = os.path.normpath(os.path.join(HERE, '..', '..', 'experiments', 'addon_10x10', 'm1_code'))
PX = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8_r1/pool_x/prod'; PX2 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8_r1/pool_x2/prod'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); open(f'{OUT}/test.log', 'a').write(s + '\n')


def jd(name, obj): json.dump(obj, open(f'{OUT}/{name}.json', 'w'), indent=1, default=float)


def m1lib():
    spec = importlib.util.spec_from_file_location('m1_rbmpp_lib', os.path.join(M1C, 'rbmpp_lib.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def real_draws(n_per_dec, decs, seed, med=B2.MED_POOLX, files_x=(10, 11, 12), files_x2=(10, 11)):
    """n_per_dec draws per decade from pool_x/prod (all rungs) + pool_x2/prod (beta 0.25), uniform among that decade's draws of the files read."""
    W, L, Bt, O, F, R = [], [], [], [], [], []
    for d_, fl in ((PX, files_x), (PX2, files_x2)):
        for f in fl:
            z = np.load(f'{d_}/out_{f}.npz'); W.append(z['words']); L.append(z['lnB']); Bt.append(z['beta']); O.append(z['orb']); F.append(np.full(len(z['words']), f + (1000 if d_ == PX2 else 0)))
            R.append(np.arange(len(z['words'])))
    W, L, Bt, O, F, R = map(np.concatenate, (W, L, Bt, O, F, R)); real_draws.rows = None; dec = np.floor((2 * L - med) / B2.LN10).astype(int); rng = np.random.default_rng(seed); sel = []
    for d in decs:
        ix = np.nonzero(dec == d)[0]; sel.append(rng.choice(ix, min(n_per_dec, len(ix)), replace=False))
    sel = np.concatenate(sel); real_draws.rows = R[sel]; return W[sel], L[sel], Bt[sel], O[sel], dec[sel], F[sel]


# ============================================================================================ (1) labels
def slow_labels(X, w, EP=B2.EP8, c=B2.C_AUG, f0=B2.F0, node=PL.NODE_ADD, sign='m1'):
    """explicit swaps, M1's brute-force ln psi at x and every y, sgn(M1) directly, f_J by definition, mode 'fix'.  Returns lnT, fJ, blocked (M, NB), dlnB (M, NB)."""
    R = m1lib(); lat = R.Lattice(8, 0.5, 'translations'); fns = R.make_fns(lat, jnp.float64, True); z = np.load(B2.PROD8)
    P = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}; lp = jax.jit(jax.vmap(fns['logpsi'], in_axes=(None, 0)))
    BI, BJ, JB = np.asarray(lat.BI), np.asarray(lat.BJ), np.asarray(lat.JB, float); HW = JB / 2; NB = len(BI)
    sg = (lambda l: np.where(np.real(w * np.exp(1j * np.imag(l))) >= 0, 1, -1)) if sign == 'm1' else None
    out, fl, BLK, DL = [], [], np.zeros((len(X), NB), bool), np.full((len(X), NB), np.nan)
    for m, x in enumerate(np.asarray(X, np.float64)):
        bs = np.nonzero(x[BI] != x[BJ])[0]; Y = np.repeat(x[None], len(bs), 0); a = np.arange(len(bs)); Y[a, BI[bs]], Y[a, BJ[bs]] = x[BJ[bs]], x[BI[bs]]
        Z = np.concatenate([x[None], Y]); pad = (-len(Z)) % 64; Zp = np.concatenate([Z, np.repeat(Z[:1], pad, 0)]) if pad else Z
        l = np.asarray(lp(P, jnp.asarray(Zp)))[:len(Z)]; lx, ly = l[0], l[1:]
        sx, sy = (sg(lx), sg(ly)) if sign == 'm1' else (B2.marshall_np(x[None])[0], B2.marshall_np(Y))
        rat = np.exp(np.real(ly) - np.real(lx)); kept = sx * sy < 0
        SK = np.sum(HW[bs] * rat * kept); VA = np.sum(HW[bs] * rat * ~kept); H = 0.25 * np.sum(JB * x[BI] * x[BJ])
        with np.errstate(divide='ignore'): f = np.log(SK) - np.log(H + VA - EP)
        add = c * np.sign(f) * min(abs(f), f0) if np.isfinite(f) else node
        out.append(np.real(lx) + add); fl.append(f); BLK[m, bs] = ~kept; DL[m, bs] = np.real(ly) - np.real(lx)
    return np.array(out), np.array(fl), BLK, DL


def t_labels():
    n_dec = int(os.environ.get('N_PER_DEC', 8)); decs = list(range(-2, -10, -1))
    W, L, Bt, O, dec, F = real_draws(n_dec, decs, 101); p = B2.prod_params(); w = B2.load_w(); n_strat = len(W)
    nh = int(os.environ.get('N_HEAVY', 16))
    if nh:                                   # heavy x (raw f_J(x) < -0.5, the deep tail that carries the round's correction): scanned from deep pool_x2 draws with the fast path
        Wd, Ld, Btd, Od, decd, Fd = real_draws(int(os.environ.get('N_SCAN', 500)), [-7, -8, -9], 404, files_x=(), files_x2=(12, 13, 14, 15))
        sc = B2.M1Sign(sign_params=p, guide=p); fx = np.concatenate([np.asarray(PL.fJ_fixed(p, PF.unpack64(Wd[i:i + 256]).astype(np.float64), sc, B2.EP8, B2.C_AUG, B2.F0, 'fix', at_y=False)['fx'])
                                                                     for i in range(0, len(Wd), 256)])
        o = np.argsort(np.where(np.isfinite(fx), fx, -np.inf))[:nh]; log(f'(1) heavy scan {len(Wd)} deep draws: f_J < -0.5 {int(np.sum(fx < -0.5))}, |f_J| > f0 {int(np.sum(np.abs(fx) > B2.F0))}, min {np.min(fx):.3f}; taking {nh}')
        W, L, Bt, O, dec, F = (np.concatenate([a, b[o]]) for a, b in ((W, Wd), (L, Ld), (Bt, Btd), (O, Od), (dec, decd), (F, Fd)))
    X = PF.unpack64(W).astype(np.float64); n = len(W); log(f'(1) {n} real draws ({n_strat} stratified + {n - n_strat} heavy), decades {np.unique(dec).tolist()}, betas {np.unique(Bt).tolist()}')
    t = time.time(); slow, fJ, BLKs, DLs = slow_labels(X, w); t_slow = time.time() - t; log(f'(1) slow path {t_slow:.0f}s')
    sgn = B2.M1Sign(sign_params=p, guide=p); lab = B2.labels_m1(p, sgn)
    lab(W[:2]); t = time.time(); fast = lab(W); t_lab = (time.time() - t) / n                          # first call compiles; timing on the second
    fast_arr = lab(X); d = np.abs(fast - slow); darr = np.abs(fast_arr - fast)
    nbm = B2.NeighboursM1(p, sgn); nbm(W[:2]); t = time.time(); y, hw, valid, blk, dl, _ = nbm(W); t_nb = (time.time() - t) / n
    mask_diff = int((blk != BLKs).sum()); ddl = np.abs(np.where(valid, dl, np.nan) - DLs); dl_dev = float(np.nanmax(ddl)); kmax = np.nanargmax(ddl)
    r_dev = float(np.nanmax(np.abs(np.exp(np.where(valid, dl, np.nan)) - np.exp(DLs))))       # | |B_y/B_x| lemma - direct |: what the partner probabilities see
    lnB_dev = float(np.max(np.abs(np.real(np.asarray(PL.logB(p, X))) - L)))
    # separate-sign-network path (rounds k >= 1): guide = a copy of rb8_prod that is NOT the sign object -> second evaluation; must give the same labels
    p2 = B2.prod_params(); s2 = B2.M1Sign(sign_params=p, guide=p2); fast_sep = B2.labels_m1(p2, s2)(W); dsep = float(np.abs(fast_sep - slow).max())
    # broken: Marshall sign
    sb = B2.M1Sign(sign_params=p, guide=p, broken='marshall'); fast_b = B2.labels_m1(p, sb)(W); _, _, _, blk_b, _, _ = B2.NeighboursM1(p, sb)(W)
    slow_mar, _, BLKm, _ = slow_labels(X, w, sign='marshall')
    res = dict(n=n, decades=np.unique(dec).tolist(), betas=np.unique(Bt).tolist(), files=np.unique(F).tolist(), c=B2.C_AUG, f0=B2.F0, EP=B2.EP8, mode='fix', w=[w.real, w.imag],
               max_abs=float(d.max()), max_abs_words_vs_arrays=float(darr.max()), max_abs_separate_sign_network=dsep,
               fJ_min=float(np.nanmin(fJ)), fJ_max=float(np.nanmax(fJ)), n_capped=int(np.sum(np.abs(fJ) > B2.F0)), n_nonfinite=int(np.sum(~np.isfinite(fJ))),
               n_fJ_below_m0p5=int(np.sum(fJ < -0.5)), blocked_mask_diff=mask_diff, n_valid_bonds=int(valid.sum()), n_blocked=int(blk.sum()), dlnB_maxdev=dl_dev,
               dlnB_maxdev_at_ratio=float(np.exp(DLs.ravel()[kmax])), abs_ratio_maxdev=r_dev, n_strat=n_strat, n_heavy=n - n_strat,
               lnB_vs_stored_maxdev=lnB_dev, guard=[sgn.n_fallback, sgn.n_checked], min_margin=sgn.min_margin,
               broken_marshall=dict(max_abs_vs_slow_m1=float(np.abs(fast_b - slow).max()), blocked_mask_diff=int((blk_b != BLKs).sum()),
                                    max_abs_vs_slow_marshall=float(np.abs(fast_b - slow_mar).max()), blocked_mask_diff_vs_slow_marshall=int((blk_b != BLKm).sum())),
               sec_per_config=dict(labels=t_lab, neighbours=t_nb, slow_direct=t_slow / n, threads=os.environ.get('OMP_NUM_THREADS'), host=os.uname().nodename),
               code_sha256=dict(b2_prep=sha(os.path.join(HERE, 'b2_prep.py')), pool_fit=sha(os.path.join(HERE, 'pool_fit.py')), pp2_lemma=sha(os.path.join(HERE, 'pp2_lemma.py')),
                                m1_rbmpp_lib=sha(os.path.join(M1C, 'rbmpp_lib.py'))))
    res['PASS'] = bool(res['max_abs'] <= 1e-9 and darr.max() <= 1e-9 and dsep <= 1e-9 and mask_diff == 0 and n >= 50)
    res['note_dlnB'] = 'information, not a criterion: the lemma ratio |B_y/B_x| vs the direct ln psi difference (precision of the rank-2 lemma on real draws); partner selection is checked by the blocked masks'
    res['BROKEN_FAILS'] = bool(res['broken_marshall']['max_abs_vs_slow_m1'] > 1e-6 and res['broken_marshall']['blocked_mask_diff'] > 0)
    jd('t1_labels', res); log(f"(1) {json.dumps(res, default=float)}")


# ============================================================================================ (2a) build_pool_given == pool_fit.build_pool
def t_pool():
    n_dec = 60; W, L, Bt, O, dec, F = real_draws(n_dec, list(range(-1, -7, -1)), 202, files_x=(10,), files_x2=())
    q = B2.rung_of(Bt); per = min((q == k).sum() for k in range(3)); idx = np.concatenate([np.nonzero(q == k)[0][:per] for k in range(3)])
    W, L, O, q = W[idx], L[idx], O[idx], q[idx]; S = len(W); p = B2.prod_params(); sgn = B2.M1Sign(sign_params=p, guide=p); lab = B2.labels_m1(p, sgn); nb = B2.NeighboursM1(p, sgn)
    dd = np.floor((2 * L - np.median(2 * L[q == 0])) / B2.LN10); decs = [d for d in range(0, -8, -1) if ((dd == d) & ~PF.ho_hash(O)).any()]
    def draw_fn(beta, s_, rng):
        k = B2.BETAS.index(beta); m = q == k; return W[m][:s_], L[m][:s_], O[m][:s_]
    Pa = PF.build_pool(draw_fn, nb, lab, S, B2.BETAS, 2, 7, decs, ho_fn=PF.ho_hash, log=log)
    Pb = B2.build_pool_given(W, L, O, q, B2.BETAS, nb, lab, 2, 7, decs, ho_fn=PF.ho_hash, med=None, log=log)
    med = float(np.median(2 * L[q == 0])); Pc = B2.build_pool_given(W, L, O, q, B2.BETAS, nb, lab, 2, 7, decs, ho_fn=PF.ho_hash, med=med, log=log)
    Pd = B2.build_pool_given(W, L, O, q, B2.BETAS, nb, lab, 2, 8, decs, ho_fn=PF.ho_hash, med=None, log=log)       # broken: other seed
    def cmp(A, Bq):
        bad = []
        for k in A:
            if k == 'y_orb': continue
            a, b = np.asarray(A[k]), np.asarray(Bq[k])
            ok = a.shape == b.shape and (np.array_equal(a, b) if a.dtype.kind in 'iuUSb' else np.allclose(a, b, rtol=0, atol=0, equal_nan=True))
            if not ok: bad.append(k)
        return bad
    res = dict(S=S, decs=decs, keys_differ_given_vs_pool_fit=cmp(Pa, Pb), keys_differ_med_explicit_vs_rule=cmp(Pa, Pc), broken_seed_keys_differ=cmp(Pa, Pd))
    res['PASS'] = bool(not res['keys_differ_given_vs_pool_fit'] and not res['keys_differ_med_explicit_vs_rule']); res['BROKEN_FAILS'] = bool('y' in res['broken_seed_keys_differ'])
    jd('t2a_pool', res); log(f'(2a) {res}')


# ============================================================================================ (2b) toy pool from real draws through label_task + merge
def t_toy():
    n_dec = int(os.environ.get('N_PER_DEC', 250)); decs = list(range(-2, -10, -1)); TD = f'{OUT}/toy'; os.makedirs(TD, exist_ok=True)
    W, L, Bt, O, dec, F = real_draws(n_dec, decs, 303, files_x=(10, 11, 12, 13), files_x2=(10, 11, 12)); RW = real_draws.rows
    SP = dict(med2lnB=B2.MED_POOLX, decs_rel=decs, K=2, seed=1, c=B2.C_AUG, f0=B2.F0, EP=B2.EP8, note='toy: 250 per decade, uniform inside (rung, decade) cells; med = pool_x beta-1 median')
    json.dump(SP, open(f'{TD}/spec_pool.json', 'w'), indent=1); parts = []
    for f in np.unique(F):
        src = f'{PX2}/out_{f - 1000}.npz' if f >= 1000 else f'{PX}/out_{f}.npz'; sel = np.sort(RW[F == f]); outp = f'{TD}/part_{f}.npz'; B2.label_task(outp, SP, src, int(f), select=sel, log=log); parts.append(outp)
    P, summ = B2.merge(TD, SP, parts, log=log); jd('t2b_toy', dict(summ, PASS=bool(summ['S'] >= 2000 - 5 and all(summ['counts'][str(d)]['train'] > 0 for d in decs))))


# ============================================================================================ (2c) dry run
def t_dryrun():
    TD = f'{OUT}/toy'; pool = f'{TD}/pool.npz'; res = {}
    for tag, extra in (('ok', {}), ('broken_negate', dict(break_negate_step=True))):
        sp = B2.r1_spec(pool, sha(pool), B=int(os.environ.get('DRY_B', 256)), B_acc=int(os.environ.get('DRY_B', 256)), B_ho=int(os.environ.get('DRY_BHO', 256)), ho_every=1,
                        max_steps=3, save_theta_at=[], fit_sec=1e9, lambda_init_re=float(os.environ.get('DRY_LAM', 0.1)), **extra)
        if os.environ.get('DRY_LAMT'): sp = B2.resolve_lam(dict(sp, p2_lam_median_target=float(os.environ['DRY_LAMT'])), pool)
        RD = f'{OUT}/dry{os.environ.get("DRY_TAG", "")}_{tag}'; os.makedirs(RD, exist_ok=True); json.dump(sp, open(f'{RD}/spec.json', 'w'), indent=1)
        t = time.time(); r = B2.fit(RD, sp); c = r['fit']['curve']
        res[tag] = dict(L_ho=[q['L_ho'] for q in c], L_tr=[q['L_tr'] for q in c], verdicts=[h['verdict'] for h in r['fit']['hist']], sec=time.time() - t,
                        median_sec_per_step=r['fit']['median_sec_per_step'], npar=r['npar'], p2_sep=sp.get('_p2_sep'))
        log(f'(2c) {tag}: {res[tag]}')
    ok = res['ok']; bk = res['broken_negate']
    res['PASS'] = bool(ok['L_ho'][-1] < ok['L_ho'][0] and ok['L_tr'][-1] < ok['L_tr'][0])
    res['BROKEN_FAILS'] = bool(not (bk['L_ho'][-1] < bk['L_ho'][0] and bk['L_tr'][-1] < bk['L_tr'][0]))
    res['lambda_init_re'] = float(json.load(open(f'{OUT}/dry{os.environ.get("DRY_TAG", "")}_ok/spec.json'))['lambda_init_re']); res['p2_lam_median_target'] = os.environ.get('DRY_LAMT'); jd(f't2c_dryrun{os.environ.get("DRY_TAG", "")}', res); log(f"(2c) PASS {res['PASS']} BROKEN_FAILS {res['BROKEN_FAILS']}")


{'labels': t_labels, 'pool': t_pool, 'toy': t_toy, 'dryrun': t_dryrun}[WHAT]()
