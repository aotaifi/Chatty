"""pool_fit8_test.py: checks (a)-(e) of the 8x8 model and labels_fn in pool_fit.py (PLAN_pfaffian.md "8x8 BUILD, route 2", piece (1b)); each check has a deliberately broken
version that must fail.   usage: python pool_fit8_test.py OUTDIR {a|b|c|d|e|all} [--small]      (CPU; JAX_PLATFORMS=cpu; --small = quick dev sizes, not for the record)
(a) logA of the pool_fit 8x8 model = Re pp2_lemma.logB (and the phase mod 2 pi) on >= 200 random S^z = 0 configs, f64, two parameter sets (production warm start with a finite-
    separation P_2; a random complex two-PP + complex RBM) and, at lam = 0, the production evaluator rbmpp_lib.make_fns.  Pass <= 1e-10 (abs).  Broken: parity dropped (phase only);
    RBM bias conjugated; flip images' sign dropped.
(b) gradient vs 4th-order central finite differences on >= 50 random parameters of every block (P_1: Re/Im f_1, P_2: Re/Im f_2, RBM: Re/Im b, W, lam: Re/Im), (b1) d logA/d theta
    for 3 configurations through jac_j, (b2) the batch LOSS gradient (grad_from_G on Net.jac rows of a synthetic pool batch).  Pass <= 1e-6 relative.
    Broken: the derivative with respect to f_2 / lam / W cut (the value is unchanged, so (a) cannot see it).
(c) one damped GN step (pool_fit.fit, arm 'pp8') on a small synthetic 8x8 pool whose target is a perturbed copy of the model: the fixed-train and held-out losses both fall.
    (c2: the same pipeline with the real Labels8, c = 4.)  Broken: break_negate_step.
(d) Labels8 (pp2_lemma.fJ_fixed mode 'fix', c = 4, f0 and E' from the spec) vs a slow path (explicit swaps, the pool_fit model's direct logA, f_J by r5_bare.fJ_of's definition
    ln SK - ln(H_xx + VA - E') with the fix applied) on >= 50 configs, plus a few with no kept bond (node rows).  Pass <= 1e-9.  Broken: c -> -c; cap off; another sign for kept / blocked.
(e) CPU seconds and MaxRSS per GN step at a small batch, the per-phase split, and the H100 scaling (DERIVED, not measured)."""
import os, sys, json, time, hashlib, resource
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import pool_fit as PF
import pp2_lemma as PL

OUT = sys.argv[1]; WHICH = sys.argv[2] if len(sys.argv) > 2 else 'all'; SMALL = '--small' in sys.argv
os.makedirs(OUT, exist_ok=True); T0 = time.time()
PROD = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'
PRODLIB = os.path.join(HERE, '..', '..', 'experiments', 'rbmpp_6x6')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
EP8 = -31.915135844116154                                             # M1 p=1 consts.json E_total of the production guide (gfmc_m13b/prod/rb8/V/consts.json)


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); open(f'{OUT}/test.log', 'a').write(s + '\n')


def jd(name, obj):
    json.dump(obj, open(f'{OUT}/{name}.json', 'w'), indent=1, default=float)


def rand_cfg(rng, B, N=64):
    k = np.argsort(rng.random((B, N)), 1); S = -np.ones((B, N)); np.put_along_axis(S, k[:, :N // 2], 1.0, 1); return S


def marshall(X):
    """test-stub sign: prod over sublattice A (x + y even) of sigma = (-1)^(# down on A).  X (M, 64) +-1 -> (M,) +-1."""
    A = np.array([(i % 8 + i // 8) % 2 == 0 for i in range(64)]); return np.prod(np.where(A[None, :], np.asarray(X, np.float64), 1.0), 1)


def other_sign(X):
    """another stub: Marshall times (-1)^(# up spins among sites 0..7 pairs) (different relative signs between neighbours)."""
    X = np.asarray(X, np.float64); return marshall(X) * np.prod(np.where(np.arange(64)[None, :] < 8, X, 1.0), 1)


def const_sign(X):
    return np.ones(len(X))


def thetas():
    SP = dict(warm_npz=PROD, p2_sigma_rel=0.3, p2_seed=3, lambda_init_re=0.1, lambda_init_im=0.0)
    th_prod = PF.warm_theta8(SP); rng = np.random.default_rng(11); N, M = 64, 16; R = np.sqrt(np.add.outer(np.arange(64) % 8, -(np.arange(64) % 8)) ** 2 + 0.0)
    cr = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    th_rand = PF.pack8(cr(N, N) * 0.35, cr(M) * 0.1, cr(N, M) * 0.08, cr(N, N) * 0.35, 0.7 * np.exp(-0.4j))
    return th_prod, th_rand, SP


# ------------------------------------------------------------------ (a)
def check_a():
    th_prod, th_rand, SP = thetas(); n = 60 if SMALL else 250; rng = np.random.default_rng(21); X = rand_cfg(rng, n); W = PF.pack64(X); out = {}
    mdl = PF.make_model8(SP); models = {'nopar': PF.make_model8(SP, _broken='nopar'), 'conjb': PF.make_model8(SP, _broken='conjb'), 'noflip': PF.make_model8(SP, _broken='noflip')}
    for name, th in (('prod_P2', th_prod), ('random', th_rand)):
        p = PF.theta8_to_pp2(th); lb = np.asarray(PL.logB(p, X)); t = jnp.asarray(th)
        lA = np.asarray(mdl['fwd_j'](t, jnp.asarray(W))); lp = np.asarray(mdl['psi_j'](t, jnp.asarray(W)))
        dre = np.abs(lA - lb.real); dim = np.abs(np.angle(np.exp(1j * (lp.imag - lb.imag)))); dre_c = np.abs(lp.real - lb.real)
        brk = {k: float(np.abs(np.asarray(m['fwd_j'](t, jnp.asarray(W))) - lb.real).max()) for k, m in models.items()}
        brk_phase = {k: float(np.abs(np.angle(np.exp(1j * (np.asarray(m['psi_j'](t, jnp.asarray(W))).imag - lb.imag)))).max()) for k, m in models.items()}
        out[name] = dict(n=n, max_dRe_logA=float(dre.max()), max_dRe_logpsi=float(dre_c.max()), max_dIm_mod2pi=float(dim.max()), logA_range=[float(lA.min()), float(lA.max())],
                         broken_max_dRe=brk, broken_max_dIm=brk_phase, pass_=bool(dre.max() <= 1e-10 and dim.max() <= 1e-10),
                         broken_fails=bool(all(max(brk[k], brk_phase[k]) > 1e-6 for k in brk)))
        log(f'(a) {name}: {out[name]}')
    # lam = 0 against the production evaluator (rbmpp_lib.make_fns, experiments/rbmpp_6x6)
    sys.path.insert(0, PRODLIB)
    import rbmpp_lib as RL
    z = np.load(PROD); lat = RL.Lattice(8, 0.5, str(z['proj'])); pr = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    lpf = jax.jit(jax.vmap(RL.make_fns(lat, use_rbm=bool(z['rbm']))['logpsi'], in_axes=(None, 0)))
    ref = np.concatenate([np.asarray(lpf(pr, jnp.asarray(X[i:i + 50]))) for i in range(0, n, 50)])
    f1, b, Wm, f2, lam = PF.unpack8(PF.pack8(z['f'].astype(complex), z['bre'] + 1j * z['bim'], z['Wre'] + 1j * z['Wim'], z['f'].astype(complex), 0.0), 64, 16)
    th0 = PF.pack8(z['f'].astype(complex), z['bre'] + 1j * z['bim'], z['Wre'] + 1j * z['Wim'], np.zeros((64, 64), complex) + 0.0, 0.0)
    lp0 = np.asarray(mdl['psi_j'](jnp.asarray(th0), jnp.asarray(W)))
    dre = np.abs(lp0.real - ref.real); dim = np.abs(np.angle(np.exp(1j * (lp0.imag - ref.imag))))
    bk = np.asarray(models['conjb']['psi_j'](jnp.asarray(th0), jnp.asarray(W)))
    bkv = float(np.max(np.abs(bk.real - ref.real) + np.abs(np.angle(np.exp(1j * (bk.imag - ref.imag))))))
    out['production_lam0'] = dict(params=PROD, params_sha256=sha(PROD), prodlib_sha256=sha(os.path.join(PRODLIB, 'rbmpp_lib.py')), n=n, max_dRe=float(dre.max()), max_dIm_mod2pi=float(dim.max()),
                                  broken_conjb=bkv, pass_=bool(max(dre.max(), dim.max()) <= 1e-10), broken_fails=bool(bkv > 1e-6))
    log(f"(a) production lam=0: {out['production_lam0']}")
    out['pp2_lemma_sha256'] = sha(os.path.join(HERE, 'pp2_lemma.py')); out['pool_fit_sha256'] = sha(os.path.join(HERE, 'pool_fit.py'))
    out['PASS'] = all(out[k]['pass_'] for k in ('prod_P2', 'random', 'production_lam0')); out['BROKEN_FAILS'] = all(out[k]['broken_fails'] for k in ('prod_P2', 'random', 'production_lam0'))
    jd('check_a', out); log(f"(a) PASS {out['PASS']} BROKEN_FAILS {out['BROKEN_FAILS']}"); return out


# ------------------------------------------------------------------ small synthetic pool (shared by b2, c, e)
def small_pool(tag='tgt', S=None, ncand=None, sigma_t=0.002, labels='perturbed'):
    """pool of the guide theta0 = warm_theta8, labels = Re logB of a perturbed copy (labels = 'perturbed') or Labels8 (labels = 'real'); returns (pool path, spec pieces)."""
    S = S or (1200 if SMALL else 2400); ncand = ncand or (3000 if SMALL else 12000); d = f'{OUT}/pool_{tag}'; path = f'{d}/pool.npz'
    th0, _, SP = thetas(); mdl = PF.make_model8(SP); os.makedirs(d, exist_ok=True); np.save(f'{d}/theta0.npy', th0)
    rng = np.random.default_rng(5); cand = rand_cfg(rng, ncand); cw = PF.pack64(cand); t0 = time.time()
    lnB = np.concatenate([np.asarray(mdl['fwd_j'](jnp.asarray(th0), jnp.asarray(cw[i:i + 256]))) for i in range(0, ncand, 256)]); log(f'[pool {tag}] {ncand} candidates logB {time.time() - t0:.0f}s, ln|B| range {lnB.min():.2f} {lnB.max():.2f}')
    draw = PF.draw_from_candidates(cw, lnB)
    # decades occupied by >= 40 draws at beta = 1 (relative to the beta = 1 median, as pool_weights does)
    dd, ll, _ = draw(1.0, 3000, np.random.default_rng(9)); med = np.median(2 * ll); dec = np.floor((2 * ll - med) / np.log(10)).astype(int)
    u, cnt = np.unique(dec, return_counts=True); decs = sorted([int(a) for a, c in zip(u, cnt) if c >= 40], reverse=True); log(f'[pool {tag}] decades with >= 40 beta=1 draws: {dict(zip(u.tolist(), cnt.tolist()))} -> decs_rel {decs}')
    rngt = np.random.default_rng(77); th_t = th0 + sigma_t * rngt.standard_normal(th0.size); np.save(f'{d}/theta_target.npy', th_t)
    guide = PF.theta8_to_pp2(th0)
    if labels == 'perturbed':
        lab = lambda cfg, orb=None: np.concatenate([np.asarray(mdl['fwd_j'](jnp.asarray(th_t), jnp.asarray(np.asarray(cfg, np.uint64)[i:i + 256]))) for i in range(0, len(cfg), 256)])
    else:
        lab = PF.Labels8(guide, marshall, EP8, 4.0, 1.0, 'fix')
    t0 = time.time()
    P = PF.build_pool(draw, PF.Neighbours8(guide, marshall), lab, S, [1.0, 0.5, 0.25], 2, 3, decs, ho_fn=PF.ho_hash, log=log); np.savez(path, **P)
    summ = PF.pool_summary(P); summ['sec_build'] = time.time() - t0; summ['sigma_t'] = sigma_t; summ['labels'] = labels; PF.jdump(f'{d}/pool_summary.json', summ)
    log(f'[pool {tag}] built S={S} in {time.time() - t0:.0f}s; no_blocked {summ["no_blocked"]}; unique x orbits {summ["unique_x_orbits"]}; counts ' + ' '.join(f'{k}:{v["all"]}' for k, v in summ['counts'].items()))
    return path, th0, SP


def pool_path(tag, **kw):
    p = f'{OUT}/pool_{tag}/pool.npz'
    return (p, np.load(f'{OUT}/pool_{tag}/theta0.npy'), thetas()[2]) if os.path.exists(p) else small_pool(tag, **kw)


# ------------------------------------------------------------------ (b)
def fd_rel(g, fd):
    """componentwise relative error with a floor (1e-3 of the largest |g| of the set) and the vector relative error."""
    sc = max(np.abs(g).max(), np.abs(fd).max()); fl = 1e-3 * sc
    return float(np.max(np.abs(g - fd) / np.maximum(np.abs(fd), fl))), float(np.linalg.norm(g - fd) / max(np.linalg.norm(fd), 1e-300))


def central4(fun, th, i, h):
    e = np.zeros(th.size); e[i] = h
    return (-fun(th + 2 * e) + 8 * fun(th + e) - 8 * fun(th - e) + fun(th - 2 * e)) / (12 * h)


def subset_idx(rng, n_each=10):
    n2 = 64 * 64; M = 16; N = 64; o = {}; o['Re f1'] = np.arange(0, n2); o['Im f1'] = n2 + np.arange(n2); b0 = 2 * n2
    o['Re b'] = b0 + np.arange(M); o['Im b'] = b0 + M + np.arange(M); w0 = b0 + 2 * M
    o['Re W'] = w0 + np.arange(N * M); o['Im W'] = w0 + N * M + np.arange(N * M); f20 = w0 + 2 * N * M
    o['Re f2'] = f20 + np.arange(n2); o['Im f2'] = f20 + n2 + np.arange(n2); o['lam'] = f20 + 2 * n2 + np.arange(2)
    return {k: np.sort(rng.choice(v, size=min(len(v), 2 if k == 'lam' else (6 if 'b' in k and 'W' not in k else n_each)), replace=False)) for k, v in o.items()}


def check_b():
    th_prod, th_rand, SP = thetas(); out = {}; rng = np.random.default_rng(31); X = rand_cfg(rng, 3); W = jnp.asarray(PF.pack64(X)); h = 1e-4
    idx = subset_idx(rng); allidx = np.concatenate(list(idx.values())); log(f'(b) {len(allidx)} parameters ' + ' '.join(f'{k}:{len(v)}' for k, v in idx.items()))
    variants = {'ok': PF.make_model8(SP), 'broken_grad_f2': PF.make_model8(SP, _broken='grad_f2'), 'broken_grad_lam': PF.make_model8(SP, _broken='grad_lam'),
                'broken_grad_W': PF.make_model8(SP, _broken='grad_W')}
    ref = variants['ok']
    for name, th in (('prod_P2', th_prod), ('random', th_rand)):
        res = {}
        fun = lambda t: np.asarray(ref['fwd_j'](jnp.asarray(t), W))                                              # (3,)
        fd = np.stack([central4(fun, th, i, h) for i in allidx], 1)                                              # (3, nsub)
        for vn, m in variants.items():
            g = np.asarray(m['jac_j'](jnp.asarray(th), m['bits'](W)))[:, allidx]
            rel = [fd_rel(g[c], fd[c]) for c in range(3)]; per_block = {}
            for k, v in idx.items():
                pos = np.searchsorted(allidx, v); per_block[k] = max(fd_rel(g[c][pos], fd[c][pos])[0] for c in range(3))
            res[vn] = dict(max_rel_componentwise=max(r[0] for r in rel), max_rel_vector=max(r[1] for r in rel), per_block=per_block)
        out[name] = dict(b1=res, n_params=len(allidx), h=h, stencil='4th-order central')
        out[name]['pass_'] = bool(res['ok']['max_rel_componentwise'] <= 1e-6)
        out[name]['broken_fails'] = {vn: bool(res[vn]['max_rel_componentwise'] > 1e-6) for vn in variants if vn != 'ok'}
        log(f"(b1) {name}: ok comp {res['ok']['max_rel_componentwise']:.2e} vec {res['ok']['max_rel_vector']:.2e}; broken " +
            ' '.join(f"{vn} {res[vn]['max_rel_componentwise']:.2e}" for vn in variants if vn != 'ok'))
    # (b2) the batch loss gradient through Net.jac + grad_from_G, on a synthetic pool batch
    pp, th0, SPp = pool_path('tgt'); P = PF.load_pool(pp); sampler = PF.PoolSampler(P, False, 1, 1.0); B = 24 if SMALL else 48; bt = sampler.batch(B)
    per_sample, grad_from_G, _ = PF._jfuns(); sel = idx
    net_l = PF.Net(ref); net_l.FCH = 3 * B                                                                   # no padding to 512 rows
    L_of = lambda m, th, c: (lambda S_: float(jnp.mean(per_sample(S_, bt['lx'], bt['ly'], bt['wb'], c)[0])))(
        net_l.fwd(jnp.asarray(th), jnp.concatenate([bt['x'], bt['y'].reshape(-1)])))
    c0 = float(np.mean(np.asarray(net_l.fwd(jnp.asarray(th0), jnp.concatenate([bt['x'], bt['y'].reshape(-1)]))[:B]) - np.asarray(bt['lx']))); resb = {}
    fdL = np.array([central4(lambda t: L_of(ref, t, c0), th0, i, h) for i in allidx])
    for vn, m in variants.items():
        net = PF.Net(m); Xall = jnp.concatenate([bt['x'], bt['y'].reshape(-1)]); S_ = net.fwd(jnp.asarray(th0), Xall); G = net.jac(jnp.asarray(th0), Xall)
        a = jnp.ones(B) / B; gt, _ = grad_from_G(G, S_, bt['lx'], bt['ly'], bt['wb'], c0, a); gt = np.asarray(gt)[allidx]
        r = fd_rel(gt, fdL); pb = {k: fd_rel(gt[np.searchsorted(allidx, v)], fdL[np.searchsorted(allidx, v)])[0] for k, v in idx.items()}
        resb[vn] = dict(max_rel_componentwise=r[0], max_rel_vector=r[1], per_block=pb)
    out['b2_loss'] = dict(pool=pp, pool_sha256=sha(pp), B=B, rows=3 * B, res=resb, pass_=bool(resb['ok']['max_rel_componentwise'] <= 1e-6),
                          broken_fails={vn: bool(resb[vn]['max_rel_componentwise'] > 1e-6) for vn in variants if vn != 'ok'})
    log(f"(b2) loss gradient: ok comp {resb['ok']['max_rel_componentwise']:.2e} vec {resb['ok']['max_rel_vector']:.2e}; broken " + ' '.join(f"{vn} {resb[vn]['max_rel_componentwise']:.2e}" for vn in variants if vn != 'ok'))
    out['PASS'] = bool(out['prod_P2']['pass_'] and out['random']['pass_'] and out['b2_loss']['pass_'])
    out['BROKEN_FAILS'] = bool(all(all(out[k]['broken_fails'].values()) for k in ('prod_P2', 'random', 'b2_loss')))
    jd('check_b', out); log(f"(b) PASS {out['PASS']} BROKEN_FAILS {out['BROKEN_FAILS']}"); return out


# ------------------------------------------------------------------ (c)
def run_fit(tag, pp, th0, spec_extra, sub):
    d = f'{OUT}/{sub}'; os.makedirs(d, exist_ok=True); np.save(f'{d}/theta0.npy', th0)
    B = 128 if SMALL else 256
    SP = dict(arm='pp8', pool=pp, warm_theta8=f'{d}/theta0.npy', B=B, B_acc=B, B_ho=B, K=2, lam_b=1.0, lam0=1e-2, lam_cap=1.0, tau=0.05, tau_min=0.01, anneal_steps=20, z_acc=2.0,
              ho_every=1, flat_tol=0.02, max_steps=1, save_theta_at=[], seed=1); SP.update(spec_extra)
    t = time.time(); r = PF.fit(d, SP); c = r['fit']['curve']
    return dict(L_ho=[q['L_ho'] for q in c], L_tr=[q['L_tr'] for q in c], verdicts=[h['verdict'] for h in r['fit']['hist']], accepted=[h['ok'] for h in r['fit']['hist']],
                sec_per_step=r['fit']['median_sec_per_step'], npar=r['npar'], spec=SP, sec=time.time() - t)


def check_c():
    out = {}
    for tag, kw in (('tgt', dict(labels='perturbed')), ('real', dict(labels='real'))):
        pp, th0, _ = pool_path(tag, **kw) if tag == 'tgt' else pool_path(tag, **kw)
        ok = run_fit(tag, pp, th0, {}, f'fit_{tag}_ok'); br = run_fit(tag, pp, th0, {'break_negate_step': True}, f'fit_{tag}_broken_negate')
        dec = lambda r: (r['L_ho'][1] < r['L_ho'][0] and r['L_tr'][1] < r['L_tr'][0])
        out[tag] = dict(pool=pp, pool_sha256=sha(pp), ok=ok, broken_negate_step=br, rel_drop_ho=1 - ok['L_ho'][1] / ok['L_ho'][0], rel_drop_tr=1 - ok['L_tr'][1] / ok['L_tr'][0],
                        pass_=bool(dec(ok)), broken_fails=bool(not dec(br)))
        log(f"(c) {tag}: ok L_ho {ok['L_ho']} L_tr {ok['L_tr']} {ok['verdicts']}; broken L_ho {br['L_ho']} L_tr {br['L_tr']} {br['verdicts']}; pass {out[tag]['pass_']} broken_fails {out[tag]['broken_fails']}")
    # multi-step on the perturbed-copy pool: the loss keeps falling (informative)
    pp, th0, _ = pool_path('tgt'); ms = run_fit('tgt', pp, th0, dict(max_steps=3 if SMALL else 6), 'fit_tgt_multistep'); out['multistep'] = ms
    log(f"(c) multistep: L_ho {ms['L_ho']} L_tr {ms['L_tr']} {ms['verdicts']}")
    out['PASS'] = bool(out['tgt']['pass_']); out['PASS_c2'] = bool(out['real']['pass_']); out['BROKEN_FAILS'] = bool(out['tgt']['broken_fails'] and out['real']['broken_fails'])
    jd('check_c', out); log(f"(c) PASS {out['PASS']} (c2 real labels {out['PASS_c2']}) BROKEN_FAILS {out['BROKEN_FAILS']}"); return out


# ------------------------------------------------------------------ (d)
def slow_lnT(th, X, sign_fn, EP, c, f0, node=-69.0):
    """per configuration: explicit swaps, the pool_fit model's direct logA of x and of every swapped y, f_J by r5_bare.fJ_of's definition
    (SK = sum_kept hw ratio, VA = sum_blocked hw ratio, den = [sum JB / 4 - sum_valid JB / 2] + VA - E', f_J = ln SK - ln den), then the fix (no-kept -> node, else c sign min(|f|, f0))."""
    lt = PF.lat8(); BI, BJ, JB, HW = lt['BI'], lt['BJ'], lt['JB'], lt['HW']; mdl = PF.make_model8({}); out = []; fl = []
    for x in X:
        lx = float(mdl['fwd_j'](jnp.asarray(th), jnp.asarray(PF.pack64(x[None])))[0]); v = np.nonzero(x[BI] != x[BJ])[0]
        Y = np.repeat(x[None], len(v), 0)
        for k, b in enumerate(v): Y[k, BI[b]], Y[k, BJ[b]] = x[BJ[b]], x[BI[b]]
        Yp = np.concatenate([Y, np.repeat(x[None], 256 - len(v), 0)])                                     # fixed shape: one compile
        ly = np.asarray(mdl['fwd_j'](jnp.asarray(th), jnp.asarray(PF.pack64(Yp))))[:len(v)]; ratio = np.exp(ly - lx)
        sx = sign_fn(x[None])[0]; sy = np.asarray(sign_fn(Y)); blk = sx * sy > 0; kpt = sx * sy < 0
        VA = np.sum(HW[v] * ratio * blk); SK = np.sum(HW[v] * ratio * kpt)
        den = 0.25 * JB.sum() - 0.5 * JB[v].sum() + VA - EP
        with np.errstate(divide='ignore', invalid='ignore'): fJ = np.log(SK) - np.log(den)
        add = c * np.sign(fJ) * min(abs(fJ), f0) if np.isfinite(fJ) else node
        out.append(lx + add); fl.append(fJ)
    return np.array(out), np.array(fl)


def check_d():
    th0, _, SP = thetas(); guide = PF.theta8_to_pp2(th0); n = 20 if SMALL else 60; rng = np.random.default_rng(41); X = rand_cfg(rng, n); res = {}
    nn_ = 4 if SMALL else 8; Xc = rand_cfg(rng, nn_)
    f0 = 0.3; slow, fJ = slow_lnT(th0, X, marshall, EP8, 4.0, f0)
    log(f'(d) raw f_J over {n} configs: min {np.nanmin(fJ):.3f} max {np.nanmax(fJ):.3f}; capped (|f_J| > f0 = {f0}) {int(np.sum(np.abs(fJ) > f0))}; non-finite {int(np.sum(~np.isfinite(fJ)))}')
    words = PF.pack64(X)
    lab = PF.Labels8(guide, marshall, EP8, 4.0, f0, 'fix'); fast = lab(words); d1 = np.abs(fast - slow)
    fast_pm = PF.Labels8(guide, marshall, EP8, 4.0, f0, 'fix')(X); dpm = np.abs(fast_pm - fast)
    brk = {}
    for kind in ('c_sign', 'no_cap', 'sign_other'):
        lb = PF.Labels8(guide, marshall, EP8, 4.0, f0, 'fix', _break=kind, sign_fn_alt=other_sign)(words); brk[kind] = float(np.abs(lb - slow).max())
    res['marshall'] = dict(n=n, f0=f0, EP=EP8, c=4.0, max_abs=float(d1.max()), max_abs_words_vs_arrays=float(dpm.max()), n_capped=int(np.sum(np.abs(fJ) > f0)), n_nonfinite=int(np.sum(~np.isfinite(fJ))),
                           lnT_range=[float(slow.min()), float(slow.max())], broken_max_abs=brk, pass_=bool(d1.max() <= 1e-9), broken_fails={k: bool(v > 1e-6) for k, v in brk.items()})
    # node rows: a constant sign has no kept bond anywhere -> every row is a node (add = -69)
    slow_c, fJc = slow_lnT(th0, Xc, const_sign, EP8, 4.0, f0)
    fast_c = PF.Labels8(guide, const_sign, EP8, 4.0, f0, 'fix')(PF.pack64(Xc)); dc = np.abs(fast_c - slow_c)
    brk_c = float(np.abs(PF.Labels8(guide, const_sign, EP8, 4.0, f0, 'fix', _break='sign_other', sign_fn_alt=marshall)(PF.pack64(Xc)) - slow_c).max())
    res['node_rows'] = dict(n=nn_, n_nonfinite=int(np.sum(~np.isfinite(fJc))), max_abs=float(dc.max()), broken_sign_other=brk_c, pass_=bool(dc.max() <= 1e-9 and np.all(~np.isfinite(fJc))), broken_fails=bool(brk_c > 1e-6))
    # a second guide: random complex two-PP, and a different E' / f0
    th_r = thetas()[1]; gr = PF.theta8_to_pp2(th_r); Xr = X[:max(10, n // 3)]
    slow_r, fJr = slow_lnT(th_r, Xr, marshall, EP8, 4.0, 0.7); fast_r = PF.Labels8(gr, marshall, EP8, 4.0, 0.7, 'fix')(PF.pack64(Xr))
    res['random_guide'] = dict(n=len(Xr), f0=0.7, max_abs=float(np.abs(fast_r - slow_r).max()), pass_=bool(np.abs(fast_r - slow_r).max() <= 1e-9))
    res['PASS'] = bool(all(res[k]['pass_'] for k in ('marshall', 'node_rows', 'random_guide')))
    res['BROKEN_FAILS'] = bool(all(res['marshall']['broken_fails'].values()) and res['node_rows']['broken_fails'])
    res['pp2_lemma_sha256'] = sha(os.path.join(HERE, 'pp2_lemma.py')); res['pool_fit_sha256'] = sha(os.path.join(HERE, 'pool_fit.py'))
    jd('check_d', res); log(f"(d) marshall {res['marshall']}; node {res['node_rows']}; random guide {res['random_guide']}; PASS {res['PASS']} BROKEN_FAILS {res['BROKEN_FAILS']}"); return res


# ------------------------------------------------------------------ (e)
def check_e():
    """phase times of one GN step on the small pool at batch sizes B (rows 3 B); MaxRSS of this process; large param-space solve timing on random G (timing only)."""
    import pool_fit as PFm
    th0, _, SP = thetas(); mdl = PF.make_model8(SP); net = PF.Net(mdl); per_sample, grad_from_G, solve_dir = PF._jfuns(); pp, _, _ = pool_path('tgt'); P = PF.load_pool(pp)
    sampler = PF.PoolSampler(P, False, 1, 1.0); th = jnp.asarray(th0); out = dict(cpu_threads=int(os.environ.get('OMP_NUM_THREADS', 0) or os.cpu_count()), host=os.uname().nodename, npar=int(th.size), steps=[])
    for B in ([32, 64] if SMALL else [128, 256, 512]):
        bt = sampler.batch(B); Xall = jnp.concatenate([bt['x'], bt['y'].reshape(-1)]); c = 0.0; rec = dict(B=B, rows=int(Xall.shape[0]))
        for rep in range(2):                                                                              # rep 0 = includes JIT compile
            t = time.time(); S = net.fwd(th, Xall); S.block_until_ready(); tf = time.time() - t
            t = time.time(); G = net.jac(th, Xall); G.block_until_ready(); tj = time.time() - t
            t = time.time(); dP, dc, prms, sc, rr = solve_dir(G, S, bt['lx'], bt['ly'], bt['wb'], c, 1e-2, 0.05); dP.block_until_ready(); ts = time.time() - t
            ab = sampler.batch(B); Xa = jnp.concatenate([ab['x'], ab['y'].reshape(-1)]); t = time.time()
            a0 = net.fwd(th, Xa); a1 = net.fwd(th + dP, Xa); a1.block_until_ready(); ta = time.time() - t
            rec[f'rep{rep}'] = dict(fwd=tf, jac=tj, solve=ts, accept_fwd2=ta, total=tf + tj + ts + ta, G_GB=float(G.size * 8 / 1e9))
        out['steps'].append(rec); log(f'(e) B={B}: {rec}'); del G
    out['maxrss_GB_after_steps'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out['pp2_lemma_sha256'] = sha(os.path.join(HERE, 'pp2_lemma.py')); out['pool_fit_sha256'] = sha(os.path.join(HERE, 'pool_fit.py'))
    jd('check_e', out); return out


def check_e2():
    """the param-space branch of solve_dir at the production batch B = 8208 (R = 3 B = 24624 rows > P + 1): time and memory on a RANDOM G (timing only), and the jac rate at a middle batch."""
    th0, _, SP = thetas(); mdl = PF.make_model8(SP); net = PF.Net(mdl); per_sample, grad_from_G, solve_dir = PF._jfuns(); out = dict(host=os.uname().nodename, cpu_threads=int(os.environ.get('OMP_NUM_THREADS', 0) or os.cpu_count()))
    Bp = 8208; R = Bp * 3; Pn = int(th0.size); rng = np.random.default_rng(0); G = jnp.asarray(rng.standard_normal((R, Pn))) * 1e-1; S = jnp.asarray(rng.standard_normal(R))
    lx = jnp.asarray(rng.standard_normal(Bp)); ly = jnp.asarray(rng.standard_normal((Bp, 2))); wb = jnp.ones((Bp, 2)); G.block_until_ready()
    for rep in range(2):
        t = time.time(); dP, dc, prms, sc, rr = solve_dir(G, S, lx, ly, wb, 0.0, 1e-2, 0.05); dP.block_until_ready()
        out[f'param_space_solve_rep{rep}'] = dict(rows=R, P=Pn, sec=time.time() - t, branch='param (R > P + 1)', G_GB=float(G.size * 8 / 1e9)); log(f"(e2) param-space solve B=8208 rep {rep}: {out[f'param_space_solve_rep{rep}']}")
    out['maxrss_GB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6; out['pool_fit_sha256'] = sha(os.path.join(HERE, 'pool_fit.py')); jd('check_e2', out); return out


if __name__ == '__main__':
    log(f'start {WHICH} small={SMALL} pool_fit sha {sha(os.path.join(HERE, "pool_fit.py"))[:12]} pp2_lemma sha {sha(os.path.join(HERE, "pp2_lemma.py"))[:12]} jax {jax.__version__} {jax.devices()}')
    fns = dict(a=check_a, b=check_b, c=check_c, d=check_d, e=check_e, e2=check_e2)
    for k in (list(fns) if WHICH == 'all' else [WHICH]): fns[k]()
    log('done')
