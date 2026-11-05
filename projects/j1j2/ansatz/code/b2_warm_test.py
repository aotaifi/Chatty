"""b2_warm_test.py: R1 warm-start selection by the registered rule (PLAN_pfaffian.md "R1 warm-start scan", rule registered 2026-11-06).  CPU.
usage: python b2_warm_test.py OUTDIR scan POOL            sep in {0.03, 0.1, 0.3} x t in {0.003, 0.01, 0.05}: ln|P_2/P_1| quantiles (pool x), start held-out / train loss
       python b2_warm_test.py OUTDIR dry POOL SEP T [STEPS]   STEPS-step dry run (b2_prep.fit, B 256) from (sep, t), then P_2's projection share at the saved steps
Projection share (metric = m5pp_sfn.py's: the copy loss's quadratic form on a function change d over a fit batch, centred value rows + bond rows,
  <d1, d2> = (1/M) sum_x (d1_x - mean d1)(d2_x - mean d2) + (1/M) sum_x wb_x sum_k (d1_x - d1_yk)(d2_x - d2_yk),  wb = lam_b rho / (K Wbar_d)):
  ln B = ln P_1 + ln(1 + lam P_2/P_1);  d_P2 = [lnA(theta) - lnA(theta, lam = 0)]_n - [same]_0,  d_total = lnA(theta_n) - lnA(theta_0);
  share = <d_P2, d_total> / |d_total|^2   (also S_fn = |d_P2|^2 / |d_total|^2).  Batch: 2048 training entries (PoolSampler seed 9001) + its partners.
  Identity check: the block ln|1 + lam P_2/P_1| from pp2_lemma.logB of P_1 and P_2 separately (100 rows).  Broken: P_1's part labelled as P_2 (share_rest = 1 - share)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import pool_fit as PF, pp2_lemma as PL, b2_prep as B2

OUT = sys.argv[1]; WHAT = sys.argv[2]; POOL = sys.argv[3]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
SEPS = [0.03, 0.1, 0.3]; TS = [0.003, 0.01, 0.05]; L_B0 = 0.147


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); open(f'{OUT}/warm.log', 'a').write(s + '\n')


def patch(P):
    med = float(P['med2lnB']); PF.pool_weights = lambda a, b, c_, d, e, f, ho_fn=None: B2.pool_weights_med(a, b, c_, d, e, f, ho_fn, med)


def scan():
    P = PF.load_pool(POOL); patch(P); mdl = PF.make_model8({}); net = PF.Net(mdl); ps, _, _ = PF._jfuns()
    HOB = PF.PoolSampler(P, True, 5001, 1.0).batch(1024); TRB = PF.PoolSampler(P, False, 7001, 1.0).batch(1024)
    def loss(th):
        r = {}
        for nm, bt in (('ho', HOB), ('tr', TRB)):
            S = net.fwd(th, jnp.concatenate([bt['x'], bt['y'].reshape(-1)])); c = float(np.mean(np.asarray(S[:bt['x'].shape[0]]) - np.asarray(bt['lx'])))
            r[nm] = float(jnp.mean(ps(S, bt['lx'], bt['ly'], bt['wb'], c)[0]))
        return r
    res = dict(pool=POOL, B0=loss(jnp.asarray(PF.warm_theta8(dict(B2.r1_spec(POOL, '', lambda_init_re=0.0))))), rows=[])
    for sep in SEPS:
        sp0 = B2.r1_spec(POOL, '', p2_sigma_rel=sep); lr = B2.p2_logratio(POOL, sp0); med = float(np.median(lr)); q = [float(np.quantile(lr, a)) for a in (0.01, 0.1, 0.5, 0.9, 0.99)]
        for t in TS:
            sp = B2.resolve_lam(dict(sp0, p2_lam_median_target=t), POOL); assert abs(sp['p2_logratio_median'] - med) < 1e-12
            L = loss(jnp.asarray(PF.warm_theta8(dict(sp))))
            row = dict(sep=sep, t=t, lam=sp['lambda_init_re'], logratio_q01_10_50_90_99=q, L_ho=L['ho'], L_tr=L['tr'], ratio_ho_vs_0147=L['ho'] / L_B0, accept=bool(L['ho'] <= 1.2 * L_B0))
            res['rows'].append(row); log(json.dumps(row))
    ok = [r for r in res['rows'] if r['accept']]
    order = sorted(ok, key=lambda r: (-r['t'], -r['sep'])); res['ranking'] = [(r['sep'], r['t']) for r in order]
    json.dump(res, open(f'{OUT}/scan.json', 'w'), indent=1); log(f"B_0 start {res['B0']}; accepted pairs in rule order: {res['ranking']}")


def dry(sep, t, steps):
    P = PF.load_pool(POOL); RD = f'{OUT}/dry_sep{sep}_t{t}'; os.makedirs(RD, exist_ok=True)
    save = [s for s in (5, 10, 15) if s < steps]
    sp = B2.resolve_lam(B2.r1_spec(POOL, B2.sha(POOL), p2_sigma_rel=sep, p2_lam_median_target=t, B=256, B_acc=256, B_ho=1024, ho_every=1, max_steps=steps,
                                   save_theta_at=save, fit_sec=1e9), POOL)
    json.dump(sp, open(f'{RD}/spec.json', 'w'), indent=1)
    r = json.load(open(f'{RD}/pool_fit.json')) if os.path.exists(f'{RD}/theta.npy') and json.load(open(f'{RD}/pool_fit.json')).get('done') else B2.fit(RD, sp)   # reuse a finished fit
    patch(P); mdl = PF.make_model8({}); net = PF.Net(mdl)
    bt = PF.PoolSampler(P, False, 9001, 1.0).batch(2048); M = int(bt['x'].shape[0]); K = int(bt['y'].shape[1]); X = jnp.concatenate([bt['x'], bt['y'].reshape(-1)])
    wb = np.asarray(bt['wb']); lnA = lambda th: np.asarray(net.fwd(jnp.asarray(th), X), np.float64)
    zl = lambda th: np.concatenate([th[:-2], [0.0, 0.0]])
    th0 = PF.warm_theta8(dict(sp)); thetas = {s: np.load(f'{RD}/theta_step{s}.npy') for s in save}; thetas[steps] = np.load(f'{RD}/theta.npy')
    A0 = lnA(th0); b0 = A0 - lnA(zl(th0))
    def ip(d1, d2):
        x1, x2 = d1[:M], d2[:M]; y1, y2 = d1[M:].reshape(M, K), d2[M:].reshape(M, K)
        v = (x1 - x1.mean()) * (x2 - x2.mean()) / M; b = np.sum(wb * (x1[:, None] - y1) * (x2[:, None] - y2), 1) / M; return v, b          # wb (M, K) per-pair weights
    # identity check of the block on 100 rows: ln|1 + lam P_2/P_1| from P_1, P_2 evaluated separately
    f1, bb, W, f2, lam = PF.unpack8(thetas[steps]); P1 = PL.PP2(f1, L=8, proj='translations', rbm='inner', b=bb, W=W); P2 = PL.PP2(f2, L=8, proj='translations', rbm='inner', b=bb, W=W)
    Xs = PF.unpack64(np.asarray(bt['x'])[:100]).astype(np.float64); l1 = np.asarray(PL.logB(P1, Xs)); l2 = np.asarray(PL.logB(P2, Xs))
    ref = np.log(np.abs(1.0 + lam * np.exp(l2 - l1))); AN = lnA(thetas[steps]); blk_n = AN - lnA(zl(thetas[steps]))
    res = dict(sep=sep, t=t, lam_init=sp['lambda_init_re'], logratio_median=sp['p2_logratio_median'], identity_block_maxabs=float(np.max(np.abs(ref - blk_n[:100]))),
               curve=[dict(step=c['step'], L_ho=c['L_ho'], L_tr=c['L_tr']) for c in r['fit']['curve']], verdicts=[h['verdict'] for h in r['fit']['hist']],
               lam_final=[float(np.real(lam)), float(np.imag(lam))], median_sec_per_step=r['fit']['median_sec_per_step'], shares={})
    rng = np.random.default_rng(1)
    for s, th in sorted(thetas.items()):
        A = lnA(th); dT = A - A0; dP = (A - lnA(zl(th))) - b0
        vt, bt_ = ip(dT, dT); vp, bp = ip(dP, dT); vpp, bpp = ip(dP, dP); Nt = vt.sum() + bt_.sum()
        sh = (vp.sum() + bp.sum()) / Nt; bs = []
        for _ in range(300):
            ii = rng.integers(0, M, M); bs.append((vp[ii].sum() + bp[ii].sum()) / (vt[ii].sum() + bt_[ii].sum()))
        dR = dT - dP; vr, br = ip(dR, dT)
        res['shares'][str(s)] = dict(share_proj=float(sh), share_proj_se=float(np.std(bs)), S_fn=float((vpp.sum() + bpp.sum()) / Nt),
                                     share_value=float(vp.sum() / vt.sum()), share_bond=float(bp.sum() / bt_.sum()), norm2_total=float(Nt),
                                     broken_P1_labelled_as_P2=float((vr.sum() + br.sum()) / Nt), sum_check=float((vp.sum() + bp.sum() + vr.sum() + br.sum()) / Nt))
        log(f'step {s}: {res["shares"][str(s)]}')
    res['accept_share_ge_0.1'] = bool(res['shares'][str(steps)]['share_proj'] >= 0.1)
    json.dump(res, open(f'{RD}/share.json', 'w'), indent=1, default=float); log(json.dumps(res, default=float))


if WHAT == 'scan': scan()
else: dry(float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 20)
