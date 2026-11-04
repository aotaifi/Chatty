"""pp2_lemma_bench.py: GPU benchmark of the complete 8x8 augmented guide (B design item (i) deliverable; lead's addition 2026-11-03).
Guide: B = sum_g phi_RBM(z_g) [P_1(g) + lam P_2(g)] over the 128 images (translations x flip), P_1 + RBM = production rb8_prod params,
P_2 = P_1 + noise (finite separation, seed 1), lam scaled so the two terms carry comparable weight at a reference configuration;
sign = sign8.SignNet XL (128 ch x 6 layers, random weights, f32) at x, every y and every two-swap endpoint z; f_J fixed (mode 'fix', c = 4).
usage: python pp2_lemma_bench.py OUTDIR [M ...]        (default M = 256 1024)
Measures per M and dtype (f64, f32): seconds per walker-step of aug_ratios, split into lemma (two_swap_ratios), CNN (sign at x, y, z),
rest; one_swap_ratios (plain P step) for the J/P ratio; peak device memory; f32 vs f64 max relative error of r1, r2 and the augmented ratio.
Options (2026-11-04, piece (3); all default to the A40-benchmarked behaviour): PP2B_PATH=old|pack|dedup (eval_path of the lemma; 'dedup' also
deduplicates the CNN calls), PP2B_CNN=XL|L (sign8 size), PP2B_DTYPES=f64,f32, PP2B_WALKERS=<npy of +-1 configs> (instead of the Metropolis prep),
PP2B_NO_SIGN=1 (skip the sign-option block).  The new paths add the unique / valid / padded endpoint counts (lemma_stats, cnn_stats, aug_stats) and the host bookkeeping seconds.
Walkers: configurations from a short Metropolis run of |B| (rbmpp_lib-style exchange moves on the production guide P_1 + RBM) so that d
is realistic, not random S^z = 0 states.
"""
import os, sys, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import pp2_lemma as PL
import sign8

OUT = sys.argv[1]; MS = [int(a) for a in sys.argv[2:]] or [256, 1024]
os.makedirs(OUT, exist_ok=True); T0 = time.time()
PROD = os.environ.get('PP2B_PROD', '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz')
EP = -0.4987565 * 64          # E' = FN p=0 energy of the production guide (PLAN_pfaffian.md B design anchors); cost-neutral
HOPS = os.environ.get('PP2B_HOPS', '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/p1_s1/blocks.npz')
SIGN_FIRST = os.environ.get('PP2B_SIGN_FIRST') == '1'
LEAN = os.environ.get('PP2B_LEAN') == '1'              # one timed repetition per component (no repeat), for M = 1024
SCH = int(os.environ.get('PP2B_SCHUNK', '16384'))       # sign_fn chunk
BCH = int(os.environ.get('PP2B_BCHUNK', '256'))         # endpoint chunk of the on-the-fly sign (B)
PATH = os.environ.get('PP2B_PATH', 'old'); CNN = os.environ.get('PP2B_CNN', 'XL'); DTS = tuple(os.environ.get('PP2B_DTYPES', 'f64,f32').split(','))
WALK = os.environ.get('PP2B_WALKERS')
C = 4.0; F0 = 3.0             # f0 is a placeholder (cost-neutral)
CONSTS = os.environ.get('PP2B_CONSTS', '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json')   # p=1 sign constants (alpha, E, w)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def log(m):
    s = f'[{time.time() - T0:7.1f}s] {m}'; print(s, flush=True); open(f'{OUT}/bench.log', 'a').write(s + '\n')


def peak():
    st = jax.devices()[0].memory_stats() or {}
    return st.get('peak_bytes_in_use', -1) / 2 ** 30


def walkers(p, M, seed=0, sweeps=40):
    """Metropolis on |B|^2 with exchange moves along anti-aligned bonds (f64 logB), sweeps x N moves... kept short: sweeps x 64 steps."""
    rng = np.random.default_rng(seed); lat = p.lat; N = lat.N
    X = np.ones((M, N)); X[:, N // 2:] = -1
    for r in X: rng.shuffle(r)
    lX = np.array(np.real(np.asarray(PL.logB(p, X, chunk=M))))
    acc = 0
    for it in range(sweeps * 8):
        anti = X[:, lat.BI] != X[:, lat.BJ]; sc = np.where(anti, rng.random(anti.shape), -1); b = sc.argmax(1)
        Y = X.copy(); a = np.arange(M); i, j = lat.BI[b], lat.BJ[b]; Y[a, i], Y[a, j] = X[a, j], X[a, i]
        lY = np.array(np.real(np.asarray(PL.logB(p, Y, chunk=M))))
        nY = (Y[:, lat.BI] != Y[:, lat.BJ]).sum(1)
        ok = np.log(rng.random(M)) < 2 * (lY - lX) + np.log(anti.sum(1) / nY)
        X[ok] = Y[ok]; lX[ok] = lY[ok]; acc += ok.mean()
    return X, acc / (sweeps * 8)


def blk(o):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, o); return o


def main():
    hops = np.load(HOPS)
    hpwt = float(np.mean(hops['hops']) / (int(hops['M']) * float(hops['dtau'])))
    p1 = PL.PP2.from_rbmpp_npz(PROD); lat = p1.lat
    r = np.random.default_rng(1); f2 = np.asarray(p1.f1) + 0.3 * np.std(np.real(np.asarray(p1.f1))) * (r.standard_normal((64, 64)) + 1j * r.standard_normal((64, 64)))
    x0 = np.ones((1, 64)); x0[0, 32:] = -1; np.random.default_rng(5).shuffle(x0[0])
    pP2 = PL.PP2(f2, L=8, proj=p1.proj, rbm='inner', b=p1.b, W=p1.W)
    lam = 0.5 * np.exp(np.real(complex(PL.logB(p1, x0)[0]) - complex(PL.logB(pP2, x0)[0])))
    p = p1.with_p2(f2, lam)
    net = sign8.SignNet(L=8, size=CNN, seed=0)
    f32 = jax.jit(lambda x: jnp.where(net.jax_logit(net.params, x.astype(jnp.float32)) >= 0, 1, -1))
    def sign_fn(x): return f32(x)
    sign_fn.jax_native = True
    R = dict(lean=LEAN, sign_first=SIGN_FIRST, host=os.uname().nodename, device=str(jax.devices()[0]), jax=jax.__version__, code_sha256={f: sha(os.path.join(HERE, f)) for f in
             ('pp2_lemma.py', 'pp2_lemma_bench.py', 'sign8.py')}, params=PROD, params_sha256=sha(PROD), n_images=int(lat.nI),
             lam=[float(np.real(lam)), float(np.imag(lam))], cnn=dict(size=CNN, npar=net.npar), eval_path=PATH, cpu_threads=os.environ.get('OMP_NUM_THREADS'), load_avg=os.getloadavg(), hops_per_walker_tau=hpwt,
             hops_source=HOPS + ' (copy of /scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/p1_s1/blocks.npz; mean hops / (M dtau))', runs={})
    t = time.time()
    if WALK: Xall = np.load(WALK).astype(np.float64)[:max(MS)]; acc = float('nan'); assert len(Xall) >= max(MS)
    else: Xall, acc = walkers(p, max(MS))
    R['walker_prep'] = dict(sec=time.time() - t, acc=float(acc), file=WALK)
    dd = (Xall[:, lat.BI] != Xall[:, lat.BJ]).sum(1); R['d_anti'] = dict(mean=float(dd.mean()), max=int(dd.max()), min=int(dd.min()))
    log(f'walkers ready: acc {acc:.3f} d mean {dd.mean():.1f} max {dd.max()}  peak {peak():.2f} GiB')
    dmax = PL._dmax_of(lat, Xall); R['dmax'] = dmax
    ref = {}
    def sign_options():
        # (B) s_G on the fly: s(z) = sign Re[w B(z) (1 + alpha (E_L(z) - E))], E_L from z's own one-swap lemma ratios (one fast eloc per
        # endpoint).  Timed on a subsample of endpoint configs (throughput is linear in the count), extrapolated to all d^2 endpoints.
        K = json.load(open(CONSTS)); wph = complex(*K['w']); al = K['alpha']; E1 = K['E']
        lat_JB = jnp.asarray(lat.JB)
        for dt in DTS:
            pc = p.cast(dt); rdt = jnp.float64 if dt == 'f64' else jnp.float32
            @jax.jit
            def sgB(z):
                zz = z.astype(rdt); lB, r = PL._one_j(pc, zz)
                EL = 0.25 * jnp.sum(lat_JB * zz[:, lat.BI] * zz[:, lat.BJ], 1) + jnp.sum(0.5 * lat_JB * r, 1)
                v = jnp.real(wph * jnp.exp(1j * jnp.imag(lB)) * (1 + al * (EL - E1)))
                return jnp.where(v >= 0, 1, -1)
            X = Xall[:256]; d = PL.two_swap_ratios(p, X, dt, dmax); Y, Z = blk(PL._configs(jnp.asarray(X, jnp.float32), d))
            nz_per_walker = float(np.asarray(d["bm"]).sum() / X.shape[0]); Zs = Z.reshape(-1, 64)[:8192]; NS = Zs.shape[0] // BCH * BCH
            for ch in (BCH,):
                blk(sgB(Zs[:ch]))
                t = time.perf_counter()
                for i in range(0, NS, ch): blk(sgB(Zs[i:i + ch]))
                tc = (time.perf_counter() - t) / NS
            Zc = Z.reshape(-1, 64)[:65536]; blk(PL._call_sign(sign_fn, Zc, SCH))
            t = time.perf_counter(); blk(PL._call_sign(sign_fn, Zc, SCH)); tcnn = (time.perf_counter() - t) / Zc.shape[0]
            R['sign_options_' + dt] = dict(endpoints_per_walker_valid=nz_per_walker, endpoints_per_walker_padded=float(np.prod(Z.shape[1:3])),
                A_cnn_s_per_config=tcnn, B_eloc_s_per_config=tc, A_cnn_s_per_walker_step=tcnn * np.prod(Z.shape[1:3]),
                B_eloc_s_per_walker_step=tc * np.prod(Z.shape[1:3]), B_over_A=tc / tcnn, peak_GiB=peak(), consts=CONSTS)
            log(f'sign options {dt}: {R["sign_options_" + dt]}'); del d, Y, Z
    if SIGN_FIRST: sign_options()
    for M in MS:
        X = Xall[:M]
        for dt in DTS:
            key = f'M{M}_{dt}'; res = {}
            # warm-up / compile on the same shapes
            NR = 1 if LEAN else 2
            blk(PL.one_swap_ratios(p, X, dt)); blk(PL.two_swap_ratios(p, X[:256], dt, dmax, eval_path=PATH))
            ts = []
            for rep in range(NR):
                t = time.perf_counter(); blk(PL.one_swap_ratios(p, X, dt)); ts.append(time.perf_counter() - t)
            res['t_one'] = min(ts)
            ts = []
            for rep in range(NR):
                PL.STATS.clear(); t = time.perf_counter(); d = blk(PL.two_swap_ratios(p, X, dt, dmax, eval_path=PATH)); ts.append(time.perf_counter() - t)
            res['t_lemma'] = min(ts)
            Xd = jnp.asarray(X, jnp.float32)
            if PATH == 'dedup':                                                      # CNN on the distinct configurations only (x, y_a, z_ab, all walkers)
                res['lemma_stats'] = dict(PL.STATS)
                for rep in range(NR):
                    PL.STATS.clear(); t = time.perf_counter(); blk(PL._signs_dedup(sign_fn, np.asarray(X), d, True, SCH, None)); res['t_cnn'] = time.perf_counter() - t
                res['n_sign_calls'] = int(PL.STATS['cnn_calls_dedup']); res['cnn_stats'] = dict(PL.STATS)
            else:
                Y, Z = blk(PL._configs(Xd, d))
                nz = int(np.prod(Z.shape[:3])); ny = int(np.prod(Y.shape[:2])); res['n_sign_calls'] = M + ny + nz
                if LEAN: blk(PL._call_sign(sign_fn, Xd.astype(jnp.int8), SCH)); blk(PL._call_sign(sign_fn, Y.reshape(-1, 64), SCH)); blk(PL._call_sign(sign_fn, Z.reshape(-1, 64)[:SCH], SCH))
                for rep in range(NR):                                                   # rep 0 = warm-up (compiles every chunk shape)
                    t = time.perf_counter()
                    blk(PL._call_sign(sign_fn, Xd.astype(jnp.int8), SCH)); blk(PL._call_sign(sign_fn, Y.reshape(-1, 64), SCH))
                    blk(PL._call_sign(sign_fn, Z.reshape(-1, 64), SCH)); res['t_cnn'] = time.perf_counter() - t
                del Y, Z
                res['n_valid_endpoints'] = int(np.asarray(d['bm']).sum()); res['n_padded_endpoints'] = int(np.prod(d['bm'].shape))
            if not LEAN: blk(PL.aug_ratios(p, X, sign_fn, EP, C, F0, 'fix', dt, dmax, sign_chunk=SCH, eval_path=PATH))
            t = time.perf_counter(); PL.STATS.clear(); o = blk(PL.aug_ratios(p, X, sign_fn, EP, C, F0, 'fix', dt, dmax, sign_chunk=SCH, eval_path=PATH)); res['aug_stats'] = dict(PL.STATS); res['t_aug'] = time.perf_counter() - t
            res['t_rest'] = res['t_aug'] - res['t_lemma'] - res['t_cnn']
            for k_ in ('t_one', 't_lemma', 't_cnn', 't_aug', 't_rest'): res[k_.replace('t_', 's_per_walker_step_')] = res[k_] / M
            res['J_over_P'] = res['t_aug'] / res['t_one']; res['lemma_over_P'] = res['t_lemma'] / res['t_one']
            res['s_per_walker_tau_aug'] = res['s_per_walker_step_aug'] * hpwt
            res['peak_GiB'] = peak()
            sel = dict(r1=np.asarray(o['r1']), r2=np.asarray(o['r2']), ratio=np.asarray(o['ratio']), bm=np.asarray(o['bm']), am=np.asarray(o['am']),
                       sy=np.asarray(o['sy']), sz=np.asarray(o['sz']))
            if dt == 'f64': ref[M] = sel
            else:
                q = ref[M]; am = q['am']; bm = q['bm']; ok = (q['sy'] == sel['sy']).all() and (q['sz'] == sel['sz']).all()
                rel = lambda a, b, m: float(np.max(np.abs(a[m] - b[m]) / np.abs(b[m])))
                nzr = q['ratio'] != 0
                res['f32_vs_f64'] = dict(r1_maxrel=rel(sel['r1'], q['r1'], am), r2_maxrel=rel(sel['r2'], q['r2'], bm),
                                         aug_maxrel=rel(sel['ratio'], q['ratio'], nzr), r2_rel_q999=float(np.quantile(np.abs(sel['r2'][bm] - q['r2'][bm]) / np.abs(q['r2'][bm]), 0.999)),
                                         signs_identical=bool(ok))
            R['runs'][key] = res; log(f'{key}: {res}')
            json.dump(R, open(f'{OUT}/bench.json', 'w'), indent=1, default=float)
            del o, d
    if not SIGN_FIRST and os.environ.get('PP2B_NO_SIGN') != '1': sign_options()
    R['p1_ref'] = dict(s_per_walker_tau=0.5142377596348524, source='/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/p1_s1/result.json (H100)')
    json.dump(R, open(f'{OUT}/bench.json', 'w'), indent=1, default=float); log('BENCH_DONE')


if __name__ == '__main__':
    main()
