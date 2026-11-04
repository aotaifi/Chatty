"""pp2_lemma_savings_test.py: piece (3) checks of the savings in pp2_lemma.py (endpoint dedup, less padding, deduplicated CNN calls).
Two stages (the runs are independent processes so that they can be farmed out; JAX_PLATFORMS=cpu):
  python pp2_lemma_savings_test.py OUTDIR run SYSTEM VARIANT M      -> OUTDIR/runs/<SYSTEM>_<VARIANT>_M<M>.{npz,json}   (aug_ratios outputs, timing, STATS)
  python pp2_lemma_savings_test.py OUTDIR compare                   -> OUTDIR/check_{a8,a6,cnn}.json  (new vs old, broken vs old; PASS / BROKEN_FAILS)
SYSTEM  a8prod : 8x8, rb8_prod params (the guide of M1 / p = 0: params_final.npz, proj translations; resolved from
                 /scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/p0_s1/{run_metadata.txt,result.json}), lam = 0, P_1 + RBM exactly as in production;
        a8two  : the same + P_2 (P_1 + noise) and lam (the pp2_lemma_bench construction); walkers = first M of 256 from a short Metropolis run of |P_1|^2
                 (savings/walkers_8x8_prod_M256_seed0.npy); sign_fn = deterministic hash sign (batch independent), c = 4, f0 = 3, E' = -0.4987565 * 64, mode 'fix'
        a6     : 6x6, pfaffian_r5 pp2_s1 theta_5 (outer RBM, full projection, lam != 0), 256 random orbit representatives of the S^z = 0 table,
                 sign_fn = the stored s_G table (pfaffian_lswo2/s_G.npy), E' = E_FN of pp2_s1, c = 4, f0 = 2
        cnn    : a8two with the L-size sign8 CNN (64 ch x 6 layers, random weights, seed 0) through make_sign_fn as the guide's sign_fn (M small)
VARIANT old (padded path, default behaviour) | pack | dedup (+ CNN dedup) | b_pair | b_trunc | b_cnnkey (deliberately broken dedup: key = bond pair
        {a, b} with the lattice index mod 2N instead of the configuration; 10% of the moves never evaluated; CNN key without sites N-2, N-1).
Compare: max abs difference of ratio, add_x, add_y, logG_x, sx, sy, sz (all entries; valid entries also separately), r1, r2, fx, fy; PASS if <= 1e-12.
"""
import os, sys, json, time, hashlib, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import pp2_lemma as PL

OUT = sys.argv[1]; STAGE = sys.argv[2]
os.makedirs(f'{OUT}/runs', exist_ok=True); T0 = time.time()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
PROD = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'
P0RUN = '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/p0_s1'
WALK8 = os.environ.get('SAV_WALKERS8', '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/savings/walkers_8x8_prod_M256_seed0.npy')
R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/pp2_s1/tab'
THETA5 = f'{R5}/theta_5.npy'; SG = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'
TOL = 1e-12
KEYS = ('ratio', 'add_x', 'add_y', 'logG_x', 'sx', 'sy', 'sz', 'r1', 'r2', 'fx', 'fy')


def log(m):
    s = f'[{time.time() - T0:7.0f}s] {m}'; print(s, flush=True); open(f'{OUT}/test.log', 'a').write(s + '\n')


def hash_sign(cfg):
    """deterministic, batch-independent +-1 function of a configuration (numpy; mixes all sites)."""
    c = (np.asarray(cfg) > 0).astype(np.uint64); N = c.shape[1]
    w = (np.arange(N, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(12345)) | np.uint64(1)
    h = (c * w[None]).sum(1, dtype=np.uint64); h ^= h >> np.uint64(29); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= h >> np.uint64(32)
    return np.where(h & np.uint64(1), 1, -1).astype(np.int8)


def guide_two(p1, seed=1):
    r = np.random.default_rng(seed)
    f2 = np.asarray(p1.f1) + 0.3 * np.std(np.real(np.asarray(p1.f1))) * (r.standard_normal((64, 64)) + 1j * r.standard_normal((64, 64)))
    x0 = np.ones((1, 64)); x0[0, 32:] = -1; np.random.default_rng(5).shuffle(x0[0])
    pP2 = PL.PP2(f2, L=8, proj=p1.proj, rbm='inner', b=p1.b, W=p1.W)
    lam = 0.5 * np.exp(np.real(complex(PL.logB(p1, x0)[0]) - complex(PL.logB(pP2, x0)[0])))
    return p1.with_p2(f2, lam), float(np.real(lam))


def setup(system):
    """-> (params, X256 walkers (float64), sign_fn, EP, f0, info)"""
    if system in ('a8prod', 'a8two', 'cnn'):
        X = np.load(WALK8).astype(np.float64); p1 = PL.PP2.from_rbmpp_npz(PROD); lam = None
        p = p1
        if system != 'a8prod': p, lam = guide_two(p1)
        sf = hash_sign; cb = None
        if system == 'cnn':                                           # random-weight CNN: centre the logit (median over x and random swaps of 64 walkers) so that the signs are balanced
            sf0 = PL.make_sign_fn('L', seed=0); net = sf0.net; rg = np.random.default_rng(0); Ys = X[:64].astype(np.int8).copy()
            for r_ in Ys:
                i_, j_ = rg.choice(64, 2, replace=False); r_[i_], r_[j_] = r_[j_], r_[i_]
            cf = np.concatenate([X[:64].astype(np.int8), Ys]); lgt = np.asarray(jax.jit(lambda x: net.jax_logit(net.params, x.astype(jnp.float32)))(jnp.asarray(cf)))
            cb = float(np.median(lgt)); sf = PL.make_sign_fn('L', seed=0, net=net, bias=cb)
        info = dict(params=PROD, params_sha256=sha(PROD), p0_run=P0RUN, p0_run_metadata_sha256=sha(f'{P0RUN}/run_metadata.txt'),
                    p0_result_json_sha256=sha(f'{P0RUN}/result.json'), walkers=WALK8, walkers_sha256=sha(WALK8), proj=p1.proj, lam_two=lam,
                    sign='sign8 L CNN seed 0, bias %.6f' % cb if system == 'cnn' else 'hash_sign', cnn_bias=cb)
        return p, X, sf, -0.4987565 * 64, 3.0, info
    R = np.load(REPS)['reps']; s1 = np.load(SG).astype(np.float64); rng = np.random.default_rng(66)
    rows = np.sort(rng.choice(len(R), 256, replace=False)); SH = np.arange(36, dtype=np.uint64)
    X = (((R[rows][:, None] >> SH[None]) & np.uint64(1)).astype(np.float64) * 2 - 1)
    import pp2_lemma_test as T6                                       # canon_idx (rep lookup)
    sign_fn = lambda cfg: s1[T6.canon_idx(cfg)]
    EP = json.load(open('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/readout/readout.json'))['tables']['pp2_s1']['E_FN']['E']
    return PL.PP2.from_cx_theta(np.load(THETA5)), X, sign_fn, EP, 2.0, dict(theta5=THETA5, theta5_sha256=sha(THETA5), s_G=SG, s_G_sha256=sha(SG), reps=REPS, sign='s_G table')


def stage_run(system, variant, M):
    p, X256, sf, EP, f0, info = setup(system); lat = p.lat
    dmax = PL._dmax_of(lat, X256); X = X256[:M]
    path = 'old' if variant == 'old' else ('pack' if variant == 'pack' else 'dedup'); br = variant[2:] if variant.startswith('b_') else None
    kw = dict(f0=f0, mode='fix', walker_chunk=128, sign_chunk=4096, dmax=dmax)
    PL.PROGRESS = lambda i, n: log(f'  {system} {variant} progress {i}/{n} walkers')
    PL.STATS.clear(); t = time.time()
    if variant == 'old': o = PL.aug_ratios(p, X, sf, EP, 4.0, **kw)
    else: o = PL.aug_ratios(p, X, sf, EP, 4.0, eval_path=path, _broken_save=br, **kw)
    jax.block_until_ready(o['ratio']); sec = time.time() - t
    d = (X[:, lat.BI] != X[:, lat.BJ]).sum(1)
    arrs = {k: np.asarray(o[k]) for k in KEYS}; arrs.update(am=np.asarray(o['am']), bm=np.asarray(o['bm']), a=np.asarray(o['a']), b=np.asarray(o['b']))
    for k in ('sx', 'sy', 'sz'): arrs[k] = arrs[k].astype(np.int8)
    tag = f'{system}_{variant}_M{M}'
    np.savez(f'{OUT}/runs/{tag}.npz', **arrs)
    meta = dict(system=system, variant=variant, M=M, dmax=dmax, sec=sec, stats=dict(PL.STATS), d_anti=dict(mean=float(d.mean()), min=int(d.min()), max=int(d.max())),
                code_sha256=sha(os.path.join(HERE, 'pp2_lemma.py')), test_sha256=sha(__file__), host=os.uname().nodename, jax=jax.__version__,
                threads=os.environ.get('OMP_NUM_THREADS', '-'), EP=EP, f0=f0, c=4.0, **info)
    json.dump(meta, open(f'{OUT}/runs/{tag}.json', 'w'), indent=1, default=float); log(f'RUN {tag}: {sec:.0f}s stats {dict(PL.STATS)}')


def maxdiff(ref, o, M):
    out = {}; am = ref['am'][:M]; bm = ref['bm'][:M]
    for k in KEYS:
        a = ref[k][:M]; b = o[k]; fa = np.isfinite(a.astype(float) if a.dtype != complex else a); fb = np.isfinite(b.astype(float) if b.dtype != complex else b)
        m = fa & fb
        dd = np.abs(a[m].astype(complex) - b[m].astype(complex)) if m.any() else np.zeros(1)
        out[k] = float(dd.max()); out[k + '_rel'] = float((dd / np.maximum(np.abs(a[m].astype(complex)), 1.0)).max()) if m.any() else 0.0   # scale-fair: |d| / max(|ref|, 1)
        if (fa != fb).any(): out[k] = float('inf'); out[k + '_rel'] = float('inf')
    out['sy_valid_mismatch'] = int((ref['sy'][:M] != o['sy'])[am].sum()); out['sz_valid_mismatch'] = int((ref['sz'][:M] != o['sz'])[bm].sum())
    out['sx_mismatch'] = int((ref['sx'][:M] != o['sx']).sum())
    return out


def load(tag):
    f = f'{OUT}/runs/{tag}.npz'
    return (dict(np.load(f)), json.load(open(f'{OUT}/runs/{tag}.json'))) if os.path.exists(f) else (None, None)


def stage_compare():
    for system, name in (('a8prod', 'a8'), ('a8two', 'a8'), ('a6', 'a6'), ('cnn', 'cnn')):
        olds = sorted(glob.glob(f'{OUT}/runs/{system}_old_M*.json'), key=lambda f: -json.load(open(f))['M'])
        if not olds: continue
        ref, rm = load(os.path.basename(olds[0])[:-5]); Mref = rm['M']; res = dict(system=system, M_old=Mref, old=rm, new={}, broken={})
        for v in ('pack', 'dedup', 'b_pair', 'b_trunc', 'b_cnnkey'):
            for f in sorted(glob.glob(f'{OUT}/runs/{system}_{v}_M*.json')):
                o, om = load(os.path.basename(f)[:-5]); M = om['M']
                if M > Mref: continue
                md = maxdiff(ref, o, M); worst = max(md[k] for k in KEYS)
                rec = dict(M=M, maxabs=md, max_all=worst, max_rel=max(md[k + '_rel'] for k in KEYS), sec=om['sec'], stats=om['stats'], host=om['host'], code_sha256=om['code_sha256'])
                (res['broken'] if v.startswith('b_') else res['new'])[f'{v}_M{M}'] = rec
                log(f'COMPARE {system} {v} M={M}: max |new - old| {worst:.3e}, sign mismatches (valid) {md["sx_mismatch"] + md["sy_valid_mismatch"] + md["sz_valid_mismatch"]}')
        res['PASS_scaled_1e-13'] = bool(res['new']) and all(max(r['maxabs'][k + '_rel'] for k in KEYS) <= 1e-13 and r['maxabs']['sx_mismatch'] + r['maxabs']['sy_valid_mismatch'] + r['maxabs']['sz_valid_mismatch'] == 0 for r in res['new'].values())
        res['PASS'] = bool(res['new']) and all(r['max_all'] <= TOL and r['maxabs']['sx_mismatch'] + r['maxabs']['sy_valid_mismatch'] + r['maxabs']['sz_valid_mismatch'] == 0
                                              for r in res['new'].values()) and any(k.startswith('dedup') for k in res['new'])
        res['BROKEN_FAILS'] = bool(res['broken']) and all(r['max_all'] > 1e-6 or r['maxabs']['sz_valid_mismatch'] + r['maxabs']['sy_valid_mismatch'] > 0 for r in res['broken'].values())
        res['tol'] = TOL; json.dump(res, open(f'{OUT}/check_{system}.json', 'w'), indent=1, default=float)
        log(f'CHECK {system}: PASS {res["PASS"]}  BROKEN_FAILS {res["BROKEN_FAILS"]} (broken max: { {k: "%.2e" % v["max_all"] for k, v in res["broken"].items()} })')


if __name__ == '__main__':
    if STAGE == 'run': stage_run(sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif STAGE == 'compare': stage_compare()
