"""pp2_lemma_savings_bench.py: CPU timing of the pp2_lemma savings (piece (3), b): seconds per walker step of the 8x8 augmented two-PP guide
on the OLD padded path, the PACK path (less padding) and the DEDUP path (+ deduplicated CNN calls), same machine, same threads, f64.
usage: python pp2_lemma_savings_bench.py OUTDIR M [M ...]        (JAX_PLATFORMS=cpu; env SB_REPS=2 SB_REAL_CNN_M=64 SB_PATHS=old,pack,dedup)
Guide and walkers as pp2_lemma_savings_test.py a8two (rb8_prod P_1 + RBM, P_2 = P_1 + noise, lam; 256 walkers from the Metropolis file); c = 4, f0 = 3.
Per M and path (best of SB_REPS interleaved repetitions, after a warm-up on one walker chunk):
  t_plain : one_swap_ratios (the plain P step done by the lemma code; NOT the persistent fast update of the production run)
  t_lemma : two_swap_ratios(eval_path)
  t_rest  : everything of aug_ratios except the lemma (fJ_fixed on the lemma output: hash-sign calls / dedup bookkeeping, configs, f_J arithmetic; CNN excluded)
  n_cnn   : number of CNN calls the path makes (old: x + all padded y + all padded z; dedup: distinct configurations), also the valid count
  CNN     : per-call cost of the L-size sign8 CNN (64 ch x 6 layers, f32, chunk 4096) from a timed sample of SB_CNN_SAMPLE configurations
            (the same configurations the guide sees: x, y_a, z_ab of the walkers); t_cnn = n_cnn x per-call (PROJECTED, labelled);
            for M = SB_REAL_CNN_M the full aug_ratios with the real CNN is also timed (old and dedup) as a check of the projection.
Split reported per walker step: lemma / CNN / plain / bookkeeping+rest.  GPU is not used.
"""
import os, sys, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import pp2_lemma as PL
import pp2_lemma_savings_test as ST          # setup(), hash_sign (its argv: OUTDIR, stage; we pass stage 'bench' which it ignores)

OUT = sys.argv[1]; MS = [int(a) for a in sys.argv[2:]]
REPS = int(os.environ.get('SB_REPS', '2')); PATHS = os.environ.get('SB_PATHS', 'old,pack,dedup').split(','); RCM = int(os.environ.get('SB_REAL_CNN_M', '64'))
NS = int(os.environ.get('SB_CNN_SAMPLE', '16384')); CH = 4096; T0 = time.time()
os.makedirs(OUT, exist_ok=True)


def log(m):
    s = f'[{time.time() - T0:7.1f}s] {m}'; print(s, flush=True); open(f'{OUT}/bench.log', 'a').write(s + '\n')


def blk(o):
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, 'block_until_ready') else a, o); return o


def lscpu():
    try: return {l.split(':')[0].strip(): l.split(':', 1)[1].strip() for l in os.popen('lscpu').read().splitlines() if ':' in l and l.split(':')[0].strip() in ('Model name', 'CPU(s)', 'Thread(s) per core')}
    except Exception: return {}


def main():
    p, X256, hsf, EP, f0, info = ST.setup('a8two'); lat = p.lat; dmax = PL._dmax_of(lat, X256); C = 4.0
    cnn = PL.make_sign_fn('L', seed=0)
    R = dict(host=os.uname().nodename, jax=jax.__version__, cpu=lscpu(), affinity=sorted(os.sched_getaffinity(0)).__len__(), omp=os.environ.get('OMP_NUM_THREADS'),
             slurm_job=os.environ.get('SLURM_JOB_ID'), code_sha256={f: ST.sha(os.path.join(HERE, f)) for f in ('pp2_lemma.py', 'pp2_lemma_savings_bench.py', 'pp2_lemma_savings_test.py', 'sign8.py')},
             guide=info, dmax=dmax, c=C, f0=f0, EP=EP, reps=REPS, paths=PATHS, cnn=dict(size='L', npar=cnn.net.npar, chunk=CH), load_start=os.getloadavg(), runs={})
    # ---- CNN per-call cost on real guide configurations (x, y, z of walkers 0..)
    Xs = X256[:64]; d = PL.two_swap_ratios(p, Xs, 'f64', dmax, eval_path='pack', walker_chunk=64)
    Y, Z = blk(PL._configs(jnp.asarray(Xs, jnp.float32), d)); bm = np.asarray(d['bm']); Zs = np.asarray(Z)[bm][:NS].astype(np.int8)
    cnn(jnp.asarray(Zs[:CH])); jax.block_until_ready(cnn(jnp.asarray(Zs[:CH])))
    ts = []
    for rep in range(3):
        t = time.perf_counter()
        for i in range(0, NS - CH + 1, CH): jax.block_until_ready(cnn(jnp.asarray(Zs[i:i + CH])))
        ts.append((time.perf_counter() - t) / ((NS // CH) * CH))
    R['cnn_s_per_call'] = min(ts); R['cnn_s_per_call_all'] = ts; log(f'CNN L: {min(ts) * 1e3:.3f} ms per call (chunk {CH}), reps {[round(x * 1e3, 3) for x in ts]}')
    del Y, Z, d
    json.dump(R, open(f'{OUT}/bench.json', 'w'), indent=1, default=float)
    kw = dict(f0=f0, mode='fix', sign_chunk=CH, dmax=dmax, walker_chunk=128)
    for M in MS:
        X = X256[:M]; res = {p_: dict(t_lemma=[], t_rest=[]) for p_ in PATHS}; tp = []
        blk(PL.one_swap_ratios(p, X[:128], 'f64'))                                          # warm-up (compile)
        kwj = dict(sign_chunk=CH, dmax=dmax, walker_chunk=128)
        for p_ in PATHS:
            d0 = blk(PL.two_swap_ratios(p, X[:128], 'f64', dmax, eval_path=p_, walker_chunk=128)); blk(PL.fJ_fixed(p, X[:128], hsf, EP, C, f0, 'fix', True, 'f64', _two=d0, eval_path=p_, sign_chunk=CH))
        log(f'M={M}: warm-up done')
        for rep in range(REPS):
            t = time.perf_counter(); blk(PL.one_swap_ratios(p, X, 'f64')); tp.append(time.perf_counter() - t)
            for p_ in PATHS:
                PL.STATS.clear(); t = time.perf_counter(); d = blk(PL.two_swap_ratios(p, X, 'f64', dmax, eval_path=p_, walker_chunk=128)); res[p_]['t_lemma'].append(time.perf_counter() - t)
                res[p_]['stats_lemma'] = dict(PL.STATS)
                if rep == 0:
                    am = np.asarray(d['am']); bmm = np.asarray(d['bm']); res[p_]['n_valid_cnn'] = int(M + am.sum() + bmm.sum()); res[p_]['n_padded_cnn'] = int(M * (1 + am.shape[1] + bmm.shape[1] * bmm.shape[2]))
                # everything of aug_ratios except the lemma: sign calls (cheap hash sign) / dedup bookkeeping, configs, f_J arithmetic, on the lemma output d
                PL.STATS.clear(); t = time.perf_counter(); o = blk(PL.fJ_fixed(p, X, hsf, EP, C, f0, 'fix', True, 'f64', _two=d, eval_path=p_, sign_chunk=CH)); res[p_]['t_rest'].append(time.perf_counter() - t)
                res[p_]['stats_aug'] = dict(PL.STATS); del o, d
                log(f'M={M} rep {rep} {p_}: lemma {res[p_]["t_lemma"][-1]:.1f}s rest (hash sign, bookkeeping, f_J) {res[p_]["t_rest"][-1]:.1f}s  load {os.getloadavg()[0]:.1f}')
        out = dict(M=M, t_plain=min(tp), t_plain_all=tp, paths={})
        for p_ in PATHS:
            r = res[p_]; st = r['stats_aug']
            n_cnn = st['cnn_calls_dedup'] if p_ == 'dedup' else r['n_padded_cnn']
            q = dict(t_lemma=min(r['t_lemma']), t_lemma_all=r['t_lemma'], t_rest=min(r['t_rest']), t_rest_all=r['t_rest'], n_cnn=int(n_cnn), n_cnn_valid=r['n_valid_cnn'],
                     n_cnn_padded_old=r['n_padded_cnn'], stats_lemma=r['stats_lemma'], stats_aug=st)
            q['t_aug_hash'] = q['t_lemma'] + q['t_rest']; q['t_cnn_projected'] = n_cnn * R['cnn_s_per_call']
            q['t_step_projected'] = q['t_aug_hash'] + q['t_cnn_projected']
            for k in ('t_lemma', 't_cnn_projected', 't_rest', 't_aug_hash', 't_step_projected'): q['s_per_walker_step_' + k[2:]] = q[k] / M
            out['paths'][p_] = q
        out['s_per_walker_step_plain'] = out['t_plain'] / M
        if 'old' in out['paths']:
            o_ = out['paths']['old']
            for p_ in PATHS:
                q = out['paths'][p_]; q['speedup_lemma_vs_old'] = o_['t_lemma'] / q['t_lemma']; q['speedup_cnn_calls_vs_old'] = o_['n_cnn'] / q['n_cnn']
                q['speedup_step_vs_old'] = o_['t_step_projected'] / q['t_step_projected']; q['speedup_step_excl_cnn_vs_old'] = o_['t_aug_hash'] / q['t_aug_hash']
        R['runs'][f'M{M}'] = out; json.dump(R, open(f'{OUT}/bench.json', 'w'), indent=1, default=float)
        log(f'M={M} summary: ' + '; '.join(f'{p_}: lemma {out["paths"][p_]["s_per_walker_step_lemma"]:.3f} s/ws, aug(no CNN) {out["paths"][p_]["s_per_walker_step_aug_hash"]:.3f}, '
                                          f'CNN calls/ws {out["paths"][p_]["n_cnn"] / M:.0f} -> {out["paths"][p_]["s_per_walker_step_cnn_projected"]:.3f} s/ws' for p_ in PATHS) + f'; plain {out["s_per_walker_step_plain"]:.4f} s/ws')
        if M == RCM:                                                                           # real CNN, full aug_ratios, as a check of the projection
            for p_ in ('old', 'dedup'):
                if p_ not in PATHS: continue
                PL.STATS.clear(); t = time.perf_counter(); o = blk(PL.aug_ratios(p, X, cnn, EP, C, eval_path=p_, **kw)); out['paths'][p_]['t_aug_real_cnn'] = time.perf_counter() - t
                out['paths'][p_]['s_per_walker_step_aug_real_cnn'] = out['paths'][p_]['t_aug_real_cnn'] / M; del o
                log(f'M={M} {p_} REAL CNN aug_ratios {out["paths"][p_]["t_aug_real_cnn"]:.1f}s = {out["paths"][p_]["s_per_walker_step_aug_real_cnn"]:.3f} s/ws (projected {out["paths"][p_]["s_per_walker_step_step_projected"]:.3f})')
            R['runs'][f'M{M}'] = out; json.dump(R, open(f'{OUT}/bench.json', 'w'), indent=1, default=float)
    R['load_end'] = os.getloadavg(); R['done'] = True; json.dump(R, open(f'{OUT}/bench.json', 'w'), indent=1, default=float); log('BENCH_DONE')


if __name__ == '__main__':
    main()
