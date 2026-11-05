"""bench_gpu.py: the H100 timing benchmark of the 8x8 guide (job A of the Pfaffian-angle H100 benchmark; PLAN_pfaffian.md "Lead decision after Astra").
One process, f64.  Stage 1: walker_chunk scan (aug_ratios c = 4 at M = 256).  Stage 2: static component timing on X0 draws of |B|^2 at M = 256 and 1024:
plain (one_swap_ratios), lemma (two_swap_ratios), sign (PhaseSign.endpoints), full aug_ratios (c = 4), c0 provider, DirectSG rows, and (extra) the same with a
two-PP amplitude.  Stage 3: live GFMC runs (gfmc_aug8.run_chain) of a few blocks per arm and M, with the per-block timing split, compile events, peak memory.
usage: python bench_gpu.py OUTDIR INPUTDIR [--quick]
"""
import os, sys, json, time, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
import gfmc_aug8 as G8, pp2_lemma as PL

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('inp'); ap.add_argument('--quick', action='store_true')
ap.add_argument('--sign', default=os.environ.get('B_SIGN', '/scratch/a/A.Otaifi/ansatz_runs/sign_scale/r8/ppw_s1/theta.npy'))
ap.add_argument('--wscan', default='64,128,256,512'); ap.add_argument('--gfmc_maxsec', type=float, default=150.0)
ap.add_argument('--ms', default='256,1024'); ap.add_argument('--m1', type=int, default=256, help='M of the walker_chunk scan'); ap.add_argument('--budget_sec', type=float, default=1e18, help='skip later stages once exceeded')
ap.add_argument('--live', default='aug:4:256:5,aug:4:1024:4,c0:0:256:5,c0:0:1024:4,sg:0:256:4,sg:0:1024:3', help='arm:c:M:blocks,...')
A = ap.parse_args(); OUT = A.out; os.makedirs(OUT, exist_ok=True); T0 = time.time()
R = dict(host=os.uname().nodename, device=str(jax.devices()[0]), jax=jax.__version__, slurm_job=os.environ.get('SLURM_JOB_ID'), quick=A.quick,
         code_sha256={f: G8.sha(os.path.join(HERE, f)) for f in ('gfmc_aug8.py', 'gfmc_aug.py', 'pp2_lemma.py', 'bench_gpu.py')}, params_sha256=G8.sha(G8.PROD8),
         sign=A.sign, sign_sha256=G8.sha(A.sign), EP=G8.EP8, f0=G8.F0_8)

if 'cuda' not in R['device'].lower() and not os.environ.get('B_ALLOW_CPU'): sys.exit('bench_gpu: no CUDA device (%s); set B_ALLOW_CPU=1 for a CPU preflight' % R['device'])


def log(m):
    s = f'[{time.time() - T0:7.1f}s] {m}'; print(s, flush=True); open(f'{OUT}/bench.log', 'a').write(s + '\n')


def save(): json.dump(R, open(f'{OUT}/bench_gpu.json', 'w'), indent=1, default=float)


def blk(o):
    jax.block_until_ready(o); return o


def best(fn, reps=3):
    """warm-up call (compiles), then best-of-reps seconds."""
    blk(fn()); ts = []
    for _ in range(reps):
        t = time.perf_counter(); blk(fn()); ts.append(time.perf_counter() - t)
    return min(ts), ts


G8.patch_timers()
p = PL.PP2.from_rbmpp_npz(G8.PROD8); lat = p.lat; ps = PL.PhaseSign.from_theta(np.load(A.sign), 8, guard=1e-8)
X = np.load(f'{A.inp}/X0_a_M1024.npy').astype(np.float64); EP = G8.EP8; f0 = G8.F0_8
dmax = int(min(lat.NB, 8 * int(np.ceil((int(np.max(np.sum(X[:, lat.BI] != X[:, lat.BJ], 1))) + 16) / 8))))
dd = (X[:, lat.BI] != X[:, lat.BJ]).sum(1); R['d_anti'] = dict(mean=float(dd.mean()), max=int(dd.max()), min=int(dd.min())); R['dmax'] = dmax
log(f'start {R["device"]} d mean {dd.mean():.1f} max {dd.max()} dmax {dmax} peak {G8.peak_gib():.2f} GiB'); save()

# ---------------------------------------------------------------- stage 1: walker_chunk scan
R['wscan'] = {}; M1 = A.m1
for wc in [int(w) for w in A.wscan.split(',')]:
    try:
        c0n = G8.PROF.n_compiles(); cs0 = G8.PROF.sec_compile()
        t = time.perf_counter(); blk(PL.aug_ratios(p, X[:M1], ps, EP, 4.0, f0, 'fix', 'f64', dmax, walker_chunk=wc, eval_path='dedup')); t_first = time.perf_counter() - t
        ts = []
        for _ in range(2):
            t = time.perf_counter(); blk(PL.aug_ratios(p, X[:M1], ps, EP, 4.0, f0, 'fix', 'f64', dmax, walker_chunk=wc, eval_path='dedup')); ts.append(time.perf_counter() - t)
        R['wscan'][str(wc)] = dict(first_call_s=t_first, best_s=min(ts), s_per_walker_step=min(ts) / M1, peak_GiB=G8.peak_gib(), compiles=G8.PROF.n_compiles() - c0n, sec_compile=G8.PROF.sec_compile() - cs0)
        log(f'wscan wc {wc}: {R["wscan"][str(wc)]}')
    except Exception as e:
        R['wscan'][str(wc)] = dict(error=str(e)[:300]); log(f'wscan wc {wc} FAILED {str(e)[:200]}'); break
    save()
ok = {k: v for k, v in R['wscan'].items() if 'best_s' in v}; WC = int(min(ok, key=lambda k: ok[k]['best_s'])); R['wchunk_chosen'] = WC; log(f'chosen walker_chunk {WC}')

# ---------------------------------------------------------------- stage 2: static components
R['static'] = {}
for M in [int(m) for m in A.ms.split(',')]:
    if time.time() - T0 > A.budget_sec: log('budget exceeded, skipping static M=%d' % M); continue
    Xm = X[:M]; res = {}
    tp, _ = best(lambda: PL.one_swap_ratios(p, Xm, 'f64', walker_chunk=WC)); res['plain'] = tp
    d0 = {}
    def lem():
        d0['d'] = PL.two_swap_ratios(p, Xm, 'f64', dmax, walker_chunk=WC, eval_path='dedup'); return d0['d']
    PL.STATS.clear(); tl, _ = best(lem); res['lemma'] = tl; res['lemma_stats_last'] = {k: v for k, v in PL.STATS.items()}
    ts_, _ = best(lambda: ps.endpoints(Xm, d0['d'], True, 'f64', 'dedup', WC)); res['sign'] = ts_
    PL.STATS.clear(); ta, _ = best(lambda: PL.aug_ratios(p, Xm, ps, EP, 4.0, f0, 'fix', 'f64', dmax, walker_chunk=WC, eval_path='dedup')); res['aug_c4'] = ta
    res['aug_stats_last'] = {k: v for k, v in PL.STATS.items()}
    res['fJ_rest'] = ta - tl - ts_
    g0 = G8.AugGuide8(p, ps, EP, 0.0, f0, mode='old', eval_path='dedup', wchunk=WC, chunk=M, dmax0=dmax, fast0=True)
    tc, _ = best(lambda: g0._aug0(Xm, min(WC, M))['ratio']); res['c0_fast'] = tc
    g0b = G8.AugGuide8(p, ps, EP, 0.0, f0, mode='old', eval_path='dedup', wchunk=WC, chunk=M, dmax0=dmax)
    tcb, _ = best(lambda: g0b._aug(Xm)['ratio'], reps=2); res['c0_via_aug_ratios_mode_old'] = tcb
    al, E, w = G8.m1_consts()
    for sgc in (256, 1024):
        try:
            sg = G8.DirectSG(p, al, E, w, chunk=sgc)
            tsg, _ = best(lambda: sg.rows(Xm.astype(np.int8)), reps=2); res[f'direct_sG_chunk{sgc}'] = tsg
        except Exception as e:
            res[f'direct_sG_chunk{sgc}'] = f'ERROR {str(e)[:200]}'; log(f'direct sG chunk {sgc} failed {str(e)[:150]}')
    res['fallback'] = dict(checked=int(ps.n_checked), fallback=int(ps.n_fallback), rate=float(ps.fallback_rate))
    res['per_step'] = {k: v / M for k, v in res.items() if isinstance(v, float)}; res['peak_GiB'] = G8.peak_gib()
    R['static'][f'M{M}'] = res; log(f'static M={M}: ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("lemma_stats_last", "aug_stats_last", "per_step")}, default=float)); save()

# ---------------------------------------------------------------- stage 2b (extra): two-PP amplitude timing, c = 4
try:
    class _a: pass
    a2 = _a(); a2.sys = 8; a2.arm = 'aug'; a2.c = 4.0; a2.eval_path = 'dedup'; a2.wchunk = WC; a2.M = int(A.ms.split(',')[0]); a2.dtau = 0.1; a2.seed = 1; a2.mode = 'fix'; a2.two_pp = True
    a2.EP = None; a2.f0 = None; a2.X0 = f'{A.inp}/X0_a_M{int(A.ms.split(",")[0])}.npy'; a2.sign = f'phase:{A.sign}'; a2.guard = 1e-8; a2.dmargin = 16; a2.sgchunk = 256; a2.theta6 = ''
    g2, X2, info2 = G8.build(a2); R['two_pp'] = {}
    for M in [int(m) for m in A.ms.split(',')]:
        if time.time() - T0 > A.budget_sec: break
        Xm = X[:M]; g2.chunk = int(np.ceil(M / WC) * WC)
        t2, _ = best(lambda: g2._aug(Xm), reps=2); R['two_pp'][f'M{M}'] = dict(aug_c4_s=t2, s_per_walker_step=t2 / M, peak_GiB=G8.peak_gib()); log(f'two-PP M={M}: {R["two_pp"][f"M{M}"]}')
    save()
except Exception as e:
    R['two_pp_error'] = str(e)[:300]; log(f'two-PP failed {str(e)[:200]}')

# ---------------------------------------------------------------- stage 3: live GFMC runs
R['gfmc'] = {}
runs = [(q.split(':')[0], float(q.split(':')[1]), int(q.split(':')[2]), int(q.split(':')[3])) for q in A.live.split(',') if q]
for arm, c, M, nb in runs:
    if time.time() - T0 > A.budget_sec: log('budget exceeded, skipping live runs from %s' % arm); break
    name = f'{arm}_c{int(c)}_M{M}'; ns = argparse.Namespace(sys=8, arm=arm, c=c, mode='fix', M=M, dtau=0.1, blocks=nb, burn=0, seed=1, X0=f'{A.inp}/X0_a_M{M}.npy', maxsec=A.gfmc_maxsec,
                  eval_path='dedup', wchunk=WC, sgchunk=256, dmargin=16, sign=f'phase:{A.sign}', guard=1e-8, EP=None, f0=None, two_pp=False, theta6='', out=f'{OUT}/live_{name}')
    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {name} {msg}'; print(line, flush=True); open(f'{OUT}/progress.txt', 'a').write(line + '\n')
    try:
        g, X0, info = G8.build(ns); info['code_sha256'] = R['code_sha256']
        r = G8.run_chain(g, X0, ns.out, ns, info, P)
        R['gfmc'][name] = dict(status=r['status'], nblocks=r['nblocks'], init=r['init'], prof_cum=r['prof_cum'])
    except Exception as e:
        R['gfmc'][name] = dict(error=str(e)[:400]); log(f'live {name} FAILED {str(e)[:300]}')
    save()
R['total_sec'] = time.time() - T0; R['peak_GiB_final'] = G8.peak_gib(); R['events'] = {k: v for k, v in G8.PROF.ev.items()}; save()
P('BENCH_GPU_DONE')
log('BENCH_GPU_DONE')
