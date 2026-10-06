"""Lattice fixed-node GFMC referee for an it2 guide spec (stored sign [+ one hop], amplitude a_g) or the ViT guide.

Algorithm identical to experiments/learned_loop_6x6/ll6_fn.py (= fn_bound_check_6x6/fn_run.py): propagator
1 - tau (H_FN - Eref), tau = min(tau_max, 0.8 / max(d_FN - Eref)), Eref = mean local energy of the population,
one move per walker per step, systematic resampling every step, mixed estimator averaged over beta >= burn,
initial walkers: M distinct states of the 256-state pool energy_krylov_vs_vit_6x6_indep.npz.
Guide 'vit': a = |psi_ViT|, s = binarised ViT phase (global phase fixed on the pool) -> common-random-number
pairing with the same seeds.
Usage: python it2_fn.py SPEC.json|vit TAG M beta burn tau_max seed1 [seed2 ...]
"""
import sys, json, time
import numpy as np
import ll6_core as C
import it2_core as I

spec_p, tag = sys.argv[1], sys.argv[2]
M, BETA, BURN, TAUMAX = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
seeds = [int(s) for s in sys.argv[7:]]
clk = I.GpuClock()
net = C.Net(I.CKPT, dtype='float32')
pool = np.load('energy_krylov_vs_vit_6x6_indep.npz')['states'].astype(np.uint64).reshape(-1)
if spec_p == 'vit':
    phi = I.vit_phase(net, pool); guide = I.VitGuide(net, phi); spec = dict(name='vit', phi=phi)
else:
    spec, bd = I.load_spec(spec_p); guide = I.HGuide(spec, net, base_dir=bd)
I.log('FN_SETUP', spec, 'M', M, 'beta', BETA, 'burn', BURN, 'seeds', seeds)
localc = {}
T0 = time.time()


def ensure_local(ss):
    miss = np.asarray([int(x) for x in np.unique(np.asarray(ss, np.uint64)) if int(x) not in localc], np.uint64)
    if len(miss) == 0: return
    d = guide.local(miss)
    own = d['own']; al = d['allowed']
    starts = np.r_[0, np.cumsum(np.bincount(own, minlength=len(miss)))]
    for i, x in enumerate(miss):
        a, b = starts[i], starts[i + 1]
        m = al[a:b]
        localc[int(x)] = (float(d['dfn'][i]), float(d['eh'][i]), d['child'][a:b][m].astype(np.uint64), d['rate'][a:b][m])


def systematic(w, rng, M):
    c = np.cumsum(w); c[-1] = 1
    return np.searchsorted(c, rng.random() / M + np.arange(M) / M, 'right')


def run(M, beta_target, burn_beta, tau_max, seed):
    rg = np.random.default_rng(seed)
    walkers = pool[rg.choice(len(pool), M, replace=M > len(pool))].copy()
    beta = 0.; Es = []; curve = []; uniq_hist = []
    while beta < beta_target:
        ensure_local(walkers); dat = [localc[int(x)] for x in walkers]
        diag = np.array([z[0] for z in dat]); el = np.array([z[1] for z in dat]); Eref = float(el.mean())
        mx = max(0., float(np.max(diag - Eref))); tau = min(tau_max, 0.8 / mx if mx > 0 else tau_max)
        curve.append((beta, tau, Eref)); uniq_hist.append(len(np.unique(walkers)))
        nxt = np.empty(M, np.uint64); bw = np.empty(M)
        for k, (x, (d, e, ys, rate)) in enumerate(zip(walkers, dat)):
            stay = 1 - tau * (d - Eref); ws = tau * rate; tot = stay + ws.sum()
            if stay < 0 or tot <= 0: raise RuntimeError(("bad", stay, tot, tau, d, Eref))
            u = rg.random() * tot
            if u < stay: y = x
            else:
                j = np.searchsorted(np.cumsum(ws), u - stay, 'right'); y = int(ys[min(j, len(ys) - 1)])
            nxt[k] = y; bw[k] = tot
        bw /= bw.sum(); walkers = nxt[systematic(bw, rg, M)]
        beta += tau
        if beta >= burn_beta:
            ensure_local(walkers); Es.append(float(np.mean([localc[int(x)][1] for x in walkers])))
    return np.asarray(Es), np.asarray(curve), uniq_hist


rows = []
for seed in seeds:
    t0 = time.time()
    Es, curve, uh = run(M, BETA, BURN, TAUMAX, seed)
    row = dict(seed=seed, Emean=float(Es.mean()), Emean_site=float(Es.mean() / 36), nE=len(Es),
               E_beta0=float(curve[0, 2]), mean_unique=float(np.mean(uh)), sec=time.time() - t0, cache_states=len(localc))
    rows.append(row)
    I.log('FN_REP', json.dumps(row))
    em = np.array([r['Emean_site'] for r in rows])
    summ = dict(spec=spec, spec_path=spec_p, tag=tag, M=M, beta_target=BETA, burn_beta=BURN, tau_max=TAUMAX, reps=rows,
                mean_site=float(em.mean()), SE_site=float(em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 1 else None,
                gpu_h=clk.hours())
    json.dump(summ, open(f'fn_{tag}.json', 'w'), indent=1)
I.log('FN_SUMMARY', tag, summ['mean_site'], summ['SE_site'], summ['gpu_h'])
