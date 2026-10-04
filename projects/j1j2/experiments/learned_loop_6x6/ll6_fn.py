"""Lattice fixed-node GFMC referee for a guide state JSON (see ll6_iter.py).

Algorithm copied from experiments/fn_bound_check_6x6/fn_run.py (= fn_guides_compare.run):
discrete-time propagator 1 - tau (H_FN - Eref), tau = min(tau_max, 0.8 / max(d_FN - Eref)), Eref = mean
local energy of the current population, one move per walker per step (stay or one allowed hop),
systematic resampling every step with weights 'tot', mixed estimator = mean local energy of the
population, averaged over beta >= burn_beta; initial walkers: M distinct states of the 256-state pool
energy_krylov_vs_vit_6x6_indep.npz.  Only the guide's local data come from ll6_core (sign chain of
any depth, amplitudes cached).
Usage: python ll6_fn.py STATE.json TAG M beta burn tau_max seed1 [seed2 ...]
"""
import sys, json, time
import numpy as np
import ll6_core as C
import jax, jax.numpy as jnp

state, tag = sys.argv[1], sys.argv[2]
M, BETA, BURN, TAUMAX = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
seeds = [int(s) for s in sys.argv[7:]]
DTYPE = 'float32'
st = json.load(open(state))
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype=DTYPE)
amp_objs = {}
def amp_of(p):
    if p not in amp_objs:
        amp_objs[p] = C.Amp(net, net.flat0 if p == 'base' else jnp.asarray(np.load(p), net.DT), cache=True, name=p)
    return amp_objs[p]
guide = C.Guide([amp_of(p) for p in st['chain_params']], st['Ts'], amp_of(st['amp_g']))
C.log('FN_SETUP', state, 'K', guide.K, 'Ts', guide.Ts, 'M', M, 'beta', BETA, 'burn', BURN, 'seeds', seeds)
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
    pool = np.load('energy_krylov_vs_vit_6x6_indep.npz')['states'].astype(np.uint64).reshape(-1)
    walkers = pool[rg.choice(len(pool), M, replace=False)].copy()
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
        if len(curve) % 10 == 1:
            C.log('PROG', seed, len(curve), 'beta', round(beta, 4), 'E', Eref, 'cache', len(localc),
                  'amp_new', {k[-30:]: v.nnew for k, v in amp_objs.items()}, 'sec', round(time.time() - T0, 1))
        if beta >= burn_beta:
            ensure_local(walkers); Es.append(float(np.mean([localc[int(x)][1] for x in walkers])))
    ensure_local(walkers); Efinal = float(np.mean([localc[int(x)][1] for x in walkers]))
    return np.asarray(Es), np.asarray(curve), Efinal, uniq_hist


rows = []
for seed in seeds:
    t0 = time.time()
    Es, curve, Efin, uh = run(M, BETA, BURN, TAUMAX, seed)
    row = dict(seed=seed, Emean=float(Es.mean()), Emean_site=float(Es.mean() / 36), tail8=float(Es[-8:].mean()),
               nE=len(Es), E_beta0=float(curve[0, 2]), E_final=Efin, curve_beta=curve[:, 0].tolist(),
               curve_E=curve[:, 2].tolist(), mean_unique=float(np.mean(uh)), sec=time.time() - t0,
               amp_evals={k: v.nnew for k, v in amp_objs.items()}, cache_states=len(localc))
    rows.append(row)
    C.log('FN_REP', json.dumps({k: v for k, v in row.items() if not k.startswith('curve_')}))
em = np.array([r['Emean'] for r in rows])
summ = dict(state=st, tag=tag, M=M, beta_target=BETA, burn_beta=BURN, tau_max=TAUMAX, dtype=DTYPE, reps=rows,
            mean=float(em.mean()), mean_site=float(em.mean() / 36),
            between_rep_SE_site=float(em.std(ddof=1) / np.sqrt(len(em)) / 36) if len(em) > 1 else None,
            wall_sec=time.time() - T0)
json.dump(summ, open(f'fn_{tag}.json', 'w'), indent=1)
C.log('FN_SUMMARY', tag, summ['mean_site'], summ['between_rep_SE_site'])
