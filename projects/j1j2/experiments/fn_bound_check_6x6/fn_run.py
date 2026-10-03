"""FN GFMC (verbatim algorithm of fn_guides_compare.py / gfmc_6x6_gr_population_size.py) with the Krylov guide and
a recorded projection-time curve E_mixed(beta).  RNG consumption identical to fgc.run, so the same seed reproduces it.
Usage: python fn_run.py T M beta burn tau_max tag seed1 [seed2 ...]
"""
import sys, os, json, time
import numpy as np
BASE = os.environ.get("FNGC_DIR", ".")
sys.path.insert(0, BASE)
T, M, BETA, BURN, TAUMAX, tag = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
seeds = [int(s) for s in sys.argv[7:]]
sys.argv = ['fn_guides_compare.py', 'fn', '6', 'krylov', '1']
import fn_guides_compare as fgc
fgc.GUIDE = 'krylov'; fgc.T = T
ensure_local = fgc.ensure_local; localc = fgc.localc; systematic = fgc.systematic

T0 = time.time()
def run(M, beta_target, burn_beta, tau_max, seed):
    rg = np.random.default_rng(seed)
    pool = np.load(os.environ.get('POOLFILE', fgc.POOL))['states'].astype(np.uint64).reshape(-1)
    walkers = pool[rg.choice(len(pool), M, replace=False)].copy()
    beta = 0.; Es = []; curve = []   # curve rows: beta_start, tau, E_mixed(walkers at beta_start) , mean weight-denominator
    uniq_hist = []
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
        if len(curve) % 5 == 1: print('PROG', seed, len(curve), 'beta', beta, 'E', Eref, 'neval', fgc.neval, 'sec', time.time() - T0, flush=True)
        if beta >= burn_beta:
            ensure_local(walkers); Es.append(float(np.mean([localc[int(x)][1] for x in walkers])))
    ensure_local(walkers); Efinal = float(np.mean([localc[int(x)][1] for x in walkers]))
    return np.asarray(Es), np.asarray(curve), Efinal, uniq_hist

rows = []
for seed in seeds:
    t0 = time.time()
    Es, curve, Efin, uh = run(M, BETA, BURN, TAUMAX, seed)
    row = dict(seed=seed, Emean=float(Es.mean()), tail8=float(Es[-8:].mean()), nE=len(Es), E_beta0=float(curve[0, 2]),
               E_final=Efin, curve_beta=curve[:, 0].tolist(), curve_tau=curve[:, 1].tolist(), curve_E=curve[:, 2].tolist(),
               mean_unique=float(np.mean(uh)), sec=time.time() - t0, neval=fgc.neval)
    rows.append(row)
    print("FN_REP", json.dumps({k: v for k, v in row.items() if not k.startswith('curve_')}), flush=True)
em = np.array([r["Emean"] for r in rows])
summ = dict(T=T, M=M, beta_target=BETA, burn_beta=BURN, tau_max=TAUMAX, reps=rows, mean=float(em.mean()),
            between_rep_SE=float(em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 1 else None)
summ["mean_site"] = summ["mean"] / 36
json.dump(summ, open(f"fn_run_{tag}.json", "w"))
print("FN_SUMMARY", tag, "mean", summ["mean"], "site", summ["mean_site"], "SE", summ["between_rep_SE"], flush=True)
