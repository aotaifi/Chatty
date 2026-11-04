"""pool8_test_drawfn.py: exactness of the PRODUCTION entry point Pool8Sampler.draw_fn at 6x6 (in-sampler tau pilot + auto thinning ON), called as pool_fit.build_pool calls it:
  for beta in betas: words, lnB, orb = draw_fn(beta, S, rng).   Every sampler is independent (own seed).
  python pool8_test_drawfn.py run  OUT.npz --eval table6|pp2 --n 12 --seed0 100 --E 4 --burn 100 --pilot 150 --S 200 [--keep_pilot K]
  python pool8_test_drawfn.py test OUT.json F1.npz ...    (bins/exact/Gaussian rule as pool8_test6.py; z from the scatter over the independent samplers)
Checks on the returned tuple (first 300 draws per sampler): words unpack to Sz = 0 +-1 configs, ln|B| equals the evaluator's, orbit id is a table rep with the table's ln|B| (6x6)."""
import sys, os, json, math, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8
import pool8_test6 as T6
LN10 = math.log(10.0); DECS = T6.DECS


def run(a):
    ev = P8.load_eval(6, a.eval); betas = [1.0, 0.5, 0.25]; tab = P8.TableEval6() if a.eval == 'pp2' else ev
    LB = np.zeros((a.n, 3, a.S)); thin = []; tau = []; chk = dict(n=0, max_dlnB=0.0, max_Sz=0, max_dorb=0.0, orb_in_reps=True); t0 = time.time()
    for k in range(a.n):
        smp = P8.Pool8Sampler(ev, betas, E=a.E, seed=a.seed0 + k, burn=a.burn, pilot=a.pilot, log=lambda m: None, keep_pilot=a.keep_pilot); rng = np.random.default_rng(a.seed0 + k)
        for q, b in enumerate(betas):
            w, l, o = smp.draw_fn(b, a.S, rng); LB[k, q] = l
            if k % 4 == 0 and q == 0:                                                   # tuple check (table lookups are slow-ish)
                c = P8.unpack_words(w[:300], 36); chk['max_Sz'] = max(chk['max_Sz'], int(np.abs(c.sum(1)).max()))
                chk['max_dlnB'] = max(chk['max_dlnB'], float(np.max(np.abs(tab.lnB(c.astype(np.float64)) - l[:300]))))
                ou = o[:300].view(np.uint64); ix = np.searchsorted(tab.reps, ou); ok = bool(np.all(tab.reps[np.minimum(ix, len(tab.reps) - 1)] == ou))
                chk['orb_in_reps'] &= ok; chk['max_dorb'] = max(chk['max_dorb'], float(np.max(np.abs(tab.la[np.minimum(ix, len(tab.reps) - 1)] - l[:300])))); chk['n'] += 300
        thin.append(smp.thin); tau.append([max(t['tau_lnB2'], t['tau_dec']) for t in smp.tau])
        if (k + 1) % max(1, a.n // 10) == 0: print(f'[drawfn] sampler {k + 1}/{a.n} {time.time() - t0:.0f}s thin {smp.thin}', flush=True)
    np.savez(a.out, lnB=LB, thin=np.array(thin), tau=np.array(tau), betas=np.array(betas), seed0=a.seed0, evalname=a.eval, keep_pilot=a.keep_pilot, E=a.E, burn=a.burn, pilot=a.pilot, S=a.S,
             chk=json.dumps(chk), sec=time.time() - t0)
    print(f'[drawfn] done {time.time() - t0:.0f}s thin {np.bincount(thin).tolist()} checks {chk}', flush=True)


def test(outp, files):
    Z = [np.load(f) for f in files]; betas = [float(b) for b in Z[0]['betas']]; LB = np.concatenate([z['lnB'] for z in Z]); n = LB.shape[0]; S = LB.shape[2]
    med, ex = T6.exact_tables(betas); res = dict(files=files, n_samplers=int(n), S_per_beta=int(S), med2lnB_exact=med, eval=str(Z[0]['evalname']), keep_pilot=int(Z[0]['keep_pilot']),
                                              thin_hist={int(k): int(v) for k, v in zip(*np.unique(np.concatenate([z['thin'] for z in Z]), return_counts=True))}, rungs=[])
    chi2 = 0.0; dof = 0; mz = 0.0
    for q, b in enumerate(betas):
        x = 2.0 * LB[:, q, :]; d = np.floor((x - med) / LN10); bins = np.where(d > DECS[0], 100, np.where(d < DECS[-1], -100, d)).astype(np.int64); rows = []
        for key in [100] + DECS + [-100]:
            fm = (bins == key).mean(1); f = fm.mean(); fe = ex[b]['frac'][key]; v = fm.var(ddof=1)
            se = math.sqrt(v / n) if v > 0 else math.sqrt(fe * (1 - fe) / (n * S))                  # empty bin: binomial floor
            rows.append(dict(bin=key, exact=fe, mean=float(f), se=se, z=float((f - fe) / se)))
        fm = x.mean(1); m = fm.mean(); se = fm.std(ddof=1) / math.sqrt(n); rows_m = dict(exact=ex[b]['mean2lnB'], mean=float(m), se=float(se), z=float((m - ex[b]['mean2lnB']) / se))
        c2 = float(sum(r['z'] ** 2 for r in rows) + rows_m['z'] ** 2); dg = len(rows); chi2 += c2; dof += dg
        mx = max(abs(r['z']) for r in rows + [rows_m]); mz = max(mz, mx); res['rungs'].append(dict(beta=b, bins=rows, mean2lnB=rows_m, chi2=c2, dof=dg, max_abs_z=mx))
    res['max_abs_z'] = mz; res['chi2'] = chi2; res['dof'] = dof; res['chi2_per_dof'] = chi2 / dof; res['PASS'] = bool(mz <= 3.0 and chi2 / dof <= 1.5)
    res['checks'] = [json.loads(str(z['chk'])) for z in Z]
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    print(f"draw_fn {res['eval']} keep_pilot={res['keep_pilot']}: samplers {n} x S {S}/beta: max|z| {mz:.2f}  chi2/dof {chi2:.1f}/{dof} = {chi2 / dof:.2f}  PASS={res['PASS']}  thin hist {res['thin_hist']}")
    for r in res['rungs']:
        print(f"  beta {r['beta']}: max|z| {r['max_abs_z']:.2f} chi2/dof {r['chi2'] / r['dof']:.2f}  worst bins {[(x['bin'], round(x['z'], 2)) for x in sorted(r['bins'], key=lambda x: -abs(x['z']))[:2]]}  <2lnB> {r['mean2lnB']['mean']:.3f}+-{r['mean2lnB']['se']:.3f} vs {r['mean2lnB']['exact']:.3f}")
    print('  tuple checks:', {k: max(c[k] for c in res['checks']) if k != 'orb_in_reps' else all(c[k] for c in res['checks']) for k in res['checks'][0]})


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('out'); ap.add_argument('--eval', default='table6'); ap.add_argument('--n', type=int, default=12)
        ap.add_argument('--seed0', type=int, default=100); ap.add_argument('--E', type=int, default=4); ap.add_argument('--burn', type=int, default=100); ap.add_argument('--pilot', type=int, default=150)
        ap.add_argument('--S', type=int, default=200); ap.add_argument('--keep_pilot', type=int, default=0); run(ap.parse_args())
    else: test(sys.argv[2], sys.argv[3:])
