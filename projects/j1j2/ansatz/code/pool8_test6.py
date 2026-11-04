"""pool8_test6.py: test (a) of pool8.py, 6x6 exactness on B_4 = theta_4 (pfaffian_r4 table).
  python pool8_test6.py OUT.json F1.npz F2.npz ...        (files from `pool8.py run --problem 6`; every ensemble of every file is an independent chain)
Exact: orbit probabilities n_r |B_r|^(2 beta) / Z (n_r = norb, B_r = exp(la_table), the table pool_fit.py 'build' reads); decades as pool_fit.pool_weights defines them,
d = floor((2 lnB - med) / ln 10), med = median of 2 lnB at beta = 1 (here the EXACT weighted median; the sampler's own median is reported next to it);
fitted decades +2..-6 (9) plus the two overflow bins (above / below) so that the fractions sum to 1.
Errors: (1) scatter of the per-chain means over the independent chains (E_tot of them), (2) tau-corrected: sqrt(2 tau var / n_samples) from the chain-averaged ACF.
z = (sampler - exact) / SE, quoted for both; chi2/dof over every bin (11 fractions - 1 constraint, + <ln|B|^2>) and rung."""
import sys, os, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8
LN10 = math.log(10.0); DECS = list(range(2, -7, -1))


def exact_tables(betas):
    la = np.load(P8.LA4).astype(np.float64); n = np.load(P8.NORB6).astype(np.float64); assert la.shape == n.shape
    x = 2.0 * la
    def wmed(w):
        o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, 0.5 * c[-1])])
    out = {}
    lw = np.log(n) + 2.0 * (la - la.max()); w1 = np.exp(lw - lw.max()); w1 /= w1.sum(); med = wmed(w1)
    dec = np.floor((x - med) / LN10).astype(np.int64)
    bins = np.where(dec > DECS[0], 100, np.where(dec < DECS[-1], -100, dec))
    for b in betas:
        lw = np.log(n) + 2.0 * b * (la - la.max()); w = np.exp(lw - lw.max()); w /= w.sum()
        f = {int(d): float(w[bins == d].sum()) for d in [100] + DECS + [-100]}
        m1 = float((w * x).sum()); m2 = float((w * x * x).sum())
        out[b] = dict(frac=f, mean2lnB=m1, var2lnB=m2 - m1 * m1)
    return med, out


def main(outp, files):
    Z = [np.load(f) for f in files]; betas = [float(b) for b in Z[0]['betas']]
    ser = np.concatenate([z['lnb_series'].astype(np.float64) for z in Z], axis=2)                  # (T, R, E_tot)
    T, R, Et = ser.shape; med, ex = exact_tables(betas)
    res = dict(files=files, T=int(T), E_tot=int(Et), betas=betas, med2lnB_exact=med, mode=str(Z[0]['mode']), broken=str(Z[0]['broken']), eval=str(Z[0]['evalname']), rungs=[])
    med_s = float(np.median(2.0 * ser[:, 0, :])); res['med2lnB_sampler_beta1'] = med_s
    zs_all = []; chi2 = 0.0; dof = 0
    for q, b in enumerate(betas):
        x = 2.0 * ser[:, q, :]                                                        # (T, E)
        d = np.floor((x - med) / LN10); bins = np.where(d > DECS[0], 100, np.where(d < DECS[-1], -100, d)).astype(np.int64)
        rows = []; zq = []
        for key in [100] + DECS + [-100]:
            ind = (bins == key).astype(np.float64); fm = ind.mean(0)                       # per chain
            f = fm.mean(); se1 = fm.std(ddof=1) / math.sqrt(Et)
            tau, M = P8.tau_int(ind.T); fe = ex[b]['frac'][key]
            v = ind.var() if ind.var() > 0 else fe * (1.0 - fe)                       # empty bin: the exact binomial variance (conservative floor, no inf z)
            se2 = math.sqrt(2.0 * tau * v / (T * Et)); se1 = max(se1, se2 if ind.var() == 0 else 0.0)
            z1 = (f - fe) / se1; z2 = (f - fe) / se2
            # rare bins (expected independent events < 50): the Gaussian z is not valid; use the exact two-sided Poisson tail (independent events = T Et / (2 tau))
            nind = T * Et / (2.0 * max(tau, 0.5)); mu = fe * nind; k = f * nind; zp = None
            if mu < 50 and not os.environ.get('POOL8_GAUSS'):
                from scipy.stats import poisson, norm
                kk = int(round(k)); p = min(1.0, 2.0 * min(poisson.cdf(kk, mu), poisson.sf(kk - 1, mu))); zp = float(np.sign(f - fe) * norm.isf(max(p, 1e-300) / 2.0))
                z1 = z2 = zp
            rows.append(dict(bin=key, exact=fe, mean=f, se_scatter=se1, se_tau=se2, tau=tau, z_scatter=z1, z_tau=z2, rare_poisson=(zp is not None), mu_events=mu, k_events=k)); zq.append(z1)
        fm = x.mean(0); m = fm.mean(); se1 = fm.std(ddof=1) / math.sqrt(Et); tau, M = P8.tau_int(x.T); se2 = math.sqrt(2.0 * tau * x.var() / (T * Et))
        mrow = dict(exact=ex[b]['mean2lnB'], mean=m, se_scatter=se1, se_tau=se2, tau=tau, z_scatter=(m - ex[b]['mean2lnB']) / se1, z_tau=(m - ex[b]['mean2lnB']) / se2)
        zr = [r['z_scatter'] for r in rows[:-0 or None]]
        c2 = float(np.sum(np.array([r['z_scatter'] for r in rows]) ** 2) + mrow['z_scatter'] ** 2); dg = len(rows) - 1 + 1
        chi2 += c2; dof += dg
        res['rungs'].append(dict(beta=b, bins=rows, mean2lnB=mrow, chi2=c2, dof=dg, max_abs_z_scatter=float(max(abs(r['z_scatter']) for r in rows + [mrow])),
                                 max_abs_z_tau=float(max(abs(r['z_tau']) for r in rows + [mrow])), tau_max_bin=float(max(r['tau'] for r in rows)),
                                 acc_move=float(np.mean([z['acc'][q] for z in Z])), swap_acc=[float(np.mean([z['swap_acc'][i] for z in Z])) for i in range(R - 1)]))
    if str(Z[0]['evalname']) == 'pp2':                         # the PP2 chain must carry the table's ln|B| on every stored config (+ Sz = 0)
        te = P8.TableEval6(); dmax = 0.0; n = 0; sz = 0
        for z in Z:
            if z['cfg'].ndim < 4: continue
            c = z['cfg'].reshape(-1, 36); l = z['lnb_thin'].reshape(-1); sz = max(sz, int(np.abs(c.sum(1)).max()))
            dmax = max(dmax, float(np.max(np.abs(te.lnB(c.astype(np.float64)) - l)))); n += len(l)
        res['pp2_vs_table_stored_cfgs'] = dict(n=n, max_abs_diff_lnB=dmax, max_abs_Sz=sz)
        print(f'  PP2 lnB vs table lnB on {n} stored configs: max |diff| {dmax:.2e}, max |Sz| {sz}')
    res['max_abs_z_scatter'] = max(r['max_abs_z_scatter'] for r in res['rungs']); res['max_abs_z_tau'] = max(r['max_abs_z_tau'] for r in res['rungs'])
    res['chi2'] = chi2; res['dof'] = dof; res['chi2_per_dof'] = chi2 / dof
    res['PASS'] = bool(res['max_abs_z_scatter'] <= 3.0 and res['max_abs_z_tau'] <= 3.0 and res['chi2_per_dof'] <= 2.0)
    json.dump(res, open(outp, 'w'), indent=1, default=float)
    print(f"{res['mode']}/{res['broken']}/{res['eval']}: chains {Et} x T {T}: max|z| scatter {res['max_abs_z_scatter']:.2f} tau {res['max_abs_z_tau']:.2f}  chi2/dof {chi2:.1f}/{dof} = {res['chi2_per_dof']:.2f}  "
          f"median sampler {med_s:.3f} exact {med:.3f}  PASS={res['PASS']}")
    for r in res['rungs']:
        print(f"  beta {r['beta']}: acc {r['acc_move']:.3f} swap {np.round(r['swap_acc'], 3).tolist()} tau_max_bin {r['tau_max_bin']:.1f}  max|z| {r['max_abs_z_scatter']:.2f}/{r['max_abs_z_tau']:.2f} "
              f"frac exact " + ' '.join(f"{v['exact']:.2e}" for v in r['bins']) + f"  <2lnB> {r['mean2lnB']['mean']:.3f}+-{r['mean2lnB']['se_scatter']:.3f} vs {r['mean2lnB']['exact']:.3f}")
    return res


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
