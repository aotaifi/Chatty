"""Analysis of the FN referee calibration: DMC (GPU, fncal_dmc.py) minus exact sector FN, per guide / M / beta window.

Usage: python fncal_analyze.py CELLDIR [OUTDIR]   (cells: cell_<guide>_M<M>_b<beta>_s<seed>.npz, several seeds are pooled)
Estimators per (guide, M, window):
  step      : the referee as used so far: equal-weight mean over STEPS of the post-step population energy, mean over populations
  step_median : median over populations of the same
  mean      : beta-TIME average (population at the start of step t, energy Eref_t, weight tau_t = its beta duration), mean over pops
  median / trim10 : median / 10% trimmed mean over populations of the time average (bootstrap SE)
  wp<p>     : time average combined with Sorella-type correction factors over the last p steps, pooled over populations:
              sum_t W_t w_t E_t / sum_t W_t w_t,  W_t = prod_{j=t-p}^{t-1} (1 - tau_j (Eref_j - E_T)),  E_T = pooled mean energy
Bias fits: bias(M) = a + b/M (weighted least squares, M >= Mmin) and the exponent alpha in |bias| ~ M^-alpha.
"""
import os, sys, glob, json
import numpy as np
from fncal_common import window_estimates, EXACT, GUIDES, N, E0_SITE

WINDOWS = {'b1.2': (0.4, 1.2), 'b2.4': (0.8, 2.4)}
PS = (20,)
rng = np.random.default_rng(12345)


def load_cells(d):
    cells = {}
    for f in sorted(glob.glob(os.path.join(d, '**', 'cell_*.npz'), recursive=True)):
        z = np.load(f)
        key = (str(z['guide']), int(z['M']), float(z['tau_max']))
        cells.setdefault(key, []).append(dict(Eref=z['Eref'], tau=z['tau'], nd=z['ndist'], beta=float(z['beta_target']), file=f))
    out = {}
    for k, lst in cells.items():
        S = max(c['Eref'].shape[1] for c in lst)
        pad = lambda a, v: np.pad(a, ((0, 0), (0, S - a.shape[1])), constant_values=v)
        out[k] = dict(Eref=np.concatenate([pad(c['Eref'], np.nan) for c in lst]), tau=np.concatenate([pad(c['tau'], 0.0) for c in lst]),
                      nd=np.concatenate([pad(c['nd'], 0) for c in lst]), beta=max(c['beta'] for c in lst), files=[c['file'] for c in lst])
    return out


def pop_means_step(Eref, tau, burn, beta):
    """referee as used so far (ll6_fn / it2_fn): equal-weight mean over STEPS of the post-step population energy."""
    e, n = window_estimates(np.nan_to_num(Eref, nan=0.0), tau, burn, beta)
    return e / N


def _tw(tau, burn, beta):
    b_after = np.cumsum(tau, 1); b_before = b_after - tau
    ov = np.clip(np.minimum(b_after, beta) - np.maximum(b_before, burn), 0, None)
    return ov * (tau > 0)


def pop_means(Eref, tau, burn, beta):
    """beta-time average: population at the start of step t (energy Eref_t) lives for beta-duration tau_t; weight = overlap of
    [beta_before_t, beta_after_t) with the window.  Removes the correlation between the adaptive step tau_t = min(tau_max,
    0.8/max(d_FN - Eref)) and the population content (an extreme walker raises Eref AND shrinks tau)."""
    w = _tw(tau, burn, beta)
    E0 = np.nan_to_num(Eref, nan=0.0)
    return (E0 * w).sum(1) / np.maximum(w.sum(1), 1e-30) / N


def weighted_pooled(Eref, tau, burn, beta, p, ET):
    """per-population (numerator, denominator) of the pooled ratio sum_t W_t w_t Eref_t / sum_t W_t w_t (w_t = beta overlap),
    W_t = prod_{j=t-p}^{t-1} (1 - tau_j (Eref_j - E_T)) (Sorella-type correction factors over the last p steps)."""
    P, S = tau.shape
    E0 = np.nan_to_num(Eref, nan=0.0)
    w = _tw(tau, burn, beta)
    f = np.where(tau > 0, 1.0 - tau * (E0 - ET), 1.0)
    lf = np.log(np.maximum(f, 1e-12))
    cs = np.concatenate([np.zeros((P, 1)), np.cumsum(lf, 1)], 1)          # cs[:, t] = sum_{j<t} lf_j
    t = np.arange(S)
    lo = np.maximum(t - p, 0)
    W = np.exp(cs[:, t] - cs[:, lo]) * w
    return (W * E0).sum(1) / N, W.sum(1)


def boot_ratio(num, den, B=400):
    n = len(num)
    est = num.sum() / den.sum()
    idx = rng.integers(0, n, (B, n))
    bs = num[idx].sum(1) / den[idx].sum(1)
    return est, bs.std(ddof=1)


def boot_stat(x, fn, B=400):
    n = len(x); idx = rng.integers(0, n, (B, n))
    return fn(x), np.std([fn(x[i]) for i in idx], ddof=1)


def trim(x, q=0.1):
    x = np.sort(x); k = int(q * len(x)); return x[k:len(x) - k].mean()


def analyze(cells, tau_max=0.025):
    res = {}
    for (g, M, tm), c in sorted(cells.items()):
        if abs(tm - tau_max) > 1e-12 or c['beta'] > 3: continue      # production protocol only; long-beta / tau studies are analysed separately
        ex = EXACT[g]
        for wn, (burn, beta) in WINDOWS.items():
            if c['beta'] < beta - 1e-9: continue
            es = pop_means_step(c['Eref'], c['tau'], burn, beta); es = es[np.isfinite(es)]
            e = pop_means(c['Eref'], c['tau'], burn, beta)
            e = e[np.isfinite(e)]
            n = len(e)
            ET = float(np.mean(e)) * N
            r = dict(guide=g, M=M, window=wn, npop=n, mean=float(e.mean()) - ex, mean_se=float(e.std(ddof=1) / np.sqrt(n)), sd_pop=float(e.std(ddof=1)))
            r.update(step=float(es.mean()) - ex, step_se=float(es.std(ddof=1) / np.sqrt(len(es))))
            m, ms = boot_stat(es, np.median); r.update(step_median=float(m) - ex, step_median_se=float(ms))
            m, ms = boot_stat(e, np.median); r.update(median=float(m) - ex, median_se=float(ms))
            m, ms = boot_stat(e, trim); r.update(trim10=float(m) - ex, trim10_se=float(ms))
            r['frac_below_exact'] = float(np.mean(e < ex)); r['frac_below_E0'] = float(np.mean(e < -0.5038096538908782))
            r['n_extreme_low'] = int(np.sum(e < ex - 5e-4))
            for p in PS:
                num, den = weighted_pooled(c['Eref'], c['tau'], burn, beta, p, ET)
                ok = np.isfinite(num) & (den > 0)
                est, se = boot_ratio(num[ok], den[ok])
                r[f'wp{p}'] = float(est) - ex; r[f'wp{p}_se'] = float(se)
            res[(g, M, wn)] = r
    return res


def fit_bias(Ms, b, se, Mmin=0):
    Ms = np.asarray(Ms, float); b = np.asarray(b); se = np.asarray(se)
    k = Ms >= Mmin
    x = 1.0 / Ms[k]; y = b[k]; w = 1.0 / se[k] ** 2
    out = {}
    if k.sum() >= 2:
        A = np.stack([np.ones_like(x), x], 1)
        C = np.linalg.inv(A.T @ (A * w[:, None])); p = C @ (A.T @ (w * y))
        chi2 = float(np.sum(w * (y - A @ p) ** 2))
        out['free'] = dict(a=float(p[0]), a_se=float(np.sqrt(C[0, 0])), b=float(p[1]), b_se=float(np.sqrt(C[1, 1])), chi2=chi2, dof=int(k.sum() - 2))
    bb = float(np.sum(w * x * y) / np.sum(w * x * x)); bse = float(1 / np.sqrt(np.sum(w * x * x)))
    chi2 = float(np.sum(w * (y - bb * x) ** 2))
    out['through0'] = dict(b=bb, b_se=bse, chi2=chi2, dof=int(k.sum() - 1))
    # power law: log|bias| vs log M for points with |bias| > 2 se
    sel = (np.abs(y) > 2 * se[k]) & (y < 0)
    if sel.sum() >= 2:
        lm = np.log(Ms[k][sel]); ly = np.log(-y[sel]); wl = (np.abs(y[sel]) / se[k][sel]) ** 2
        A = np.stack([np.ones_like(lm), lm], 1); C = np.linalg.inv(A.T @ (A * wl[:, None])); p = C @ (A.T @ (wl * ly))
        out['powerlaw_alpha'] = float(-p[1]); out['powerlaw_alpha_se'] = float(np.sqrt(C[1, 1]))
    return out


if __name__ == '__main__':
    cells = load_cells(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else '.'
    res = analyze(cells)
    rows = [dict(r) for r in res.values()]
    json.dump(rows, open(os.path.join(out, 'fncal_table.json'), 'w'), indent=1)
    for r in rows:
        print(f"{r['guide']:7s} M={r['M']:5d} {r['window']} n={r['npop']:6d} step={r['step']*1e6:7.2f}+-{r['step_se']*1e6:5.2f} "
              f"stepmed={r['step_median']*1e6:7.2f} | tavg={r['mean']*1e6:7.2f}+-{r['mean_se']*1e6:5.2f} "
              f"med={r['median']*1e6:7.2f}+-{r['median_se']*1e6:5.2f} trim={r['trim10']*1e6:7.2f} "
              + ' '.join(f"wp{p}={r[f'wp{p}']*1e6:7.2f}+-{r[f'wp{p}_se']*1e6:5.2f}" for p in PS) + ' (1e-6/site)')
