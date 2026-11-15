"""vardec_analyse.py: estimators from the T-VARDEC records.  usage: python vardec_analyse.py RUN_OUT [--gate -5] (reads checkpoint.npz) -> prints the table, writes result.json.
Records (per gated sample, authoritative decade <= gate): chain, sweep, dec, lw (weights, f32 sampler 2 Re ln psi), Hxx, EL, F, pair, Fp, pairp.
Tempered sampling |psi|^(2p): weight omega = exp((1-p) (lw - MREF)); estimator of E_{|psi|^2}[q] = sum omega q / sum omega over ALL walker-samples (den[t, k] = sum of omega over the walkers of chain k at sweep t).
S(x) = 1_F (single + pair) = boundary contribution of x (-4 sum_{y not in F} H_xy r_y); set F (c = -1) and F' (E_L > H_xx)."""
import json, math, sys
import numpy as np

MREF = -78.4616959545097


def binerr(x, minbins=32):
    x = np.asarray(x, float); e = []
    while len(x) >= minbins:
        e.append(float(x.std(ddof=1) / math.sqrt(len(x)))); n = len(x) // 2 * 2; x = x[:n].reshape(-1, 2).mean(1)
    return e


def chain_series(rec, den, K, mask_fn, val_fn, nsw):
    """per chain, per sweep: sum_omega q / mean_t den(t)  (so the chain mean of the series IS the ratio estimator)."""
    out = np.zeros((nsw, K)); m = mask_fn(rec)
    if m.any():
        np.add.at(out, (rec['sweep'][m], rec['chain'][m]), rec['omega'][m] * val_fn(rec)[m])
    return out / den.mean(0)[None, :]


def summarize(ser, nsite=64.0):
    """ser (nsw, K) series of one quantity -> dict value/site, errors, per-chain."""
    K = ser.shape[1]; per = ser.mean(0) / nsite; val = float(per.mean())
    scat = float(per.std(ddof=1) / math.sqrt(K)) if K > 1 else float('nan')
    bl = [binerr(ser[:, k] / nsite) for k in range(K)]
    eb = float(np.sqrt(sum((max(b) if b else float('nan')) ** 2 for b in bl)) / K)
    return dict(value=val, err_scatter=scat, err_binning=eb, err=float(max(scat, eb)), per_chain=per.tolist(), binning_levels_chain0=bl[0])


def analyse(rec, den, p, K, gate, nsite=64.0, decmin=-16):
    nsw = den.shape[0]; R = dict(rec); R['omega'] = np.exp((1.0 - p) * (R['lw'] - MREF) + (R['lw64'] - R['lw'])); R['F'] = R['F'].astype(bool); R['Fp'] = R['Fp'].astype(bool)
    single = 4.0 * (R['Hxx'] - R['EL']); S_F = single + 4.0 * R['pair']; S_Fp = single + 4.0 * R['pairp']; win = R['EL'] < R['Hxx']
    q = dict(
        m_gated=(lambda r: np.ones(len(r['dec']), bool), lambda r: np.ones(len(r['dec']))),
        m_F=(lambda r: r['F'], lambda r: np.ones(len(r['dec']))),
        m_F_window=(lambda r: r['F'] & win, lambda r: np.ones(len(r['dec']))),
        m_F_good=(lambda r: r['F'] & ~win, lambda r: np.ones(len(r['dec']))),
        total_F=(lambda r: r['F'], lambda r: S_F), single_F=(lambda r: r['F'], lambda r: single), pair_F=(lambda r: r['F'], lambda r: 4.0 * r['pair']),
        total_F_window=(lambda r: r['F'] & win, lambda r: S_F), total_F_good=(lambda r: r['F'] & ~win, lambda r: S_F),
        single_F_window=(lambda r: r['F'] & win, lambda r: single), single_F_good=(lambda r: r['F'] & ~win, lambda r: single),
        pair_F_window=(lambda r: r['F'] & win, lambda r: 4.0 * r['pair']), pair_F_good=(lambda r: r['F'] & ~win, lambda r: 4.0 * r['pair']),
        m_Fp=(lambda r: r['Fp'], lambda r: np.ones(len(r['dec']))),
        total_Fp=(lambda r: r['Fp'], lambda r: S_Fp), single_Fp=(lambda r: r['Fp'], lambda r: single), pair_Fp=(lambda r: r['Fp'], lambda r: 4.0 * r['pairp']))
    out = dict(gate=gate, p=p, chains=K, sweeps=nsw, n_gated=int(len(R['dec'])), n_F=int(R['F'].sum()), n_F_window=int((R['F'] & win).sum()), n_F_good=int((R['F'] & ~win).sum()), n_Fp=int(R['Fp'].sum()))
    ess = lambda m: float(R['omega'][m].sum() ** 2 / max((R['omega'][m] ** 2).sum(), 1e-300)) if m.any() else 0.0
    out['ess'] = dict(F=ess(R['F']), F_window=ess(R['F'] & win), F_good=ess(R['F'] & ~win), Fp=ess(R['Fp']))
    series = {}
    for name, (mf, vf) in q.items():
        ser = chain_series(R, den, K, mf, vf, nsw); series[name] = ser
        d = summarize(ser, 1.0 if name.startswith('m_') else nsite); out[name] = d
    out['window_share_of_mF'] = float(out['m_F_window']['value'] / out['m_F']['value']) if out['m_F']['value'] else float('nan')
    out['inconclusive_lt50'] = dict(F_window=out['n_F_window'] < 50, F_good=out['n_F_good'] < 50, F=out['n_F'] < 50, Fp=out['n_Fp'] < 50)
    # per decade
    tab = []
    for d in sorted(set(R['dec'].astype(int).tolist())):
        sel = R['dec'].astype(int) == d; sub = {k: v[sel] for k, v in R.items() if isinstance(v, np.ndarray) and len(v) == len(sel)}
        den_tot = den.sum(); om = sub['omega']; Fm = sub['F']
        s = lambda m, v: float((om * m * v).sum() / den_tot)
        sg = 4.0 * (sub['Hxx'] - sub['EL']); w_ = sub['EL'] < sub['Hxx']; one = np.ones(len(om))
        tab.append(dict(decade=d, n=int(sel.sum()), mass_gated=s(one, one), n_F=int(Fm.sum()), mF=s(Fm, one), mF_window=s(Fm & w_, one), single_per_site=s(Fm, sg) / nsite,
                        pair_per_site=s(Fm, 4.0 * sub['pair']) / nsite, total_per_site=s(Fm, sg + 4.0 * sub['pair']) / nsite, n_Fp=int(sub['Fp'].sum()), mFp=s(sub['Fp'], one),
                        total_Fp_per_site=s(sub['Fp'], sg + 4.0 * sub['pairp']) / nsite))
    out['per_decade'] = tab
    return out, series


def fmt(o):
    g = lambda k: f"{o[k]['value']:+.3e} +- {o[k]['err']:.1e}"
    L = [f"gate {o['gate']}  p {o['p']}  chains {o['chains']}  sweeps {o['sweeps']}  n_gated {o['n_gated']}  n_F {o['n_F']} (window {o['n_F_window']}, good {o['n_F_good']})  n_F' {o['n_Fp']}",
         f"  m_F {g('m_F')}  window share {o['window_share_of_mF']:.3f}  m_F' {g('m_Fp')}",
         f"  F : single {g('single_F')}  pair {g('pair_F')}  total dE/site {g('total_F')}",
         f"      window-x boundary {g('total_F_window')} (n {o['n_F_window']})  good-x boundary {g('total_F_good')} (n {o['n_F_good']})",
         f"  ESS (weights) F {o['ess']['F']:.0f} F_window {o['ess']['F_window']:.0f} F_good {o['ess']['F_good']:.0f} F' {o['ess']['Fp']:.0f}",
         f"  F': single {g('single_Fp')}  pair {g('pair_Fp')}  total dE/site {g('total_Fp')}  (n {o['n_Fp']})"]
    return '\n'.join(L)


if __name__ == '__main__':
    run = sys.argv[1]; z = np.load(f'{run}/checkpoint.npz', allow_pickle=True); p = float(z['p']); K = int(z['K'])
    for g in [float(x) for x in z['gates']]:
        rec = {k[len(f'g{int(g)}_'):]: z[k] for k in z.files if k.startswith(f'g{int(g)}_')}
        o, ser = analyse(rec, z['den'][:int(z['nsw'])], p, K, g); print(fmt(o))
