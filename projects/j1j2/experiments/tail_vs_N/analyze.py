#!/usr/bin/env python3
"""Analysis + figure for tail_vs_N: reads results/tail_vs_N/tvn_N*.json, writes metrics.json, table fragments, figure."""
import json, os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', '..', 'results', 'tail_vs_N')
CH = dict(norb=0, ncfg=1, mass=2, c_quad=3, c_exact=4, eq_min=5, ex_min=6, eq_max=7, ex_max=8)


def cum_curve(h, ch):
    """cumulative share from the bulk: arrays (thr log10 phi^2 at bin lower edge, share of channel on configs >= thr)."""
    BPD = 8
    x = np.asarray(h[CH[ch]]); c = np.cumsum(x) / x.sum()
    thr = -(np.arange(len(x)) + 1) / BPD
    return thr, c


def crossing(thr, cum, level):
    """log10 phi^2 at which the bulk-side cumulative share reaches `level` (linear in log10, within the bin)."""
    i = int(np.searchsorted(cum, level))
    if i >= len(cum): return thr[-1]
    t_hi = thr[i] + 1 / 8                         # upper edge of bin i (cum before this bin = cum[i-1])
    c0 = cum[i - 1] if i > 0 else 0.0
    f = (level - c0) / max(cum[i] - c0, 1e-300)
    return t_hi - f / 8


def below(h, ch, t):
    """sum of channel over configurations with log10 phi^2 < t (bin-interpolated)."""
    x = np.asarray(h[CH[ch]]); BPD = 8
    edges_hi = -np.arange(len(x)) / BPD           # upper edge of bin
    tot = 0.0
    for b in range(len(x)):
        hi, lo = edges_hi[b], edges_hi[b] - 1 / BPD
        if hi <= t: tot += x[b]
        elif lo < t: tot += x[b] * (t - lo) * BPD
    return tot


def metrics(st):
    h = st['hist']; Ncfg = st['N_cfg']
    out = dict(k=st['k'], guide=st['guide'], G=st['G'], Q=st['Q'], Q_over_G=st['Q'] / st['G'],
               rms_delta=st['rms_delta'], eps_trial=st['E_trial_per_site_above_E0'], E_F_a=st['E_F_a_per_site_above_E0'],
               E_FN=st['E_FN_per_site_above_E0'], exact_check=st['G_exact_decomp'] / st['G'],
               second_order=[r['Q_eps2_over_G'] for r in st['second_order']])
    mass = np.asarray(h[CH['mass']]); norb = np.asarray(h[CH['norb']]); ncfg = np.asarray(h[CH['ncfg']])
    for ch in ('c_quad', 'c_exact', 'eq_min', 'ex_min', 'eq_max', 'ex_max'):
        thr, cum = cum_curve(h, ch)
        r = {}
        for lev, nm in ((0.5, '50'), (0.2, '80tail')):          # bulk share 50%  <=>  tail share 50%;  bulk 20% <=> tail 80%
            t = crossing(thr, cum, lev)
            r['t' + nm] = t
            r['xt' + nm] = t + np.log10(Ncfg)                     # normalised: log10(phi^2 / uniform)
            r['mass_below_' + nm] = below(h, 'mass', t)
            r['ncfg_below_' + nm] = below(h, 'ncfg', t)
            r['norb_below_' + nm] = below(h, 'norb', t)
            r['frac_cfg_below_' + nm] = r['ncfg_below_' + nm] / Ncfg
        out[ch] = r
    if len(h) > 10:                               # Lorenz: smallest set of configurations (ranked by gain density) holding X of G
        g = np.cumsum(h[10]); nc = np.cumsum(h[9]) / Ncfg
        for lev in (0.5, 0.8):
            i = int(np.searchsorted(g, lev)); out[f'support{int(lev*100)}'] = float(nc[min(i, len(nc) - 1)])
            out[f'support{int(lev*100)}_n'] = float(nc[min(i, len(nc) - 1)] * Ncfg)
    tr = st['truncation']
    out['trunc'] = tr
    return out


def load(res=RES):
    data = {}
    for f in sorted(glob.glob(os.path.join(res, 'tvn_N*.json'))):
        d = json.load(open(f))
        if d['cluster'] not in data: data[d['cluster']] = d; continue
        cur = {(s['k'] if s['k'] is not None else 'vit'): s for s in data[d['cluster']]['states']}
        for s in d['states']:                                   # later file wins, but keeps truncation curves of an earlier run
            key = s['k'] if s['k'] is not None else 'vit'
            if key in cur and not s.get('truncation'): s['truncation'] = cur[key].get('truncation', [])
            cur[key] = s
        data[d['cluster']]['states'] = [cur[k] for k in sorted(cur, key=lambda z: (z == 'vit', z if z != 'vit' else 0))]
    return data


if __name__ == '__main__':
    data = load()
    allm = {}
    for cl, d in data.items():
        allm[cl] = dict(N=d['N'], D=d['D'], E0_per_site=d['E0_per_site'], states=[metrics(st) for st in d['states']])
    json.dump(allm, open(os.path.join(RES, 'metrics.json'), 'w'), indent=1, default=float)
    print('wrote metrics.json', {k: len(v['states']) for k, v in allm.items()})
