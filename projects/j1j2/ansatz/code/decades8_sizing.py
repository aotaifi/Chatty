"""decades8_sizing.py: DERIVED pool size for the fitted relative decades -2..-9 (PLAN_pfaffian "8x8 decade range" follow-up (1)).
Occupancy f_{d,q} and in-cell IS efficiency eps_{d,q} (pool_fit band-uniform weights exp(-2 beta lnB), eps = ESS/n) per (beta, decade) from the decades8 draws
(run8L + run8L2); decades relative to the median of 2 lnB over the beta = 1 draws.  ESS_d = 0.8 sum_q S_q f_{d,q} eps_{d,q} >= ratio P for every d (pool8_sizing.py),
S_1 >= 2e4; LP per rung subset.  Cost with independent chains per rung: sum_q S_q thin_q s_w, thin_q = ceil(2 tau_q), s_w = 0.33 s/sweep/walker (pool8).
Error of eps: half-split of the files (odd / even), reported as the spread of the resulting S.
usage: python3 decades8_sizing.py OUT.json files..."""
import sys, json, glob, itertools, math
import numpy as np
from scipy.optimize import linprog
out = sys.argv[1]; files = sorted(sys.argv[2:])
P, RATIO, SW = 1.8e4, 3.0, 0.33
TAU = {1.0: 1.2, 0.5: 0.7, 0.25: 0.6, 0.125: 0.59, 0.0625: 0.61}       # pool8 results (1, .5, .25); decades8 pilots median (.125, .0625)
DECS = list(range(-2, -10, -1)); LN10 = math.log(10.0)
def load(fs):
    b = []; l = []
    for f in fs: z = np.load(f); b.append(z['beta']); l.append(z['lnB'])
    return np.concatenate(b), np.concatenate(l)
def table(beta, lnB, med):
    dec = np.floor((2 * lnB - med) / LN10).astype(int); tab = {}
    for q in sorted(set(beta.tolist()), reverse=True):
        m = beta == q; n = m.sum(); tab[q] = {}
        for d in DECS:
            c = m & (dec == d); k = int(c.sum())
            if k < 2: tab[q][d] = (k / n, 0.0, k); continue
            lw = -2 * q * lnB[c]; w = np.exp(lw - lw.max()); eps = (w.sum() ** 2 / (w * w).sum()) / k
            tab[q][d] = (k / n, float(eps), k)
    return tab
def lp(tab, sub):
    A = -0.8 * np.array([[tab[q][d][0] * tab[q][d][1] for q in sub] for d in DECS])
    if np.any(np.all(A >= 0, axis=1)): return None
    o = linprog(np.ones(len(sub)), A_ub=A, b_ub=-RATIO * P * np.ones(len(DECS)), bounds=[((2e4 if q == 1.0 else 0), None) for q in sub])
    return o.x if o.success else None
def size(tab):
    rows = []
    for k in range(1, 6):
        for sub in itertools.combinations(sorted(tab, reverse=True), k):
            if 1.0 not in sub: continue
            x = lp(tab, sub)
            if x is None: continue
            ch = sum(x[i] * math.ceil(2 * TAU[q]) for i, q in enumerate(sub)) * SW / 3600
            rows.append(dict(rungs=list(sub), S_q=[float(v) for v in x], S_total=float(x.sum()), core_h=float(ch)))
    return sorted(rows, key=lambda r: r['core_h'])
beta, lnB = load(files); med = float(np.median(2 * lnB[beta == 1.0]))
tab = table(beta, lnB, med); rows = size(tab)
res = dict(note='DERIVED', P=P, ratio=RATIO, s_w=SW, tau=TAU, med2lnB=med, n_draws=int(len(beta)),
           frac_eps={str(q): {str(d): tab[q][d] for d in DECS} for q in tab}, rows=rows[:15])
halves = []
for h in (files[0::2], files[1::2]):
    b_, l_ = load(h); t_ = table(b_, l_, float(np.median(2 * l_[b_ == 1.0]))); r_ = size(t_)
    best_same = [r for r in r_ if r['rungs'] == rows[0]['rungs']]
    halves.append(dict(best=r_[0], same_rungs_as_full=best_same[0] if best_same else None))
res['halves'] = halves
json.dump(res, open(out, 'w'), indent=1, default=float)
for q in tab: print(q, ' '.join(f"{d}:{tab[q][d][0]:.3f}/{tab[q][d][1]:.2f}({tab[q][d][2]})" for d in DECS))
for r in rows[:8]: print(r['rungs'], 'S_q', [f'{v:.2e}' for v in r['S_q']], f"S_tot {r['S_total']:.2e} core-h {r['core_h']:.0f}")
for h in halves: print('half best', h['best']['rungs'], f"{h['best']['S_total']:.2e} {h['best']['core_h']:.0f}", 'same rungs', None if h['same_rungs_as_full'] is None else (f"{h['same_rungs_as_full']['S_total']:.2e}", round(h['same_rungs_as_full']['core_h'])))
