"""pool8_sizing.py: DERIVED sizing of the 8x8 pool from the measured tables (pool8_measure.py jsons).  Usage: python pool8_sizing.py OUT.json P ratio m_g1.json m_g3.json m_g4.json ...
ESS_d(S) = 0.8 sum_q S_q f_{d,q} eps_{d,q} >= ratio P for every fitted decade d in +2..-6 (0.8 = training share after the 20% held-out hash).  Per rung subset (must contain beta = 1, S_1 >= 2e4 for the median): LP over S_q.
Cost (a) independent chains per rung (no swaps): sum_q S_q thin_q s_w;   (b) one PT ladder of R rungs (one draw per rung per thin sweeps): max_q S_q thin R s_w,  thin = ceil(2 max tau over the ladder)."""
import sys, json, itertools, math
import numpy as np
from scipy.optimize import linprog
out, P, ratio, files = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
DECS = [str(d) for d in range(2, -7, -1)]; tab = {}; tau = {}; sw = []
for f in files:
    r = json.load(open(f)); sw.append(r['s_per_sweep_per_walker_core'])
    for b, t in r['sizing']['frac_eps'].items():
        if b in tab and r['E_tot'] <= tab[b][1]: continue
        tab[b] = (t, r['E_tot'] * r['T_recorded']); tau[b] = r['sizing']['tau_per_rung'][b]
s_w = float(np.mean(sw)); need = ratio * P; res = dict(note='DERIVED', P=P, ratio=ratio, s_per_sweep_walker=s_w, rows=[])
betas = sorted(tab, key=float, reverse=True)
def lp(sub):
    A = -0.8 * np.array([[tab[b][0][d][0] * tab[b][0][d][1] for b in sub] for d in DECS]); keep = [i for i in range(len(DECS)) if np.any(A[i] < 0)]
    if len(keep) < len(DECS): return None
    o = linprog(np.ones(len(sub)), A_ub=A, b_ub=-need * np.ones(len(DECS)), bounds=[((2e4 if b == '1.0' else 0), None) for b in sub]); return o.x if o.success else None
for k in (1, 2, 3, 4):
    for sub in itertools.combinations(betas, k):
        if '1.0' not in sub: continue                        # pool_fit's decade median needs a beta = 1 chain (S_1 >= 2e4 enforced)
        x = lp(sub)
        if x is None: continue
        thin = math.ceil(2 * max(tau[b] for b in sub)); ind = sum(x[i] * math.ceil(2 * tau[b]) for i, b in enumerate(sub)) * s_w / 3600
        pt = x.max() * thin * len(sub) * s_w / 3600
        res['rows'].append(dict(rungs=list(map(float, sub)), S_q=[float(v) for v in x], S_total=float(x.sum()), core_h_independent=float(ind), core_h_PT_same_rungs=float(pt)))
res['rows'].sort(key=lambda r: r['core_h_independent'])
json.dump(res, open(out, 'w'), indent=1)
for r in res['rows'][:12]: print(r['rungs'], 'S_q', [f'{v:.2e}' for v in r['S_q']], f"S_tot {r['S_total']:.2e}  core-h indep {r['core_h_independent']:.0f}  PT {r['core_h_PT_same_rungs']:.0f}")
print('s_w', s_w)
