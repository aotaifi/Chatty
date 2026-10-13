"""Analysis of an 8x8 FN-DMC run (dc_fn.py, PREREG E2 + amendment 1): protocol window [0.8, 2.4], late-window transient
check ([2.4, 3.2] minus [0.8, 2.4], paired per population), beta profile in bins of 0.4.
  python dc_fn_analyze.py fn.npz OUT.json
"""
import sys, json
import numpy as np
import dc_fn as F

z = np.load(sys.argv[1]); E, T = z['Eref'], z['tau']; N = 64
E0, SE0 = -0.4988474, 1.06e-5           # psi_P plain VMC (results/loop_8x8 FU1a)
E1, SE1 = -0.4989687, 1.41e-5           # psi_1 direct VMC pooled (REVIEW_lanczos8x8)


def win(a, b):
    pm = F.pop_means(E, T, a, b, N)
    return pm, float(pm.mean()), float(pm.std(ddof=1) / np.sqrt(len(pm)))


pp, mp, sp = win(0.8, 2.4)
res = dict(npop=len(pp), protocol=dict(E=mp, se=sp, sd_pop=float(pp.std(ddof=1))))
bmax = float(np.cumsum(T, 1).max())
if bmax >= 3.2 - 1e-9:
    pl, ml, sl = win(2.4, 3.2); d = pl - pp
    res['late_2.4_3.2'] = dict(E=ml, se=sl)
    res['late_minus_protocol'] = dict(d=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
    pq, mq, sq = win(1.6, 3.2); res['late_1.6_3.2'] = dict(E=mq, se=sq)
    res['transient_detected'] = bool(res['late_minus_protocol']['d'] < -2 * res['late_minus_protocol']['se'])
res['profile'] = {f'{a:.1f}-{a + 0.4:.1f}': win(a, a + 0.4)[1:] for a in np.arange(0.0, min(bmax, 3.2) - 0.39, 0.4)}
q = res['late_1.6_3.2'] if res.get('transient_detected') else res['protocol']
res['quoted'] = q
res['gain_vs_psiP'] = dict(d=q['E'] - E0, se=float(np.hypot(q['se'], SE0)), resolved=bool(q['E'] - E0 + 2 * np.hypot(q['se'], SE0) < 0))
res['vs_psi1'] = dict(d=q['E'] - E1, se=float(np.hypot(q['se'], SE1)), new_best_bound=bool(q['E'] + 2 * q['se'] < E1))
res['vs_RBMPP'] = dict(d=q['E'] + 0.498886, below_2sigma=bool(q['E'] + 2 * q['se'] < -0.498886))
json.dump(res, open(sys.argv[2], 'w'), indent=1); print(json.dumps(res, indent=1))
