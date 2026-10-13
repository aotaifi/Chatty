"""F1 summary: N_1/2 table, pre-registered rule-of-thumb fits log N_1/2 = log C - p log x (x = energy error per site,
or w_s per site), and the post-hoc Lanczos skewness rule N_1/2(L) ~ 2 kappa3^2/kappa2^3 (per-site cumulants), applied
also to the 8x8 guides (a8_skew.json).
  python a0f_summary.py a0f_dilution.json a8_skew.json OUT.json
"""
import sys, json
import numpy as np

R = json.load(open(sys.argv[1])); SK = json.load(open(sys.argv[2])); out = dict(table={}, fits={})
for g, r in R.items():
    k2, k3 = r['c2'] / 16, r['c3'] / 16
    out['table'][g] = dict(E_err_site=r['E_err_site'], w_s_copy=r['w_s'], w_s_site=r['w_s'] / 16,
                           skew_copy=r['c3'] / r['c2'] ** 1.5, skew_rule_L=2 * k3 ** 2 / k2 ** 3,
                           **{f'N_half_{s}': r[s]['N_half'] for s in ('K', 'L', 'Lc')},
                           **{f'ratio192_{s}': r[s]['ratio_N192'] for s in ('K', 'L', 'Lc')},
                           **{f'ratio64_{s}': r[s]['ratio_N64'] for s in ('K', 'L', 'Lc')})
for s in ('K', 'L', 'Lc'):
    for xk in ('E_err_site', 'w_s_site'):
        rows = [(t[xk], t[f'N_half_{s}']) for t in out['table'].values() if t[f'N_half_{s}'] is not None]
        x = np.log([a for a, _ in rows]); y = np.log([b for _, b in rows])
        p, c = np.polyfit(x, y, 1); dev = np.max(np.abs(y - (p * x + c)))
        out['fits'][f'{s}_{xk}'] = dict(n=len(rows), C=float(np.exp(c)), p=float(-p), max_dev_factor=float(np.exp(dev)),
                                         excluded=[g for g, t in out['table'].items() if t[f'N_half_{s}'] is None])
out['skew_rule_8x8'] = {b: dict(N_half_L_estimate=2 * v['kappa3'] ** 2 / v['kappa2'] ** 3,
                               N_half_L_trimmed=2 * (v['m3_trim999'] / 64) ** 2 / (v['m2_trim999'] / 64) ** 3)
                        for b, v in SK.items()}
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for g, t in out['table'].items():
    print(g, {k: (round(v, 4) if isinstance(v, float) and abs(v) < 10 else (round(v) if isinstance(v, float) else v)) for k, v in t.items()})
for k, v in out['fits'].items(): print(k, v)
print(out['skew_rule_8x8'])
