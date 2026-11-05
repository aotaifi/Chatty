"""m5p2x_plateau_gate.py: reads OUT/plateau.json (m5p2x_plateau.py), prints the numbers and ONE 'VERDICT PLATEAU ext1' line (thresholds only; the lead decides).  usage: python3 m5p2x_plateau_gate.py PLATEAU_JSON"""
import sys, json
d = json.load(open(sys.argv[1])); assert d.get('done'), 'plateau.json not done'; x = d['decision']
print(f"L_ho single global 1800 {d['L_ho_single'][0]:.4e} vs 2100 {d['L_ho_single'][1]:.4e}: rel {x['lho_single_rel_change']:.4f} (not gating)")
print(f"L_ho window means: A4 (1770..1800 vs 2070..2100) {x['lho_win4_rel_change']:.4f} (< 0.015: {x['cond_A4']});  A16 (1650..1800 vs 1950..2100) {x['lho_win16_rel_change']:.4f} (< 0.015: {x['cond_A16']})")
print(f"e(theta_1800) {x['e_1800']:.6e}  e(theta_2100) {x['e_2100']:.6e}  diff {x['e_diff_signed']:+.3e}  |diff| {x['e_abs_diff']:.3e} (< 1e-7: {x['cond_B']})")
if 'trend' in d: t = d['trend']; print(f"E_FN trend 1200 {t['e_1200']:.6e}, 1500 {t['e_1500']:.6e}, 1800 {t['e_1800']:.6e}, 2100 {t['e_2100']:.6e}; change per 100 steps {t['per_100_steps'][0]:+.3e}, {t['per_100_steps'][1]:+.3e}, {t['per_100_steps'][2]:+.3e}")
if 'r5_like' in d: r = d['r5_like']; print(f"r_5-like: theta_1800 {r['r_1800']:.4f}  theta_2100 {r['r_2100']:.4f}  tail average 1200..2100 {r['r_tail_avg']:.4f}")
print(f"VERDICT PLATEAU ext1: {'PLATEAU REACHED' if x['plateau_A4_B'] else 'NOT PLATEAUED'} (B E_FN {'ok' if x['cond_B'] else 'fails'}, A4 {'ok' if x['cond_A4'] else 'fails'}); with the literal 16-point window: {'PLATEAU REACHED' if x['plateau_A16_B'] else 'NOT PLATEAUED'}")
