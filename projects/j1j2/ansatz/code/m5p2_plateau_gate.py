"""m5p2_plateau_gate.py: mechanical reading of OUT/plateau.json (written by m5p2_plateau.py) against the registered plateau rule of PLAN_pfaffian.md "M5' rerun".
  python3 m5p2_plateau_gate.py PLATEAU_JSON   prints the numbers and ONE verdict line 'VERDICT PLATEAU: ...' (thresholds only; the lead decides).
  Plateau iff (A) |L_ho(1200) - L_ho(1500)|/L_ho(1500) < 1.5% AND (B) |e(theta_1200) - e(theta_1500)| < 0.1e-6."""
import sys, json
d = json.load(open(sys.argv[1])); assert d.get('done'), 'plateau.json not done'
x = d['decision']; s = d['steps']; L = d['L_ho']; m = d['L_ho_window_means']
print(f"L_ho step {s[0]} {L[str(s[0])]:.4e}, step {s[1]} {L[str(s[1])]:.4e}: rel change {x['lho_rel_change']:.4f} (< 0.015: {x['lho_cond_lt_1p5pct']}); window means {m[0]:.4e} vs {m[1]:.4e} (rel {x['lho_window_mean_rel_change']:.4f}, not the condition)")
print(f"e(theta_{s[0]}) {x['e_1200']:.6e}  e(theta_{s[1]}) {x['e_1500']:.6e}  diff {x['e_diff_signed']:+.3e}  |diff| {x['e_abs_diff']:.3e} (< 1e-7: {x['e_cond_lt_0p1e6']})")
if 'r5_like' in d: r = d['r5_like']; print(f"r_5-like: theta_1200 {r['r_1200']:.4f}  theta_1500 {r['r_1500']:.4f}  tail-average theta_5 {r['r_tail_avg']:.4f}")
print(f"VERDICT PLATEAU: {'PLATEAU REACHED' if x['plateau'] else 'NOT PLATEAUED'} (L_ho {'ok' if x['lho_cond_lt_1p5pct'] else 'fails'}, E_FN {'ok' if x['e_cond_lt_0p1e6'] else 'fails'})")
