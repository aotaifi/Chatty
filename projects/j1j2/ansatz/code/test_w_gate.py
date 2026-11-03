"""gate for Test W: python3 test_w_gate.py OUTDIR -> one VERDICT line from w_result.json (done, controls, table check, KILL comparison)."""
import json, sys
r = json.load(open(sys.argv[1] + '/w_result.json')); k = r.get('kill', {}); c = r.get('controls', {}); p = r.get('part2_check', {})
ok = bool(r.get('done')) and c.get('self_kappa_bond_minmax') == [1.0, 1.0] and p.get('pass_1e_8', False)
print(f"VERDICT: done={r.get('done')} self_kappa_bond={c.get('self_kappa_bond_minmax')} perm_absmax_bond={c.get('perm_kappa_bond_absmax'):.3g} table_check_pass={p.get('pass_1e_8')} (max rel {p.get('max_rel_dev_amp')}) "
      f"KILL sum(<-9)={k.get('sum_lt_m9'):.4f} sum(>=-9)={k.get('sum_ge_m9'):.4f} near_node_KILLED={k.get('KILL_near_node_reshaping')} controls_ok={ok}")
