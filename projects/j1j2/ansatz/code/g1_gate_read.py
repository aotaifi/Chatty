"""Gate reader for a Test G gate run (cx_g1fit.py --mode gate).  usage: python3 g1_gate_read.py RUNDIR/cx_g1.json
Prints one VERDICT-ready line: control 1, the five per-site energies and differences, MM decrease, steps, deep-draw counts."""
import json, sys
d = json.load(open(sys.argv[1])); c = d.get('controls', {}).get('c1_tax0', {}); g = d.get('gate', {})
if d.get('mode') == 'probe2':
    q = d.get('probe2', {})
    if not d.get('done'): print(f"GATE_FAIL probe2 not done, stop={d.get('stop')}"); sys.exit(0)
    print(f"{'GATE_OK' if q['pass_70pct'] and q['pass_bound'] else 'GATE_FAIL'} probe2 steps={q['steps']} accepted={q['accepted']} "
          f"lower_by_2sigma={q['sig2_lower']} ({q['frac_sig2']:.2f}, need 0.70) MM bound dE_FN/site={q['MM_bound_dE_site']:.3e} +- {q['MM_bound_se_site']:.1e} "
          f"s/step={q['sec_per_step_median']:.1f}"); sys.exit(0)
if d.get('stop') or not d.get('done'):
    print(f"GATE_FAIL done={d.get('done')} stop={d.get('stop')} c1_ok={c.get('ok')} c1_rel={c.get('rel')}"); sys.exit(0)
ok = c.get('ok') and g.get('MM_decrease') is not None
print(f"{'GATE_OK' if ok else 'GATE_FAIL'} c1_rel={c.get('rel'):.1e} E_FN[A0]={g.get('E_FN_A0'):.6e} E_FN[chi0]={g.get('E_FN_chi0'):.6e} "
      f"E_FN[A1]={g.get('E_FN_A1'):.6e} Delta_0={g.get('Delta_0'):.4e} tau_0={g.get('tau_0'):.4e} tau0/Delta0={g.get('tau0_over_Delta0'):.3f} "
      f"MM_decrease={g.get('MM_decrease')} steps={g.get('inner_steps')} accepted={g.get('inner_accepted')} deep={g.get('coverage_total')}")
