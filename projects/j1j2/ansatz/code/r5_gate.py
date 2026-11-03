"""r5_gate.py: mechanical reading of pfaffian_r5/readout/readout.json against PLAN_pfaffian.md "Two-PP class (d)" (the lead decides; this applies the stated thresholds only).
r_5 = (1.2184484e-5 - e(theta_5)) / 0.7817e-5 per arm (theta_5 = mean of the 7 iterates 600..1500).
KILL (d): both pp2 seeds r_5 <= 0.05 AND e_aug >= 4.2e-6 (each seed).  NOT ATTRIBUTABLE: r_5(pp2_sK) - r_5(ctl_sK) < 0.03 for K = 1 or K = 2 (seed-matched).
The COST gate (CPU timing at 10x10, KILL of deployment > 20x) is not read here.  Also checks: control, fit steps == 1500 per arm, theta_5 cluster vs local.   usage: python3 r5_gate.py [READOUT_JSON]"""
import sys, json
RJ = sys.argv[1] if len(sys.argv) > 1 else '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/readout/readout.json'
Q = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'
d = json.load(open(RJ)); assert d.get('done'), 'readout not done'; assert d['control']['pass_'], 'control failed'
V = d['verdict_numbers']; arms = d['arms']
print(f"control e(theta_4) rel {d['control']['rel']:.1e}; arms read: {' '.join(arms)}")
for a in arms:
    fit = json.load(open(f"{d['dirs'][a][0]}/cx_r1.json")); steps = fit.get('fit', {}).get('steps'); th = d['theta'][a]
    shaok = th['theta_5_cluster_sha256'] == th['theta_5_local_sha256'] or th['max_abs_diff_cluster_vs_local'] < 1e-12
    print(f"  {a:7s} [{d['cluster'][a]}] npar {th['npar']}  steps {steps}{'' if steps == 1500 else ' (NOT 1500)'}  e {V[a]['e']:.6e}  r_5 {V[a]['r_5']:.4f}  e_aug {V[a].get('e_aug', float('nan')):.6e} (c* {V[a].get('cstar')}, "
          f"{d['tables'][a].get('scan_status')})  theta_5 sha {'OK' if shaok else 'MISMATCH'}" + (f"  lambda(theta_5) {th['lambda_theta5']}" if 'lambda_theta5' in th else ''))
pp = [a for a in ('pp2_s1', 'pp2_s2') if a in V]; missing = [a for a in ('pp2_s1', 'pp2_s2', 'ctl_s1', 'ctl_s2') if a not in V]
kill = len(pp) == 2 and all(V[a]['r_5'] <= 0.05 and V[a].get('e_aug', 0) >= 4.2e-6 for a in pp)
diffs = {k: V[f'pp2_s{k}']['r_5'] - V[f'ctl_s{k}']['r_5'] for k in (1, 2) if f'pp2_s{k}' in V and f'ctl_s{k}' in V}
nattr = any(v < 0.03 for v in diffs.values())
print(f"VERDICT R5: KILL (d) {'YES' if kill else 'no'}{' (incomplete: ' + ' '.join(missing) + ' missing)' if missing else ''}; "
      f"r_5(pp2) - r_5(ctl) {', '.join(f's{k} {v:+.4f}' for k, v in diffs.items())} -> {'NOT ATTRIBUTABLE' if nattr else 'attributable'}"
      f"{'' if len(diffs) == 2 else ' (only ' + str(len(diffs)) + ' seed pair(s) complete)'}")
