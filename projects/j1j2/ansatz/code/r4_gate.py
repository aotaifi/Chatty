"""r4_gate.py: mechanical reading of pfaffian_r4/readout/readout.json against PLAN_pfaffian.md "Round 4" (the lead decides; this applies the stated thresholds only).
r_4 = (1.2522e-5 - e(theta_4)) / 0.7878e-5; r_4 >= 0.178 kills the DECLINE (H_q); bare drop e(bar) - e(theta_4) <= 0.5e-6 kills PROGRESS; q = r_4 / 0.178.
Also checks the fit: steps == 1500 in the winner's cx_r1.json and the theta_4 sha agreement.   usage: python3 r4_gate.py [READOUT_JSON]"""
import sys, json
RJ = sys.argv[1] if len(sys.argv) > 1 else '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/readout/readout.json'
d = json.load(open(RJ)); assert d.get('done'), 'readout not done'; assert d['control']['pass_'], 'control failed'
T = d['tables']; e4 = T['theta_4']['e']; eb = 1.2521612014025783e-05
r4 = (1.2522e-5 - e4) / 0.7878e-5; drop = eb - e4; q = r4 / 0.178
fit = json.load(open(f"{d['fit_dir']}/cx_r1.json")); steps = fit.get('fit', {}).get('steps')
th = d['theta']; shaok = th['theta_4_cluster_sha256'] == th['theta_4_local_sha256'] or th['max_abs_diff_cluster_vs_local'] < 1e-12
print(f"winner {d['winner']}  fit steps {steps}  control rel {d['control']['rel']:.1e}  theta_4 sha cluster {th['theta_4_cluster_sha256'][:12]} local {th['theta_4_local_sha256'][:12]} (max|diff| {th['max_abs_diff_cluster_vs_local']:.1e})")
for k in ('theta_4', 'final_1500') + tuple(f'ckpt_{s}' for s in (600, 750, 900, 1050, 1200, 1350)):
    if k in T: print(f"  {k:11s} e {T[k]['e']:.6e}" + (f"   c* {T[k]['cstar']} e_aug {T[k]['e_aug_min_site']:.6e} ({T[k]['scan_status']})" if 'cstar' in T[k] else ''))
dec = 'KILLS the decline (H_q)' if r4 >= 0.178 else 'does not kill the decline'
prg = 'KILLS progress' if drop <= 0.5e-6 else 'does not kill progress'
print(f"VERDICT R4: e(theta_4) {e4:.6e}  r_4 {r4:.4f}  q {q:.2f}  bare drop {drop*1e6:.4f}e-6  -> {dec}; {prg}; fit steps {'OK' if steps == 1500 else 'NOT 1500: ' + str(steps)}; theta_4 sha {'OK' if shaok else 'MISMATCH'}")
