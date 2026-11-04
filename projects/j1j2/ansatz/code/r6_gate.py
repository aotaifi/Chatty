"""r6_gate.py: mechanical reading of pfaffian_r6/readout/readout.json against PLAN_pfaffian.md "Round 6 two-PP + parameter-matched control" (the lead decides; thresholds only).
R6 (two-PP round 6 from pp2_s1's theta_5 to T_6): r_6 = (8.3319e-6 - e(theta_6)) / 5.0735e-6.   KILL: r_6 <= 0.1 (the second PP saturates like the first).
M5 (single PP + active RBM alpha 72, round 5 from theta_4 to T_5): r_5 = (1.2184484e-5 - e(theta_5^M5)) / 0.7817e-5.
  r_5_own = (e_start - e(theta_5^M5)) / (e_start - 4.367093186728886e-6), e_start = bare e of M5's initial theta (lead 2026-11-03, M5 at rbm_init_sigma 3e-3).
  "structure matters" iff r_5(two-PP seed 1) - r_5_own(M5) > 0.15 (lead 2026-11-03: the rule applies to r_5_own), r_5(two-PP seed 1) from pfaffian_r5/readout/readout.json (0.4929; PLAN rounds to 0.34);
  otherwise the round-5 gain is attributed to parameter count.
Predictions (not gates): r_6 0.2-0.4 (bare 6.3-7.3e-6), aug 2.5-2.9e-6; r_5(M5) 0.05-0.15, aug 3.9-4.2e-6.
Also checks: both controls, fit steps == 1500 per arm, theta_R cluster vs local.   usage: python3 r6_gate.py [READOUT_JSON]"""
import sys, json
RJ = sys.argv[1] if len(sys.argv) > 1 else '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6/readout/readout.json'
R5J = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/readout/readout.json'
d = json.load(open(RJ)); assert d.get('done'), 'readout not done'; assert d['control']['pass_'] and d['control_theta5']['pass_'], 'control failed'
V = d['verdict_numbers']; arms = d['arms']
r5pp = json.load(open(R5J))['verdict_numbers']['pp2_s1']['r_5']
print(f"controls: e(theta_4) rel {d['control']['rel']:.1e}, e(theta_5 pp2_s1) rel {d['control_theta5']['rel']:.1e}; arms read: {' '.join(arms)}")
RN = {'r6': 'r_6', 'm5': 'r_5'}; PRED = {'r6': ((0.2, 0.4), (2.5e-6, 2.9e-6)), 'm5': ((0.05, 0.15), (3.9e-6, 4.2e-6))}
for a in arms:
    fit = json.load(open(f"{d['dirs'][a][0]}/cx_r1.json")); steps = fit.get('fit', {}).get('steps'); th = d['theta'][a]
    shaok = th['theta_R_cluster_sha256'] == th['theta_R_local_sha256'] or th['max_abs_diff_cluster_vs_local'] < 1e-12
    r = V[a][RN[a]]; ea = V[a].get('e_aug', float('nan')); (rl, rh), (al, ah) = PRED[a]
    print(f"  {a:3s} [{d['cluster'][a]}] npar {th['npar']}  steps {steps}{'' if steps == 1500 else ' (NOT 1500)'}  e {V[a]['e']:.6e}  {RN[a]} {r:.4f} (pred {rl}-{rh}: {'in' if rl <= r <= rh else 'MISSED ' + ('HIGH' if r > rh else 'LOW')})"
          f"  e_aug {ea:.6e} (c* {V[a].get('cstar')}, {d['tables'][a].get('scan_status')}; pred {al:.1e}-{ah:.1e}: {'in' if al <= ea <= ah else 'MISSED ' + ('HIGH' if ea > ah else 'LOW')})"
          f"  theta_R sha {'OK' if shaok else 'MISMATCH'}" + (f"  lambda(theta_6) {th['lambda_thetaR']}" if 'lambda_thetaR' in th else '')
          + (f"  RBM rms {th['rbm_block_rms_thetaR']:.3e}" if 'rbm_block_rms_thetaR' in th else '')
          + (f"  e_start {V[a]['e_start']:.6e} r_5_own {V[a]['r_5_own']:.4f}" if 'r_5_own' in V[a] else ''))
out = []
if 'r6' in V:
    out.append(f"R6 KILL {'YES (r_6 <= 0.1: the second PP saturates like the first)' if V['r6']['r_6'] <= 0.1 else 'no'} (r_6 {V['r6']['r_6']:.4f})")
else:
    out.append('R6 missing')
if 'm5' in V:
    ro = V['m5']['r_5_own']; dlt = r5pp - ro; lo, hi = sorted((0.34, r5pp - 0.15)); edge = lo <= ro <= hi
    out.append(f"M5 {'STRUCTURE MATTERS' if dlt > 0.15 else 'PARAMETER COUNT (round-5 gain attributed to parameter count)'} (r_5(pp2_s1) {r5pp:.4f} - r_5_own(M5) {ro:.4f} = {dlt:+.4f}, rule > 0.15; r_5 vs theta_4 {V['m5']['r_5']:.4f}"
               + ('; NOTE r_5_own(M5) lies between 0.34 (PLAN rounding) and r_5(pp2_s1) - 0.15: the two readings differ, the lead decides' if edge else '') + ')')
else:
    out.append('M5 missing')
print('VERDICT R6: ' + '; '.join(out))
