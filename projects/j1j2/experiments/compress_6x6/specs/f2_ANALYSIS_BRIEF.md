# F2 analysis brief, arm @ARM@ (local copy of the Paderborn H100 run @REM@; local dir @LOC@; window @WIN@, lambda_v @LV@, fit budget @FS@ s)
Question (README compress_6x6, "Test F2", pre-registered 2026-10-23 12:10): does fitting what fixed node uses (an extra loss lambda_v L_v on the log-ratios over
sign-violating bonds s_2(x) H_xy s_2(y) > 0, bonds sampled in proportion to w_xy = phi(x) phi(y) |H_xy|) lower the refit tax of the O2b loop?
One round-1 fit as in F1 (target log phi_1 = o2b_17431886/round_1/target_la.npy, warm start a_1 = o2b_17431886/round_0/theta.npy, same batches and optimiser; cx_f2fit.py = cx_f1fit.py
+ the L_v rows/acceptance/held-out term).  tax = E_FN(hop(a_2)) - E_FN(hop(phi_1)); reference hop(phi_1) = 4.951226e-6; O2b round-1 tax 5.93021e-6; F1 arm A (3x3, objective as O2b,
converged at 1070 steps) 5.60e-6; F1 arm B (5x5) 4.58e-6.  Arm C = window 3, arm D = window 5 (outer ring free); lambda_v set so lambda_v L_v = old loss at step 0 (held-out sets;
logged as F2.lambda_v in cx_f2.json), same lambda_v in C and D; the fit's own convergence rule (held-out TOTAL objective flat, acceptance >= 0.5, lam_rel below cap).
Controls (arm names): ctl1 = lambda_v 0, 3x3, 80 steps: tax 5.959e-6 (F1 arm A 80-step control f1_ctl1) within 2%.  lvchk = brute-force L_v (cx_f2_lvcheck.py): lv_check_pass true.  rC/rD = s/step.
Files: @LOC@/cx_f2.json: control {rel_hop_phi1, expect_tax, tax_rel_dev}, hop_phi1, hop_a2, tax, tax_vs_ref_hop_phi1,
  fit {steps, fit_sec, converged, reason, median_sec_per_step, L_ho_curve [(step, held-out TOTAL loss)], curve_parts [(step, L_old_ho, L_v_ho)], F2 {lambda_v, L_ho_old_step0, L_v_ho_step0, ...},
  err_by_decade, bond_error_total (target-weighted rms log-ratio error on violating bonds = sqrt(L_v over the whole table)), bond_error_warm_total, E_FN_zero_hop, E_FN_splice},
  stop (present only on CONTROL FAILED / FIT FAILED); @LOC@/round_1/{cx_t1s.json (hist, curve), fit.log, progress.txt, theta.npy}; la_table.npy stays on Paderborn.
Gate: no Monte Carlo chains (exact solves, one fit): quotable.py does not apply.  Read the numbers directly:
  python3 -c "import json;d=json.load(open('@LOC@/cx_f2.json'));print(d.get('stop'),d.get('control'),d.get('tax'),d.get('lv_check_pass'),d.get('fit',{}).get('steps'),d.get('fit',{}).get('median_sec_per_step'),d.get('fit',{}).get('curve_parts',[])[-3:])"
Decision (README "Test F2", arms C and D; tax_C, tax_D; min = min(tax_C, tax_D)):
  - min tax <= 3.0e-6: the objective is the lever; re-run O2b with it (round tax ~3e-6 puts the fixed point near tax / (1 - 0.34) ~ 4.5e-6, half the bar).
  - 3.0e-6 < min tax <= 4.0e-6: partial; combine with the 5x5 and test a larger lambda_v once.
  - min tax > 4.0e-6: the ansatz class is the limit; next is hidden-fermion rows or the conceptual angle's hop-sign idea.
  - control failed, or a fit that stopped on its budget without converging: report, decide with the PI.
  - Lead's prediction: C 3.5-4.5e-6, D 3-4e-6; ~45% partial, ~30% lever.
Read also: did L_v on the held-out bond set fall (curve_parts) while the old loss rose?  The worker reports the numbers to the lead only; the lead decides (projects/j1j2/PLAN.md).
