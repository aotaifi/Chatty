# F1 analysis brief, arm @ARM@ (local copy of the Paderborn H100 run @REM@; local dir @LOC@; window @WIN@, fit budget @FS@ s)
Question (README compress_6x6, "Test F1", pre-registered 2026-10-22 21:40): is the refit tax of the O2b loop (~6e-6 in every round) capacity or under-training?
One round-1 fit as O2b ran it: target log phi_1 = o2b_17431886/round_1/target_la.npy, warm start a_1 = o2b_17431886/round_0/theta.npy, same objective and batches
(cx_f1fit.py = copy of cx_t1s.py + options bf_window / freeze_outer).  tax = E_FN(hop(a_2)) - E_FN(hop(phi_1)); reference hop(phi_1) = 4.951226e-6; O2b round-1 tax
5.93021e-6 (80 steps, 3x3).  Arm A = window 3, arm B = window 5 (outer ring free), both fit_sec = 10800.  Controls: ctl1 (window 3, 80 steps) must give 5.93e-6 within 5%;
ctl2 (window 5, outer ring frozen, 80 steps) must equal ctl1 (tax 1e-3 rel, log-amplitudes on 4096 orbits 1e-10, python3 cx_f1_compare.py).
Files: @LOC@/cx_f1.json: control {rel_hop_phi1, rel_hop_o2b_a2, expect_tax, tax_rel_dev, tax_within_5pct}, hop_phi1, hop_a2 {EFN, alpha_V, ...}, tax, tax_vs_ref_hop_phi1,
  fit {steps, fit_sec, converged, reason, median_sec_per_step, L_ho_curve [(step, held-out loss)], L_ho_step0, err_by_decade {d: mean, rms, rms_heldout}, bond_error_total,
  E_FN_zero_hop, E_FN_splice}, stop (present only on CONTROL FAILED / FIT FAILED); @LOC@/round_1/{cx_t1s.json (hist, curve), fit.log, progress.txt, theta.npy}; la_table.npy stays on Paderborn.
Gate: no Monte Carlo chains (exact solves, one fit): quotable.py does not apply.  Read the numbers directly:
  python3 -c "import json;d=json.load(open('@LOC@/cx_f1.json'));print(d.get('stop'),d['control'],d.get('tax'),d['fit']['steps'],d['fit']['median_sec_per_step'],d['fit']['L_ho_curve'][-3:])"
Decision (README, for arms A and B; tax_A, tax_B; 'steps' and the held-out curve say whether the fit was still moving):
  - tax_A <= 3.5e-6: under-training dominates; the lever is the per-round optimiser and budget (cost question).
  - tax_A > 4.5e-6 and tax_B <= 0.6 x tax_A: capacity is the lever; re-run O2b with 5x5.
  - tax_A > 4.5e-6 and tax_B > 4.5e-6 (both): neither; the limit is the fit objective or the ansatz class. Next: fit the quantity E_FN uses (ratios on sign-violating bonds)
    or a different class (hidden-fermion rows).
  - anything else (e.g. A in 3.5-4.5e-6, or B between 0.6 and 1 x A, or a control failed): report, decide with the PI.
  - Lead's prediction: A 4.5-5.5e-6, B 0.6-0.8 x A; about 40% on 'both > 4.5e-6'.
The worker reports the numbers to the lead only; the lead decides (projects/j1j2/PLAN.md).
