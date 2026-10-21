# delta_today (append only)
- 2026-10-21 refresh2_17409866 seed 1 done: E_FN 9.22e-5 (seed 0 9.16e-5); refit FAIL; fitted-net flip labels recall 5e-7.
- 2026-10-21 t0_17410220 read: pre-registered kill rule fires (eps PP/ViT 0.78).
- 2026-10-21 T0b 17410565 submitted by a worker; launcher read by the lead after submission (should have been before).
- 2026-10-21 step-back: tail networks -> verdict switch; next cheapest test G0 (gamma-extrapolated FN).
- 2026-10-21 lead wrongly relabelled the T0 kill as "inconclusive" (0e1929b), reverted to the pre-registered verdict.
- 2026-10-21 G0 pre-registered (27507c4); worker building cx_g0.py. Another session (ge25sup identity) pushed T0-phys da596a2 and submitted 17410698.
- 2026-10-21 T0b 17410565 read and verified from cx_t0b.json: eps tracks dE (slope 0.72); B2/ViT 1.507 direct, 1.32 at equal dE; T0 kill stands; T1 not run unless the PI asks.
- 2026-10-21 T0-phys 17410698 read: one-hop ViT/PP 1.23 (inconclusive by the pre-registered 1.2/2 bar; RBM+PP one-hop 1.29e-5); stripe/Neel ratio 5.7x bulk on X_0.1 (confirmed via moment ratio, not plaquettes); J1 Marshall violation 0.40 on X_0.1 vs 0.013 bulk; W_2 92% Marshall-wrong.
- 2026-10-21 G0 launcher + cx_g0.py read by the lead before sbatch (operator = pre-registration; stoquastic for gamma >= 0); go given.
- 2026-10-21 Fable training audit: refit NOT converged (28/200 accepted, lam capped from step 30; verified); eps~sqrt(L2) is the L2-objective signature; Pfaffian route back to inconclusive; next T1s.
2026-10-21 16:55 VERDICT job 17410913: STALLED [/project/theorie/a/A.Otaifi/chatty_compress6/runs/g0_17410913/ANALYSIS.md]
- 2026-10-21 G0 17410913: control (i) pass (1.0185827e-4); control (ii) LOBPCG stalled >28 min from noisy start, worker scancelled its own job; controls (ii)/(iii) switched to eigenvector-residual form, resubmitting.
2026-10-21 17:23 VERDICT job 17411369: COMPLETED, none. The gate did not run, because Bash is denied in this session (don't-ask mode), so `quotable.py` and the plot could not be produced. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/g0_17411369/ANALYSIS.md]
- 2026-10-21 G0 17411369 FAIL (verified): min <H>_phi^gamma at gamma=0 for all bases, ViT 7.57e-5 = 26% of gap; controls pass.
