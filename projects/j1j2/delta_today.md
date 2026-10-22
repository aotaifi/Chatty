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
- 2026-10-21 T1s: launcher + specs read; Amendment T1s-1 (fd21d17); go given with sha256 code-drift guard.
2026-10-21 18:44 VERDICT job 17412547: COMPLETED, none. Bash was denied (don't-ask mode), so I did not run quotable.py or read cx_t1s.json, and no plot was made. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s_vit_17412547/ANALYSIS.md]
- 2026-10-21 T1s arm (a) ViT 17412547 done: converged (acc 0.98, moving 4.4e-2), decade -11 rms 0.124, bulk 0.018-0.023, E_FN 8.18e-5 -> ViT reading: representability limit (>=0.1 converged). Others pending/running.
2026-10-21 19:33 VERDICT job 17412548: COMPLETED, not run. Bash was denied in this session, so `quotable.py` and the plot did not run. I read `cx_t1s.json` directly with Read. The brief says `quotable.py` does not apply to this job anyway. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s_vitns_17412548/ANALYSIS.md]
2026-10-21 20:31 VERDICT job 17412549: COMPLETED, gate not run. Bash was denied (don't-ask mode), so `quotable.py` and the plot did not run. Numbers below were read from `cx_t1s.json`. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s_pp_17412549/ANALYSIS.md]
2026-10-21 22:45 VERDICT job 17412550: COMPLETED, not produced. Bash was denied in this session, so I could not run quotable.py or the brief's gate command. The numbers below are read directly from cx_t1s.json with Grep/Read. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s_ppbf_17412550/ANALYSIS.md]
2026-10-21 22:46 T1s worker: 4 jobs COMPLETED (exit 0, sha256 guard matched): 17412547 vit, 17412548 vit_nostrat (control 1), 17412549 pp, 17412550 ppbf; runs /project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s_*_174125{47,48,49,50}/cx_t1s.json; code cx_t1s.py at cd4a444 (sha256 842c333c...). Converged (amendment T1s-1): vit yes, vit_nostrat no (budget), pp yes, ppbf no (budget). E_FN: 8.18e-5, 8.93e-5, 2.82e-5, 2.96e-5. MaxRSS 10.7-11.7 GB of 38 GB.
- 2026-10-21 T1s all 4 verified: ViT conv -11=0.124 (repr. limit by rule; noise-floor caveat), PP conv -11=0.073 E_FN 2.82e-5 (inconclusive), PPBF not conv 0.074 E_FN 2.96e-5; no arm passes 1e-5.
- 2026-10-21 T1s-2 launcher/specs read; budgets raised (amendment T1s2-1, fd0433f) so convergence is reachable; go given.
2026-10-22 00:48 VERDICT job 17418856: COMPLETED, none. Bash was denied (don't-ask mode), so I could not run the brief's gate command or `quotable.py`. No numbers are quoted. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s2_pp_17418856/ANALYSIS.md]
2026-10-22 01:51 VERDICT job 17418858: COMPLETED, none. The gate did not run, and I have no numbers. Bash was denied ("don't ask" mode), so I could not run quotable.py or read cx_t1s.json. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s2_vit_17418858/ANALYSIS.md]
2026-10-22 04:39 VERDICT job 17418857: COMPLETED, none. The gate did not run, because Bash was denied ("don't ask" mode) on my first call. That call was a `--help` check plus a read of `cx_t1s.json`. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s2_ppbf_17418857/ANALYSIS.md]
2026-10-22 04:40 T1s-2 worker: 3 jobs COMPLETED (exit 0, sha guard matched): 17418856 pp, 17418857 ppbf, 17418858 vit (full A40); runs /project/theorie/a/A.Otaifi/chatty_compress6/runs/t1s2_*_1741885{6,7,8}/cx_t1s.json; code at 75310bd. Converged (T1s-1 + tau floor): none (all out of budget). r(-11): pp 0.0695, ppbf 0.0661, vit 0.1133. E_FN 2.61e-5 / 2.42e-5 / 6.89e-5; splice E_FN 7.94e-6 / 6.03e-6 / 8.17e-6. MaxRSS 10.6-12.4 GB.
- 2026-10-22 G1 code+launcher read; go with 2 h wall (~52 min compute, over the 0.5 slice-h pre-reg estimate).
2026-10-22 06:45 VERDICT job 17426232: TIMEOUT [/project/theorie/a/A.Otaifi/chatty_compress6/runs/g1_17426232/ANALYSIS.md]
- 2026-10-22 G1 17426232 PASS (verified): PP+bf T1s-2 one-hop E_FN 7.06e-6 < 1e-5 < refs (RBM+PP 1-hop 1.29e-5); timeout after 4/6 bases (refs only). Next: T2 design.
- 2026-10-22 O1/O2 pre-registered (ee4d488); O1 worker launched; Fable O2 design review launched (adversarial, before O2 runs).
2026-10-22 09:53 VERDICT job 1: DISAPPEARED [/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/b0c3a6ee-1f88-4816-9c14-ae183c72ff38/scratchpad/wt/r1/ANALYSIS.md]
2026-10-22 09:53 VERDICT job 1: DISAPPEARED [/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/b0c3a6ee-1f88-4816-9c14-ae183c72ff38/scratchpad/wt/r2/ANALYSIS.md]
2026-10-22 09:54 VERDICT job 1: DISAPPEARED [/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/b0c3a6ee-1f88-4816-9c14-ae183c72ff38/scratchpad/wt/r3/ANALYSIS.md]
- 2026-10-22 Fable O2 review: fit tax 6.1e-6 leaves ~3e-6 for the loop; noiseless SHDMC rises to wrong node (Caffarel 2609.01301); amendment O2-1 (80b39ab): O2a fit-free chain first, RISE verdict, fit-tax viability criterion. O2a waits for O1.
2026-10-22 10:10 VERDICT job 17430543: COMPLETED, gate not run (exact solves, no MC chains, quotable.py does not apply). Bash was denied here, so I read cx_o1.json and progress.txt directly. Control ok=True: ppbf E_FN 7.05625e-6 vs ref 7.056e-6 (rel 3.6e-5). [/project/theorie/a/A.Otaifi/chatty_compress6/runs/o1_17430543/ANALYSIS.md]
- 2026-10-22 O1 17430543 PASS (verified): alpha by VMC energy gives ppbf E_FN 7.11e-6 vs oracle 7.06e-6. Next O2a.
- 2026-10-22 O2a 17430739 submitted (2cf5250); s2/s0 global-sign swap noted in expelled split labels.
- 2026-10-22 O2a 17430739 verified: exact hop+FN chain 2.37e-5 -> 8.1e-6 -> 4.5e-6 -> 2.9e-6 -> 2.1e-6 -> 1.7e-6 (c 0.32-0.64); map fast; next O2b.
2026-10-22 10:27 VERDICT job 17430739: COMPLETED, no gate verdict. Bash was denied, so quotable.py did not run (the brief also says it does not apply to exact solves). The numbers below come straight from cx_o2a.json and progress.txt. [/project/theorie/a/A.Otaifi/chatty_compress6/runs/o2a_17430739/ANALYSIS.md]
- 2026-10-22 O2b launcher read; amendment O2b-1 (f1a6132): round-0 fit 10800 s; go given (~14 A40-h).
- 2026-10-22 migrate: PLAN.md rewritten to current state; O2b being submitted by worker; session to be cleared after O2b is in the queue.
- 2026-10-22 O2b job 17431886 submitted (PENDING, wall 19:15, setsid watcher).
