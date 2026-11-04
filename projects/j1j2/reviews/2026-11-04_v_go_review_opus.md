# V go/no-go: adversarial second opinion (worker, 2026-11-04; read-only; own CPU check in /scratch/a/A.Otaifi/astra/v_go_opus/)

The claim the go rests on: the round-5 storage step keeps ~0.94 of its exact-table r when it is fitted from 4e5 samples
(r_V 0.4636 vs r_5 0.4929; pfaffian_v/readout/readout.json "verdict_numbers"), so it will also hold at 8x8.
The PASS stands. "Survives sampling" is much narrower than it sounds, and the 8x8 bar is wrong.

## 1. Does V support "storage survives sampling at 6x6"?
What V actually sampled: only which configurations are covered. The draws are exact i.i.d. from the table CDF (pool_fit.py:84-89, Exact6.draw_fn).
The partners come from exact B_4 ratios. The labels are read from the target table (pool_fit.py:104-113). The sign is the exact s_G.
None of these parts was tested: an MCMC/tempered sampler, a stored sign, sampled E' and f0, an 8x8 model (pool_fit.py:28: "an 8x8 model is not in this file").
Checks I ran myself (gap.py -> gap.log/gap.json; full 15.8M-orbit tables; kappa = per-decade stored fraction, n-weighted, as in Test W):
- Labels from a table: harmless in itself (ln T is a local function of B_4, s_G, E', f0, computable per sample at 8x8); the risk is in s_G and E' (Q2).
- r_5 like-for-like: YES. T_5^b1 - T_5(old) is exactly 0 on every orbit in decades -5..-13 (the fitted band).
  2/16/91 orbits change in -14/-15/-16, n|B_4|^2 weight 5.5e-11. The kappa tables of both fits against T_old and T_b1 are identical to 3 digits. The confound recorded in PLAN_pfaffian.md:1478-1479 is void.
- Leakage: it cannot inflate r_V, because grading is the exact E_FN of the full table. Minor: 73,560 held-out orbits appear as training partners y, because the held-out mask is on x only (pool_fit.py:151, :309).
  Generalisation, measured directly: on orbits that never entered the pool (98-99% of every deep decade), V's kappa is
  0.253/0.273/0.299/0.335/0.374/0.355 in decades -9..-14.
  The exact-table fit pp2_s1 gives 0.252/0.273/0.294/0.323/0.351/0.330 on the same orbits. So V stores AS MUCH OR MORE off-pool.
- Held-out about 2x train: this is real memorisation, but it is local and small. On the pool's own orbits V's kappa is higher than off-pool:
  -13: 0.454 vs 0.374, and -12: 0.356 vs 0.335. Rms residual 15% lower there (0.0865 vs 0.1021); the exact fit shows no such gap (0.357 vs 0.351).
  The overfit sits where the training ESS is lowest (relative decades -5/-6: ESS 31.7k/23.2k against P = 5334; pool_summary "ess").
  The held-out loss itself does not track E_FN in either fit. The exact fit's L_ho rose from 5.7e-3 to 1.2-3.3e-2 while E_FN fell
  (pfaffian_r5/pp2_s1/fit/cx_r1.json L_ho_curve). Do not use it as evidence either way.
- Off-pool, V's correlation with the correction is better but its rms residual is worse: 0.1021 vs 0.0985 at -13, 0.0530 vs 0.0514 at -12.
  The 0.029 r deficit is extra noise in bond differences, not missing coverage.
- Single seed, single pool: the PASS margin is 0.114 in r, about 4x the E_FN walk (+-0.2e-6, i.e. 0.026 in r; PLAN.md line "Round 3").
  PASS robust; the ratio 0.94 is not: the 0.029 gap is one walk unit, and the pool-to-pool scatter is unmeasured (c4).
- Iterate averaging: the same 7-iterate mean is used in both arms (readout.json fit_spec.save_theta_at; cx_r1.json), so it is like-for-like.
  The per-iterate e of V is ungraded, so how much the average rescues V relative to the table fit is unknown. Augmented: 3.346e-6 vs 3.258e-6 (readout.json aug_b1), i.e. 0.92 of the augmented gain (my ratio (4.367-3.346)/(4.367-3.258)).
- 8x8 coverage: the overfit already appears at ESS/P ~ 4-6 per decade. At 8x8, P ~ 1.8e4 (B design) is 3.4x larger.
  The log-amplitude band also widens, about sqrt(64/36) = 1.33x more decades (estimate, not measured). At S = 4e5 the deep decades would then sit at ESS/P ~ 1, well inside the memorisation regime. Keeping 6x6's ratio needs S ~ 1.5-2e6.

## 2. Commit to the 8x8 loop test now? No. Build and check first.
- The bar is sharper than PLAN.md records: CNN-MPS 6x6 VMC -0.5038070(2) (laqx src TeX l.763) is 2.65e-6/site above OUR E0
  (-18.137147540071613/36; sign_comb/r6grade/v1_sG.json). PLAN.md:493 has ~3.2e-6, which comes from the paper's ROUNDED ED -0.503810 (TeX l.759).
  2.65e-6 is below every loop guide we have:
  - R6 with exact s_G: 2.81e-6.
  - Deployable stored sign: R6 3.70e-6 (S_a1) and 3.56e-6 (S_comb); R5 4.12e-6 and 3.97e-6 (v2_S_a1.json, v2_S_comb.json).
  - V's guide with a stored sign would be ~3.35 + 0.86 = 4.2e-6 (estimate).
  At 6x6, with exact tables, the deployable loop loses to a published plain VMC by 0.9-1.5e-6.
  An 8x8 win over p=1 and Varbench (both RBM+PP) would therefore not show that the loop beats strong networks.
- 8x8 competitor that makes the result mean something: the strongest network we can train at 8x8 (CNN-MPS D=20 via laqx, or our ViT) + FN p=0.
  Its E_L costs ~2N evaluations per step with no fast ratios (PLAN.md:495-496). That is the same cost class as our augmented guide:
  ~185x plain per walker step measured (B design benchmark, A40), maybe ~40-60x after the savings.
  Equal cost per walker step is roughly met; compare at equal TOTAL GPU-h (training + projection).
  There is no published 8x8 CNN-MPS number (PLAN_tooling.md:107), so we must produce this bar ourselves; first at 6x6: CNN-MPS (or ViT) + FN p=0 against the stored-sign loop, from exact tables. That costs CPU and a small amount of GPU.
- Must exist before any graded 8x8 point:
  - an 8x8 tempered MCMC pool sampler (only the exact 6x6 CDF exists today) and the 8x8 pp2 model in pool_fit;
  - the gfmc_aug class with its c=0 equivalence test;
  - the savings (dedup, padding, smaller CNN).
- c1-c5:
  - c1 (8x8 energy sign gate): NECESSARY, but its tolerance of 1e-5/site (PLAN_pfaffian.md:1418-1419) is as large as the whole predicted R1 margin over p=1
    (Astra -0.498951..-0.498967 vs -0.498946(5): 0.5-2.1e-5). Tighten it to <= 3e-6, i.e. the grading target.
  - c3 (exact c>0 validation of the evaluator at 6x6): NECESSARY and cheap. The 8x8 grading code is new.
  - c5 (same-sign comparisons): NECESSARY and free; it is a bookkeeping rule. c4 (independent pools): NECESSARY for the "0.94" claim and for sizing S at 8x8; not needed for the PASS itself. Cheap: CPU pool + ~2.4 H100-h.
  - c2 (E' and f0 from samples): NICE-TO-HAVE. f0 is irrelevant on the fitted band (target change exactly 0 on -5..-13, above;
    capping changes E_FN by <= 1.5e-10 per the f_J audit). The E' sensitivity is a cheap 6x6 CPU check.

## 3. The A40 duplicate 17695322
It is worth nothing as evidence. It reproduces the Paderborn trajectory exactly: 101 of 101 common progress rows (steps 10-1010) agree in
held-out and train loss to all printed digits (fit_th/round_1/progress.txt vs pdb2/fit/round_1/progress.txt). It will return the same theta_V.
It was useful only as a determinism check across hardware, now shown. Recommend cancelling it (the lead/PI decides; I cancelled nothing).
A replica is better, with a NEW POOL SEED (c4; the pool is built on CPU from the tables) and the same fit seed. A second fit seed on that new pool is optional.
That is the pre-registered "second independent pool" (PLAN_pfaffian.md:1413), and it measures the pool scatter that the 0.94 ratio needs.

## 4. The single most likely way 8x8 fails, and the cheapest check that would reveal it
The stored sign, not sampling. At 6x6 the deployable sign already costs 0.86-0.88e-6 (r6grade), about a quarter of the augmented error.
At 8x8 the network is smaller: there is no XL, the L-size net has 0.9-1.2% configuration error, and the savings precondition asks for an
additional ~4x fewer CNN parameters. The loop's predicted margin over p=1 is only 0.5-2.1e-5. A sign tax of the order of the c1 tolerance erases it,
and R0 fails before R1 starts.
Cheapest check (CPU, exact, no 8x8 spend): train the EXACT CNN class and size planned for 8x8 (after the 4x reduction) on s_G at 6x6.
Grade its tax on R5/R6 with v_bare exactly as r6grade did. If the tax exceeds ~1.5e-6 at 6x6, stop before any 8x8 GPU spend.
Second check (~1 H100-h): train that net at 8x8 on s_G labels (26 s per 2e5 samples) and measure the blocked-bond-weighted wrong fraction on held-out tempered samples, against the same quantity at 6x6.
Not checked: per-iterate grades of V; the E' sensitivity; the 8x8 decade widths (estimated above, not measured); whether S_a1 is the class planned for 8x8.

VERDICT Q1: PASS is real and not an artifact: like-for-like confirmed (target identical on the fitted band), off-pool storage equals the exact fit, mild memorisation only in decades -12/-13. But V tested coverage only, not MCMC, the stored sign or the 8x8 model.
VERDICT Q2: NO-GO for 8x8 grading now. First beat (or re-bar against) CNN-MPS: 2.65e-6, not 3.2e-6, which already beats our deployable 6x6 loop (3.56-4.12e-6). Tighten c1 to <= 3e-6. c1, c3, c5 necessary; c4 necessary for the ratio; c2 nice-to-have.
VERDICT Q3: Cancel 17695322 (it reproduces pdb2 exactly through step 1010, so it carries zero information). Spend the slot on a new-pool-seed replica (c4).
VERDICT Q4: Most likely failure is the stored-sign tax at 8x8 with the smaller CNN. Cheapest reveal: grade the planned 8x8 CNN class at 6x6 on R5/R6 (CPU, exact) before any 8x8 spend.
