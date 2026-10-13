# RBM+PP guides for the FN/Krylov loop, exact 6x6 J1-J2 (J2/J1 = 0.5, PBC) — results 2026-10-13

Pre-registration and amendments: `experiments/rbmpp_6x6/README.md`. All energies per site relative to the exact
E0/N = -0.50380965 ("dE"); relative error = dE / 0.50380965. Tables are exact on all 15,804,956 orbit
representatives (full space group x spin-flip projection). Figure: `fig_rbmpp_loop_6x6.png`.

## 1. The guides (what was trained, how, and the check)
Ansatz exactly as Nomura & Imada (2021): RBM with 16 complex hidden units (no symmetry imposed) times a pair product
with a real, full 36 x 36 f; trained by VMC with SR (natural gradient), 'simple' published projection
(translations x {1, C4} x spin flip). Own JAX code (tests pass, incl. a singlet test of the fermion signs and a 4x4
exact-enumeration check of the estimator).

| guide | how | train cost (RTX 2080 Ti h) | VMC, published projection | VMC, full projection | exact table (full projection) | rel. error |
|---|---|---|---|---|---|---|
| A: RBM+PP converged | T1 (PP, 3000 SR steps) -> T2 (+ RBM, 900 steps) -> T4 (1500 steps, 8192 samples, SR shift 0.02) | ~7.0 (1.34 + 0.74 + 4.92) | -0.503727(3) (two evals: -0.5037248(36), -0.5037290(38)) | -0.5037579(30) | **dE 4.74e-5** (-0.5037622) | 9.4e-5 |
| PP alone | T1, 3000 steps (flat after ~400 steps) | 1.34 (0.2 to the plateau) | -0.5036901(61) | -0.5037496(40) | **dE 6.30e-5** | 1.25e-4 |
| B2: RBM+PP early | T3d checkpoint, step 95 (from scratch) | ~0.08 | - | -0.503664(11) | **dE 1.54e-4** (ViT: 1.32e-4) | 3.0e-4 |
| B1: RBM+PP early | T3d checkpoint, step 70 | ~0.06 | - | -0.503407(16) | **dE 3.87e-4** | 7.7e-4 |

- **Table check (exact vs VMC, same state, full projection):** A 1.4 sigma, PP 0.75 sigma, B2 0.8 sigma, B1 1.2 sigma
  (B1/B2 compared on the complex state). PASS for all.
- **Pre-registered target (published projection <= -0.503745): NOT reached.** We get -0.503727, 1.8e-5/site short.
  Cause (partly pinned, inconclusive): optimiser-limited. With 2048 samples and SR shift 1e-3 both RBM+PP runs
  plateaued at the PP-alone energy (-0.50368 .. -0.50370 for 3000+ steps while lr halved); 4x samples + shift 0.02
  (T4) moved the plateau down by 3e-5 at the same steps and at equal wall-clock. T4 is flat over its last 1000 steps
  (250-step means within 3e-6), so the next step (more samples again, or longer annealing) was not tested.
- **With the full projection our RBM+PP is at the published level** (4.74e-5 vs published 4.5e-5 with the simple
  projection), and the **PP alone (6.3e-5) is already 2x better than our ViT (1.32e-4)** at ~0.2-1.3 GPU-h on a
  2080 Ti. The full point-group projection is worth 3-6e-5/site here (also for the ViT it was worth 2.4e-5).

## 2. Loop on each guide (exact tables, no networks; cx_step, 16 iterations)
Iteration = clipped-Lanczos target Lc_0.3 (new sign = sign of 1 + alpha (E_L - E), amplitude log max(|f|, 0.3),
alpha energy-optimal) or the old semi-implicit FN target SI_1. "+pL": exact p-step joint Lanczos on the guide.

| guide (start dE) | E_FN guide | w_s guide | +1L / +2L / +3L | Lc k=1 | k=2 | k=4 | k=8 | k=16 (rel.) | SI_1 k=16 |
|---|---|---|---|---|---|---|---|---|---|
| ViT 1.32e-4 (ref., cx_step_long16) | 1.02e-4 | 1.1e-4 | 2.55e-5 / 8.6e-6 / 3.8e-6 | 2.62e-5 | 1.16e-5 | 4.91e-6 | 1.90e-6 | 6.1e-7 (1.2e-6) | 7.5e-6 |
| **A** 4.74e-5 | 3.52e-5 | 1.1e-5 | 7.4e-6 / 2.5e-6 / 1.1e-6 | 7.49e-6 | 3.26e-6 | 1.59e-6 | 7.53e-7 | **2.98e-7 (5.9e-7)** | 3.9e-6 |
| PP 6.30e-5 | 4.77e-5 | 2.4e-5 | 1.46e-5 / 5.9e-6 / 2.6e-6 | 1.47e-5 | 7.55e-6 | 4.01e-6 | 2.04e-6 | 8.70e-7 (1.7e-6) | 7.9e-6 |
| B2 1.54e-4 | 1.22e-4 | 9.2e-5 | 4.83e-5 / 2.19e-5 / 9.2e-6 | 4.86e-5 | 2.82e-5 | 1.60e-5 | 8.00e-6 | 2.94e-6 (5.8e-6) | 2.5e-5 |
| B1 3.87e-4 | 2.92e-4 | 3.1e-4 | 1.51e-4 / 6.3e-5 / 2.3e-5 | 1.51e-4 | 8.94e-5 | 5.06e-5 | 2.55e-5 | 1.01e-5 (2.0e-5) | 6.4e-5 |

Per iteration E_FN, w_s and the +1/+2 Lanczos polish are in `runs/step_*.json` (e.g. A, Lc: w_s 1.1e-5 -> 8.4e-7
(k=1) -> 1.5e-8 (k=16); E_FN 5.7e-6 (k=1) -> 2.5e-7 (k=16)).

Readings:
1. **One Lc iteration = one Lanczos step, from every guide:** Lc(k=1) is within 1-2% of guide + 1L, and
   "+1L on Lc(k)" equals Lc(k+1) to 1-3% at every k. The old SI target is 5-13x slower (k = 16).
2. **(A) add-on, table level: PASS by a wide margin.** From the best RBM+PP, 16 iterations remove 99.4% of the error
   (4.74e-5 -> 2.98e-7, relative 5.9e-7; DMRG(8192) is at 1e-5). The pre-registered 30% bar after 8 iterations
   (98.4%) is passed — but this is the exact table loop, not a network loop (not run).
3. **Equal hops: joint Lanczos still wins** (panel b): A at 4 hops: loop 1.59e-6 vs guide + 4L 4.8e-7 (3.3x);
   B2: 1.60e-5 vs 3.4e-6 (4.7x). Iterating one step at a time is greedy; the joint polynomial is better at equal
   matvecs. The loop's case rests on cost per iteration at large N (one hop of a stored net), not on 6x6 hop counts.
4. **The speed of the loop depends on the guide, not only on its energy.** B2 starts like the ViT (1.54e-4 vs
   1.32e-4) but ends 5x higher after 16 iterations (2.9e-6 vs 6.1e-7); the PP starts 2x lower than the ViT and
   ends 1.4x higher. Early-training RBM+PP checkpoints are slow. Cause not pinned (hypothesis: their error sits in
   low-lying excitations that a one-step polynomial removes slowly; testable from the Lanczos variances in the tables).
5. **(B) method:** from a moderate guide the loop does the work but slowly: B1 (rel. 7.7e-4) needs 5 iterations to
   pass the published RBM+PP (dE 4.5e-5), B2 2 iterations, PP and A pass it immediately (they already are at that level).

## 3. Old-target network loop on the PP guide (wt_loop, unchanged, realistic write-back)
`runs/wtloop_PP.json`: 8 iterations (20k Adam steps each), dE 6.30e-5 -> 5.64 / 4.63 / 4.08 / 4.08 (dead fit,
frac 0) / 3.53 / 3.18 / 3.90 (write-back blow-up, frozen-FN frac -2e8) / 2.45e-5; E_FN 4.77e-5 -> 2.16e-5.
Removes 61% in 8 iterations (ViT: 76%); both known failure modes of the unchanged wt_loop appeared. Guide + 1L on
the loop guide at it. 8: 1.17e-5 vs PP + 1L 1.46e-5. Superseded by the Lc target (section 2); kept for the record.

## 4. Cost (GPU-h, sacct)
- Total of this work: 25.4 GPU-h (RTX 2080 Ti and A40 24 GB slices), above the 15 GPU-h brief. Of it: guide
  training 14.3 (incl. 6.5 h on the two plateaued RBM+PP runs T2/T3 that located the optimiser cause, and 0.8 h
  lost to the PP mask bug), 5 exact tables 2.4 (+ 1.0 h of duplicate B1/B2 tables), candidate VMC pre-filter 1.3,
  T4 eval 1.3, loops 5.0 (wt_loop PP 2.2; cx_step 0.5-1.0 per guide).
- The ViT guide cost was not measured here (see results/literature/J1J2_BENCHMARKS_COST.md for published costs).

## 5. Open problems
- Reach the published RBM+PP energy with the published projection (more samples / slower annealing; 1.8e-5 short).
- Network (callable) version of the Lc loop on the RBM+PP guide (stage 2 of the step-size test; cx_chop3), with
  3 seeds; the table result above is the ideal it must follow.
- Why do early RBM+PP checkpoints converge slower than the ViT at equal energy (section 2, reading 4)?
- Equal-hop comparison still lost at 6x6; the loop's case needs the 8x8+ cost-per-iteration argument.
