# Three label-free loop rounds beat one Lanczos step at 6x6 by 0.79e-6 (same sign); the cost does not yet win

All energies are exact fixed-node values (6x6, J2 = 0.5), per site above E0. No Monte Carlo.

- **Result:** the augmented guide B_2 e^{c f_J[B_2]}, refit for two label-free rounds under the competitor's signed-Lanczos sign s_G, reaches
  5.207e-6 (seed 1) and 5.290e-6 (seed 2). Network + one signed Lanczos step + FN (competitor, same sign) reaches 5.574e-6.
  Margin 0.28-0.37e-6; seed spread 0.083e-6 (pfaffian_r2g/readout/g{1,2}/readout.json).
- **Path:** round 1 under s1 (L-SWO), 1.018e-5 -> 8.34e-6; switching the Jacobi factor to s_G's walls gives 6.07e-6 untrained
  (pfaffian_lswo2/prep_result.json); round 2 under s_G gives 5.2-5.3e-6. Bare network per round: drop 1.3-1.6e-6.
- **Why it works:** label-free refit to the network's own Jacobi-corrected state stores more than an exact-label fit (R1b 15% vs
  L-SWO 37% of a round's gain); the correction itself is a direction the PP cannot absorb (Test Theta v2, held-out 0-4%),
  so the Jacobi factor stays an explicit layer and the base only refits the bulk.
- **Sign and amplitude interact** (Test X): s_G helps the loop's magnitude by 0.95e-6 and the competitor's by 4.45e-6.
- **Not shown:**
  - equal cost (a Jacobi level costs ~5.4x a plain PP per walker step, the same two-hop class as one Lanczos step;
    training 0.58 H100-h + ~3.5 H100-h + 7.6 A40-h vs the competitor's VMC 14.3 GPU-h);
  - deployable sign (s_G is computed, not stored; sign angle);
  - sampled fits (exact tables used);
  - sizes > 6x6;
  - seeds independent only in the optimiser path.
- **Round 3 (added 2026-10-31):** the official value (the pre-registered kill rule, theta at step 1200) is augmented 4.786e-6,
  0.79e-6 below the competitor; averaging the fit's tail checkpoints gives 4.644e-6 (pfaffian_r3avg/readout/readout.json).
  One seed. The per-round law and the equal-cost comparison follow.
- **Next:** no round 4 (lead, 2026-10-31). The scale test needs the cost law below.

## Per-round table under s_G (exact FN, per site above E0)

| round k | guide B_k | fit | bare e(B_k) | aug min e (c*) | bare drop | gap g_k = bare - aug | r_k = drop / g_{k-1} |
|---|---|---|---|---|---|---|---|
| 1 | B_1 (L-SWO, trained under s1) | 520 steps, converged | 1.5453e-5 | 6.069e-6 (4.0) | - | 0.938e-5 | (0.368 under s1) |
| 2 | B_2 seed 2 | 950 steps, converged | 1.4090e-5 | 5.290e-6 (3.99) | 1.363e-6 | 0.880e-5 | 0.145 |
| 2 | B_2 seed 1 | 2926 steps, budget | 1.3839e-5 | 5.207e-6 (3.98) | 1.614e-6 | - | 0.172 |
| 3 | theta(1200) (official) | 1200 steps | 1.3133e-5 | 4.786e-6 (3.9) | 0.957e-6 | 0.835e-5 | 0.109 |
| 3 | theta_bar (mean of 6 checkpoints, 1350-2100, seed 1) | 2100 steps + average | 1.2522e-5 | 4.644e-6 (4.0) | 1.568e-6 | 0.788e-5 | 0.178 |

Sources: pfaffian_lswo2/prep_result.json, pfaffian_r2g/readout/g{1,2}, pfaffian_r3/readout/r3_1200, pfaffian_r3avg/readout.
The augmented guide gains about half the bare drop per round (ratio 0.57, 0.53, 0.41).

**Fit r_k = r_1 q^k.** Two matched (single-iterate) rounds under s_G give q = 0.109 / 0.145 = 0.75, but the seed spread of r_2
(0.145-0.172) and the E_FN walk along a fit (0.1-0.2e-6 per 150 steps, i.e. +-0.02 in r_3) make q = 0.63-0.95. With averaging,
round 3 stored r = 0.178, as much as round 2 without it. So **a declining r is not established**: the decline is the
same size as the protocol and noise difference. Under s1, r fell 0.37 -> 0.11, but its round 2 was unconverged (A40, 935 steps).

**Hypothesis H_q (not established):** under one protocol (averaged fits), r_{k+1} = 0.75 r_k and the augmented gain stays
half the bare drop. From theta_bar, the loop would then close 30% of the remaining bare error (1.25 -> 0.88e-5) and 41% of the
augmented error (4.64 -> 2.8e-6) over all further rounds; with q = 0.5 only 11% / 15%. The prediction for round 4 (averaged,
from theta_bar) is r_4 = 0.11-0.15, bare 1.13-1.17e-5, augmented 4.0-4.3e-6. It kills H_q if r_4 >= 0.178 (no decline, the
loop keeps its rate) and kills "the loop progresses" if the bare drop is <= 0.5e-6 (r_4 <= 0.063). This corrects my earlier
round-4 prediction (1.17-1.21e-5), which applied the single-iterate r to an averaged start. Round 4 costs about 1.5 H100-h.

## Equal-cost comparison at 6x6

| cost | ours (PP + one Jacobi factor) | competitor (RBM+PP + one signed Lanczos step) | status |
|---|---|---|---|
| walker step vs its own plain network | 5.4x a plain PP (measured, T2, pfaffian_t2/t2_result.json; C kernel; op model 10.6x) | 25-36x by op count IF the RBM dominates: the d^2 ~ 1300 two-swap ratios each cost O(M) (M = alpha N hidden units, alpha 4 assumed), with no shared-inverse saving, giving (n^3 + d^2 (64 + M)) / (n^3 + d M) ~ 25x. For the PP part alone it is the same constant class as ours (board 23). Not measured | UNRESOLVED until measured |
| network training | 4.4 H100-h on top of the competitor's PP start (rpT1_pp): F1 real fit 1.2 + complex copy 0.7 + round 1 0.58 + round 2 0.89 + round 3 (to step 1200) ~1.0 | 14.3 GPU-h on RTX 2080 Ti / A40 slices (results/rbmpp_6x6/README.md section 4; 6.5 h of it on plateaued runs) | see below |
| exact FN solves for E', c* and readout | ~1 CPU-h per round (15.8M orbits; only possible at 6x6) | one FN solve | not counted for either |

Conversion: our own code measured 26.8 s/step on an A40 vs 2.99 s/step on an H100 (9.0x, pfaffian_lswo2 vs pfaffian_r2g).
Typical fp32 throughput gives 2-3x. So the competitor's 14.3 GPU-h is 1.6-7 H100-h.
**Verdict at 6x6:**
- Training: tie or lose, ours 4.4 H100-h plus their PP start against their 1.6-7 H100-h in total.
- Walker step: unresolved. The RBM part decides it, and it is unmeasured.
- Accuracy: win, 4.79e-6 vs 5.57e-6.
- No equal-cost accuracy comparison has been made.
The rounds would also need FN-QMC runs for E' and c* at scale. They are free here only because 6x6 allows exact solves.

## Stop rule

Along a fit at a fixed target, E_FN walks by 0.1-0.2e-6 per 150 steps, which is noise in directions the copy loss is flat in.
The mean of the tail iterates (Polyak-Juditsky) beat the best iterate by 0.35e-6 (1.252e-5 vs 1.287e-5). Future rounds use a
fixed step budget plus tail averaging.
