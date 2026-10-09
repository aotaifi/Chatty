# Compressing the loop's guide stack into one callable network: exact 6x6 pre-tests (2026-10-09)

Pre-registration and amendments: `experiments/compress_6x6/README.md` (commits 48a4b98, amendment 1, amendment 2,
all before the P2 evaluations). Code: `experiments/compress_6x6/`. Raw results: `runs/*.json` here; full run
directories on ws1 under `/project/theorie/a/A.Otaifi/chatty_compress6/runs/`. Figure: `fig_compress_6x6.png/.pdf`.
Exact 6x6 J1-J2, J2/J1 = 0.5, full symmetric sector (15.8M orbits). Energies per site, relative to E0.

## Verdict
**P2 fails for every arm; P3 was not run (it was conditional on P2). Inconclusive, not a conceptual limit.**

| arm (student of la_k) | k = 3: share <H> / share E_FN | k = 6: share <H> / share E_FN | cost (psi_P passes) | tail collapse |
|---|---|---|---|---|
| A' warm ViT, Adam (lr 1e-6 by the scan rule, 4000 steps) | 8.2% / 9.6% | 7.6% / 10.2% | 1.00 | yes |
| A' warm ViT, damped Gauss-Newton (400 steps) | 1.3% / 1.7% | 6.7% / 9.9% | 1.00 | yes |
| A ViT from scratch (lr 1e-3, 4000 steps) | not run (amendment 2) | -8720% / -5130% (<H> 7.9e-3, E_FN 3.1e-3) | 1.00 | yes |
| B bond-ratio net (no amplitude) | continuation -50x / -45x table (2 seeds) | -87x / -82x | 0.06-0.11 | yes |
| pass bar | >= 70% / >= 70% | >= 70% / >= 70% | <= 3 | no |

- **Cause (what failed): none of the students moved toward la_k.** The warm ViT ends where it starts. Its held-out
  bond log-ratio error is indistinguishable from that of the untrained psi_P in every decade (fig. c). Its own loss
  fell by only 1-11%. The from-scratch ViT and the spin-only ratio net land 4-20x above the warm start in bond log-ratio error.
- **Precision target (P1): 3e-3 per bond, already violated by psi_P in the bulk.** One write-back step keeps >= 90%
  only for white bond log-ratio noise <= 0.003 (log V, log W noise <= 0.001).
  - psi_P differs from la_3 by 0.005 (1e-7 decade) to 0.3 (1e-14) in |H|-weighted bond log-ratio, i.e. 2-100x the
    tolerance.
  - A distillation that keeps >= 70% must therefore move the bond ratios of the network in EVERY decade, the tail
    included.
- **Not a conceptual limit.** The teacher is a function of x alone, and nothing here shows that a ViT cannot
  represent it. Specifically:
  - the warm ViT is optimizer-limited: Adam raises the loss at every lr >= 3e-6 and is flat at 1e-6;
    sample-space Gauss-Newton is pinned at maximal damping (median lam 500-750 x tr/M, accept rate 0.42 = noise,
    step norm ~1e-6). This is the stiff landscape of stall_6x6 test 1;
  - the scratch ViT and B are optimizer/capacity-limited at this budget: their loss was still falling (A: 0.026 at
    4000 steps; B: 4.8e-3 at 20k steps, -5% per 4k).
  - Held-out = training to within 1.0-1.2x everywhere, so generalisation is not the problem; precision is.

## P1: sensitivity of one write-back step to V, W errors (`runs/cx_replay.json`, `runs/cx_p1ext.json`)
Replay check: every wtLOOP8 iteration reproduced to all printed digits (frac 0.745736 / 0.658892 / 0.762104 /
0.722097 / 0.510609, <H> exact).
Stored net k, guide k-1 exact; noise enters the net input and the semi-implicit target. Entries: kept share of the
unperturbed step (frac(eta)/frac(0)) for iterations 1 / 2 / 3.

| eta | iid log-noise on V, W | white log-a noise, per-bond rms eta |
|---|---|---|
| 0.001 | 0.99 / 0.99 / 0.99 | 1.00 / 1.00 / 1.00 |
| 0.003 | 0.93 / **0.90 / 0.89** | 0.98 / 0.97 / 0.96 |
| 0.01 | 0.25 / -0.13 / -0.25 | 0.73 / 0.61 / 0.56 |
| 0.03 | -5.9 / -8.9 / -10.3 | -1.4 / -2.7 / -2.9 |
| 0.1 | -71 / -111 / -128 | -25 / -39 / -44 |

- **eta\*_VW = 0.001, eta\*_bond = 0.003** (largest tested eta with >= 0.9 in all three iterations).
- The exact semi-implicit target is even more sensitive (iid V, W at 0.003: 0.75 / 0.69 / 0.61 of an ideal
  iteration vs 0.85 unperturbed). The step is a log-difference of 1 + W and 1 + H_xx + V - E, so a white relative
  error eta in V, W becomes white log-amplitude noise of order eta in the update. That costs ~0.33 eta^2 per site,
  the stall_6x6 0.004-rms criterion.
- These are white-noise models, the worst case. A smooth student error costs less, which is why the stored-net
  continuation (below) was planned as the direct test.

## P2: distillation at k = 3 (<H> 6.29e-5) and k = 6 (<H> 4.03e-5)
References, same sign s_k (`runs/cx_replay.json`):
- psi_P amplitude: <H> 1.273e-4 / 1.297e-4; E_FN 9.29e-5 / 9.31e-5.
- Stack: E_FN 5.25e-5 / 3.47e-5.
- Stack + one exact Lanczos step: 2.17e-5 / 1.55e-5.

| run | <H>(A, s_k) | E_FN(A, s_k) | A + 1 Lanczos | own loss start -> end | continuation frac (table 0.722 / 0.511) | SI target frac from A |
|---|---|---|---|---|---|---|
| A' Adam k3 | 1.220e-4 | 8.91e-5 | 3.66e-5 | 2.24e-4 -> 2.05e-4 | -4.78 | 0.884 |
| A' GN k3 | 1.264e-4 | 9.23e-5 | 3.82e-5 | 2.24e-4 -> 2.22e-4 | -4.53 | 0.879 |
| A' Adam k6 | 1.229e-4 | 8.71e-5 | 3.71e-5 | 5.16e-4 -> 4.58e-4 | -2.57 | 0.892 |
| A' GN k6 | 1.237e-4 | 8.73e-5 | 3.75e-5 | 5.16e-4 -> 4.80e-4 | -2.45 | 0.887 |
| A scratch k6 | 7.926e-03 | 3.09e-03 | 1.17e-03 | 0.875 -> 0.026 | -0.60 | 0.838 |

Held-out |H| a_y/a_x-weighted rms bond log-ratio error per decade of a_k^2 (k = 6, seed 0; fig. c):

| decade | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 | 1e-11 | 1e-12 | 1e-13 | 1e-14 |
|---|---|---|---|---|---|---|---|---|---|
| psi_P (untrained) | - | 0.0075 | 0.013 | 0.026 | 0.055 | 0.11 | 0.20 | 0.36 | 0.52 |
| A' Adam | 0.0050 | 0.0070 | 0.0125 | 0.024 | 0.050 | 0.12 | 0.21 | 0.32 | 0.50 |
| A' GN | 0.0062 | 0.0077 | 0.013 | 0.025 | 0.051 | 0.12 | 0.22 | 0.33 | 0.53 |
| A scratch | 0.094 | 0.13 | 0.18 | 0.27 | 0.39 | 0.63 | 0.91 | 1.4 | 1.9 |
| B (seed 0) | 0.031 | 0.050 | 0.076 | 0.12 | 0.20 | 0.33 | 0.48 | 0.75 | 1.21 |
| tolerance eta\*_bond | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 |

- **Continuation** (stored FEAT net k+1 driven by the student's own log a, V, W, no retraining) is strongly negative
  for every student.
  - For A' this mainly reflects that the student is still ~psi_P: the stored net was trained on the features of the
    table guide la_k and is out of distribution on psi_P-like features.
  - The student's own semi-implicit target (SI frac 0.88-0.89 of an ideal iteration) is as good as the table's
    (0.86). A RETRAINED FEAT net on the student guide would be the fair continuation; that is P3, which was not run.
- **B** gets log V, log W wrong by 0.05 / 0.025 rms (a_k^2-weighted), 25-50x eta\*_VW. Its continuation step on the
  exact log a is -36 to -44 (frac), i.e. -50x to -87x the table step. P1 predicts exactly this.
- Cost: A/A' = 1.00 psi_P pass per configuration (same ViT, same 16 images; measured 0.997-1.037); B 0.06-0.11
  passes. Both are independent of k by construction.
- Optimizer evidence (own loss, training orbits; held-out batches equal to within 5%):
  - A' Adam rises first, then returns slightly below the start (k3: 2.24 -> 2.42 at 1k -> 2.05e-4 at 4k);
  - A' GN is flat after 100 steps;
  - B: 0.50 -> 0.0114 (4k) -> 0.0050 (16k) -> 0.0048 (20k);
  - A scratch: 0.88 -> 0.054 (1k) -> 0.033 (2k) -> 0.026 (4k), still falling.

## What this means for the loop
- The re-basing problem is the original write-back problem in a sharper form. The accumulated correction must be
  stored to <= 3e-3 in every bond log-ratio down to a^2 ~ 1e-14. The base itself is off by 5e-3 to 0.5 there.
- A converged ViT cannot be fine-tuned there by first-order or sample-space second-order steps at these batch sizes.
- Neither a from-scratch ViT nor a spin-only ratio net gets near it in ~0.5 GPU-h.
- What was not tested, and is the next candidate (reviewer E4 "cheaper first try"): a zero-hop residual that takes the
  base's INPUT-GRADIENT features (one backward pass: a linearised log-ratio for every bond, ~3 base passes). These
  give the student one-hop information without a hop. The FEAT result (one-hop inputs: 89%) says this information
  is what spin-only students lack.
- Alternative that needs no compression: cap the depth (reviewer E6). Frozen-base B for 2 iterations, then exact
  Lanczos + calibrated FN, compared at 8x8 with ViT + Lanczos p1/p2. For reference: stack + Lanczos at k = 6
  (1.55e-5) is below psi_P + Lanczos (2.55e-5), but the stack is a 6-hop table.

## Compute (ws1; sacct)
- Full A40: replay + P1 2.0 h; P1 extension 0.95 h; A' Adam 2 x 0.63 h; A' GN 2 x 0.55 h; A scratch ~0.65 h;
  B 4 x 0.21 h + eval 0.1 h; smokes 0.07 h.
- A40 24 GB slices: lr scans 0.55 h, smoke 0.04 h.
- 2080 Ti: 0.15 h (two failed B attempts, out of memory).
- Total ~7.4 GPU-h, of which ~0.6 on slices (<= 8 budget).
