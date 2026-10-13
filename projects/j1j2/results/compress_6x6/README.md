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

# Round 2 (2026-10-09/11): attacking the recursion
Pre-registration b33c794, amendment 4 before the diagnostic run. Figure `fig_round2_6x6.png/.pdf`; raw:
`runs/cx_chop_s0.json`, `cx_chop_s1.json`, `cx_struct.json`, `cx_vw.json`, `cx_rebase_diag.json`.

## Verdict
**Test 1 (constant-hop loop) narrowly fails: 67% of the table loop's gain at iteration 6 (bar 70%), then a plateau at
<H> ~7.0e-5. Test 3 (amortised V/W) narrowly fails at k = 3 (0.89, bar 0.90) and clearly at k = 6 (0.48). Test 2:
student errors are 1.5-2x more tolerable than white noise; re-base errors are LESS tolerable (outliers).**
- Cause of the plateau (amendment-4 diagnostic): neither the optimizer nor the capacity limits the re-base. Its own
  loss floors at ~5e-6 with 37k parameters at 60k steps and with 111k parameters alike. The one-hop features of the
  frozen psi_P are the limit: they hold ~84% of the accumulated correction and no more.
- Not a conceptual limit. The fix is to give the re-base more information about the guide's neighbourhood than
  psi_P's own one-hop sums (next step below).

## Test 1: constant-hop loop (re-base every iteration into a frozen-base-feature net)

| it | frac s0 / s1 | <H> before re-base (s0) | <H> re-based guide s0 / s1 | retention (this iteration's gain) s0 / s1 | retention since psi_P s0 / s1 | E_FN s0 | G + 1 / 2 Lanczos (s0) | table loop <H> / E_FN |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.71 / 0.67 | 9.94e-5 | 1.007e-4 / 1.005e-4 | 0.95 / 1.00 | 0.96 / 1.00 | 7.89e-5 | 3.13e-5 / 1.17e-5 | 9.67e-5 / 7.62e-5 |
| 2 | 0.62 / 0.78 | 8.26e-5 | 8.51e-5 / 8.28e-5 | 0.86 / 0.74 | 0.95 / 0.89 | 6.91e-5 | 2.83e-5 / 1.05e-5 | 7.87e-5 / 6.44e-5 |
| 3 | 0.78 / 0.00* | 6.67e-5 | 7.48e-5 / 8.40e-5 | 0.56 / - | 0.88 / 0.97 | 6.18e-5 | 2.70e-5 / 1.00e-5 | 6.29e-5 / 5.25e-5 |
| 4 | 0.76 / 0.64 | 6.04e-5 | 7.18e-5 / 7.68e-5 | 0.21 / 0.53 | 0.84 / 0.90 | 5.91e-5 | 2.64e-5 / 9.96e-6 | 5.20e-5 / 4.40e-5 |
| 5 | 0.76 / 0.71 | 5.74e-5 | 6.94e-5 / 7.26e-5 | 0.17 / 0.30 | 0.84 / 0.86 | 5.65e-5 | 2.56e-5 / 9.95e-6 | 4.79e-5 / 4.08e-5 |
| 6 | 0.74 / 0.77 | 5.56e-5 | **7.02e-5 / 7.03e-5** | -0.06 / 0.16 | 0.81 / 0.83 | 5.68e-5 | 2.54e-5 / 1.00e-5 | 4.03e-5 / 3.47e-5 |
| 7 | 0.76 | 5.51e-5 | 6.91e-5 | 0.07 | 0.82 | 5.56e-5 | 2.50e-5 / 9.97e-6 | 3.64e-5 / 3.15e-5 |
| 8 | 0.75 | 5.44e-5 | 7.03e-5 | -0.08 | 0.80 | 5.58e-5 | 2.50e-5 / 1.01e-5 | 3.16e-5 / 2.75e-5 |

\* Seed 1, iteration 3: the write-back fit returned f = 0 (own loss flat at its initial value for all 20k steps, a dead
Adam run), so that iteration did nothing; the loop continued.
- **Pre-registered rule: FAIL, narrowly.** Share of the table loop's accumulated <H> gain at iteration 6 is 0.674
  (seed 0) and 0.673 (seed 1); the bar is 0.70 (<H> <= 6.78e-5). <H> falls in 5 of 6 iterations for both seeds.
- **The write-back is not the problem.** Capture from the callable guide stays at 0.62-0.78 in every iteration, as in
  the table loop (0.42-0.76), and the semi-implicit target from the guide's own V, W stays at 0.87-0.89.
- **The re-base is the problem.** Its retention of each new iteration's gain falls 0.95 -> 0.86 -> 0.56 -> 0.21 ->
  0.17 -> ~0. Its accumulated retention stays at 0.80-0.84. The |H|-weighted rms bond error of the re-base grows
  0.001 -> 0.007, above the P1 tolerance 0.003 from iteration 3.
- **Against Lanczos at equal hop count:**
  - The callable guide + 1 Lanczos step reaches 2.5e-5, the same as psi_P + 1 Lanczos (2.55e-5).
  - + 2 steps reaches 1.0e-5 vs 8.6e-6 for psi_P + 2 Lanczos steps. This is no gain over Lanczos.
- **Cost per sample, measured hop counts, independent of k:**
  - guide value: one hop of psi_P (78 evaluations at 6x6, ~135 at 8x8);
  - guide V, W: two hops (2.8e3 / 8.9e3);
  - edge-loss training sample: three hops (~1e5 / ~4e5).
  At the measured 1.9e4 psi_P evaluations/s per A40 that is 4 ms / 0.15 s / ~5 s per sample at 6x6.

**Diagnostic of the plateau (amendment 4; same write-back table, iteration 6, seed 0; chop re-base: -0.06):**

| re-base variant | own loss end | retention of iteration 6's gain | retention since psi_P |
|---|---|---|---|
| chop (fresh, 37k, 20k steps, lr 3e-3) | 8.7e-6 | -0.06 | 0.81 |
| warm start from the iteration-5 re-base, 20k, lr 1e-3 | 4.9e-6 | 0.17 | 0.85 |
| fresh, 60k steps, lr 1e-3 | 4.9e-6 (5.4e-6 at 40k) | 0.22 | 0.86 |
| fresh, C = 64 (111k parameters), 40k, lr 1e-3 | 5.8e-6 | 0.09 | 0.84 |

- **Reading (pre-registered): none reaches 0.7, so neither optimizer nor capacity.** A lower lr and longer training
  help a little (-0.06 -> 0.22), but the loss floors at ~5e-6 whatever the size. The frozen base's one-hop
  features (log a_P, V_P, W_P) cannot carry the accumulated correction beyond ~85%. This is the same finding as
  option B (6d(b)): the base cannot see how the neighbours changed.

## Test 2: structured vs white tolerance (one write-back step from la_k with the actual error fields)
kept = frac / unperturbed frac; size = exact |H| a_x a_y-weighted rms bond error of c e.

| field | k | bond rms -> kept |
|---|---|---|
| white | 3 | 0.0010 -> 0.99, 0.0030 -> 0.94, 0.0099 -> 0.36, 0.030 -> -5.1 |
| white | 6 | 0.0010 -> 0.99, 0.0030 -> 0.91, 0.0098 -> 0.01, 0.030 -> -8.2 |
| A' student | 3 | 0.0020 -> 1.01, 0.0059 -> 0.95, 0.020 -> -0.40 |
| A' student | 6 | 0.0031 -> 1.02, 0.0092 -> 0.68, 0.031 -> -8.7 |
| A scratch student | 6 | 0.0023 -> 0.97, 0.0069 -> 0.62, 0.023 -> -3.5, 0.069 -> -29 |
| base-feature re-base of la_k | 3 | 0.0020 -> 0.88, 0.0066 -> -0.37, 0.013 -> -5.3 |
| base-feature re-base of la_k | 6 | 0.0051 -> 0.46, 0.017 -> -2e5 |
| white log V, log W | 3 | logV 0.001 -> 0.98, 0.003 -> 0.83, 0.010 -> -0.79 |

- eta\*_struct (kept = 0.9, log-linear interpolation): white 0.0033 (k = 3) / 0.0031 (k = 6); A' student 0.0062 /
  0.0045; A scratch 0.0029. **By the pre-registered rule the student errors are "more tolerable":** at 0.006 the A'
  error costs 0.05 against ~0.4 for white noise at k = 3.
- **The re-base errors are LESS tolerable than white.** At c = 1 (the actual re-base) the semi-implicit target from
  the perturbed V, W diverges (SI 'kept' -8e4 / -2e5). So the global rms hides a few configurations with very large
  errors (rare orbits the re-base never sampled). In the constant-hop loop the write-back uses the re-based guide's
  own V, W, so these outliers do not wreck the capture (0.62-0.78). They are part of the guide's energy error.
- B's V/W fields: every point is NaN (a bug in the log-space interpolation with V = 0 orbits). B's errors (log V 0.05)
  are 50x the white V/W tolerance (0.001-0.003), so this does not change any reading.
- Caveat: the replay applies the step computed from the perturbed guide to the exact guide. For log-amplitude errors
  this is an upper bound on the damage; test 1 measures the real thing.

## Test 3: amortised V/W from one hop of the base

| k | exact-V/W capture | V_P, W_P instead (no training) | amortised capture (rel.) | rms log V / log W error, held-out | training |
|---|---|---|---|---|---|
| 3 | 0.722 | -0.386 | 0.645 (**0.894**) | 0.0046 / 0.0018 | 0.0042 / 0.0017 |
| 6 | 0.511 | -9.30 | 0.243 (**0.476**) | 0.0103 / 0.0054 | 0.0096 / 0.0050 |

- **Pre-registered rule: FAIL** (bar 0.9 at both k).
  - At k = 3 the amortised V, W nearly suffice (0.894). The error (log V 0.0043) is 4x the white V/W tolerance but
    costs only 11%: structured again.
  - At k = 6 the base's one-hop information is not enough (log V error 0.018).
- Held-out = training, and the loss was flat over the last 10k steps. Same cause as test 1: with psi_P's one-hop
  quantities as the only guide information, the later guides' V, W are out of reach.

## Next step (proposal)
All three results point at one cause. The frozen base's one-hop sums (V_P, W_P, log a_P) do not carry enough
information about how the neighbours' amplitudes changed. Fixes in order of cost:
1. **Per-bond base information instead of sums.** Feed the re-base (and the V/W predictor) psi_P's per-bond
   log-ratios r_b(x) (all 144 bonds, one hop of the base, the same evaluations as V_P, W_P) through a bond-equivariant
   layer, instead of only their two sums. The cost is unchanged (one hop of the base) and the information is
   strictly richer. Pre-test: the iteration-6 re-base diagnostic and test 3 at k = 6 with these inputs, pass = 0.7 /
   0.9.
2. **Re-base onto the previous re-based guide's features** (a rolling base: base_{k+1} = G_k, re-base every 2-3
   iterations), so that the features see the changed neighbours. Cost per sample: one hop of G_k = two hops of psi_P,
   still independent of k.
3. Only if both fail: two-hop base features (8.9e3 base evaluations per sample at 8x8).

## Compute (round 2)
- Full A40: chop seed 0 1.50 h, struct 1.85 h, vw 0.43 h.
- 24 GB slices: chop seed 1 2.51 h, smokes 1.3 h (one hit its time limit, one failed on a schedule bug), diagnostic
  0.79 h.
- Total 3.8 A40-h + 4.6 slice-h = 8.4 GPU-h counted naively, ~6.1 A40-equivalent if a 24 GB slice counts as half an
  A40.

# Round 3 (2026-10-11/12): per-bond inputs, rolling base, residual re-base
Pre-registration 2dc074b, amendments 5-7 (in `experiments/compress_6x6/README.md`), each before its run.
Figure `fig_round3_6x6.png/.pdf`. Raw: `runs/cx_pt3_*.json`, `runs/cx_chop_inc_s*.json`, `runs/cx_struct_Bfix.json`.

## Verdict
**The constant-hop loop PASSES with residual re-bases: 82% / 85% of the table loop's gain at iteration 6 (bar 70%;
round 2: 67%).** The remaining gap to the table grows after iteration 6. At equal hop count the loop still does not
beat psi_P + Lanczos.
- **The round-2 plateau was a distillation-target problem, not missing information.**
  - Re-distilling the WHOLE correction la_w - log|psi_P| into a fresh net hits a loss floor of ~5e-6 whatever the
    inputs: psi_P scalars, per-bond ratios, or even the previous guide G_5's own features (from which the increment
    was made).
  - Fitting only the new increment la_w - G_k, sampled by its own Dirichlet weight, keeps 0.78 with psi_P's scalars
    alone and 0.995 with G_5's features.
  - The new increment is ~4% of the accumulated correction in the Dirichlet metric. A fresh full fit under-samples it
    and cannot resolve it.
- The pre-registered per-bond (BondNet) and rolling-base arms fail. They are superseded by the residual re-base, which
  keeps the cost constant without them.

## Pre-tests (re-base of the iteration-6 write-back, seed 0; share of the iteration's gain kept)

| re-base | inputs | kept | own loss end |
|---|---|---|---|
| full, fresh (round 2) | spins + psi_P scalars | -0.06 | 8.7e-6 |
| full, best optimizer / size variant (round 2) | same | 0.22 | 4.9e-6 |
| **(1a) full, BondNet** | + psi_P's 144 bond log-ratios + kept flags | **0.10** (fail) | 6.1e-6 |
| **(3) full, rolling base** | + one-hop scalars of G_3 (two hops of psi_P) | **0.08** (fail) | 9.5e-6 |
| full, diagnostic | + one-hop scalars of G_5 (previous guide) | 0.21 | 7.5e-6 |
| **residual (amendment 6)** | spins + psi_P scalars | **0.78** (pass) | 1.3e-6 |
| residual, upper bound | + G_5 scalars | 0.995 | 4e-8 |

**(1b) BondNet V/W predictor.** k = 6: -0.06 of the exact capture (fail; scalar predictor 0.48). k = 3: 0.92
(scalar 0.89). The extra per-bond input did not help the V/W prediction either.

## Constant-hop loop with residual re-bases (8 iterations, 2 seeds)
G_{k+1} = G_k + h_k(x; spins, log a_P, V_P, W_P). h_k is a fresh 37k FEAT net fit to la_w - G_k (edge loss, sampling
by the increment's node weight, Adam 1e-3, 20k steps). The write-back is unchanged (FEAT net on G_k's own log a, V, W).

| it | frac s0 / s1 | <H>(G) s0 / s1 | E_FN s0 / s1 | re-base kept s0 / s1 | G + 1 Lanczos s0 / s1 | G + 2 Lanczos s0 / s1 | table <H> / E_FN |
|---|---|---|---|---|---|---|---|
| 1 | 0.72 / 0.75 | 9.90e-5 / 9.65e-5 | 7.79e-5 / 7.60e-5 | 0.98 / 1.00 | 3.09e-5 / 3.02e-5 | 1.16e-5 / 1.14e-5 | 9.67e-5 / 7.62e-5 |
| 2 | 0.69 / 0.78 | 8.03e-5 / 7.90e-5 | 6.55e-5 / 6.44e-5 | 0.95 / 0.78 | 2.69e-5 / 2.71e-5 | 1.00e-5 / 1.00e-5 | 7.87e-5 / 6.44e-5 |
| 3 | 0.75 / 0.00* | 6.98e-5 / 7.89e-5 | 5.82e-5 / 6.44e-5 | 0.66 / - | 2.56e-5 / 2.69e-5 | 9.44e-6 / 9.96e-6 | 6.29e-5 / 5.25e-5 |
| 4 | 0.61 / 0.71 | 6.50e-5 / 6.76e-5 | 5.47e-5 / 5.68e-5 | 0.50 / 0.81 | 2.44e-5 / 2.49e-5 | 9.13e-6 / 9.22e-6 | 5.20e-5 / 4.40e-5 |
| 5 | 0.73 / 0.76 | 5.93e-5 / 5.73e-5 | 5.05e-5 / 4.86e-5 | 0.51 / 0.85 | 2.31e-5 / 2.21e-5 | 8.81e-6 / 8.21e-6 | 4.79e-5 / 4.08e-5 |
| 6 | 0.66 / 0.69 | **5.68e-5 / 5.45e-5** | 4.83e-5 / 4.64e-5 | 0.29 / 0.32 | 2.22e-5 / 2.17e-5 | 8.67e-6 / 8.20e-6 | 4.03e-5 / 3.47e-5 |
| 7 | 0.73 / 0.59 | 5.06e-5 / 5.34e-5 | 4.33e-5 / 4.54e-5 | 0.68 / 0.17 | 2.04e-5 / 2.12e-5 | 8.21e-6 / 8.19e-6 | 3.64e-5 / 3.15e-5 |
| 8 | 0.68 / 0.67 | 5.53e-5 / 5.04e-5 | 3.97e-5 / 4.28e-5 | -0.63 / 0.39 | 2.65e-5 / 2.02e-5 | 1.18e-5 / 8.02e-6 | 3.16e-5 / 2.75e-5 |

\* Seed 1, iteration 3: the write-back fit died again (f = 0, own loss flat; the same initialisation seed 1003 as in
round 2). This is an Adam/initialisation failure, worth a restart rule (re-seed when the own loss does not move).
- **Pre-registered rule: PASS.**
  - Share of the table loop's gain at iteration 6: 0.819 (seed 0) and 0.845 (seed 1), mean 0.83 >= 0.70, both
    >= 0.60.
  - <H> falls in 6 of 6 iterations for both seeds (seed 1 iteration 3 by 6e-8, the dead fit).
  - At iteration 8: 0.76 / 0.81. Seed 0's last re-base lost 63% of that iteration's gain, a single bad fit.
- E_FN of the callable guide at iteration 8: 3.97e-5 / 4.28e-5 (table 2.75e-5). That is below the RBM+PP <H>
  (4.47e-5). <H> itself is still 5.0-5.5e-5.
- **The re-base is still the bottleneck.** Its retention drops from ~0.9 (iterations 1-2) to 0.3-0.7 later, with
  single bad fits. The write-back keeps 0.59-0.78 every time.
- **Against Lanczos at equal hop count:** guide + 1 Lanczos step reaches 2.0e-5 at iteration 8, below psi_P + 1
  Lanczos (2.55e-5) but above psi_P + 2 Lanczos (8.6e-6). The guide costs one hop of psi_P, so the fair comparison
  for "guide + 1 step" is psi_P + 2 steps: the loop still loses. Guide + 2 Lanczos is 8.0e-6 at three hops.
- **Cost per sample, independent of k:**
  - guide: one hop of psi_P (78 / ~135 evaluations at 6x6 / 8x8) plus k small nets on the same features
    (37k-parameter CNNs; negligible next to the ViT);
  - its V, W: two hops (2.8e3 / 8.9e3);
  - an edge-loss write-back sample: three hops.
  At 1.9e4 psi_P/s per A40 that is ~4 ms / 0.15 s / ~5 s per sample at 6x6.

## B's V/W replay (bug fixed: V can be -1e-17 by round-off)

| scale c | log V / log W rms | kept |
|---|---|---|
| 0.03 | 0.0015 / 0.0007 | 1.01 |
| 0.1 | 0.0050 / 0.0024 | 0.76 |
| 0.3 | 0.015 / 0.0072 | -2.3 |
| 1 | 0.050 / 0.024 | -50 |

- White V/W noise at log V 0.001 / 0.003 / 0.010 keeps 0.98 / 0.83 / -0.79, so B's structured error is somewhat more
  tolerable than white.
- At full size (c = 1) B is 15-50x off.

## Next step (proposal)
1. Make the residual re-base robust:
   - restart rule for dead fits (re-seed if the own loss has not fallen 10% by 2k steps);
   - accept a re-base only if its exact-identity frozen-FN estimate on held-out batches improves; otherwise keep the
     previous h, which costs nothing;
   - 40k steps when the increment is small.
   This attacks the 0.3-0.7 late retention and the single bad fits (iteration 8, seed 0).
2. Then the 8x8 one-iteration test with the residual-re-base guide, against the projected ViT + Lanczos p1/p2 at
   equal hops. On 6x6 the loop does not yet beat psi_P + Lanczos at equal hops, so 8x8 should be framed as the loop's
   add-on value (G + Lanczos vs psi_P + Lanczos at the same hops), not as a standalone win.

## Compute (round 3)
24 GB slices only:
- pre-tests: BondNet re-base 0.25 h, BondNet V/W 0.6 h, G_3 0.15 h, G_5 0.15 h, residual 0.3 h;
- B fix 0.25 h;
- loops 2 x 1.47 h.
About 4.6 slice-h, ~2.3 A40-equivalent.

# Round 4 (2026-10-12): robust residual re-base, 3 seeds
Pre-registration ce63e83 (`experiments/compress_6x6/README.md`, round 4). Figure `fig_round4_6x6.png/.pdf`. Raw:
`runs/cx_chop2_s{0,1,2}.json`, `runs/cx_cost.json`, `runs/cx_lanczos_ref.json`.

## What was run, step by step
1. **Start.** The guide is the symmetrised ViT psi_P (exact 6x6 tables). Its sign comes from an exact Krylov sign step
   after every iteration, as in the table loop.
2. **Write-back, each iteration.**
   - From the current callable guide G_k, compute its own one-hop quantities log a, V and W. V is the weight of the
     guide's sign-violating neighbours; W is that of its sign-allowed neighbours.
   - Compute the semi-implicit FN target T from them.
   - Train a fresh 37k-parameter FEAT net (a CNN with log a, V, W as inputs) to T: edge (Dirichlet) loss, 20k Adam
     steps, lr 3e-3.
3. **Residual re-base.**
   - Train a second fresh 37k net h_k, seeing only the spins and psi_P's own log a, V, W (one hop of psi_P), on the
     increment just added.
   - The new callable guide is G_{k+1} = G_k + h_k. Settings: 20k steps (k <= 2) or 40k (k >= 3), lr 1e-3.
4. **Robustness, applied to both fits.**
   - 20% of the configurations are held out of training.
   - Gradient clipping (norm 1).
   - A fit whose loss does not move is re-seeded (up to 3 tries, lr halved each time).
   - A fit is accepted only if its loss on the held-out configurations beats "no change".
5. **Measurements, exact, per iteration:**
   - capture (share of the ideal FN iteration the write-back keeps);
   - re-base retention (share of that iteration's <H> gain kept by G_{k+1});
   - <H> and E_FN (the fixed-node energy, a variational bound, with G_{k+1} as guide);
   - G_{k+1} + 1 and + 2 exact Lanczos steps (one Lanczos step = one application of H, i.e. one more hop).
6. **References.** psi_P + p exact Lanczos steps, p = 0..4: 1.319e-4 / 2.551e-5 / 8.59e-6 / 3.82e-6 / 1.95e-6.
   Measured evaluation cost vs k.

## Result
**One line: PASS. The callable constant-hop guide keeps 85% of the table loop's gain at iteration 6 and 86% at
iteration 8 (3-seed means), but at equal hop count it is ~2x worse than psi_P + Lanczos.**

| it | capture s0/s1/s2 | re-base retention s0/s1/s2 | <H> s0/s1/s2 | E_FN s0/s1/s2 | table <H> / E_FN |
|---|---|---|---|---|---|
| 1 | 0.68/0.67/0.64 | 0.98/0.99/0.98 | 1.004/1.006/1.024e-4 | 7.89/7.89/8.01e-5 | 9.67e-5 / 7.62e-5 |
| 2 | 0.66/0.72/0.70 | 0.85/0.83/0.84 | 8.42/8.31/8.47e-5 | 6.86/6.78/6.89e-5 | 7.87e-5 / 6.44e-5 |
| 3 | 0.74/0.75/0.76 | 0.68/0.67/0.67 | 7.29/7.20/7.32e-5 | 6.07/5.98/6.07e-5 | 6.29e-5 / 5.25e-5 |
| 4 | 0.69/0.72/0.60 | 0.64/0.87/0.62 | 6.52/6.12/6.71e-5 | 5.52/5.18/5.65e-5 | 5.20e-5 / 4.40e-5 |
| 5 | 0.74/0.74/0.75 | 0.81/0.42/0.46 | 5.65/5.68/6.17e-5 | 4.83/4.85/5.23e-5 | 4.79e-5 / 4.08e-5 |
| 6 | 0.58/0.65/0.54 | 0.30/0.68/0.73 | **5.44/5.15/5.66e-5** | 4.65/4.41/4.83e-5 | 4.03e-5 / 3.47e-5 |
| 7 | 0.66/0.88*/0.01 | 0.18/0.73/- | 5.30/4.46/4.84e-5 | 4.51/3.84/4.19e-5 | 3.64e-5 / 3.15e-5 |
| 8 | 0.69/0.64/0.71* | 0.25/0.39/0.64 | **5.10/4.23/4.40e-5** | 4.34/3.62/3.78e-5 | 3.16e-5 / 2.75e-5 |

\* the first write-back try stalled (dead fit: own loss did not move); it was detected and re-seeded, and the second
try was accepted.
- **Pass rule.**
  - Share of the table loop's <H> gain at iteration 6: 0.846 / 0.878 / 0.822, mean 0.849 (bar 0.70, no seed below
    0.60).
  - At iteration 8: 0.807 / 0.893 / 0.877, mean 0.859 (bar 0.70). **PASS.**
- **Robustness.**
  - The dead-initialisation failure (round 2/3: seed 1, iteration 3) did not recur with gradient clipping. Two other
    stalls (seed 1 iteration 7, seed 2 iteration 8) were caught and re-seeded.
  - No re-base was rejected, and no retention is negative any more; the worst is 0.18 (round 3: -0.63).
  - Remaining weakness: seed 2 iteration 7, an accepted write-back that captured only 0.01. The held-out acceptance
    only tests "better than nothing".
- Seeds 1 and 2 reach <H> 4.2-4.4e-5 at iteration 8, below RBM+PP (4.47e-5). E_FN is 3.6-4.3e-5.
- **Equal-hop comparison.** The callable guide costs one hop of psi_P; + p Lanczos steps make 1 + p hops.

| hops of psi_P | guide (it 8) + Lanczos, s0/s1/s2 | psi_P + Lanczos |
|---|---|---|
| 1 | 5.10 / 4.23 / 4.40e-5 (guide alone) | 2.55e-5 (p = 1) |
| 2 | 2.02 / 1.76 / 1.80e-5 (+1 step) | 8.59e-6 (p = 2) |
| 3 | 8.10 / 6.98 / 7.06e-6 (+2 steps) | 3.82e-6 (p = 3) |

  psi_P + Lanczos wins by ~2x at every hop count. The guide's own sign uses an exact global Krylov step, an advantage
  the Lanczos column does not get.
- **Cost per configuration vs k** (measured on a 24 GB A40 slice: one symmetrised psi_P evaluation 5.4e-5 s, one
  small net 1.1e-5 s):

| | k = 0 | k = 4 | k = 8 | small-net share at k = 8 |
|---|---|---|---|---|
| 6x6 guide value (78 psi_P + k small nets) | 4.19e-3 s | 4.23e-3 s | 4.28e-3 s | 2.1% |
| 6x6 guide V, W (2.8e3 psi_P + 78 k small nets) | 0.152 s | 0.156 s | 0.159 s | 4.3% |
| 8x8 guide value (135 psi_P + k) | 7.25e-3 s | 7.29e-3 s | 7.34e-3 s | 1.2% |
| 8x8 guide V, W (8.7e3 psi_P + 135 k) | 0.470 s | 0.476 s | 0.482 s | 2.5% |

  The cost is linear in k with a small slope; psi_P's hops dominate. An edge-loss training sample costs <= 9x the
  V, W cost (an upper bound; shared neighbours reduce it).
  - 8x8 rows: measured 6x6 throughputs x 8x8 hop counts. The 8x8 ViT is larger, so these are lower bounds for the
    psi_P part.
- **Convergence.** Every fit was trained to its step budget; the held-out losses are in `fit_w` / `fit_r` of each
  iteration's record. The late re-bases still keep only 0.2-0.7 of their iteration's gain. That, not the write-back
  (0.54-0.76 apart from the one stalled fit), is what separates the callable guide from the table loop.

## Compute (round 4)
24 GB slices: three loops 1.71 / 2.64 / 1.75 h, cost 0.01 h, Lanczos reference 0.01 h. One duplicate Lanczos
reference ran on a full A40 (0.01 h). Total 6.1 slice-h (~3 A40-equivalent).

## Next step (proposal)
- The constant-hop guide now works on 6x6 (85% of the table loop at bounded cost). Its value is as a starting point,
  not against Lanczos: at equal hops Lanczos on psi_P is 2x better.
- The fair add-on comparison at 8x8 is "guide after k iterations + p Lanczos" vs "psi_P + (p+1) Lanczos", with the
  calibrated FN referee.
- Code for the 8x8 worker: `cx_chop2.py` + `robust_fit` / `fit_edge` / `tail_proposal` / `FeatNet` in `cx_lib.py`.

# Step size, stage 2 (2026-10-13): callable loop with the clipped-Lanczos target
Pre-registration and amendments S2-1 and S2-2: `experiments/compress_6x6/README.md` ("Step size"). Code `cx_chop3.py`.
Figure `fig_step2_6x6.png/.pdf` (`cx_figure5.py`).
Raw files:
- `runs/cx_chop3_s{0,1}.json`: main arm;
- `runs/cx_chop3_pt_{f3,f4,r60k}.json`: k = 1 pre-tests;
- `runs/cx_chop3_prev_s0.json`: sign-matched re-base test.

## What was run, step by step (exact 6x6 tables, J2/J1 = 0.5)
1. **Start.** The guide is psi_P, the symmetrised ViT: <H> - E0 = 1.319e-4 per site.
2. **Target, each iteration k.** From the current callable guide G_k, with the sign s_k kept as a table:
   - the exact one-hop quantities V, W and the local energy E_L = H_xx + V - W;
   - the clipped-Lanczos factor f = 1 + alpha (E_L - E). Its step alpha is energy-optimal, found exactly on the tables
     before any fit;
   - the target log-amplitude change d = log max(|f|, 0.3) and the new sign s_k sgn(f).
3. **Write-back.**
   - A 37k-parameter FEAT CNN sees the spins and G_k's own log a, V, W and asinh-compressed E_L (amendment S2-1).
   - It is trained to d with the edge (Dirichlet) loss in the metric of the target state. Adam, 20k steps, lr 3e-3,
     gradient clipping, 20% of configurations held out. A fit is accepted only if its held-out loss beats "no change".
   - Result: la_w = G_k + d_fit.
4. **Residual re-base.**
   - A second 37k net h_k sees only one hop of psi_P: log a_P, V_P, W_P and asinh E_L,P.
   - It is fit to the increment la_w - G_k in the metric of la_w. Adam, lr 1e-3, 20k steps (k <= 2) or 40k (k >= 3).
   - The new callable guide is G_{k+1} = G_k + h_k. It costs one hop of psi_P plus k small nets.
5. **Exact measurements, every iteration:**
   - <H> of the exact target, of la_w and of G_{k+1};
   - capture = (write-back gain) / (exact gain);
   - retention = (gain after re-base) / (write-back gain);
   - share = capture x retention;
   - E_FN (fixed-node energy, a variational bound);
   - G_{k+1} + 1 and + 2 exact Lanczos steps;
   - sign error w_s (ground-state weight on wrong signs);
   - held-out loss curves.
6. Two seeds x 4 iterations.

## Result
**One line: FAIL at the pre-registered bar (share >= 0.70 at every k). k = 1 passes (0.84), k = 2-4 do not (0.66, 0.31,
0.09). The write-back is near-exact from k = 2 on; the gain is lost in the re-base, whose error stays at a fixed absolute
size (~0.013 bond-rms) while the increments shrink.**

| k | alpha | flips | capture s0/s1 | retention s0/s1 | share s0/s1 (mean) | <H> s0/s1 | stage-1 exact | E_FN s0/s1 | w_s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.156 | 0.24 | 0.84/0.84 | 1.00/1.00 | 0.84/0.84 (**0.84**) | 4.34/4.27e-5 | 2.62e-5 | 2.94/2.92e-5 | 7.2e-6 |
| 2 | -0.075 | 0.09 | 0.97/0.96 | 0.69/0.67 | 0.67/0.65 (**0.66**) | 2.50/2.53e-5 | 1.16e-5 | 1.89/1.91e-5 | 4.7e-6 |
| 3 | -0.104 | 0.11 | 1.00/1.00 | 0.33/0.29 | 0.33/0.29 (**0.31**) | 2.06/2.14e-5 | 7.06e-6 | 1.58/1.64e-5 | 3.3e-6 |
| 4 | -0.099 | 0.13 | 1.01/1.02 | 0.09/0.08 | 0.09/0.08 (**0.09**) | 1.96/2.05e-5 | 4.91e-6 | 1.49/1.57e-5 | 2.4e-6 |

- **Secondary measures.** Trajectory share (start - <H>) / (start - stage 1): 0.84 / 0.89 / 0.89 / 0.88. Error ratio
  to stage 1: 1.6 / 2.2 / 3.0 / 4.1, growing.
- **Comparison.** Against round 4 (same machinery, semi-implicit target), one iteration now does what 8 iterations did
  there: <H> 4.3e-5 after k = 1, against 4.2-5.1e-5 after k = 8. Iterations 2-4 add little: the loop plateaus at
  ~2.0e-5.
- **Equal hop count.** The guide costs 1 hop of psi_P; each Lanczos step adds 1.

| hops | guide k = 4 (+0 / +1 / +2 Lanczos), s0 / s1 | psi_P + p Lanczos |
|---|---|---|
| 1 | 1.96 / 2.05e-5 | 2.55e-5 |
| 2 | 0.92 / 0.98e-5 | 0.86e-5 |
| 3 | 4.64 / 4.94e-6 | 3.82e-6 |

  At 1 hop the callable guide beats psi_P + one Lanczos step by 1.2-1.3x. This is the first callable guide to do
  so (round 4: 4.2-5.1e-5). At 2 hops it ties. At 3 hops Lanczos is 1.2-1.3x better.

### Where the gain is lost (diagnosis)
- **k = 1, write-back.**
  - With (log a, V, W) only: capture 0.25. The held-out loss floors at 2.6e-4 from 12k steps, a representation limit:
    d is a sharp function of E_L = D0 + V - W, and the net sees only log V and log W.
  - Adding asinh(E_L) as a column: capture 0.84 (loss 4.6e-5).
  - In the 60k control run the same arm reached held-out loss 7e-8 and capture 1.000. The remaining 0.16 at k = 1 is
    therefore optimisation variance between runs, not a limit.
- **k = 1, re-base.**
  - Without E_L,P: retention 0.57.
  - With E_L,P: retention 1.00 (loss 2e-4 of its start).
  - The 60k-step control without E_L lowered the loss 8x, but the guide blew up to <H> 0.61 per site: an error of
    +20.7 in log-amplitude on configurations that the target almost never samples, which carry 80% of the new guide's
    weight. A guard against this was added for later runs.
- **k >= 2, re-base (the plateau).**
  - The write-back is near-exact (capture 0.96-1.02, error bond-rms 0.006).
  - The re-base's held-out loss floors: final / start 0.35, 0.74, 0.95 at k = 2, 3, 4, flat over the last half of
    training.
  - Its error has a FIXED absolute size: bond-rms 0.012-0.014 at every k >= 2, while the increment itself shrinks
    (bond-rms roughness 0.028, 0.020, 0.018). So retention falls toward 0.
  - The error is spread over the bulk, not tail blow-ups: < 3e-7 of the weight sits on configurations with error > 1.
- **Which information is missing.**
  - Hypothesis H1 (amendment S2-2): the re-base features split psi_P's neighbours by psi_P's own sign. 24% + 9% of
    configurations have flipped by k = 2.
  - Test: the same features with the neighbours split by the guide's current sign. Retention at k = 2 fell to
    0.55 (main 0.69). **H1 refuted.** The run was stopped after k = 2, because the pre-registered reading was
    already decided.
  - Remaining cause, consistent with round 3's loss floor (~5e-6 there):
    - d_k depends on the guide's own amplitude ratios at the neighbours, G_k(y) - G_k(x). These include the earlier
      increments h_j(y).
    - The h_j are functions of psi_P's one-hop features at y, i.e. two-hop information of psi_P that a
      one-hop-of-psi_P re-base cannot see.
  - The 4-feature write-back on G_k's own one-hop quantities captures ~1.0. So the information IS in one hop of
    G_k, i.e. two hops of psi_P.

### Convergence evidence
- Write-back losses flatten by 12-16k of 20k steps. Held-out / training loss ratio 1.0-1.3: no overfitting.
- Re-base losses at k >= 3 are flat for the last 20k of 40k steps.
- Every fit was accepted on the first try. No stalls and no blow-ups in the main arm; the guard never fired in S2-2.

## Compute
24 GB A40 slices:
- smoke 0.20 h;
- pre-tests 0.47 + 0.44 + 0.28 h;
- main arm 0.87 + 0.94 h;
- S2-2 test 0.93 h.

Total 4.1 slice-h (budget 8).

## Open problems / next
1. **Two-hop re-base.** Give h_k the one-hop quantities of a FIXED two-hop anchor, e.g. G_1 (V_1, W_1, E_L,1). The
   cost is constant in k (two hops of psi_P) and would test the remaining cause directly. Alternatively, stop re-basing
   and accept k-hop guides for small k (k <= 2: guide + 1 Lanczos already ties psi_P + 2 Lanczos).
2. **k = 1 write-back optimisation variance** (capture 0.84 vs 1.00 at identical settings): lower lr or several
   restarts selected by held-out loss.
3. **The blow-up failure mode** of edge-loss fits on unsampled configurations (seen once at 60k steps) needs a sampled
   guard at scale; the exact-table guard used here is not available there.
4. **Scaling claim.** At 6x6 a single clipped-Lanczos iteration gives a 1-hop callable guide better than psi_P + 1
   Lanczos. Whether this survives at 8x8 (where the SI target was 3.6x over-damped) is the decisive next test, and
   it needs only k = 1.

# Fold-back diagnostics D1 / D2 (2026-10-13)
Pre-registration and amendment FB-1: `experiments/compress_6x6/README.md` ("Fold-back diagnostics D1 / D2"). Code:
`cx_lightcone.py`, `cx_pairint.py`. Raw files: `runs/cx_lightcone_D1.json`, `runs/cx_lightcone_D2.json`,
`runs/cx_pairint.json`. Compute: 0.80 + 0.41 + 0.01 slice-h.

## What was run, step by step
1. **Rebuild the stage-2 state at iteration k = 3** (seed 0). The guide is G_3 = log|psi_P| + h_1 + h_2 (stored nets),
   with sign s_3.
   - The exact clipped-Lanczos target d_3 was recomputed.
   - Sanity checks passed: <H> 2.4951e-5 and target 1.1876e-5 reproduce the run to all printed digits.
2. **D1: four re-base fits of the exact d_3.** Same 37k net, same initialisation, Adam lr 1e-3, held-out acceptance.
   - R0: psi_P's one-hop features (round 4 / stage 2).
   - RA: R0 plus G_3's own local energy as an extra input.
   - RB: R0 with the target scaled to unit size.
   - RC: R0 with 3x the steps (120k).
3. **D2: exact light-cone test, no fits.** On 2048 samples of the target state I rebuilt G_3's local energy at x from
   G_3 at every neighbour x^b.
   - The neighbours' psi_P features were computed with psi_P at the second-hop configurations x^{bb'}, taken exact
     only when bond b' lies within distance rho of b. Beyond rho they come from the additive (first-hop-only)
     approximation.
   - From each rebuilt local energy, the induced error of the target d_3 was computed.
4. **D2a / FB-1: pair interaction I(b, b').** This is the departure from additivity, la(x^{bb'}) + la(x) - la(x^b) -
   la(x^{b'}), together with the analogous sign product, by bond distance.
   - Measured for psi_P and for the exact ground state psi0, on 1024 samples each.

## Result
**One line: the fold-back is information-limited; the missing information is the guide's own local energy (two
hops of psi_P). At 6x6 it is not local: the exact ground state itself has O(1) pair interactions at every distance
on the lattice, and the light-cone approximation is worse than no neighbour information.**

| D1 arm | retention (exact-target share) | <H> | re-base bond-rms error | held-out loss final / start |
|---|---|---|---|---|
| main stage-2 arm (write-back output as target) | 0.33 | 2.06e-5 | 0.0149 | 0.74 |
| R0: psi_P one hop | 0.018 | 2.47e-5 | 0.0169 | 0.98 |
| **RA: + G_3's own E_L** | **0.989** | **1.20e-5** | **0.0057** | **0.11** |
| RB: R0, target scaled to unit rms | 0.331 | 2.06e-5 | 0.0147 | 0.75 |
| RC: R0, 120k steps | 0.024 | 2.46e-5 | 0.0169 | 0.98 |

Pre-registered readings:
- **(I) information: CONFIRMED.** RA = 0.99 >= 0.85, while R0, RB and RC are all <= 0.5. With the right input, the
  same net, optimiser and step count fit the target to a tenth of its held-out loss.
- **(F) optimiser floor: refuted.** RC: 3x the steps changes nothing (0.024).
- **(S) target scale: a real but secondary effect.**
  - The exact target is small (pointwise rms 0.0065). At its raw scale the Adam fit does not move.
  - Normalising it lifts retention from 0.02 to 0.33, the main-arm value, but not to 0.7.
  - Normalising targets costs nothing and should be the default in every fit.

| D2 level (target-state samples) | exact psi_P evaluations per bond | error of E_L,3 | induced error of d_3 (rms) |
|---|---|---|---|
| no neighbour information (u(x^b) := u(x)) | 0 | 0.16 | 0.017 |
| h_j at x^b with x's features | 0 | 0.16 | 0.018 |
| rho = 0 (shared-site pairs exact) | 6.7 | 1.23 | 0.112 |
| rho = 1 | 25 | 0.41 | 0.039 |
| rho = sqrt2 | 38 | 0.31 | 0.037 |
| rho = 2 | 52 | 0.27 | 0.027 |
| exact (all 80) | 80 | 1e-7 | 1e-8 (sanity) |

- **Light cone at 6x6: NOT viable** (pre-registered: some level with rho <= sqrt2 at <= 0.25 x the no-information
  error).
  - Every partial level is worse than using no neighbour information.
  - Reason: the additive approximation corrupts the asinh E_L feature of x^b by 0.7-1.5 standard deviations even at
    rho = 2.
- For scale:
  - the main re-base's pointwise error is 0.0041 and the target's own rms is 0.0065;
  - on the fit proposal (tail) samples all levels are far off (d error 0.55-1.1).

| bond distance | 1 | sqrt2 | 2 | sqrt5 | 2.83 | 3 | 3.16 | 3.61 |
|---|---|---|---|---|---|---|---|---|
| rms I, psi_P | 1.68 | 1.28 | 1.02 | 0.95 | 0.82 | 0.61 | 0.43 | 0.75 |
| rms I, exact psi0 | 1.69 | 1.29 | 1.02 | 0.94 | 0.80 | 0.61 | 0.42 | 0.72 |
| sign non-additivity, psi0 | 19% | 13% | 10% | 9% | 6% | 2.6% | 1.6% | 5.4% |

- **FB-1 reading: the non-additivity belongs to the 6x6 ground state, not to the network.** psi0 and psi_P agree to
  2%.
- So no base network can make the light cone local at 6x6. Whether the interactions decay at 8x8 and beyond, at the
  near-critical J2 = 0.5, is open and needs a sampled 8x8 measurement.

# Rolling re-anchor test (2026-10-13)
Pre-registration and amendments RA-1, RA-2: `experiments/compress_6x6/README.md`. Code: `cx_reanchor.py`,
`cx_reanchor_diag.py`. Raw files: `runs/cx_reanchor_ra_{ms_s0,ms_s1,ad_s0}_k1.json`, `runs/cx_reanchor_diag.json`.
Compute: 2.55 slice-h (budget 4). This covers the two minSR jobs cancelled at 6 min by amendment RA-1.
The k = 3 seed-1 job crashed reading the k = 2 "stopped" record; it would not have run anyway.

## What was run, step by step
1. **Start.** The stage-2 start: psi_P with its sign, <H> 1.319e-4.
2. **Target.** The exact clipped-Lanczos target: alpha -0.156, 24% of configurations flip sign, target <H> 2.62e-5.
3. **Fine-tune.** The whole symmetrised ViT (155k parameters), warm-started, was fine-tuned for 12 min to the
   target log-amplitude with the stage-2 edge loss.
   - Sampling: increment-weighted, 20% of orbits held out.
   - minSR (damped Gauss-Newton with value rows, trust region, acceptance on a fixed validation set): seeds 0 and 1.
   - Adam at lr 1e-6: seed 0.
4. **Exact evaluation.** The network was tabulated on all 15.8M orbits and its energy computed exactly with the
   target sign.
5. **Splice diagnostic.** Each fitted table was spliced with the exact target by probability decade, and the exact
   energy of each splice computed.

## Result
**One line: FAIL. Every arm makes the energy worse at k = 1 (share -0.14 / -0.12 minSR, -0.10 Adam; pre-registered
fallback triggered). The cause is located: the fit is good in the bulk and wrong in the tail (configurations
with a^2 < 1e-9, 7.7% of the weight), where the step's gain lives.**

| arm | held-out edge loss end / start | share | <H> | bond-rms error | E_FN | + 1 Lanczos |
|---|---|---|---|---|---|---|
| start (psi_P, old sign) | | 0 | 1.319e-4 | 0.052 | 1.02e-4 | 2.55e-5 |
| old amplitude, new sign | | -0.03 | 1.347e-4 | | | |
| minSR seed 0 (60 steps, acceptance 0.27) | 0.50 | **-0.139** | 1.467e-4 | 0.047 | 9.26e-5 | 4.11e-5 |
| minSR seed 1 (160 steps, acceptance 0.14) | 0.52 | **-0.119** | 1.446e-4 | 0.048 | 9.24e-5 | 4.20e-5 |
| Adam lr 1e-6 (1108 steps) | 0.67 | **-0.097** | 1.422e-4 | 0.055 | 9.74e-5 | 4.39e-5 |
| exact target | | 1 | 2.62e-5 | 0 | | |

Pointwise error by probability decade (minSR seed 1; the target's own size is 0.031 overall):

| decade of a_t^2 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 1e-10 | 1e-11 | 1e-12 | 1e-13 | 1e-14 | 1e-15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.006 | 0.003 | 0.003 | 0.006 | 0.013 | 0.034 | 0.090 | 0.21 | 0.39 | 0.64 | 0.91 |

Splices (share; identical pattern for all three arms, minSR seed 1 shown):

| cut: a_t^2 below the cut is the tail | 1e-7 | 1e-8 | 1e-9 | 1e-10 | 1e-11 | 1e-12 |
|---|---|---|---|---|---|---|
| tail weight | 52% | 24% | 7.7% | 1.9% | 0.37% | 0.06% |
| fit in the bulk + exact target in the tail | 0.99 | 0.97 | 0.91 | 0.76 | 0.52 | 0.29 |
| exact target in the bulk + fit in the tail | -0.12 | -0.11 | -0.09 | -0.02 | 0.16 | 0.43 |
| fit in the bulk + start in the tail | -0.04 | -0.05 | -0.06 | -0.06 | -0.02 | +0.02 |

- **What this means.**
  - The ViT fine-tune reproduces the increment well where a_t^2 >= 1e-9. With the tail exact, 0.91-0.99 of the
    gain would be kept.
  - The step's gain needs the amplitude to drop near the new nodes (24% sign flips). That happens in the tail, and
    the network does not follow it there: its errors grow to 0.1-0.9.
  - The tail cannot simply be left alone either: bulk fit + old tail gives about -0.05, because the new sign needs
    the new tail amplitudes.
- This is the same barrier as writeback_tail (spin-only nets 20-31%) and P2 (psi_P itself is off by 0.05-0.5 in
  these decades).
- Not shown to be a conceptual limit: no tail-steered sampling or extra capacity was tried. But it is consistent
  across optimisers, seeds and three earlier campaigns.
- Method checks:
  - fp32 floor: edge rms 7e-6 against an increment of 0.042 (ratio 2e-4), so the test was meaningful;
  - blow-up guard not triggered (weight 1.3-1.7e-4);
  - all fits accepted on held-out data.

# Sign-channel check (2026-10-13, lead request)
Pre-registration: `experiments/compress_6x6/README.md` ("Sign-channel check"). Code `cx_signchan.py`. Raw files:
`runs/cx_signchan_RA.json`, `runs/cx_signchan_S2.json`. Compute: 0.70 slice-h.

## What was run, step by step
1. **Pairs.** Each pair has a fitted guide and the exact amplitude it was meant to store, with the same current sign:
   - the three k = 1 re-anchored ViTs vs the exact k = 1 target;
   - stage-2 G_2 vs the exact k = 1 target;
   - stage-2 G_3 vs the exact k = 2 target.
2. **Next sign step from each amplitude.** Rule C: clipped-Lanczos sign. Rule K: Krylov sign, threshold optimal at
   fixed amplitude.
3. **Scoring.** The exact fixed-node energy (E_FN) of each new sign, computed with the exact reference amplitude as
   the common guide. Also each guide's own E_FN.
4. **Iteration.** For one re-anchored ViT and for G_3: two more sign steps, with the exact FN amplitude of each
   chain's own sign between steps.

## Result
**One line: formally FAIL at step 1, where the absolute sign gains are tiny; PASS once an FN amplitude step
follows (step 2-3 ratios 0.92-1.02). The guide's OWN E_FN is dominated by its amplitude (4-5x), so the premise
"E_FN depends only on the sign" does not hold for lattice FN.**

| pair | rule | E_FN start (common amp.) | fitted-amp. sign | exact-amp. sign | ratio of gains | disagreeing weight | own E_FN of fitted guide + its new sign |
|---|---|---|---|---|---|---|---|
| re-anchor minSR s1 | C | 1.951e-5 | 2.291e-5 | 1.929e-5 | -15.6 | 4.4e-6 | 9.28e-5 |
| re-anchor Adam | C | 1.951e-5 | 2.369e-5 | 1.929e-5 | -19.2 | 4.9e-6 | 9.73e-5 |
| re-anchor minSR s0 | C | 1.951e-5 | 2.244e-5 | 1.929e-5 | -13.4 | 4.1e-6 | 9.32e-5 |
| re-anchor minSR s1 / Adam / s0 | K | 1.951e-5 | 2.108 / 2.181 / 2.077e-5 | 1.900e-5 | -3.1 / -4.5 / -2.5 | 2.0-2.8e-6 | 9.1-9.6e-5 |
| stage-2 G_2 | C | 1.951e-5 | 1.933e-5 | 1.929e-5 | **0.83** | 1.6e-6 | 2.89e-5 |
| stage-2 G_2 | K | 1.951e-5 | 1.925e-5 | 1.900e-5 | 0.50 | 5.6e-7 | 2.87e-5 |
| stage-2 G_3 | C | 1.258e-5 | 1.247e-5 | 1.222e-5 | 0.29 | 4.5e-7 | 1.90e-5 |
| stage-2 G_3 | K | 1.258e-5 | 1.230e-5 | 1.219e-5 | 0.71 | 2.4e-7 | 1.86e-5 |

Iterated with exact FN amplitudes between steps (rule C), E_FN after each step; gains are measured from the common
start:

| chain | step 1 | step 2 | step 3 | ratio of gains at steps 2 / 3 |
|---|---|---|---|---|
| re-anchor minSR s1: fitted-amp. first sign | 2.291e-5 | 1.378e-5 | 1.126e-5 | **0.95 / 0.92** |
| same, exact-amp. first sign | 1.929e-5 | 1.347e-5 | 1.055e-5 | |
| stage-2 G_3: fitted-amp. first sign | 1.247e-5 | 9.27e-6 | 7.52e-6 | **1.02 / 0.98** |
| same, exact-amp. first sign | 1.222e-5 | 9.32e-6 | 7.45e-6 | |

<H> and E_FN side by side. Every E_FN here is the guide's own fixed-node energy, per site, minus E0.

| guide | <H> | E_FN |
|---|---|---|
| psi_P (start) | 1.319e-4 | 1.02e-4 |
| exact k = 1 target (one hop of psi_P) | 2.62e-5 | 1.95e-5 |
| exact k = 2 target from G_2 | 1.58e-5 | 1.26e-5 |
| stage-2 seed 0, it 1..4 | 4.34 / 2.50 / 2.06 / 1.96e-5 | 2.94 / 1.89 / 1.58 / 1.49e-5 |
| stage-2 seed 1, it 1..4 | 4.27 / 2.53 / 2.14 / 2.05e-5 | 2.92 / 1.90 / 1.64 / 1.57e-5 |
| re-anchor minSR s0 / s1 / Adam (k = 1) | 1.467 / 1.446 / 1.422e-4 | 9.26 / 9.24 / 9.74e-5 |
| RBM+PP | 4.47e-5 | |

## Readings
- **Step 1 fails by the pre-registered bar, but the absolute numbers are small.**
  - With the exact amplitude as FN guide, the next sign step gains only 2-5e-7 from the k = 1 target and 3.5-3.9e-7
    from the k = 2 target.
  - The fitted amplitude costs 1.3-4.2e-6 for the re-anchored ViTs, but only 0.4-2.6e-7 for the stage-2 guides.
- **Sign errors from imperfect amplitudes do not accumulate if an exact FN amplitude step follows.** At steps 2-3
  both chains agree within 2-8%.
- **The lattice E_FN of a guide depends strongly on its amplitude.** Same sign s_1:
  - 1.95e-5 with the exact target amplitude;
  - 2.94e-5 with stage-2 G_2;
  - 9.2-9.7e-5 with the re-anchored ViTs.
  - Cause: the wall term sum_violating |H| a(y)/a(x) uses the amplitude at the nodes, i.e. on rare configurations.
- So the rare-region amplitude problem is NOT irrelevant for an FN-reported energy, unless the reported E_FN comes
  from an amplitude that is itself the FN solution (self-consistent FN).
- With exact FN amplitudes the sign chain reaches 7.45e-6 after three steps from the k = 2 state. That is a table
  result (it needs phi_FN as the next guide).

# Hybrid re-anchor test (2026-10-13)
Pre-registration and amendment HY-1: `experiments/compress_6x6/README.md`. Code: `cx_hybrid.py`, `cx_reanchor.py`
(state / fit_only), `cx_hybrid_diag.py`. Raw files: `runs/cx_hybrid_k1.json`, `runs/cx_hybrid_k2.json`,
`runs/cx_hybrid_diag.json`. Compute: about 1.0 slice-h (budget 3).

## What was run, step by step
1. **The guide.** It is the re-anchored ViT plus one correction net (37k parameters). The net reads one hop of the
   current ViT (log A, log V, log W, asinh E_L), is fit to the whole residual between the exact target and the ViT,
   and is refit each iteration, not stacked.
2. **k = 1.** I reused the re-anchored ViT of the earlier test (same target), then fit the correction.
3. **k = 2.** The ViT was fine-tuned again to the new exact target (warm, minSR, 12 min), tabulated, and a new
   correction fit. Everything was evaluated exactly: <H> and the guide's own E_FN.
4. **k = 3.** Not run: the continuation rule needs share >= 0.3 at k = 2.
5. **Diagnostic (HY-1).**
   - Decade splices of the k = 2 guide with the exact target.
   - The same correction refit with ONE extra input, asinh E_L of the current guide G_2. That input costs a second
     hop.

## Result
**One line: k = 1 PASSES (share 0.96), k = 2 FAILS (0.23). The k = 2 shortfall is the one-shell information gap:
with the guide's own local energy as one extra input the same net reaches 0.92.**

| | share | <H> | own E_FN | ViT alone <H> / E_FN | + 1 Lanczos |
|---|---|---|---|---|---|
| hybrid k = 1 | **0.963** | 3.01e-5 | 2.17e-5 | 1.45e-4 / 9.24e-5 | 1.40e-5 |
| stage-2 k = 1 (same hop cost) | 0.84 | 4.34e-5 | 2.94e-5 | | |
| exact target k = 1 | 1 | 2.62e-5 | 1.95e-5 | | |
| hybrid k = 2 | **0.230** | 2.64e-5 | 1.71e-5 | 1.30e-4 / 8.17e-5 | 1.42e-5 |
| stage-2 k = 2 (same hop cost) | 0.66 | 2.50e-5 | 1.89e-5 | | |
| exact target k = 2 (from hybrid G_2) | 1 | 1.40e-5 | 1.08e-5 | | |
| HY-1 upper bound k = 2 (+ E_L of G_2, 2 hops) | **0.925** | 1.52e-5 | 1.11e-5 | | |

HY-1 splices at k = 2 (share), cut by decade of the target state:

| cut | 1e-8 | 1e-9 | 1e-10 | 1e-11 | 1e-12 |
|---|---|---|---|---|---|
| tail weight | 24% | 7.7% | 1.9% | 0.37% | 0.06% |
| fit in the bulk + target in the tail | 0.98 | 0.78 | 0.71 | 0.60 | 0.45 |

**Readings.**
- **The hybrid fixes iteration 1.** The ViT carries the bulk and the one-hop correction the tail: 0.96, against 0.84
  for stage 2 and -0.12 for the ViT alone.
- **From k = 2 on, the same one-shell gap as D1 returns, now in the tail.**
  - The k-th increment is a function of E_L of the full guide (ViT + correction).
  - That needs the correction at the neighbours, i.e. one hop of the ViT at the neighbours: two hops.
  - With it the fit reaches 0.92; without it, 0.23.
- **The ViT absorbs the bulk but not the previous correction's tail.** On its own, the ViT is far off at k = 2
  (1.30e-4).
- **On E_FN the hybrid beats stage 2 at the same hop cost** (2.17e-5 vs 2.94e-5 at k = 1; 1.71e-5 vs 1.89e-5 at
  k = 2). It still misses the exact targets (1.95e-5, 1.08e-5).

# Lanczos-wrapped network, pre-test W0 (2026-10-13)
Pre-registration: `experiments/compress_6x6/README.md` ("Lanczos-wrapped network"). Code `cx_wrap_pretest.py`.
Raw file: `runs/cx_wrap_pretest.json`. Compute: 0.1 slice-h. No fits.

## What was run
- The guide is G = (1 + beta (H - E)) N, with optimal beta (one exact Lanczos step on the table).
- N is either the exact k = 1 clipped-Lanczos target N* or the stored re-anchored ViT fit to it (N_fit).
- The two were spliced by probability decade, to see whether N's tail errors still matter once the wrapper is
  applied.

## Result
**One line: NOT SOUND (pre-registered). Wrapping does not make N's tail errors irrelevant: with N exact in the bulk
and the fit in the tail (cut 1e-9), the wrapped state keeps only 4-7% of the gain. In the wrapped exact state the
inherited term dominates almost nowhere (< 1e-4 of the weight down to a^2 = 1e-12).**

| N | wrapped <H> | wrapped own E_FN |
|---|---|---|
| exact target N* | 1.15e-5 | 9.15e-6 |
| ViT fit, minSR s1 / Adam | 4.20e-5 / 4.39e-5 | 3.15e-5 / 3.38e-5 |
| psi_P (= psi_P + 1 Lanczos) | 2.55e-5 | 1.89e-5 |

| cut | tail weight | fit bulk + exact tail (share_w, s1 / Adam) | exact bulk + fit tail (share_w, s1 / Adam) |
|---|---|---|---|
| 1e-8 | 24% | 0.97 / 0.97 | 0.02 / 0.01 |
| 1e-9 | 7.7% | 0.91 / 0.92 | **0.07 / 0.04** |
| 1e-10 | 1.9% | 0.77 / 0.78 | 0.20 / 0.17 |

Inheritance in the wrapped N*: share of each decade's weight where |beta (E_L - D)| > |1 + beta (D - E)|.

| decade | 1e-8 | 1e-10 | 1e-11 | 1e-12 | 1e-13 | 1e-14 | 1e-15 |
|---|---|---|---|---|---|---|---|
| weight | 29% | 5.8% | 1.5% | 0.31% | 0.05% | 0.006% | 0.0006% |
| inherited-dominant fraction | 0 | 1.5e-7 | 2e-6 | 1.1e-4 | 1.8e-3 | 0.018 | 0.071 |

**Reading.**
- Inheritance dominates only where a step CREATES new nodes, i.e. the configurations flipped by that step.
- Once a step is absorbed into N, those configurations are N's own tail. The next wrapper multiplies them by an O(1)
  factor and does not regenerate them.
- So the wrapper cannot supply the tail that N failed to learn, and training N through it cannot down-weight N's
  tail. No training run (per the pre-registration).
