# Write-back with energy-metric (tail) sampling, exact 6x6 oracle harness

Question (PI + Research Workspace seq 100, 2026-10-08): the write-back of one FN amplitude iteration into a residual
network failed (`results/writeback/DESIGN.md`: 17% of the gain with oracle data, < 1% realistic). Was the training
*distribution* the problem rather than the network? The FN update is white-noise rough in the energy metric and 81% of
its gain sits on 7.6% of the phi^2 weight, so samples x ~ phi^(2 beta) rarely visit the configurations that carry it.
Hypothesis: sample the energy metric itself (a mixture proposal with an explicit tail component proportional to the
per-configuration kept-edge residual weight, i.e. importance sampling of the quadratic form Q) and fit edge differences
of the log-amplitude with the FN graph-Laplacian weights.

Code: `wt_exact.py` (exact projections, no training), `wt_run.py` (training arms on the `writeback` harness),
`wt.sbatch`, `specs/`. Results: `results/writeback_tail/`.

## Definitions (all per site, exact 6x6 sector k = 0, A1, flip+; J2/J1 = 0.5)
- Guide (a, s) = (|psi_P|, s_P), symmetrised ViT. phi = Perron vector of H_FN[a, s]; delta = log phi - log a.
- K = allowed-edge (s_x s_y = -1) part of |H_off|. FN Laplacian L = diag(phi K phi) - Phi K Phi.
  Q(e) = e^T L e / N = 1/2 sum_{xy allowed} |H_xy| phi_x phi_y (e_x - e_y)^2 / N (identity I2 of `amp_design/DESIGN_MEMO.md`).
- Frozen-FN gain of one exact iteration G0 = E_f[a] - E_FN = 3.01e-5 (exact); Q(delta) = G0 to second order.
- Captured fraction of a write-back b = a exp(f): **frac = (E_f[a] - E_f[b]) / G0** (exact frozen-FN Rayleigh quotient,
  the primary metric of `experiments/writeback/README.md`). Quadratic version: 1 - Q(delta - f)/Q(delta).
- Per-configuration node weight of the residual (always >= 0): c_e(x) = 1/2 sum_{y allowed} |H_xy| phi_x phi_y (e_x - e_y)^2.
- Bulk / tail: per-configuration phi^2 >= 1e-10 / < 1e-10 (the 1e-10 cut separates 92.3% of the mass and 19% of the
  gain from 7.6% of the mass and 81% of the gain).

## Arms (Step 1, oracle phi_FN; all Adam, warm-up + cosine, lr 3e-3, B = 1024 configurations x K = 8 neighbour
evaluations per step, 20k steps: equal steps and equal network evaluations per arm)
Distributions:
- **(i) old**: x ~ phi^(2 beta), beta = 0.5; K bonds uniform among valid bonds (exactly arm E-M of `writeback`).
- **(ii) tail**: x ~ q = 1/2 phi^(2 beta) + 1/2 c_delta / Q(delta) (explicit tail component = node weight of the
  quadratic form at f = 0); K bonds drawn proportional to the FN edge weight |H_xy| phi_y/phi_x (allowed bonds only);
  self-normalised importance weights p/q.
- **(ii-a) adaptive tail**: as (ii), but the tail component is refreshed every 2000 steps to c_e of the current
  residual e = delta - f (exact table of the network); this samples the energy metric of what is still missing.
Losses: (i) the exact frozen-FN edge identity of `writeback` (nonlinear in g = b/phi). (ii) quadratic edge-difference
loss 1/2 sum w_xy ((f_x - f_y) - (delta_x - delta_y))^2 (the FN Laplacian least-squares fit of log-ratio differences).
A cross arm (ii proposal + (i) loss) separates the effect of the distribution from that of the loss.
Bases / architectures:
- **R28k**: residual CNN 32 ch x 4 layers (28k parameters, the `writeback` net), exact D4 x flip symmetrisation.
- **R111k**: 64 ch x 4 layers (wider).
- **R28k+a**: the same CNN plus the base log-amplitude log a(x) as an input (Fourier embedding of the standardised
  log a, concatenated with the pooled CNN features, two-layer MLP head, zero-initialised output). Justification: the
  gain sits at the low-amplitude configurations, and *where* the tail is (depth log a) is a non-local property of the
  ViT that a 3x3 CNN must otherwise rediscover; log a(x) is free in a real loop (b = a e^f evaluates a anyway).
- **Linear structured bases (exact, no training)**: piecewise-constant functions of log a (64 quantile bins), of
  (log a, log of the guide's FN violating weight V(x) = sum_{y violating} |H_xy| a_y/a_x) (32 x 32 bins), and of
  (log a, V, kept-edge weight W(x) = sum_{y allowed} |H_xy| a_y/a_x) (16^3 bins). These quantities are local to the
  guide (zero or one hop of the base network). Their maximal quadratic capture is exact (coarse-grained Laplacian).
- **Bulk restriction** (exact, no training): the best *arbitrary* f supported on the bulk only (f = 0 on the tail;
  CG solve of L_SS f = (L delta)_S); also tail-only, and bulk cuts 1e-9 / 1e-11 / 1e-12. Plus a trained arm whose
  samples are restricted to the bulk (the network extrapolates to the tail).

## Pre-registered rule (written 2026-10-08 10:05, before any run of this test)
- **Step 1 PASS (reopen the write-back route)** if the best trained arm with the (ii)/(ii-a) distribution captures
  **frac >= 0.50** of one FN iteration's frozen-FN gain on oracle data (exact evaluation, best-validation parameters).
  Then Step 2.
- **Step 2** (only after a Step 1 PASS): same arm with realistic data: MCMC/exact-sampled configurations from the tail
  proposal built from guide quantities only, local edge terms evaluated with the network and the guide only (no exact
  phi_FN at neighbours beyond what the FN step provides), and a same-capacity fixed-sign VMC control at equal samples.
  PASS if the FN-tail arm captures >= 0.5 of the gain AND >= 1.5x the gain of the VMC control.
- **If Step 1 < 0.50: stop.** Report which part of the tail the network cannot represent (decade decomposition of the
  remaining residual Q(delta - f), bulk/tail restricted bounds, linear-basis bounds), no Step 2.
- Secondary reading (not part of the rule): (ii) vs (i) at equal architecture says whether the distribution was the
  bottleneck (> 1.5x would support the hypothesis even if the 50% bar is missed); the bulk-restricted bound says
  whether capturing the gain requires the tail at all.
- Budget: <= 10 GPU-h (RTX 2080 Ti or full A40; no V100 / A40 slices). Full fp32 networks, float64 exact algebra.

## Amendment 1 (2026-10-08 10:35, after the smoke run and the exact projections, before any main training result)
- **Correction of a parenthetical.** The bulk cut stays per-configuration phi^2 >= 1e-10 as defined (98.2% of the phi^2
  mass, 41% of the quadratic gain). The numbers quoted next to it (92.3% mass / 19% gain) belong to the 1e-9 cut.
  Both cuts are reported.
- Estimator check (smoke, 64 x 1024 samples, f = 0 and a perturbed head): all proposal / bond combinations reproduce
  the exact frozen-FN and quadratic values within 1.2 SE. Their SE at f = 0 is similar (4.2-4.9e-7 per 1024 samples).
- Exact projections already in (`wt_exact.py`, no training; they do not change the rule): arbitrary f on the bulk
  only captures 21% (cut 1e-10) / 6% (cut 1e-9); tail-only f captures 72% / 91%. The old beta = 0.5 proposal already
  puts 39% of its samples below 1e-10 (phi^2 itself: 1.8%; the tail mixture: 49%).
- The quadratic edge loss and the exact frozen-FN identity differ at third order (Q(delta) = 3.08e-5 vs G0 = 3.01e-5),
  so a second cross arm is added: TAX-R28 = adaptive tail proposal + exact identity loss. Intermediate exact table
  evaluations are dropped (4.5 min each); training curves come from the common validation estimator (SE ~1e-7).
  The 111k arms are capped at 70 min each (the old E-L cap was 1 h); one tail arm (TA) for the wide net.
- Rule unchanged: Step 1 PASS iff the best (ii)/(ii-a) arm reaches exact frac >= 0.50.

## Amendment 2 (lead + PI requests during the run; recorded afterwards, 2026-10-08 15:00)
- ~10:50, before any optimizer result: optimizer arms added at the same data/loss budget, Adam kept as baseline:
  minSR on the residual net (old and adaptive-tail distribution), energy-metric Gauss-Newton / Levenberg-Marquardt
  (matrix-free CG on J^T L J; trust ratio judged on an independent batch after the same-batch version overfitted in the
  smoke run), and minSR fine-tuning of the symmetrised ViT itself (old distribution only, budget).
- ~11:20: three-iteration "jump" arm: fit b = |psi_P| exp(f) to phi_3 of the ideal exact loop (frozen H_FN[phi_2, s_2]),
  then a fixed-sign VMC-minSR polish at the Krylov sign of phi_3; report exact <H> and E_FN after fit and polish.
- Budget raised to 15 GPU-h. The 50% reopen rule is unchanged and applies to every arm.
- Hyper-parameters chosen on short smoke scans (validation only): minSR eta 0.05, shift 1e-4 tr/B, step cap 0.01 rms;
  ViT minSR eta 0.002, cap 0.001; GN batch 8192 x 8 edges, 20 CG steps, 25 min wall-clock per arm.

## Amendment 3: Step 2 pre-registration (2026-10-08 ~22:30, before any Step-2 run)
Trigger: Step 1 passed with guide-neighbourhood inputs (FEAT-TA-R28: 88.7% exact, held-out tail 89.7%; job wtKA2).
Lead GO; merged with the itfit one-hop-feature route (its semi-implicit FN step is used as the realistic FN target).

**Realistic data, no phi_FN anywhere in training, proposal, loss, validation or model selection.**
- Net: FEAT (28k residual CNN + inputs log a, V, W of the frozen guide; 37k parameters), b = |psi_P| exp(f).
- Guide-only quantities (exact tables here; in a real loop these are guide evaluations on the one-hop neighbourhood):
  V, W, H_xx, the FN diagonal D = H_xx + V, the guide's frozen-FN energy E_a (a scalar, exact here, a VMC estimate in a
  loop), the semi-implicit FN step T(x) = log(1 + W) - log(1 + (D - E_a)) (itfit, tau = 1; 87% of one FN iteration as
  an exact unprojected step).
- Proposal (static, guide-only): q = 1/2 a^(2 beta) + 1/2 node weight of the guide-metric quadratic form of T,
  c_a(x) = 1/2 sum_{y kept} |H_xy| a_x a_y (T_x - T_y)^2, beta = 0.5; exact sampling from this table (stands in for
  MCMC); self-normalised weights b^2/q (VMC arms) or a^2/q (regression arms).
- Samples: B = 256 configurations per step, 20k steps, for every arm; K = 8 sampled valid bonds per configuration
  (weight nvalid/K) for every local term, so all arms use the same samples and the same network evaluations per step.

**Arms** (Adam: lr 3e-3 warm-up + cosine; minSR: sample-space natural gradient, shift 1e-3 tr/n, step cap):
- FN target, regression: **F-SI-Adam**, **F-SI-SR**. Edge-difference least squares of f + log a against T in the
  guide metric (weights |H_xy| a_y/a_x on kept bonds). minSR form for least squares (sample space over edges).
- FN target, energy: **F-VMC-Adam**, **F-VMC-SR**. Frozen-FN energy E_f[b] by VMC: local energy
  E_L(x) = D(x) - sum_{kept} |H_xy| b_y/b_x from the sampled bonds.
- Control: **C-VMC-Adam**, **C-VMC-SR**. Fixed-sign <H>(b, s_P) by VMC with the same inputs, samples and bonds.

**Metrics (exact, final parameters; no selection):** frac = exact frozen-FN gain captured (as in Step 1);
<H>(b, s_P). Convergence evidence: each arm's own realistic objective on fixed validation batches every 1000
steps, plus exact evaluations at 10k and 20k steps. A phi-based validation estimate is logged as a diagnostic only.

**Rule (lead, pre-registered):**
- PASS if the best FN arm reaches frac >= 0.5 **and** >= 1.5x the frac of the best control arm (same metric).
- If PASS: three-iteration loop with the FEAT net (guide_k -> features, T, proposal -> fit -> exact table -> Krylov
  sign), exact <H> and E_FN per iteration vs RBM+PP (4.5e-5) and the ideal loop (4.30e-5 after 3 iterations).
- Otherwise: report which part fails (target, estimator noise, optimizer) and stop.

**Cost (counted, `wt_hopcount.py`, `results/writeback_tail/hopcount.json`):**

| lattice | distinct one-hop configurations | distinct two-hop configurations |
|---|---|---|
| 6x6 | 77 | ~2.8e3 |
| 8x8 | 135 | ~8.9e3 |
| 10x10 | 210 | ~2.2e4 |

- One hop is what V, W and T at a sampled x need.
- With K sampled bonds, the edge terms need the inputs at K neighbours, i.e. K x one-hop guide evaluations: 616 at
  6x6, 1080 at 8x8 for K = 8.
- A full local energy needs all two-hop configurations.

## Amendment 4 (2026-10-08 ~22:45, after the Step-2 smoke, before any Step-2 main result)
- Smoke (40 steps per arm, validation only) showed:
  - Adam at lr 3e-3 makes both VMC arms diverge within 20 steps; the frozen-F diagnostic rose 25x.
  - minSR at eta 0.3 diverges on the SI regression; minSR at eta 0.05 drifts upward on both VMC arms.
  - The realistic VMC energy estimate has SE ~6e-3 per site per 256-sample batch, 200x the 3e-5 gain.
- Step sizes for the main runs, set symmetrically for FN and control arms:
  - VMC arms: Adam lr 3e-4; minSR eta 0.01, shift 1e-3 (the itfit VMC-control setting);
  - SI regression: Adam lr 3e-3 unchanged; minSR eta 0.05.
- The step 2 controls are given the most favourable setting; the rule is unchanged.
- The lead's priority: the same-model fixed-sign VMC control for the itfit realistic-feature loop. That loop is
  `results/itfit_6x6/runs/itF.json`: 5.1k-parameter MLP on 13 frozen-base one-hop features, LM-GN on semi-implicit
  targets, captures 0.95 / 0.80 / 0.69 per iteration. Its control is run exactly as the itfit worker pre-registered it
  (`experiments/itfit_6x6/specs/main_F.json`, arms F-VMC-iii-e01 / -e03):
  - same model and features, tempered samples;
  - minSR on <H>(b, s_k), eta 0.01 / 0.03, shift 1e-3;
  - 6.6e8 network evaluations or 20 min per loop iteration, at least the FN arm's 0.8-2.2e8, so the control gets more;
  - 3 loop iterations with Krylov signs.
  - Pass test: the FN loop captures >= 1.5x the control in each iteration, with >= 0.5.
- Code unchanged (`experiments/itfit_6x6/itfit_run.py` at e8e72c2); specs `specs/itF_F-VMC-iii-e0{1,3}.json`.
