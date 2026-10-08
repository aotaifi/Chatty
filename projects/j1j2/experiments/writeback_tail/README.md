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
