# Write-back of one FN iteration with energy-metric (tail) sampling, exact 6x6 oracle harness

Date 2026-10-08. Pre-registration, definitions and amendments: `experiments/writeback_tail/README.md` (committed
before any training result). Code: `experiments/writeback_tail/` (`wt_exact.py`, `wt_run.py`, `wt_figure.py`, `specs/`).
Run JSONs: `runs/`; exact projections: `wt_exact.json`. Figure: `writeback_tail_6x6.png` / `.pdf`.
All numbers per site, J2/J1 = 0.5, exact symmetric sector, guide (a, s) = symmetrised ViT (|psi_P|, s_P).
"Captured" = exact share of the frozen-FN gain of one FN iteration, frac = (E_f[a] - E_f[b]) / (E_f[a] - E_FN),
G0 = 3.01e-5; b = |psi_P| exp(f) (residual nets) or b = |psi_P,theta| (ViT itself). Best-validation parameters.

## Verdict
**INCONCLUSIVE (repo rule 1): no arm reaches the pre-registered 50% bar, so no Step 2; but the shortfall is not pinned
to a conceptual limit and one lever is still moving (section 6).** The best arm (adaptive tail proposal, 111k residual CNN,
Adam) captures **30.8%** of one exact FN iteration with oracle data; the same net on the old distribution 15.2%.
- **The hypothesis is half right.** The distribution matters and interacts with capacity: tail sampling raises the 28k
  net from 17.9% to 20-22% and the 111k net from 15.2% to 30.8% (2.0x); with the old samples extra width *hurt*.
  minSR needs the tail even more (5.4% -> 18.1%). But no arm, optimizer or basis comes near 50%.
- **Capturing the gain requires the tail.** The best arbitrary correction restricted to the 98% of the phi^2 mass above
  1e-10 reaches 20% (exact); 50% needs it to be right down to 1e-11 (> 1.2M orbits); the tail alone carries 72%.
- **What the net cannot represent** is the configuration-rough part of delta at every depth: the best net keeps ~40% of
  each bulk decade and 20-27% below 1e-12, where a third of the gain sits (section 4).
- **Optimizers (lead/PI request):** Adam > minSR (needs tail; 18%) >> energy-metric Gauss-Newton (0.3%, overfits each
  batch). minSR on the symmetrised ViT itself makes the frozen-FN energy *worse* (-22%; <H> -7e-6).
- **Three-iteration "jump" (PI):** the fit keeps 18.5% of the 3-iteration frozen-FN gain (2.4e-5/site); <H> with the target
  sign 1.161e-4 vs psi_P 1.29e-4, target 4.3e-5, RBM+PP 4.5e-5: a partial jump, far from RBM+PP. Every VMC-SR
  polish made it worse (best 1.187e-4).

## 1. Exact projections (no training): does the gain need the tail?
| projection (exact, quadratic optimum in the FN metric Q) | phi^2 mass | orbits | captured (exact frac) |
|---|---|---|---|
| any f on the bulk, per-config phi^2 >= 1e-9 | 92.3% | 115k | **5.6%** |
| any f on the bulk, >= 1e-10 (pre-registered cut) | 98.2% | 428k | **19.6%** |
| any f on the bulk, >= 1e-11 | 99.6% | 1.21M | 44.9% |
| any f on the bulk, >= 1e-12 | 99.95% | 2.70M | 72.3% |
| any f on the tail only, < 1e-10 | 1.8% | 15.4M | 72.3% |
| any f on the tail only, < 1e-9 | 7.7% | 15.7M | 90.9% |
| piecewise constant in log a (256 bins) | | | -0.9% (quadratic 3.2%) |
| piecewise constant in (log a, log V), 32x32 | | | 0.1% (quadratic 4.9%) |
| piecewise constant in (log a, log W), 32x32 | | | 3.9% (quadratic 6.3%) |
| piecewise constant in (log a, log V, log W), 16^3 (1650 groups) | | | **24.6%** |
| piecewise constant in delta itself, 64 bins (oracle reference) | | | 98.0% |

V, W = sign-violating / kept-edge weight of the guide (one hop of the base network). Reading:
- **Yes, the gain needs the tail.** An arbitrary correction on the 98% of the phi^2 mass above 1e-10 reaches 20%; reaching
  50% needs the correction right down to per-configuration phi^2 ~ 1e-11, i.e. on > 1.2M orbits.
- The gain is not a function of depth (log a: 3% quadratic) nor of depth plus the local FN weights (25%).
- The old beta = 0.5 proposal already put 39% of its samples below 1e-10 (phi^2: 1.8%, tail mixture: 49%).
  ESS of the importance weights: 9.7% (old) vs 5.9% (mixture).

## 2. Trained arms (Step 1, oracle phi_FN)
All arms: 20k steps x 1024 configurations x 8 neighbour evaluations (Adam, minSR), or the same wall-clock (25 min,
Gauss-Newton; 2000 steps x 128 for the ViT). "old" = x ~ phi^(2 beta), beta = 0.5, uniform bonds (the 2026-10-07 E-M
distribution). "tail" = 1/2 phi^(2 beta) + 1/2 node weight of Q(delta), bonds drawn by FN edge weight;
"adaptive tail" = tail component refreshed every 1000 steps to the node weight of the current residual Q(delta - f).
Loss "edge" = FN-Laplacian least squares of edge differences; "identity" = exact frozen-FN edge identity.

| arm | optimizer | net | distribution | loss | **captured (exact)** | quadratic | <H>(b, s_P) (start 1.319e-4) |
|---|---|---|---|---|---|---|---|
| O-R28 | Adam | CNN 28k | old | identity | 17.9% | 19.4% | 1.253e-4 |
| T-R28 | Adam | CNN 28k | tail | edge | 20.0% | 25.9% | 1.240e-4 |
| TA-R28 | Adam | CNN 28k | adaptive tail | edge | 20.4% | 25.9% | 1.240e-4 |
| TX-R28 | Adam | CNN 28k | tail | identity | 11.5% | 13.8% | 1.277e-4 |
| TAX-R28 | Adam | CNN 28k | adaptive tail | identity | 20.9% | 21.8% | 1.241e-4 |
| O-R111 | Adam | CNN 111k | old | identity | 15.2% | 17.4% | 1.264e-4 |
| TA-R63 (follow-up) | Adam | CNN 63k | adaptive tail | edge | 22.8% | 28.1% | 1.232e-4 |
| **TA-R111** | Adam | CNN 111k | adaptive tail | edge | **30.8%** | 35.0% | **1.199e-4** |
| O-R28a | Adam | CNN 28k + log a input | old | identity | 19.4% | 19.8% | 1.247e-4 |
| T-R28a | Adam | CNN 28k + log a input | tail | edge | 21.9% | 27.2% | 1.233e-4 |
| TA-R28a | Adam | CNN 28k + log a input | adaptive tail | edge | 20.6% | 26.2% | 1.239e-4 |
| TB-R28 | Adam | CNN 28k | tail, bulk configurations only | edge | 7.6% | 16.6% | 1.283e-4 |
| SR-O-R28 | minSR | CNN 28k | old | identity (local energies) | 5.4% | 8.7% | 1.300e-4 |
| SR-TA-R28 | minSR | CNN 28k | adaptive tail | identity (local energies) | 18.1% | 18.8% | 1.250e-4 |
| GN-O-R28 | Gauss-Newton/LM | CNN 28k | old | edge | 0.3% | 0.7% | 1.318e-4 |
| GN-TA-R28 | Gauss-Newton/LM | CNN 28k | adaptive tail | edge | 0.3% | 0.7% | 1.318e-4 |
| SR-O-ViT | minSR | the symmetrised ViT itself (155k) | old | identity (local energies) | **-21.8%** | 8.2% | 1.249e-4 |
| exact FN step | | | | | 100% | 100% | 0.841e-4 |

Optimizer notes (smoke scans in `runs/` logs; settings in `experiments/writeback_tail/specs/`):
- minSR (Chen & Heyl): sample-space solve, b^2/q-weighted QGT, frozen-FN local energies from the exact edge form;
  eta 0.05, diagonal shift 1e-4 x tr/B, step capped at 0.01 rms change of f. Smaller shifts or eta 0.2 diverged (NaN);
  shift 1e-3/1e-2 was slower.
- Gauss-Newton / Levenberg-Marquardt in the energy metric (matrix-free CG on J^T L J, 20 CG steps, batch 8192 x 8 edges
  per outer step). Accepting steps by the fit on the same batch drives the damping to 1e-8 and every step overfits the
  batch (no validation gain); judging the trust ratio on an independent batch drives the damping to its cap after
  7-9 accepted steps. A full-batch GN over the 15.8M orbits costs 4.5 min per function evaluation (16 images) and was
  not run.
- ViT minSR (eta 0.002, step cap 0.001 rms, B = 128 configurations x 8 neighbours x 16 images): validation improved for
  400 steps and then oscillated above the start. The best-validation parameters *raise* the exact frozen-FN energy by
  6.6e-6 (-22% of G0) while lowering <H>(b, s_P) by 7e-6: the ViT moves configurations the samples do not see.

## 3. Three-iteration target (PI: a jump, not a local step)
Target = phi_3 of the ideal exact loop from psi_P (FN -> Krylov sign -> FN -> Krylov -> FN); frozen Hamiltonian
H_FN[phi_2, s_2], whose Perron vector is phi_3; target sign s_T = Krylov sign of phi_3. Same residual factor
b = |psi_P| exp(f) (28k CNN + log a input), adaptive tail proposal, exact-identity loss, Adam 20k steps
(`runs/wtJ3_16903064.json`). The quadratic edge loss is unusable here: for this far target it lowered its own value
while the exact frozen-FN energy rose (smoke run, 1000 steps), so the exact identity was used.

| state | frozen-FN gain captured (G0 = 1.28e-4) | <H>(., s_P) | <H>(., s_T) | E_FN(., s_T) |
|---|---|---|---|---|
| psi_P (start) | 0 | 1.319e-4 | 1.290e-4 (*) | 9.30e-5 (*) |
| **fit to phi_3** | **18.5%** (2.4e-5/site) | 1.205e-4 | **1.161e-4** | **8.67e-5** |
| fit + VMC-minSR polish (1000 steps x 256 samples, eta 0.01) | -0.8% | 1.325e-4 | 1.296e-4 | 9.30e-5 |
| fit + VMC-minSR polish (300 x 1024, eta 2e-3) | 2.9% | 1.313e-4 | 1.277e-4 | 9.23e-5 |
| fit + VMC-minSR polish (300 x 1024, eta 5e-4) | 14.8% | 1.229e-4 | 1.187e-4 | 8.80e-5 |
| exact phi_3 (target) | 100% | | 4.30e-5 | 4.87e-5 (E_FN of its own guide) |
| RBM+PP (reference) | | | 4.5e-5 | |

(*) smoke run with f ~ 0 (rms 6e-4, 0.02% of the gain), i.e. psi_P to the digits shown. All energies per site above E0.

- The jump is partial: the fit keeps 18.5% of the three-iteration frozen-FN gain (about the same fraction as for one
  iteration, but 4x the absolute gain: 2.4e-5 vs 0.6e-5/site). <H> with the target sign falls 1.29e-4 -> 1.16e-4,
  far from the target (4.3e-5) and from RBM+PP (4.5e-5).
- Every fixed-sign VMC polish (realistic: samples from b^2, full local energies, no phi_FN) made the fit worse, in
  proportion to its step size (<H>(s_T) 1.161e-4 -> 1.296 / 1.277 / 1.187e-4): the per-step VMC noise (estimates scatter
  by ~1e-4/site at 256-1024 samples) is far above the 1e-6-1e-5 effects. Polishing at this level would need ~1e6+
  samples per step. Polish-only runs from the saved fit: `runs/wtJ3pol_16903904.json` (they reproduce the fit to 1e-15).
- The majorisation argument does not forbid a jump, but the representation limit of section 4 applies to the far
  target as well: the network keeps the same ~20% share.

## 4. What the network cannot represent
Per-decade capture (quadratic form, node split of Q), share of each decade's gain that is captured:

| per-config phi^2 decade | 1e-8 | 1e-9 | 1e-10 | 1e-11 | 1e-12 | 1e-13 | 1e-14 |
|---|---|---|---|---|---|---|---|
| share of the gain Q(delta) | 5.4% | 12.6% | 21.6% | 25.5% | 19.6% | 9.8% | 3.2% |
| captured fraction of that decade (TA-R111) | 40% | 37% | 38% | 37% | 32% | 27% | 20% |
| captured fraction (TA-R28, adaptive tail, 28k) | 31% | 29% | 29% | 27% | 23% | 18% | 13% |
| captured fraction (O-R28, old distribution, 28k) | 24% | 22% | 21% | 20% | 18% | 15% | 11% |
| captured fraction (O-R111, old distribution, 111k) | 24% | 21% | 20% | 17% | 15% | 12% | 9% |
| captured fraction (TA-R63, adaptive tail, 63k) | 34% | 31% | 32% | 30% | 25% | 20% | 14% |

- No decade is represented: even in the bulk decades the best net keeps ~40%, and the capture falls with depth to 20-27%
  below 1e-12, where a third of the gain sits. The residual stays configuration-rough (rms error 0.0063 vs rms
  delta 0.0086).
- Tail sampling does what it should: at 28k it raises the capture in every decade (1.2-1.4x), and at 111k it is what
  makes the extra capacity useful (old 15% -> tail 31%; with the old distribution the wider net is *worse* than the
  narrow one, as on 2026-10-07). This supports "distribution before capacity", but the combination stops at 31%.
- Restricting the samples to the bulk (TB-R28) loses most of the gain (7.6%): the net does not extrapolate into the
  tail. The exact restricted optimum agrees: any f on 98% of the mass reaches 20%.
- Optimizers: Adam is best. minSR needs the tail distribution (5% -> 18%, 3.4x) and still trails Adam; energy-metric
  Gauss-Newton overfits each batch. minSR on the converged ViT itself makes the frozen-FN energy worse.

## 5. Conceptual limit vs method
**Conceptual (exact, independent of network, optimizer and sampler; 6x6):**
- The gain of one FN iteration is configuration-rough in the FN metric (0.97 of white noise) and sits in the tail:
  59% on configurations with per-configuration phi^2 < 1e-10, which are 97% of the 15.8M orbits.
- **Any correction that captures >= 50% must be accurate on configurations of weight < 1e-11** (the best arbitrary
  correction restricted to phi^2 >= 1e-11, i.e. 99.6% of the mass and 1.2M orbits, reaches 45%; >= 1e-12: 72%).
- In a realistic (sampled, no phi_FN) setting the energy-estimator noise at 1e3-1e4 samples per step is ~1e-4/site, far
  above the 1e-6-1e-5 effects (2026-10-07 VMC arms; every VMC polish of section 3 raised <H>). This is a cost statement
  that scales with the sample count, not a no-go.

**Method (open, not a limit):**
- Representation/learnability of a rough tail function by a callable net: capture rises with capacity *only* with tail
  sampling (28k 20% -> 111k 31%); whether it saturates below 50% is untested (larger nets, section 6).
- Whether the tail correction generalises to unseen orbits at all (learnability control, section 6).
- Inputs: the nets see only spins (+ log a); the correction is generated by the guide's one-hop neighbourhood
  (V, W bins alone give 25%). Mechanism-aware inputs: section 6.
- Optimizers: second-order methods failed for identified technical reasons (per-batch GN interpolates each batch; ViT
  minSR selected on a sampled frozen-F estimate that does not see the configurations it changed), not tested in their
  best form (global GN with validated damping: section 6).

## 6. Follow-up (lead/PI/consultation, 2026-10-08 afternoon)
Diagnosis of the ViT minSR arm: the step was selected on the *frozen-F* validation estimate (tail-mixture samples,
exact identity), not on <H>. That estimate said +6% at step 400 while the exact frozen-F said -22%; the quadratic
capture was +8% (its per-decade split loses below 1e-14 and above 1e-7, both rarely sampled), so the 30-point gap
to the exact value is beyond second order: large changes of f on few configurations. The ViT update damaged
configurations the validation never saw (an estimator blind spot, not a
shift/step-size divergence: the step was capped at 0.001 rms of f and the shift was 1e-4 tr/B).

Global Gauss-Newton (consultation item 4; job wtKGN 16976430): one fixed pool of 16 x 4096 configurations x 8 edges
(524k residuals, 18x the 28k parameters), 30 CG steps per outer iteration, damping accepted only if the exact-identity
frozen-F estimate falls on 16 independent batches. 8 steps accepted in 20 min, then every step rejected while the
damping rose 1000x; validation estimate 0.33% of the gain (no exact evaluation: the job stopped on an exact CG
convergence, rs = 0, in the 16th outer step). With curvature accumulated globally and damping validated on the true
objective, the energy-metric GN direction of this net still does not generalise beyond the pool.

Capacity with the adaptive tail proposal (Adam, 20k steps, exact frac): 28k 20.4%, **63k 22.8%**, 111k 30.8%
(old proposal: 28k 17.9%, 111k 15.2%). The 63k net puts |f| > 0.1 on 32% of the orbits and up to |f| = 2 in the deep
tail (max |f| 0.07 above phi^2 = 1e-8, 0.32 above 1e-10, 0.55 above 1e-12).

Capacity: a 445k residual CNN (C128) at the 28k/111k learning rate (3e-3) had not started learning after 7000 steps
(validation frozen-F above the start; the 111k net was at 12% by then). Stopped after 1.4 A40-h; replaced by a 250k net
(C96) at lr 1e-3.

Queued on ws1 at the time of writing (cluster fair-share: the user's other sessions hold ~730 running jobs, so these
start by backfill; expected starts 20:45-05:40):
| job | content |
|---|---|
| wtC48 16948146 | 63k residual CNN, adaptive tail (capacity point; equal-parameter control for the tail expert) |
| wtC96 16998622 | 250k residual CNN, adaptive tail, lr 1e-3 (capacity point) |
| wtEXP 16961949 | tail expert: frozen TA-R28 bulk + sigmoid gate below phi^2 = 1e-8 x fresh 28k tail net trained only on tail samples (80% of tail orbits); captured % overall, per decade, and on held-out tail orbits |
| wtKA 16976428 | learnability control (orbit split, all edges incident to test orbits withheld, real target); mechanism-aware inputs (log a, guide violating weight V = FN wall term, kept weight W) + adaptive tail; edge-sampling proposal q_xy ~ \|H_xy\| phi_x phi_y |
| wtKB 17009145 | the same learnability arm on a target shuffled within (log a, FN weighted degree) bins, rescaled to equal Q |
| wtKGN 16976430 | global Gauss-Newton: one fixed pool of 16 x 4096 x 8 edges, 30 CG steps, damping accepted/rejected on the exact frozen-F identity on independent batches |
Specs: `experiments/writeback_tail/specs/{cap_C48,cap_C96,expert,consult_A,consult_B,consult_GN}.json`.

## 7. Next step
1. Do not build the loop on write-back yet (no arm passes); the route is inconclusive, not closed. FN/Krylov stays a sign step and calibrated referee; amplitude by ordinary VMC.
2. Finish the queued follow-up (section 6). The one lever that still moves is **capacity x tail sampling** (28k -> 111k: 20% -> 31%, curve still rising at the
   end of the cosine schedule). If revisited: a 0.4-0.5M residual CNN with the adaptive-tail proposal on the same oracle
   harness (about 4 A40-h), before anything realistic. It would need a further 1.6x to pass.
3. Do not pursue energy-metric Gauss-Newton on minibatches or minSR fine-tuning of the converged ViT for this purpose.

## Compute
About 11.5 GPU-h in total (A40 and RTX 2080 Ti on ws1; within the 15 GPU-h ceiling set for the optimizer arms):
exact projections 0.1, smoke/debug 1.6 (mostly 2080 Ti memory failures), 28k Adam arms 3.3, 111k arms 2.6, minSR 1.0,
Gauss-Newton 1.0, ViT minSR 0.55, three-iteration fit and polish 1.3. Runs on ws1 `/project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/`.
