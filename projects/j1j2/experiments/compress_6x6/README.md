# Compressing the FN/Krylov loop's guide stack into one callable network (exact 6x6, J2/J1 = 0.5)

Pre-registration, written 2026-10-09 before any decisive run. Results go to `results/compress_6x6/`.

## Why
The 6x6 loop with realistic write-back (`results/writeback_tail/` 6c) passes RBM+PP only as a TABLE: guide k is
la_k = la_{k-1} + f_k(x, log a_{k-1}, V_{k-1}, W_{k-1}), and V, W need the previous guide on the one-hop
neighbourhood, so a callable guide recurses (135^k base calls at 8x8). Spin-only re-basing of the accumulated
correction failed (writeback_tail 6d(a): 10.6% / -318% / 2.6% kept); base-feature re-basing keeps 68-84% but costs
two hops. Reviewer E3: no conceptual no-go (la_k is a function of x alone); the obstacle is log-ratio precision on
~135 bonds. Chatty (workspace seq105): distil the WHOLE a_k into one flat callable A_theta(x) that never calls a_{k-1}
or its neighbourhood; symmetry-respecting ViT student; explicit log-a target + neighbour log-ratio loss in the
frozen-FN Dirichlet metric; re-base every 2-3 accepted steps.

## Setup (all exact on the 15.8M-orbit symmetric sector; code in this directory)
- `cx_replay.py`: replays `wtLOOP8_17090119` from its stored FEAT nets (read-only), checks every iteration against
  the stored record, saves (la_k, s_k), k = 1..6, to `/project/theorie/a/A.Otaifi/chatty_compress6/tables/`, runs P1
  and computes the references at k = 3, 6 (same sign s_k throughout).
- `cx_distill.py`: P2 students, training, exact evaluation.
- Teacher at k: (la_k, s_k), <H> 6.29e-5 (k = 3) and 4.03e-5 (k = 6); start psi_P 1.319e-4.

## P1: sensitivity of one write-back step to errors in V and W (sets the precision target)
- Iterations 1-3 of wtLOOP8, guide k-1 exact, stored net k (no retraining), write-back applied to the exact guide.
- Noise models, eta in {0.01, 0.03, 0.1}, each entering the net input AND the semi-implicit target T:
  - 'VW': V -> V exp(eta xi_V), W -> W exp(eta xi_W), xi iid N(0, 1) per orbit (reviewer E5(i));
  - 'bond': V, W and the log-a input computed from la + (eta/sqrt 2) xi, i.e. a student with white pointwise error
    whose bond log-ratio error has rms eta.
- Measured: frac of the ideal FN iteration with the stored net, frac of the exact semi-implicit target, rms of the
  induced log V, log W errors.
- **Tolerance eta\*** (per model) = the largest tested eta at which the stored-net frac stays >= 0.9 x its
  unperturbed value in all three iterations; "< 0.01" if none (then 0.01 is used as an upper bound below).
  eta\*_bond is the per-bond log-ratio tolerance; eta\*_VW the tolerance on log V, log W.
- Replay check: unperturbed fracs must reproduce wtLOOP8 (0.7457 / 0.6589 / 0.7621) to 1e-4.

## P2: distil la_k (k = 3 and k = 6) into ONE callable student
Arms:
- **A' (vit_warm):** the 6x6 ViT (4 layers, d = 60, 155k parameters), symmetrised exactly as psi_P,
  log A = log |sum_{g in D4 x flip} psi_theta(g x)| (4 patch translations inside), warm-started from the checkpoint,
  so A = psi_P at step 0. Adam, peak lr 1e-4.
- **A (vit_scratch):** the same ViT from random init, amplitude-only symmetrisation log A = mean_g Re log psi_theta(g x)
  (a random complex ViT has no meaningful phase to sum coherently). Same cost as A'. Adam, peak lr 1e-3.
- **B (bond):** bond-output ratio net. Periodic residual CNN (C = 64, 6 layers) -> per-site embedding -> per-bond-type
  MLP head; outputs R_b(x) ~ la_k(x^b) - la_k(x) for all 144 bonds in one pass; V, W follow from R (neighbour signs
  from s_k). Random D4 x translation x flip image per sample. Adam, peak lr 1e-3, B = 256, all bonds, 20k steps.
  B is not an amplitude: it is the feature supply of the reviewer's scheme (a scalar companion such as A/A' carries
  log a). It is scored on fidelity and on the continuation step, not on energy shares.
- **C (optional):** direct spin encoder + one edge-aggregation layer at equal inference budget. Run only if
  >= 1.5 GPU-h remain after A', A, B; otherwise recorded as not run.

Loss (A, A'): L = L_ratio + 0.1 L_value in the metric of the target guide a_k,
- L_ratio = E_{x ~ a_k^2} sum_{valid bonds b} (J_b/2) a_k(x^b)/a_k(x) (e(x) - e(x^b))^2 / N, K = 8 bonds sampled
  uniformly per x (weight n_valid/K), e = log A - la_k: the |H|-Dirichlet form of the target guide (kept bonds = the
  frozen-FN kinetic metric, violating bonds = the wall term that sets V);
- L_value = Var_{a_k^2}(e). Never pointwise MSE alone.

Sampling: x ~ 1/2 n a_k + 1/2 Dirichlet node weight of D_k = la_k - log|psi_P| in the guide-k metric (writeback
pre-test (a)), self-normalised importance weights to a_k^2. Held-out: 20% of orbits (writeback_tail hash) are never
sampled and every bond into them has weight 0. Batch B = 128 configurations x 8 bonds for A/A'.
Optimizer: Adam with warmup (300 steps) + cosine to 2% of the peak; the FINAL parameters are evaluated (no selection).
Convergence evidence: own loss (training-orbit and held-out batches) every 500 steps.
minSR: not run in P2 (a Jacobian over 16 images x 4 translations x 1152 rows per step is ~10x an Adam step, and minSR
overfit the small batches of the write-back regression). Revisited only if the A' Adam loss is still falling at the
end and budget remains.

Seeds: A' and B: seeds 0 and 1 at both k. A: seed 0 at both k; seed 1 only where seed 0 reaches share_H >= 0.4
(a second seed cannot move a verdict that is far from the bar).

Scores (exact):
- share_H = [<H>(psi_P amp, s_k) - <H>(A, s_k)] / [<H>(psi_P amp, s_k) - <H>(la_k, s_k)];
- share_EFN = the same with the FN energy of the guide (A, s_k) (exact FN ground state);
- secondary: frozen-FN Rayleigh quotient under H_FN[a_k, s_k]; pre-test (a) 'kept' (relative to <H>(psi_P, s_P));
  <H> with the student's own Krylov sign;
- callable student + one exact Lanczos step (vs psi_P + Lanczos 2.55e-5; reviewer bar: >= 2x below);
- continuation: the stored FEAT net k+1 driven by the student's own log a, V, W (no retraining), frac of the ideal FN
  iteration from the student guide, relative to the table's (iteration 4: 0.722, iteration 7: 0.511);
- per decade of a_k^2 (1e-7 .. <= 1e-15), held-out vs training orbits, 2048 orbits each: |H| a_y/a_x-weighted rms
  log-ratio error over all valid bonds, unweighted rms, rms log V and log W errors, pointwise rms;
- cost: wall-clock of one student evaluation per configuration / one psi_P evaluation (same batch, fp32, A40).

**Pass (scalar arm A or A'):** at both k = 3 and k = 6:
- share_H >= 0.70 AND share_EFN >= 0.70 (mean over seeds; no seed below 0.60);
- no tail collapse: held-out weighted rms log-ratio error <= 3 eta\*_bond in every decade from 1e-9 to 1e-14;
- cost <= 3 base-equivalent passes per configuration, independent of k (true by construction for A/A'; measured).
Reported, not part of the pass (reviewer E5(ii)): held-out weighted rms log-ratio error <= eta\*_bond in every decade.
Generalisation flag: held-out / training rms > 1.5 in any decade.

**Pass (B):** continuation frac with B's V, W >= 0.70 x the table continuation frac at both k (mean over seeds), no
tail collapse (same definition), cost <= 3. A B pass means "the one-hop features can be supplied by one forward pass";
it does not by itself make a callable guide.

Reading: P2 passes if A or A' passes. A negative result is "inconclusive (cause: X)" unless the cause is a
conceptual limit (e.g. capacity: loss flat at a floor with held-out = training; optimizer: loss still falling).

## P3 (only if P2 passes)
- From the passing student at k = 6 (and k = 3 if budget remains): two more loop iterations, fresh FEAT net on the
  student guide's own log a, V, W (exactly wt_loop: 20k Adam steps x 256 x 8 bonds, Krylov sign), then exact referee.
- Compare with the table loop iterations 7, 8 (<H> 3.64e-5, 3.16e-5; E_FN 3.15e-5, 2.75e-5).
- Pass: each iteration keeps >= 0.5 of its ideal FN iteration and the energy keeps falling.
- Also: callable guide + one exact Lanczos step vs psi_P + one Lanczos step (2.55e-5).

## Prior expectations (written before the runs)
- A from scratch: low (< 30%): the full la_k must be learned to ~1e-3 in log-ratio, which VMC never achieved for this
  architecture; at this budget a from-scratch fit is likely optimizer-limited.
- A': the most likely to work, but the converged ViT sits in a stiff landscape (stall_6x6 test 1); 40-70%.
- B: ratios are local-ish and supervised for every bond, so V, W may be easier than log a itself; unknown.

## Budget
<= 8 A40-h on ws1 (full A40, cip,inter): replay + P1 ~0.8, smoke 0.3, six ViT runs ~4.5, four B runs ~1.2, P3 ~1.
Runs: `/project/theorie/a/A.Otaifi/chatty_compress6/runs/`. Code and specs in this directory.

## Amendment 1 (2026-10-09 11:00, after smoke tests and the first P1 points, before any P2 run)
- **P1 extension.** At iteration 1 every pre-registered eta already loses far more than 10% of the step (VW model:
  eta 0.01 -> frac 0.19 vs 0.75; bond model: eta 0.01 -> 0.54). So eta\* < 0.01 for both models. To locate it,
  eta = 0.001 and 0.003 are added (`cx_p1ext.py`, same models; guides from the replay tables). The eta\* rule is
  unchanged (largest tested eta with frac >= 0.9 x unperturbed in all three iterations).
- **A' learning rate.** Smoke (`cxSMsl_17092262`, target = psi_P itself, so the gradient is fp32 noise): Adam at
  peak lr 1e-4 moved the warm ViT from loss 4.5e-11 to 5e-2 in 20 steps (stiff landscape, stall_6x6 test 1). The
  pre-registered 1e-4 is therefore replaced by a short scan on k = 3, seed 0: 600 steps at lr in {1e-6, 3e-6, 1e-5,
  3e-5}, choose the largest lr whose training-orbit own loss at step 600 is lowest (own loss only; no energy, no
  held-out data). A from scratch: scan {3e-4, 1e-3} the same way. The chosen lr is used for both k and both seeds.
- **Steps (from measured throughput).** ViT students: 0.33 s/step (B = 128 x 8 bonds, 16 images x 4 translations,
  fp32) on a 24 GB A40 slice with the jax 0.8.2 venv; the full-sector ViT table takes ~14 min. A and A' get 4000 Adam
  steps (~22 min) each. B: 20k steps (12 ms/step).
- **Hardware.** Runs may use the A40 vGPU slices (`cx_slice.sbatch`, cluster/A40_SLICES_HOWTO.md, jax 0.8.2 venv;
  fp32 ViT outputs differ from jax 0.10.2 at rms 1e-4, below the 3e-3 target). On slices the ViT training forward
  is evaluated in 4 sequential image groups with rematerialisation (`remat: 4`; same function, less memory).
- Warm-start check (smoke): log A' - log|psi_P| on 4096 orbits: rms 1.0e-4, max 3.4e-3 (fp32 floor of the coherent
  16-image sum).

## Amendment 2 (2026-10-09 12:50, after the lr scans, before any P2 evaluation)
**Replay** (`cxREP_17091969`, 2.0 A40-h): every iteration reproduces wtLOOP8 (frac and <H> to all printed digits).
References (same sign s_k): k = 3: <H> stack 6.289e-5, psi_P amplitude 1.273e-4; E_FN stack 5.252e-5, psi_P
9.294e-5; table continuation (iteration 4) frac 0.722. k = 6: <H> 4.030e-5 vs 1.297e-4; E_FN 3.468e-5 vs 9.309e-5;
continuation (iteration 7) 0.511. Stack + one exact Lanczos step: 2.17e-5 (k = 3), 1.55e-5 (k = 6).

**lr scans (k = 3, seed 0, 600 steps, own training-orbit loss; start 2.24e-4 for the warm ViT):**

| arm | lr | loss at 200 / 400 / 600 |
|---|---|---|
| A' | 1e-6 | 2.24e-4 / 2.51e-4 / **2.20e-4** |
| A' | 3e-6 | 5.34e-4 / 3.85e-4 / 2.73e-4 |
| A' | 1e-5 | 4.57e-4 / 4.21e-4 / 3.26e-4 |
| A' | 3e-5 | 8.79e-4 / 4.61e-4 / (n/a) |
| A | 3e-4 | 0.271 / - / 0.156 |
| A | 1e-3 | - / - / **0.113** |

By the amendment-1 rule A' uses lr 1e-6 and A lr 1e-3. Every A' lr first RAISES the loss (Adam's per-parameter steps
hit the stiff directions of the converged ViT); at 1e-6 the loss is flat (-2% in 600 steps). A from scratch is three
orders of magnitude above the warm start's loss after 600 steps.

Consequences (decided now, before any evaluation):
- **New arm A'-GN (the state-of-the-art optimizer for this least-squares problem):** same student, same loss, written
  as residuals (value sqrt(lam iw/B)(e_x - m), edge sqrt(iw w/(B N))(e_x - e_y)); damped Gauss-Newton in sample space
  (minSR form: (J J^T + lam I) a = r, step -J^T a), B = 128 x 8 bonds (1152 residuals), lam relative to tr(JJ^T)/M,
  starting at 1e-2, halved on acceptance, x4 on rejection; a step is accepted only if the loss falls on 3 FRESH
  training-orbit batches (paired old/new). It is evaluated exactly like A'. k = 3 and k = 6, seed 0; seed 1 if
  share_H >= 0.4. Step count set from its smoke throughput (amendment 3 if changed).
- A' with Adam: seed 0 only at both k (the scan shows a flat loss; a second seed cannot move the verdict).
- A from scratch: k = 6, seed 0 only (k = 3 adds nothing at a loss 500x the warm start's).
- Fidelity decades now include 1e-6 and 1e-5 (the replay's psi_P fidelity reference covers 1e-7 .. <= 1e-15 only).
- New jobs run with the jax 0.8.2 venv on full A40s too (measured faster: 0.33 vs 0.5 s/step; fp32 differences at
  rms 1e-4).

# Round 2 (lead/PI decision 2026-10-09: attack the recursion, do not cap the loop). Pre-registered 15:10, before any run.
Budget <= 8 GPU-h (A40 slices preferred). Exact 6x6, same lab, same references. Negative = "inconclusive (cause)".

## Test 1: constant-hop loop (`cx_chop.py`) - main
Each iteration k, starting from G_0 = log|psi_P|, s_0 = s_P:
- (w) write-back from the CALLABLE guide G_k exactly as wt_loop / F-SI-Adam: FEAT net on G_k's own (log a, V, W),
  semi-implicit target, guide-metric tail proposal, edge loss, Adam 20k x 256 x 8 -> la_w = G_k + f_k; Krylov sign
  s_{k+1} of la_w (exact, as in the table loop).
- (r) re-base: distil la_w into a FRESH base-feature net G_{k+1} = log|psi_P| + g(x; spins, log a_P, V_P, W_P of the
  frozen psi_P): the pre-test (a) 'featbase' arm (68-84% of the accumulated gain), edge (Dirichlet) loss in the metric
  of (la_w, s_{k+1}), proposal 1/2 n a_w + 1/2 Dirichlet node weight of la_w - log|psi_P|, Adam 20k x 256 x 8.
- Cost, independent of k: G_k(x) = one hop of psi_P (78 base evaluations at 6x6, ~135 at 8x8); V_k, W_k(x) = two hops
  (2.8e3 / 8.9e3); an edge-loss training sample needs features at the 8 sampled neighbours too, i.e. three hops of the
  base (~1e5 / ~4e5; measured hop counts `writeback_tail/hopcount.json`, measured psi_P throughput 1.9e4 / s per A40).
- Runs: seed 0 for 8 iterations, seed 1 for 6 iterations (budget).
- Reported per iteration: capture frac of the write-back, <H> of la_w (before re-base) and of G_{k+1}, re-base
  retention (step: share of this iteration's <H> gain kept; accumulated: share of the gain since psi_P kept), E_FN of
  G_{k+1}, G_{k+1} + 1 and + 2 exact Lanczos steps; against the table loop wtLOOP8 and psi_P + Lanczos p1 / p2
  (2.55e-5 / 8.6e-6).
- **Pass:** after 6 iterations G_6 keeps >= 70% of the table loop's accumulated <H> gain (<H>(G_6) <= 6.78e-5; table
  4.03e-5 from 1.319e-4), mean over the two seeds, neither seed below 60%, and <H> falls in at least 5 of the 6
  iterations. Reported in addition: whether G_k + 1 Lanczos step goes below 1.27e-5 (2x below psi_P + Lanczos).

## Test 2: structured tolerance (`cx_struct.py`)
- Replay of one write-back step from the exact table guide la_k with the stored wtLOOP8 net k+1 (k = 3 -> iteration 4,
  frac 0.722; k = 6 -> iteration 7, 0.511), with the guide replaced by la_k + c e for the ACTUAL error fields e of
  candidate predictors:
  - the P2 students: A' Adam (k = 3, 6), A from scratch (k = 6);
  - the base-feature re-base of la_k (the test-1 distillation, fit here);
  - B's actual log V, log W errors (k = 3, interpolated in log space);
  - white noise as the reference.
  Scales c chosen to span kept 0.1-1.
- Size measures (exact): |H_xy| a_x a_y-weighted rms bond difference of c e; a^2-weighted rms of log V, log W.
- Reported: kept share (frac / unperturbed frac) vs size, structured vs white.
- **Reading (pre-registered):** structured errors are "more tolerable" if, at equal weighted bond rms, their loss of
  capture is <= half the white-noise loss. eta\*_struct = the bond rms at which kept = 0.9, interpolated log-linearly
  between tested scales.

## Test 3: amortised V/W (`cx_vw.py`)
- For the table guides la_3, la_6: a 2-output FEAT-type CNN (spins + log a_P, V_P, W_P, i.e. one hop of the base)
  predicts u_V = log(V_k + 1e-6), u_W = log(W_k + 1e-6) (the FEAT input transform). Loss: standardised squared error,
  weighted by the FEAT measure, tail proposal of T_k, 20% held-out orbits, Adam 3e-3, 20k x 512.
- The predictions replace V_k, W_k in the next write-back step (stored net k+1, net input and target). Zero-training
  baseline: V_P, W_P themselves.
- **Pass:** capture with the amortised V, W >= 0.9 x the exact-V/W capture at both k. Also reported: rms log V,
  log W errors (held-out vs training), the baseline.
- If test 3 passes, the guide's V/W cost drops from two hops of the base to one, so the whole loop costs one hop of
  psi_P per evaluation (training samples two).

## Amendment 4 (2026-10-11, after tests 1-3 finished, before the diagnostic run)
Test 1 plateaus near <H> 7.0e-5. The write-back capture from the callable guide stays at 0.62-0.78. The re-base
step retention, however, falls 0.95 -> 0.86 -> 0.56 -> 0.21 -> 0.17 -> ~0.
The re-base own-loss curves are still falling at 20k steps and spike up to 10x (Adam at 3e-3), and their final loss
grows from 0.2e-6 (iteration 1) to 9e-6 (iteration 7). So the next question is whether the optimizer, the capacity or
the base features limit the re-base. Diagnostic (`cx_rebase_diag.py`), on the saved write-back table la_w of seed 0,
iteration 6 (chop re-base kept -6% of that iteration's gain):
- (a) warm start from the iteration-5 re-base, 20k steps at lr 1e-3;
- (b) fresh, 60k steps at lr 1e-3;
- (c) fresh, C = 64 (~111k parameters), 40k steps at lr 1e-3.
Reading:
- If (a) or (b) keeps >= 0.7 of the iteration's gain, the plateau is optimizer-limited, and the fix is warm-started,
  longer, lower-lr re-bases.
- If only (c) does, it is capacity.
- If none does, the frozen base's one-hop features are the limit, and the fix is richer base information (two-hop or
  input-gradient features).

# Round 3 (lead/PI 2026-10-11): per-bond base information. Pre-registered before any run.
Cause found in round 2: the frozen base's one-hop SUMS (log a_P, V_P, W_P) cannot carry the accumulated correction
beyond ~85% (re-base loss floor ~5e-6 at any size or step count). Fix tested here, at the same cost (one hop of psi_P):
**BondNet** (`cx_bond.py`).
- Inputs: the spins, psi_P's 144 per-bond log-ratios r_b(x) = log a_P(x^b) - log a_P(x), each bond's kept/violating
  flag under the current sign, the bond type, and the one-hop scalars (log a_P, V_P, W_P).
- Layers: site CNN (3 conv layers) -> bond messages from (h_i + h_j, h_i h_j, r_b, tanh r_b, kept_b, valid_b, type)
  -> summed onto the two sites -> site layer -> pooled readout.
- Translation-equivariant; exactly D4 x flip symmetric at evaluation (16 images, bond features permuted with the
  sites); one random image per training sample.

Pre-tests:
- **(1a) Re-base, iteration 6** (seed 0, the case where the scalar-feature re-base kept -0.06 of the iteration's gain
  and at most 0.22 with any optimizer or size).
  - Setup: BondNet, edge loss in the metric of (la_w, s_6), the same proposal, Adam lr 1e-3, 30k steps.
  - **Pass: retention of the iteration's gain >= 0.7.**
- **(1b) Amortised V/W at k = 6** (scalar-feature predictor: 0.48), and k = 3 (0.89).
  - Setup: BondNet with 2 outputs, residual on (log V_P, log W_P), the same loss, sampling and held-out split as
    test 3.
  - **Pass: capture >= 0.9 x exact at k = 6.**
- **(3) Rolling-base arm, iteration 6.**
  - Setup: re-base features = psi_P's one-hop scalars + the one-hop scalars of the iteration-3 re-based guide G_3. G_3
    is itself a psi_P-one-hop net, so its features cost two hops of psi_P at every k. FeatNet with 6 features, lr 1e-3,
    20k steps.
  - **Pass: retention >= 0.7.**

Follow-ups, conditional:
- **(2)** If (1a) passes: the constant-hop loop with the BondNet re-base, 8 iterations, seeds 0 and 1, Adam lr 1e-3,
  30k steps per re-base; everything else as round 2. Compared with the table loop, and at equal hops with psi_P + 1/2
  Lanczos steps.
  - Pass as round 2: >= 70% of the table loop's accumulated gain at iteration 6, mean of the seeds, neither seed below
    60%.
- If (3) passes and budget remains: the same loop with the rolling-base re-base, one seed.
- Bug fix: B's V/W replay (round 2, test 2) was NaN because V can be -1e-17 by round-off and its log was taken. Fixed
  with max(V, 0); the B points are re-run.

Budget <= 8 GPU-h, slices preferred.

## Amendment 5 (2026-10-11 22:20, after the three round-3 pre-tests, before this run)
All three round-3 pre-tests fail (results section below): BondNet re-base 0.10, BondNet V/W at k = 6 -0.06,
rolling base with G_3 0.08. One more decisive diagnostic, for the hypothesis that each new increment f_k needs the
IMMEDIATELY previous guide's one-hop information.
- Re-base iteration 6 with features = psi_P's one-hop scalars + the one-hop scalars of G_5. G_5 is rebuilt from the
  stored iteration-5 re-base; its V, W are taken under s_6, since s_5 was not saved.
- If it keeps >= 0.7, the hypothesis holds: a fixed-hop base cannot hold the loop's increments, and every iteration
  needs one more hop.

## Amendment 6 (2026-10-11 23:10, after the G_5 diagnostic, before this run)
- **The missing-hop hypothesis is refuted.** With psi_P's scalars + the one-hop scalars of the immediately previous
  guide G_5 (the very inputs from which the increment f_6 was made), the re-base keeps only 0.21.
- So the re-base is limited by the FIT itself. Across every input set, size and step count the loss floors at ~5e-6.
- Suspect: the distillation targets the whole accumulated correction la_w - log|psi_P|, and samples by ITS Dirichlet
  weight. The new increment is ~4% of that in the Dirichlet metric and is under-sampled and under-resolved.
- **Test: residual re-base.** G_6 = G_5 + h, with h fit to the increment la_w - G_5 alone and sampling targeted at
  the increment's own node weight. Two input sets for h:
  - (i) psi_P's one-hop scalars only. This gives a constant-hop guide: one hop of psi_P plus a stack of k small nets
    on the same features, cost linear in k but tiny;
  - (ii) + G_5's one-hop scalars, as an upper bound.
- FeatNet, edge loss, Adam lr 1e-3, 20k steps.
- **Pass (i): retention >= 0.7.** If it passes, the constant-hop loop is rerun with residual re-bases (seed 0,
  8 iterations) within the remaining budget.

## Amendment 7 (2026-10-11 23:25): residual re-base PASSES its pre-test; loop launched
- Residual re-base with psi_P's one-hop scalars only, sampling targeted at the increment: **retention 0.784** of
  iteration 6's gain (pass >= 0.7), 0.961 since psi_P. Own loss 6.4e-6 -> 1.3e-6.
- As pre-registered in amendment 6, the constant-hop loop is now run with residual re-bases (`rebase_net: inc`),
  8 iterations, seeds 0 and 1 (`specs/chop_inc_s*.json`). Pass rule as in round 2.
- Cost of the guide: one hop of psi_P plus k small nets on the same features (the base dominates).

# Round 4 (lead/PI 2026-10-12): robust residual re-base, 3 seeds. Pre-registered before any run.
Code `cx_chop2.py` (loop) and `robust_fit` in `cx_lib.py`; cost `cx_cost.py`. Budget <= 6 GPU-h on slices.
- **Dead-fit bug.** In round 2 and round 3 the seed-1 write-back at iteration 3 (init seed 1003) returned exactly its
  f = 0 loss after moving early. The net had collapsed to a constant output: the edge loss ignores constants, and the
  GELU hidden layer was dead after an early large Adam update. Fixes:
  - global-norm gradient clipping 1.0;
  - stall/collapse detection (own loss at 2k steps >= 99.5% of the start and the final loss within 0.1% of the start)
    triggers a re-seed, with lr halved per retry and <= 3 tries.
- **Held-out acceptance.** 20% of orbits (the usual hash) are excluded from training: never sampled, incident edges
  weight 0. A fit is accepted only if its exactly symmetric edge loss on 16 held-out batches beats f = 0.
  - A rejected fit is retried.
  - If no try is accepted, the write-back contributes f = 0 or the re-base h = 0, and this is recorded.
  - Applied to both the write-back and the re-base.
- **Re-base steps.** 20k for k <= 2 and 40k for k >= 3 (the increments are small from iteration 3 on in rounds 2-3);
  Adam lr 1e-3.
- **Runs.** 8 iterations, seeds 0, 1, 2.
- **Reported per iteration:** capture frac, re-base retention, <H>, E_FN, guide + 1 / 2 Lanczos steps (equal-hop
  comparison: guide + 1 step against psi_P + 2 steps), and number of small nets. Measured cost per configuration of
  the guide value, its V, W and an edge-loss sample vs k at 6x6 and 8x8 (psi_P and small-net throughput measured on
  the same GPU; expected linear in k for the small-net part).
- **Pass:**
  - >= 70% of the table loop's accumulated <H> gain at iteration 6 for the mean over 3 seeds, no seed below 60%;
  - in addition the 3-seed mean at iteration 8 >= 70% of the table loop's gain there.
  - Reported: whether robustness removed the bad re-bases (number of rejected/retried fits; worst single-iteration
    retention).

## Step size (pre-registered 2026-10-12, before the run)

**Why.** 8x8 follow-up: the semi-implicit FN target is 97% collinear with a Lanczos step, T ~ -(E_L - E)/(1+W), i.e. a
Lanczos step with a per-configuration step 1/(1+W) that is too damped (the uniform optimal step gains 3.6x more).
On 6x6, 8 table-loop iterations (3.16e-5) do not reach one Lanczos step (2.55e-5). The loop's scaling edge exists only
if one iteration (constant hop cost) can gain about as much as one Lanczos step and iterations compound.
Note: the Krylov sign step is the sign of a Lanczos factor 1 + alpha(E_L - E); the full Lanczos amplitude |f| goes to
zero at the flipped configurations (the "damped flip" of cx_l1bump).

**Stage 1 (cx_step.py, exact tables, no networks).** From psi_P, iterate k = 1..4 times each variant: exact Lanczos (L),
clipped Lanczos amplitude log max(|f|, eps) with eps = 0.3, 0.1, 0.03 (bounded, learnable), the current target (SI_1),
and the energy-optimally scaled target lam*T (SI_opt). Step parameter energy-optimal per iteration.

**Pass (stage 1):** a bounded variant (Lc_eps or SI_opt) reaches
- k = 1: error <= 1.2 x psi_P + 1 Lanczos (<= 3.06e-5 per site),
- k = 2, 3: <= 1.5 x psi_P + 2 / 3 Lanczos steps (<= 1.29e-5, <= 5.7e-6),
- with bond-rms roughness of its increment <= 3 x that of SI_1 at k = 1 (learnability proxy).
Sanity: L at k = 1 reproduces 2.55e-5.
**Then stage 2:** network fit (round-4 machinery) of the passing variant; the network loop must keep >= 70% of the
stage-1 gain per iteration. **Fail (energy):** no bounded variant reaches the k = 1 bar -> the per-iteration gap is a
property of bounded targets; cause to be pinned before any conclusion. **Fail (roughness only):** stage 2 decides.

### Step size: stage-1 result (job 17160258, results/compress_6x6/runs/cx_step.json)
Error per site after k iterations (exact tables, no networks):

| k | exact Lanczos, iterated | clipped Lanczos eps=0.3 | current target SI_1 | SI, optimal scale | psi_P + k Lanczos (joint) |
|---|---|---|---|---|---|
| 1 | 2.55e-5 | 2.62e-5 | 9.17e-5 | 5.50e-5 | 2.55e-5 |
| 2 | 1.09e-5 | 1.16e-5 | 6.91e-5 | 4.11e-5 | 8.59e-6 |
| 3 | 6.39e-6 | 7.06e-6 | 5.46e-5 | 2.81e-5 | 3.82e-6 |
| 4 | 4.33e-6 | 4.91e-6 | 4.42e-5 | 2.03e-5 | 1.95e-6 |

Against the pre-registered bars: k = 1 PASS (Lc_0.3 2.62e-5 <= 3.06e-5); k = 2 PASS (1.16e-5 <= 1.29e-5); k = 3 FAIL
(7.06e-6 > 5.7e-6). Cause of the k = 3 fail: the bar was set against the JOINT 3-step Lanczos polynomial; one step at a
time (greedy) is worse even for exact Lanczos (6.39e-6), so no iterate-one-step method can pass it — a bar error, not a
target error. Roughness proxy: FAIL at k = 1 (Lc_0.3 0.052 vs 3 x SI_1 = 0.029; the first step flips 24% of
configurations), PASS at k >= 2 (0.020, 0.011, 0.008). Clipping at eps = 0.3 costs only 3-13% of the energy.
Diagnosis of the current target: T = log(1+W) - log(1+H_xx+V-E) is a difference of logs of extensive quantities, so
its per-configuration change is ~ (E_L - E)/N while the Lanczos change alpha (E_L - E) has alpha ~ fixed (-0.157 at
6x6, -0.161 at 8x8): the semi-implicit target is over-damped by a factor that grows with N (8x8: 3.6x). Even
optimally scaled it reaches only 5.5e-5 at k = 1. Decision: stage 2 with the clipped-Lanczos target (eps = 0.3).

### Step size: stage 2 pre-registration (2026-10-12, before any stage-2 run; code `cx_chop3.py`)
Question: does the callable network loop (round-4 machinery) keep the per-iteration gain of the clipped-Lanczos step?

Iteration k from the callable guide G_k (sign s_k kept as a table, as before):
- **(t) Target, exact on the tables.** One-hop quantities of (G_k, s_k): V, W, local energy E_L = D0 + V - W, E = <H>.
  f = 1 + alpha (E_L - E), d = log max(|f|, 0.3), s_{k+1} = s_k sgn f. alpha is energy-optimal for (G_k + d,
  s_{k+1}), chosen on the exact tables before any fit (as `cx_step.lanczos_state`).
- **(w) Write-back.** FEAT net (37k, round 4) on G_k's own (log a, V, W), fit to d with `robust_fit`: edge loss, tail
  proposal of d, 20k Adam steps, lr 3e-3, clipping, held-out acceptance. The edge loss is taken in the metric of the
  TARGET state (G_k + d, s_{k+1}). Reason: the energy error of the fitted state is, to second order, the Dirichlet form of
  (d_fit - d) in the metric of the state it approximates; round 4's re-base already uses its target's metric. The sign
  change is part of the target, so the old guide's kept bonds are the wrong metric near the new nodes.
- **(r) Residual re-base**, exactly round 4: h_k on psi_P's one-hop features, fit to la_w - G_k in the metric of
  (la_w, s_{k+1}), 20k (k <= 2) / 40k (k >= 3) Adam steps, lr 1e-3; G_{k+1} = G_k + h_k.

Feature pre-test (k = 1, seed 0, one iteration each): round-4 features (log a, V, W) vs the same + a standardised E_L
column. Reason for the extra column: d is an explicit function of E_L = D0 + V - W, which is linear in V and W, while
the net sees log V and log W. **Rule:** the E_L arm is used for the main runs if its k = 1 capture exceeds the
3-feature arm's by >= 0.05; otherwise the round-4 features are kept.

**Main runs:** 2 seeds x 4 iterations.

Measured per iteration (exact):
- alpha, flips (share of configurations whose sign changes), bond-rms roughness of d;
- <H> of the guide, of the exact target state, of the write-back la_w and of G_{k+1};
- capture = (<H>(G_k) - <H>(la_w)) / (<H>(G_k) - <H>(exact target));
- retention = (<H>(G_k) - <H>(G_{k+1})) / (<H>(G_k) - <H>(la_w));
- iteration share = capture x retention = (<H>(G_k) - <H>(G_{k+1})) / (<H>(G_k) - <H>(exact target));
- bond-rms of the write-back error d_fit - d (target metric) and of the re-base error;
- E_FN of G_{k+1}, G_{k+1} + 1 and + 2 exact Lanczos steps, sign error w_s, wall time;
- training-batch and held-out loss curves of every fit (every 2000 steps).

**Pass (primary; this is the README bar "keep >= 70% of the stage-1 gain per iteration"):** iteration share >= 0.70 at
every k = 1..4, mean over the 2 seeds, no seed below 0.60.

Secondary, reported (they decide the scaling reading, not the pass):
- trajectory share (H_start - <H>(G_k)) / (H_start - Lc_0.3(k)), with stage-1 Lc_0.3 = 2.62e-5, 1.16e-5, 7.06e-6,
  4.91e-6;
- comparison at equal k with round 4 and the table loop;
- comparison at equal hops with psi_P + p Lanczos (2.55e-5, 8.59e-6, 3.82e-6, 1.95e-6).
  Hop count: G_k costs one hop of psi_P at every k; + p Lanczos steps make 1 + p hops.

Known risk (stage 1): the k = 1 target is 5.5x rougher (bond-rms 0.052) than the SI target, because 24% of the
configurations flip sign. If k = 1 misses 0.70, the cause is first located (capture or retention). Then remedies are
tried in this order, each on k = 1 only, before any conclusion:
1. 3x more steps;
2. width C = 64;
3. the "direct" arm (below);
4. a smaller first-step alpha (0.5 alpha*).

**Direct arm** (seed 0, 4 iterations; reported, not part of the pass):
- No write-back net: the re-base fits the exact d (la_w = G_k + d).
- d is an explicit one-hop function of G_k, so the write-back net costs the same hop as the exact formula and adds
  only fit error.
- This arm tells whether the write-back net is needed at all.

Budget: <= 8 slice-h (pre-test 2 x ~0.4, main 2 x ~1.2, direct ~1).

#### Amendment S2-1 (2026-10-13 02:50, after the feature pre-test, before the main runs)
Feature pre-test, k = 1, seed 0 (runs `c3PTf3_17164046`, `c3PTf4_17164047`; exact target 2.62e-5 from 1.319e-4):

| arm | write-back held-out loss (f = 0: 7.31e-4) | capture | re-base retention | iteration share | <H> after re-base |
|---|---|---|---|---|---|
| (log a, V, W) | 2.61e-4 (flat from 12k steps) | 0.25 | 0.995 | 0.25 | 1.053e-4 |
| + E_L column | 4.59e-5 | 0.84 | 0.57 | 0.48 | 8.15e-5 |

- By the rule, the E_L column is used (capture +0.59).
- k = 1 still fails (share 0.48). Cause located: the re-base. It fits the write-back output on psi_P's
  (log a, log V, log W) WITHOUT the local energy. That is the coordinate whose absence capped the 3-feature write-back
  at 0.25 with a loss floor (flat from 12k steps, i.e. a representation limit, not an optimiser limit). Consistent with
  this, the 3-feature write-back output, itself a smooth function of those features, was re-based with retention 0.995.
- Remedy, deviating from the pre-registered order (justified by the located cause): add psi_P's asinh-compressed local
  energy to the re-base features (`base_EL`). It costs nothing extra: E_L,P = D0 + V_P - W_P comes from the same hop of
  psi_P.
- Remedy (1) of the pre-registered order (3x steps, 60k, no E_L) runs in parallel at k = 1 as the control
  (`chop3_pt_r60k`).
- Rule:
  - If the seed-0 main run (E_L in both nets) reaches k = 1 share >= 0.70, it continues to 4 iterations and seed 1
    follows with the same settings.
  - Else, if the 60k control reaches >= 0.70, the main runs switch to it.
  - Else the diagnosis continues with remedies (2)-(4) before any conclusion.
- The direct arm uses the same re-base features.
- The pass rule is unchanged.

#### Amendment S2-2 (2026-10-13 08:40, after both main seeds, before this run)
Main arm (E_L in both nets), seeds 0 / 1, iteration share = capture x retention:

| k | capture | re-base retention | share | re-base held-out loss, final / start |
|---|---|---|---|---|
| 1 | 0.84 / 0.84 | 1.00 / 1.00 | 0.84 / 0.84 | 2e-4 / 2e-4 |
| 2 | 0.97 / 0.96 | 0.69 / 0.67 | 0.67 / 0.65 | 0.35 / 0.36 |
| 3 | 1.00 / 1.00 | 0.33 / 0.29 | 0.33 / 0.29 | 0.73 / 0.75 |
| 4 | 1.01 / 1.02 | 0.09 / 0.08 | 0.09 / 0.08 | 0.92 / 0.98 |

- The pre-registered main arm FAILS at k = 2, 3, 4.
- The write-back is not the problem from k = 2 on: capture 0.96-1.02.
- The loss is in the re-base: its held-out loss floors (k = 3, 4: flat over the last half of 40k steps). That is an
  information or representation limit, not an optimiser limit.
- Hypothesis H1: the increment d_k is a function of G_k's one-hop quantities, whose neighbours are split into kept and
  violating by the guide's CURRENT sign s_k. The re-base features split psi_P's neighbours by psi_P's own sign s_P.
  By k = 2, 24% + 9% of the configurations have flipped sign, so the features no longer carry the split the target
  was computed with.

Test (seed 0, 4 iterations, `chop3_prev_s0`):
- Re-base features = psi_P's (log a, log V, log W, asinh E_L) with the neighbours split by s_k, the guide's sign at the
  start of iteration k (for k = 1 this is s_P, i.e. the main arm).
- Cost at 6x6: one hop of psi_P plus sign look-ups at the neighbours, which any energy evaluation of the guide needs
  anyway. At scale, each stored increment h_j needs its own sign s_j at the neighbours (cost linear in k, open).
- Blow-up guard: reject a re-base whose new guide puts > 1e-3 of its weight on configurations where the re-base error
  exceeds +1 in log-amplitude; retry with a new seed, at most 2 retries.
  - Reason: the k = 1 control without E_L at 60k steps (`c3PTr60`) reached held-out loss 2.1e-5. Its guide still
    blew up to <H> 0.61 per site, an error of +20.7 on configurations that carry 80% of the new guide's weight.

Reading:
- H1 is supported if the re-base retention at k = 2..4 reaches >= 0.7 (main arm: 0.69 / 0.33 / 0.09).
- A rise that stays below 0.7 points to the remaining missing information: the amplitude change at the neighbours,
  i.e. two hops of psi_P.
- No rise refutes H1.
- The pass bar is unchanged. A pass of this arm would need seed 1 before any claim.

### Step size: stage-2 result (2026-10-13; details in `results/compress_6x6/README.md`)
- **FAIL** against the pre-registered bar. Mean share per iteration 0.84 / 0.66 / 0.31 / 0.09 (bar 0.70 at every k).
- k = 1 passes. One iteration gives <H> 4.3e-5, against 4.2-5.1e-5 after 8 round-4 iterations.
- The write-back is near-exact from k = 2 on (capture 0.96-1.02).
- The re-base on one hop of psi_P has a fixed absolute error (~0.013 bond-rms), so retention falls to 0.09 and the
  loop plateaus at ~2.0e-5.
- Amendment S2-2 (sign-matched re-base features) is refuted: retention 0.55 at k = 2.
- Remaining cause: the increments need two-hop information of psi_P, i.e. the guide's own neighbours.
- At 1 hop the k = 4 guide (2.0e-5) beats psi_P + 1 Lanczos step (2.55e-5). At 2 hops it ties psi_P + 2 Lanczos steps.

## Fold-back diagnostics D1 / D2 (pre-registered 2026-10-13, before any run; code `cx_lightcone.py`)
**Question.** Why does the residual re-base keep 1.00 / 0.68 / 0.31 / 0.09 at k = 1..4 (stage 2), with a fixed
absolute error of ~0.013 bond-rms? There are three candidate causes:
- (I) **information / light cone.** d_k is a sharp function of E_L of G_k. That quantity needs G_k at the neighbours,
  i.e. the earlier increments h_j at the neighbours, i.e. psi_P's one-hop features at the neighbours (a second hop).
- (F) **fit floor.** An Adam noise or scale floor of the 37k net, independent of the inputs.
- (S) **target scale.** The increments shrink, so a fixed absolute error looks like falling retention.

Already on file:
- At k = 1 the same net, optimiser and steps reached a held-out loss of 1.7e-7 from 8.6e-4 (5000x). There the target
  is a function of E_L,P, which is an input.
- At k >= 2 the held-out loss floors at 2e-5 to 3e-5 (ratio 0.35-0.92).
- This already argues against a generic optimiser floor. D1 tests it directly.

**State.** Seed-0 main arm (run `c3s0_17165794`), iteration k = 3:
- G_3 = log|psi_P| + h_1 + h_2 (stored params), sign s_3 = s_2.npy;
- exact clipped-Lanczos target d_3, recomputed exactly as cx_chop3.
Sanity checks (abort if they fail):
- <H>(G_3, s_3) = 2.4951e-5 within 1e-9;
- <H> of the exact target = 1.1876e-5 within 1e-8.

**D1 (fits; re-base of the EXACT d_3 in the metric of the target state; robust_fit, Adam lr 1e-3, same init seed 53
in all arms, 40k steps unless stated).**
Arms:
- R0 = round-4/S2 features (psi_P one hop: log a, log V, log W, asinh E_L,P);
- RA = R0 + asinh E_L of G_3 itself (the information test);
- RB = R0 with the target divided by its rms and the output multiplied back (the scale test);
- RC = R0 with 120k steps (the optimiser test).
Measured: retention r = (<H>(G_3) - <H>(G_3 + h)) / (<H>(G_3) - <H>(target)); re-base bond-rms error; held-out loss
final / start.
Readings:
- (I) confirmed if RA >= 0.85 and R0, RB, RC <= 0.5.
- (S) if RB >= 0.7. (F) if RC >= 0.7.
- If RA < 0.7, the fit is limited even with the information: (I) is not confirmed and the cause is open.

**D2 (exact, no fits).** The light-cone construction.
- y = x^b is a neighbour of x. psi_P's one-hop features of y need psi_P at x^{b b'}.
- Level rho: exact psi_P (amplitude and sign) at x^{b b'} if the bond distance d(b, b') <= rho (minimum periodic site
  distance; 0 = shared site, always exact because the validity changes; b' = b gives x itself).
- Otherwise use the additive approximation:
  - la(x^{bb'}) ~ la(x^b) + la(x^{b'}) - la(x);
  - s(x^{bb'}) ~ s(x^b) s(x^{b'}) s(x).
  This needs only the first hop.
- Levels: '0th' (u_3(x^b) := u_3(x), i.e. no neighbour information), 'Fx' (h_j at the spins of x^b but with x's
  features), rho = 0, 1, sqrt2, 2, and exact (sanity: must reproduce the G_3 table to 1e-5).
- For each level:
  - the approximate E_L,3(x) = D0 + sum_b (J_b/2) s_3(x) s_3(x^b) exp(G^_3(x^b) - G_3(x)), with
    G^_3(x^b) = la_P(x^b) + h_1 + h_2 evaluated on the level's features of x^b;
  - the induced error of the target, d(E^) - d(E) (offset removed; alpha and E as in d_3);
  - the weight on which the clipped sign sgn(1 + alpha(E - E)) disagrees.
- Samples: 2048 x from |psi_target|^2 and 2048 from the fit proposal Q.
- Cost reported: exact second-hop evaluations per bond, n_rho.
- Reference: pointwise error of the main-arm re-base h_3 against d_3 on the same samples.
- D2a: ratio interaction I(b, b') = la(x^{bb'}) + la(x) - la(x^b) - la(x^{b'}) and sign interaction
  s(x^{bb'}) s(x) s(x^b) s(x^{b'}), by bond distance, on 1024 samples of |psi_P|^2. Also the rms error of the four
  standardised features of x^b at each level (edge-weighted and unweighted).
Readings:
- Information explains the floor quantitatively if the '0th' d-error is >= 0.5 x the main-arm re-base error.
- **The light-cone re-base is viable at 6x6** if some level with rho <= sqrt2 reduces the d-error to <= 0.25 x the
  '0th' error AND the flip-disagreement weight to <= 1e-3.
- If it needs rho = 2 or exact, locality is not demonstrated at 6x6 (6x6 is too small to see a decay beyond d = 2;
  then 8x8 sampled data decide).
Budget: D1 ~1.1 slice-h, D2 ~0.4 slice-h (total <= 2 GPU-h).

#### Amendment FB-1 (2026-10-13 10:20, after D2 and D1 arms R0/RA, before this run)
- D2a found O(1) ratio interactions of psi_P out to the largest bond distance on 6x6:
  - rms I = 1.68 at distance 1, 1.02 at 2, 0.82 at 2.83;
  - sign non-additivity 19% at distance 1 and 7% at 2.83.
- D2b: every light-cone level is WORSE than ignoring the neighbours' information.
- The question is whether this is a property of the network guide (symmetrisation = log of a sum, global attention)
  or of the 6x6 ground state itself (finite size, near-critical J2 = 0.5).
- `cx_pairint.py` repeats D2a on the exact ground state |psi0| (and on psi_P as a reproduction check). It also adds
  the mean contribution error r_b' |sigma e^I - 1| of the additive approximation per bond distance.
- This is descriptive, with no pass/fail. Reading:
  - if psi0 shows the same O(1) interactions, 6x6 cannot test locality and only a larger lattice can;
  - if psi0 is markedly more additive (rms I at distance >= 2 below 0.3), the non-locality is the guide's, and the
    light cone would need a local base network.

## Rolling re-anchor test (pre-registered 2026-10-13, lead/PI approved; before any run; code `cx_reanchor.py`)
**Question.** Can the base network itself absorb each loop increment, so that the fold-back has no hop radius
(design file section 5.3)?
- The guide is the fine-tuned symmetrised ViT at every k (0 hops). The next target comes from one exact hop of it.
- There is no stack and no re-base.

**Iteration k (exact 6x6 tables, J2/J1 = 0.5; sign kept as a table, as in stage 2):**
1. **State (la_k, s_k).** For k = 1 it is the stage-2 start (log|psi_P|, s_P): <H> 1.319e-4, exact target 2.62e-5.
   For k >= 2, la_k is the full-sector table of the previous iteration's fine-tuned network.
2. **Target.** Exact clipped-Lanczos target from (la_k, s_k), exactly as stage 2: f = 1 + alpha (E_L - E),
   d = log max(|f|, 0.3), s_{k+1} = s_k sgn f, alpha energy-optimal. Target state la_t = la_k + d.
3. **Fine-tune** of the symmetrised ViT (warm: the checkpoint for k = 1, the previous iteration's parameters after
   that) to log A = la_t.
   - Loss: the stage-2 edge loss on the kept bonds of the target state (la_t, s_{k+1}), weight (J/2) a_t(y)/a_t(x).
   - Sampling: increment-weighted, i.e. 1/2 a_t^2 + 1/2 Dirichlet weight of d (lab.proposal_q). 20% of orbits held
     out (never sampled, their edges weight 0).
   - **minSR arm (main):** damped Gauss-Newton in sample space.
     - Batch 384 configurations x 8 kept bonds; the update is -J^T (J J^T + lam I)^{-1} r.
     - Trust region: the predicted rms change of log A on the batch is <= 0.5 x the target's pointwise rms.
     - Acceptance on a FIXED set of 16 training-orbit validation batches (256 x 8), old and new compared on the same
       samples. lam is halved on acceptance and multiplied by 4 on rejection (bounds 1e-6 to 1e2, relative to
       tr(JJ^T)/M).
     - Change vs P2's GN, with the reason: P2 accepted steps on 3 fresh 128 x 8 batches. Its batch-to-batch loss
       spread (+-40%) exceeded any real per-step gain (~1%), so acceptance was a coin flip (rate 0.41), lam rose to
       1e3 and the steps shrank to |d| ~ 1e-6. The fixed paired validation set removes that noise.
   - **Adam arm (control):** lr 1e-6 (the P2 scan value), warmup-cosine, batch 256 x 8.
   - Both arms: 12 min of fine-tuning per iteration on a 24 GB A40 slice.
   - Held-out acceptance: keep the fine-tuned parameters only if the edge loss on 16 fixed held-out batches is
     below its value at the start. Otherwise revert.
4. **Exact evaluation.** Full-sector table of the network (~14 min).
   - Blow-up guard as in S2-2: reject if > 1e-3 of the new state's weight sits where la_new - la_t (offset removed)
     exceeds +1.
   - Measured: <H>(la_new, s_{k+1}); iteration share = (<H>_k - <H>_new) / (<H>_k - <H>_target); bond-rms error in
     the target metric; E_FN; + 1 / + 2 exact Lanczos steps; w_s; own and held-out loss curves.
5. **fp32 precision floor (lead's request).** At the start of every iteration, on 1024 target-metric samples and
   their kept edges: log A in fp32 vs fp64 (same parameters). Floor = edge-weighted rms of the bond differences of
   (fp32 - fp64), compared with the same estimator for the increment d.
   - The test is meaningful only if floor <= 0.1 x the increment's edge rms. Otherwise that iteration is
     "inconclusive (cause: fp32 floor)".
   - P2 smoke: pointwise 1.0e-4 rms, max 3e-3. Stage-2 increment bond-rms at k = 1, 2, 3: 0.052, 0.028, 0.020.
   - So the expected ratio is about 1e-2, well clear.

**Target normalisation.** For the ViT it is not a separate option:
- the GN solution is linear in the residual, and Adam is invariant to the gradient scale;
- the trust region above is set relative to the target's size.
For the small-net fits, normalisation is now the default in `robust_fit` (D1: 0.02 -> 0.33). Runs before 2026-10-13
11:00 used the unnormalised fit.

**Runs.**
- minSR: seeds 0 and 1, k = 1, 2, 3, one job per iteration.
- Adam: seed 0, k = 1.
- Continuation rule: an arm goes on to k + 1 only if its share at k >= 0.30.

**Pass / fail:**
- **PASS:** minSR iteration share >= 0.70 at every k = 1, 2, 3 (mean over the 2 seeds, no seed below 0.60).
- **Fallback (lead's rule):** both optimisers < 0.30 at k = 1 means the warm ViT cannot hold increments at the
  required precision. Then cap the loop at <= 2 hops + Lanczos polish, and test a local base architecture only if
  8x8 pair interactions allow it.
- **Otherwise inconclusive (cause: X),** diagnosed from:
  - the acceptance rate and lam history;
  - own vs held-out loss;
  - the per-decade bond error (tail vs bulk);
  - the fp32 floor.

**Budget:** <= 4 slice-h (7 jobs x ~0.55 h).

#### Amendment RA-1 (2026-10-13 10:52, after the first 20 minSR steps of jobs 17174449/50, before any evaluation)
- **Observation.** The minSR direction predicted rms changes of log A of 2.7-20 on its own batch, against a trust
  radius of 0.016. So every step was cut to 0-1% (own loss 6.34e-4 -> 5.80e-4 in 20 steps), while lam fell.
- **Cause.** The edge loss leaves the common mode of each sampled star (x and its 8 neighbours) free. The
  minimum-norm sample-space solution uses it.
- **Fix.** P2's value rows: sqrt(0.1 iw/B)(e_x - mean), with residual e = log A - la_t. This is a Tikhonov term in
  function space, in the target metric. Steps are accepted on the same combined objective, on the fixed validation
  set. The trust radius is measured on the x samples.
- The two minSR jobs were cancelled at 6 min; the Adam control (unaffected) continues.
- Held-out acceptance, pass rule and budget are unchanged.

#### Amendment RA-2 (2026-10-13 11:40, after the three k = 1 jobs, before this diagnostic)
- **k = 1 result.** Every arm is NEGATIVE:
  - minSR seed 0: share -0.14; seed 1: -0.12;
  - Adam: -0.10;
  - for reference, the new sign with the old amplitude gives <H> 1.347e-4, a share of -0.03.
- The fits nevertheless halve the held-out edge loss (minSR 0.50-0.52, Adam 0.67). The pointwise error is
  0.003-0.006 in the bulk decades (a_t^2 >= 1e-8), but 0.09 / 0.21 / 0.39 / 0.64 / 0.91 at 1e-11 ... 1e-15.
- **The pre-registered fallback is triggered:** both optimisers < 0.30 at k = 1. So the k = 2, 3 jobs stop by the
  continuation rule.
- Before the cause is written down (rule 1), `cx_reanchor_diag.py` splices each fitted table with the exact target
  (and with the start) by probability decade of the target state, and computes exact <H>. No fits; ~0.2 slice-h.
- Reading:
  - if "fit in the bulk + exact target in the tail" recovers >= 0.7, the loss sits in the tail decades the ViT
    cannot follow;
  - if not, the bulk fit itself is energetically wrong (e.g. the edge errors are correlated in an energy-costly way).

## Sign-channel check (pre-registered 2026-10-13, lead request; before any run; code `cx_signchan.py`; <= 1 slice-h)
**Lead's premise and a correction.** The premise: the reported energy is E_FN of the loop's sign, so amplitude
errors on rare configurations matter only through the next sign decision.
- On the lattice this holds only in part. The fixed-node Hamiltonian H_FN[a, s] has the diagonal wall term
  sum_violating |H_xy| a(y)/a(x), so E_FN depends on the guide amplitude, and most strongly on the ratios across
  violating bonds, i.e. at the nodes, in the rare region.
- Measured already (P2 replay, same sign s_3): E_FN = 9.29e-5 with psi_P's amplitude vs 5.25e-5 with the stack's
  amplitude.
- So two quantities are reported:
  - (i) the sign channel: E_FN of the new sign with a COMMON amplitude;
  - (ii) the guide's own E_FN.
- An amplitude-free score would be the self-consistent E_FN* of a sign (iterate a <- phi_FN[a, s]). It is not
  computed here (budget).

**Pairs (fitted guide, exact reference with the same current sign):**
- RA-s0, RA-s1, RA-adam: the three k = 1 re-anchored ViT tables vs the exact target la_P + d_1, sign s_1.
- S2-G2: stage-2 seed 0, G_2 = la_P + h_1 vs la_P + d_1, sign s_1.
- S2-G3: G_3 = G_2 + h_2 vs G_2 + d_2, sign s_2.

**Next sign step from each amplitude (both rules):**
- (C) clipped-Lanczos sign: s' = s sgn(1 + alpha (E_L - E)), alpha energy-optimal for the clipped state (as stage 2);
- (K) global Krylov step: the same form, with alpha chosen to minimise <H>(a, s') at FIXED amplitude.

**Metric.**
- Gain = E_FN[la_ref, s] - E_FN[la_ref, s'], i.e. the fixed-node energy with the reference amplitude as the common
  guide.
- Computed for s' from the fitted amplitude (Gain_f) and from the reference amplitude (Gain_r).
- Also reported: E_FN[la_f, s'_f] (the guide's own amplitude), E_FN[la_ref, s] and E_FN[la_f, s].

**Iteration (rule C, pairs RA-s1 and S2-G3, as the time cap allows).** Steps 2-3 use the exact FN amplitude of
each chain's own sign between steps (a <- phi_FN[a, s']), then the next sign step.
- This isolates how a first-step sign difference propagates.

**Pass** ("rare-region amplitude errors are irrelevant to the FN-reported sign channel"): Gain_f >= 0.8 Gain_r at
every step, for both rules and all pairs. Otherwise report by how much and where.

Two jobs (RA pairs; stage-2 pairs), each capped at 28 min of work after setup, items in this priority order:
1. rule C, step 1;
2. rule K, step 1;
3. the iteration.
Also tabulated without computation: <H> and E_FN side by side for all stage-2 (both seeds) and re-anchor guides.

## Hybrid re-anchor test (pre-registered 2026-10-13, lead/PI approved; before any run; code `cx_hybrid.py`; <= 3 slice-h)
**Idea (design 5.5).** The guide is G_k = log A_theta_k + c_k, with two parts:
- the re-anchored ViT, which carries the common configurations (0 hops);
- one correction net c_k (FEAT net, 37k parameters). Its inputs are ONE hop of the current ViT: log A, log V, log W
  and asinh E_L of (A_theta_k, s_k).
- c_k is refit every iteration to the whole residual, not stacked. So the hop cost is constant: the guide value
  costs 1 ViT hop, its local energy 2.

**Iteration k (exact 6x6 tables; sign kept as a table, as stage 2):**
1. Exact clipped-Lanczos target from (G_k, s_k): la_t = G_k + d_k, s_{k+1}. Start: G_1 = log|psi_P|.
2. **Re-anchor.** minSR fine-tune of the ViT (warm from theta_k) to la_t, exactly as the re-anchor test (amendment
   RA-1 settings, 12 min, held-out acceptance). Then the full table la_A.
   - k = 1 reuses the already-run minSR seed-1 fit to the identical target (`ra_ms_s1_k1_17174477`, table and
     parameters) to save budget.
3. **Correction.** c_{k+1} is fit by robust_fit to the residual la_t - la_A:
   - normalised target (now the default), metric of the target state;
   - features: one hop of the new ViT (la_A, s_{k+1}), standardised in the target metric;
   - Adam lr 3e-3, 30k steps, held-out acceptance;
   - blow-up guard: reject if > 1e-3 of the new guide's weight sits on +1 excess, then c = 0.
4. G_{k+1} = la_A + c_{k+1}. Exact evaluation:
   - <H>(G_{k+1}, s_{k+1}) and iteration share = (<H>(G_k) - <H>_new) / (<H>(G_k) - <H>_target);
   - the guide's own E_FN (fn_solve);
   - the same two quantities for the ViT alone (la_A);
   - + 1 Lanczos step; bond-rms error.

**Runs.** One seed (budget), k = 1, 2, 3, one job per iteration. The next iteration runs only if share >= 0.30.

**Pass:** iteration share >= 0.70 at every k = 1, 2, 3, scored on <H>.

**Also scored on the guide's own E_FN.** The baseline to beat is stage-2 seed 0 at the same k:
- <H> 4.34 / 2.50 / 2.06e-5;
- E_FN 2.94 / 1.89 / 1.58e-5;
- that guide costs 1 hop of psi_P and the same 2 hops for E_L.

**Readings:**
- Pass on <H> together with E_FN <= the stage-2 value at each k: the hybrid is the constant-cost fold-back.
- Pass on <H> but not on E_FN: report it; that is the amplitude-at-the-nodes issue of the sign-channel check.
- Share < 0.7: locate the shortfall (the correction's held-out loss and capture of the residual, splices by
  decade as in RA-2), with no conclusion before the cause is known.

#### Amendment HY-1 (2026-10-13 16:20, after hybrid k = 1 and k = 2, before this diagnostic)
- **Result so far.**
  - k = 1: share **0.963**. <H> 3.01e-5 (target 2.62e-5; stage-2 4.34e-5); own E_FN 2.17e-5 (stage-2 2.94e-5).
  - k = 2: share **0.230**. <H> 2.64e-5 (target 1.40e-5; stage-2 2.50e-5); own E_FN 1.71e-5 (stage-2 1.89e-5).
  - k = 3 stops by the continuation rule.
- **Hypothesis.** The same one-shell gap as D1, now in the tail.
  - d_2 is a function of E_L of the GUIDE G_2 = ViT_1 + c_1. That needs c_1 at the neighbours, i.e. one-hop
    features of ViT_1 at the neighbours: two hops.
  - The correction c_2 sees only one hop of ViT_2.
- **Test** (`cx_hybrid_diag.py`, about 0.3 slice-h, no new ViT fit; uses the k = 2 ViT table):
  - (a) splices of the k = 2 hybrid guide with the exact target by probability decade (as RA-2);
  - (b) the same correction fit with ONE extra input, asinh E_L of G_2 (exact table), the information upper bound.
- Reading:
  - (b) >= 0.85 confirms that the k >= 2 shortfall is the one-shell information gap;
  - (b) < 0.7 points to the fit or the ViT part; (a) locates it.

## Lanczos-wrapped network: exact pre-test W0 (pre-registered 2026-10-13, before any run; code `cx_wrap_pretest.py`)
**Proposal (lead).** Store the guide as G = (1 + beta (H - E)) N, with N a spin-only network (the re-anchored ViT).
- Evaluating G costs one hop of N.
- Rationale: in the tail the wrapped amplitude is inherited from N on common neighbours, beta sum_y H_xy N(y). So
  N would need to be precise only on common configurations.

**Question before any training.** Does wrapping make N's tail errors irrelevant? This is decided exactly, with no
fits, using the stored k = 1 tables:
- N* = the exact clipped-Lanczos target of psi_P (what the re-anchor fits aimed at);
- N_fit = the re-anchored ViT tables (minSR seed 1, Adam). Both have sign s_1.
- W(X) = exact energy of the wrapped state (1 + beta (H - E)) X, with the optimal beta (one exact Lanczos step on
  the table).

Measured:
- W(N*), W(N_fit);
- splices by decade of N* (cuts 1e-8, 1e-9, 1e-10): (a) fit in the bulk + N* in the tail; (b) N* in the bulk + fit
  in the tail;
- share_w(X) = (W(N_fit) - W(X)) / (W(N_fit) - W(N*));
- own E_FN of the wrapped N* and the wrapped N_fit;
- an inheritance statistic for the wrapped N*: by decade, the weight where the inherited term |beta (E_L - D)|
  exceeds the own term |1 + beta (D - E)|.

**Reading:**
- **SOUND** (go to the training test): (b) at the 1e-9 cut gives share_w >= 0.8, i.e. N's tail errors below 1e-9
  cost <= 20% of the wrapped gain.
- **NOT SOUND**: (b) < 0.5. The wrapper inherits, but N's own tail errors still dominate the wrapped energy.
  No training run; report.
- In between: report, and decide the training test with the lead.

Budget ~0.3 slice-h.

## Route B: a spin-only tail representation (pre-registered 2026-10-13, PI decision; before any run; code `cx_tailnet.py`; <= 3 slice-h)
**Why spin-only nets fail on a^2 < 1e-9 (design reason, before compute).**
1. **The precision required there (0.003 in bond log-ratio) is below the base's own error.** The ViT is off by
   0.05-0.5 in those decades (P2). The residual to learn is therefore the increment plus the base's tail error.
2. **The tail target has an inherited structure.** For a rare x the step gives la_new(x) ~ log sum_y |H_xy| a(y), a
   log-sum-exp over the ~70 neighbours' amplitudes. A spin-only net must emulate the base at all neighbours
   implicitly, which is a different and harder function class than the base itself. This is why one-hop inputs
   reached 89% while spin-only students stayed at 20-31% (writeback_tail).
3. **The data per orbit is tiny.** 7.7% of the weight is spread over ~1e7 orbits, so generalisation must come from
   structure.
   - Learnability is shown with one-hop inputs (held-out = train), not with spins alone.
   - fp32 is not the limit (floor 7e-6 edge rms).
4. **What could make it work.**
   - Give the tail net the neighbours' information at ZERO hop cost: the base's input gradient. One backward pass
     gives a linearised log-ratio r^_b = sum_{i in b} (d la / d x_i)(-2 x_i) for every bond, and from it V^, W^, E^_L.
   - Add capacity (C = 64).
   - Use energy-matched sampling (1/2 |psi_t|^2 + 1/2 Dirichlet weight of the residual, i.e. the tail proposal).
   - Literature: |psi|^alpha / overdispersed sampling for NQS gradients (arXiv:2507.05352) addresses sampling, not
     tail representation. I found nothing on input-gradient features for NQS.

**G0 (pre-test, about 10 min).** On 2048 samples of the target state and 2048 of the tail proposal: exact one-hop
log-ratios of ViT_1 (the k = 1 re-anchored ViT, ra_ms_s1_k1) against the linearised r^_b; exact E_L of (ViT_1, s_1)
against E^_L.
- Gradient features are useful if corr(asinh E^_L, asinh E_L) >= 0.9 on the tail-proposal samples.
- Otherwise arm TG is dropped.

**T1 (iteration 1, exact tables).**
- Additive tail net T on top of ViT_1: G = la_A + T, fit to the residual la_t1 - la_A (robust_fit: normalised,
  energy-matched tail proposal, held-out acceptance, guard).
- Arms:
  - TS: spins + la_A, C = 64, 40k steps;
  - TS32: same with C = 32, the capacity control;
  - TG (only if G0 passes): + linearised (log V^, log W^, asinh E^_L), C = 32. This needs the input gradient on
    all 15.8M orbits, about 50 min.
- References: ViT alone -0.12; hybrid with exact one-hop inputs 0.963.
- **Pass:** share >= 0.70 at iteration 1. Then iteration 2 of the full loop for the best passing arm (ViT
  re-anchor + new tail net), pass >= 0.70, within the budget.
- Otherwise: "inconclusive (cause: rare-configuration precision of spin-only inputs)", or a concrete cause if the
  diagnostics locate one.

## Adaptive-depth guide (pre-registered 2026-10-13, lead idea; before any run; code `cx_adaptive.py`; <= 1 slice-h)
**Idea.** The hard structure is localised in CONFIGURATION space (near the base's nodes), not in real space.
- Evaluate with one hop by default. Descend one more hop only for configurations flagged as near-node by a cheap
  one-hop criterion.
- Over iterations this is a branching process. The cost stays bounded iff the branching ratio r (mean number of
  flagged neighbours of a flagged configuration) is < 1.

**Test** (hybrid iteration 2, exact tables; reuses HY-1). The correction c_2 (one hop of ViT_2) gets one extra input:
- E_mix(x) = exact E_L of the guide G_2 (two hops) if x is flagged;
- otherwise the zeroth-order one-hop value E_L^0 = E_L of (ViT_1, s_2), i.e. the earlier correction's differences
  at the neighbours dropped.
- References: HY-1 with the exact E_L everywhere (p = 1): share 0.925; no extra column: 0.230.

**Flag criteria** (all one-hop or zero-hop, thresholds set as quantiles of |psi(G_2)|^2):
- F1: |asinh((E_L^0 - E)/sd)| (large local energy);
- F2: V^0/(1 + W^0) (wall-to-kept ratio of ViT_1 with sign s_2);
- F3: |c_1(x) - mean| (the size of the earlier correction at x; zero hop).

**Measured for each criterion at p_flag = 1%, 2%, 5%:**
- (a) p_flag under |psi(G_2)|^2 and under the tail measure Q;
- (b) branching ratio r: mean number of flagged neighbours of flagged configurations, from 2000 flagged samples;
- the rms error of the asinh-standardised E_mix against the exact column;
- (d) cost per guide value in hop units: 1 + p_flag n_nb / (1 - r), with n_nb the mean number of valid bonds (about
  72 at 6x6, 145 at 8x8). Extrapolated to 8x8 with the same p_flag.
- (c) **fits:** c_2 refit (robust_fit, 30k Adam, seed as HY-1) for p = 0 (E_L^0 everywhere) and for F1, F2, F3 at
  p_flag = 2%. Share of the iteration-2 step, own E_FN.

**Pass:** some criterion reaches share >= 0.70 with cost <= 3 hops at 6x6 AND r < 0.5.
- Also reported: whether the same p_flag keeps the cost <= 3 hops at 8x8. That needs p_flag <~ 1.4% (1 - r).

## Bond-field fixed node (brainstorm idea 1) + Delta_B (idea 2): exact kill test (pre-registered 2026-10-13, lead approved; before any run; code `cx_bondfield.py`; <= 2 slice-h)
**Idea 1.** The lattice FN bound needs only:
- a sign field s;
- positive reciprocal ratios r_xy = 1/r_yx on the sign-violating ("bad") bonds. These enter the wall diagonal
  sum_bad |H_xy| r_xy.
Amplitudes, tails included, are generated by the projector and never stored. min_r E_FN(s, r) = E_s*, the best
variational energy with sign s, reached by iterating r <- ratios of the current FN solution.

**For each sign s** (s_P = ViT sign; s_1 = the k = 1 clipped-Lanczos sign from psi_P; s0 = exact sign as a sanity
check, where (b) and (c) must give 0), exact lowest eigenvalue of H_eff(s, r) (fn_solve):
- (a) r from the ViT, |psi_P|;
- (b) r from |psi0|;
- (c) self-consistent optimum: iterate r <- ratios of phi_FN from (a), until the change is < 1e-8 per site (at
  most 20 solves);
- (d) a trained bond-ratio field. It is parametrised as ratios of a_hat = |psi_P| exp(f_theta), so the reciprocity
  r_yx = 1/r_xy holds exactly. f_theta is fit ONLY on the bad bonds of s to the converged log(phi*/|psi_P|), with an
  edge loss weighted by |H_xy| phi*_x phi*_y. That equals the second-order expansion of E_FN, since
  dE ~ sum_bad |H| phi_x phi_y (cosh(d log r) - 1).
  - d1: f on spins + log|psi_P(x)|. Zero extra hop: a GFMC walker step already evaluates psi_P at all neighbours.
  - d2: f on psi_P's one-hop features (log a, log V, log W, asinh E_L). One extra hop per walker step.
  - Both FEAT nets as stage 2 (robust_fit, normalised target, 30k Adam, held-out acceptance).

**Idea 2.** E_B = ground energy of the sign-free Hamiltonian (all off-diagonal elements -|H_xy|), computed by Lanczos
on 2 D0 - H.
- Delta_B = (E0 - E_B)/N. Stated in advance: |s_x H_xy s_y| = |H_xy|, so Delta_B does NOT depend on the stored sign.
  The sign quality enters released-node / path methods only through the guiding function, not through Delta_B.
- Check: the Perron vector must have one sign in the sector basis (trivial-character sector).

**Pass (idea 1, lead's bar):**
- (d) recovers >= 70% of the (a) -> (c) gap, for the best stored sign; AND
- (c) for the best stored sign is below psi_P + 2 Lanczos (8.6e-6 per site).
- Also reported against ViT + 2 Lanczos + variance extrapolation (1.5e-6 from p = 0, 1, 2; not variational).
- Kill the r part if (d) recovers < 30%. Kill the route if (c) is not below 8.6e-6 for s_P and s_1.

## Bond-field FN, round 2: training the field on the FN energy (pre-registered 2026-10-19, lead approved; before any run; code `cx_bondfield2.py`; <= 4 slice-h; exact 6x6, sign s_1)
**Gradient check (derived here).** H_eff(r) = H on kept bonds + diag(D0 + sum_bad |h_xy| r_xy), with r_yx = 1/r_xy.
- For the normalised ground state phi, Hellmann-Feynman gives dE/dlog r_xy = |h_xy| (r_xy phi_x^2 - phi_y^2 / r_xy).
  It vanishes at r = phi_y/phi_x. This is the lead's formula.
- With phi held fixed, E_phi(r) = <phi|H_eff(r)|phi> = const + sum_{bad pairs} |h| phi_x phi_y (t + 1/t - 2), with
  t = r phi_x / phi_y.
  - It majorises E_FN: E_FN(r) <= E_phi(r), with equality at the current r.
  - Its gradient at the current r equals the Hellmann-Feynman gradient (envelope theorem).
- So "descend E_FN" is done by minimising the exact majoriser. Loss per bad bond: |h| phi_x phi_y 2 (cosh delta - 1),
  with delta = log r - log(phi_y/phi_x).
- Then re-solve phi and refit. A refit is accepted only if the exact E_FN decreases, which makes the loop monotone.
- Symmetry: r = ratios of a_hat = |psi_P| exp(f), with f orbit-invariant (FEAT nets on orbit features), so H_eff
  keeps the full symmetry and the orbit tables apply.

**Runs:**
1. **Bad bonds per configuration.** The number of valid bonds with s_1(x) s_1(x^b) > 0, under |phi|^2 for the field
   of (c), and under |psi_P|^2. This sets the hop cost of fields with one-hop features: one extra hop of psi_P per bad
   neighbour of a walker.
2. **(c) converged further:** up to 60 self-consistent solves, or until the change is < 1e-9 per site. This fixes
   the ceiling (20 solves gave 4.62e-6).
3. **Iterate-and-refit,** starting from a_hat = |psi_P|, phi_0 = phi_FN; up to 6 outer rounds, each a fresh 30k-step
   FEAT fit with the exact majoriser loss on bad bonds only, then an exact re-solve. Feature arms:
   - A: psi_P's one-hop features with neighbours split by s_1 (log a, log V, log W, asinh E_L).
     Cost: 1 + n_bad psi_P hops per walker step.
   - B: A + the current FN solution's own one-hop quantities at x (log phi, log V_phi, log W_phi, asinh E_L,phi).
     **Honest cost:** phi is never stored in a GFMC run, so these inputs are NOT available to a walker. B is an
     information upper bound only.
   - C (reciprocal edge net on the local spin patch + psi_P's bond ratio): conditional, only if budget remains after
     1-3; it needs a custom wall operator.
4. **Pass:** the trained field (arm A, or C if run) reaches E_FN <= 1.0e-5 per site at the stated hop cost. That is
   below psi_P + 1 Lanczos (1.89e-5 in E_FN) and approaching psi_P + 2 Lanczos (8.6e-6).
   - B passing alone means: information sufficient, but not available to a walker.
   - Also reported: the fraction of the converged ceiling's gap that is closed.

#### Provenance note (2026-10-19)
- The round-2 pre-registration text above and the cx_lib.py changes (cosh loss, violating-first sampling) first
  landed on main in commit ac6b89e (2026-10-19 13:45:13). That was another worker's commit, which swept in the
  uncommitted files from the shared worktree; the committed text equals the text I wrote.
- The code (cx_bondfield2.py, specs) followed in e649189 (13:45:43).
- The three round-2 jobs (17354415-17) were still PENDING when this note was written, so the pre-registration
  precedes every run.
- The RESEARCH_MAP.md change in ac6b89e is not from this route.

#### Amendment BF2-1 (2026-10-19 14:40, after arm A rounds 1-2 and ceiling solves 1-15, before anything below)
- **Observation.** Arm A tracks the exact self-consistent iteration closely: round 1 gives 6.09e-5 vs exact solve 1
  6.00e-5; round 2 gives 4.55e-5 vs 4.33e-5.
- **Problem with the original cap.** The exact iteration itself needs about 10 solves to reach 1.0e-5 (solve 10:
  1.04e-5), so a 6-round cap cannot reach the pass bar by construction.
- **Change.** Arm A continues from its round-6 field for up to 10 more rounds (16 in total). It is a resume job;
  the field is saved every round.
- **Arm B (the information upper bound, not available to a walker) is dropped** to keep the budget (<= 4 slice-h).
- **The pass bar is unchanged.**
- **Add-on (lead / AVSL request): error structure** of the final trained field against the converged ceiling
  field phi*, on bad bonds sampled from |phi*|^2 with weights |h| phi*_x phi*_y. Reported:
  - weighted rms of delta = trained minus exact bad-bond log-ratio;
  - corr(delta, exact log-ratio);
  - rms by bond type (NN / NNN);
  - correlation of delta between bad-bond pairs of the same configuration, vs bond distance;
  - corr(delta_k, delta_{k+1}) between successive rounds.
- **Noise test.** E_FN of the phi* field plus independent antisymmetric noise of equal weighted rms (per orbit pair,
  hashed), against E_FN of the trained field. The wall is built directly; there is no global amplitude.

## Expulsion / veto test (pre-registered 2026-10-19, consultant Fable + PI caveat; before any computation; code `cx_veto.py`; exact 6x6, sign s_1, no training, <= 1 slice-h)
**Input.** phi* = the self-consistent field after the round-2 ceiling run (up to 60 solves, `bf2c` la_star). The
ceiling is also Aitken-extrapolated from its last three solves.
- X_tau = {x : phi*_x / |psi_P(x)| < tau} for tau = 1e-1, 1e-2, 1e-3 (both amplitudes normalised over the sector).

**(2) Veto field.** E_FN(s_1, r_hyb), with ViT ratios on the support and a = 0 on X_tau (implemented as
la_P - 40 on X_tau).
- Exact decomposition of the ViT-field majoriser gap at phi*, sum_pairs |h| (sqrt(r_P) phi*_x - phi*_y/sqrt(r_P))^2,
  into three bond classes:
  - bulk: both ends with phi*/a_P >= 0.3;
  - near-node in the support: one end in [tau, 0.3);
  - into-expelled: one end in X_tau.
- Readings: E_FN <= 1e-5 means next train the veto as a zero-hop sign net; > 3e-5 means a bulk-ratio (amplitude)
  problem; in between, report.

**Expulsion as a wrong-sign detector (PI caveat: expelled = sign inconsistent with its neighbours, not necessarily
wrong).** For each tau: precision and recall of X_tau against {s_1 != s0}, by orbit-weighted count, by |psi0|^2 weight
and by |psi_P|^2 weight. phi* weight is listed too, though it is ~0 on X by construction.

**(3) Flip test.** s_2 = s_1 flipped on:
- (i) all of X_tau;
- (ii) X_tau intersected with the Krylov flip set (rule K with ViT amplitudes: s_1 sgn(1 + alpha(E_L - E)), alpha
  energy-optimal at fixed amplitude);
- (iii) oracle: X_tau intersected with {s_1 != s0}.
For each tau and variant: E_FN(s_2, ViT ratios). For tau = 1e-2 also its own c-optimum (6 solves + Aitken).
- If (i) or (ii) gives an E_FN below E_FN(s_1, ViT ratios) = 9.54e-5, the expelled set acts as the next sign step at
  fixed cost. The best variant is then iterated 2 more times (X recomputed from the new phi*, with 6 solves each),
  budget permitting.

#### Amendment V-1 (2026-10-19 15:00, after the math review, before any veto/flip computation)
- **Change of input.** The round-2 ceiling job ran on a slow 16 GB slice. At 25 solves (3.65e-6, ratio of successive
  decrements ~0.9) it would have hit its time limit before saving phi*, so it was cancelled. The veto job now
  computes the ceiling itself on a full A40, from phi_FN(psi_P, s_1):
  - (1a) the plain self-consistent iteration, 40 solves;
  - (1b) L-BFGS (m = 10) on the global log-amplitude a_hat, with the exact Hellmann-Feynman gradient
    dE/da_hat(z) = n_z [ -phi_z^2 V(z) + e^{a_hat(z)} sum_{x bad nbr} |h| phi_x^2 e^{-a_hat(x)} ], 30 evaluations
    from the same start.
  - Logged per solve: E_FN, the decomposition E_FN = f(phi) + D(phi; r) (D = sum_bad |h| (sqrt(r) phi_x -
    phi_y/sqrt(r))^2, certified descent), and min / quantiles of phi_FN/|psi0|.
  - Prediction from the math review: convergence is sublinear (face structure), not geometric.
  - phi* = the lower of the two endpoints. Also recorded: W_B = sum_bad |h| phi*_x phi*_y per site.
- **Flip arm (iv)** (math review): flip x in Z_tau with KKT multiplier mu_x = s_x (H s phi*)_x > 0. Flipping Z
  alone is energy-neutral at phi*; the next FN solve decides.
- **Pre-registered predictions (math review):**
  - Z_1e-2 covers >= 80% of the |psi0|^2 weight of W = {s_1 != s0};
  - after flipping (arm iv), E_FN falls monotonically toward exact within a few rounds;
  - falsifiers: a finite-field geometric limit of the ceiling, or Z disjoint from W.
- **Iteration.** If arm (iv) or (i) lowers E_FN below the s_1 ceiling, 2 more rounds of the best non-oracle arm (10
  fixed-point solves each, then Z, mu, flip).
- **Budget:** <= 1.5 slice-h, up from 1. The extra covers the ceiling that the cancelled round-2 job did not deliver.

#### Amendment V-2 (2026-10-19 15:45, during the ceiling phase of the veto job, before any flip computation)
- **Problem.** The only free GPUs are 16 GB slices, at ~1.3 min per FN solve. One job cannot finish ceiling +
  flips + iterations within its time limit.
- **Change.** Split into two jobs:
  - job `veto16` (17356522, running, unchanged settings) computes the ceiling (40 fixed-point solves + 30 L-BFGS
    evaluations) and saves phi*. It may time out in its flip phase; those partial flip numbers are not used.
  - A second job reruns the flip stage from the saved phi*, with the same flip/iterate settings.
- Nothing pre-registered changes.

#### Amendment V-3 (2026-10-19 16:20, after the ceiling phase, before the L-BFGS re-run and any flip result)
- **Ceiling job result.**
  - 40 fixed-point solves: 2.509e-6, Aitken 2.05e-6. min log(phi/psi0) stays near -30, i.e. expelled
    configurations.
  - L-BFGS stopped after ONE evaluation: scipy's default projected-gradient tolerance (1e-5) exceeds the gradient
    norm in total-energy units. The L-BFGS arm was therefore not executed.
- **Fix.** gtol = ftol = 1e-30. L-BFGS is re-run as its own job: 30 evaluations from phi_FN(psi_P, s_1), ceiling only.
- phi* for the flip stage stays the 40-solve fixed point, as already queued in job vetoF.

#### Amendment V-4 (2026-10-19 16:30, after the first three flip lines of job vetoF, before any detector statistic was read)
- **Bug.** The exact-sign table s0 differs from s_1 by a global sign: the flip job printed wrong-sign weight 1.000.
  In addition, s0 = +1 was assigned where psi0 vanishes exactly by symmetry.
- Consequence: precision/recall and the oracle arm (iii) would be meaningless.
- **Fix.** Align s0 by the sign of sum |psi0|^2 s0 s_1, and exclude configurations with psi0 = 0 from {s_1 != s0}.
- The flip job was cancelled and is rerun with the fix. Arms (i), (ii) and (iv) and the veto E_FN do not depend on
  s0.

#### Amendment V-5 (2026-10-19 21:45, after the flip results, before this run)
- **Result of the flip stage.** At 6 solves the flipped signs s_2 reach 1.65e-5 (arms i / iii / iv) and 1.80e-5
  (arm ii), against s_1's 1.81e-5 after 6 solves. The pre-registered iterate condition compared them with s_1's
  40-solve ceiling, so it did not fire.
- That comparison is unequal, so it is repeated at an equal solve count:
  - s_2a = s_1 flipped on all of X_0.1;
  - s_2b = s_1 flipped on X_0.1 with mu > 0 (KKT);
  - each with 40 fixed-point solves from ViT ratios, against s_1's 2.51e-6 at 40 solves.
- Then one more round from the better of the two: s_3 = KKT flip at tau = 0.1 from that phi*, 30 solves.
- Reading:
  - s_2 below 2.51e-6 at 40 solves: the expelled set acts as the next sign step and the loop closes at fixed cost;
  - otherwise not.
- Code `cx_s2ceil.py`, <= 1.2 slice-h.

## Amplitude refresh after the flip (pre-registered 2026-10-21, lead approved; before any run; code `cx_refresh.py` + `cx_reanchor.py` options target_la / edges='all'; <= 3 slice-h)
**Hypothesis.** The spin-only ViT re-anchor failed on rare configurations that were mostly wrong-signed or newly
flipped. After the flip (s_2 = s_1 flipped on X_0.1) the sign handles those, so the ViT may now be precise enough on
the support.

**Steps:**
- (A) phi*(s_2): 40 self-consistent solves from ViT ratios (cx_s2ceil gave 9.15e-7), saved.
- (B) Fine-tune the warm symmetrised ViT to log phi*(s_2):
  - cx_reanchor machinery: minSR with value rows, trust region, fixed validation acceptance, held-out acceptance,
    15 min;
  - edge loss over ALL valid bonds (kept and bad) in the wall metric (|h| phi_x phi_y);
  - X_0.1 is held at its flipped sign (it is part of s_2).
- (C) E_FN(s_2, ratios of the fitted ViT): one solve, and after 3 own solves. Also:
  - an oracle-veto variant: a = 0 where phi*(s_2)/|fitted ViT| < 0.1; this uses phi*, so it is an upper bound;
  - the majoriser-gap decomposition at phi*(s_2): bulk bad bonds / near-node (ratio in [0.1, 0.3)) / bonds into X,
    for the fitted field and for plain ViT ratios;
  - the cost of one refresh (fit seconds, steps).
- References:
  - E_FN(s_2, ViT ratios), recomputed here, against the tau = 0.1 flip value 1.0186e-4;
  - s_1 + ViT ratios 9.54e-5;
  - the ceiling 9.15e-7.

**Pass:** one-solve E_FN(s_2, fitted ViT) <= 1.0e-5, i.e. the fitted ViT is a usable zero-hop guide.

#### Amendment AR-1 (2026-10-21, audit request; written while job 17409419 was in stage B, before any fit or E_FN result was read)
- **Second seed** of the ViT fit (seed 1, same settings).
- **Decade splices** (per seed). E_FN(s_2) with one solve, for a field taking the fitted ViT ratios where phi*(s_2)
  has probability decade >= t and the exact phi*(s_2) ratios below, for t = -8, -9, -10, -11. This shows which decades
  carry the remaining E_FN gap.
- **Flip labels from the fitted network** (per seed), with phi_f = the FN solution of the fitted-ViT field (one
  solve, what a GFMC run with that guide would give):
  - expelled set X_f = {phi_f / |fitted ViT| < 0.1} and KKT set {x in X_f : mu_x(phi_f) > 0};
  - compared with the exact labels from phi*(s_2) and |psi_P|;
  - for both: precision and recall against W_2 = {s_2 != s0} (aligned, psi0 zeros excluded) by |psi0|^2 weight,
    plus the overlap between the fitted and exact label sets.
- Code `cx_refresh2.py`, a follow-up job after 17409419. Total budget ~3.3 slice-h, 0.3 over.
- The H_FN-SR test is not started unless the refit reaches ~3e-5 or better.

## Test T0: determinant/Pfaffian bases as the bad-bond field (pre-registered 2026-10-21, consultant Fable; before any run; code `cx_t0.py`; exact 6x6, no training, <= 1 slice-h)
**Input.** phi*(s_2) from the amplitude-refresh run (refresh_17409419). The sign is always s_2; the bases give
amplitude ratios only.
Bases:
- ViT (psi_P);
- PP alone (rbmpp_6x6 `sym_tables_PP.npz`, dE 6.3e-5);
- converged RBM+PP (`sym_tables_A.npz`, dE 4.7e-5);
- for reference, the refreshed ViT (seed 0, `refresh_17409419/vit`).

Measured:
- **(a)** E_FN(s_2, r = base ratios), one solve, with and without an (oracle) veto a = 0 on
  X_0.1(base) = {phi*(s_2)/|base| < 0.1}.
- **(b)** On bad bonds of s_2, weighted |h| phi*_x phi*_y and binned by the probability decade of phi*(s_2) at x:
  - rms of delta = Delta log base - Delta log phi*, total and NN / NNN;
  - the pointwise error e = log base - log phi*, offset aligned in the phi* metric. Its fitted eps (model e ~ eps/a)
    is eps_d = rms_d(e) x a_d, with a_d = 10^{d/2} the decade's amplitude. eps is reported per decade and as the
    median over decades 1e-9 .. 1e-13.
- **(c)** a(x) = |PP(x)| |1 + alpha (E_L(x) - E)|, with E_L the local energy of (|PP|, s_2) and one scalar alpha
  minimising E_FN(s_2, a) (bounded scalar search, <= 12 solves). Also with the veto.

**Decision (consultant's rule):**
- eps(PP) within 2x of eps(ViT) means the determinant bias is dead;
- eps(PP) down >= 5x, or (c) < 5e-5, means propose T1.

#### Results: amplitude refresh (refresh_17409419, refresh2_17409866; read by the lead 2026-10-21) - FAIL
- One-solve E_FN(s_2, fitted ViT): 9.16e-5 (seed 0), 9.22e-5 (seed 1); bar 1.0e-5. Plain ViT ratios 1.019e-4; oracle
  veto 8.18e-5; ceiling 9.15e-7. CORRECTION (Fable audit, verified by the lead from fit/hist): the fit had NOT converged. Only 28/200 (seed 1: 35/200) steps were accepted, and the trust region sat at its cap from step 24-30 onward. Late accepted steps moved by 1e-5, against 3e-3 early. The flat held-out curve is a stall, not a minimum. The refit numbers are a lower bound on what a ViT can do.
- Decade splices (seed 0 / seed 1): fitted ViT down to decade t, exact phi*(s_2) below that:
  - t = -8: 2.0 / 2.2e-6;
  - t = -9: 5.8 / 5.9e-6;
  - t = -10: 1.68 / 1.68e-5;
  - t = -11: 3.49 / 3.51e-5.
  The ViT is good enough for a^2 >= 1e-9. Decades -10 and -11 cost 2.9e-5, and the configurations below 1e-11 cost
  5.7e-5, about 60% of the gap.
- Flip labels from the fitted network are useless: recall 5e-7 by weight for both X and KKT. The exact labels on s_2
  recall 0.13, with KKT precision 0.998. The KKT step needs the oracle phi*: the circularity found in the audit is
  confirmed.
- The H_FN-SR test is not started; its gate was 3e-5.

#### Results: T0 (t0_17410220; read by the lead 2026-10-21) - PRE-REGISTERED KILL RULE FIRES: the determinant (fixed pairing) tail bias is dead. Labelled hypothesis (consultant's; lead ~20%): local backflow moves the node; its kill test is T1 arm (ii) > 3e-5 with decade error below 1e-11 > 0.1, cost <= 4 A40-h
| base | dE_VMC | E_FN(s_2) | + veto | eps median, decades 1e-9..1e-13 | bond-error rms, decades -9 / -10 / -11 / -12 |
|---|---|---|---|---|---|
| ViT | 1.32e-4 | 1.02e-4 | 9.67e-5 | 5.7e-7 | 0.085 / 0.155 / 0.31 / 0.58 |
| ViT refreshed | - | 9.16e-5 | 8.18e-5 | 3.4e-7 | 0.056 / 0.10 / 0.19 / 0.33 |
| PP | 6.3e-5 | 5.37e-5 | 5.26e-5 | 4.5e-7 | 0.089 / 0.153 / 0.27 / 0.47 |
| RBM+PP | 4.74e-5 | 3.76e-5 | 3.67e-5 | 3.0e-7 | 0.065 / 0.11 / 0.20 / 0.38 |
| (c) PP x \|1 + alpha(E_L - E)\|, alpha -0.108 | - | 2.37e-5 | 2.13e-5 | - | - |
- eps(PP) / eps(ViT) = 0.78, inside the factor 2: by the pre-registered rule the determinant bias is dead. The backflow variant was not tested by T0; it is a hypothesis (above), not a result.
- (c) < 5e-5 meets the second clause, but (c) uses E_L, which is a one-hop quantity. Its gain comes from the hop and
  not from the determinant. The equal-hop reference psi_P + 1 Lanczos (E_FN 1.89e-5, ViT sign) is better, so (c)
  does not rescue the determinant.
- With any zero-hop guide, E_FN(s_2) / dE_VMC is 0.77 (ViT), 0.85 (PP) and 0.79 (RBM+PP). The improved sign
  (ceiling 9.15e-7) buys only about 20% over the guide's own VMC energy, because the guide's ratios set E_FN.

#### Test T0b: tail precision at equal global accuracy (pre-registered 2026-10-21, before any run; cx_t0 bases only, no training, <= 0.5 slice-h)
- Question: is eps an architecture property, or does it track the guide's global accuracy?
- Bases: the four T0 bases plus the early RBM+PP checkpoints B2 (dE 1.54e-4, close to the ViT's 1.32e-4) and B1
  (dE 3.87e-4), from rbmpp_6x6 rpTabB2_17166873 and rpTabB1_17166872. Measured as in T0 (a) and (b).
- Decision:
  - eps(B2) <= eps(ViT) / 2 means the pairing family has a tail bias at equal accuracy. Backflow T1 stays a live
    option for the PI.
  - eps(B2) within 1.5x of eps(ViT), with all six bases on one monotone eps(dE) curve (no base off it by more than
    1.5x), means tail precision tracks global accuracy. A better tail then means a better VMC state, so T1/T2 cannot
    show a loop edge. Recommend route C (ViT/RBM+PP + 1-2 Lanczos + calibrated FN at 8x8-14x14).
  - Anything between these two is inconclusive and is reported as such.
#### Amendment T0-phys (2026-10-21, written after reading the T0 numbers, before any of the computations below; code `cx_t0phys.py`; <= 0.5 slice-h; separate from T0b)
**Why (1).** The lead's T0 verdict stands (kill rule fires). The two branches of the T0 rule disagree, and the lead resolved it with the equal-hop reference psi_P + 1 Lanczos. Part (1) makes that resolution an identical-construction comparison:
- eps(PP)/eps(ViT) = 0.78 falls within 2x, so the "determinant bias is dead" branch fires;
- (c) = 2.37e-5 < 5e-5, so the "propose T1" branch fires.

The (c) construction |PP| |1 + alpha (E_L - E)| is one Lanczos-like hop. The fair control (rule 5) applies the same
hop to the ViT. The earlier table gives E_FN 1.95e-5 for the exact k = 1 clipped target of psi_P, but that uses the
s_1 sign and the clipped form, so it is not an identical construction.

**(1) Same-construction control.** Run (c) exactly as in T0 (same E_L definition with sign s_2, same bounded search
over +-3/sdE, 12 evaluations, same veto and statistics) for base = ViT and base = RBM+PP.

Reading, fixed now:
- ViT one-hop <= 1.2 x PP one-hop (<= 2.84e-5): the (c) trigger comes from the hop, not the determinant. The rule
  reads "determinant bias dead" and T0 gives no evidence for T1.
- ViT one-hop >= 2 x PP one-hop (>= 4.73e-5): the determinant basis earns the gain, and T1 is proposed.
- In between: inconclusive, reported to the lead with the numbers.

**(2) Physics add-on (consultant prediction: a stripe / domain-wall enhancement near the nodes).**

Sets:
- X_0.1, the expelled set used for s_2. It is recomputed from phi*(s_1) (veto16 la_star) and psi_P, and checked
  against s2.npy.
- The remaining wrong-sign set W_2 = {s_2 != s0}.
- The psi0 probability decades d = -5 .. -15, each on its own.
- Bulk = decades >= -8. Tail = decades -10 .. -12.

Measured per set, as a |psi0|^2-weighted mean (and an unweighted orbit-count mean):
- (i) The fraction of valid J1 bonds out of x whose exact sign product s0_x s0_y = +1. This is Marshall-violating:
  Marshall predicts -1 on every J1 bond.
- (ii) The weight share of Marshall-wrong configurations, s0_x != Marshall(x) after a global alignment.
- (iii) The configuration moments
  - M_Neel^2 = (sum_i sigma_i (-1)^(x+y) / N)^2;
  - M_stripe^2 = (sum_i sigma_i (-1)^x / N)^2 + (sum_i sigma_i (-1)^y / N)^2.
- (iv) The 2x2-plaquette shares: Neel (checkerboard) vs stripe (two parallel columns or rows, antiparallel to each
  other).
- (v) The density of ferromagnetic J1 bonds (domain walls).

Prediction counts as confirmed if the stripe plaquette share or M_stripe^2 / M_Neel^2 on X_0.1 and in the tail is
>= 1.5x the bulk value. It is refuted if both are <= 1.1x.

## Test G0: gamma-FN on exact tables, so the reported energy does not depend on the tail (pre-registered 2026-10-21, lead; before any run; code `cx_g0.py`; exact 6x6, no training, <= 0.5 slice-h)
**Idea.** This uses Sorella-Capriotti PRB 61, 2599 (2000), after ten Haaf et al. PRB 51, 13039 (1995). Work in the
amplitude basis u (psi = s u). The gamma-FN operator keeps the kept bonds of H_FN, puts back the sign-violating ("bad")
off-diagonals as -gamma H_xy, and weights the wall term by (1 + gamma):
  H^gamma = D0 + (1 + gamma) V_a - W_kept - gamma B_bad,
where V_a(x) = sum_bad H_xy a_y / a_x is the wall term and B_bad is the bad-bond hopping.
- gamma = 0 is our H_FN; gamma = -1 is H itself.
- In cx_lib terms: Dg_gamma = D0 + (1 + gamma)(Dg - D0), B x = 0.5(h1 + s h2) - D0 x, and
  A_gamma x = Dg_gamma x - 0.5(h1 - s h2) - gamma B x.
- Hellmann-Feynman: <psi^gamma|H|psi^gamma> = E^gamma - (1 + gamma) dE^gamma/dgamma, the tangent extrapolation to
  gamma = -1. It is a variational upper bound, and walkers estimate it from two FN runs.

**Measured.** Sign s_2 throughout (refresh_17409419). Ratios a from the ViT, PP and RBM+PP (as in T0), at
gamma = 0, 0.25, 0.5, 1, 2:
- E^gamma (the Perron ground energy of H^gamma);
- <H> on phi^gamma, exact, with sign s_2;
- the chord extrapolation of E^gamma to gamma = -1, an estimate and not a bound.
All are per site, relative to E0.

**Controls (each must be able to fail):**
- (i) gamma = 0 reproduces T0: E_FN of the ViT = 1.0186e-4.
- (ii) Exact a = |psi0| with the exact sign s0 gives E^gamma = E0 to 1e-9 for every gamma.
- (iii) A broken version, with the sign of the gamma B term flipped, must fail control (ii); report both runs.

**Decision (primary = min over gamma of <H> on phi^gamma, ViT ratios):**
- PASS if <= 1e-5 or within 3x of the ceiling 9.15e-7. The reported energy then does not need a tail network: T1 is
  moot for the energy, and the open problem is the sign labels.
- FAIL if it removes < 50% of the gap from 1.0186e-4 to the ceiling.
- In between: partial; report the fraction removed, per base.
- The chord estimate is reported but never decides.

#### Results: T0b (t0b_17410565; read by the lead 2026-10-21) - the tail eps tracks global accuracy; no pairing advantage
| base | dE_VMC | E_FN(s_2) | eps (median, decades 1e-9..1e-13) |
|---|---|---|---|
| RBM+PP | 4.74e-5 | 3.76e-5 | 3.02e-7 |
| PP | 6.3e-5 | 5.37e-5 | 4.47e-7 |
| ViT | 1.32e-4 | 1.02e-4 | 5.72e-7 |
| B2 (RBM+PP, early) | 1.54e-4 | 1.46e-4 | 8.62e-7 |
| B1 (RBM+PP, early) | 3.87e-4 | 3.69e-4 | 1.37e-6 |
- The vit / pp / rbmpp rows reproduce t0_17410220 to 3 digits; vit is bit-identical.
- eps rises monotonically with dE_VMC: eps ~ dE^0.72 from RBM+PP to B1. E_FN(s_2) / dE_VMC is 0.77-0.95 for all
  five bases.
- Direct ratio eps(B2) / eps(ViT) = 1.507, 0.007 above the pre-registered 1.5x cap. B2 is 16% less accurate than the
  ViT. Interpolated to equal dE (log-log, between RBM+PP and B2), the pairing family's eps is 7.5e-7: the ViT is 0.76x
  of it and PP is 1.15x, both inside 1.5x. Read at equal accuracy, which is what the rule asks, this is the second
  branch: tail precision tracks global accuracy, and at equal accuracy the pairing family is no better than the ViT
  on the tail (if anything 1.3x worse).
- Consequence: a better tail means a better variational state, so the T0 kill stands. Extrapolating the measured slope
  (derived, unvalidated, past measurement): eps ~ 1e-8 would need dE_VMC ~1e-6.

#### Results: T0-phys (t0phys_17410698, runs/cx_t0phys.json; 15 min, one A40 slice)
**Checks.** s_2 equals s_1 flipped on the recomputed X_0.1 (0 mismatches). X_0.1 holds 9.3% of the orbits and 6.0e-6
of the |psi0|^2 weight.

**(1) The one-hop construction |base| |1 + alpha (E_L - E)| with sign s_2 (identical search for every base):**

| base | zero-hop E_FN | one-hop E_FN (alpha) | + veto | one-hop / own dE_VMC |
|---|---|---|---|---|
| ViT | 1.02e-4 | 2.90e-5 (-0.099) | 2.61e-5 | 0.22 |
| PP | 5.37e-5 | 2.37e-5 (-0.108) | 2.13e-5 | 0.38 |
| RBM+PP | 3.76e-5 | 1.29e-5 (-0.122) | 1.08e-5 | 0.27 |

- ViT / PP after one hop = 1.23. That is just above the 1.2 bar and below 2, so the pre-registered reading is
  **inconclusive**.
- The lean is toward "the hop, not the determinant":
  - PP's advantage shrinks from 1.9x with no hop to 1.23x after one hop;
  - relative to its own VMC error, the ViT gains the most from the hop.
- This leaves the T0 kill and the T0b verdict (eps tracks global accuracy) unchanged.

**(2) Physics of the near-node sets.** All values are |psi0|^2-weighted means. Random Sz = 0 reference values:
- M_Neel^2 about 0.029, M_stripe^2 about 0.057;
- Neel plaquettes 0.125, stripe plaquettes 0.25;
- ferromagnetic J1 bonds about 0.49.

| set (psi0 weight) | J1 Marshall-violating (s0) | Marshall-wrong config | M_Neel^2 | M_stripe^2 | ratio S/N | plaq Neel / stripe | ferro J1 |
|---|---|---|---|---|---|---|---|
| bulk, decades >= -8 (0.76) | 0.013 | 0.003 | 0.153 | 0.041 | 0.27 | 0.46 / 0.28 | 0.27 |
| tail, decades -10..-12 (0.077) | 0.111 | 0.156 | 0.049 | 0.061 | 1.25 | 0.25 / 0.31 | 0.40 |
| X_0.1 (6.0e-6) | 0.400 | 0.239 | 0.053 | 0.081 | 1.52 | 0.24 / 0.29 | 0.39 |
| W_2 = {s_2 != s0} (1.2e-6) | 0.463 | 0.923 | 0.053 | 0.069 | 1.29 | 0.33 / 0.33 | 0.34 |

- **Prediction confirmed by the pre-registered rule.** M_stripe^2 / M_Neel^2 is 5.7x the bulk value on X_0.1 and 4.7x
  in the tail (bar 1.5x).
- The plaquette criterion alone would not confirm it: the stripe share is 1.04x / 1.09x the bulk value.
- Mechanism, stated precisely: Neel order collapses toward its random value, M_Neel^2 down 2.9x. The stripe moment
  rises 2.0x over the bulk and sits 1.4x above random. Ferromagnetic J1 bonds (domain walls) rise 1.4x. These are
  stripe-tilted, Neel-disordered configurations, not clean stripe domains.
- **Marshall violation is the strongest signal.**
  - 40% of the J1 bonds out of X_0.1 have the non-Marshall sign product, against 1.3% in the bulk (30x). That is 3.6x
    the tail average at comparable weight. The fraction rises smoothly with decade: 0.054 / 0.097 / 0.15 / 0.19 / 0.24
    at decades -9..-13.
  - On W_2, 92% of the configurations have a non-Marshall exact sign. The remaining sign errors of s_2 sit where the
    true sign departs from Marshall and the guide sign has stayed Marshall-like.
#### Fable training audit (results/consult/fable_training_audit_2026-10-21.md) - read by the lead 2026-10-21
- The T0/T0b evaluation is sound. The conclusion "a precise tail = a better variational state; the Pfaffian route is
  dead" does NOT follow.
- 80-91% of each network's L2 error sits at a^2 < 1e-10 (Fable, measured). Its eps = (1.2-1.8) x sqrt(L2/N_conf) for
  all six bases. Every network was trained on a |psi|^2-weighted objective, so "eps tracks dE" is that objective's
  signature, not a representability limit.
- The T0 kill rule could only give "within 2x" at equal L2. It kills "a fixed pairing is automatically right"; it
  says nothing about a node-moving ansatz trained on the tail.
- Verdict status: T0 kill = fixed VMC-trained PP only. The Pfaffian+backflow route is INCONCLUSIVE (cause: no network
  has been trained with weight on the tail). Next: the decade-stratified fit (T1s, pre-registration below).

## Test T1s: tail-weighted (decade-stratified) fit to phi*(s_2) (pre-registered 2026-10-21, lead, from the Fable training audit; before any run; <= 4 slice-h)
**Question.** Is the tail error a representability limit or the training objective? The question is asked per arm.

**Target and sampling:**
- Target: log phi*(s_2) (refresh_17409419/la_star_s2.npy). The sign is s_2 and is fixed.
- Sampling: decade-stratified. Equal counts per |phi*|^2 decade from -5 to -13, uniform within a decade, at least 2048
  samples per step.
- Loss: per-configuration (log a - log phi*)^2 plus a bond term on bad bonds of s_2, each renormalised per decade.

**Arms:**
- (a) ViT, warm psi_P;
- (b) PP f-only, warm from sym_tables_PP;
- (c) PP + local backflow with a 3x3 receptive field, zero-initialised.

**Optimiser and convergence:**
- minSR or SR with a trust region.
- Converged means: acceptance >= 50%, trust region below its cap over the last 20% of steps, and a flat stratified
  held-out loss. Run until converged or out of budget, and report which.

**Readings, per arm:**
- Objective was the cause: rms log error at decade -11 <= 0.03 with the bulk (decades >= -8) <= 0.01.
- Representability limit: a verifiably converged fit still has >= 0.1 at decade -11.
- Otherwise inconclusive.

**Physics pass:** E_FN(s_2, fitted ratios), one solve, <= 1e-5.

**Controls:**
- The broken-version check: the arm-(a) loss evaluated with the decade weights switched off reproduces the refresh
  fit's tail error profile.
- The unchanged warm-start psi_P reproduces the T0 eps 5.72e-7.

#### Results: G0 (g0_17411369; read and verified by the lead 2026-10-21) - FAIL
**Controls:** all pass.
- (i) gamma = 0 reproduces 1.0185827e-4.
- (ii) Eigenvector residual with the exact psi0: 1.3e-15 x |E0| at gamma 0, 0.5, 2.
- (iii) The broken copy fails, with residual 0.22 / 0.87 x |E0|.
- Every solve reached tolerance (res <= 9.7e-11), and the Perron vector had no negative weight.

**<H> on phi^gamma, in units of 1e-5, at gamma = 0 / 0.25 / 0.5 / 1 / 2:**
- ViT: 7.57 / 8.27 / 8.84 / 9.71 / 10.86.
- PP: 4.23 / 4.56 / 4.81 / 5.19 / 5.67.
- RBM+PP: 2.93 / 3.17 / 3.35 / 3.63 / 3.98.

**Reading:**
- The minimum is at gamma = 0 for every base, and <H> rises monotonically with gamma. Chord estimates at gamma = -1:
  8.18 / 4.52 / 3.14e-5.
- The primary measure (ViT, 7.57e-5) removes 26% of the gap to the ceiling 9.15e-7. That is below 50%: FAIL.
- The gamma extrapolation does not free the reported energy from the tail ratios. The only gain comes from evaluating
  <H> on the FN state at gamma = 0, which is 0.57-0.67 x dE_VMC, against 0.8 for E_FN.

#### Amendment T1s-1 (2026-10-21, lead; written before submission, after reading cx_t1s.py and launch_t1s.sh)
- "Converged" in the T1s pre-registration had two gaps:
  - (i) a frozen optimiser passed it;
  - (ii) the thresholds were not numbers.
- The added rules:
  - **Converged** means all of the following over the last 20% of steps (at least 50 steps):
    - acceptance >= 0.5;
    - lam below its cap at every step;
    - the held-out loss changing < 2%;
    - the median predicted rms change of log A per accepted step >= 1e-3, so the parameters are still moving.
  - A fit that is flat but frozen is reported as "stalled", never as converged, and cannot give the
    "representability limit" reading.
- Control 1 is a training run, decided before any run: draws from |phi*|^2, no per-decade renormalisation, with the
  refresh fit's 900 s budget.
- Note on PP's parameter count: the trained PP f is a general 36x36 matrix with 1260 active entries, not 630.

#### Results: T1s (runs t1s_{vit_17412547, vitns_17412548, pp_17412549, ppbf_17412550}; verified by the lead from cx_t1s.json 2026-10-21)
Rms log error per |phi*|^2 decade, -5 .. -13, count-weighted. The sha guard and control 2 pass in every arm.

| arm | -5 | -8 | -9 | -10 | -11 | -12 | -13 | converged | E_FN(s_2) |
|---|---|---|---|---|---|---|---|---|---|
| (a) ViT, stratified | 0.023 | 0.021 | 0.028 | 0.056 | 0.124 | 0.213 | 0.305 | yes (acc 0.98, move 4.4e-2, held-out 0.1%) | 8.18e-5 |
| control 1, ViT, unstratified | 0.007 | 0.011 | 0.021 | 0.054 | 0.129 | 0.247 | 0.399 | no (budget) | 8.93e-5 |
| (b) PP f-only | 0.005 | 0.016 | 0.023 | 0.038 | 0.073 | 0.153 | 0.317 | yes (acc 1.00, move 5.0e-2, held-out 1.7%) | 2.82e-5 |
| (c) PP + 3x3 backflow | 0.010 | 0.015 | 0.022 | 0.038 | 0.074 | 0.163 | 0.333 | no (budget; held-out still falling 20%) | 2.96e-5 |

Pre-registered readings:
- (a) ViT: converged with 0.124 >= 0.1 at decade -11, so the reading is "representability limit". Stratifying moved its
  tail by only 4% (0.129 -> 0.124) and cost bulk accuracy (0.007 -> 0.023).
- (b) PP: inconclusive (0.073 lies between 0.03 and 0.1).
- (c) PP + 3x3 backflow: inconclusive, because it had not converged.
- Physics pass (E_FN <= 1e-5): no arm passes.

Lead's reading:
- Trained on the tail, the pairing structure does what T0 could not show. With 1260 parameters against the ViT's
  155k, PP reaches a 1.7x lower error at decade -11 and 1.5x lower at decade -10. Its E_FN falls from 5.37e-5 (VMC
  table) to 2.82e-5, against the ViT's 8.18e-5. The T0 kill is therefore overturned for the PP family.
- Caveat on "converged" for (a) and (b): 78/90 and 119/120 accepted steps were "within noise", and the parameters
  still moved by about 5% in log A per step. The plateau may be the noise floor of 228 samples per decade per step,
  not a minimum. The ViT "representability limit" reading carries this caveat until a run with more samples per step
  settles it.

## Test T1s-2: is the T1s plateau a sampling noise floor? (pre-registered 2026-10-21, lead; before any run; cx_t1s.py, spec changes only; ~6 slice-h)
**Setup:**
- Same code, target and readout as T1s.
- Changes per step:
  - B = 8192 samples per step (912 per decade), and B_acc = 8192;
  - the trust radius tau is annealed from 0.05 to 0.01, halved each time the held-out loss is flat (< 2%) over 20
    steps.
- Warm starts: arms (b) PP, (c) PP+backflow and (a) ViT, each from its own T1s endpoint (theta.npy).
- Fit budgets: 3600 s for PP and ViT, 5400 s for PP+backflow.
- Convergence: the T1s-1 rule, plus "tau has reached 0.01".

**Readings, per arm (r = decade -11 rms):**
- Noise floor: r falls by >= 1.5x from its T1s value. The plateau was sampling, and the arm continues toward 1e-5.
- Converged limit: r falls by < 1.2x and the arm converged. For the ViT this confirms the representability limit;
  for PP / PP+backflow, a limit at that r.
- Between these: inconclusive.

**Pass (the T1 bar, unchanged):** E_FN(s_2) <= 1e-5. Also reported: E_FN with the fitted ratios on bulk decades
(>= -9) and exact phi* below (a splice), to show which decades still carry the gap.

#### Amendment T1s2-1 (2026-10-21, lead; before submission, after reading launch_t1s2.sh and the specs)
At the measured speeds (s/step x 4), the pre-registered budgets give ppbf about 72 steps and the ViT about 54. The
convergence rule needs >= 60 steps of annealing plus a 50-step window, so those two arms would end "out of budget" by
construction: a test that cannot settle.
- New fit budgets: pp 3600 s (unchanged, about 116 steps), ppbf 10800 s (about 144 steps), ViT 9000 s (about 134
  steps).
- The readings and the pass bar are unchanged.

#### Results: T1s-2 (t1s2_pp_17418856, t1s2_ppbf_17418857, t1s2_vit_17418858; verified by the lead 2026-10-22)
| arm | -9 | -10 | -11 | -12 | r(-11): T1s -> T1s-2 | converged | E_FN(s_2) | splice E_FN (fitted >= -9, exact below) |
|---|---|---|---|---|---|---|---|---|
| PP | 0.025 | 0.039 | 0.070 | 0.141 | 0.073 -> 0.070 (1.05x) | no (budget; tau stopped at 0.025) | 2.61e-5 | 7.9e-6 |
| PP + 3x3 backflow | 0.021 | 0.035 | 0.066 | 0.138 | 0.074 -> 0.066 (1.12x) | no (budget; tau stopped at 0.025) | 2.42e-5 | 6.0e-6 |
| ViT | 0.023 | 0.050 | 0.113 | 0.200 | 0.124 -> 0.113 (1.10x) | no (lam at cap on 16 window steps) | 6.89e-5 | 8.2e-6 |

Pre-registered readings:
- The noise-floor reading (>= 1.5x) fails for all three arms: 4x samples and 3x the steps move r by 5-12%.
- The converged-limit reading cannot apply, because no arm converged. Strictly, every arm is inconclusive.
- The pass bar (E_FN <= 1e-5) is not met.

Lead's reading:
- All three splices land at 6-8e-6, so the whole remaining gap sits at |phi*|^2 <= 1e-10 for every network.
- The best zero-hop guide is now PP+backflow at 2.42e-5, against the original ViT's 1.02e-4.
- More sampling or training time on this objective is unlikely to close the remaining 2.4x: r(-11) must fall from
  0.066 to about 0.03.

## Test G1: one hop on the tail-trained amplitudes (pre-registered 2026-10-22, lead; before any run; cx_t0 part-(c) machinery; no training, <= 0.5 slice-h)
**Construction.** For each tail-trained table (t1s2_ppbf_17418857, t1s2_pp_17418856, t1s2_vit_17418858 la_table.npy;
for reference also the T1s ones):
- a(x) = |base(x)| |1 + alpha (E_L(x) - E)|, where E_L is the local energy of (|base|, s_2);
- a bounded scalar search over alpha, exactly as in T0 (c) / T0-phys (1);
- E_FN(s_2, a), one solve, with and without the oracle veto.

**Equal-hop references:**
- psi_P + 1 Lanczos: 1.89e-5;
- ViT one-hop: 2.90e-5 (T0-phys);
- PP one-hop: 2.37e-5;
- RBM+PP one-hop: 1.29e-5 (all VMC tables).

**Readings (best tail-trained Pfaffian arm, no veto):**
- PASS: E_FN <= 1e-5 and below every equal-hop reference.
- Partial: below 1.29e-5 but above 1e-5.
- No edge: >= 1.29e-5, i.e. no better than the VMC-trained RBM+PP with the same hop.

#### Results: G1 (g1_17426232; verified by the lead 2026-10-22) - PASS
The control reproduces T0-phys's ViT one-hop (2.9048e-5, rel 7e-12). The job timed out after 4 of 6 bases; the two
T1s pp/vit bases were references only and do not affect the reading.

| base (tail-trained) | zero-hop E_FN | one-hop E_FN | + veto | alpha |
|---|---|---|---|---|
| PP + 3x3 backflow (T1s-2) | 2.42e-5 | **7.06e-6** | 5.43e-6 | -0.114 |
| PP (T1s-2) | 2.61e-5 | 8.06e-6 | 6.35e-6 | -0.119 |
| PP + 3x3 backflow (T1s) | 2.96e-5 | 7.98e-6 | 6.18e-6 | -0.112 |
| ViT (T1s-2) | 6.89e-5 | 1.52e-5 | 1.42e-5 | -0.085 |

- **PASS.** 7.06e-6 <= 1e-5, and it is below every equal-hop reference: RBM+PP one-hop 1.29e-5, psi_P + 1 Lanczos
  1.89e-5, ViT one-hop 2.90e-5. It is 1.8x below the best of them.
- The tail-trained ViT with the same hop reaches only 1.52e-5. The pairing structure, once trained on the tail, is
  what gets below 1e-5.
- CORRECTIONS (lead, 2026-10-22):
  - The sign s_2 is ALSO oracle-derived (the flip of X_0.1 was computed from phi*(s_1) on exact tables). G1 is a
    representability result: oracle sign + oracle target + alpha tuned on the exact E_FN.
  - One seed throughout.
  - The fair equal-hop comparison is RBM+PP one-hop (same s_2): 1.8x. psi_P + 1 Lanczos used the ViT sign.
- **What this does NOT show** (the T2 questions):
  - (i) the fit target was the exact phi*(s_2) and alpha was chosen on the exact E_FN, both oracle quantities at 6x6;
  - (ii) the cost at 8x8-14x14 with fast Pfaffian updates;
  - (iii) an edge over plain SR VMC of the same architecture at equal GPU-h.

## De-oracling the G1 guide (pre-registered 2026-10-22, lead; before any run). Order O1 -> O2 -> O3 -> O4, one oracle removed at a time; exact solves stand in for walkers until O4.

### O1: alpha without the exact E_FN (no training; <= 0.5 slice-h; code cx_g1 + one option)
- Bases: t1s2_ppbf, t1s2_pp, t1s2_vit (G1). Sign s_2.
- alpha_V = argmin of the exact variational energy <H> of the one-hop state s_2 |a| |1 + alpha(E_L - E)|. This
  quantity is walker-accessible as a VMC estimate.
- Report E_FN(s_2, alpha_V), compared with the G1 optimum (7.06e-6 for ppbf).
- **PASS:** E_FN(ppbf, alpha_V) <= 1e-5.
- **FAIL:** > 1e-5. In that case alpha needs an FN-based rule, e.g. a two-point mixed estimate.

### O2: the loop with a walker-producible target (sign s_2 still oracle; <= 8 A40-h)
**Round k:**
- (a) Guide g_k = network a_k + one hop, with alpha by the O1 rule.
- (b) phi_k = the one-shot FN solution of g_k (one sec.fn_solve), which is what a long GFMC run with g_k samples.
- (c) Fit a_{k+1} to log phi_k: warm start from a_k, decade-stratified, the cx_t1s machinery, B = 8208, at most 40 min.
- (d) E_FN(g_{k+1}).

**Start:** a_0 = the VMC-trained PP table (zero-hop 5.37e-5) with zero backflow. This is a start that needs no
oracle.

**Measured each round:**
- E_FN(g_k);
- the per-round contraction c_k = E_FN(g_{k+1}) / E_FN(g_k);
- the fit error per decade against phi_k.

Reference: the exact iteration contracts by 0.6-0.65 per solve from a fitted ViT (refresh "own" solves:
9.2 -> 4.3 -> 2.5 -> 1.7e-5).

**Decision:**
- **PASS:** E_FN(g_k) <= 1e-5 within 4 rounds.
- **STALL, stop early:** two consecutive rounds with c_k > 0.9 while E_FN > 1.5e-5. The loop then does not converge
  at constant cost per round, and the cause is diagnosed from the per-decade fit error against phi_k.
- **Viability flag:** a projection to 8x8 from the measured c_k and the s/round. If it needs > 15 rounds to reach
  1e-5 from a VMC start, the loop is not viable as designed, even on a PASS, and this is reported as such.

### O3 (after an O2 pass): KKT flip labels from the network's own FN solution instead of phi*; sign s_1 -> s_2' without the oracle. Pre-registered after O2.
### O4: walkers instead of exact solves (8x8 problem; tail-steered sampling). Pre-registered after O3.

#### Amendment O2-1 (2026-10-22, lead; from results/consult/fable_O2_design_review_2026-10-22.md, before any O2 run)
The review's points are accepted; numbers were spot-checked against G1 and T1s-2.
- **Fit tax.** The best fit of the exact phi* plus one hop already sits at 7.06e-6, against the 9.15e-7 ceiling, so
  only about 3e-6 of the 1e-5 bar is left for the loop. The fit-free chain from 2.37e-5 is itself marginal at 4
  rounds (estimated by the review, not measured).
- **Unnamed failure.** Noise-free self-healing DMC (Caffarel et al., arXiv:2609.01301) is O2's steps (b)-(c). Its
  energy RISES monotonically to a wrong node. O2 did not name that signature.

**Changes:**
1. **O2a first: the fit-free reference chain**, phi_{k+1} = FN solution of hop(phi_k), from the same VMC-PP start and
   with the same alpha rule (O1). No training; 4 rounds; about 1.5 A40-h. It measures the rate of the map itself.
2. **O2b: the fitted loop**, as pre-registered, run after O1 passes. Each round it also reports:
   - the fit tax = E_FN(hop(a_{k+1})) - E_FN(hop(phi_k));
   - the splice;
   - the per-decade MEAN fit error, as well as the rms;
   - the expelled set {phi_k / |g_k| < 0.1}, split into s_2-right and s_2-wrong |psi0|^2 weight (oracle diagnostics
     only).
3. **Contraction** c_k is defined on E_FN - 9.15e-7.
4. **New verdict RISE:** c_k > 1 in any round. Stop.
5. **Viability criterion** (added; the absolute 1e-5 bar stays the PASS bar): in each of the first 3 rounds the
   fitted loop keeps >= 50% of the exact chain's gain, i.e. fit tax < 0.5 x the exact gain. If it does not, the
   representation, not the map, limits the loop, and the loop is not viable at constant cost as designed.
6. **Reading a STALL:**
   - O2a's rate > 0.9: the map is too slow;
   - fit tax >= the exact gain: a representation floor;
   - small tax but rising s_2-right expelled weight: the spurious-attractor mechanism of self-healing DMC.
7. **Budget:** about 2.7 A40-h per fitted round (measured T1s-2 speeds), so O2b is about 12 A40-h, sized from rates.
