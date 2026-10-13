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
