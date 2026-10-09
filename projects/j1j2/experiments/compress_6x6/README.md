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
