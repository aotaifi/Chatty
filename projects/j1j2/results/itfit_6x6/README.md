# Imaginary-time fitting with the frozen FN Hamiltonian as the loop's amplitude write-back (exact 6x6)

Pre-registration and amendments: `experiments/itfit_6x6/README.md`. Code: `experiments/itfit_6x6/`. All energies per site,
dE = E - E0, exact symmetric sector of the 6x6 J1-J2 model at J2/J1 = 0.5. Figure: `itfit_6x6.png` / `.pdf`.

## Verdict
**Not PASS by the pre-registered rule; INCONCLUSIVE (cause: representation of a one-hop correction in a zero-hop
network, not stiffness, not sampling).**
- **Stiffness is solved, and it is not the bottleneck.** lambda_max(F) = 1.6e6, so the explicit Euler target needs
  ~5e4-8e4 steps per FN iteration. The **semi-implicit (diagonal-implicit) step** t = (a + tau K a)/(1 + tau (D - E))
  is local (one hop of the network and of the guide), positive, and captures **87% of one FN iteration in one step,
  97% in two**. Unprojected, a loop with two such steps tracks the ideal FN loop within 5%.
- **The network cannot store the step.** Fitting the 28k-parameter residual CNN to these targets with stochastic natural
  gradient (minSR form, 2 x 1500 iterations per loop iteration) keeps **3.7% of one FN iteration with tempered
  sampling (iii), 0.45% with FN-walker samples (ii), 0.7% with |b|^2 samples (i)**, far below the 50% bar; every arm
  stops after loop iteration 1 by the pre-registered 0.10 rule. The exact phi_FN target (ORACLE, same budget) reaches
  8.3%, so realistic ITE targets reach ~45% of what this network + optimiser can hold at this budget. The fit loss is still falling slowly (not converged), at ~1% of the target per
  1000 iterations; the sibling one-shot test with exact phi_FN targets saturates at <= 18% after 20k steps.
- **One exact hop restores it.** The same, essentially unchanged network plus one exact semi-implicit hop
  (one hop of the network, like the composite sign + one hop) recovers 82-87% of the first FN iteration:
  <H> 1.32e-4 -> 8.9e-5, E_FN 1.02e-4 -> 7.10e-5 (ideal 7.84e-5 / 6.52e-5). With no learnable write-back the composite
  loop stalls after that first iteration (E_FN 7.10e-5 -> 7.09e-5 -> 7.07e-5).
- Reason (Stage 0): the FN update is a local-equilibrium correction on wall configurations, an explicit function of
  three one-hop quantities of the guide: kept-edge weight W(x) = sum_kept |H_xy| a(y)/a(x), wall
  V(x) = sum_viol |H_xy| a(y)/a(x), and the Ising diagonal H_xx. A zero-hop CNN residual has to re-derive ratios of
  neighbour amplitudes of the base network, which it cannot (the update is white-noise rough in its input).

## Stage 0: stiffness (exact, `itfit_exact.json`, `itfit_exactloop.json`)
Monitored risk (PI): the FN wall makes F stiff. Measured on F_0 = H_FN[|psi_P|, s_P] (G_0 = 3.01e-5):
- **lambda_max(F) = 1.61e6** (Lanczos, converged at m = 10; equal to max D; H bandwidth ~35), set by a wall on a
  configuration with phi^2 = 3e-25. Largest stable explicit step tau_stab = 2/(lambda_max - E_FN) = **1.24e-6**; the
  positivity limit of the explicit target tau_pos = 1.8e-6 > tau_stab (explicit at tau_pos diverges).
- Wall V(x) = sum_viol |H_xy| a(y)/a(x), by decade (phi^2 weight / share of the ideal gain):
  V < 1: 6.5% / 0.06%; 1-10: 92% / 65%; 10-100: 1.4% / 34%; 100-1e3: 2e-5 / 0.8%; > 1e3: 3e-8 / 0.01%.
  The gain sits on walls 1-100; the stiffness comes from walls ~1e6 that carry nothing.
- Exact unprojected steps, captured fraction of one FN iteration (one step / n90):

| scheme | 1 step | 2 steps | n90 | cost per step |
|---|---|---|---|---|
| explicit, tau = 0.9 tau_stab | 5.0e-5 | 1.0e-4 | ~5e4 (extrap.; ITE time beta90 ~ 0.09 -> ~8e4) | 1 matvec |
| explicit, optimal tau (1-hop Rayleigh-Ritz = Ledinauskas-Anisimovas adaptive dt) | 0.23 | 0.24 | ~9e3 (0.33 at n = 200) | 2 matvecs |
| Krylov 2-hop Rayleigh-Ritz, restarted | 0.24 | 0.26 | ~3e3 (0.36 at n = 80) | 3 matvecs |
| **semi-implicit, tau = 1** (local, positive) | **0.866** | **0.970** | **2** (0.99 at 3) | 1 matvec |
| semi-implicit, tau = 0.03 / 0.1 / 0.3 / 100 | 0.51 / 0.73 / 0.83 / 0.88 | 0.74 / 0.91 / 0.96 / 0.975 | 4 / 2 / 2 / 2 | 1 matvec |
| implicit, tau = 0.3 (global CG solve) | 0.966 | 0.998 | 1 | 22 matvecs (21 CG) |
| implicit, tau = 0.01 | 0.30 | 0.49 | 9 | 6 matvecs (5 CG) |

  No energy-increase event in any semi-implicit or implicit run. **Stiffness decision (pre-registered rule): KEEP**,
  semi-implicit target with K = 2 steps per loop iteration, tau = 1 (tie-break in amendment 1).
- Unprojected loop with K exact semi-implicit steps per iteration (= perfect fits; every iteration adds hops):

| loop | E_FN of guides, it 1 / 2 / 3 / after 3 | <H> (Krylov sign) after it 1 / 2 / 3 | frac per it |
|---|---|---|---|
| ideal FN loop | 1.02e-4 / 6.52e-5 / 4.87e-5 / 3.81e-5 | 7.84e-5 / 5.58e-5 / 4.30e-5 | 1 |
| tau = 1, K = 2 | 1.02e-4 / 6.80e-5 / 5.13e-5 / 4.05e-5 | 8.32e-5 / 6.01e-5 / 4.64e-5 | 0.97 / 0.96 / 0.96 |
| tau = 1, K = 1 | 1.02e-4 / 7.29e-5 / 5.72e-5 / 4.62e-5 | 9.17e-5 / 6.91e-5 / 5.46e-5 | 0.87 / 0.86 / 0.85 |

## Stage 1: trained arms (exact evaluation; `runs/itA1.json`, `runs/itA2.json`, `runs/itC.json`)
Residual b = |psi_P| exp(f), f = 28k-parameter symmetric CNN; semi-implicit targets, tau = 1, K = 2 steps,
2 x 1500 stochastic natural-gradient (minSR form) iterations, eta = 0.3, lam = 1e-3 (amendment 3). frac = captured
fraction of loop iteration 1 (exact frozen-FN energy, G_0 = 3.01e-5). "+F_k hop": one exact semi-implicit step of F_0
on top of the fitted net. "composite": the one-hop guide c = T_{F[b, s']}[b] with the new Krylov sign s'.
Start: <H> 1.319e-4, E_FN 1.018e-4; Krylov sign alone: <H> 1.266e-4, E_FN 0.943e-4.

| arm | data / fit | frac (net) | + F_0 hop | composite guide | <H> net (Krylov) | <H> composite | E_FN net / composite (next guide) | evaluations |
|---|---|---|---|---|---|---|---|---|
| ITE-Q-iii | tempered x ~ b, energy-metric GN | **0.037** | 0.882 | 0.835 | 1.253e-4 | 8.83e-5 | 9.39e-5 / 7.09e-5 | 2.5e9 |
| ITE-Q-ii | FN mixed a phi (walkers), energy metric | 0.0045 | 0.870 | 0.823 | 1.265e-4 | 8.89e-5 | 9.43e-5 / 7.10e-5 | 2.6e9 |
| ITE-Q-i | x ~ b^2 (VMC), energy metric | 0.0066 | 0.872 | 0.825 | 1.264e-4 | 8.88e-5 | 9.42e-5 / 7.10e-5 | 2.6e9 |
| ITE-P-iii | tempered, pointwise infidelity (p-tVMC) | 0.0092 | 0.873 | 0.826 | 1.263e-4 | 8.87e-5 | 9.42e-5 / 7.09e-5 | 2.0e9 |
| ITE-Q-iii, eta = 1 | K = 1, 500 iterations | 0.0056 | 0.871 | 0.824 | 1.264e-4 | 8.89e-5 | 9.43e-5 / 7.10e-5 | 4.2e8 |
| ORACLE | exact phi_0 as target, tempered, energy metric | 0.083 | 0.899 | 0.849 | 1.233e-4 | 8.69e-5 | 9.30e-5 / 7.03e-5 | (2.5e9) |
| VMC-i (minSR on <H>(b, s_P), eta 0.01 / 0.03 / 0.1, 192 steps) | x ~ b^2 | -0.037 / -0.109 / -0.126 | | | <H>(s_P) 1.323e-4 / 1.349e-4 / 1.348e-4 | | | 1.6e7 |
| ideal FN iteration | | 1 | | | 7.84e-5 | | 6.52e-5 | |
| exact ITE, K = 2, no fit | | 0.970 | | | 8.32e-5 | | 6.80e-5 | |

Composite loop C-P-i (guide = one exact hop on the stored net, net distilled from the next target, 3 iterations):

| it | frac of the composite guide | <H> composite (Krylov sign) | E_FN of the next guide | net fit: val-loss ratio |
|---|---|---|---|---|
| 1 | 0.823 | 8.89e-5 | 7.10e-5 | 0.987 |
| 2 | 0.007 | 8.84e-5 | 7.09e-5 | 0.947 |
| 3 | 0.013 | 8.80e-5 | 7.07e-5 | 0.952 |

**Verdict by the pre-registered rule:** best (ii)/(iii) arm captures 0.037 < 0.50 in loop iteration 1: **not PASS**;
the 1.5x VMC condition is moot (the VMC control lost energy at every step size). The composite arm captures 0.82 in
iteration 1 but 0.01 in iterations 2-3: it fails its own reading (amendment 2) too. No energy-increase event in any
ITE step; lambda_max stays 1.57-1.61e6 on every F_k.

## Convergence evidence and diagnosis (pre-registered order)
1. **Optimiser convergence: not converged, slow and steady.** Symmetrised validation fit loss after 1500 iterations
   (per step): ITE-Q-iii 0.976 / 0.979 of its start, Q-i 0.993 / 0.983, Q-ii 0.991 / 0.999, P-iii 0.996 / 0.992,
   ORACLE 0.899 / 0.965 (curves in `runs/*.json`, `val_hist`). The kept fraction of each exact step's gain is
   0.4-2% (Q-iii 2.1% per step; ORACLE 5.8% / 2.7%). Scan (amendment 3): progress ~linear in eta and iterations;
   eta = 1 does not help (0.56%). A Levenberg-Marquardt variant with per-step validation stalls (P = 28k > batch rows,
   batch noise > batch signal). At ~1-2% of the target per 1000 iterations, one FN iteration would need >= 1e4-1e5
   iterations; the sibling one-shot test (`experiments/writeback_tail/`, exact phi target, 20k minSR/Adam steps)
   saturates at 15-18%. So more iterations do not reach 50% with this network.
2. **tau / stiffness: solved.** The exact semi-implicit targets keep 87% / 97% of the gain (no energy increase,
   K = 2 exact loop within 5% of the ideal loop). The projected loss is entirely in the fit: cumulative fitting loss
   5.1e-5 per site per loop iteration (the whole step).
3. **Sampling: matters, but at the 1% level.** Tempered x ~ b with weights to b^2 (iii) beats b^2 (i) and the FN
   mixed distribution (ii) 5-8x (3.7% vs 0.7% / 0.45%), in line with the tail concentration of the gain (81% on 7.6%
   of the phi^2 weight). The walker distribution (ii) is the worst: it under-samples the tail like b^2.
4. **Representation: limiting.** Exact targets at equal budget (ORACLE) reach only 8.3%. The semi-implicit target is
   an explicit rational function of three one-hop quantities of the guide (kept weight W, wall V, Ising diagonal
   H_xx) and the gain sits on wall configurations whose value is fixed by their neighbours. A zero-hop residual CNN
   has to re-derive neighbour-amplitude ratios of the frozen ViT from the spin pattern; it cannot. One exact hop on the
   unchanged net gives back 82-87% of the iteration (columns "+F_0 hop", "composite"), but a stored zero-hop net
   cannot carry the bulk change of later iterations either (composite loop stalls).
**Verdict: INCONCLUSIVE (cause: representation, zero-hop network vs one-hop correction).** Not a conceptual limit
of FN-ITE: the information and the stiffness are fine; the write-back representation is the open problem.

## Next step (proposal, needs a design decision)
A residual factor that sees the one-hop structure of the FROZEN base: f(x) = g_theta(x, log a_P(x), W_P(x), V_P(x))
with W_P, V_P computed from one hop of |psi_P| (no recursion: the base never changes; cost one shell of the base per
amplitude evaluation, the same class as the composite sign + one hop). Iteration 1 is exactly representable (the
semi-implicit target is a function of W_P, V_P, H_xx). Cheap exact pre-test before any training: project the ideal
loop's iteration-2 and -3 updates onto functions of these features (binned, as in `writeback_tail/wt_exact.py`, but
with H_xx added) and require >= 50% per iteration. Parametrisation (a) (fine-tuning the ViT) is not a fix: same
zero-hop limitation and 22 min per exact table on an A40 (`vitprobe.json`: 8.2e-5 s per configuration forward).

## Deviation from amendment 3
The longer VMC control runs (VMC-i, VMC-iii, 60 min each) were queued but cancelled unstarted after 2 h in the
cip queue: the verdict is fixed by the first condition (0.037 < 0.5), and at every scanned step size minSR on
<H>(b, s_P) with B = 1024 lowered nothing (it raised <H>). The VMC control reported above is the pre-registered scan
(192 steps per eta, 1.6e7 evaluations, ~1% of the ITE-Q evaluation count). A fair VMC comparison at equal evaluations
(~3e4 minSR steps, ~8 s each on an A40 with exactly symmetrised local energies) was not affordable.

## Compute (ws1, `/project/theorie/a/A.Otaifi/chatty_itfit/runs/`)
9.7 GPU-h of the 15 GPU-h budget: Stage 0 0.4 (2080 Ti), exact loop reference 0.15, smokes/debugging ~1.5 (two OOMs
on the 2080 Ti, a zero-head GN degeneracy), inner-optimiser scan 0.9, VMC scan 1.3, main arms 6.1 (A40:
itA1 2.8, itA2 1.7, itC 1.3), ViT throughput probe 0.14. Files: `itfit_exact.json`, `itfit_exactloop.json`,
`scan_inner_q.json`, `scan_vmc.json`, `vitprobe.json`, `runs/itA1.json` (Q-iii, Q-ii, ORACLE, eta = 1),
`runs/itA2.json` (Q-i, P-iii), `runs/itC.json` (composite), `summary_rows.json`; network parameters stay on ws1.

## Follow-up: residual on one-hop features of the frozen base (amendments 4-5, 2026-10-08 evening; stopped when merged into writeback_tail)
- **Exact pre-test (`itfit_feat.json`): GO.** Best frozen-base one-hop family per ideal iteration captures 0.97 / 0.82 / 0.66
  (smooth basis; 256 bins of T(tau=1)); the current guide's one-hop features hold 0.975 / 0.98. A perfectly fitted
  linear base-feature residual in the loop: frac 0.97 / 0.77 / 0.52, E_FN 1.02e-4 -> 6.61e-5 -> 5.46e-5 -> 4.90e-5.
- **Trained (`runs/itF.json`; MLP 5.1k parameters on 13 base features, energy-metric LM-GN, semi-implicit targets,
  realistic samples, exact table clamped to the fitted range on validation configurations):**
  F-Q-iii (tempered): frac **0.951 / 0.801 / 0.686**, <H>(Krylov) 8.42e-5 / 6.50e-5 / 5.48e-5, E_FN 6.85e-5 / 5.54e-5 /
  4.74e-5 (ideal 6.52e-5 / 4.87e-5 / 3.81e-5); with one exact hop on top <H> 6.52e-5 / 5.29e-5 / 4.53e-5 (RBM+PP 4.47e-5).
  F-Q-ii (FN-walker samples): 0.949 / 0.798 / 0.614, E_FN 6.85e-5 / 5.57e-5 / 4.93e-5. Fit validation loss fell
  1.5e-5 -> 3.5e-10 per step (converged). First run without the clamp blew up (`runs/itF_unclamped.json`).
- The capture criterion (>= 0.5 per iteration) is met in all three iterations; the same-model VMC control did not run
  (stopped by the coordinator: route merged into `writeback_tail` Step 2), so the 1.5x VMC condition is untested here.
