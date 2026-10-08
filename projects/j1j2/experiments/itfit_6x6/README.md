# Imaginary-time fitting with the frozen FN Hamiltonian as the loop's amplitude write-back (exact 6x6)

Question: can projected imaginary-time evolution with the frozen fixed-node Hamiltonian F = H_FN[a_k, s_k]
(never the physical H) write one FN iteration back into a callable network, iterated in the FN/Krylov loop?
Method family: p-tVMC / projected ITE (Sinibaldi et al. arXiv:2305.14294; Gravina, Savona, Vicentini arXiv:2410.10720),
fixed-target ITE (Ledinauskas & Anisimovas arXiv:2307.15521). Context: `results/writeback/DESIGN.md` (one-shot fit:
17% with oracle data and Adam, < 1% with realistic VMC estimators; INCONCLUSIVE, cause Adam), `results/stall_6x6/README.md`,
`results/amp_design/DESIGN_MEMO_draft1.md` (identities I1 majorisation, I2 ground-state transform),
STATUS.md 2026-10-03 (one Euler target captured 38-77% of the frozen gain on 4x4, the Adam fit failed).
The sibling test `experiments/writeback_tail/` (one-shot fit, sampling/optimizer arms) is not touched; its stall/sector
code is reused read-only (`../stall_6x6/st6_sector.py`; small helpers are copied, not imported).

Code: `itfit_common.py` (exact sector helpers), `itfit_exact.py` (Stage 0, no training), `itfit_run.py` (Stage 1),
`itfit.sbatch`, `specs/`, `itfit_figure.py`. Results: `results/itfit_6x6/`. ws1: code `~/chatty_itfit/code`, runs
`/project/theorie/a/A.Otaifi/chatty_itfit/runs/`.

## Pre-registration (written 2026-10-08, before any run of this test)

### Definitions (all per site, exact sector k = 0, A1, flip+, J2/J1 = 0.5, E0 = -0.50380965)
- Loop: guide (a_k, s_k) -> F_k = H_FN[a_k, s_k] -> amplitude write-back a_{k+1} (here: projected ITE on F_k)
  -> Krylov sign step s_{k+1} = Krylov(a_{k+1}, s_k) -> next iteration. Start (a_0, s_0) = (|psi_P|, s_P), the
  symmetrised ViT with its own binarised sign, exactly as the ideal exact loop of `stall_6x6` (E_FN of the guides
  1.02e-4 -> 6.5e-5 -> 4.9e-5 -> 3.8e-5). The Krylov sign enters after each amplitude update (with the Krylov sign
  already applied to psi_P, E_FN would be 0.943e-4; not used, to keep the ideal-loop reference exact).
- F = D - K: D = FN diagonal (H_xx plus the sign-violating "wall" W(x) = sum_{y viol} |H_xy| a_k(y)/a_k(x)),
  K = kept (sign-consistent) hops, K >= 0. phi_k = Perron vector of F_k, E_FN,k its eigenvalue.
- Frozen energy E_f,k[b] = <b|F_k|b>/<b|b>. Ideal frozen gain G_k = E_f,k[a_k] - E_FN,k (G_0 = 3.01e-5).
- **Captured fraction of loop iteration k: frac_k = (E_f,k[a_k] - E_f,k[a_{k+1}]) / G_k** (exact). Same metric for the
  VMC control (its a_{k+1} scored on its own F_k).
- Loop gain after iteration k: exact E_FN(a_0, s_0) - E_FN(a_k, s_k), and <H>(a_k, s_k) - E0 (Krylov sign included).
  References: psi_P <H> 1.32e-4; ideal loop guide <H> 7.8e-5 (1 it) / 4.3e-5 (3 it); RBM+PP 4.47e-5.

### Stage 0: stiffness of F (exact, no training) -- monitored risk (PI request)
Risk: F has a stiff FN wall (on 4x4 lambda_max(F) = 2.4e3 vs H bandwidth ~20; explicit ITE needed 50-100 exact steps
for 92-98%). If the explicit step needs hundreds of steps per FN iteration, every step needs a converged fit and the
fitting error accumulates: projected explicit ITE is then unaffordable regardless of the network.
Measured exactly on F_0 (and lambda_max + wall summary on every F_k in Stage 1):
1. lambda_max(F) (Lanczos, 100 steps, checked against max D and the Gershgorin bound), the largest stable explicit
   step tau_stab = 2/(lambda_max - E_FN), and the positivity limit tau_pos = 1/max_x (E_L^F(x) - E) of the explicit
   target a(1 - tau (E_L^F - E)) (log fits need t > 0).
2. Distribution of the wall W(x) by decades: configurations, phi^2 and a^2 weight, share of the ideal gain
   (I2 node weight of delta = log phi - log a), share of Var(delta).
3. Exact (unprojected) step schemes, captured fraction vs step n, steps to 90% (n90; extrapolated from the last-20%
   geometric rate if not reached), energy-increase events, negative-target weight, matvecs per step:
   - explicit Euler t = (1 - tau (F - E)) a, tau = 0.9 tau_stab and tau = tau_pos;
   - optimal explicit (1-hop Rayleigh-Ritz in span{a, F a}, restarted; = Euler with exact line search);
   - Krylov 2-hop target: Rayleigh-Ritz in span{a, F a, F^2 a}, restarted;
   - **semi-implicit (diagonal-implicit) Euler** t = (a + tau K a) / (1 + tau (D - E)), tau in {0.03, 0.1, 0.3, 1, 3}:
     the stiff diagonal (wall) is treated implicitly, the kept hops explicitly. It is local (one hop of a and of the
     guide), always positive, has phi as its fixed point, and is the diagonal (Jacobi) preconditioning of the
     energy-metric Newton step (the I2 Laplacian has diagonal D(x) - E_FN at phi);
   - **implicit Euler** t = (1 + tau (F - E))^-1 a, tau in {0.03, 0.3, 3}, solved exactly by Jacobi-preconditioned CG.
     With samples it is not a target but a fit with the operator on the trial side: minimise
     sum_x mu(x) |[(1 + tau (F - E)) b'](x) - a(x)|^2 / a(x)^2 (one hop of b' per sample, gradients through the hop;
     at tau -> infinity this is local-energy variance minimisation of F). Reported for comparison only.
   - captured gain after ONE step vs tau for each scheme.
4. **Stiffness keep/kill criterion (decided now):** affordable = K_max = 15 ITE steps per loop iteration (each step
   needs one converged fit, ~1.5-2 A40-min with the residual CNN; 3 loop iterations x 4 ITE arms ~ 6 GPU-h).
   - KEEP: a LOCAL scheme (explicit or semi-implicit) reaches >= 90% of G_0 within 15 exact steps. Stage 1 uses the
     local scheme and tau with the smallest n90 that has no energy-increase event; K = its n90.
   - KEEP WITH CAVEAT: none reaches 90% within 15, but one reaches >= 60% within 15: K = 15; the 50% pass bar then
     requires the fits to keep >= 5/6 of the exact per-step gain.
   - STOP (stiffness-limited at this budget): no local scheme reaches 60% within 15 steps. Report n90 per scheme,
     which decades of the gain converge slowly, and whether implicit/Krylov variants fix it at what cost; Stage 1 is
     then reduced to the oracle ceiling + one ITE arm as a diagnostic; verdict INCONCLUSIVE (cause: stiffness of the
     explicit/semi-implicit step) unless the slow part is shown to be intrinsic to F.

### Stage 1: projected ITE with training (exact evaluation, realistic training data)
Parametrisation (b) (residual factor), as in `writeback`: b_theta = |psi_P| exp(f_theta), f = residual CNN 32 ch x 4
layers (28k parameters), exactly D4 x flip symmetrised (all 16 images in every evaluation, including Jacobians),
zero-initialised head, |psi_P| frozen; f is warm-started across ITE steps and loop iterations (one net carries the
cumulative correction). Parametrisation (a) (fine-tuning the 155k-parameter symmetrised ViT; 64 ViT passes per
configuration, a full symmetrised table ~23 min on a 2080 Ti) is estimated at >= 2 A40-h per loop iteration and arm
(tables per ITE step or 2.3e7 ViT passes per fit iteration); it is not run at this budget. The smoke job measures its
throughput to make this a number, not a guess.

Per ITE step n of loop iteration k (b_n = current network, exact table each step):
- target t_n = scheme(b_n) on F_k (scheme and tau from Stage 0), computed exactly; in a real loop t_n(x) needs b_n
  and a_k on the one-hop shell of x (counted as evaluations, see below).
- fit: theta_{n+1} from theta_n by n_in Gauss-Newton (Levenberg-Marquardt) iterations, each on a fresh batch of
  B = 1024 configurations, minSR-form solve d = J^T (J J^T + lam tr(JJ^T)/R)^-1 r (Chen & Heyl arXiv:2302.01941),
  accept/reject on a fixed validation batch of the step (2048 configurations), lam x4 on reject, /2 on accept.
  Not Adam.
- **Q-fit (main): GN in the energy metric of the target**, residual on sampled edges:
  Q_n(e) = 1/2 sum_x mu(x) sum_{y kept} |H_xy| (t_n(y)/t_n(x)) (e_x - e_y)^2, e = log b' - log t_n,
  K_b = 4 kept bonds per x drawn uniformly among the kept bonds of x (weight n_kept/K_b).
- P-fit (secondary, standard p-tVMC): pointwise natural-gradient fit, residual sqrt(mu(x)) (e_x - <e>), i.e. the
  infidelity to t_n to second order, minSR.
Training distributions mu (exact sampling from tables; no MCMC autocorrelation, equal for all arms):
- (i) x ~ b_n^2 (VMC samples) + neighbours, unweighted;
- (ii) x ~ a_k phi_k (the FN mixed distribution that walkers give), unweighted (walkers give no phi(x) values);
- (iii) x ~ b_n (tempered beta = 1/2), self-normalised weights b_n to the b_n^2 measure (the energy-metric edge
  weight then becomes |H_xy| t_y, bounded; draft-1 memo).
Arms (all from the same start, same network, 3 loop iterations unless stopped):
1. ITE-Q-i, 2. ITE-Q-ii, 3. ITE-Q-iii (main arms);
4. ITE-P-iii (fit-metric control: pointwise infidelity fit, otherwise as 3);
5. VMC-i: same-capacity fixed-sign VMC, minSR on <H>(b, s_k) (the physical H at the loop's sign), x ~ b^2,
   local energies over all valid bonds; Krylov sign step after each loop iteration exactly as the ITE arms;
6. VMC-iii: as 5 with tempered beta = 1/2 sampling + self-normalised weights (the control gets the same trick);
7. ORACLE (loop iteration 1 only): Q-fit with the exact phi_0 as the target of every step (same K x n_in GN
   iterations, distribution iii): the ceiling of this network + optimiser for one FN iteration.
VMC step size eta from a short smoke scan {0.01, 0.03, 0.1} at the start (best exact <H> after 200 steps).
**Equal evaluations:** cost = forward network evaluations on configurations, counting target shells as a real loop
would need them (2 shells of 1 + n_valid(x) configurations per target point: b_n and the guide a_k) plus one per fit
point; VMC: one shell per sample plus one per sample. The VMC arms get the evaluation count of the ITE-Q arms of the
same loop iteration (about 10x more VMC steps than ITE fit iterations). Counts are logged exactly.
Monitored per ITE step (exact, from the step's table): E_f,k[b_n] (energy-increase events), E_f,k[t_n] (gain of the
exact target vs tau), kept fraction of the step's exact gain (E_f[b_n] - E_f[b_{n+1}]) / (E_f[b_n] - E_f[t_n]),
cumulative fitting loss sum_n (E_f[b_{n+1}] - E_f[t_n]), validation fit loss per GN iteration. Per loop iteration:
lambda_max(F_k), wall summary, frac_k, <H>(a_{k+1}, s_k), <H> and E_FN with the Krylov sign s_{k+1}.

### Pass rule (decided now)
- **PASS** iff the best ITE arm with distribution (ii) or (iii) has **frac_k >= 0.50 in every completed loop
  iteration** (>= 2 completed) **AND** its loop gain in exact E_FN after the last completed iteration is
  **>= 1.5x the gain of the better VMC control** at equal evaluations (a control gain <= 0 satisfies this if the ITE
  gain is > 0). Then: report the realistic cost and propose the 8x8 sampled version.
- Otherwise not PASS. Before any verdict other than INCONCLUSIVE, diagnose in this order:
  (1) optimiser convergence: validation fit loss per GN iteration flat? kept fraction of each step's exact gain;
  (2) tau/stiffness: projected vs exact unprojected curve at the same K, energy-increase events;
  (3) sampling: (i) vs (ii) vs (iii);
  (4) representation: the ORACLE ceiling. If ORACLE < 0.5 the 28k residual CNN cannot hold one FN iteration with this
  optimiser and the ITE route is representation-limited at this capacity (next lever: non-local/pair-product factor
  or larger net), not refuted.
- Arms with frac_1 < 0.10 stop after loop iteration 1 (except the VMC controls when an ITE arm continues).
- Budget <= 15 GPU-h (full A40 or RTX 2080 Ti; no V100 / A40 slices), full fp32 networks, float64 exact algebra.
  Planned: Stage 0 <= 1.5, smoke <= 1.5, Stage 1 <= 10, reserve 2.
- Any change of K, tau, n_in, B or eta after Stage 0 / smoke is recorded as an amendment BEFORE the Stage 1 results.

## Amendment 1 (2026-10-08 12:40, after Stage 0, before any Stage 1 training result)
**Stage 0 result (exact, `results/itfit_6x6/itfit_exact.json`, job 16902333, 2080 Ti 24 min):**
- lambda_max(F_0) = **1.61e6** (Lanczos converged by m = 10; = max D, Gershgorin 1.61e6); the H bandwidth is ~35.
  The largest wall sits on a configuration with phi^2 = 3e-25. tau_stab = 2/(lambda_max - E_FN) = **1.24e-6**;
  the positivity limit of the explicit target tau_pos = 1.8e-6 is above tau_stab (explicit at tau_pos diverges).
- Wall W(x) (phi^2-weighted mean 3.0): 93.5% of the phi^2 mass has W > 1, 1.4% has W > 10 and carries 35% of the
  ideal gain; W > 100 holds 2e-5 of the mass and 0.8% of the gain. The gain sits on walls 1-100; the stiffness is set
  by walls ~1e6 that carry nothing.
- Exact steps (captured fraction of G_0 after n steps; n90):
  explicit at 0.9 tau_stab: 5.6e-5 per step, 4.7% after 1000 steps, n90 ~ 5e4 (extrapolated; the ITE time to 90%
  from the implicit tau = 0.01 run is beta ~ 0.09, i.e. ~8e4 explicit steps);
  optimal explicit (Euler with exact line search = 1-hop Rayleigh-Ritz, the Ledinauskas-Anisimovas adaptive dt):
  23% / 25% / 33% after 1 / 3 / 200 steps (n90 ~ 9e3); Krylov 2-hop RR: 24% / 36% after 1 / 80 restarts;
  **semi-implicit (diagonal-implicit) Euler: one step 51% (tau 0.03), 73% (0.1), 83% (0.3), 87% (1), 88% (>= 10);
  two steps 91-97% for tau >= 0.1 (n90 = 2), 99% in 3-5 steps, no energy-increase event**;
  implicit Euler (global CG solve, 35-47 matvecs per step): tau = 0.3 97% in one step; tau = 0.01: n90 = 9.
- Unprojected loop with K exact semi-implicit steps per iteration (`itfit_exactloop.json`): tau = 1, K = 2:
  E_FN of the guides 1.02e-4 -> 6.80e-5 -> 5.13e-5 -> 4.05e-5 (ideal loop 6.52e-5 / 4.87e-5 / 3.81e-5),
  <H> with the Krylov sign 8.32e-5 / 6.01e-5 / 4.64e-5 (ideal 7.84e-5 / 5.58e-5 / 4.30e-5), frac 0.97 / 0.96 / 0.96.
**Stiffness decision (pre-registered rule): KEEP.** The explicit step is killed by the wall (n90 ~ 5e4-8e4 >> 15);
the local semi-implicit step reaches 90% in 2 steps. Stage 1 uses the semi-implicit target with **K = n90 = 2**.
Tie-break among the tau with n90 = 2 (not specified before; fixed now, before any Stage 1 result): the smallest tau
whose two-step fraction is within 0.01 of the best (0.975 at tau = 100): **tau = 1** (0.970). It avoids the pure
Jacobi limit (tau -> infinity), whose bipartite -1 mode is undamped.

**Optimiser detail (fixed after the smoke runs, which had no usable fit result):** the residual net's zero-initialised
head makes every Jacobian column except the 33 head columns vanish, so a validated GN step can never leave the
head-only subspace (all steps were rejected in the first smoke). Changes: head initialised with std 1e-4 (f ~ 1e-4,
exact start energies are measured and reported), Levenberg-Marquardt with Marquardt column scaling (Jacobian columns
normalised to unit rms) and up to 4 re-solves with x4 damping on the same batch before a step is rejected; lam starts
at 1 (relative to tr(JJ^T)/R). With P = 28k parameters > R rows the undamped GN step interpolates the batch and does
not generalise (synthetic check), so the damping is set by the validation batch. The VMC control keeps standard minSR
(no column scaling) with eta from the scan, as pre-registered.

## Amendment 2 (2026-10-08 13:25, after smoke runs whose fits were broken or only 12 iterations long; before any Stage 1 result)
- **Training:** one random D4 x flip image per configuration (shared with its sampled bonds) in the GN Jacobian and
  residuals (16x cheaper; the fit losses are quadratic in f, so by convexity the symmetrised f does at least as
  well); validation, acceptance and every exact evaluation use the exactly symmetrised f. B = 2048 (Q-fit, 8192 edge
  rows), 4096 (P-fit); up to n_in = 150 GN iterations per ITE step, stopped earlier when 3 consecutive iterations fail
  all 4 LM re-solves or the validation loss improves by < 0.1% over 25 iterations (convergence evidence = these
  curves, saved per step).
- **Diagnostics added to every loop iteration (all arms, exact):** (a) frac of T_{F_k}[b], one exact semi-implicit
  step of the current F_k applied to the fitted network (how much of the remaining error one exact hop repairs);
  (b) the **one-hop composite guide** c = T_{F[b, s']}[b] (semi-implicit step of the network's own FN Hamiltonian with
  the new Krylov sign s'): frac on F_k, <H>(c, s'), E_FN(c, s'). c is computable with one hop of the network, like
  the composite sign + one exact hop of the sign work.
- **New exploratory arm C-P-i (composite amplitude + one hop)**, motivated by Stage 0 (the semi-implicit step is an
  explicit one-hop function of the guide and captures 87% of an FN iteration): guide of iteration k is
  c_k = T_{F[n_k, s_k]}[n_k] (c_0 = |psi_P|); target t_k = T_{F_k}[c_k] (K = 1); the zero-hop net n_{k+1} is
  fitted to t_k by the pointwise natural-gradient fit under b^2 samples (the hop is expected to re-equilibrate the
  wall configurations from their neighbours, so the stored net mainly needs the bulk); 3 loop iterations. Its targets
  need two hops of n_k in a real loop (counted as (1 + n_valid)^2 per target point). **Reading, decided now:** the
  composite route is viable at 6x6 if the guide's frac_k >= 0.5 in each of 3 loop iterations AND its loop E_FN gain
  is >= 1.5x that of the better VMC control. It is a separate question from the pre-registered PASS rule for the
  zero-hop write-back, which is unchanged.
- **VMC budget:** the VMC arms get the evaluation count of the ITE-Q arms' first loop iteration, capped at 2.5 GPU-h
  per arm; if the cap binds, the comparison is reported at the VMC's actual (lower) count, which favours the ITE arms.

## Amendment 3 (2026-10-08 14:15, after the inner-optimiser scan, before the main Stage 1 runs)
- **Inner optimiser = stochastic natural gradient in minSR form** (fresh batch every iteration, fixed step eta, every
  step taken, symmetrised validation loss every 100 iterations, best-validation parameters kept), as in p-tVMC inner
  loops. The LM variant with per-step validation acceptance (amendment 1) stalls: with P = 28k > rows the batch
  signal is below the batch noise, so almost every single step fails a validation check although the average step
  helps (smokes: 0.1-0.35% of the fit loss in 10-12 iterations, lam driven to the cap).
- Scan on the first target (Q-fit, distribution iii, B = 1024 x K_b = 4... run with B = 2048, 400 iterations, exact
  frac of the fitted net, `results/itfit_6x6/scan_inner_q.json`): eta = 0.03 / 0.1 / 0.3 -> frac 0.20% / 0.38% /
  1.0% (validation fit loss -0.2% / -0.4% / -1.6%). Main runs use **eta = 0.3, lam = 1e-3, B = 1024 (Q, 4096 edge
  rows) / 4096 (P), n_in = 1500 iterations per ITE step, K = 2**; one sensitivity arm ITE-Q-iii-eta1 (eta = 1,
  loop iteration 1 only). Progress is roughly linear in eta and in the number of iterations at this budget, i.e. the
  fits are NOT converged; convergence evidence = the saved validation curves.
- VMC controls: one loop iteration each (the ITE arms are expected below the 0.10 stop rule), evaluation budget of an
  ITE-Q arm's first loop iteration (2.47e9 network evaluations) capped at 75 min of GPU time; eta from the VMC scan.
