# Writing the FN update back into a network: diagnosis, candidates, decisive test (2026-10-07)

Context: `results/stall_6x6/README.md`. The exact FN step from the symmetrised ViT psi_P lowers <H> by 4.8e-5/site, but
every projection onto a callable network loses it. Needed: projection error <= ~0.004 rms in log a; the update itself
has rms 0.011. All numbers per site, J2/J1 = 0.5, 6x6, exact symmetric sector. Test code: `experiments/writeback/`.

## 0. Bottom line
**Verdict: PARTIAL by the pre-registered rule; the write-back route as designed is closed.** Diagnosis:
1. The energy cost of a write-back error is a Dirichlet form on the H-graph. For the frozen FN Hamiltonian it is exact:
   Q_FN(e) = E_f[b] - E_FN (verified to 1e-11). Rough errors cost 0.34 sigma^2 = (2/3)|E/N| sigma^2 (intensive).
2. The FN update itself is **as rough as white noise** in that metric (0.97 of white noise; 0.5 in the H metric).
3. **81% of its gain sits in low-amplitude configurations holding 7.6% of the phi^2 weight.**
4. So no smooth or compact family carries it, and a pointwise (phi^2-weighted) loss ignores the edge structure that
   sets the cost: on identical data it keeps half of what the energy loss keeps.

Exact 6x6 test, fresh residual CNN on the symmetrised ViT:
- With oracle phi_FN and the right metric, the factor keeps **17%** of one iteration (infidelity loss 8%, 4x wider
  net 13%).
- With realistic VMC data at equal cost, it keeps **< 1%**, and the FN target ties plain fixed-sign VMC (0.8% vs
  0.9%).
- The reviewer's pre-registered rule therefore drops the FN amplitude target. The sign step and the FN bound survive.

Ranked: (1) energy-metric residual factor on a frozen base, (2) natural-gradient energy fits of the full net,
(3) infidelity fits. Only (1) moved the energy here ((2) gains ~1e-6/step on the ViT, stall_6x6), and not enough.

## 1. Diagnosis: which errors cost energy
**1.1 The metric is a Dirichlet form on the H-graph, not L2.** Write the trial state as b = phi g at fixed sign.
For the stoquastic H_FN = H_FN[a_k, s_k] with Perron vector phi (H_FN phi = E_FN phi), the ground-state transform is exact:

  E_f[b] - E_FN = 1/2 sum_{x,y allowed} |H_xy| phi_x phi_y (g_x - g_y)^2 / sum_x phi_x^2 g_x^2 ,

where "allowed" are the sign-consistent hops (the violating ones sit on the FN diagonal). For <H> at fixed sign the same
form holds with signed edge weights (frustrated edges negative) plus a local-energy term; this is the edge Laplacian of
`amp_design/DESIGN_MEMO.md`. An error e = log b - log phi therefore costs through its **edge differences** e_x - e_y,
weighted by phi_x phi_y |H_xy|. A pointwise loss weights e_x^2 by phi_x^2. Infidelity is the same pointwise loss:
1 - F = Var_{phi^2}(e) + O(e^3).

**1.2 Rough errors cost a universal, intensive amount.** For iid log-noise of rms sigma,
dE = sigma^2 |<H_offdiag>|/N exactly to second order. For an SU(2) singlet the off-diagonal (exchange) part is 2/3 of
the energy, so dE = (2/3)|E/N| sigma^2 = **0.336 sigma^2** (measured: 0.33-0.35, `interp_fn_update.json`). The coefficient
is the kinetic energy per site, so it is the same at 8x8 and 10x10. In the FN metric the white-noise coefficient is
kappa_FN = <sum_{y allowed}|H_xy| phi_y/phi_x>_{phi^2}/N = **0.420** (exact).

**1.3 The FN update itself is not smooth in this metric.** delta = log phi_FN - log|psi_P| has rms 0.0086 under phi^2
(0.011 under psi0^2). Exact frozen-FN gain G0 = Q_FN(delta) = 3.01e-5, so Q_FN(delta)/Var(delta) = 0.41 = **0.97 of
white noise** in the FN metric. In the H metric (fit to the seven exact partial steps, <H>(a e^{t delta}, s_P) =
1.319e-4 - 6.0e-5 t + 1.25e-5 t^2) the curvature is 0.17 Var(delta), **0.5 of white noise**. Consequences:
- No smooth or low-dimensional family can carry it: 44 symmetric clusters 0.7%, 7 one-hop features 8% (stall_6x6).
- Under-shooting is cheap, scrambling is not: an error along delta (b = a e^{(1-eps) delta}) costs ~3.5e-5 eps
  (linear), whereas an error with white-noise roughness costs 0.34 sigma^2, i.e. the whole gain at sigma = 0.012.
- The requirement is Q(e) <= 0.1 x gain ~ 5e-6/site. Only for rough e does that mean rms <= 0.004.

**1.4 Where the gain sits.** Decomposition of the frozen-FN gain over decades of the per-configuration weight phi_x^2
(exact; positive per-configuration edge sums):
| per-config phi^2 | share of phi^2 mass | share of Var(delta) | share of the gain Q_FN(delta) | share of white-noise cost |
|---|---|---|---|---|
| >= 1e-9 | 92.3% | 11% | 19% | 89% |
| 1e-12 .. 1e-10 | 7.6% | 75% | 66% | 11% |
| < 1e-12 | 0.05% | 14% | 15% | 0.1% |

**81% of the gain sits in configurations that hold 7.6% of the phi^2 weight**, where delta is large. The FN update is a
tail correction: the low-amplitude configurations next to high-amplitude ones (weight phi_x phi_y on the edge, but
phi_x^2 in any pointwise loss). This is the amplitude analogue of the "self-sealing" sign errors (stored_signs). In
practice: sampling x ~ phi^(2 beta) with beta = 0.5 cut the standard error of the energy-metric loss 9x at equal batch.

**1.5 Why every projection so far failed.**
1. *Wrong metric.* L2/fidelity fits reduce the phi^2-weighted pointwise error, but the residual a network leaves is
   rough at the configuration level (networks fit the smooth part first: spectral bias, Rahaman et al. arXiv:1806.08734;
   Xu et al. arXiv:1901.06523). The L2 fit to phi_FN left rms 0.012 and lost the whole gain (stall_6x6 test 2).
2. *Refitting the whole network scrambles the base.* The ViT's own error (rms 0.045 vs psi0) was placed by VMC in soft
   modes: (<H>(|psi_P|, exact sign) - E0)/rms^2 = 1.56e-4/0.0445^2 = 0.079, 0.23 of white noise. Any refit of all weights moves part of that large error into
   stiff modes; a 0.011 update cannot survive that. Supervised fits to |psi0| *lowered* the pointwise error and *raised*
   E (stall_6x6 test 1).
3. *The update lies outside the network's tangent space.* E_f[b] >= <H>_b with equality at b = a, so at a VMC-stationary
   network the frozen-FN objective is stationary too (review finding 1). Inside the same family the FN target can only
   help through signs, new parameters or conditioning; its gain lives in the complement of the tangent space. SR/RGN on
   the trained ViT gains ~1e-6/step (1e-5 rms parameter steps raise E by 1e-2), large-N fixed-sign VMC < 1e-5 (AMP6).
4. *The update sits in the tails* (1.4): any loss estimated on phi^2 or b^2 samples sees 7.6% of the relevant
   configurations, so the estimator variance, not only the metric, limits a sampled projection.

## 2. Candidates (ranked)
**R1. Frozen base x fresh residual factor, trained in the energy metric, accumulated over iterations.**
b = a_base exp(f_theta), a_base = |psi_P| frozen for the whole loop, f_theta a new exactly symmetric network with f = 0 at
start. Iteration k minimises E_f[a_k, s_k](b) over theta, warm-starting theta from iteration k-1 (one net carries the
cumulative correction; the base is never refit). Loss: the identity of 1.1. At 6x6 it is estimated exactly on
x ~ phi^2 with one-hop ratios; at 8x8 it is ordinary fixed-sign VMC on the stoquastic H_FN with samples from b^2
(local energy needs a_k and b on the one-hop neighbours: the zero-hop cost of AMP6, ~1e5 samples per 20 A40-min).
The Gauss-Newton matrix of this loss is J^T L_FN J with L_FN the allowed-edge Laplacian, i.e. the RGN/A_+ damping of
draft 4, so the natural-gradient optimiser is already built.
- Controls: Q_FN(e) directly; keeps the base's soft-mode errors untouched (point 2); new parameters near f = 0 are
  well conditioned (point 3). Precedents for correction factors on a frozen or physical base: RBM+PP
  (Nomura et al. arXiv:1709.06475; arXiv:2005.14142), neural Jastrow on Gutzwiller states (Ferrari et al.
  arXiv:1906.00463), neural backflow (Luo & Clark arXiv:1807.10770).
- Open risk: whether a modest f can *represent* delta (point 1.3). This is what the test below decides.
- Cost: P_f ~ 3e4-1e5, so N >= P_f is affordable at 8x8 (AMP6: generalisation solved at N = 1e5).

**R2. Natural-gradient / RGN fitting of the full network in the energy metric** (draft 4, A_+ damping; MinSR
Chen & Heyl arXiv:2302.01941; Rende et al. arXiv:2310.05715). Right metric, wrong parametrisation at the ViT: points 2-3
cap it at ~1e-6/step. Useful as the optimiser *inside* R1 (on theta only) and from scratch, not for write-back into a
converged net.

**R3. Infidelity / supervised projector fits with control variates** (Sinibaldi et al. arXiv:2305.14294; Gravina,
Savona, Vicentini arXiv:2410.10720; supervised wave-function optimisation, Kochkov & Clark arXiv:1811.12423;
Ledinauskas & Anisimovas arXiv:2307.15521; Giuliani et al. arXiv:2303.08902; teacher-student gate fits,
Jonsson et al. arXiv:1808.05232). These control 1 - F = Var_{phi^2}(e): low-variance estimators, but the wrong norm.
They certify the energy only if the residual is <= 0.004 in *all* directions. Westerhout et al. (arXiv:1907.08186)
find poor generalisation of supervised fits in frustrated magnets; Wu et al. (arXiv:2305.03394) compare supervised
losses. Use only to
warm-start f. Adding an edge-difference (Sobolev-type, Czarnecki et al. arXiv:1706.04859) term turns R3 into R1.

**Keep two routes as parallel arms at every size (PI).** Same residual network, same samples and network evaluations,
referee = exact <H>/E_FN at 6x6 and the calibrated FN referee at 8x8:
- *Energy route* (frozen <H_FN> or fixed-sign <H>; one estimator, two Hamiltonians). Expected to scale best in its
  metric: the objective is intensive and is the energy itself, so a better loss value is a better state at any N. The
  sampled estimator self-averages (per-site SE ~ sigma(E_L)/(N sqrt M) with sigma(E_L)^2 ~ N), so the samples needed
  per step at fixed per-site resolution fall like 1/N while each sample costs ~N network evaluations: roughly
  N-independent cost per step. Frozen <H_FN> and fixed-sign <H> share the gradient at b = a (review finding 1) and
  differ only at second order. Fixed-sign <H> should gain at least as much per step; frozen <H_FN> has a positive
  Laplacian Hessian (no frustrated edges), which should matter more as frustration and N grow. Risk: the gain sits in
  the tails (1.4), and the tail weight fraction shrinks with N. Tempering (beta < 1) has to become stronger and
  importance weights degrade.
- *Pointwise route* (infidelity / supervised fit to a_FN, with control variates). Expected to degrade with N.
  - The energy tolerance on the *rough* error is intensive: sigma_rough <= sqrt(dE_iter / kappa) ~ 0.004 at any N.
    But 1 - F ~ Var(e) is extensive for smooth errors, so the energy-relevant part becomes a vanishing fraction of the
    loss.
  - Fidelity estimators need samples of the target (phi^2), which FN walkers do not give (they give a phi), so it
    needs forward walking or reweighting, whose variance grows with N.
  - It stays useful as a cheap, low-variance warm start. At 6x6 it already captures 2x less than the energy loss on
    identical data.

Not ranked but recorded:
- *Projector-step representations*: Lanczos/power steps (Sorella arXiv:cond-mat/0009149; Hu et al. arXiv:1304.2630)
  are exact but cost one hop per step and recurse like the signs did; one hop = 8% here.
- *Non-local factor families*: a pair-product (Pfaffian) or GCNN residual (Roth & MacDonald arXiv:2104.05085) is the
  next architecture if a CNN residual is representation-limited.
- FN/projector + NQS on lattices exists only for sign-free models (Inack et al. arXiv:1809.03562; Brodoloni et al.
  arXiv:2407.05978); FN-SR (Sorella & Capriotti arXiv:cond-mat/9902211) is the closest classical analogue.

## 3. Decisive test (exact 6x6; pre-registered in `experiments/writeback/README.md`)
Representability test with oracle data plus realistic same-capacity controls. Start: guide (|psi_P|, s_P); exact
phi_FN; frozen-FN gain G0 = 3.01e-5/site; the exact step lowers <H>(s_P) 1.319e-4 -> 0.841e-4. Model
b = |psi_P| exp(f), f = residual CNN (exactly D4 x flip x translation symmetric, zero-initialised head). All arms:
Adam, 20k steps, equal network evaluations per step, samples ~ phi^(2 beta) or psi_P^(2 beta) with beta = 0.5.
Exact sector numbers at the final parameters (`runs/*.json`; figure `writeback_6x6.png`):

| arm | data | loss | gain captured, Q_FN | <H>(b, s_P) | after Krylov: <H> / E_FN | 1 - F |
|---|---|---|---|---|---|---|
| start psi_P | | | 0 (Q_FN = 3.01e-5) | 1.319e-4 | 1.266e-4 / 0.943e-4 | 7.05e-5 |
| **E-M** (28k) | oracle phi_FN | frozen-FN energy (exact edge identity) | **17%** (2.49e-5) | 1.256e-4 | 1.202e-4 / 0.913e-4 | 5.19e-5 |
| I-M (28k) | oracle phi_FN | infidelity | 8.4% (2.76e-5) | 1.286e-4 | 1.232e-4 / 0.929e-4 | 5.72e-5 |
| Vfn-M (28k) | realistic VMC | frozen-FN energy | 0.8% | 1.317e-4 | (best-by-val: 1.267e-4 / 0.943e-4) | 6.76e-5 |
| Vh-M (28k) | realistic VMC | fixed-sign <H> | 0.9% | 1.316e-4 | (best-by-val: 1.269e-4 / 0.943e-4) | 6.70e-5 |
| E-L (111k) | oracle phi_FN | frozen-FN energy | 12.7% (2.63e-5) | 1.272e-4 | 1.219e-4 / 0.922e-4 | 5.50e-5 |
| E-M, lr 1e-2 | oracle phi_FN | frozen-FN energy | 3.9% (2.89e-5) | 1.306e-4 | 1.252e-4 / 0.938e-4 | 6.36e-5 |
| exact FN step | | | 100% | 0.841e-4 | 0.784e-4 / 0.652e-4 | 0 |
| 7 one-hop features (stall_6x6) | exact | frozen-FN energy | 7.7% | 1.271e-4 (capacity) | | |

- **Verdict by the pre-registered rule: PARTIAL** (oracle E arm 17%, between 10% and 50%). With *perfect* data and the
  right metric, a 28k residual CNN keeps one sixth of one FN iteration. Its residual stays white-noise rough
  (Q_FN(e)/(kappa Var e) = 1.13): what it cannot represent is exactly the rough tail part of delta.
- **The metric matters (control I-M):** at equal data the energy loss captures 2.0x the gain of the infidelity loss, and
  it even reaches the lower infidelity (5.19e-5 vs 5.72e-5). The pointwise loss wastes capacity on the phi^2-heavy
  configurations that carry little of the gain.
- **Realistic data kill it at this budget (Vfn-M, Vh-M):** with VMC local energies (no phi_FN) and the same network
  evaluations (1.3M samples x 144 bonds), both energy estimators capture < 1%; the sampled objective's standard error
  (2.3e-5/site per 4096 samples, smoke check) is the size of the whole gain. FN target and plain fixed-sign VMC tie
  (0.8% vs 0.9%): **by the reviewer's pre-registered rule the FN amplitude target is dropped** (tie), and with the
  realistic FN arm < 0.1 the cost case worsens. The oracle arm shows the estimator, not only the network, is the
  bottleneck: the zero-variance edge identity reached 17% where realistic VMC reached 1%.
- **Not a capacity floor at this budget (E-L, lr probe):** 4x the parameters captured *less* at equal steps (12.7% vs
  17%), and lr 1e-2 was worse than 3e-3 (3.9%). The E-M curve flattens at 15-20k steps (figure a). What
  limits it is fitting a rough, tail-concentrated function by stochastic gradients. Extra width does not help.
- Compute: about 4.7 GPU-h (A40 3.5, RTX 2080 Ti 1.2, mostly smoke/memory debugging on the 2080 Ti) (A40 and RTX 2080 Ti), of the 6 GPU-h budget. Runs on ws1
  `/project/theorie/a/A.Otaifi/chatty_writeback/runs/`; code `experiments/writeback/`.

## 4. Next step
1. **Stop writing FN amplitude updates back** (no loop, no 8x8 write-back run). The amplitude is obtained by ordinary
   VMC; FN/Krylov is kept for signs (to be decided by the Lanczos-vs-Krylov comparison) and as the variational referee.
2. **If the amplitude write-back is ever revisited**, the only open lever is the representation of a rough tail
   correction. Two parallel arms (PI), same harness (`wb_run.py`, oracle phi_FN, about 1 GPU-h each):
   - a non-local factor (pair-product/Pfaffian residual, as in RBM+PP);
   - the energy loss with tail-tempered sampling (beta = 0.25).

   Revisit only if one of them passes 50% with oracle data. Then rerun the realistic VMC arms, sign step and FN
   referee included, before any 8x8 cost.
3. Record in FAILED_ROUTES: "write-back of the FN amplitude update into a residual network". Why it failed: the update
   is white-noise rough and tail-concentrated; 17% even with oracle data, < 1% with realistic samples.
