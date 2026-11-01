# AVSL-08 — actual M128 FN walkers and rare-correction search (2026-10-19)

**Status: completed 20-site diagnostic; not a scalable FN–Krylov RL algorithm.**

## Why this is not another ground-state sign-training exercise
We test the existing **post-Krylov sign chamber** and its approximate FN-optimized guide. A learned *exploration* policy starts from real FN walker configurations, moves one/two legal Heisenberg Hamiltonian hops to propose which **symmetry orbit to verify**, and receives the **signed magnitude** of a verified frozen-guide single-orbit energy change. The policy does **not** learn the ground-state sign vector or use ED ground-state sign labels. All trial proposals are measured; harmful changes are not accepted.

The actual stochastic populations are produced using the **existing `K1FNEngine.run_fn`**, M=128, importance-transformed FN propagation and systematic reconfiguration. The run starts with an independent (a_{guide}^2)-Metropolis **initialization**, *not* the earlier toy (q\propto a) root sampler. The late/burn snapshots are real branching FN histories at finite projection length. **However** `amp` from `sign20_training_data.npz` is an *approximate* FN guide, and its independently computed (a^2)-weighted local-energy standard deviation is **0.259**, so it is NOT an exact FN eigenvector. Its true stationary mixed law would be (f\propto a_{guide}\phi_{FN}), which is not generally (a_{guide}^2). Convergence with imaginary time, M128 population bias, and statistical independence are not established.

A closely related **real mixed FN sampler had already been implemented** in `results/scalable_sign_2026-10-18/j1j2_walker_only_amplitude20.py`, using a different analytic Jastrow guide; this experiment isolates **current post-Krylov guide walker support and learned proposer vs cheap outward controls**, not first-ever real FN sampling.

## Hidden exact-oracle audit: where are the valuable corrections?
At this 20-site frozen guide, full post-hoc enumeration of 4,654 symmetry orbits reports:
- **198 individually profitable sign-flip orbits** with gain `>1e-12` in total-energy units. Their **summed individual gains** are `0.0008001565242074627` (NOT jointly achievable gain).
- Their total (a_{guide}^2) probability is `4.8317771103635376e-05`. **This is guide-squared mass, not a measured stationary FN mixed mass.**
- The actual **M128 FN walkers almost never visit them**: only 1 positive orbit in 12 short beta=.38 FN populations (1,536 post-burn snapshots/population) and 2 positive orbit hits in 12 beta=3 populations (2,688 post-burn snapshots/population). There is strong observed support suppression in the tested finite-beta FN chains, not a proven limiting FN equilibrium law.
- Looking at **one legal Hamiltonian hop** from *all* observed walker snapshots (which is **much more expensive than checking only 120 proposed orbits**) makes roughly 49 positive orbits reachable per short-beta population and 61 at beta=3. This frontier captures **91–94% of the SUM of positive isolated-orbit gains** in this 20-site guide, despite not including most positive orbit IDs. One-hop accessibility alone does not mean a cheap algorithm knows which to choose.

## Comparators, cost protocol
- `walker`: verify only directly visited FN walker orbit(s).
- `random2`: two randomly selected legal Hamiltonian hops starting from FN walkers; verify the end orbit.
- `learned`: same two-hop exploration with a 9-feature geometry/history stochastic linear softmax policy, learned **on-policy** on *independent real FN trajectories*. Update with the **signed continuous frozen-amplitude gain**, positive or negative; keep=0. Fixed gradient clipping and `\epsilon=0.18`. No cheap proxy rewards and no ED labels.
- `tempered`: independent (a_{guide}^{0.5})-Metropolis search, where *all initialization and proposal-chain guide queries* are charged separately.
- Two cost regimes: (1) equal attempted candidate trajectories, and (2) equal **120 distinct terminal orbit energy verifications** per arm. The second is a better oracle-call comparison but **does not** equalize entire end-to-end cost. All training and FN generation query costs are recorded.
- Each learned arm spends **400 attempted training trajectories PER seed**, about 340 distinct scored orbits and ~9,940 distinct amplitude lookups (beta=.38, averaged). Random exploration has **no training cost**. Verification itself costs ~4,400–4,500 additional distinct amplitudes/arm/seed. None of this can yet be claimed to amortize across iterations or sizes.
- Six or twelve independently seeded train/FN/test blocks; train and evaluation FN walker streams use separate seeds but the **underlying sign chamber and amplitude guide are the same** and orbit identities can overlap. The policy is *not* evaluated on a changed FN sign iteration.

## Results (number of profitable **distinct** single-orbit proposals)

| FN projection setting | Seed blocks | Unique verifier calls/arm total | Walker | Random 2-hop | Continuous-energy learned | Tempered |
|---|---:|---:|---:|---:|---:|---:|
| beta=0.38, burn=.12 | 12 | 1,440 | 0 | **39** | 24 | 23 |
| beta=3.0, burn=2.5 | 12 | 1,440 | 2 | 27 | **49** | 23 |
| beta=4.0, burn=3.5 | 6 | 720 | 0 | **15** | 14 | 10 |

For beta=.38, alternate **equal 160 attempted paths**, 12 seed blocks: walker 0, random 43, learned 29, tempered 26.

At beta=3 the learned policy is ahead of random in discovered isolated positives and their combined *bookkeeping* sum, but at beta=.38 and beta=4 it does not win. Descriptive paired count difference at beta=3 is +1.83 profitable orbits per 120 verified calls per run (12 seeds, 95% Student-t interval [+0.48,+3.18]); the beta=4 six-seed result is -0.17 and very noisy. **Projection depth was investigated after observing the initial beta=.38 result; beta=3's apparent advantage is therefore exploratory and NOT a preregistered robust discovery.**

The fully correct test is still missing: **does the learned exploration enable better signed Krylov-verified updates and independently reprojected FN energies after subsequent FN iterations, per *amortized* cost?** No accepted joint sign function is produced here; scores are exact single-orbit frozen-guide energy gains. The active "verifier" in this toy is a 20-site exact boundary-energy oracle, **NOT the actual K1 sign decision**. It is essential not to conflate this with the goal.

## Project-lead interpretation
1. **Support bottleneck is real in the tested finite-beta walkers.** Valuable corrections often lie outside directly visited FN support.
2. **A local reachable frontier exists** and is highly enriched in potential single-orbit frozen-guide energy gain. Search *allocation* inside that frontier—rather than arbitrary farther hops—is the promising question.
3. **Current RL is not a compelling winner.** It sometimes identifies more positive orbits per charged terminal score (beta=3), but this reverses with FN projection settings, and its 400 training labels/seed make it more expensive overall. A fresh learner trained separately on each seed **cannot** establish amortization or scaling.
4. **Next discriminating experiment:** freeze a candidate-prioritization learner after training on earlier FN states, then use it without new training to allocate a *fixed number of actual K1 verifier calls* on a **changed** FN state; measure full joint update and independent subsequent FN energy. Compare to FN-rooted random one-hop and existing region-local Krylov `C_w` at matched total amplitude/bond queries. Reuse existing FN mixed-sampler and correction-level amplitude-ratio diagnostics from 2026-10-18, not another unconstrained 20-site test.

## Reproduction / code & raw data
- `avsl08_mixed_fn20.py`: FN engine adapter for tilted 20-site geometry and on-policy continuous-reward proposer, complete oracle audit, cost counters.
- `test_avsl08_mixed_fn20.py`: 4 new physics checks, checking (H_{xx}), tilted neighbors, exact signed local-energy row, M128 branch/reconfigure path, orbit symmetry. **21/21 related regression checks pass**, including independent full-H sign-energy identities.
- `avsl08_mixed_fn20_compact_results.json`: 12+12+12+6 raw-run summary, per-seed positive counts, exact query costs, matched differences and hashes.
- `raw/avsl08_mixed_fn20_allruns.tar.gz.b64`: all individual JSON outputs (12 short attempted, 12 short unique-budget, 12 beta3 unique-budget and 6 beta4 unique-budget), compressed and base64-encoded. To extract, decode and untar.
- Input amplitude/sign array `sign20_training_data.npz` (not committed; exact binary SHA-256 in summary), regenerated from `results/scalable_sign_2026-10-18/fn_collapse_exact20_krylov_v2.py` using the existing preparation script. DO NOT train on `psi0`/ED ground state.
- Original branch `avsl/experiment-05-onpolicy-costed-2026-10-19` / draft PR #1; all runs on isolated Mac scratch with source working tree untouched. **No 6×6/8×8 HPC or training job submitted.**
