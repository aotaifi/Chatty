# AVSL Experiment 04 — 20-site held-out sign-chamber transfer (2026-10-19)

## Question
Can a small learned candidate selector, trained on one sign chamber, **transfer without retraining** to another sign chamber and discover useful symmetry-orbit sign corrections with bounded local guide queries? This is a *contextual-bandit / active-learning candidate-ranking test*; it is **not** GRPO, DAPO policy-gradient RL, or a completed self-consistent FN–Krylov loop.

## Input and strict separation
- J1–J2 Heisenberg, J2/J1 = 0.5, tilted 20-site periodic torus, Sz=0 Hilbert dimension 184,756. Symmetry orbits use the same 20 translations and global spin inversion as the earlier 20-site experiments.
- **Training chamber A:** Marshall signs. **Testing chamber B:** signs after the previous energy-optimized Krylov update. Their sign tables differ on ~32.95% of full configurations (~1.26% exact ground-state probability mass). No training or reward labels from chamber B were used.
- A control held the **same post-Krylov amplitude fixed** in both chambers; the stronger follow-up used two distinct exact *FN amplitudes*. Marshall FN amplitude was freshly solved from the Marshall sign class with the current guide potential (E_FN = -10.3847165404; residual 6.81e-8); the post-Krylov FN amplitude came from the existing three-post-Krylov-iteration table. Therefore the stronger comparison changes **both signs and amplitudes**, without altering the lattice size.
- Train and evaluation *orbit keys* are disjoint under deterministic 20% holdout across both sign chambers, including over different simulated root seeds; no exact ground-state signs or ED amplitudes train the selector.
- Training rewards are `gain_a(g)=4∑_{x∈g,y∉g} H_xy s_x s_y a_x a_y` for a **whole translation + spin-inversion symmetry orbit g**. Gains are evaluated through explicitly counted calls to a precomputed exact-FN amplitude backend. No full-space scoring/ranking is made. Select candidates first from an amplitude-tempered Metropolis frontier and one-hop neighbors. The Metropolis chain is **not an FN/GFMC population**.
- Features: cheap bonds *observed along the frontier* (signed couplings and encounter counts), orbit spin geometry, representative amplitude and orbit size; no full orbit energy/neighbor scoring before candidate selection. The model is a 2-layer MLP 32→24 trained on binary `single-orbit gain > 0` targets. Keep/no-flip is zero and unprofitable labels are preserved.
- Baselines: same-pool random selection, raw signed-bond heuristic, and separately trained **geometry-only** MLP (all observed-bond features zeroed). Each arm audits B=22 held-out orbit candidates and greedily selects at most 12 of those if their **marginal** gains are positive, recomputing interactions between selected orbits.
- Equal **22 expensive orbit evaluations**, similar but not *exactly equal* number of unique amplitude queries; report measured calls. All mean energies below are normalized wavefunction expectation **gains** (positive = lower energy).

## Experiment design and costs
- Training: 12 independent exploration roots in Marshall chamber ×150 queried orbit labels = **1,800 reward labels**; 327 were beneficial with true Marshall FN amplitudes. Mean amplitude queries summed across the 12 training episodes = **about 129,000** (roughly 130k; not amortized), **far too expensive for a scalable claim**.
- Evaluation: 24 held-out independent Metropolis root seeds for each of three different MLP initializations (the same held-out roots reused; this is **24 roots, not 72 independent physical trials**). Test frontend has ~6,800 unique guide queries in total per arm including shared exploration/frontier; the model-specific extra queries vary modestly across arms.

## Fixed-a energy gains after transferring Marshall → post-Krylov

| Selector | Mean gain, three MLP initializations / common 24 test roots | Beneficial audited orbits, approximate mean |
|---|---:|---:|
| Learned full policy | 0.00003116 / 0.00003181 / 0.00004438 | 5.75 / 6.42 / 6.17 |
| Geometry-only MLP | 0.00001563 / 0.00002434 / 0.00002861 | 5.58 / 5.63 / 5.50 |
| Signed-bond heuristic | 0.00000690 | 0.125 |
| Random | 0.00001260 | 0.458 |

Across three initializations the full policy's *mean* gain is 0.00003578; the geometry-only control's mean is 0.00002286. Per-seed paired root-bootstrap intervals of **full minus geometry-only** are:
- seed 42: +0.00001552 [0.00000632, 0.00002617]
- seed 7: +0.00000747 [0.00000076, 0.00001432]
- seed 99: +0.00001576 [0.00000446, 0.00002955]
The bootstrap resamples 24 correlated walker-root tests; it is *conditional* on the guide, training data and policy seed, and cannot establish size generalization.

Matched-amplitude *isolation control* (same post-Krylov a for Marshall training and post-Krylov testing), held-out roots and same policy variants:
- Full learned: 0.00001905 / 0.00002548 / 0.00001656;
- Geometry-only: 0.00000151 / 0.00000087 / 0.00000210;
- Signed-bond heuristic 0.00000690 and random 0.00001260.
Thus observed signed-bond features can contribute incremental predictive value when crossing sign chambers, but these are still **oracle-amplitude small-basis tests**, not proof of useful actual walker exploration.

## Independent validation
- Reconstructed 117 randomly sampled symmetry orbits: both sign chambers and the amplitude tables are invariant within tested orbits.
- Direct full-Hamiltonian energy differences for 12 selected arms (three held-out test episodes × four selectors) agree with separately computed crossing-edge gain within 1e-9.
- Separate Marshall FN residual 6.81e-8 with norm-one positive vector; energy -10.3847165404.
- All scored gains refer to **fixed-guide** energy. No reoptimized FN solve after the policy update, no rollback under a learned inexact amplitude, no evaluation of out-of-pool generalization of the **sign net**.

## Interpretation, status, and strict caveats
**Encouraging, not promoted.** The learned policy beats geometry-only, signed heuristic and random in mean immediate frozen-guide energy gain on this genuinely different 20-site sign+amplitude chamber, across three seeds; this is evidence of local feature transfer. It does *not* demonstrate that a model will discover important corrections from realistic 128 FN walkers, nor that this gain survives FN projection, nor favorable scaling to 6x6/8x8.

The key risk remains that rewards were trained using globally computed FN amplitudes and expensive orbit labels. All local calls are counted, but producing the exact-FN amplitude oracle was excluded from cost. The augmented training set touches a substantial fraction of this small Hilbert space across roots.

**Next discriminator:** replace exact FN callable amplitudes by the 128-walker mixed-distribution guide; keep the distinct-chamber holdout, but evaluate the *same* learned policy against random / signed / geometry with a hard **total** query+walker-step budget. An independent post-hoc 20-site FN solve and direct physical Hamiltonian energy can check false positives. Introduce a *no-flip/rollback* control and report candidate-edge ratio error. Do not scale to 6x6 merely because this transfer test is positive.

## Reproducibility and source links
Original scripts/results from the conversation were bundled in `avsl_chamber_transfer20_repro.tar.gz` (conversation download artifact, not currently checked into GitHub):
- `avsl_chamber_transfer20.py` (shared-a isolation)
- `avsl_transfer20_fnchambers.py` (distinct FN amplitudes)
- `avsl20_geometry.py` (torus and amplitude-cached local sampling)
- `avsl_make_marshall_fn20.py` (separate FN amplitude solve)
- `avsl_transfer20_fnchambers_audit.py` (independent checks)
- JSON results: `avsl_transfer20_fnchambers_results.json`, `avsl_transfer20_fnchambers_seed7.json`, `avsl_transfer20_fnchambers_seed99.json`, `avsl_transfer20_fnchambers_audit_results.json`.
- Binary input `sign20_training_data.npz` can be recreated by `projects/j1j2/results/scalable_sign_2026-10-18/prepare_sign20.py`. The experiments currently expect files at `/mnt/data`. Reproduce in an isolated workspace.

No source code is claimed to be committed by this report unless separately uploaded and verified; do not lose the conversation archive before completing that upload.
