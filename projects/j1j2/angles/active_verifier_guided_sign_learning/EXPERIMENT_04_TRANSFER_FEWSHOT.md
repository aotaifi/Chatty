# AVSL Experiment 04 — 20-site transfer, opportunity ceiling and few-shot adaptation (2026-10-18/19)

## Status / scientifically valid claim

**Completed numerical benchmark** for sign-chamber transfer at 20 sites. A candidate-ranking network pre-trained on Marshall signs transfers poorly to post-Krylov signs, but a small number of *new-chamber* correction labels adapts the network somewhat faster than training it from scratch. This is an offline **fixed-amplitude** sign-ranking benchmark with an *exact FN amplitude oracle* used at candidate evaluation. It is **not** a demonstration of scalable on-policy RL, FN self-consistency or improved sign learning at 6x6/8x8.

## Setup and primary controls

- Square-lattice spin-1/2 J1-J2 at J2/J1=0.5, 20-site tilted torus, 184,756 Sz=0 configurations; whole symmetry-orbit flips, frozen common FN amplitude.
- Source sign chamber: Marshall. Transfer target: Krylov-updated signs. Candidate pools from the existing sampler. Training and held-out evaluation use non-overlapping orbit assignments.
- 22 expensive candidate-orbit evaluations per held-out test, 20 held-out trials, two independent neural parameter seeds where specified.
- Report *physical normalized fixed-a Hamiltonian energy differences*, not raw numerator gains; compare unchanged signs, random candidates, observed-signed-bond heuristic, geometry+amplitude features, learned bond-feature model.
- Exact FN amplitudes are available as a **callable oracle**, not a prediction from walkers. The counting excludes that oracle's underlying computation and amortized source-pretraining cost unless explicitly stated.

## Chamber-transfer failure and opportunity ceiling

| Method | Marshall mean frozen-a gain | Post-Krylov mean frozen-a gain |
|---|---:|---:|
| Marshall-trained bond-feature policy | 0.000249 | 0.00000104 |
| Geometry + amplitude network | 0.000029 | 0.00000177 |
| Observed-signed-bond heuristic | 0.000730 | 0 |
| Random | 0.000171 | 0.00000032 |

Post-Krylov has fewer individually profitable orbits (1,458 -> 201 across the complete relevant symmetry-reduced sector). However, an **oracle** that ranks the *same held-out candidate pool* by exact gain would achieve ~0.0000325 per 22 candidate checks, compared to ~0.00000104 for the transferred policy. Therefore difficulty increases, **but the learned transfer ranking is also genuinely weak**.

The oracle ranking scans held-out gains and is explicitly an *upper ceiling*, never an algorithm available to a scalable sampler.

## Adapting a source model to new-chamber labels

Existing Marshall source model pretraining: **1,980 oracle-labeled examples** (not included in target-label budget). Target-label samples were fixed by the same episode-level proposal process. Here ‘new labels’ means post-Krylov candidate gain measurements; it does *not* mean exact ED sign labels.

| Target labels | Mean target fixed-a gain |
|---|---:|
| Source model unchanged; 0 labels | 0.00000104 |
| Source model warm-start; 108 labels (4 positive) | 0.0000108 |
| Source model warm-start; 360 labels (**8 positive**) | **0.0000149** |
| Target model trained from scratch on 360 labels | **0.0000124** |
| Source model warm-start; 1,980 labels (49 positive) | 0.0000225 |

- Results reproduced with model seeds 19 and 29. These are paired held-out trials under two model seeds, not 40 independent physics systems.
- Warm-start at 360 target labels outperformed scratch by ~20% on the benchmark metric, a **modest** advantage; it does *not* include source pretraining cost.
- The 108-label arm uses ~4 positives; 360 has only 8, so variance and overfitting risks are severe.
- A single mixed-chamber model trained on 990 Marshall and 990 post-Krylov labels improves target ranking versus Marshall-only training, but does not beat chamber-specific training.
- Direct independent full-H comparisons for 64 few-shot sign updates agreed with the crossing-edge energy formula to maximum absolute error **2.57e-14**. This is a formula audit, not proof that the guide generalizes.

## Interpretation, explicit non-claims

1. The *location and value* of rare profitable corrections depend strongly on the current sign chamber. A fixed policy can become obsolete after Krylov.
2. A few current-chamber labels may allow meaningful policy adaptation; this is analogous to DAgger/on-policy collection and active supervised ranking, **not** yet GRPO/DAPO training.
3. The geometry-only shortcut persists, so improvement from observed-bond features remains **unproved** as a transferable mechanism.
4. Source-pretraining and exact-amplitude query costs are omitted in the attractive few-shot numerator. No favorable end-to-end scaling is yet established.
5. Single-step frozen-a energy improvement is not the same as improvement after full FN amplitude relaxation or after replacing an explicit orbit flip by a sign NN that generalizes to untouched states.
6. Learning profitable candidates is not the same as discovering rarely proposed high-impact states: proposal support is still fixed by the walker/neighbor explorer.

## Next *discriminating* experiment (do not start a broad campaign)

Before larger CNNs or GRPO policy gradients, repeat **one source-to-new-chamber adaptation round** with a fixed 128-walker budget, a **callable imperfect guide**, and **total evaluation costs**. Score:

- (a) opportunity recalled by the candidate proposal support before ranking;
- (b) verifier precision/false-positive physical energy, ideally with uncertainty controls;
- (c) ranking advantage over geometry-only and signed-bond at equal **total** training + new-label + query cost;
- (d) downstream energy after accepting a *joint* symmetry-respecting sign update and rerunning FN; include keep/no-flip;
- (e) size-consistent local defect markers / edge-relative-sign error, not a global Krylov threshold.

**Stop rule:** if ranking needs exact amplitude queries or a geometry-only model matches it, do not promote it as scalable active discovery. If discovery support misses most post-Krylov oracle opportunities, improve exploration before optimizing the neural ranker.

## Reproduction / raw data

All archived in:
`projects/j1j2/experiments/active_verifier_guided_sign_learning/`

- `avsl_transfer20.py`, `avsl_transfer20_results.json`: source-to-target benchmark.
- `avsl_transfer20_validate.py`, `avsl_transfer20_validation.json`: full-H verification of initial transfer.
- `avsl_transfer20_opportunity_audit.py`, `avsl_transfer20_opportunity_audit.json`: opportunity counts.
- `avsl_transfer20_oracle_ceiling.py`, `avsl_transfer20_oracle_ceiling.json`: within-pool ranking ceiling.
- `avsl_transfer20_pooled.py`, `avsl_transfer20_pooled_run.py`, two seed result JSONs, validation code/results: mixed-chamber training.
- `avsl_transfer20_fewshot.py`, `avsl_transfer20_fewshot_results.json`, validation code/results: target-budget warm start against scratch.

Some scripts contain original `/mnt/data` absolute paths and require the 20-site `scalable_sign_2026-10-18/` data-generation stage. Run in a clean scratch workspace; do **not** execute against the dirty Mac checkout without reviewing paths and dependencies.

*Research decision:* preserve the numerical evidence, but don't infer that a successful 4x4 selector has solved 20-site transfer or that a 20-site warm start proves 6x6 scaling.
