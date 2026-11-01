# Experiment 05 preregistration — on-policy, costed AVSL adaptation
Date: 2026-10-19. Status: **DESIGN ONLY; no physical simulation or training launched.**
Parent: `angles/active_verifier_guided_sign_learning/HANDOFF_2026-10-19.md`.
Relevant evidence: `angles/active_verifier_guided_sign_learning/EXPERIMENT_04_CHAMBER_TRANSFER_ADAPTATION.md`, `results/scalable_sign_2026-10-18/WALKER_ONLY_NCE_2026-10-18.md`, `DESIGN_scalable_loop_2026-10-13.md`.

## Decision and falsifiable question
**TEST** a learned *exploration/allocation* policy, not a larger full-space sign classifier or GRPO implementation. Can adaptation in a genuinely **new, self-consistently sampled FN sign chamber** recover more **independently verified post-FN energy improvement per end-to-end compute** than a local Krylov-region rule and an equally costed geometry-only explorer?

If proposal support, trustworthy verification, or size-consistent generalization fails, stop **before** training a composite-sign network. A positive 20-site offline classification result alone is insufficient.

## Existing evidence and interpretation
- 20-site post-Krylov frozen-a single-orbit opportunities: 201/4,654, vs 1,458/4,654 in Marshall. Hindsight top-22 in held-out pools: ~3.25e-5 mean gain; Marshall-pretrained policy gets 1.0363e-6. The hindsight ranking uses unavailable global labels.
- 360 target labels: warm-start 1.49094e-5 vs scratch 1.23926e-5; geometry-only warm 1.42148e-5. Only 8 target positives; two NN seeds and shared 20 test pools. No evidence of statistically independent across-chamber amortization.
- Optimistic **label-only** break-even if each subsequent chamber preserves this same mean gain and consumes 360 labels: 1980 / [360*(1.49094e-5/1.23926e-5 - 1)] = 27.08, or **28 future chambers**. This is not a forecast: it ignores proposal/amplitude/model cost, target changes and diminishing opportunities. The geometry-only comparator may dominate.
- Test-frontier cost already measured: ~134.2 distinct roots; 2,340.85 orbit candidates; 4,395.05 unique amplitudes after frontier; 9,224.4 after **all** bond features; ~15,834.2 sign lookups; ~655 further unique amplitudes for 22 candidate scores. Source training episodes recorded ~12,000 unique amplitude cache entries each (18 episodes). These are **per-episode unique** counts, not globally unique across episodes or hardware latency.
- Earlier failures constrain this design: FN collapse/suppression is not a sign label; K2 may worsen energy; walker-MLE/NCE local ratios can fail exactly on correction edges; independent log-amplitude noise caused false positives; 20-site learned amplitude sign gates became harmful beyond a few flips; 8x8 walker-MLE unseen-neighbor ratios drifted far more than seen ones. Never use ED signs or the full exact FN vector as training labels/backends.
- A frozen-a energy improvement, for a *precisely defined callable amplitude a*, is a legitimate variational-energy statement for `a*s`. It does not certify improvement at `phi_FN[a,s]` or after amplitude reoptimization. Calling a learned approximation to FN amplitudes a 'verifier' of true FN progress is incorrect. Independent sampling is required to estimate the former without selection bias; a fresh fixed-node evaluation is required for the latter.

## Physical protocol (one chamber at a time; no blind expensive runs)
**Stage 0 — API and latency preflight (cheap; no training).**
1. Reuse the **callable** `SquareJ1J2` and `CachedAmplitude` interfaces in `experiments/callable_fn_loop.py`; use the 128-walker FN engine as reference for actual walker histories (`experiments/gfmc_6x6_krylov_M128_gr.py`). Audit spin sector, symmetry, time weighting, population control, independent RNG streams and neighbor enumeration.
2. Require `log_a(x)`, `sign(x)`, `neighbors(x)`, `FN_walker_snapshot()` to operate without a `2^N` lookup, full `S^z=0` vector, exact FN eigenvector, or cached global symmetry-orbit membership. A bounded `O(N)` orbit expansion for a candidate is permitted, but must be charged.
3. Measure a realistic 8-label/small-frontier **latency calibration** at fixed device, batch size, walker sweep count and signs. Do not set an arbitrary equal-count query budget as a substitute for measured cost.
4. Verify invariants on a tiny toy graph / 20-site hold-out: exact pairwise fixed-a crossing-energy identity; joint flips are not additive; independent VMC sign-update estimator agrees with direct full-H; translation/spin-flip symmetry; changing the local rule in an untouched decoupled block does not change another block's local decision. Full-space exact arrays, if used, are **audit-only** behind an inaccessible evaluator boundary.

**Stage 1 — single real FN chamber, capped scout (no policy NN update).**
1. Start with a callable `a_t(x)>0` and composite `s_t(x)` on a fresh post-Krylov/FN chamber. Unlike Experiment 04, do **not** artificially hold a single exact amplitude table fixed across different sign chambers. Freeze `(a_t,s_t)` during a scout; FN mixed walkers have distribution `f_t ∝ a_t*phi_FN[a_t,s_t]`, **not** `a_t^2`. Use **M=128** as the existing pilot setting, record decorrelated walker histories.
2. For each observed walker root, consider legal J1/J2 one-hop neighbors, then **at most two-hop** bounded exploratory rollouts when latency allows. Record proposal probabilities, path, encounter count, symmetry representatives, and first-seen state. No all-orbit enumeration during scouting. Share seeds for paired comparisons, but run **independent per-arm query caches and counters**.
3. Compare at equal *calibrated total compute*, not simply 22 or 64 labels: (A) signed-boundary prioritization; (B) geometry-only ranking; (C) frozen transferred policy; (D) uniform/tempered-neighbor explorer; (E) nonlearned size-consistent `C_w` region-Krylov challenger; (F) **keep**. The trained-from-scratch policy joins only Stage 2 when enough genuine target labels exist.
4. Stage-1 ceiling: **64 scored candidate proposals across a chamber per arm**, **2 joint accepted changes maximum**; stage-0 hardware calibration sets a separate time/energy cap, with early abort on excess. The '64' is a protocol maximum, not a promise of positive examples. Retain **all** negative/no-flip labels and costs.
5. Record *support before ranking*: newly discovered candidate motifs, unique roots/bonds, distinct orbit-equivalent proposals, scoreable fraction, repeat discovery across independent walker populations, and verified positive yield **per total CPU/GPU second**. On 20-site evaluation-only audits, measure missed full-space opportunities; **never** require this observable at 6x6/8x8.

**Stage 2 — conditional on Stage 1, active target adaptation.**
1. If the baseline explorer shows repeatable positive, confidently assessable outcomes and manageable local-ratio cost, collect on-policy, actively chosen target-chamber labels with checkpoints at `B_target={64,192,360}` (cumulative **attempted**, not only retained positive, evaluations). Retrain a small explorer/ranker, retaining negative/keep anchors and versioned replay. Compare: frozen-source, warm-start + fresh data, **scratch on identical fresh data**, geometry-only with its cheaper real feature cost, signed-bond and `C_w` handcraft, keep. No GRPO relative-rank reward on all-negative batches; if a policy-gradient experiment is later justified, use **keep-relative** rewards and charge discarded proposals.
2. Use within-chamber trajectory episodes, with behavior-policy version, sampling propensity and resulting action logged; train only on earlier episodes. Report *held-out walker populations and held-out **later FN chambers***, not just disjoint labels/orbits from the same frozen chamber.
3. **Exploration first.** Only after this wins may a callable new composite sign model `s_{t+1}(x)` be distilled from the approved local rule and old-guide keep anchors. **Do not** train on ED ground-state signs, use an unconstrained global K2 threshold, or stack independent 'correction factor' NNs. Audit unrequested changes in both seen and fresh neighbor states; roll back if distillation harms paired physical energy. Cached orbit flips alone cannot establish intensive progress as N grows.
4. Advance on policy only after accepting a joint new sign rule and rerunning FN for the same amplitude backend; version the new chamber. Maximum one accepted chamber transition for initial pilot. Repeatable gains on fresh chambers, not extra SGD epochs, are the target.

## Candidate unit and size-consistency
At 20 sites, retain *single symmetry orbit* as a code regression/label-scoring comparator only. For physical scalability, a learned explorer should nominate **local windows/defect components or parameterized local rules** that generalize across many configurations, with a callable translation/spin-flip-equivariant `eta(x)∈{±1}` and `s'(x)=s(x)*eta(x)`. Include connected-components-of-marked-windows `C_w` from 2026-10-13 as a hard baseline, with the **decoupled-block parity consistency test** and size-normalized features (local per-bond excess, not extensive global fields). False merges, overlapping-window parity errors and unseen-neighbor cost must be audited. A local rule affects many configurations: evaluate it **jointly**, not by summing scores of sampled single-orbit flips.

## Independent verification and energy hierarchy
For **the same frozen callable amplitude** `a_t`, write `eta(x)=s'(x)/s_t(x)`. With `q_t(x)=a_t(x)^2/Z_t` and symmetric real H,
```
D_fixed := E[a_t*s_t] - E[a_t*s']
         = E_{x~q_t} sum_{y!=x} H_xy * a_t(y)/a_t(x)
                * s_t(x)*s_t(y) * [1 - eta(x)*eta(y)].
```
Estimate `D_fixed` with *independent* `a_t^2` VMC samples, paired old/new local energies, decorrelated/block errors and the **actual joint sign function**. The FN mixed `a_t*phi_FN` population cannot replace `q_t` without controlled reweighting. Any finite-sample energy selector is optimistic on its own selection sample: reserve fresh independent verification chains and apply a one-sided, multiple-test-aware confidence rule. Do not promote marginal `D_fixed>0` based solely on approximate local neighbor-ratio scores.

Energy ladder, explicitly separate:
- **screen**: cheap observed bonds/local features; not physical certification;
- **select**: local callable-ratio Krylov/energy proxy and keep-relative labels; not FN certification;
- **verify**: independent paired `D_fixed` for the *exact candidate callable guide*; report uncertainty and false-positive rate;
- **accept**: independently estimated `D_FN=E_FN[s_t,a_t]-E_FN[s',a_t]` from reprojected FN walkers, including independent replica uncertainty / population bias; if inconclusive, **keep**;
- **generalize**: recheck `D_fixed` and `D_FN` on the distilled sign function, not only explicit flipped samples.

Use a guarded batched/rollback action (initial cap two), no 'least-bad negative flip' reward. The final metric is *energy improvement per site after FN, relative to keep*, at matched total budget. If none is credible, record zero and halt.

## Cost accounting — each arm pays its own costs
Implement an append-only JSONL event ledger. Required per event: `run_id, code_commit, model_hash, chamber_id, t, phase, arm, walker_seed, proposal_seed, training_seed, device, event_type, wall_s, cpu_s, gpu_s, n_walker_steps, n_neighbor_hops, n_guide_model_calls, n_guide_unique_states, n_sign_model_calls, n_sign_unique_states, n_orbit_members_expanded, n_feature_edges, n_verifier_candidates_attempted, n_vmc_sweeps, n_fn_reprojection_steps, n_optimizer_steps, n_replay_examples, cache_hits, cache_misses`. Use **accumulated increments**, and also publish distinct-state **per-run** cache keys to detect double counting. Include cache construction, burn-in, hardware warm-up (reported separately), failed proposals, rejected dynamic groups, teacher generation, all NN fits, independent verifier, FN re-projections, rollbacks, and data preparation.
- Record **wall time plus CPU core-hours and GPU-hours**, numerical neighbor probes, distinct amortized model evaluations and peak memory. Batch size/model inference latency matters; raw 'unique amplitudes' is not a hardware-independent currency.
- Pretrained source policy: publish (i) *marginal cost* given a pre-existing checkpoint; (ii) **fully loaded** cost including all source labels/exploration/model training, amortized over `K={1,4,16,32}` accepted *independent* chambers. Scratch and geometry-only arms must receive the same total budget, including any funds transferred from absent pretraining.
- Primary Pareto: net FN energy gain/site versus **total GPU-equivalent expense (or measured hardware time on the same host)**. Also publish label/proposal efficiency and event-level raw costs. Do not assert transfer wins based on verification-count matching.

## Holds, failure gates, decision
- **HOLD** if the callable-guide API accesses a global amplitude/sign vector, if symmetry/signed-edge identities fail, or if query costs explode at two-hop expansion.
- **HOLD** if independent sign-energy verification cannot separate gain from zero or if accepted updates become adverse under independent FN projections. A small number of promising singles cannot justify a joint rule.
- **REJECT benefit of NN transfer** unless warm-start beats equal-total-cost geometry-only, scratch and the handcrafted `C_w` baseline on independent *new FN chambers*, with uncertainty reported. If not, retain a cheap on-policy explorer.
- **REJECT scalable sign claims** unless the decoupled-block test passes exactly, edge-relative-sign defect metrics improve where reference signs are available, and end-to-end gain **per site** does not collapse from 20 sites through 6x6. 8x8 only after a successful, budgeted 6x6 gate.
- **STOP training** when candidate support is the bottleneck: train/change the bounded explorer, not the ranker. A null scout at B=64 is **inconclusive about nonexistence** of rare sign defects, but sufficient to stop speculative neural training.

## Reproducibility / implementation tasks (NOT YET IMPLEMENTED)
1. Add a walker-accessible `Backend` adapter for `callable_fn_loop.py` and actual 128-walker FN; avoid importing `avsl_transfer20.py` in production because it loads `sign20_training_data.npz` into global arrays at import time.
2. Introduce budget ledger and independent per-arm caches before rerunning policy baselines. Explicitly report feature cost: in `avsl_transfer20.py`, `local_features` is computed for the entire pool *before* ranking; that invalidates equal end-to-end cost comparisons.
3. Implement joint local-rule VMC energy difference and regression test against direct H on the bounded 20-site referee. Implement decoupled-block `C_w` unit tests.
4. Freeze named configs/RNG streams, source commit/checkpoint SHA, optimizer/version, executable command, JSONL events, raw accepted/rejected candidates and measured wall GPU/CPU spend *before* large runs. Preserve all results and negative runs in GitHub.
5. Prepare a short readout answering: does **support** exist? are predictions **verifiable**? does **NN exploration** win at equal fully loaded cost? does the **local sign rule** remain size-consistent? Do not run the 6x6/8x8 training stage until all gates are visibly passed.

**Status at preregistration:** Literature-inspired on-policy verifier-guided discovery remains a hypothesis. The only completed new result here is a reanalysis of existing archive and a fully costed, falsifiable protocol; no fresh numerics.

## Infrastructure progress on this branch (no new physics result)
Two standalone, pure-Python files are now included:
- `avsl_onpolicy_core.py`: explicit per-arm metered callable caches and append-only incremental JSONL cost events; paired same-amplitude Hamiltonian energy-difference estimator, **requiring independent a² chains**; marked-window overlap component combiner.
- `test_avsl_onpolicy_core.py`: eight deterministic tests covering independent joint-update Hamiltonian energy identity, nonadditivity, no-flip/global-sign invariance, FN-mixed sampling rejection, exact uniform q audit, component merge/parity and ledger counters.

The tests prove only mathematical/software invariants on a small graph, **not** an unbiased stochastic estimator on actual correlated walker histories, independent FN energy improvements or size consistency of a fully constructed local-window rule. The callable frontends deliberately contain neither ED signs nor global amplitude tables.

**Unimplemented:** isolated 128-walker FN adapter, an independently seeded equilibrated a² VMC sampler, adaptive proposal runner, actual C_w region-excess predicate, empirical ratio-error calibration, fully timed per-arm baselines, and independent reprojected FN energies. Consequently no new transferable policy or acceptance decision is claimed. Stage 0 latency/performance calibration is the next permitted run, then Stage 1 only after the resource guard is verified.
