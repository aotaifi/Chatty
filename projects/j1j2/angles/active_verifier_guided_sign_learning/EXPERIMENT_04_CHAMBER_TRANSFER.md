# AVSL experiment 04: 20-site sign-chamber transfer and opportunity audit
Date: 2026-10-19. **Diagnostic, not demonstration of scalable learning.**

## Question
A policy trained to prioritize energy-profitable symmetry-orbit sign corrections in the Marshall chamber was tested without retraining in a post-Krylov chamber. Are tiny post-Krylov gains caused by a lack of profitable corrections or poor policy transfer?

## Controlled transfer protocol
The same 20-site FN amplitude vector is deliberately held fixed in both sign chambers (Marshall and the guide after the Krylov update). Root configurations use toy Metropolis sampling from a^1, plus one-hop neighbors; this is **not** the self-consistent FN mixed-walker distribution. Candidate orbits are held out by a stable hash split; 22 orbit scores are queried per test trial. The training policy was fit on 1,980 unique labeled symmetry orbits out of 4,654 total (~42.5%), over 18 training episodes. Its training labels are whether an **individual full symmetry-orbit sign flip** improves fixed-amplitude energy, *not* the Krylov sign target. Full a(x), current signs and basis membership are oracle lookups for queried configurations, not callable learned NQS functions. The model ranks discovered orbit candidates; it does not generate genuinely new multi-step paths.

Independent direct-Hamiltonian checks agreed with the orbit crossing-edge energy formula within about 2.4e-14.

## Original held-out transfer results (20 seeds, 22 evaluated orbits per test)
Mean *normalized fixed-a energy gain* (positive = beneficial):
| Ranker | Marshall | Post-Krylov |
|---|---:|---:|
| Learned bond-feature policy, initialization seed 19 | 0.000249136 | 0.000001036 |
| Geometry plus amplitude network, seed 19 | 0.000028618 | 0.000001774 |
| Observed signed-bond heuristic | 0.000729884 | 0 |
| Random | 0.000170509 | 0.000000320 |

**Initialization sensitivity is serious.** A second existing initialization (seed 29; same test seeds) yielded learned gain 0.000700866 on Marshall and 0.000002283 post-Krylov. Thus the apparent Marshall underperformance by ~3x for seed 19 is not stable across just these two seeds. Neither learned seed establishes robust, competitive cross-chamber transfer.

## New full-orbit opportunity audit (evaluation oracle, not an allowed policy)
Enumerating every symmetry orbit with the same fixed a to measure how many *individually profitable flips* exist:
- Number of orbits: **4,654**.
- Marshall: **1,458** individually profitable orbits (~31.3%).
- Post-Krylov: **201** (~4.3%).
- Sum of positive *individual* normalized gains: Marshall 0.02280516; post-Krylov 0.00080016 (~28.5x smaller). **These sums cannot be realized simultaneously** because sign-orbit flips interact; they are diagnostic opportunity measures, not achievable total energy drops.
- Mean discovered frontier from 140 walker roots + one hop: ~2,341 orbits, representing ~429 beneficial Marshall orbits or ~57.6 beneficial post-Krylov orbits.
- Held-out one-fifth of that frontier contains ~78.3 individually profitable Marshall orbits and ~10.4 post-Krylov orbits; on average only about 5% of all profitable orbits appear in each hold-out pool.

**Oracle 22-evaluation ceiling:** Rank the *same held-out pool* using exact frozen-a single-orbit gains of every pool member (a deliberately expensive hindsight oracle), then greedily accept the best 22 candidates with interaction corrections:
| Chamber | Oracle hindsight ceiling mean gain | Learned seed-19 gain | Learned / oracle |
|---|---:|---:|---:|
| Marshall | 0.00255388 | 0.000249136 | 9.6% |
| Post-Krylov | 0.0000325237 | 0.000001036 | 3.5% |

This proves **both** effects: (i) Krylov leaves many fewer profitable corrections and (ii) the current ranking policy captures little of their *remaining* useful gain at equal 22-score budget. The hindsight oracle violates the deployment cost by scoring the whole holdout pool; it is only a ranking-performance reference, not a runnable competitor at equal cost. A full-space discovery oracle has a still larger cost.

## Next discriminator (running separately)
Train **directly in the post-Krylov sign chamber** using the identical 1,980-orbit label budget, holdout split, features and 22-query evaluation protocol. Compare in-chamber policy to Marshall-trained zero-shot policy across **multiple initializations**. If in-chamber policy is still poor, blame features, reward imbalance and opportunity sparsity, not only distribution transfer. If in-chamber policy succeeds, adapt via versioned replay / active continual learning rather than hoping a static correction classifier transfers.

## Important scaling failures/limitations
- 1,980 expensive training labels are **42.5% of the complete 20-site orbit space**; training data acquisition is not favorable scaling.
- Per-episode frontier evaluation touches many configurations. Evaluating all candidate amplitudes is already expensive and needs to be counted in future comparisons.
- Frozen amplitudes are not a self-consistent new FN solution in the other sign chamber.
- Exact FNs are not used for target ED labels; however, **the fixed-a amplitudes themselves remain globally stored** and accessible on demand.
- Maximizing number of beneficial flips does not maximize energy improvement. Maintain a zero-action baseline, fixed physical cost, full-orbit symmetry and interaction-aware greedy updates.
- We have **not** learned a trajectory exploration policy; the current net is a candidate ranker. No GRPO/DAPO reinforcement-learning success claim.

## Paths
Source and archived raw results:
`projects/j1j2/experiments/active_verifier_guided_sign_learning/avsl_transfer20.py`
`.../avsl_transfer20_results.json`
`.../avsl_transfer20_validate.py`
`.../avsl_transfer20_validation.json`.

Evaluation audits were generated as `avsl_transfer20_opportunity_audit.py`, `avsl_transfer20_oracle_ceiling.py` and their JSON outputs; attach/commit alongside this report for exact reproduction. Both enumerations are strictly **post-hoc oracle audits**.

**Decision:** Do not scale this static learned ranker to 6x6. Separate rare-candidate support, reward signal quality, and policy transfer; evaluate actual on-policy acquisition next.
