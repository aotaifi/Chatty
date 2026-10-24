# 20-site chamber transfer / few-shot adaptation verdict — 2026-10-24

**Status:** a useful diagnostic; NOT a demonstrated scalable AVSL algorithm. Reproduces the 2026-10-19 runs after tool interruption. This note distinguishes transferred physics, fixed-a ranking, label cost, and exploration cost.

## Why this experiment
A first learned 4x4 ranking rule exploited spin geometry and showed no strong observed-bond-specific benefit. We therefore trained a 20-site ranking model in one real sign chamber (Marshall) and tested *without retraining* after the Krylov sign update. A source-to-target transfer is valuable only if it improves **energy recovered per realistically expensive query**, not just accuracy on a tiny exact basis.

## Benchmark / control
- Tilted 20-site periodic J1-J2 at J2/J1=0.5, S^z=0; 184,756 spin configurations, 4,654 translation+spin-flip orbit keys.
- Chambers: (i) Marshall and (ii) post-Krylov sign, at the **same frozen exact FN amplitude vector**. Consequently this is *sign-chamber shift only*, not true FN-relaxed transfer.
- Pool: common amplitude-tempered Metropolis roots and one-hop neighbors (toy walker, **not GFMC**); orbit-held-out 20% deterministic split.
- Each test episode can verify 22 candidate orbits, then greedily combine up to 12. Direct Hamiltonian energy audits agree with stored crossing-edge gains at ~3e-14.
- Targets for policy training: the **binary indicator of positive frozen-a single-orbit energy gain**, NOT the Krylov sign labels or exact ground-state signs. An exact FN amplitude array and signs are available as callable *on-demand* lookup backends, hence this is **not yet oracle-free**.
- Training labels: 18 episodes × 110 = 1,980 unique source-chamber orbit score queries. The original Marshall training frontier queried 220,798 amplitudes **summed across episodes**, with repeats across episodes (not 220,798 globally unique). That accumulated figure already exceeds the full 20-site basis dimension. Even a seemingly good candidate ranker has not established query/sample efficiency.

## Previously audited transfer result (20 held-out root episodes, 22 verification labels per test)
| Ranker | mean frozen-a gain Marshall | mean gain post-Krylov |
|---|---:|---:|
| CNN trained Marshall on bond + geometry features | 0.000249136 | 0.000001036 |
| CNN trained Marshall on geometry+amplitude only | 0.000028618 | 0.000001774 |
| Observed signed-bond heuristic | 0.000729884 | 0 |
| Random orbit screening | 0.000170509 | 0.000000320 |

Thus Marshall-trained policy has poor zero-shot transfer. Post-Krylov has fewer profitable orbit flips than Marshall (201 vs 1,458 global single-orbit opportunities), so opportunity is genuinely depleted, **but** the target heldout candidate pool still has an independently calculated oracle ceiling of 0.000032524 for 22 evaluations. The zero-shot CNN reaches ~3.2% of that mean ceiling.

## Training and adapting on the target chamber
Training a fresh policy on 1,980 post-Krylov labels raises post-Krylov energy gain to 0.000021734 (network seed 19); geometry+amplitude features alone do **better**, 0.000028558. Another seed produced about 0.000022735 (bond) vs 0.000025305 (geometry); this is still evidence against an observed-bond-specific feature benefit.

Pooled-chamber CNN (990 source and 990 target labels, 1,980 total labels) achieves post-Krylov mean gains:
- seed 19: 0.000012687 (bond-inclusive) versus 0.000005800 (geometry+amp).
- seed 29: 0.000011297 (bond-inclusive) versus 0.000002807 (geometry+amp).
Pooled bond inputs may help at mixed-chamber training, but neither beats full target-only retraining at the same label count; the common frontier and source-training expenses must be counted separately. A direct-H pooled validation checked 160 updates: max error 3.17e-14.

**Few-shot same-chamber adaptation on Marshall-initialized policy**, 2 net seeds, 20 heldout frontier episodes:
| New target labels | Positive target labels | Mean gain warm-start | Mean gain from scratch |
|---:|---:|---:|---:|
| 108 | 4 | 0.000010844 | 0.000010500 |
| 360 | 8 | 0.000014909 | 0.000012393 |
| 720 | 12 | 0.000015316 | 0.000013169 |
| 1,980 | 49 | 0.000022476 | 0.000022284 |

The 360-label warm-start yields about a 20% relative gain over scratch in this restricted setup. **It is not a 20% reduction in end-to-end compute**, because source-model pretraining and generation of 18 candidate frontiers were already paid for. The 64 independent direct-H checks for the few-shot solutions agreed to max 2.56e-14.

## Independent opportunity audit
- Marshall: 1,458 globally positive single-orbit frozen-amplitude flips; post-Krylov: 201.
- Conditional on the same heldout/frontier pools and 22 expensive verifier evaluations, post-Krylov **oracle** mean frozen-a energy gain is 0.000032524; these oracle rankings require full candidate scoring and are exclusively retrospective ceilings.
- The post-only geometry model achieves ~88% of this conditional ceiling, but no full-space FN reoptimization, walker-only candidate generation, or sign-network generalization was tested.

## Interpretation and falsifiers
1. **Evidence for chamber adaptation, not zero-shot transfer.** The required policy is conditional on the *current sign chamber*. Replay/few-shot updates may help, but must not fossilize earlier mistaken signs.
2. **Geometry shortcut persists.** Bond-aware learning is not yet better than geometry-only controls in a target-specific setting.
3. **Not yet RL:** an offline supervised positive/negative ranking experiment; no group-policy gradient, no on-policy policy-driven frontier and no end-to-end FN/Krylov loop.
4. **Not a scalable amplitude method:** all reward labels use exact amplitudes via an on-demand oracle. 18 frontiers require >200k cumulative amplitude lookups despite only 20 sites; gain-per-unit work is unproven.
5. **No claim of size consistency:** global proposed-orbit energy ranks and frozen-guide amplitudes do not establish intensive energy scaling.
6. **Data leakage control is orbit-based**; nearby configurations and Hamiltonian neighbors remain highly correlated between splits, so the 20 held-out trials are not statistically independent lattice problems.

## Recommendation
**Stop tuning another 20-site oracle classifier.** The next decisive experiment must be *on-policy and walker-accessible*: policy chooses exploration frontier under fixed 128-walker FN budget; compare tempering+one-hop, signed-bond heuristic, pretrained/adapted policy and geometry-only network at the **same total amplitude/sign/edge requests, expensive verifier labels, and optimizer steps**. Require independent true frozen-a AND reoptimized FN energy audits on small sizes. Report useful new correction orbits / cost, heldout sign-network performance and intensive energy per site, with paired seeds. If the learned policy does not win at equal total cost, archive AVSL learned-frontier direction as an unsuccessful route.

## Reproduction
Scripts/results in `projects/j1j2/experiments/active_verifier_guided_sign_learning/`:
- `avsl_transfer20.py`, `avsl_transfer20_results.json`, `avsl_transfer20_validate.py`
- `avsl_transfer20_opportunity_audit.py`, `avsl_transfer20_oracle_ceiling.py`
- `avsl_transfer20_pooled_run.py`, `avsl_transfer20_pooled.py`, `avsl_transfer20_pooled_validate.py`
- `avsl_transfer20_fewshot.py`, `avsl_transfer20_fewshot_validate.py`
The existing scripts assume `/mnt/data/sign20_training_data.npz`, use NumPy, PyTorch and `avsl_transfer20.py` with the prepared exact 20-site benchmark. The few-shot script requires source weights `avsl_transfer20_results.pt`, generated by running the original source train command first. Pooled script is self-contained generated copy of source with only the label distribution changed.
