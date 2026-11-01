# AVSL-08 — Actual FN-mixed walker support and rare-correction exploration
**2026-10-19 · 20-site tilted square-lattice J1–J2 at J2/J1=0.5 · preliminary research result**

## Research question
Can learned exploration from **genuine fixed-node mixed walkers** find rare, energetically useful sign corrections that walkers themselves miss, more efficiently than unlearned exploration? The sign-updater/FN–Krylov loop is the objective; this test isolates *exploration support*, not a replacement ground-state sign learner.

## What was implemented

The existing FN sign field `s_k` and a **positive but unconverged** post-Krylov guide amplitude `a_k` come from the archived 20-site experiment. We implemented a continuous-time branching fixed-node projector, not a Metropolis surrogate:

- For a Hamiltonian edge `H_xy=h>0`, a jump `x→y` is allowed only if `s(x)s(y)=-1`, with importance rate `k_xy=h a(y)/a(x)`.
- The local importance potential is `E_L(x)=H_xx + Σ_y h_xy s(x)s(y)a(y)/a(x)`. Gillespie jumps are exact in continuous time inside each interval; Feynman–Kac weights `exp(-∫ E_L dt)` are incorporated by systematic population resampling.
- FN walker distribution tends toward `f(x)∝a(x)φ_FN(x)`, **not** the older Metropolis toy law `a(x)`. Finite populations have correlated ancestors and population-control bias.
- Six **independent** heldout mixed-FN root populations, M=128 each, `β=10`, burn-in `β=4`, `dt=0.15`, recording every four resampling intervals (1,280 recorded snapshots per population). A separate real mixed-FN population provides RL training roots. **The lookup guide and exact φ are precomputed 20-site oracle arrays—not a callable learned guide or large-N solver.**

The policy is the same minimal geometry-only 9-feature, two-hop stochastic explorer as AVSL-06. **Its input never contains exact FN amplitudes, exact signs of new configurations, or exact gain labels**; those are behind the charged verifier. It trains on **400 attempted** proposals from a separate true FN walker population, using signed continuous rewards `tanh(ΔE_fixed-a/1e-5)` (including negative outcomes). Terminal energy verification for a candidate flips one complete translation/complement symmetry orbit. This is a physically exact **frozen-guide single-orbit gain**, **not yet K1 verification**, a combined callable sign correction, or a reprojected FN energy decrease.

## Physically validated distribution

- Exact 20-site FN reference energy: **−10.417697678503282**.
- Four independent M=128 projected walker estimates of the corresponding mixed-distribution local energy: **−10.41325366, −10.42351867, −10.41272314, −10.41697206**. Mean: **−10.41661688**. Per-run deviations are compatible with noisy finite populations; we do not claim precise convergence.
- Independent tests check rates, reverse detailed balance, the exact FN diagonal/offdiagonal operator identity, global sign invariance, energy identity `〈E_L〉_{aφ_FN}=E_FN`, genuine stochastic hops, and the full positive-orbit audit. **15/15** combined tests passed.
- The correct mixed law is relatively close to `a²` in this particular guide/chamber: total variation distance **0.00466048**. Therefore **do not** generalize this toy amplitude mismatch to poorly guided 6×6/8×8 systems.

## The support bottleneck is real

Across all **4,654** translation/complement symmetry orbits, the exact frozen-`a` oracle finds **198** strictly profitable individual-orbit sign flips (gain greater than `1e-12`). Their **combined mass under the exact FN mixed distribution** is only **3.75141773154403e−5**, i.e. **0.0037514%**. That is *probability mass*, not their combined energy gain, which is nonadditive across flips.

Each of six FN population runs recorded **274–310 distinct symmetry orbits**. **Zero** of the 198 profitable orbits appeared in any run's recorded FN snapshots. This is consistent with their tiny mixed probability; six populations alone should not be interpreted as a statistical proof of a universal suppression law.

## Discovery comparison

All arms start from **the same heldout mixed FN roots** and have private oracle caches. They do not receive the positive-orbit reference list. A policy proposal searches two physical Hamiltonian hops; tempered exploration instead uses eight symmetric Metropolis bond proposals targeting `a^0.5` from a real FN root.

### First comparison: equal heldout attempts (6 × 160)

| Method | Distinct positive orbits discovered, summed over six independent blocks |
| --- | ---: |
| Direct FN walker-state proposals | 0 |
| Random two-hop exploration | 10 |
| Tempered search | 8 |
| Fixed geometry heuristic | 11 |
| **Learned two-hop explorer** | **19** |

**Not cost-comparable:** the policy consumed 400 additional *training* verification attempts and one extra FN walker population.

### Fairer comparison: charge those 400 extra training labels

Controls now receive 227 test attempts per heldout FN population (`6×227=1362`); learned policy receives 160 each plus 400 attempts for source training (`960+400=1360`).

| Method | New positive orbit discoveries | Total attempted verifier checks |
| --- | ---: | ---: |
| Direct walker states | 0 | 1,362 |
| **Random two-hop search** | **19** | 1,362 |
| Tempered eight-proposal search | 10 | 1,362 |
| **Geometry heuristic** | **19** | 1,362 |
| **RL (400 train + 960 test)** | **19** | 1,360 |

- 17 of RL's 19 discovered positives were **not previously labeled during its training**. This tests more than replay, although it is the same sign chamber.
- Actual distinct terminal orbit verifications: RL **343 train + 895 test = 1,238**; random **1,253**. Stagewise distinct guide amplitude configurations: RL **10,115 train + 32,224 test = 42,339**, random **41,622**. Bond probes: RL **316,640** including training, random **318,160**.
- **Furthermore** RL needs an *additional* genuine FN source walker run to generate its training roots: **75,876** distinct guide configurations (254,011 guide amplitude requests), not charged to the random controls. This makes RL strictly *more expensive* for these tests overall.
- The six paired block differences `RL−random` in discovered positives were `[-1,+1,-3,-2,+3,+2]`, mean **0**, descriptive Student-t 95% interval **[−2.48,+2.48] corrections per heldout block**. The small sample and repeated physical chamber preclude strong claims.

## Decision

**Confirm the support bottleneck**: low-mass profitable sign-correction orbits are missed in real FN mixed snapshots, but short local exploration from visited configurations can reach valuable ones.

**Do not claim RL beats random exploration or that the FN–Krylov loop has improved.** With comparable orbit-verification/guide-query expenditure, this simple continuous-energy-reward RL policy finds exactly as many profitable orbits as random and geometry-only controls, *before accounting for its extra FN source-population cost*. Scaling this specific explorer unchanged to 6×6 is not justified.

The next research question is **how to select which low-probability configurations warrant expensive Krylov verification**, not whether to add more scalar-reward training steps. A corrected next test would retain real FN walkers, replace direct energy-label queries by a costed local **K1 verification**, and test whether *targeted deeper exploration* beats depth-matched random exploration at a fixed full budget. Only then combine accepted changes into a consistent callable `s_{k+1}` and independently rerun FN before making loop-improvement claims. This stage has **not** been implemented.

## Reproduction and artifacts

Experiment sources in this draft PR:
- `avsl_fn_mixed20.py` (branching sampler and optional exact FN reference audit)
- `avsl_mixed_support_pilot20.py` (FN support audit, continuous reward RL, arms and budget matching)
- `test_avsl_fn_mixed20.py` and the earlier AVSL-06/07 tests
- `avsl_cooperative20.py` dependency for the quotient symmetry oracle and bond generator
- `avsl_mixed20_compact_report.json` with paired seed counts, full query costs, numerical verification, SHA-256 and limitations
- `raw/avsl_mixed20_raw.tar.gz.b64` containing the complete six-run comparison JSON, four validation populations' summaries, and six original recorded FN-root samples
- `reference/fn_collapse_exact20_krylov_v2.py` and `reference/prepare_sign20.py` regenerate the 20-site guide under the archived temporary path. The input data used for these results has SHA-256 `5d15ad5ce2cc0f2b5dd9ba41c4c14d9f4b63d1dca5004acf85cd15509b1e208f`.

Example invocation after regenerating `/tmp/avsl05-20site-structural-data/sign20_training_data.npz` and its FN reference:
```bash
OPENBLAS_NUM_THREADS=1 python avsl_fn_mixed20.py \
  --data /tmp/avsl05-20site-structural-data/sign20_training_data.npz \
  --make-ref --ref-out /tmp/avsl_fn20_ref_post.npz \
  --out /tmp/avsl_fn20_validation.json --beta 12 --burn 5 --seeds 4

OPENBLAS_NUM_THREADS=1 python avsl_mixed_support_pilot20.py \
  --data /tmp/avsl05-20site-structural-data/sign20_training_data.npz \
  --reference /tmp/avsl_fn20_ref_post.npz \
  --out /tmp/avsl_mixed20_budget_adjusted.json \
  --seed 28109 --beta 10 --burn 4 --walkers 128 --record 4 \
  --train-budget 400 --budget 160 --test-seeds 6 \
  --match-training-labels --save-roots
```

The user’s original working tree is unchanged. Experiments were executed in isolated Mac scratch, not on the cluster. This is **20-site proof-of-support**, not a proof of 6×6 scalable learned correction or post-FN energy decrease.
