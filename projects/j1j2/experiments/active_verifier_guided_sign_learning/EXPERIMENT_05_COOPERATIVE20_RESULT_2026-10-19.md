# AVSL Experiment 05A — bounded cooperative-orbit audit, 2026-10-19

**Verdict: REJECT cooperative clusters of at most eight Hamiltonian-connected symmetry orbits as the current leading explanation for AVSL's poor 20-site transfer. TEST the original experiment's callable-guide/verification bottlenecks next.** This is a bounded exact-20-site reference diagnostic, not a scalable algorithm.

## Controls and method
- Reconstructed the archived `20`-site `4×5` tilted `S^z=0` J1–J2 `J2/J1=0.5` guide (6 Marshall FN rounds, K1 signs, 3 FN rounds); reference amplitude and sign vectors were available through an **exact evaluator**. Final prepared FN energy from regeneration: `-10.411321676977218`. No ED sign labels trained.
- Gain-blind exploration: `256` roots × `4` seeds (`30260–30263`) × `4` different root laws. Each root grows a random Hamiltonian-neighbor-connected patch of `8` translation/spin-flip symmetry orbit vertices. **No energy scores were consulted to propose the patch.**
- Root laws: toy Metropolis `pi ∝ a`, Metropolis `pi ∝ a²` (a² VMC law), Metropolis `pi ∝ sqrt(a)`, and uniform `S^z=0` state roots (20-site-only non-deployable access control). **These are not FN mixed `a*phi_FN` walkers.**
- Within a frozen patch, enumerated all `2^8` subset actions to calculate an audit-only ideal joint-cut gain and compared with conservative greedy singles, a best single, and gain-blind random subsets. Every number is the **normalized 20-site total frozen-`a` physical variational energy gain**, not gain per site, a neural-network result, or post-FN improvement.

## Result
Each row aggregates `1024` **overlapping** eight-orbit patches across four seeds in the same sign chamber. Means include no-flip/zero-gain outcomes.

| Root law | Positive subset patches | Positive pair with both individual gains nonpositive | Mean best-subset gain | Mean greedy-single gain | Extra gain of best subset |
|---|---:|---:|---:|---:|---:|
| `pi∝a` | 148 | **0** | `2.77780724e-6` | `2.76567469e-6` | `1.21325519e-8` |
| `pi∝a²` | 117 | **0** | `2.27831839e-6` | `2.26993796e-6` | `8.38042216e-9` |
| `pi∝sqrt(a)` | 183 | **0** | `1.95949742e-6` | `1.95792055e-6` | `1.57686600e-9` |
| Uniform state roots | 263 | **0** | `1.59433848e-6` | `1.59415302e-6` | `1.85458111e-10` |

- No positive oracle subset made *entirely* of individually nonpositive constituent flips was found across **4,096 sampled patches**. The unrestricted subset oracle exceeded the conservative greedy baseline in `4/2/4/3` patches (respectively). The relative extra mean gain was `0.437% / 0.368% / 0.080% / 0.012%` of the respective mean within-patch oracle gain. Rare individual patches do show more noticeable synergistic increments; the result only concerns *the sampled distribution mean*.
- The gain formula `G(A∪B)=G(A)+G(B)-8W_AB/Z` is algebraically correct when `W_AB` counts each undirected crossing between disjoint sets once. But this identity does not demonstrate that beneficial collective moves are reachable or matter here.
- Independent **full sparse H** expectations checked `128` joint sign actions (oracle subsets and random subsets, drawn from all 16 sample runs): max `|ΔE_direct−ΔE_boundary|=2.3314683517128287e-15`; amplitude symmetry error `7.632783294297951e-17` over `223` sampled orbits; zero sign symmetry violations. The physical energies agree. The diagonal cancels exactly for sign-only changes.

## Proposal-support warning (crucial)
A **post-hoc whole 20-site oracle audit** found `4,654` symmetry orbits, `198` individually profitable flips above the numerical threshold `G>1e-12`; the old handoff counted `201` at a looser strict-positive cutoff (the discrepancy concerns negligible gains, not evidence of new physics).

- `pi∝a` 4-seed union: visited `2,981` orbits, including `91/198` significant positive singles (46.0%) but **92.8%** of the summed positive *single-action* gain.
- Union of all 4 root laws × 4 seeds: visited `4,568/4,654` orbits (**98.2%**), `94.95%` of significant positive orbit identities, **99.990%` of summed positive single-action gain.
- These are hindsight coverage measurements, **not** gain obtainable by a scalable policy; the union is effectively enumerating most orbit representatives at this tiny N. The four sampling laws should never be pooled to claim cheap walker discovery.
- Each `pi∝a` seed averaged `27,843.75` **unique exact amplitude array lookups** after scoring; `a²`, `sqrt(a)`, and uniform roots averaged separately and all cache/request/probe counts are in the results JSON. The ~1-second CPU runtimes are *not* estimates of NQS/128-walker inference costs; exact amplitude preparation is excluded.

## Scientific interpretation and next step
**ACCEPT** the exact cut-energy algebra, and the tiny sampled-cluster enhancement. **REJECT** "individually negative but collectively profitable small connected clusters" as a demonstrated primary failure mode of the current 20-site post-Krylov AVSL. **PARK** a larger global sparse QUBO search: it would mostly optimize an exponential-size, non-deployable orbit lookup; it is not the decisive test of support/compressibility. **TEST** the already preregistered Stage-0 128-walker/callable-guide inference and independent a² energy-verifier latency/accuracy, before any NN training.

The experiment **does not** test larger/nonconnected cooperative clusters, a trained or hand-designed size-consistent local sign function, newly sampled FN chambers, network generalization, or a certified after-FN energy reduction. Because the small-patch cooperative mechanism is near null in all four controls, running a *compression network trained to imitate oracle clusters* would add little information; the `C_w` baseline remains required for later real sign-rule learning.

## Reproducibility
GitHub branch `avsl/experiment-05-onpolicy-costed-2026-10-19`, experiments directory:
- `avsl_cooperative20.py`: gain-blind patch sampling, exact frozen-a boundary evaluation, per-orbit caches/counters and 2^8 within-patch hindsight solver.
- `avsl_cooperative20_validate.py`: independent sparse full-H energy and symmetry audit.
- `avsl_cooperative20_support.py`: explicitly non-deployable, evaluation-only global support ceiling.
- `cooperative20_compact_results.json`: all 16 seed-wise summaries, counter values, support audit, SHA-256 hashes of **each full raw run JSON and input NPZ**, and full-H validation summary.
- Full per-patch raw JSON files and the data-regeneration log remain on the isolated Mac scratch at `/tmp/avsl05-20site-structural-data/`; raw SHA values in the compact JSON support reconciliation. The exact FN NPZ binary was regenerated from archived scripts, **not committed**.

Minimal reproduction on a NumPy/SciPy CPU environment (never in the dirty original Mac checkout):
```sh
# Stage copies of fn_collapse_exact20_krylov_v2.py and prepare_sign20.py
# from projects/j1j2/results/scalable_sign_2026-10-18/ in an isolated workdir.
# Replace literal /mnt/data in those *copies* with the workdir and run:
python3 prepare_sign20.py
# Then, in the experiments/active_verifier_guided_sign_learning directory:
for law in tempered_a a2 a_half uniform_states; do
  for seed in 30260 30261 30262 30263; do
    python3 avsl_cooperative20.py --data /scratch/SIGN_DATA/sign20_training_data.npz \
      --anchors 256 --seed "$seed" --root-law "$law" \
      --out "/scratch/SIGN_DATA/cooperative20_${law}_${seed}.json"
  done
done
python3 avsl_cooperative20_support.py \
 --data /scratch/SIGN_DATA/sign20_training_data.npz \
 --directory /scratch/SIGN_DATA --out /scratch/SIGN_DATA/support.json
# For support.py archived naming, the tempered_a files are historically named
# cooperative20_<seed>.json (without 'tempered_a_'), so rename/copy those four.
# Validate direct H with all 16 result paths using:
python3 avsl_cooperative20_validate.py \
 --data /scratch/SIGN_DATA/sign20_training_data.npz \
 --out /scratch/SIGN_DATA/validation.json --ncheck 64 <all-16-result-JSON-paths>
```
**No HPC job was submitted, no neural training run, and no claim of scalability.** The previous Experiment 05 preregistration and strict expensive-run stopping gates still apply.
