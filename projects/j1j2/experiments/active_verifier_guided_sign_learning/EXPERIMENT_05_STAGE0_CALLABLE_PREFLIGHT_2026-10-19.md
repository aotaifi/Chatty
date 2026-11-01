# AVSL Experiment 05 — Stage-0 callable 6×6 FN integration preflight (2026-10-19)

**Gate: PASS correctness / FAIL cost scalability for naive recursive K1 as implemented / NOT YET TESTED with the actual NQS checkpoint.** This was an inexpensive integration smoke run, not a trained-model physics result. **Do not launch sign NN training from this outcome.**

## Implemented and executed
- `avsl_stage0_callable_preflight.py` connects existing `projects/j1j2/experiments/callable_fn_loop.py` `SquareJ1J2(6,0.5)` + `K1FNEngine` to the Experiment-05 `MeteredCallable` and incremental `CostLedger`. It uses **128 actual projector walkers** but an analytic, positive, translation-covariant Jastrow test backend (`g1=.045,g2=-.025`): **NOT** the real 6x6/8x8 ViT/NQS checkpoint, and not a physical self-consistent FN chamber.
- Disjoint initial-threshold and independent a² VMC chains (seeds `241719-241722` vs `330519-330522`); symmetric bond-exchange Metropolis accepts with `min(1,exp[2(loga_y-loga_x)])`. True FN mixed walkers are **never** fed to the a² paired estimator.
- Candidate controls `eta=+1` (keep), `eta=-1` (global flip), and `eta=s_M/s_K1` (K1→Marshall). The latter is a reversible non-learned **negative/contrast** arm, not a proposed sign solution.
- Hard bound: `L=6`, `M=128`, `beta_target<=0.12`, `burn<=1000`, `draws<=256`, `chains<=8`; this execution was much smaller than limits.
- Smoke executed on Mac `Darwin-arm64-Python3.13.7` with NumPy/SciPy. Four a² chains × 16 retained samples each after just 32 bond proposals burn-in, two-proposal thinning. **Unconverged statistical error**, sufficient only for sentinel checks.
- Source: `avsl_stage0_callable_preflight.py`; six new focused tests in `test_avsl_stage0_callable_preflight.py`, plus eight existing audit tests. **14/14 tests pass** in 0.169 s. Source SHA-256 at run: `d4620e492af7b27f8f335e1266464281e80eb16bc5eb87dfccaa8c84dad656b9`. Full JSON: `stage0_synthetic_smoke_result.json`.

## Measured small-run costs
```text
L=6, J2/J1=0.5, M=128, beta_target=0.025, 1 FN propagation slice
input threshold roots:      64 (4 independent chains × 16)
projector mixed snapshots:  128 (95 distinct)
NEW exact callable guide configurations DURING THE FN SLICE: 352,978
all stages guide distinct configurations:                      356,475
all stages guide batch/model calls:                              3,756
all stages sign requests needing model resolution:              3,501
all stages lattice neighbor enumerations:                       11,229
all stages enumerated neighbor edges:                           857,260
FN slice wall clock:                                             9.7293 s
entire run wall clock (including ledger and verifier):           16.6065 s
```
The FN stage alone required **about 2,758 previously unseen guide configurations per walker**. The live engine's `ensure_local → ensure_sign(x and one-hop neighbours) → ensure_r(all) → amplitude of their neighbours` reaches deeper neighborhoods than the walker frontiers. **This is a structural state-count cost**, not a measured neural-inference time. If a trained NN needs expensive forward passes for those states, the burden may be prohibitive. Model-batch optimization, cache reuse, truncated/local sign proposals, and/or genuinely cheaper signed-edge querying must be investigated *before* any training campaign.

### Independent frozen-a energy verifier sentinels
```text
Independent chains: 4; 16 sampled x per chain; a² Metropolis;
n retained = 64; no autocorrelation-time or stationarity test.
keep:          delta E = 0, stderr = 0             [PASS EXACT]
global eta=-1: delta E = 0, stderr = 0             [PASS EXACT]
K1->Marshall:  delta E = +4.04887 ± 1.13467       [NOT A PHYSICS RESULT]
```
The last sign/energy result has no physics interpretation due to the deliberately synthetic amplitudes, insufficient burn-in/independence, one short projector step and no genuine NQS checkpoint. Sign of the delta follows `E[K1]-E[Marshall]`.

## Accounting notes
- The `CostLedger` saved append-only `events.jsonl` in isolated Mac scratch `/tmp/avsl05_stage0_synthetic_smoke_seed240519/`: **23,759 incremental events, 14,150,264 bytes**, SHA-256 `d5c556e331bb44b0030119b8ac74b4ea0541b472e38e891275d540fff910903c`. It is **not mirrored to GitHub**, but all aggregate costs needed for the decision and the run seed/source were recorded. Any future real-model pilot MUST archive ledger and aggregate counters per phase in the run artifact.
- Counter semantics: `n_guide_model_calls` counts batch callable invocations; `n_guide_unique_states` counts distinct amplitude configurations; `n_sign_model_calls` counts batches with unknown signs. `wall_s` for underlying inference is a subset of total wall time; don't double count inference events and outer FN runtime. Unknown GPU time omitted.
- Existing prototype ledger is still **incomplete for fully loaded arm comparisons**: not every verifier neighbor query is an increment in `n_neighbor_hops`. The summary `lat_neigh_edges` counts all calls. Correct phase-partitioned accounting before any candidate arm comparison.
- The independent a² verifier gate is only mathematically validated here. It lacks independent-chain autocorrelation diagnostics, multiple-comparison control, projector convergence and NQS ratio-calibration tests.

## Next gate and scientific decision
1. **HOLD** expensive exploratory NN training; naive callable K1 recursion consumed 352,978 unseen amplitudes for just one M128 slice.
2. Before running any expensive 6x6/8x8 FN work, prepare a **real NQS checkpoint** and its JAX/NetKet/Flax runtime. These libraries were absent in default `python3` on both the Mac and `ws1` at this check. The standalone `vit8_callable_amplitude_adapter.py` exists but does not make default Python environments usable. Do not extrapolate synthetic Jastrow CPU wall time to NQS cost.
3. Run an **amplitude-only latency/memory calibration** on a representative, **small, fixed set** (e.g. 32 / 128 / 512 inputs) with the real callable backend. Combine measured per-batch costs with observed frontier size **before** deciding whether a bounded real FN step is affordable.
4. The leading hypothesis is now **walker-accessible amplitude/sign-query economics**, not eight-orbit cooperative energy gain. If per-slice costs are too high, refactor sign proposals so local rules do not require exact recursive K1 over deep neighborhoods, then rerun the same gate.
5. If small-batch latency and target storage satisfy the agreed cap, perform actual 128 walker source guide verification with a fresh run directory, held-out independent a² verifier and replay/no-flip controls. No FN convergence or energy discovery is claimed here.

To reproduce the synthetic sentinel run from repo experiment directory:
```sh
OPENBLAS_NUM_THREADS=1 python3 -m unittest -v test_avsl_stage0_callable_preflight test_avsl_onpolicy_core
OPENBLAS_NUM_THREADS=1 python3 -u avsl_stage0_callable_preflight.py \
 --out /scratch/avsl05-stage0-seed240519 --seed 240519 \
 --L 6 --J2 .5 --M 128 --chains 4 --draws 16 --burn 32 --thin 2 \
 --fn-beta-target .025 --tau-max .025
```
Execution creates `result.json` and `events.jsonl`, does not modify project state or launch a cluster job. Use a **new empty out directory**; existing log directories are never overwritten.
