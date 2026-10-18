# J1-J2 iterative sign-learning / scalable local-correction handoff (2026-10-18)

## Objective, not the benchmark
Develop a sign-repair mechanism for frustrated spin-1/2 J1-J2 at J2/J1=0.5 that scales to 6x6/8x8 and beyond, instead of merely fitting the exact signs or maximizing 20-site CNN accuracy. FN amplitudes can reveal overlooked regions, while the Krylov/local Hamiltonian step decides corrections. Corrections must be reversible and must avoid collective overcorrection. Test sampling/access and parameter costs rather than granting global amplitudes for free.

## Mathematical test
At frozen amplitudes a_x>0 and current signs s_x, define the energy gain from reversing one configuration x:
G_x = 4 a_x s_x sum_{y!=x} H_xy s_y a_y.
A symmetry-orbit sign flip must account for all external bonds; multiple flips interact, so use the *marginal* gain after earlier accepted flips (greedy Ising coordinate descent). This guarantees decreasing **fixed-amplitude** energy for accepted flips, not decreasing energy after FN amplitudes are updated or after a neural network generalizes to unobserved states.

## Exact 20-site findings
- S^z=0 dimension 184,756, 20-site tilted 4x5 torus, J2/J1=0.5. Starting point: Marshall signs/J2=0 amplitudes -> 6 frozen-sign FN rounds -> optimized Krylov -> 3 frozen-sign FN rounds.
- Measured FN suppression correlates strongly with rare Krylov flip candidates but is *not* a reliable sign label. Correct-sign configurations can also collapse. Naively reweighting whole-space samples by suppression worsened a pilot CNN.
- Corrected CNN pilot: 3,000 AdamW steps, five seeds, one-hop candidate support, energy-optimized Krylov labels, 20% symmetry-orbit holdout. Mean fixed-a energy: ordinary -10.414802484, FN-suppression-weighted -10.414689265, individual-energy-score-weighted -10.413678531, unchanged-guide -10.414447972. These results are **small-system diagnostics**, not scaling verdicts.
- Global-oracle greedy orbit updates: 40 symmetry orbits (~1,600 configurations) recover 94.4% of full optimized-Krylov energy gain (0.000910056 vs 0.000963543); they use globally known amplitudes.
- Sampler-discovered pool + local bond evidence, B=300 scored symmetry orbits, 40 maximum accepted orbit flips, 12 seed trials: screening by *mean observed signed coupling* recovers mean +0.000394493 fixed-a energy versus random +0.000124592, sign-only local field +0.000096996, smallest amp +0.000047286. Approximate distinct amplitudes queried: 15,217 observed-signed vs 16,031 random. This ranks useful local candidates and does not establish training benefit.
- CRUCIAL scalability limitation: exact full-space FN amplitudes still exist as an **on-demand oracle backend** to the sampler. The local Metropolis chain is a toy (targets amplitude^1), not a GFMC walker; costs of computing/learning those amplitudes are absent. The fixed-a energy gate can fail with inaccurate amplitudes.

## New imperfect-amplitude diagnostic (12 seeds; identical clean-amplitude candidate pools)
Symmetry-preserving Gaussian noise in per-orbit log-amplitude, sampled via a deterministic hash/seed. Proposed flips and energy gates use noisy amplitudes; the final delta E is independently scored against the original frozen guide.
| log-amplitude sigma | mean actual energy gain | trials with worsening energy / 12 | mean *predicted* gain |
|---|---:|---:|---:|
| 0.0 | +0.000394493 | 0 | +0.000394493 |
| 0.1 | +0.000361091 | 0 | +0.000463184 |
| 0.3 | -0.000960771 | 11 | +0.001152190 |
| 0.5 | -0.011381067 | 12 | +0.007264962 |
Result: naive energy gating is not robust to realistically imperfect amplitude estimates; sigma is a **synthetic perturbation** and is not yet calibrated to an actual neural guide.

## Next decisive experiments
1. Benchmark cheap finite-hop bond screening at fixed *total* query/flop/walker budget, and measure out-of-pool generalization and actual energy change.
2. Replace the full FN-amplitude oracle with a callable variational guide; perturb, regularize and calibrate `log[a(y)/a(x)]` on visited frustrated edges, not arbitrary global amplitudes. Distinguish guide error from finite-walker noise.
3. Derive *robust* marginal-gain confidence intervals from amplitude-ratio error bounds. Accept an orbit flip only when the **worst-case** energy improvement exceeds zero; allow backtracking/rollback.
4. Train networks on per-iteration **composite updated signs**, not ED ground-state signs or stacked Krylov correction factors. Preserve the previous guide on untouched configurations and evaluate energy, held-out error and sampled query cost.
5. Only then move to 6x6/8x8. A negative short CNN result is not evidence that scalable sign correction is impossible.

## Validation protocol / interpretation
ED target labels are permitted for evaluation only. Whole-orbit corrections enforce translation and global spin-flip symmetries; make sure geometry includes all intended symmetries. A local greedy energy decrease is measured at frozen amplitudes and cannot guarantee FN self-consistent descent. Direct independent H expectation checks should agree with local crossing-edge energy differences. The sampled candidate *support* is bounded by whatever the explorer proposes; reweighting a distribution alone cannot find zero-support configurations.

## Code and data
Files in this directory are archived in the same dated experiment and are designed to run with NumPy/SciPy/PyTorch. The original scripts were executed under `/mnt/data` and include hardcoded path references. Use the committed `reproduce.py` runner to stage copies with paths rewritten into an isolated scratch directory; it never edits archived originals.
Recreate data with `fn_collapse_exact20_krylov_v2.py` + `prepare_sign20.py`, audit true Krylov labels with `audit_sign20_targets.py`, obtain next-step suppression using `measure_suppression20_fast.py`, run the corrected CNN with `train_sign20_corrected.py`, and run `j1j2_budgeted_orbit20.py` / `j1j2_noisy_oracle_diagnostic.py`. Independent local-E audit: `validate_j1j2_budgeted20.py`.


## New robustness diagnostics — 2026-10-18 evening
- The signed crossing-edge contribution from a symmetry orbit g into orbit h is `W_gh=sum_{x in g,y in h} H_xy s_x s_y a_x a_y`. For current flipped set F, replace W_gh by -W_gh when h belongs to F. Set P=sum max(W,0), N=sum max(-W,0). A single-orbit marginal flip improves *frozen-amplitude* energy when P>N.
- Under per-neighbor bounded log-amplitude error |log(a_hat_h/a_h)| <= eps, **regardless of a positive multiplicative error at g**, a sufficient guarantee is `exp(-eps) P > exp(eps) N`; equivalently `eps < 0.5 log(P/N)` (when N>0). This is a bound on neighbor amplitude uncertainty; a bound on *neighbor/central ratios* can serve in its place. Gaussian sigma is NOT a deterministic error bound.
- Exact-guide orbit ranking, 12 seeds, 88 greedily accepted orbits: epsilon certification thresholds quantiles (min,p10,p25,median,p75,p90,max) = (0.01459,0.02084,0.02496,0.05361,0.06627,0.06843,0.09886). At eps=0.01/0.03/0.05/0.07/0.1, the number of these 88 flips certifiable individually is 88/52/48/2/0, respectively. A conservative robust rule with eps=0.1 certifies **no** flips on the short tested seeds. Source: `j1j2_margin_precision20.py`, `j1j2_margin_precision20_results.json`, `j1j2_robust_gate20.py`.
- Two consecutive fixed-sign FN amplitude solutions are MUCH better correlated than independent Gaussian per-orbit errors. With old-amplitude sampled candidate pools and old-amplitude greedy orbit decisions, actual gain evaluated at the next FN amplitude is positive for **all 12 trials**, mean +0.000405666 versus old-guide gain +0.000394493; zero originally chosen individual sequential corrections became adverse under the next amplitude. Mean unweighted median |neighbor log-ratio drift| is ~0.0373; p90 ~0.244; weighted RMS ~0.0464. This only tests two **successive FN solutions near convergence**, not an actual neural guide. Source: `j1j2_real_fn_drift20.py`, `j1j2_real_fn_drift20_results.json`.
- Direct full-H expectation independently confirmed the FN-continuation energy gains for seeds 201 and 211 to ~1e-14, using `j1j2_real_fn_drift_validation20.py`.
- Interpretation: **Ratio accuracy on important frustrated edges** is the bottleneck; a 20-site classifying CNN score does not establish scale-up. The next critical experiment requires an imperfect *callable NQS* amplitude and real query/training cost, ideally calibrating an empirical high-confidence bound on neighbor amplitude ratios before committing an orbit flip.

## Reproduction file organization
- `fn_collapse_exact20_krylov_v2.py` builds exact 20-site Hamiltonian and FN solver.
- `prepare_sign20.py` writes `sign20_training_data.npz` (states, initial exact-FN guide amplitude, current signs, ED labels for scoring).
- `audit_sign20_targets.py` writes `audit_sign20_targets.npz` with energy-optimized Krylov corrections; ED signs are NOT a training target.
- `measure_suppression20_fast.py` writes `sign20_suppression.npz` with measured suppression and another FN amplitude iteration.
- `j1j2_orbit_priority20.py` is imported by `train_sign20_corrected.py` (CPU/PyTorch).
- `j1j2_budgeted_orbit20.py` depends on `j1j2_local_query_diagnostic.py`; the noise, precision, and continuation scripts depend on these.
- Code currently uses literal `/mnt/data` working directories: **to reproduce on other machines, stage scripts/data there or replace these literal paths in copies with your dedicated scratch directory**. The uncommitted Mac workspace is independent; do not stage unrelated files.
- Large binary `.npz` data are **not included** in the GitHub text archive; regenerate them with the preparatory scripts. A conversation-local archive of scripts/results was also generated separately.

## Fresh-workspace commands
Requires Python 3.11+ with `numpy` and `scipy` (plus `torch` for the CNN arm). From this directory:
```bash
python reproduce.py --workdir "$HOME/j1j2_20_repro" --stage data
python reproduce.py --workdir "$HOME/j1j2_20_repro" --stage diagnostics
python reproduce.py --workdir "$HOME/j1j2_20_repro" --stage cnn
```
The `data` stage may need significant RAM/CPU to diagonalize the exact 20-site problem. Results are written only into the requested scratch directory; none of the Mac repository's unrelated uncommitted changes are touched.

## Prior 6x6/8x8 caution (historical)
Earlier 8x8 FN-amplitude fits looked accurate on sampled configurations while **crucial rare Hamiltonian-neighbor ratios were off-support**, and the next FN step worsened. Sampled SR/amplitude optimisation also failed near convergence, whereas 4x4 exact-force SR improvements were oracle-limited. Therefore the new requirement of ~2–7% log-ratio precision on delicate 20-site orbit corrections is a **hard hurdle**, not an already achieved 8x8 capability. Do not scale without controlled held-out edge-ratio errors, calibrated confidence intervals, and comparable access cost.

## 2026-10-18: callable CNN amplitude guide and equal-pool ranking (new)

New reproducible scripts and outputs:
- `j1j2_neural_local20.py`, `j1j2_neural_local20_results.json`
- `j1j2_neural_matched_pool20.py`, `j1j2_neural_matched_pool20_results.json`

Setup: 20-site tilted torus, same frozen-sign chamber as above. Two 4,881-parameter translation-invariant CNNs learn log FN amplitudes using 600 symmetry-orbit labels visited in local true-FN Metropolis exploration; both use the same training configurations, optimizer initialization and 900 steps. The second adds a loss on adjacent log-amplitude *differences*. Training used ~6,500 distinct full-space exact-FN amplitude queries per seed (optimistic teacher). **These training labels would not be directly accessible from a real GFMC mixed walker distribution**; this does not solve the amplitude handoff.

Independent tests: candidate explorers are learned-guide Metropolis samplers; one-hop spin-exchange neighborhoods; 300 eligible orbit scores computed via callable model only; greedy up to 40 flips; exact FN vectors and ED signs kept out of selection and used only for energy verification. Two training seeds × three test seeds per model.

- Same-guide exploratory pools: ordinary pointwise loss mean true fixed-a energy gain +0.000188313 (61% of per-pool exact-FN oracle gain); edge-ratio loss +0.000507152 (94% of its own different-pool oracle gain). Both positive in six of six runs. **Pools differ by learned-guide walker behavior; not a controlled head-to-head comparison.**
- Matched-pool ablation: walk each pool using the *pointwise* guide, but score the same 120 walker seeds and their one-hop candidate pool with either model. Pointwise +0.000188313, edge-ratio-trained +0.000261125 (**39% higher**), ~3,554–3,556 distinct network orbit evaluations per scorer. Edge-trained matched-pool actual gains were at least equal in all 6 cases, higher in 5.
- Independent cross-check: direct full Hamiltonian expectation agreed with the symmetry-orbit edge formula within 1.6e-14 for representative selected sets.
- On separately encountered bond sets, exact-FN-weighted log-ratio RMSE was **0.313 (point)** vs **0.322 (edge)**, not a global RMSE improvement. Training on ratios may improve *decision-relevant regions* or guide exploration; the exact mechanism is not established.
- Greedy proposed improvements are not guaranteed with approximate guide ratios: 5/37 pointwise flips and 4/74 edge-trained flips actually increased true marginal energy in their sequential contexts, even though total energy improved in every tested run.
- A naive robust margin bound with eps=0.1 in neighbor log-amplitude errors accepted **zero** corrections in this study; eps=0.05 suppressed almost all benefit. Need better *calibrated per-edge uncertainty* rather than a global error threshold.
- Main scaling caveats: only N=20, 2 training seeds, exact on-demand FN amplitudes used for *training oracle* and offline validation, stored table of existing signs still used as a sign oracle, and ~3,600 of the **4,654** symmetry orbits were queried in each 20-site test. The enumeration fraction is large because 20 sites is tiny; cost at larger N must be measured in calls/operations/visits, not finite-size percentage. No demonstrated 6x6/8x8 scaling, iterative FN reoptimization or generalization of a learned sign network.

**Next gating experiment:** replace teacher access to `a_{FN}(x)` with labels or local ratios estimable from actual mixed FN walkers (plus uncertainty), then run the same fully online query-counted sign update. Separately replace the sign lookup table with a callable stored-sign net. Maintain a no-update energy baseline and monitor true marginal losses, not only aggregate energy gain.

Reproduce locally (scripts currently assume data lives at `/mnt/data`): run `python prepare_sign20.py`, `python audit_sign20_targets.py`; then `python j1j2_neural_local20.py --seeds 2 --steps 900 --train-orbits 600 --trials 3 --budget 300`; and finally `python j1j2_neural_matched_pool20.py`. Files `sign20_training_data.npz` and `audit_sign20_targets.npz` are generated, not checked into GitHub.

## 2026-10-18: four-training-seed extension (supersedes two-seed pilot)

Run the existing `j1j2_neural_local20.py` directly with `--seeds 4 --steps 900 --train-orbits 600 --trials 3 --budget 300`, then run `j1j2_neural_matched_pool20.py` (updated to four seeds). Archived JSON outputs now contain all eight model fits and all twelve matched candidate-pool pairs.

**Four independent training seeds × three search trials per seed:**
- Different learned-guide exploration pools: pointwise-log-amplitude CNN mean true fixed-a energy improvement **+0.000256903**; joint pointwise+neighbor-log-ratio CNN **+0.000460079** (+79%); per-pool exact FN oracle gains +0.000328556 and +0.000494886, respectively. One of twelve point-guide searches worsened energy slightly; all twelve edge-guide searches improved energy.
- **Same walker-discovered candidate pools** (defined by pointwise-guide walker): point CNN **+0.000256903**, edge CNN **+0.000293586**, a **14.3% paired improvement** at 3,596.5 vs 3,596.75 distinct CNN orbit evaluations. Edge wins 7, ties 4, loses 1. **Training-seed variation is important; three trials within one training seed are not statistically independent.**
- Independent FN-weighted log-neighbor-ratio RMSE: point CNN 0.2884; edge CNN 0.2927. General ratio regression accuracy did NOT improve despite the better sign corrections. No causal mechanism yet established.
- At final oracle FN amplitudes, 9/86 greedily accepted point-model flips and 7/135 edge-model flips had *unfavorable individual marginal gains*. Aggregate gain therefore does not certify every decision. Global uncertainty bound eps=0.1 in log amplitudes suppresses all updates; reliable uncertainty gating remains unsolved.
- Total 20-site symmetry orbits: 4,654; each test used ~3,600 distinct callable-guide orbit evaluations. That's a large *fraction* of this tiny system, not evidence of large-scale performance. The computational algorithm only visits sampled local bonds, with naive query complexity O((number of visited walker states + number of scored candidates) × N), each CNN query itself O(N) for fixed channels.
- Primary unsolved issue: **direct exact FN amplitudes are the training teacher** (~6,500 distinct oracle queries per training run). GFMC mixed walkers do *not* directly expose that amplitude at arbitrary spin configurations. Existing finite-size sign guide is still a lookup table. These oracle components must be eliminated before claiming 6×6/8×8 scalability.

**Next proposed test:** Replace the exact-FN teacher with a realistic estimator of neighbor *log amplitude ratios* from mixed FN walker data, maintain independent held-out bond ratios and confidence calibration, then test whether the local greedy correction still improves energy using only observable sampled information. Later replace the stored guide sign table by a callable sign network. Do NOT escalate to 8×8 before a successful oracle-free small-system replication.

## Latest continuation: walker-only FN amplitude learning (2026-10-18)
See [WALKER_ONLY_FN_MIXED_2026-10-18.md](WALKER_ONLY_FN_MIXED_2026-10-18.md) and `j1j2_walker_only_amplitude20.py` / `j1j2_walker_only_eval20.py` for the exact no-FN-label 20-site mixed-walker diagnostic. Important negative result: at 128 walkers, candidate-bond RMS log-ratio error is 0.638 and only 1/4 proposed sign-update sets lowers the exact frozen-FN-amplitude energy. The experiment establishes feasible label-free amplitude fitting but **not** safe or scalable sign correction. `reproduce.py` now supports a `walker` stage after `data`.
