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
Files in this directory are archived in the same dated experiment and are designed to run with NumPy/SciPy/PyTorch. The original scripts were executed under `/mnt/data` and include hardcoded path references; use `J1J2_DATA_DIR` or adjust their base path to this directory for local replay.
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
