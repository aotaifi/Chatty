# paired_fn: paired lattice fixed-node GFMC for two guides (shared tool)

Spec: `/scratch/a/A.Otaifi/astra/paired_fn/ASTRA_paired.md` (section 3 recommended estimator, section 5 interface, section 2 diagnostics,
section 4 validation). PLAN_sign.md "ASTRA ON THE PAIRED ESTIMATOR", "BUILD". Built 2026-11-12 by a worker; not committed (the lead commits).

## Convention
`Delta = E_FN[B] - E_FN[A]` (B minus A). Engine series are total energies; summaries divide by `NS` (36 at 6x6, 64 at 8x8).
The q0r JSON field `Delta` and G2 `CASES` use A - B: convert explicitly. H_xy = J_b/2 > 0; a bond is kept when psi(x) psi(y) < 0
(`gfmc_fn_lib.GuideBase.finish`); a general signed H needs a provider that tests H_xy rr < 0.

## Interface
```python
import providers as P, engine as E, analysis as AN
out = E.run(guide_A, guide_B, X0_A, X0_B, mode='coupled', dtau=0.1, burn_blocks=..., measure_blocks=...,
            pair_seed=..., reference_energy=0.0, output=DIR, memory_tau=None,
            coupling='split_rates+config_parent_coupling', NS=36, maxsec=None)
out['series']  # per block: emix_A/B, egrow_A/B, delta, delta_grow, A_/B_ {ess,mmax,top10,span,zero,num,den,lnscale},
               # alpha, n_common, same_pre, same_post, sep_flux, cut_flux, matched_time, shared/A_only/B_only hops,
               # F_count_/F_weight_/F_ess_frac_{A,B}, burn flag;  out['anc']: ancestral family counts / ESS at lags 10,100,1000
AN.pair_stats(eA, eB, cost_pair, saA, saB, cost_A, cost_B, NS)   # S_A, S_B, S_D, S_AB, rho, C_pair, C_ind, kappa_work (eqs 11-13)
```
Provider contract (`providers.py`): `rows(S) -> RowBatch(rate (k,NB), dFN (k,), [valid, keep, sign, log_abs, guide_id])`,
`move(S, b)`, `key(S)` (configuration key for the resampling coupling), optional `sign(S)`, `guide_id`. Both arms must use the same
bond IDs. Providers: `ConfigGuide` (wraps any unchanged `gfmc_fn_lib.GuideBase`, e.g. `TableGuide`, `BareM1Sign`; state = +-1 configs),
`DenseGuide` (small dense H, independent rows; stage 1), `OrbitTableGuide` (exact 6x6 orbit table, `g2lib.rows` convention).
Broken variants for validation only: provider `broken='no_vsf'|'inverse_ratio'`, `walls_sign=`/`walls_sig=` (changed walls omitted);
engine `broken='force_A_parents'` (coupled), `'inverse_hop_ratio'` (reweight_A).

## Modes
- `coupled` (default): two populations with their own configurations, rates, block-local weights and multinomial parent laws. Per bond b:
  both arms swap at min(r_A, r_B), A only at r_A - min, B only at r_B - min (each marginal is exactly its own process). Block end: maximal
  categorical coupling of the two parent laws aggregated by configuration key (`resample_coupled`).
- `union` (experimental): one coordinate, reference rates max(r_A, r_B), two carried log masses (eq 8), common parents p = (p_A + p_B)/2,
  offspring mass v/(M p) (eq 9). Unsupported arm weights are -inf, never clipped.
- `reweight_A` (experimental): walk with A's rates, B mass = A mass x R_{B/A} (eq 5), parents by p_A. Every visited state is checked; a
  B-only kept edge raises `SupportViolation` (hard failure). `memory_tau` resets R every memory_tau blocks (finite memory: biased, eq 10).
- `single`: one standalone multinomial arm (for standalone variances and costs in kappa_work).
Multinomial resampling throughout (not the production engine's systematic resampling): validate against the production engine only in the
finite-M limit, not bitwise.

## Files
`providers.py`, `engine.py`, `analysis.py`, `stage1_checks.py` (stage 1), `guides6.py` (6x6 reference guides and pairs; saved guides in
`/scratch/a/A.Otaifi/ansatz_runs/paired_fn/guides`), `probe8.py` + `probe8_a40.sbatch` (8x8 rho probe, proposal, not run).

## Validation status (2026-11-12)
| stage | status | evidence |
|---|---|---|
| 1 rows / finite propagators | **PASS** (32 checks) | `/scratch/a/A.Otaifi/ansatz_runs/paired_fn/stage1/stage1.json`, `stage1.log` |
| 2 exact 6x6 calibration | not started (lead's amendment: stop after stage 1 for the 8x8 probe proposal) | |
| 3 systematics (M, dtau, burn) | not started | |
| 4 broken versions at 6x6 | not started | |
| 8x8 rho probe | launcher written, not submitted, not smoke-tested | `probe8_a40.sbatch` |
Status labels: algebra-pass (stage 1) / calibration-pass (stage 2) / production-qualified (stages 3-4 + 8x8 pilot). Current: **algebra-pass**.

Stage 1 content: production `TableGuide` rows == independent dense rows (max err 0); E_L == (H psi)/psi; joint generator marginals
(4e-16, 6e-16); resampler marginals and P(equal configuration) = alpha (z <= 1.03, broken force-A law z = 383); finite-T propagator
(T = 0.1..2, 60 replicas x M 2000, four-spin system) for coupled/union on amplitude-only, sign-only and amplitude+sign pairs and reweight_A
on the amplitude-only pair, all |z| < 2.6; coupled arms vs standalone arms |z| < 1.8; reweight_A raises on both sign-changing pairs and on
the two-state counterexample; broken versions miss: inverse-ratio provider (z 33-57), inverse hop ratio in reweight_A (miss +0.128,
z 78; Astra +0.136 at T=2), no V_sf (z 437), changed walls omitted (miss +0.347, z 122), forced A parents (z 51), unchecked A-only on an
unsupported pair (E_B miss -0.0405 at T=2, Astra -0.0402); exact zero controls (identical, global flip) Delta == 0 bitwise in all three modes;
two-state: union -0.994(4), coupled -0.998(4) vs exact -1; unchecked A-only -1.245(4) at T=0.5 (Astra's -1.25) and -0.5 at T=20 (finite-M
loss of state 0); the X-state ConfigGuide path agrees with the dense propagator.
Observation (4-spin, amplitude+sign pair, identical starts): paired finite-T variance / independent = 1.31 (no gain on that toy pair).
