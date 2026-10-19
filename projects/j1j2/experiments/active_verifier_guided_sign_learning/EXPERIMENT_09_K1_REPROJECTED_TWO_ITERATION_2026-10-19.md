# AVSL-09 — Costed Krylov-field verification, composite signs, independent FN reprojection
**2026-10-19 · 20-site tilted J1–J2, J2/J1=.5 · exploratory, NOT scalable success**

## Why this test
Prior AVSL-06/07/08 experiments counted individual frozen-amplitude orbit-energy wins. They **did not** implement a genuine Krylov verifier, a callable composite sign update, or a new FN solve after a proposal. AVSL-09 supplies those missing steps in an exact-reference **20-site** setting, then probes reuse of a frozen learned policy across an FN/sign change.

## Online construction
Given a genuine stored mixed FN population and positive guide `a`, current symmetry-covariant signs `s`, and a selected candidate configuration `x`:

1. Explore from walker roots by zero hops (walker), one or two random Hamiltonian hops, one-hop geometry bias, or a frozen two-hop stochastic AVSL-08 policy (plus a one-hop zero-shot ablation). Sampling sees only legal swaps and local geometry, **not** exact energy labels.
2. Compute the actual **local first-order Krylov field**
   `r(x)=H(s*a)(x)/(s(x)*a(x)) = H_xx + sum_y H_xy s(x)s(y)a(y)/a(x)`,
   requiring guide/sign evaluations at `x` and all valid one-hop neighbors. Deduplicate by translation/spin-complement symmetry orbit `g(x)`.
3. Fix `T=0` for the entire pilot. If `r(x)>T`, mark the *complete* symmetry orbit as flipped. This implements the sign of `(T-H)psi` at the inspected configurations. `T=0` is exploratory; full optimal global threshold was **not** used for online selection.
4. The entire resulting sign field is **callable on any 20-site configuration**:
   `s_new(x)=s_old(x)*(-1 if g(x) in verified_flip_orbits else +1)`.
   It respects the enumerated symmetry, and it accounts for interactions between accepted flips via the independent full energy evaluation. It is **NOT** a large-N generalizing sign network; the orbit exception list can grow.
5. Run the **independent full 20-site fixed-node eigenvalue solver** using the unchanged guide `a` and the composite `s_new`. Independently report `E_FN[a,s_old]-E_FN[a,s_new]`. The exact full-space amplitudes are available through a callable reference backend but are *not* policy inputs.
6. Charge proposal hops, feature enumeration, symmetry expansion, distinct Krylov checks, and unique guide/sign lookups. The learned explorer's **400 previously spent** single-orbit energy-label training checks and separate FN walker training population are separately charged and cannot be ignored.

The projected reference Hamiltonian reproduces `E_FN[a,s]=-10.417697678503282`, and independent tests verify the Krylov local field against `H psi / psi`, global symmetry invariance, the full-H energy-difference identity, and the no-update FN energy.

### Offline diagnostic, not available to a scalable policy
Enumerating all 4,654 symmetry orbits after the fact:
- 198 individually energy-profitable orbit flips.
- Fixed `T=0` Krylov local field identifies 177 of those 198, among 193 threshold-positive orbits. Flipping all 193 globally would produce frozen-`a` energy gain `0.0007226412`.
- Exact globally optimized threshold on this fixed guide is about **−2.4968**, selecting 264 orbits and producing joint frozen-`a` gain `0.0009635426`. This oracle threshold was **not** used to choose online policy outcomes.
- These checks show the **local physical verifier is informative**. Scarce relevant candidate support remains the selection bottleneck.

## First K1 → FN update, 6 independent genuine FN populations
Each method gets **160 attempted** candidate selections per population; all roots were archived from real `M=128` branching FN samplers, `β=10`, burn `β=4`. The learned policy is frozen from AVSL-08 and was trained on the same pre-update sign/amplitude chamber, using independent FN-root seeds.

| Candidate proposal | Average K1-positive symmetry orbits accepted | Mean **reprojected** FN energy reduction | Positive FN reductions out of 6 | Mean distinct K1 checks | Mean unique amplitude configurations |
|---|---:|---:|---:|---:|---:|
| Walker only | 0.00 | 0 | 0 | 87.0 | 2,928.7 |
| Random 1 hop | 1.17 | **5.39e−5** | 4 | 134.3 | 4,555.5 |
| Geometry 1 hop | 1.33 | **7.70e−5** | 4 | 133.2 | 4,583.7 |
| Learned 1 hop, frozen 2-hop network | 0.67 | 3.18e−5 | 3 | 133.7 | 4,562.3 |
| Random 2 hops | 1.67 | 2.64e−5 | 5 | 149.8 | 5,374.2 |
| Learned 2 hops | 3.33 | 6.77e−5 | 5 | 150.0 | 5,344.7 |

All six full-FN outputs are archived, including per-run accepted orbit IDs, fresh K1 field scores, proposed-sign physical energy gain and FN residual. FN eigen-residuals in this run were at most about `3.1e−7`, substantially smaller than the main nonzero FN energy gains.

**Interpretation:** Frontier exploration produces sign modifications with **independently verified post-FN benefit**, whereas directly examining walkers gave no sign updates in these six populations. Learned two-hop gets more K1-positive candidates, but neither it nor its zero-shot one-hop use establishes a cost-effective advantage over simple geometry/random alternatives. This is a proof of a working local frontier + Krylov + finite-list sign + FN stack in the toy system.

## Second update: frozen policy across an actual changed FN chamber
We selected three of the heldout first-stage populations (indices 0, 2, 5), with methods random1, geometry1 and learned2. After first K1 changes, we **reprojected FN**, used the resulting positive FN eigenvector as the **refreshed guide**, and generated a new real branching FN population at `M=128, β=6, burn β=2`. We did **not retrain** the policy, and spent 160 more candidate proposals and K1 verifications on the new FN state. We applied a new callable sign update and independently reprojected FN.

**Critical confound discovered:** `E_FN[a,s]` depends on *both* sign field `s` and guide `a`. Substituting `a_new = phi_FN[a,s]` and rerunning FN often lowers the energy **with zero sign changes**. It would be wrong to attribute the whole observed second-step gain (~`5.6e−3`) to Krylov or RL.

Accordingly, for **each intermediate state**, we separately computed the no-second-sign-update control `E_FN[a_1,s_1]`. The correct **second-sign incremental benefit** is
`E_FN[a_1,s_1] - E_FN[a_1,s_2]`, holding `a_1` fixed between both solves:

| Method | Mean **guide-only** second-stage FN energy reduction | Mean **additional second-K1 sign** FN reduction (3 blocks) |
|---|---:|---:|
| Random 1 hop | 0.00558493 | **0.00005778** |
| Geometry 1 hop | 0.00559801 | **0.00007641** |
| Frozen learned 2 hops | 0.00559475 | **0.00008195** |

Second-step sign corrections: random1 `[1,1,0]`, geometry1 `[2,1,1]`, learned2 `[4,2,6]`.

**Interpretation:** The frozen learned policy does locate further beneficial K1 sign corrections after a changed FN projection, and the sign-only benefit survives a guide-refresh no-update control. Its very small numerical advantage over geometry (8.19e−5 vs 7.64e−5) with only three data points is **not robust evidence of generalization or amortization**. The much larger amplitude-refresh improvement must not be sold as a learned sign effect.

## What has *not* been demonstrated
- No **scalable** sign representation: the output is a 20-site enumerated orbit exception list, not a neural `s_{k+1}(x)`.
- No 6×6 or 8×8 NN experiment; 20-site reference amplitudes are exact tables accessed through an on-demand interface.
- No implementation of the previously designed **region-local connected-component Krylov** `C_w`; current K1 verifier is one physical local-field sum with fixed global `T=0`.
- The AVSL-08 policy was trained on expensive exact individual-orbit energy rewards (400 trials) in the *same starting chamber*. The next chamber reused that policy, but we did not demonstrate a fair training-cost break-even horizon.
- The initial threshold was chosen after inspecting an exploratory full-space K1 diagnostic, so the six-seed results are exploratory, not confirmatory.
- Independent root populations do not imply independent sign chambers. The second-stage uses only 3 handpicked blocks; report all per-block data.
- The nearest valuable single-orbit oracle labels are not necessarily the same as the global optimized K1 sign decision.

## Physics and cost gates before 6×6
1. Preserve per-phase FN energy **with and without** guide refresh; otherwise a false RL improvement is easy to claim.
2. Compare **actual local K1 checks** to random1, random2, geometry1, targeted bond and region-local `C_w` with the *same total amplitude, sign and proposal costs*, charging source training and all FN population generation.
3. Require a **compact callable sign corrector** that generalizes to unseen configurations without enumerating symmetry orbits or recursively calling expensive nested K1 fields.
4. Check that learned ranking predicts sign improvement on a **new FN chamber**, not merely its source guide; use heldout matched populations and a preregistered global or local threshold.
5. Only then test a 6×6 one-hop frontier with current walkers, the existing baseline FN–Krylov loop, and energy per site plus local wrong-sign edge rate.

## Reproduction
- `avsl09_k1_diagnostic.py`: exact *posthoc* K1 threshold opportunity audit.
- `avsl09_k1_fn_loop20.py`: real K1 candidate evaluation, coherent sign-field changes, full FN reprojection (six roots).
- `avsl09_two_iteration20.py`: generate new FN walkers after first sign and amplitude update, apply second frozen-policy K1 update and FN reproject.
- `avsl09_second_step_control20.py`: **required** same-guide no-second-sign-update FN control.
- `test_avsl09_k1_fn_loop20.py`: five extra validation tests (all pass).
- `avsl09_compact_result.json` / `raw/avsl09_k1_fn_full.json` / `raw/avsl09_two_iteration20_controlled.json` / `raw/avsl09_k1_diag.json` contain complete counts, energies, orbits, and limits.
- Input: same 20-site `sign20_training_data.npz` as AVSL-08; frozen prior policy from `avsl_mixed20_budget_adjusted.json` (full results in AVSL-08 archive); independent FN checker in `reference/fn_collapse_exact20_krylov_v2.py`.

**Reproduction of second stage** needs the `newdata_path` intermediate amplitude/sign NPZ files from `avsl09_two_iteration20.py`; those are not GitHub-tracked since they use complete 20-site arrays. Run the first-stage and second-stage scripts sequentially to regenerate them. Seeds and options are stored in the raw JSON metadata. All source runs were executed on isolated local Mac scratch; user's primary working tree was not overwritten.

**Decision:** Continue verifier-allocation research; do not claim present RL beats simple search on end-to-end cost. For a useful next result, remove orbit-table dependency and prove cost-efficient transfer at 6×6 with a compact sign rule.
