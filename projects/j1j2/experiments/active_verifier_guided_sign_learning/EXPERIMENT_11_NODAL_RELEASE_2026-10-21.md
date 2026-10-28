# AVSL-11 — Fixed-node nodal-release objective and rare-sign discovery (2026-10-21)

**Status:** completed local 4x4 and 20-site calculations, independently reprojected selected sign corrections, heldout M128 walker proposal study. No scalable 6x6/8x8 iteration demonstrated.

## Why we changed direction
Previous AVSL06-10 optimized the *fixed-amplitude* sign-flip energy or discovery counts. That objective can mark an actual fixed-node (FN) energy improvement as harmful. We derived and independently validated an FN Rayleigh-release certificate for **one complete symmetry orbit sign flip**, using the SAME positive guide amplitude `a` in the old and new FN Hamiltonians.

With `K_xy=s(x)H_xy s(y)` and normalized **exact old FN ground-state** `phi>0`, the release certificate is

`G_FN(G) = Σ_{(x∈G,y∉G)} K_xy [sqrt(a_y/a_x) phi_x + sqrt(a_x/a_y) phi_y]^2`.

The Rayleigh principle guarantees

`E_FN(s,a) - E_FN(flip_G(s),a) >= G_FN(G)`.

Thus `G_FN>0` is a **sufficient**, not necessary, condition for lower reoptimized FN energy. Crucially, this target depends on amplitudes of visited **neighbors** even when `phi_x≈0` at the candidate itself. The exact phi is an EVALUATION ORACLE and is not available on large systems.

## 4x4 exact physical verification
Post-Krylov 16-site J1-J2=.5, Sz=0, 12,870 configurations / 441 translation-and-spin-inversion orbits: **19** frozen-a positive single flips vs **71** positive exact release certificates. Direct sparse Rayleigh identity independently matched 30 orbit perturbations to <6e-14, and full FN reprojection verified the bound. Example orbit 210: frozen-a gain **-1.73804e-4**, FN certificate **+1.185217e-3**, true independently reprojected FN gain **+1.371407e-3**. Exhaustive 94 simple translation-invariant binary plaquette parity/count sign templates produced no positive frozen-a update, so blindly applying block motifs is not a solution.

## Tilted 20-site exact physical verification
Archived FN guide is approximate, not the lowest FN eigenstate. Solved *the same* fixed-node operator for the exact positive state with diagonal-preconditioned Rayleigh-Ritz: `E_FN=-10.417697678503302`, residual `2.87e-8`, guide overlap `0.999868`. Full FN eigenvector and full-basis orbit labels used ONLY to audit:

- **198** frozen-a positive single-orbit flips vs **245** positive FN-release certificates.
- **52** release-certified orbits looked harmful/neutral in frozen-a test; **5** frozen-a positive orbits lacked certification (lack of certificate is inconclusive).
- 52 newly certified orbits carry `9.02e-6` total guide-squared mass and `2.67e-5` SUM of isolated certificates, compared with `9.47e-4` summed across all 245. **These are small individual opportunities, not jointly additive energy gains.**
- Three release-certified / frozen-a-negative candidates were independently reprojected using the SAME guide, giving true FN energy improvements `+7.15e-6`, `+4.83e-6`, `+4.33e-6`; frozen-a scores were negative `-4.26e-6`, `-2.82e-6`, `-1.44e-6`. Their direct Rayleigh checks matched the certificate <1e-9; FN residuals ~1–2e-8.

## Cheap walker-neighbor screening: no candidate amplitude
With observed M128 FN mixed roots `y`, multiplicities `c_y`, accessible *root* guide values `a_y`, and callable current signs `s(x)`, define a sign-balance score for a candidate symmetric orbit G:

`B(G) = [Σ_(y root, x~y, x∈G) c_y H_xy s_x s_y /a_y] / [Σ_(same) c_y H_xy /a_y]`.

This is a normalized observed forbidden-minus-allowed edge pressure; the candidate `a_x` cancels **assuming symmetry-related guide amplitudes are equal**, so no expensive off-support *amplitude* query is needed for ranking. This is NOT a certified energy score and NOT a replacement for Krylov. Candidate `s(x)` evaluations and symmetry hashes still cost computation. The non-normalized pressure and positive-only forbidden-edge heuristics performed WORSE than random in separate AVSL11 tests.

**Heldout protocol:** Ranking chosen with exploratory seeds 18001–18012. Test on 12 independent heldout FN population seeds 18101–18112 for each beta. 128-walker FN branching/reconfiguration; finite β, approximate fixed guide, unknown mixing/bias. At equal 120 **distinct terminal scored orbits per seed**:

| β | Arm | mean FN-certificate-positive orbits | mean previously frozen-a-negative, FN-certified | mean SUM of positive certificates (not joint gain) |
|---|---|---:|---:|---:|
| .38 | Neighbor sign balance | **9.17** | **1.58** | 4.60e-5 |
| .38 | Random one-hop frontier | 3.00 | 0.58 | **5.24e-5** |
| .38 | Strict J2 two-hop | 4.75 | 0.58 | 3.97e-5 |
| 3.0 | Neighbor sign balance | **9.00** | **1.42** | 3.82e-5 |
| 3.0 | Random one-hop frontier | 4.75 | .83 | **4.35e-5** |
| 3.0 | Strict J2 two-hop | 5.42 | .42 | 3.01e-5 |

Paired count advantages versus random frontier: β=.38 +6.17 [descriptive 95% Student-t CI +3.75, +8.59]; β=3 +4.25 [2.49, 6.01]. However **overall gain magnitude does not outperform random**: this is proposal enrichment, **not a solved scalable energy optimizer**. Screening all walker-adjacent bonds took ~28,500 (β=.38) or ~45,240 (β=3) bond-table inspections / seed and ~356 or 566 unique source-guide configurations; full end-to-end model-query costs not equalized against J2 walk. Existing current s(x) on 20 site is an oracle table, not a demonstrated cheap 6x6 sign callable.

## Project lead's next gate
Do **not** begin another RL reward or tempering sweep. Integrate (1) cheap sign-balance to shortlist candidates, (2) independent estimate of **FN Rayleigh release magnitude and confidence** from actual mixed-walker information, and (3) a **globally callable symmetry-respecting correction η(x)**, followed by independent FN reprojection and cost audit. The full exact old FN phi and full orbit index must be used only on 20-site audits. If significant off-support release magnitude cannot be estimated affordably, stop and redesign before 6x6.

**Code:** `avsl11_nodal_release_core.py` includes generic FN identity and callable sign-balance. Full local implementations, source 20-site reference, unit tests and 24 heldout raw JSON records are in the ChatGPT-generated AVSL11 research bundle. The Mac/HPC was not used. Results are provisional physics experiments, not journal-ready sign-learning claims.
