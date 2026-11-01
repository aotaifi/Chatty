# AVSL-10 — Compact local sign rules: unsafe extrapolation, promising Krylov-verifier allocation
**2026-10-19 · 20-site tilted J1–J2 = 0.5 · exploratory, not a proven scalable algorithm**

## Question and decision
AVSL-09 established that selected K1 corrections lower *independently reprojected FN energies*, including after new FN iterations. But its accepted sign structure is a finite list of symmetry orbits. Can a small **callable, size-independent local sign-correction model** generalize rare verified corrections to unseen configurations without enumerating the Hilbert space? If not, can the same model at least cheaply prioritize **genuine** Krylov verification calls on FN-walker frontiers?

**Decision:** Directly trusting this learned sign rule is unsafe. The model is useful **as a candidate ranker behind Krylov verification**, but most of its gain appears to reflect the very strong **lowest-guide-amplitude-first** baseline. We do not yet have an amortized neural advantage or a scalable compact output sign rule.

## Model and strictly allowed inputs
A correction sign `eta_theta(x)` would multiply the already callable old sign `s_k(x)`. Features are **translation and global spin-complement invariant** and cheap per configuration:
- A 2×2 spin-window motif histogram with 8 complement-identified bins.
- A 2×3 motif histogram with 32 complement-identified bins, pooled over all N translated windows.
- NN and NNN antiparallel-spin fractions.
- Optional 43rd feature: **one** positive guide-amplitude query `(log a_k(x))/N`.

Thus model complexity is **42 or 43 scalar features** followed by one regularized logistic linear layer. At inference this requires O(N) spin operations and either zero or one current-guide calls per configuration, with *no exact FN amplitudes, K1 fields, ED labels, or symmetry orbit lookup needed*. The 6×6 `tilt=0` feature routine runs in tests. It is **not** yet a validated 6×6 sign updater.

**Training data:** Genuine archived `M=128` mixed-FN walker populations (source blocks 0,1,2), proposals one or two legal Hamiltonian hops from real walker roots, and the *actual physical local K1 field* `r(x)=H(sa)(x)/(s(x)a(x))`; binary decision `r(x)>T=0`. Training used **2,640 attempted Krylov queries**, 2,246 stagewise distinct verifier evaluations, **1,214 globally distinct labeled symmetry orbits** with **18 positives**, and **74,203 stagewise unique guide amplitude configurations** (summed across separately cached methods). This does **not** include the earlier AVSL08 learned proposer’s 400 energy-label training trials and its separate FN population. No ED sign labels or exact full-orbit label audit entered training.

## Experiment A: Direct compact sign-rule extrapolation FAILED

The decision threshold was fixed using only TRAIN K1-labeled examples: maximum of the 99.5th percentile negative score and the 25th-percentile positive score. Then this globally callable rule predicted signs across all 184,756 20-site computational states, and a **separate exact 20-site FN solver** checked the joint result.

| Proposed direct sign correction | Predicted orbits flipped | Unseen predicted orbits | K1-positive amongst unseen (posthoc) | Change in independently projected FN energy |
|---|---:|---:|---:|---:|
| 42-feature geometry-only rule | 62 | 54 | 6 | **−0.0270951** |
| 43-feature geometry + log guide amplitude | 286 | 277 | 75 | **−0.00303037** |

Here **negative** sign means the new sign rule made the FN energy **worse** (we always report `E_FN[old]−E_FN[new]`). This is not a Monte Carlo fluctuation: the independent exact FN eigenresidual was approximately `1e−8`.

The guide-assisted model had 75/277=27.1% positive K1 labels amongst unseen selected orbits, versus 193/4,654=4.1% K1-positive across the full orbit space, which is meaningful **enrichment for ranking**. Nevertheless **~73% of its unseen predicted flips were false positives under this Krylov decision criterion**. Coupled, noncommuting signed changes can also harm even when independent K1 labels look favorable.

**Stricter abstention did not rescue direct flipping.** We increased the threshold using training-negative-score quantiles up to 100%. At the 100th percentile, training-selected precision was 100%, but extrapolation still flipped **112** orbits and worsened independently reprojected FN energy by **−0.000248869**. Perfect training precision with 18 rare positives provides no credible unseen-region guarantee. Do not deploy this direct network sign function on 6×6.

## Experiment B: Use compact model *only to allocate paid K1 checks*

All methods rank candidates from **identical** one-hop frontiers of archived *heldout* real FN mixed root populations 3,4,5 (no shared training root streams). For each test population:
- 3,000 attempted bond moves from genuine FN walkers, ~721–753 distinct deduplicated symmetry orbits proposed.
- At most 160 K1 checks per method, stopping around **4,000 unique guide amplitude configurations**. Guide-model and amplitude-rarity scoring queries count toward this cap; geometry/random scores use no amplitudes.
- Only an actual `r(x)>0` K1-verified symmetry orbit is accepted. The full joint finite orbit correction list is independently reprojected FN, with no assumed additivity of singleton gains.

| Search/ranking from same frontier | Mean **post-FN energy decrease**, 3 populations | Mean accepted K1-positive orbits | Mean unique guide configurations |
|---|---:|---:|---:|
| Random | **1.158e−4** | 1.67 | 4,013 |
| Trivial high-NNN-antiparallel geometry rank | 0 | 0 | 4,017 |
| **Lowest a_k(x) first**, no training | **3.442e−4** | 7.33 | 4,018 |
| Learned 42-feature local motifs | **2.530e−4** | 3.67 | 4,025 |
| Learned 43-feature motifs + guide | **4.261e−4** | 8.00 | 4,020 |

The trained model beats random at this heldout cost, but **an untrained amplitude-rarity rule captures most of the benefit**. In 2 of the 3 heldout populations it found the **exact same accepted positive K1-orbit set** as the learned guide-conditioned model. The one additional population drives the extra learned improvement.

This strongly suggests a simple new baseline: **search low-guide-amplitude configurations near FN walkers and use Krylov to decide signs**. Note that small a(x) itself is **not a trustworthy flip label**; it is only a candidate priority.

## Experiment C: Frozen model on **changed FN sign chambers**

We independently test whether the *same frozen pretrained* ranker works after actual FN–Krylov changes.

- Three distinct 20-site sign chambers generated by first-stage `geometry1` sign updates from AVSL-09 (starting root-block indices 0,2,5); use each resulting FN projected positive amplitude as the **new guide**.
- For each chamber, generate *new genuine M128 mixed-FN walkers* at β=6, burn=2, then generate a new common one-hop frontier.
- Evaluate two independent FN walker/proposal seeds for each chamber, six matched tests total. **Only three genuinely distinct sign chambers**, not six.
- No retraining or changing coefficients/threshold. Each arm capped to ~4,000 unique amplitudes for candidate scoring + K1 verification.
- Compare an independent **no-second-sign-update** FN calculation at the *same refreshed guide* against a new exact FN solve after K1-verified composite signs. Thus the reported improvement isolates the additional **sign step**, not the large automatic gain from changing the guide.

| Candidate ranking | Mean **second-sign-only** FN improvement across six chamber×seed tests |
|---|---:|
| Random | **6.857e−5** |
| Lowest amplitude first | **3.899e−4** |
| Learned motifs only | **3.621e−4** |
| Learned motifs + amplitude | **4.703e−4** |

The learned 43-feature ranker beats amplitude rarity in the mean across both repeats, but the extra gain is concentrated in only **two** of the three changed sign chambers. As a fraction of the model's standardized weight vector squared norm, the **guide-amplitude coefficient contributes ~69.6%**; much of the learned priority is simply amplitude rarity.

**Evidence:** The freeze-and-transfer test does show a viable *search → actual K1 verification → coherent sign updates → independent FN energy gain* cycle over changed FN states, with a compact callable **ranking** model.

**No claim:** We have NOT learned a reliably safe **sign-update function** on unseen configurations, and cannot show the extra model training expense has amortized.

## Critical economic caveat
Ranking comparisons charge candidate score amplitude queries *plus* K1 verifier amplitudes. However, the source learner required 2,640 verifier attempts, globally 1,214 distinct K1 labels, and ~74k stagewise amplitude configurations, plus costs of FN source populations and original AVSL08 RL proposer training. Those are not paid by the 3+6 heldout queries. A single cheap amplitude-based heuristic is a much stronger baseline than earlier random/geometry tests. **No amortized advantage has been demonstrated.**

All signs newly applied in the exact 20-site audit are still stored via a **finite list of symmetry-orbit exceptions**. These use full-basis orbit expansion (not scalable) and are *not* the 42/43-feature network: direct neural sign extrapolation failed. We must not conflate a compact **verifier-allocation policy** with a compact **global sign correction**.

## Recommendation for the next stage
1. **Use lowest-amplitude-first + real K1 as the new default non-ML frontier baseline**, not random or a weak geometry ranker.
2. Investigate a more trustworthy **compact sign representation**: e.g., region-local connected-component Krylov `C_w` from the October 13 size-consistency design, possibly with a residual model trained on actual K1 labels. Test global energy against exact small systems before 6×6.
3. Since `a(x)` can be tiny, study finite-walker/model amplitude-ratio uncertainty and the risk of denominator-driven false positives; exact 20-site K1 checks may be overly optimistic at large sizes.
4. Evaluate source-training amortization over **genuinely new sign chambers**, not only additional seeds from the same three changed sign fields.
5. Scale to 6×6 only after verifying a compact sign rule or a solver that can evaluate the new sign cheaply on every FN neighbor; maintain exact guide/sign query accounting.

## Artifacts
- `avsl10_compact_local_rule20.py` — symmetry-invariant O(N) feature extractor, supervised K1 training from real FN frontiers, direct sign-rule FN audit.
- `avsl10_abstention20.py` — train-only quantile abstention sweep and independent FN control.
- `avsl10_costed_ranker20.py` — heldout FN frontier, explicit paid scoring + Krylov verifier selection, joint signs and FN.
- `avsl10_cross_chamber20.py` — changed-chamber frozen-policy FN replicas with independent no-sign energy control.
- `test_avsl10_local_rule20.py` — symmetry, complement, 6×6 features and training-policy checks; 25 selected tests from AVSL09/10 physics and cost infrastructure passed together.
- `avsl10_compact_result.json`, `raw/avsl10_raw.tar.gz.b64` — numerical results, raw scored candidate/verification logs, source labels, model weights and two changed-chamber replicate sets.
- Earlier dependencies `avsl09_k1_fn_loop20.py`, `avsl_fn_mixed20.py`, `reference/fn_collapse_exact20_krylov_v2.py` and archived mixed-FN root arrays from AVSL08.

**Status:** exploratory 20-site mechanism, source training costly, scalable sign correction unresolved.
