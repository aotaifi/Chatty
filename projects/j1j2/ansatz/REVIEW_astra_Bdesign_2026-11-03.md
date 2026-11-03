**The pivotal claim is unproven: sampled refitting must preserve gains in the FN energy’s bond metric. I would fund a narrowed V before committing to the 8×8 build.**

1. **Attempt to break that claim.** Deterministic labels remove label noise; they do not establish coverage or generalisation. The proposed `P/S = 0.045` does **not** justify “r shrinks by ~5%.” In Θ v2, I verified 524,288 training pairs and `P/S = 0.0052`, yet the near-minimum-norm projection scored **0.447 in training and −0.477 held out**; damped GN scored **0.224 and −0.009**. This is a different fit, but directly defeats parameter count as sufficient justification.

   My [CPU coverage check](/scratch/a/A.Otaifi/astra/bdesign/coverage_check.json), using theta₄’s exact table, gives only **19,288 starting samples in decade −14**, versus 40,000 requested, for 400,000 independent draws equally allocated across the proposed temperatures. That decade carries **11.2%** of Test W’s bond weight. Partners or adaptive allocation could repair this; neither is demonstrated. Moreover, the existing fit stratifies by **target** amplitude and samples target-ratio-weighted partners; the proposal stratifies by **base** amplitude and leaves partner probabilities unspecified.

2. **Predictions versus evidence:**

| Prediction | Assessment |
|---|---|
| V: `r_sampled ≥ 0.35` | Plausible hypothesis, unsupported quantitatively. It allows 29% degradation from 0.49; neither that tolerance nor the claimed 5% degradation follows from Θ/Test W. |
| R0: −0.49894(1) | The central value is a reasonable analogy; ±10⁻⁵ is uncalibrated. The cited 6×6 comparison gives 6.069 versus 5.574 ×10⁻⁶ above E₀, after an earlier refit—not the proposed untouched production guide. The previous cross-size forecast in the 10×10 report missed by **18–68 ×10⁻⁶/site**. |
| R1: −0.49896…−0.49900 | The optimistic end lacks support. Round 5 converted only **0.288** of its bare gain into augmented gain. Applying that measured ratio to the proposed `r=0.2–0.5`, `g=1.9×10⁻⁴`, and R0 gives **−0.498951…−0.498967**. This conditional extrapolation is itself unvalidated; −0.49900 is another **33×10⁻⁶** below its optimistic endpoint. |

The last calculation is independently reproduced in the [arithmetic check](/scratch/a/A.Otaifi/astra/bdesign/arithmetic_check.json).

3. **Cost.** Twenty–25 H100-h is an optimistic allowance, not an established estimate. The measured 5.5×/10.9× ratios are **CPU walker-step ratios**, excluding sign evaluation and active RBM work—not GPU evaluation rates. At the proposal’s own CNN rate, `128²` calls cost **0.54 seconds per walker-step before deduplication**. CNN batching/caching therefore matters alongside rank-4 GPU efficiency. Benchmark the complete guide at realistic walker batches, symmetry projection and precision, including memory use. The 1.42-hour label arithmetic checks out conditionally; the six-second GN step and grading costs remain unmeasured. Listed build tasks sum to **3.5–6 worker-days**, excluding integration and validation.

4. **Missing controls that affect interpretation.** Require:

   - An 8×8 **energy-based sign-storage control**: the `10⁻⁴` wrong-weight gate is 93× looser than the cited 6×6 result and does not bound FN tax.
   - Validated `f_J` fixes, sampled determination of **E′ and the cap**, and sensitivity checks. The current whole-table maximum-weight check cannot certify unseen 8×8 spikes.
   - Exact 6×6 **c>0** validation of the complete evaluator; c=0 does not exercise the Jacobi machinery.
   - Independent pools/fits, frozen final evaluation, and population/projection convergence. Two GFMC chains do not measure fit variability.
   - Bare/augmented comparisons using the **same stored sign**, plus matched initialisation, symmetry and equal-cost controls. Otherwise gains cannot be attributed specifically to refitting.

5. **My probability: approximately 15%**, a subjective pre-gate judgment. At final uncertainty `5×10⁻⁶`, beating the stipulated benchmark by three combined sigma requires **E < −0.4989792**, before unresolved systematic errors.

   The cheapest decisive test is **one finite-pool replay of round 5 with exact sign and corrected target held fixed**, then exact FN grading—roughly the proposed V budget, **~3 H100-h plus CPU**. Replicate with an independent pool only if promising; add CNN storage afterward. This directly tests the unsupported sampling claim without conflating it with sign storage.