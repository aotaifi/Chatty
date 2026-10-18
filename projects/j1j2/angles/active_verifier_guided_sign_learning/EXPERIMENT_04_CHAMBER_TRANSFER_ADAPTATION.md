# Experiment 04 — 20-site chamber transfer, pooled training, and few-shot adaptation
Date: 2026-10-19. **Status: controlled 20-site diagnostic; not a scalable RL success.**

## Key distinction
A learned policy *ranks candidates* for an expensive **frozen-amplitude individual symmetry-orbit sign-flip energy score**. It neither learns the exact GS signs nor learns an optimized Krylov step; frozen-a individual-orbit gain and a subsequent FN/Krylov iteration are different operations. The evaluator uses a globally available exact 20-site FN amplitude/sign table **as an on-demand oracle**. The root/candidate sampler is a toy amplitude-tempered Metropolis chain, not a GFMC walker. This is a transfer **ranking** test, not yet reinforcement learning with on-policy trajectory rewards.

**Control setup:** S^z=0 20-site tilted torus, dimension 184,756, translation + spin-flip symmetry orbits (4,654 total); same frozen amplitudes for two sign chambers (Marshall versus post-Krylov); 20 evaluation episodes (seeds 30260–30279), fixed 22 newly scored candidate orbits per method per episode, 20% deterministic held-out-orbit split; candidate pool from common local sampled roots and their Hamiltonian neighbors. Exact ED ground-state labels are never used for training. Source training 18 episodes × 110 labels = 1,980 expensive orbit-gain labels; 20-site guide amplitude oracle calls for candidate discovery/verification are **additional costs**, not included in the simple label budget.

## Opportunity audit (post-hoc FULL-SPACE oracle, not a deployable policy)
- All 4,654 orbits: Marshall contains 1,458 individually profitable frozen-a flips; post-Krylov contains only **201**.
- Within the identical 20 held-out candidate pools, the hindsight top-22 selection can recover mean post-Krylov energy gain about **3.25e-5**; actual zero-shot Marshall-trained policy obtains **1.0363e-6**, only ~3.2% of that hindsight bound. Post-Krylov opportunity is both scarce and largely missed by the frozen policy.
- The 22-score ceiling itself is an **oracle ranking across all candidates** in each test pool, and its cost is not a fair baseline. Use it only to estimate ranking regret.
- The score formula was independently audited via full H expectation at fixed a.

## Controlled chamber train/test
| Training labels | 20-site chamber tested | Bond-feature policy gain | Geometry/amp-only policy gain | Signed-bond heuristic | Random |
|---|---|---:|---:|---:|---:|
| 1,980 Marshall-only | Marshall | 2.49136e-4 | 2.86178e-5 | 7.29884e-4 | 1.70509e-4 |
| 1,980 Marshall-only | post-Krylov | 1.03630e-6 | 1.77403e-6 | 0 | 3.19905e-7 |
| 1,980 post-Krylov-only | post-Krylov | 2.17342e-5 | 2.85576e-5 | 0 | 3.19905e-7 |
| 990 Marshall + 990 post | post-Krylov | 1.26869e-5 | 5.80024e-6 | 0 | 3.19905e-7 |

The pooled network is better than pure zero-shot but worse than training directly on the target chamber, using the same total labels. A second initialization gives pooled bond-model post gain 1.12972e-5, corroborating direction, **not** generalizable superiority. The two chambers share amplitudes: no self-consistent FN transfer is demonstrated.

## First few-shot incremental teacher-data test
Source model = Marshall model pretrained on 1,980 labels; retain source normalization; fine-tune the 13-input local-feature CNN for 700 AdamW updates (lr=8e-4) on post-Krylov labels sampled from the *same fixed training episodes*. Scratch comparator uses same target labels, randomly initialized net, 700 AdamW steps and lr=3e-3. Two initialization seeds 19,29; **20 identical held-out test episodes**, each selecting/scoring 22 orbits. Changes after pretraining are not free: report total source + target labels for warm-start.

| Added post labels | Positive target examples | Warm-start mean gain | Scratch target-only mean gain |
|---:|---:|---:|---:|
| 108 | 4 | 1.08437e-5 | 1.05004e-5 |
| 360 | 8 | **1.49094e-5** | 1.23926e-5 |
| 720 | 12 | 1.53161e-5 | 1.31692e-5 |
| 1,980 | 49 | 2.24757e-5 | 2.22841e-5 |

At 360 labels, warm-start increases measured gain relative to scratch by ~20%, but **pretraining cost is additional 1,980 labels**; the improvement needs amortization over many sign iterations before becoming cost-effective. At 1,980 target labels, the advantage is marginal.

**Feature ablation** (same few-shot labels, heldout sets, budgets): at 360 labels, warm-start full bond + local geometry gives 1.49094e-5; geometry/amplitude-only gives 1.42148e-5. They are nearly equivalent. With 1,980 target labels, geometry-only results varied considerably across seeds. **No clear evidence that the encountered-bond features are the reason adaptation works.**

## Independent physics checks
- For the pooled-chamber experiment, 160 selected updates checked against direct full-H computation: max absolute energy discrepancy **3.17e-14**, and 100 sampled symmetry orbits passed amplitude/sign invariance checks.
- For 64 few-shot changes (two seeds, two methods, four budgets, four test episodes each), direct full-H fixed-a energy checks agreed with local crossing-edge scores to max **2.56e-14**. This validates the *energy accounting*, not the ranking generalization or the true FN energy change.
- Unchanged guide is a legitimate zero-gain baseline; all accepted changes in these trials have positive **fixed-amplitude** gain by construction with exact callable a, not with real approximate FN guides.

## Decision
1. **Do not claim a scalable discovery/RL method.** We have shown target-chamber feedback dramatically improves candidate ranking; 360 post labels can alter an old policy, but the geometry-only shortcut and oracle amplitude backend remain.
2. **Next: deployable chamber shift.** Replace the oracle amplitudes and global sign table with a callable, walker-trained amplitude and sign representation. Compare frozen sign-policy, few-shot adapted policy, and *handcrafted region-local field* at equal **total** sampler, teacher, amplitude query and training cost. Include non-flip anchors and cross-fitted independent energy checks.
3. **Train to propose useful corrections, not just classify sign errors.** Keep the verified positive/negative correction examples in a versioned replay store and train on genuinely new chambers (ideally a FN-refresh change), with a held-out orbit **and** held-out chamber evaluation. Error at bonds adjacent to corrections and actual post-FN energy are mandatory.
4. **Size consistency:** coordinate-local/global orbit flips are only a diagnostic. A size-consistent region-based sign rule should be tested per `DESIGN_scalable_loop_2026-10-13.md` before concluding anything about 6x6/8x8.
5. **Training label scaling:** measure positive discoveries per incremental *verification query*, not just mean gain per 22 scored orbits; source pretraining cost and repeated amortization explicitly shown.

## Repro / locations
- `projects/j1j2/experiments/active_verifier_guided_sign_learning/avsl_transfer20.py` (canonical benchmark).
- `avsl_transfer20_results.json`, `avsl_transfer20_validate.py`, `avsl_transfer20_validation.json` (Marshall zero-shot and independent check).
- `avsl_transfer20_opportunity_audit.py/.json`, `avsl_transfer20_oracle_ceiling.py/.json` (post-hoc ceilings, not deployable labels).
- Follow-up local artifacts: `avsl_transfer20_train_post_results.json`, `avsl_transfer20_pooled_results.json`, `avsl_transfer20_fewshot.py`, `avsl_transfer20_fewshot_results.json`, `avsl_transfer20_fewshot_ablation.py/.json`, `avsl_transfer20_fewshot_validate.py/.json`. These were produced in the controlled Python working environment; preserve copies separately if not mirrored into the repository.
- `sign20_training_data.npz` (generated benchmark input) and the archived 20-site preparation scripts in `projects/j1j2/results/scalable_sign_2026-10-18/`.

**Scientific bottom line:** The bottleneck is no longer *just* finding rare potential flips. Transfer across FN/Krylov chambers, reliable local amplitude ratios, and the cost of incremental supervision must all be addressed. Small-chamber adaptivity has a positive controlled signal; mechanism and scaling remain open.

## Additional query-cost audit (important fairness correction)
The **22 verification scores** were budget-matched across methods, but **full end-to-end physics-query costs were NOT**. In the original transfer data's 20 test episodes, the common proposal frontend averages 134.2 root configurations and 2,340.85 candidate symmetry orbits; amplitude-query cache holds 4,395.05 unique states after the initial frontier and **9,224.4** after all candidate bond features were evaluated. Sign-lookups after feature gathering average **15,834.2**. A typical 22-orbit verification introduces ~655 additional unique amplitude lookups (varies by candidate). All arms were charged the *same shared frontend* in our diagnostic, but geometry-only rankings would not need every signed-bond feature computation in a real deployment. Therefore **same number of verification labels is not equivalent to equal total runtime/query cost**. This could erase the small warm-start advantage in a fully cost-matched deployment; do not use current results to claim improved scaling.

New machine-readable summary in `projects/j1j2/experiments/active_verifier_guided_sign_learning/transfer_adaptation_summary.json`; reproducible few-shot source in `transfer_fewshot_repro.py`. Raw high-volume per-episode files remain in the conversation working artifacts if not present in GitHub. The test requires the previously generated 20-site `sign20_training_data.npz` and Marshall source model checkpoint; those binaries are not committed here.
