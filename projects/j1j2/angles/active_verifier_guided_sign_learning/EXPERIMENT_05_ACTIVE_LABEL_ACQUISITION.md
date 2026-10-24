# AVSL experiment 05 — Active label acquisition under chamber shift (2026-10-24)

## Question
After training a candidate-ranking model on Marshall signs, can we adapt it to post-Krylov signs with *fewer expensive sign-verifier labels* by choosing informative new-chamber orbit examples rather than random training examples?

**This is NOT on-policy active exploration**, a GRPO/RL policy, or proof of scaling. It is an **offline active labeling** control using the same available frontier and on-demand *exact* 20-site FN amplitudes. A policy that picks which states the walkers actually visit has not been tested.

## Protocol / controls
- J1-J2 tilted 20-site torus, J2/J1=0.5, dimension 184,756, 4,654 symmetry orbits.
- Initial ranking network from 1,980 verified Marshall-chamber training labels (`avsl_transfer20.py`); all four arms start with this **same expensive pretraining**.
- Same 18 new-chamber training root episodes, common amplitude-tempered toy Metropolis + one-hop candidate frontier, same orbit split and candidate supply. Within each arm, exactly **20 previously unlabeled post-Krylov orbit candidates per episode**, 360 new-chamber labels total.
- Four acquisition arms: random; highest predicted success under source policy; closest to source 50%-probability decision boundary (uncertainty sampling); observed signed-bond score. Include *negative/no-flip labels* in training, do not dynamically discard zero-success episodes.
- Train same warm-start MLP with weighted BCE for 700 AdamW steps; two training seeds (19,29). No ED ground-state signs in training. The target is whether *a single symmetry-orbit sign flip gives a positive frozen-amplitude gain*, not true Krylov labels.
- Evaluate all arms on the same 20 disjoint held-out-orbit root episodes; 22 candidate orbit verifications per episode, then greedy joint energy-gated combination; **actual frozen-guide H** directly checks selected flip gain. All methods same label budget, not necessarily precisely the same number of amplitude lookups because queried neighbors differ.

## Results
| Acquisition at 360 new-chamber labels | Positive labels found | Mean held-out frozen-a energy gain | Gain / random |
|---|---:|---:|---:|
| Random | 6 | 0.000013020 | 1.00× |
| Source top-score | 59 | 0.000011275 | 0.87× |
| Source uncertainty | 15 | **0.000019189** | **1.47×** |
| Observed signed-bond score | 2 | 0.000006650 | 0.51× |

Uncertainty beats random on **30 / 40** paired seed × episode comparisons (1 tie, 9 losses); paired 20-episode mean gain delta +0.000006169, naive 95% bootstrap interval [0.00000285,0.00000978] when sampling episode roots and averaging across the two net seeds. **This uncertainty interval is descriptive only**: root episodes on a finite shared Hamiltonian are correlated, and two training seeds are inadequate for robust generalization statements. Direct-H validation checked 40 selected update instances with max absolute energy error 2.30e-14.

**Cost:** 360 verifier labels in every arm. Summed per-episode *unique within each episode* amplitude queries (can be repeated across episodes) including label expansions:
- random: 175,614; extra labeling-relative-to-common-frontier queries 10,559
- source top: 176,302; extra 11,247
- source uncertainty: 176,077; extra 11,022
- signed bonds: 174,632; extra 9,577

Thus all four acquisition arms still make roughly **175k amplitude queries**, comparable to the complete 184,756-state basis, just to acquire 360 labels. Although the uncertainty arm improves energy ranking **at approximately equal label/query cost to random**, its absolute cost and oracle backend make this approach non-scalable as implemented. The extra 1,980 label/source frontier pretraining cost is excluded from those numbers and must be charged in any full lifecycle comparison.

## Interpretation
- **Meaningful:** For a fixed candidate pool, uncertainty sampling of a fallible source ranking policy can train a better *target-chamber ranker* than random labeling with equal budget. This is related to active learning / DAPO's emphasis on informative groups, but **not GRPO/DAPO itself**: we used offline binary labels and BCE, not policy gradients or sampled-reward groups.
- **Counterexample to maximizing positives:** Source-top acquisition found 59 positives vs uncertainty's 15, yet learned a worse ranker. Positive-label prevalence is not the scientific objective.
- **Significant caveat:** The positive result is all on a tiny 20-site fixed-amplitude oracle instance with a geometry-dominated learned representation; active acquisition across novel models, FN chambers and lattice sizes remains open.
- **No physics guarantee:** Frozen-a sign energy improvement does not certify improved converged FN energy, exact sign correctness, or global sign-network generalization.

## Decision
Promote *uncertainty-guided acquisition* to a candidate exploration arm, **not** a proven scalable solution. The next decisive test is **128-walker on-policy frontier generation with callable imperfect amplitudes and signs**, compare uncertainty, passive tempered+neighbor, geometry baseline and signed-bond heuristics under identical *total* query and training budgets, and measure independently validated fixed-a / post-FN energy. If it does not outperform simple baselines in **energy improvement per unique expensive query**, stop tuning another offline 20-site oracle classifier.

## Reproduction and source
`projects/j1j2/experiments/active_verifier_guided_sign_learning/avsl_transfer20_active_acquisition.py`, `avsl_transfer20_active_acquisition_validate.py`. Prior source policy requires `avsl_transfer20_results.pt` regenerated by `avsl_transfer20.py`. The full run data (`avsl_transfer20_active_acquisition_results.json`) and direct-H validation JSON were also saved as conversation artifacts. Python 3 / numpy / torch. The scripts use `/mnt/data` scratch paths.
