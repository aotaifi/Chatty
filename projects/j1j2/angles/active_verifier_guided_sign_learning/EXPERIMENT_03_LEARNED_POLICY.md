# AVSL policy-learning pilot — 18 October 2026

## Research question
Can reward-labelled active candidate selection find more beneficial **complete-symmetry-orbit sign corrections** per costly guide evaluation than random selection or the existing locally observed signed-bond heuristic? This is **contextual-bandit supervised policy learning**, NOT GRPO or DAPO policy-gradient training. The principal intended advance is a scalable rare-sign exploration strategy, not yet achieved.

## Exact setup
- 4×4 periodic spin-½ J₁–J₂ at J₂/J₁=0.5, Sz=0, 12,870 configurations, 441 translation+global-spin-inversion orbits.
- Analytic callable amplitude guide a(x)=exp(-κ Hxx), Marshall signs; no ED or FN amplitude labels. Symmetry-orbit sign reversal is the action; *keep* has zero reward.
- Candidate pool from amplitude-tempered local Metropolis sampling plus one-Hamiltonian-hop expansion. Cheap features consist of observed signed bonds and/or local spin-bond geometry; no orbit ID.
- Exact **frozen-analytic-guide** marginal energy gain of each audited orbit, calculated from explicitly queried neighbor guide amplitudes, is the training reward/acceptance signal. All negative outcomes retained. Greedy acceptance recomputes marginal energy after previous accepted orbits, so coupled flips cannot be counted as additive.
- Train a 13-feature 2×32 tanh MLP binary reward-classification/ranking policy on 3,600 randomly acquired orbit labels, from 45 start-state/κ runs. Of these 151 orbits yielded positive single-orbit gain. 800 AdamW steps, 3 independent initializations. Training includes κ=.75/.85/.9/1.0/1.1; test κ=.8/.9/1.0 and independent walker seeds 2100–2111.
- **Strict orbit holdout:** orbit keys are assigned to held-out (~20%) versus training (~80%) with a deterministic hash. Test reward labels never enter training, including via other walker seeds or κ values. Features can access unlabeled encountered frontier configurations on the test side (transductive).
- Main comparison: B=22 fully audited distinct orbits from identical frontiers, at most 12 accepted greedily, and count every unique guide amplitude queried for frontier expansion and orbit audit. All methods get the *same* frontiers and candidate-budget B; number of distinct amplitude calls differs slightly and is reported. Test has 12 seeds ×3 κ =36 correlated cases.

## Observations
Independent full 4×4 Hamiltonian expectation checks agreed with the crossing-bond formula to ~1e−13 in the first two selected test cases.

### Main (3 trainings, same ranking results; numbers for seed42)
| Selection | Beneficial audited orbits (mean per test, B=22) | Mean **normalized physical energy** improvement | Mean unique test guide queries |
| --- | ---: | ---: | ---: |
| Random | 0.69 | 0.0000951 | 4255 |
| Existing observed signed-bond heuristic | 2.17 | 0.0003304 | 4250 |
| Learned policy | **3.75** | **0.0004604** | 4237 |

Learned vs signed: **23 wins, 13 ties, 0 losses** in the 36 tests. Correlated seed-cluster bootstrap for average normalized gain difference: +0.0001300, 95% percentile interval [+0.0000872,+0.0001699] on these test seeds. This reflects the fixed training population and this particular tiny guide family, not robustness across physical systems.

Training cost (not free): across 45 training runs the walker/frontier/label collection used **222,550 unique-guide-queries summed over runs**, including **52,854 additional verifier queries** after frontier expansion; there were 3,600 reward labels. At 4×4 this **exceeds exhaustive sector enumeration** by a wide margin, although it might amortize over many future tasks. No favorable end-to-end scaling claim follows.

### Ablation: **geometry shortcut**
**Removing ALL encountered-bond features left the learned policy's test performance unchanged** (3.75 positives, mean raw gain 209.34). Observed-bonds-only features (plus κ) gave 3.72 positives, mean raw gain 203.37. Therefore the initial positive result cannot be attributed to learning from walker-observed signed bonds: the policy is largely exploiting simple spin geometry and the analytic-guide form on a very small lattice. This is a central negative finding for the proposed mechanism.

### Smaller-frontier stress test
With only 40 sampled seed configurations and B=18 audit actions, a policy trained on 20 start/κ runs found a mean 2.17 profitable orbits, vs 0.375 for signed-bond heuristic and 0.417 for random, over 24 held-out root/κ evaluations (8 roots ×3 κ). Approx 1,858 test guide queries per learned trial. This is positive for *cheap static geometric prediction*, not evidence of adaptive exploration on FN walkers.

### Out-of-guide stress test
A model trained at κ=.75/.85/.9/1/1.1 was also tested at κ=.7 and κ=1.2 with B=8,12,18 and 40 sampled seeds. On κ=.7 no tested flips were profitable: the model correctly found zero, but so did the heuristic. At κ=1.2 the learned model still found some useful orbits (around 2 per trial); the observed-bond heuristic found almost none. However κ perturbation within one analytic guide family is not true distribution shift to a different FN sign chamber or system size.

## Interpretation and **NO-GO for promoting as scalable RL**
- **What worked:** a very small network can learn a reusable action-value proxy and rank useful corrections on unseen 4×4 symmetry orbits, in an honest held-out test. It surpassed a hand-coded ranking baseline for this guide family.
- **What did not work as intended:** The learned feature ablation shows that **walker-observed bond information was not necessary**. The result resembles learning a small-cluster analytic geometry classifier, not scalable active discovery of rare FN signs.
- **What remains missing:** no training from noisy NQS amplitudes, no actual FN-walker induced state distribution, no longitudinal sign-network updates/recovery, no 6×6/8×8 scaling, no amortized end-to-end query advantage, and no RL policy-gradient optimizer. Reward is from the very guide that supplies features, not an independent exact-FN verifier.
- **Recommended next discriminator:** train/evaluate the same candidate policy on **two different sign chambers** and/or the real 20-site frozen Krylov guide, require a 20-site-orbit holdout and test ablation of geometry, compare true post-FN physical gain and queries. Use a verified FN guide with callable learned signs and amplitudes; do not assume global exact amplitude lookup is affordable.

## Reproduction
In the same directory (Python 3.11+; numpy/scipy/torch):

```bash
python avsl_learned_policy4.py --train-runs 9 --test-runs 12 --steps 800 --seed 42 --out policy4_results.json
python avsl_policy4_audit.py policy4_results.json policy4_audit.json --checks 2
python avsl_policy4_feature_ablation.py
python avsl_policy4_budget_check.py --checkpoint policy4_results.pt --out policy4_budgetcheck.json
```

Requires `avsl_probe4.py` and `avsl_probe4_validation.py` from the same experiment folder. Some script defaults target `/mnt/data`; redirect output arguments when running elsewhere. All promotion criteria should be re-evaluated in a truly unseen guide family and with actual FN walker sampling before calling this an AVSL scaling success.

## Additional frozen-policy second-chamber stress test (separate, NOT clean held-out generalization)
Starting from the first run's accepted learned orbit corrections, change the Marshall sign chamber while keeping analytic amplitudes fixed; then re-evaluate the *unmodified* learned policy on locally available candidates. This is not an FN refresh or a retrained iterative sign net.

- With candidate selection restricted to the **original held-out orbit partition**, all selection methods found zero beneficial orbits at B=22: the first step had already exhausted almost all profitable corrections inside that narrow partition. This test is inconclusive.
- Allowing **all discovered candidate orbits** after the sign update, the learned model found 10.28 positive single-orbit actions per B=22 vs signed-edge heuristic 0.31 and random 0.75 across 36 seed/κ tests. Mean normalized frozen-guide energy improvement was about +0.001963 vs +0.000162 and +0.000112, respectively. Direct full-H energy verification of representative second-round choices agreed at ~1e-13.
- **Critical caveat:** The reopened candidate pool contains orbit identities encountered during the original training, although their sign labels and surrounding environment have changed. Hence the apparent dramatic advantage does **not** establish out-of-support generalization, scalable active learning, or valid FN self-consistent progress. It is a stress-test observation, not a claim of a working iterative RL loop.
- Exact reproduction: `experiments/active_verifier_guided_sign_learning/avsl_policy4_second_round_all.py` and `avsl_policy4_second_round_all_results.json` on GitHub, plus the original trained `.pt` checkpoint kept in the local artifacts. Re-training the network deterministically from the committed script reproduces that checkpoint under matching Python/PyTorch versions.
