# GRPO / DAPO dynamic sampling: relevance to FN–Krylov — 2026-10-18

## Source-accurate summary

- **GRPO**, introduced in DeepSeekMath (Shao et al., 2024, arXiv:2402.03300), uses **group-relative standardized rewards** for multiple sampled responses to the same prompt, avoiding a separate value-model/critic. This is an update method, not a mechanism that discovers states outside the proposal policy's support.
- **DAPO** (Yu et al., ByteDance Seed + Tsinghua, 2025, arXiv:2503.14476; NeurIPS 2025) extends GRPO with *Clip-Higher*, *dynamic sampling*, token-level policy-gradient loss and *overlong reward shaping*. Dynamic sampling resamples groups with mixed **binary outcome rewards**, excluding homogeneous-reward groups that give zero standardized advantage. The authors' public implementation is based on verl; published AIME results are about language-model reasoning, not quantum Monte Carlo.
- **OpenAI process supervision** (Lightman et al., 2023) and active feedback argue for verifying valuable intermediate decisions, but in AVSL an approximately computed Krylov label is not independently verified unless assessed with a separate physical energy evaluation.
- Related primary references:
  - https://arxiv.org/abs/2402.03300
  - https://arxiv.org/abs/2503.14476
  - https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html
  - https://github.com/BytedTsinghua-SIA/DAPO
  - https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/

## Mapping: task, policy, group, reward, verifier

A **task** is a starting spin configuration together with the current sign and amplitude guide; an **action** is a bounded path through the Hamiltonian-neighbor graph terminating in a candidate symmetry-orbit sign correction (or **no flip**). Sample K proposals for the same task/root under a policy. Every proposal incurs exploration and sign/amplitude query cost. A reward is a physically meaningful, independently evaluated **frozen-current-guide energy gain** when available; its value *does not guarantee* true FN or sign-ground-state progress after relaxation. If a critic uses the same approximate amplitudes as the candidate policy, predicted gains are not independent verification.

Given rewards R_1,...,R_K, group-relative advantage is A_i=(R_i - mean(R))/std(R) when std>0. A zero-variance group has no such gradient. With **binary success rewards** R_i=1[gain_i>0], the probability of an informative group under independent proposals of per-candidate success probability p is

```
P(mixed | K,p)=1-(1-p)^K-p^K.
```

For rare corrections (p small), this is ~Kp; each surviving useful group can require O(1/(Kp)) attempted groups. **Filtered proposals still consume all physics queries**. Dynamic filtering therefore does not solve the original problem of discovering low-probability states, nor does it make the entire algorithm cheaper per successful correction.

For **continuous energy gains**, even if all K actions are harmful, the rewards often have nonzero variance. Their relative advantage can accidentally assign a *positive* update to the least-bad harmful flip. **Include the zero-reward “keep” action as an explicit group anchor**, but NOTE that *ordinary mean-centered GRPO still can increase a harmful action's probability*: for rewards [keep=0, flip1=-3, flip2=-2, flip3=-1], the least-bad flip (-1) has a **positive** standardized advantage (~0.447). The anchor makes keep the best reward but does **not** certify all optimizer gradients. For physical sign acceptance, use a **keep-relative advantage** A_i=(R_i-R_keep)/scale=R_i/scale, a keep-versus-flip ranking objective, or explicit verified positive-gain gating. All-negative groups are essential *no-flip training examples*, not useless physics data. A reward estimated using the same inaccurate amplitude model is still not an independent verifier.

## First controlled diagnostic — software/physics sanity test, NOT RL training

Code: `projects/j1j2/experiments/active_verifier_guided_sign_learning/` (or local backup `avsl_probe4.py`, `avsl_dapo_groups4.py`). 4x4 periodic spin-1/2 J1–J2, J2/J1=0.5, S^z=0, **analytic callable amplitude** a(x)=exp(-kappa*H_xx), Marshall sign table given analytically. Root states are sampled by amplitude-tempered Metropolis and neighbors added on-demand, with unique amplitude queries counted. Whole translation/spin-flip symmetry-orbit single corrections are evaluated at frozen guide, without ED/FN amplitudes in the selection stage.

Compare random candidate order, rank from observed signed bonds, and 50/50 interleaved ranking. For each of 12 independent root seeds, examine B=80 orbit candidates grouped as G=4, hence 20 groups. Record single-orbit positive gains, fraction of groups with mixed binary outcomes, and continuous anchored-reward variance; include rejected-group query costs. Revisit other guide kappa values because a guide already locally stable offers no profitable single-orbit flips.

| Diagnostic setting | Candidate acquisition | Profitable tested orbits / 80 | Mixed binary groups / 20 | All-negative groups / 20 |
|---|---|---:|---:|---:|
| kappa=0.9, 12 seeds | random | 4.00 | 3.58 | 16.42 |
| kappa=0.9, 12 seeds | observed signed bond priority | 5.67 | 4.67 | 15.33 |
| kappa=0.32, 12 seeds | either | 0 | 0 | 20 |

At kappa=0.9, active acquisition discovered +1.67 profitable single-orbit candidates per seed (8/12 pairwise wins, 1 tie) and its *sum of positive single-orbit numerator gains* was larger in 10/12 paired trials. Unique amp queries averaged 5,212 versus 5,253 for random, including ~4,094 initial common frontier queries. Improvement is modest, small-sample, and **not** independent replication of the 20-site FN results.

At kappa=0.32, 4x4 *whole-orbit, current-guide* gains were all <=0, checked also over all 441 symmetry orbits by a post-hoc enumeration audit. The DAPO binary filter would discard all groups; continuous reward with **explicit keep=0** remains informative for learning *not* to flip. At kappa=0.9, only 23 of 441 full-space single-orbit flips help (post-hoc only). Full-H independent evaluation of greedy selected orbit flips agreed with crossing-bond energy formula to numerical tolerance; the large unnormalized gains in these scripts are **energy numerators** and should not be compared across kappa without normalizing by sum_x a(x)^2.

**Interpretation:** Dynamic sampling can make a *chosen optimizer minibatch* contain more group-relative gradients, but after paying for all rejected candidates it gives no automatic per-oracle-query discovery improvement. A physically safe no-flip anchor often makes nominally “uninformative” all-negative groups useful. Our next stage is an actual **learned exploratory policy**, not a GRPO library integration.

## Decision on whether to train GRPO today

**NO** full GRPO/DAPO sign policy yet. Our action space comprises a correlated graph of spin configurations and symmetry orbits, and our verifier has not been calibrated for imperfect amplitudes. An imitation/active-learning *frontier selection* experiment should precede complex policy gradients.

**First policy comparison to implement next:** Equal **total** guide-query/edge-evaluation budget, M independent start states, K=4 candidate *exploration rollouts* per start, include “do nothing” (reward zero), evaluate one verified orbit per rollout when budget permits. Arms: fixed tempered/hop exploration, signed-bond-prioritized frontier, and a learned selection policy trained from retained graded rewards (all-negative groups retained). Report new positive **non-overlapping accepted corrections** per cost, true frozen-a energy, false-positive loss under perturbed guide, and held-out roots. Record effects after updating signs and repeating FN; add size-consistent local Krylov-region proposals from `DESIGN_scalable_loop_2026-10-13.md` before claims about large N.

## Scientific caveats and falsifiers
1. DAPO dynamic filtering does not enlarge proposal support. If the policy never visits the rare boundary, no training label is recovered.
2. Action policies must respect spin/inversion/translational symmetries; the source experiment's orbit-based gain formula is valid only for these fixed symmetries of the guide.
3. Reward should distinguish proposed Krylov flip, accepted *energy-decreasing* flip, and eventually validated post-FN change; they are not identical.
4. Gaussian uncertainty or replica disagreement is not a guarantee on a local amplitude ratio.
5. In all numerical studies, compare against unchanged sign, baseline tempered-neighbor sampling and fixed-query-cost alternatives. Cross-check physical Hamiltonian energies on accessible sizes before moving to 6x6/8x8.
