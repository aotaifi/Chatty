# Reward design for AVSL (first-pass, 2026-10-18)

## Is this "RL: reward for lowering energy, punish for raising energy"?
**Yes as an eventual policy-learning formulation, but start with a contextual bandit.** Reward should measure the effect of the *entire accepted sign action under a specified guide*, not mark each Krylov-proposed flip as good by definition. The first useful policy chooses *which configuration/orbit to query next*, not the ground-state sign of every configuration.

## Policy and environment
- Context = FN walker state, encountered signed H bonds, local frustration/uncertainty features, replay statistics, current callable guide version.
- **Acquisition action** = which symmetry-respecting candidate orbit / neighborhood to investigate next. It incurs guide-neighbor evaluations and perhaps verification cost.
- Optional **commit action** = flip/keep one orbit or a small group. If a neural sign policy generalizes beyond inspected orbits, measure the full changed sign function: the local reward for individually flipped orbits is NOT its guaranteed collective reward.
- Transition = recompute FN weights and walker distribution with the updated sign and amplitude model; delayed effects cause credit assignment.
- Terminal metric = independent physical Hamiltonian energy of the full callable wavefunction, FN energy where available, and query/walker budget. Intensive size-consistent diagnostics as N increases.

For a fixed amplitude a and two guide sign functions s and s', define
  R_frozen = E[s,a] - E[s',a] - lambda * cost(action).
This is exactly physically meaningful at frozen a if computed from the *entire s'* and a, but does NOT guarantee that reoptimizing FN gives better energy. Define the *delayed* reward
  R_FN = E_FN[s,a_updated] - E_FN[s',a'_updated] - lambda * cost,
when independent FN estimators or conservative confidence bounds permit.
Avoid claiming a noisy self-consistent energy estimator is a deterministic verifier.

For a symmetry-orbit action F with frozen a, its unnormalized energy gain is
  G(F) = 4 sum_{x in F, y outside F} H_xy s_x s_y a_x a_y.
The denominator Z=sum_x a_x^2 is unchanged under pure sign flips. A **marginal** gain must be recomputed after earlier flips due to interactions among actions.

## Why not policy gradient first?
1. There is a cheap exact local reward on a *callable* frozen guide for explicitly queried actions (conditional on enough neighbor queries); use it for supervised ranking / contextual bandit before noisy REINFORCE/PPO-style updates.
2. Frozen guide rewards can be sparse. Our 4x4 analytic guide kappa=0.32 is a verified **zero-single-orbit-positive** case, even though this says nothing about exact signs. No immediately rewarding action means an immediate-reward RL policy cannot bootstrap from this reward alone.
3. Our high-value objective is *long-term FN energy per computation*, but its delayed FN projection and estimator errors are hard; full RL may become useful when bandit-style immediate rewards demonstrably miss productive long-horizon actions.
4. A flip-policy proposal is not the same as a Krylov label or an independent verifier. Krylov is a teacher/proposer; the physical energy difference is the reward for the joint accepted action. Model-based self-critique without independent validation can amplify amplitude errors.

## Required controls before an RL learner
- Compare passive, signed-bond active and disagreement-driven acquisition with the same unique evaluated states and verification budget.
- Include a stay-put action, imperfect-ratio and joint-action counterexamples, and the no-reward basin.
- Treat exact ED/FN data as post-hoc scoring only, never labels available to the acquisition agent.
- Keep diversity/replay to avoid repeated visit to easy high-amplitude states and overfitting false positives.
- At 6x6/8x8 explicitly verify the action representation is size-consistent; whole-configuration global Krylov threshold is not automatically size-consistent.

## Sources
- OpenAI Let's Verify Step by Step (process vs outcome supervision / active feedback): https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/
- OpenAI Reinforcement fine-tuning guide (programmable graders, bootstrap requirement): https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning
- Anthropic Constitutional AI (AI feedback / critique-revision) https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
- Ross et al., DAgger, AISTATS 2011 (on-policy dataset aggregation) https://proceedings.mlr.press/v15/ross11a.html
- ByteDance/Tsinghua DAPO, 2025 (dynamic sampling) https://dapo-sia.github.io/
