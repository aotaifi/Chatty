# Literature-to-physics transfer ledger — 2026-10-18

**Scope:** public methods, not claims about private/current internal model training recipes. Separate what papers actually demonstrated from what is merely an analogy to frustrated quantum-magnet sign learning.

## 1. OpenAI — Let's Verify Step by Step / process supervision (2023)
OpenAI, "Improving mathematical reasoning with process supervision" (May 31, 2023):
https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/
Paper: https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf

**Actually demonstrated:** rewards for intermediate steps can outperform outcome-only reward models on math problems; paper also describes *active learning* reducing annotation cost by selecting valuable completions for feedback.
**Physics translation:** prioritize expensive Krylov/FN audits for configurations/orbits where a step-level label is valuable; label `candidate → verified sign correction`, rather than use only noisy final energy as a training signal.
**Mismatch:** our approximate-amplitude Krylov field is not a ground-truth human label. A reduction in frozen-guide energy is not a guaranteed reduction in reoptimized FN energy. Demand independent evaluation and quantify false positives.

## 2. OpenAI — Weak-to-strong generalization (2023)
https://openai.com/index/weak-to-strong-generalization/
**Actually demonstrated:** stronger models can sometimes generalize beyond weak training labels; behavior depends on task and method, with explicit limitations.
**Physics translation:** learn a composite sign model from *partial*, noisy local teacher labels without assuming the labeler is perfect; preserve old-guide predictions elsewhere.
**Mismatch:** without additional inductive bias/validation, a network can amplify false Krylov flips. No guaranteed weak-to-strong theorem for frustrated sign functions.

## 3. Anthropic — Constitutional AI / RLAIF (2022)
https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
**Actually demonstrated:** generated critique/revision and AI feedback can replace some costly human labels in training an assistant.
**Physics translation:** iterative sign proposal → independent critic/verifier → revision → replay. Keep uncertainty and version metadata so an erroneous target is not permanent.
**Mismatch:** do not call a self-scored Krylov rule an independent verifier. Unlike preference modeling, frozen-guide Hamiltonian calculations offer a precise but *limited* variational test.

## 4. Anthropic — Reward tampering / reward hacking (2024–25)
https://www.anthropic.com/research/reward-tampering
https://www.anthropic.com/research/emergent-misalignment-reward-hacking
**Actually demonstrated:** optimizing a flawed proxy can produce exploitative shortcuts rather than intended behavior.
**Physics translation:** if we train on amplitude-guided predicted gain, the learner may preferentially generate states on which its own imperfect guide predicts large, fictitious gains. Audit on held-out/independent physical-H/FN measurements and test predicted versus actual gains.
**Mismatch:** no social/agentic deception is implied by a frustrated spin model; the general mathematical issue is optimizer exploitation of an imperfect surrogate.

## 5. DAgger — Ross, Gordon, Bagnell (AISTATS 2011)
https://proceedings.mlr.press/v15/ross11a.html
**Actually demonstrated:** dataset aggregation for supervised imitation learning on policy-induced state distributions; established guarantees require expert and assumptions.
**Physics translation:** after each sign-network update, collect *new* states from the FN distribution induced by that sign model; append to a replay dataset while revisiting previously uncertain labels.
**Mismatch:** classic DAgger assumes a queried, sufficiently good expert. Krylov with approximate amplitudes is a fallible teacher; DAgger guarantees do not automatically carry over.

## 6. Prioritized Experience Replay — Schaul et al. (2015)
https://arxiv.org/abs/1511.05952
**Actually demonstrated:** more frequent replay of high-priority experiences improves RL data efficiency on tested benchmarks.
**Physics translation:** replay verified high-gain bond/correction examples plus past false flips, *with a substantial ordinary/tempered floor*. Version labels and correct for bias where weighted training would otherwise optimize the wrong distribution.
**Mismatch:** a correction's rarity alone does not imply large physical energy importance.

## 7. ByteDance Seed / Tsinghua AIR — DAPO / dynamic sampling (2025)
https://dapo-sia.github.io/
https://github.com/BytedTsinghua-SIA/DAPO
**Actually demonstrated:** dynamic sampling and diversity-focused techniques improve training efficiency of reasoning-model RL; in a particular grouping scheme, batches with all-positive/all-negative rewards have little learning signal.
**Physics translation:** prioritize *informative* orbits on which conservative sign-update judgments disagree or the predicted gain is near a threshold; do not spend all compute revisiting high-confidence unchanged orbits.
**Mismatch:** in our sign problem, rare correct signs and high-confidence no-flip examples are valuable *anchors*; filtering all easy examples would encourage catastrophic false flips.

## 8. Hypotheses not established by any of these papers
- That active sampling can overcome extremely small *proposal probability* for relevant defect regions.
- That a locally verifiable *fixed-guide* energy correction predicts an improvement after FN projection.
- That an approximate learned amplitude gives calibrated enough bond ratios to label rare sign flips.
- That LLM-style uncertainty/replay yields a size-consistent sign correction on 6x6 and 8x8.
- That optimizing sign classification generalizes beyond the original teacher and simultaneously improves variational physics.

## Working synthesis (ranked by plausibility)
1. **On-policy frontier discovery with tempered restarts + baseline** (DAgger + active learning). Reasonably transferable; easiest falsification at equal *unique guide-evaluation cost*.
2. **Independent local verification with uncertainty calibration** (process supervision + reward-hacking warning). Necessary for correctness, but reliable independent labels may be too expensive; current key challenge.
3. **Versioned replay, keep/no-flip anchors, composite sign function** (PER + weak-to-strong). Plausible mitigation for forgetting and misleading correction factors, but prior sign-learnability tests already showed large networks can learn composite signs; not a novel result.
4. **Direct training of a local correction field rather than full amplitude**. Interesting if one can find a source of supervisory information that does *not* require the expensive amplitude object; otherwise just moves the problem.

## Literature position
AVSL is a *physics-specific research hypothesis inspired by these methods*. Novelty, if found, must lie in a genuinely scalable physically justified proposal/verification rule with improved sign energy per query, **not** in applying existing active-learning or replay terminology to an ordinary CNN.
