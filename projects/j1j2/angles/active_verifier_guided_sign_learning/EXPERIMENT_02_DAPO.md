# Experiment 02 — DAPO / GRPO reward-group filtering (pre-RL diagnostic)

**Date:** 2026-10-18. **Status:** complete software/physics diagnostic, not a trained policy. See `GRPO_DAPO_DYNAMIC_SAMPLING.md` for scientific interpretation and citations.

## Precise question
DAPO discards groups in which sampled outcomes have zero variance (for binary correctness, all-correct or all-incorrect). When truly energy-decreasing sign flips are rare, does this improve **discovery per expensive guide query**, or just optimizer minibatch density?

## Controlled 4x4 experiment
- Analytic, callable a(x)=exp(-kappa Hxx) and Marshall sign, J2/J1=0.5; no exact FN/ED amplitudes available to the acquisition algorithm.
- kappa=0.32 and 0.9; 12 independent seeds, 180 Metropolis draws per seed; each candidate comes from the one-hop frontier.
- B=80 scored *full symmetry-orbit* candidates per arm, grouped into G=4; all observed edge, amplitude and verifier calls counted, including would-be discarded groups.
- Compare random, observed signed-bond priority, and 50/50 interleaving. Reward=positive if the frozen-analytic-guide energy numerator falls on one-orbit flip. Groups selected by DAPO-style **binary mixed success**, no actual policy gradients applied.

| Kappa | Priority | Useful flips among 80 | Mixed groups among 20 | All-negative groups among 20 |
|---|---|---:|---:|---:|
| 0.9 | random | 4.00 | 3.58 | 16.42 |
| 0.9 | observed signed bonds | 5.67 | 4.67 | 15.33 |
| 0.9 | 50/50 | 3.42 | 3.08 | 16.92 |
| 0.32 | all three | 0.00 | 0.00 | 20.00 |

For kappa=0.9 the best arm averaged 5212 unique guide amplitudes vs 5253 for random, including ~4094 common frontier acquisitions; active beat random on *count* in 8/12 seeds (one tie), on summed positive **unnormalized single-orbit gains** in 10/12. Gains from separate proposed corrections are not additive if they are applied jointly.

At kappa=0.32 all orbit flips are individually harmful (independent exhaustive audit of all 441 orbit reps); discarding homogeneous binary groups discards the only available correct label, **keep the sign**. Continuous rewards carry relative information in every group, but mean-centered GRPO advantages can assign a positive advantage to a *less harmful* flip. Example rewards [keep=0, flip1=-3, flip2=-2, flip3=-1] produce a positive standardized advantage for flip3. A no-flip anchor alone does not prevent this; **keep-relative rewards or pairwise keep-vs-flip preferences** are required to avoid rewarding harmful sign actions.

## Theoretical observation
If binary beneficial-flip probability on a candidate is p and G=4 candidates are independently sampled, the fraction of groups retained is P_mix=1-p^G-(1-p)^G. For p->0, P_mix ~ Gp. **Resampling until a desired gradient density does not improve physical beneficial-flip discovery per attempted physics query**; query costs of all rejected proposals must be included.

## What this does not establish
- No RL policy/network was trained; all candidate choices are fixed heuristics.
- This uses an intentionally simple 4x4 amplitude guide, not FN walkers or a realistic neural guide.
- A selected correction is validated at **frozen guide amplitude** only; it may fail after full FN projection.
- Probe selection is not proof of size consistency or large-N sample efficiency.
- The conserved magnetization basis, global sign chamber and symmetry orbits are known explicitly for 4x4, but not used as an oracle of beneficial flips in acquisition.

## Reproduce
From `projects/j1j2/experiments/active_verifier_guided_sign_learning/` with Python and NumPy:
```bash
python avsl_probe4.py --kappa .9 --B 40 --seeds 12 --n-seeds 180 --out /tmp/avsl_priority.json
python avsl_dapo_groups4.py --trials 12 --budget 80 --out /tmp/avsl_dapo_groups4_results.json
python validate_4x4_energy.py
```
Saved per-seed reproducible summary: `avsl_dapo_groups4_summary.json`. Code and full-trial summary were rerun from the GitHub `main` branch on MacBook-Air-2.local (isolated `/tmp`, working tree untouched).

## Next *hypothesis* to test
Train a contextual exploration policy with **graded, keep-relative** reward on scarce labeled root/frontier groups, using actual walker-query costs and a random/tempered exploration floor. Compare against a fixed signed-bond policy, track unique newly discovered profitable orbits and false positives on held-out orbit identities, and do not accept any flip without independent variational auditing. Before full GRPO, quantify proposal support: does the induced policy ever find the rare defect regions? If not, no clever group normalization will solve the obstruction.
