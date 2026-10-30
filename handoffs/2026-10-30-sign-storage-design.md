# 2026-10-30 Sign-storage angle (j1j2): design on paper, no runs

**Built.** A mechanism for storing each loop round's sign in a second network, judged by the exact fixed-node (FN) energy,
not a classification score. Plan file: projects/j1j2/PLAN_sign.md; equations: board blocks 24-26 (artifact
LwUGJoTxGJku3k6mTVgMVP, v51); blind second opinion: Astra cj-344a55c0d108 (/scratch/a/A.Otaifi/astra/sign_angle/).

**Why.** The PI's loop updates the sign every round with one signed Lanczos step. Without a stored sign, round k needs k
nested steps (cost d^(k+1)). The question is whether storage error stays < 0.5e-6/site at 6x6.

**Rejected.** Synthetic injected errors as the first test (mine): Astra's replay of real N=36 sign networks is as cheap
and tests real error patterns. CNN as the deployed class: the step needs the sign at ~d^2 two-swap configurations per
walker move, ~1.6e5 at 14x14. Stacked per-step factor networks: unlearnable (TRIED_IDEAS A10). Taking the sign from the
amplitude network: one network, two jobs.

**Three things to understand.**
1. A sign error costs only on bonds whose sign PRODUCT changed (block 24): a flipped region costs only at its edge. The
   phi^2-weighted wrong-sign weight w misprices errors.
2. A stored sign with floor < 0.5e-6/site already exists: N=36 CNN, bound 1.7e-7 (block 26). Storage itself is not the problem.
3. The open term: the next signed Lanczos step repairs an isolated wrong sign, but scales its magnitude by
   rho = 2|alpha|kappa - 1 (block 25). That distortion, not the sign, may break the budget. Replay R0 (CPU, frozen
   networks, full step, exact FN) decides it. Prediction: ground-state XL net passes by ~10x; composite m2 M net fails,
   with the magnitude part dominant.

Correction made today: my first isolated-error coefficient (2 phi^2 K) was off by a factor 2; the exact bounds are on
board 24. Next: the PI's go for R0.
