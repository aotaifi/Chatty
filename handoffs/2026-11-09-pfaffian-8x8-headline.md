# Pfaffian angle: 8x8 bare grade, R0 and the p=1 headline (2026-11-09)

## What was done
- R1's bare c=0 grade: E_1 -0.498803(32) against B_0 -0.498755(39) through the same code (E_1 - E_0 = -4.95e-5, 6.5 sigma). That is 11 sigma below p=0.
- R0: the new p=1 driver walks exactly like M1's own code from the same start (to 5e-14). The 80-block R0 run was not quotable.
- The p=1 headline (Paderborn 4186187 + 4186886), same code for both arms, M 256:

  | arm | E / site |
  |---|---|
  | A = B_1 + its own step | -0.4989545(34) |
  | B = B_0 + M1's step | -0.4989390(45) |

  - A - B = -1.55e-5(56), 2.7 sigma; A - M1 = -8.5e-6(61), 1.4 sigma.
  - The registered kill fires (it needs more than 3 sigma on both).
- Running: the c=0 attribution control (theorie A40 17852869) and the K=3 round-6 test at 6x6 (A40 17871864/5).

## Why it was built this way
- The headline is paired in code (A against B through one driver). M1's reference came from another code, and R0 sat 2 sigma off it.
- B_1 got its own alpha/E/w by M1's VMC rule, as registered ("pipeline unchanged, network swapped"). Its w is rotated 28.5 deg from M1's, so M1's constants would have been wrong for arm A.

## Alternatives rejected
- Extending R0 to quotable: the trajectory-identity test is stronger and cheaper.
- M=1024 for R0: M1 used 256, so the population-control bias matches only at 256.
- Racing copies: a race needs a cancel, and a cancel needs the PI's word in the angle's own session.

## Three things to understand
1. The refit and the Lanczos step fix largely the same error. The step gains 1.83e-4 on B_0 but only 1.52e-4 on B_1. Only about a third of R1's 4.65e-5 bare gain survives p=1.
2. p=1 local energies are heavy-tailed: 1 + alpha(E_L - E) goes to 0 near psi_1's nodes. R0 and arm B disagree by 3.8 sigma by naive errors, so the block errors may be too small.
3. The express QoS cap (1800 H100-min a month) counts REQUESTED wall time while a job runs. Request tight limits, or a second job cannot start.

State and job ids: projects/j1j2/PLAN_pfaffian.md, "STATE (2026-11-09 21:30)".
