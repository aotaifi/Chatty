# Sign angle, 2026-11-15: exact deep sign costs energy at 8x8 (G(-5) kill + T-VARDEC)
**What was measured**
- G(-5) (cip 17968152): s306 guide vs. same amplitude with the exact one-step Lanczos sign at |psi|^2 decades <= -5. Paired FN gives
  Delta = +6.4(3.5)e-6/site (cost lean, not quotable). The registered kill fired: the gain is below 5e-6 at 2 sigma. Astra confirmed it.
- T-VARDEC (cip 17994730): the exact trial-energy (VMC) change of the same flip set is +2.44(25)e-5/site, with 98.6% of the flipped weight
  in a "window" where flipping raises the energy. The local-field rule F' (flip only where E_L > H_xx) gains only 5.6e-7/site.
**Why**
- PI goal: a better sign every round with a scalable method. We tested whether the exact sign in the low-amplitude tail is worth storing at 8x8.
  It is not, so both stored-mask routes at 8x8 are closed operationally (PLAN_sign.md "G(-5) RESULT", "T-VARDEC RESULT").
**Rejected alternatives**
- Hybrid "Lanczos AND E_L > H_xx": identical to F' at 8x8 (H_xx - E >= 15.9 > 1/|alpha| for every state), and its trial gain is 9x below the bar.
- Symmetry-correct stored sign (A51): closed 11-13.
- G_max (a first-order bound on ANY deep sign change, forward walking, ~4-6 A40-h): parked until the PI decides on T-BAD.
**Three things to understand**
1. The Lanczos sign flips x when E_L(x) > E + 1/|alpha| = E + 5.68. A single flip lowers the trial energy only if E_L(x) > H_xx, and H_xx - E >= 15.9 always.
   So every flip with E + 5.68 < E_L < H_xx raises the energy. Measured: those are almost all the flips.
2. Trial energy is not FN energy: the FN cost (+6.4e-6) is 0.26x the trial cost (+2.44e-5). The FN weight phi differs from |psi|^2 (Astra: Q(phi) bounds).
   A 3-state counterexample shows that a flip that lowers the trial energy can raise the FN energy.
3. G(-5) is not a ceiling over all stored masks: an imperfect mask that omits harmful flips could beat it. What is closed is storing the Lanczos labels, plus F'.
Files: PLAN_sign.md (sections above), TRIED_IDEAS A52/A53, runs/j1j2/sign_strong/{g5/cip,vardec/out/prod2}, /scratch/a/A.Otaifi/astra/sign_g5/ASTRA_g5.md.
