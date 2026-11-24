# The 75%-per-round sign-copy bar has no supported route at constant cost (2026-11-24, sign angle)

## What was built
- Field copy: the same CNN fitted by MSE to the normalized field v = h/Lambda, deployed as sgn of its output (Astra route 2).
  Run: runs/j1j2/sign_bad/tfield, A40 job 18149700.
- Diagnosis from saved outputs: tfield/diag/diag_err.py -> diag_err.json.
- Blind Astra review of what is left: runs/j1j2/sign_bad/astra_wall/ASTRA_wall.md.

## Results (leak-free 60k panel; one seed, no error bars)
- Field copy: q 0.273, worse than the current classifier copy CF (0.457). Error rms 0.075 against the 0.045 the bar needs.
- On flips, the error points back to the old sign (+0.081 against mean |v| 0.071); 54% of flips come out with the wrong sign.
- Gaussian noise on the exact v gives q 0.78 at rms 0.045 and 0.36 at rms 0.075. The map from rms to q holds; the copy is not accurate enough.
- Oracle bounds (Astra, re-computed by me):
  - Choosing per row between CF and the field copy: q 0.567.
  - Repairing every CF missed flip, false flips untouched: q 0.732.
  - So both kinds of regret must fall by more than half.

## Rejected and why
- Residual fit against an old-sign baseline: for linear least squares it is the same projection, so q does not change.
- Margin-filtered map (Astra route 3): a perfect copy reaches only 0.726.
- Longer training: rms ~ steps^-0.064, so rms 0.045 needs about 2900x the steps.

## Three things to understand
1. q is an energy share, q = 1 - R/G, with regret R weighted by |h|. The bar q >= 0.75 means R <= G/4.
2. The failure is the flip correction lost in the fit. Write v = b - 2bF, with b the old-sign baseline and F the flip indicator: the net keeps b and loses -2bF on flips, which sit one exchange away from keeps.
3. Open, and the PI's call: keep the per-round bar (then the angle closes negative) or judge cumulative gain along the deployed copy.
   That route is Astra route 1. It needs the copy not to degrade over rounds, which is unmeasured.
