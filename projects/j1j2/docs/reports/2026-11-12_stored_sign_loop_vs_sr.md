# Stored Lanczos sign loop beats SR training of the same sign network (8x8, bad start)

- Finding: from the Marshall sign at 8x8 (J2/J1 = 0.5, amplitude fixed to |M1|, c = 0 fixed-node GFMC), three rounds of "one Lanczos sign step -> store in a CNN -> deploy the CNN" lower the FN energy every round, and beat SR (variational) training of the same CNN at matched compute by 1.06e-3/site (59 sigma).
- Energies per site (2 chains each, quotable.py QUOTABLE):
  - Marshall: -0.495522(23) (sign_bad/r1/fn_marshall)
  - loop round 1: -0.498158(17) (r1/fn_net); round 2: -0.498658(9) (r2/fn_net); round 3: -0.498709(10) (r3/fn_net)
  - SR-trained phase net, hard sign, 7.7 A40-h = loop rounds 1-3 labels + fits: -0.497651(15) (ctrlC/fn_net_slice)
  - reference: sgn(M1) on |M1| -0.498754(2.3e-6) (sign_node/prod, F arm)
- Gains per round: 2.64(3)e-3, 5.00(19)e-4, 5.1(13)e-5; gap to sgn(M1) contracts x5.4, x6.2, x2.1 (slowing; storage loss suspected, 64x10 net lean -2.2(14)e-5, inconclusive).
- Cost: one round = ~1-1.5 A40-h labels (exact one-step sign on samples + one-swap neighbours, offline) + ~0.75 A40-h CNN fit; deploying the stored CNN costs 3.3x a plain FN step. The exact step at walker time would cost ~150x (not used).
- Why SR lost (hypothesis): its hard sign moved only 1.6% of configurations from Marshall (Marshall trap, arXiv:2002.04613, 2011.11214); the loop supplies each improved sign directly and storage is supervised classification.
- How this could be wrong: one SR setting (minSR, eta 0.05, 0.3 rad start), still falling slowly; fixed amplitude, c = 0 (no Jacobi head); one size (8x8); one network class (64x6 CNN, capacity-limited: held-out label disagreement ~10%, chance in the deepest decades). Controls passed: labels vs direct recomputation (1e-10..1e-11), Marshall-in-place reproduces previous labels exactly, wrong alpha changes labels, FN generic path = BareM1Sign bitwise.
- Not shown: that the loop goes BELOW a strong network's own sign (the angle's main goal; T-STRONG pending), or that it scales beyond 8x8.
- Records: PLAN_sign.md "T-BAD FINAL" through "CONTROL C RESULT"; TRIED_IDEAS A50.
