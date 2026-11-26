# Sign copy loop: ceiling at round 3, 6.7e-5/site above sgn(M1) (2026-11-25, sign angle)

## What was built
- Round 4 of the constant-cost copy (CF class): a BCE refit of CF on its own C-free labels, plus three softened initialisations (runs/j1j2/sign_bad/r4cf, r4cf_soft).
- FN energy grade of CF against n_2 with the branched paired grader (8 pairs, job 18168074; runs/j1j2/sign_bad/branched/cf_vs_n2/result.json).

## Results
- Round 4 kept share q_4 = 0.104 on the leak-free panel (bar 0.40). Soft inits give 0.10-0.12. Predicted round-4 FN gain is 0.4-1.1e-5/site, under the 1.6e-5 resolution.
- CF vs n_2: -2.91(1.19)e-5/site (-2.45 sigma). The first-order estimate said -9.6e-5; CF realizes only 0.30 of it.
- Gap to sgn(M1), per round: 3.23e-3 (Marshall), 5.96e-4, 9.6e-5, then 6.7(1.5)e-5 (CF). 97.9% of the Marshall gap is closed.
- CF and n_3 are equal in energy (+0.6(1.4)e-5), though first order put CF ahead by 3.2e-5.

## Rejected and why
- More rounds: the fit has a fixed point. Labels change on 1.4% of rows, the refit returns nearly the same function, and the rows missed in round 3 stay missed (G_4 1.30e-4 against R_3 1.14e-4).
- Softened head init: the head scale is relearned within 5% of training, so q does not change.
- Round 4 measured on FN (2026-11-26, job 18211780, branched/cf4_vs_cf): -6.4(6.2)e-6/site, 1.0 sigma; at most 1.75e-5 (95%). Gap after round 4: 6.1(1.6)e-5.

## Three things to understand
1. q_k = 1 - R_k/G_k is the share of the exact round gain the copy keeps. The loop dies once the next round's exact gain G is made of the last round's errors R.
2. The first-order energy from the panel is noisy (bootstrap sd ~2.9e-5 per net; half of it on 5-6 flips), not shown to be biased. Grade on FN energy before trusting it.
3. Open, and the PI's call: close the angle at 6.7e-5/site, or pay for a copy that has lower error on the same table. No constant-cost route found one.
