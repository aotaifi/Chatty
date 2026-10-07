# Exact baseline: one Lanczos step vs one Krylov sign step (6x6 J1-J2, J2/J1 = 0.5, periodic)

Date: 2026-10-07. Requested by the independent review (`results/review/REVIEW_2026-10-07.md`, finding 3).
Question: at equal hop cost, does ONE Lanczos step beat our ONE Krylov sign step?

Everything is exact (no sampling): fully symmetric sector (k=0, A1, flip+, 15.8M orbit reps), the exact GPU code of
the stall diagnosis (`experiments/stall_6x6/st6_sector.py`), one RTX 2080 Ti, 4 min of GPU time (job 16863817).
Energies are dE = (E - E0)/N per site with E0 = -0.50380965 (RBM+PP: dE = 4.5e-5). Variance is sigma^2 = <H^2> - <H>^2 per site.
Code: `experiments/lanczos_baseline_6x6/` (`lanczos_lib.py`, `test_lanczos.py`, `run_lanczos6x6.py`). Raw numbers: `lanczos_baseline_6x6.json`.

Starting states (the stall-work tables): **psi_P** = D4 x flip projected ViT, own sign (dE 1.319e-4, reproduced);
**ViT(rep)** = ViT evaluated on canonical reps, own binarised sign (dE 2.093e-4, reproduced; the "ViT as in the stall work").
Lanczos step: psi_1 = (1 + alpha H) psi, alpha from the 2x2 generalised eigenproblem in span{psi, H psi} (2 matvecs).
p = 2 (3 matvecs) is listed for the extrapolation. Krylov step: exact energy-optimal threshold on the same amplitude
(sign structure only). FN = exact lattice fixed-node energy with the state as guide.

| quantity (dE per site) | psi_P | ViT(rep) |
|---|---|---|
| start <H> / sigma^2 | 1.319e-4 / 6.8e-4 | 2.093e-4 / 1.52e-3 |
| (2) Krylov sign step, <H> | 1.266e-4 | 2.006e-4 |
| (1) Lanczos p=1, <H> / sigma^2 / alpha | **2.55e-5** / 1.22e-4 / 0.085 | **5.28e-5** / 3.41e-4 / 0.119 |
| Lanczos p=2, <H> / sigma^2 | 8.6e-6 / 3.9e-5 | 1.89e-5 / 1.14e-4 |
| variance extrapolation to sigma^2 = 0 from p=(0,1) / (0,1,2) | 2.2e-6 / 1.5e-6 | 7.7e-6 / 5.1e-6 |
| (3) Krylov then Lanczos p=1 (p=2), <H> | 3.61e-5 (1.36e-5) | 5.36e-5 (1.96e-5) |
| Lanczos p=1 then Krylov sign on psi_1, <H> | 2.54e-5 | 5.21e-5 |
| (4) FN: psi / Krylov guide / Lanczos-p1 guide / Krylov-then-Lanczos-p1 guide | 1.018e-4 / 0.943e-4 / **1.89e-5** / 2.81e-5 | 1.479e-4 / 1.370e-4 / 3.87e-5 / 4.01e-5 |

## Verdict
**Yes, by a wide margin. One exact Lanczos step lowers <H> 20x more than one Krylov sign step.**
- From psi_P the gain is 1.06e-4 (Lanczos) against 5.3e-6 (Krylov). The remaining error is 2.6e-5 vs 1.27e-4, i.e. 5x lower.
- One Lanczos step is worth about 5 exact FN iterations (guide <H> 4.3e-5 after 3, 8.2e-6 after 12) and already beats RBM+PP (4.5e-5) from psi_P.
- It gains 1.6e-4 on the ViT(rep) start (4x lower error than Krylov, 5.3e-5 vs 2.0e-4).
- Krylov then Lanczos is worse than Lanczos alone (3.6e-5 vs 2.6e-5): the sign step is redundant, and Lanczos changes the sign structure itself. Lanczos then Krylov gains only 1.6e-7.
- The Lanczos state also makes a far better fixed-node guide: 1.9e-5 vs 0.94e-4 (Krylov guide) and 1.0e-4 (psi_P).
- Variance extrapolation (Hu-Becca-Parola-Sorella) from (psi_P, psi_1) lands at 2.2e-6, within 5e-6 of E0; three points give 1.5e-6. It is not safe when the start is poor: from the Krylov guide the two-point extrapolation overshoots below E0 (-1.2e-5).

Caveats on "equal hop cost":
- Evaluating psi_1(x) = psi(x) + alpha sum_x' psi(x') costs the same ~72-connected-neighbour evaluations as the local energy that the Krylov step needs, and alpha needs <H^2> and <H^3> (a second hop) from samples.
- psi_1 is not a network. Each use pays the hop again, and writing it back into a network is the projection problem of the stall diagnosis (the update is a 0.01-rms-class smooth correction that L2/SR projections lose).
- So the exact result says the sign step is far from the best use of one hop. It does not say a network trained on psi_1 keeps the gain; the review's baseline (Lanczos on the stored ViT, no retraining) is the number to beat.

## Checks
- `test_lanczos.py`: alpha and Ritz energies/variances equal the dense/explicit Krylov-basis values to 1e-9 on a random dense matrix and on the `Sector` class (synthetic sector, same code path as the GPU runs); alpha is the global minimiser of the Rayleigh quotient of (1 + alpha H) psi; tiny J1-J2 clusters (4x3 dense, 4x4 sparse ED, noisy Marshall start) check alpha, E_p and the variance against dense/explicit values (all pass; the extrapolation is only good when the start is already close, as here for psi_P, not for these crude 4x4 starts).
- The run reproduces the stall-work values: psi_P 1.319e-4, ViT(rep) 2.093e-4, FN 1.018e-4 / 1.479e-4, Krylov 1.2665e-4 / 2.006e-4.
