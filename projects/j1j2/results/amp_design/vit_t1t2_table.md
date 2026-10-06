| arm | seeds | CPU-h | eps<H> (guide or psi) | reduction vs net | eps_FN | reduction vs net FN | w_s |
|---|---|---|---|---|---|---|---|
| T1 Krylov signs, rgntr, N=10000, 6 steps | 2 | 0.53 | 6.56e-05 [6.0e-05,7.1e-05] | 82% | 4.38e-05 | 86% | 1.4e-06 |
| T1 Krylov signs, rgntr, N=100000, 6 steps | 2 | 0.59 | 4.86e-05 [4.2e-05,5.5e-05] | 86% | 3.23e-05 | 90% | 1.4e-06 |
| T1 Krylov signs, hybrid, N=10000, 6 steps | 2 | 0.34 | 1.15e-04 [1.1e-04,1.2e-04] | 68% | 9.68e-05 | 70% | 1.4e-06 |
| T1 Krylov signs, hybrid, N=100000, 6 steps | 2 | 0.28 | 1.09e-04 [1.1e-04,1.1e-04] | 69% | 9.14e-05 | 71% | 1.4e-06 |
| T1 Krylov signs, sr, N=10000, 6 steps | 2 | 0.28 | 8.73e-05 [7.5e-05,1.0e-04] | 76% | 5.73e-05 | 82% | 1.4e-06 |
| T1 Krylov signs, sr, N=100000, 6 steps | 2 | 0.29 | 6.19e-05 [5.5e-05,6.9e-05] | 83% | 4.19e-05 | 87% | 1.4e-06 |
| T1 Krylov signs, c1, N=10000, 6 steps | 2 | 0.28 | diverged (5.9e-01) | - | 1.7e-01 | - | - |
| T1 Krylov signs, c1, N=100000, 6 steps | 2 | 0.27 | diverged (7.6e-01) | - | 2.6e-01 | - | - |
| T1 net signs, rgntr, N=10000, 6 steps | 1 | 0.42 | 2.43e-04 [2.4e-04,2.4e-04] | 32% | 2.19e-04 | 31% | 2.4e-04 |
| T1 net signs, rgntr, N=100000, 6 steps | 1 | 0.42 | 2.54e-04 [2.5e-04,2.5e-04] | 29% | 2.29e-04 | 28% | 2.4e-04 |
| T1 net signs, hybrid, N=10000, 6 steps | 1 | 0.33 | 2.60e-04 [2.6e-04,2.6e-04] | 27% | 2.39e-04 | 25% | 2.4e-04 |
| T1 net signs, hybrid, N=100000, 6 steps | 1 | 0.32 | 2.39e-04 [2.4e-04,2.4e-04] | 33% | 2.23e-04 | 30% | 2.4e-04 |
| T2 loop (Krylov sign + 3 RGN-TR)/iter, 10 iters | 2 | 2.62 | 5.81e-05 [5.0e-05,6.6e-05] | 84% | 3.88e-05 | 88% | 7.3e-07 |
| T2 loop (Krylov sign + 1 RGN-TR)/iter, 30 iters | 1 | 3.03 | 5.50e-05 [5.5e-05,5.5e-05] | 85% | 3.52e-05 | 89% | 3.3e-07 |
| T2 net signs fixed, RGN-TR, 30 steps | 2 | 2.04 | 2.20e-04 [2.2e-04,2.2e-04] | 39% | 2.00e-04 | 37% | 2.4e-04 |
| T2 complex ViT (sign+amp), RGN-TR | 2 | 6.58 | 7.49e-05 [5.8e-05,9.2e-05] | 79% | 4.45e-05 | 86% | 9.7e-06 |
| T2 loop, adaptive lam/N (Krylov sign + 3 RGN-TR)/iter | 2 | 3.41 | 7.54e-06 [6.2e-06,8.9e-06] | 98% | 3.96e-06 | 99% | 0.0e+00 |
| T2 net signs fixed, RGN-TR adaptive | 1 | 3.66 | 1.89e-04 [1.9e-04,1.9e-04] | 47% | 1.75e-04 | 45% | 2.4e-04 |
| T2 complex ViT (sign+amp), RGN-TR adaptive | 1 | 10.16 | 5.76e-05 [5.8e-05,5.8e-05] | 84% | 3.05e-05 | 90% | 1.9e-06 |
| T2 standard VMC (SR, complex ViT, N=4000, lr 0.05, 600 steps) | 1 | 4.60 | 3.28e-04 | 10% | 2.88e-04 | 10% | 2.1e-04 |
| reference: same SR VMC, long run on 16-core `cluster` nodes, at 5 CPU-h | 1 | 5 | 3.21e-04 | 11% | | | |
| reference: same SR VMC, long run on 16-core `cluster` nodes, at 20 CPU-h | 1 | 20 | 2.28e-04 | 37% | | | |
| reference: same SR VMC, long run on 16-core `cluster` nodes, at 50 CPU-h | 1 | 50 | 1.38e-04 | 62% | | | |

**At equal CPU-h** (seed mean, log-interpolated; variational eps: loop/net-sign = <H> of the guide, VMC = <H> of psi)

| CPU-h | loop (adaptive, spi3) | loop spi1 | net signs RGN-TR | complex RGN-TR (all runs) | SR VMC |
|---|---|---|---|---|---|
| 0.25 | 1.14e-04 (68%, n=2) | 9.65e-05 (73%, n=1) | 2.75e-04 (23%, n=1) | 2.80e-04 (23%, n=3) | 3.73e-04 (-3%) |
| 0.5 | 6.11e-05 (83%, n=2) | 5.82e-05 (84%, n=1) | 2.45e-04 (32%, n=1) | 2.16e-04 (40%, n=3) | 3.57e-04 (2%) |
| 1 | 3.52e-05 (90%, n=2) | 5.50e-05 (85%, n=1) | 2.26e-04 (37%, n=1) | 1.43e-04 (61%, n=3) | 3.40e-04 (6%) |
| 2 | 1.70e-05 (95%, n=2) | 5.50e-05 (85%, n=1) | 2.08e-04 (42%, n=1) | 1.00e-04 (72%, n=3) | 3.47e-04 (4%) |
| 3 | 9.02e-06 (97%, n=2) | 5.50e-05 (85%, n=1) | 1.94e-04 (46%, n=1) | 8.34e-05 (77%, n=3) | 3.22e-04 (11%) |
| 4 | - | - | - | 7.33e-05 (80%, n=3) | 3.37e-04 (7%) |
| 6 | - | - | - | 6.74e-05 (81%, n=3) | - |
