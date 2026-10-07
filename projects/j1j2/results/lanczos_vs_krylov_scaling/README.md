# One Lanczos step vs one Krylov sign step as N grows (exact, J1-J2, J2/J1 = 0.5, periodic tori, N = 16..36)

Date: 2026-10-07. Question: on 6x6 one Lanczos step beats one Krylov sign step 20x in gain (`../lanczos_baseline_6x6/README.md`). A fixed number of
Lanczos steps is a global operation; does its per-site gain shrink with N while the Krylov sign step (configuration by configuration) stays size-intensive?

Everything is exact (full k=0, A1, flip+ sector vectors, no sampling): N = 16 (4x4), 20 (T(4,0;1,5)), 24 (T(4,0;0,6)), 28 (T(4,0;1,7)), 32 (T(4,4;-4,4)), 36 (6x6);
dims 107 ... 15.8M. Code: `experiments/lanczos_vs_krylov_scaling/` (`scaling.py`, `run.sbatch`, `aggregate.py`), reusing unmodified the sector/Krylov code of the depth-vs-N test
(`closed_fn_krylov_sym6x6.py`, `torus_cluster.py`) and `lanczos_lib.py` (alpha from the 2x2 problem in span{psi, H psi}). Raw numbers: `lvk_N*.json` (guides A, B, Cx, Cm),
`lvk2_N*.json` (A2, C2), `lvk3_N*.json` (J); `tables.md`, `cost.md` are the aggregator outputs reproduced below. Cluster jobs 16897859, 16897884, 16897986 (ws1 `cluster` partition, 16 CPUs;
total about 3 CPU-h; N=36 takes 12 min). dE = (E-E0)/N per site; w_s = wrong-sign weight vs exact psi0 (global sign optimised).

## Guides (start states, all exactly available)
- **A** exact |psi0| + Marshall; **B** |GS(J2=0)| + Marshall (amplitude and sign both imperfect); both deterministic.
- **Cx** |psi0| e^eps + exact signs, **Cm** same noise + Marshall signs. eps ~ iid N(0, sigma^2) per symmetry orbit, sigma = 0.018 at ALL N (calibrated so the 4x4 start error is 1.0e-4/site; the start error then stays 1.08-1.15e-4 at every N, i.e. it is size-intensive by construction). Mean over seeds (64, 32, 16, 8, 4, 4 for N = 16..36).
- Added because Cx has no sign error (Krylov trivially useless) and A/Cm are dominated by the Marshall sign error (Lanczos at a disadvantage): **A2** exact |psi0| + the signs after two Krylov steps from Marshall (the depth-vs-N ladder, w_s 1e-6..4e-5, N-dependent), **C2** the Cx noise on A2's signs,
  **J** |psi0| exp(kappa (n_antiparallel NN - mean)) + exact signs, kappa = 0.0067 fixed (a smooth, extensive, Jastrow-type amplitude mis-tuning; start 0.9-1.0e-4/site at all N).
- Operations: K = one Krylov sign step (exact energy-optimal threshold); L1 = one Lanczos step (1+alpha H)psi, optimal alpha; L2 = two Lanczos steps (lowest Ritz vector of span{psi,H psi,H^2 psi}); L1,L1 = two sequential single-alpha steps; K+L1 = Krylov then Lanczos.
  Tables below list dE after each operation, gains (start minus after, per site) and w_s. Note that the clusters are different tori (the N=32 and N=36 Marshall start errors are much smaller per site than for N <= 28: cluster-specific, not a trend); compare methods at the same N rather than fitting absolute gains.

## Verdict
**The expectation is not borne out for N <= 36: the per-site gain of one Lanczos step does NOT shrink with N, and the Krylov sign step is not size-intensive in any special way.** Both gains are controlled by what the start error is made of.
- **Lanczos (L1)** removes a fixed FRACTION of the start error that is independent of N: Cx 0.85 -> 0.92, J 0.88 -> 0.88 (flat), C2 0.85 -> 0.91, A2 0.81 -> 0.91, A/Cm 0.80 -> 0.73 (cluster-dependent, no monotone trend), B 0.62 -> 0.56. At fixed per-site start error the per-site gain is therefore flat (Cx: 9.5e-5, 9.9e-5, 1.00e-4, 1.01e-4, 1.00e-4, 9.9e-5; J 8.8e-5 -> 7.9e-5, tracking the start error, the fraction is flat at 0.88). The remaining error even shrinks (Cx 1.7e-5 -> 0.9e-5 /site). L2 removes 0.93-0.99 of it (Cx 0.956 -> 0.986), also N-independent.
- **Krylov (K)** removes exactly the sign-error part of the energy and nothing else: gain ~ 0 when there are no wrong signs (Cx, J: gain <= 3e-7 of 1e-4), 0 when the amplitude is too wrong for any flip to lower E (B: exactly 0 at all N), 0.84-0.96 of the start error for Marshall starts (A, Cm), 0.98-1.00 for A2. Its gain scales with the wrong-sign weight w_s of the guide (C2: K gain 2e-6 ... 6e-5, w_s 9e-7 ... 4e-5), not with N as such.
- **Ratio gain(K)/gain(L1)**: A 1.19, 1.20, 1.19, 1.20, 1.16, 1.16 and A2 1.24 ... 1.08 (K slightly ahead, flat); B, Cx, J: K = 0 (Lanczos infinitely better, 1e3-1e5 for Cx/J); C2 (noise 1e-4 + small residual sign error): 0.02, 0.05, 0.19, 0.16, 0.18, 0.39 for N = 16..36.
  The C2 rise is the growth of w_s of the fixed-depth (d=2) Krylov ladder signs with N (8.9e-7 -> 4.4e-5, cf. `sign_design_tests`), i.e. a property of that guide family, not of the two operations. A naive extrapolation of that column would cross 1 near N ~ 45-60, but this is only a trend (6 clusters of different shape, a guide whose sign error is defined by a fixed depth); at fixed w_s the ratio would stay flat. We find no evidence that Krylov overtakes Lanczos at larger N through the scaling of the operations themselves.
- Where Krylov wins (A, A2: by ~1.1-1.2x in gain, i.e. 1.8-4.3x lower remaining error after K than after L1 for A) the guide is almost purely sign-wrong with perfect amplitude. For K+L1 (K first) the error is lowest of all one-hop-pair combinations there (A: 4e-5 ... 1.3e-4 vs L2 7e-4 ... 1e-3 per site) but the gain over L1 alone vanishes for the sign-good guides (Cx, J, C2: K adds < 5%).
- Caveats. (i) Total start error N*dE is only 4e-3 ... 4e-2 here, far below the sector gap (O(1)). In this perturbative regime a smooth linear filter (1+alpha H) removes a fixed fraction of the error spectrum whatever N is. A decay of the per-site Lanczos gain with N (product-state heuristic: gain/site ~ N^(-1/2)) is expected only once N*dE exceeds the gap scale, i.e. at N of order 10^2-10^3 for 1e-4/site; that regime cannot be reached exactly and is NOT tested here. (ii) The two error families (iid orbit noise, Jastrow-type smooth mis-tuning) are model errors; a trained network's error can be different (the 6x6 ViT numbers of the baseline: fractional Lanczos gain 0.80 and 0.75, in line with the 0.73-0.92 here). (iii) The K+L1 and the L2 values are exact-vector statements, not sampled-network ones (see the baseline README on writing psi_1 back).

## Cost per evaluation (neighbour evaluations per configuration)
Number of connected configurations of a configuration x = number of antiparallel NN+NNN bonds (each antiparallel bond gives one distinct flipped neighbour), averaged with |psi0|^2 (guide-independent to the accuracy needed):

| N | antiparallel NN bonds | antiparallel NNN bonds | neighbours per config (1 hop) | 2-hop estimate (neighbours^2) |
|---|---|---|---|---|
| 16 | 22.46 | 14.36 | 36.82 | 1356 |
| 20 | 27.86 | 18.20 | 46.07 | 2122 |
| 24 | 33.43 | 21.85 | 55.29 | 3057 |
| 28 | 38.99 | 25.53 | 64.52 | 4162 |
| 32 | 45.24 | 26.85 | 72.09 | 5198 |
| 36 | 50.60 | 30.98 | 81.58 | 6655 |

So one hop costs ~2.3 N neighbour evaluations (N=36: 82, in the ~2N ballpark of the baseline note), growing linearly in N.
- Krylov sign step: 1 hop per configuration (local energy r = H psi/psi from the neighbours; the grouped threshold pass reuses the same matrix elements): ~2.3 N evaluations/config, i.e. N-linear. Exact sector cost: 1 H-matvec + 1 sparse sweep.
- Lanczos step: evaluating psi_1(x) = psi(x) + alpha sum_x' psi(x') costs the same 1 hop, ~2.3 N per evaluation; but alpha needs <H>, <H^2>, <H^3>, and <H^3> needs a second hop: ~(2.3 N)^2 evaluations per sample (upper bound, 6.7e3 at N=36), and each later use of psi_1 pays the hop again. Exact: 2 matvecs for L1, 3 for L2.
- Both are O(N) per use of the corrected state; the extra O(N^2) of Lanczos is a one-off estimate of alpha (can be amortised over samples / estimated on a subset).

## Figure
`fig_gain_vs_N.png` / `.pdf`: per-site gain (start minus after) vs N, log10 axes, points + thin lines, one panel per guide type (dotted = start error); last panel = gain(K)/gain(L1). Missing orange curve = Krylov gain ~ 0.

## Tables

**Guide A: exact |psi0| + Marshall** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; single deterministic guide)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.156e-02 | 5.514e-04 | 2.362e-03 | 8.170e-05 | 8.015e-04 | 1.022e-03 | 1.101e-02 | 9.201e-03 | 0.84 | 1.27e-02 | 3.41e-04 | 1.59e-03 |
| 20 | 1.058e-02 | 4.390e-04 | 2.100e-03 | 4.237e-05 | 7.065e-04 | 8.973e-04 | 1.014e-02 | 8.480e-03 | 0.84 | 1.36e-02 | 2.76e-04 | 7.35e-04 |
| 24 | 1.102e-02 | 7.031e-04 | 2.369e-03 | 8.708e-05 | 8.797e-04 | 1.090e-03 | 1.032e-02 | 8.654e-03 | 0.84 | 1.75e-02 | 5.15e-04 | 1.70e-03 |
| 28 | 1.080e-02 | 5.746e-04 | 2.278e-03 | 6.766e-05 | 7.815e-04 | 9.958e-04 | 1.023e-02 | 8.523e-03 | 0.83 | 1.97e-02 | 4.87e-04 | 1.69e-03 |
| 32 | 6.192e-03 | 7.858e-04 | 1.523e-03 | 8.240e-05 | 5.775e-04 | 7.328e-04 | 5.406e-03 | 4.669e-03 | 0.86 | 1.36e-02 | 6.52e-04 | 4.06e-03 |
| 36 | 7.341e-03 | 1.143e-03 | 1.998e-03 | 1.337e-04 | 8.152e-04 | 1.023e-03 | 6.198e-03 | 5.343e-03 | 0.86 | 1.96e-02 | 1.08e-03 | 8.83e-03 |

**Guide B: |GS(J2=0)| + Marshall** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; single deterministic guide)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4.061e-02 | 4.061e-02 | 1.556e-02 | 1.556e-02 | 5.316e-03 | 7.829e-03 | 0.000e+00 | 2.505e-02 | nan | 1.27e-02 | 1.27e-02 | 9.01e-03 |
| 20 | 4.043e-02 | 4.043e-02 | 1.833e-02 | 1.833e-02 | 9.313e-03 | 1.149e-02 | 0.000e+00 | 2.210e-02 | nan | 1.36e-02 | 1.36e-02 | 9.89e-03 |
| 24 | 4.168e-02 | 4.168e-02 | 1.935e-02 | 1.935e-02 | 9.686e-03 | 1.204e-02 | 2.985e-09 | 2.233e-02 | 7480056.36 | 1.75e-02 | 1.75e-02 | 1.32e-02 |
| 28 | 4.211e-02 | 4.211e-02 | 2.041e-02 | 2.041e-02 | 1.081e-02 | 1.313e-02 | 5.428e-11 | 2.169e-02 | 399680208.36 | 1.97e-02 | 1.97e-02 | 1.63e-02 |
| 32 | 2.893e-02 | 2.893e-02 | 1.090e-02 | 1.090e-02 | 4.806e-03 | 6.054e-03 | 1.066e-14 | 1.803e-02 | nan | 1.36e-02 | 1.36e-02 | 1.23e-02 |
| 36 | 3.234e-02 | 3.234e-02 | 1.427e-02 | 1.427e-02 | 7.383e-03 | 8.879e-03 | 2.570e-12 | 1.807e-02 | 7032726283.41 | 1.96e-02 | 1.96e-02 | 1.82e-02 |

**Guide C-x: noisy |psi0| + exact signs** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; mean over seeds)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.125e-04 | 1.122e-04 | 1.729e-05 | 1.728e-05 | 4.981e-06 | 6.464e-06 | 2.650e-07 | 9.518e-05 | 359.18 | 0.00e+00 | 2.16e-07 | 1.03e-07 |
| 20 | 1.148e-04 | 1.147e-04 | 1.579e-05 | 1.577e-05 | 4.200e-06 | 5.430e-06 | 3.821e-08 | 9.898e-05 | 2590.03 | 0.00e+00 | 5.61e-08 | 1.64e-08 |
| 24 | 1.137e-04 | 1.137e-04 | 1.362e-05 | 1.362e-05 | 3.153e-06 | 4.150e-06 | 3.197e-08 | 1.001e-04 | 3131.87 | 0.00e+00 | 3.96e-08 | 2.09e-08 |
| 28 | 1.123e-04 | 1.123e-04 | 1.123e-05 | 1.123e-05 | 2.204e-06 | 2.945e-06 | 2.611e-08 | 1.011e-04 | 3872.04 | 0.00e+00 | 3.50e-08 | 2.41e-08 |
| 32 | 1.114e-04 | 1.114e-04 | 1.137e-05 | 1.136e-05 | 2.228e-06 | 2.997e-06 | 6.409e-09 | 1.001e-04 | 15615.12 | 1.11e-16 | 1.01e-08 | 6.97e-09 |
| 36 | 1.077e-04 | 1.077e-04 | 9.106e-06 | 9.105e-06 | 1.509e-06 | 2.053e-06 | 4.886e-09 | 9.858e-05 | 20175.34 | 0.00e+00 | 5.63e-09 | 4.50e-09 |

**Guide C-m: noisy |psi0| + Marshall** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; mean over seeds)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.165e-02 | 6.793e-04 | 2.409e-03 | 1.012e-04 | 8.223e-04 | 1.049e-03 | 1.097e-02 | 9.242e-03 | 0.84 | 1.27e-02 | 3.54e-04 | 1.73e-03 |
| 20 | 1.069e-02 | 5.561e-04 | 2.149e-03 | 5.901e-05 | 7.331e-04 | 9.286e-04 | 1.014e-02 | 8.542e-03 | 0.84 | 1.36e-02 | 2.78e-04 | 7.81e-04 |
| 24 | 1.112e-02 | 8.163e-04 | 2.446e-03 | 9.954e-05 | 9.156e-04 | 1.137e-03 | 1.031e-02 | 8.679e-03 | 0.84 | 1.75e-02 | 5.14e-04 | 1.80e-03 |
| 28 | 1.090e-02 | 6.890e-04 | 2.377e-03 | 8.062e-05 | 8.457e-04 | 1.069e-03 | 1.021e-02 | 8.526e-03 | 0.83 | 1.97e-02 | 4.87e-04 | 1.77e-03 |
| 32 | 6.304e-03 | 9.015e-04 | 1.628e-03 | 9.418e-05 | 6.408e-04 | 8.088e-04 | 5.402e-03 | 4.676e-03 | 0.87 | 1.36e-02 | 6.53e-04 | 5.14e-03 |
| 36 | 7.450e-03 | 1.270e-03 | 2.135e-03 | 1.463e-04 | 9.031e-04 | 1.128e-03 | 6.179e-03 | 5.315e-03 | 0.86 | 1.96e-02 | 1.09e-03 | 1.08e-02 |

**Guide A2: exact |psi0| + signs after 2 Krylov steps** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; single deterministic guide)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.319e-06 | -1.110e-16 | 2.557e-07 | 1.110e-16 | 7.776e-08 | 1.033e-07 | 1.319e-06 | 1.063e-06 | 0.81 | 8.85e-07 | 0.00e+00 | 0.00e+00 |
| 20 | 5.566e-06 | 2.365e-08 | 5.597e-07 | 2.371e-09 | 1.259e-07 | 1.605e-07 | 5.543e-06 | 5.007e-06 | 0.90 | 3.04e-06 | 1.32e-08 | 2.67e-08 |
| 24 | 2.315e-05 | 1.977e-07 | 2.645e-06 | 1.849e-08 | 5.790e-07 | 7.680e-07 | 2.295e-05 | 2.050e-05 | 0.89 | 1.38e-05 | 1.06e-07 | 1.64e-07 |
| 28 | 1.944e-05 | 3.119e-07 | 1.813e-06 | 2.336e-08 | 3.625e-07 | 4.728e-07 | 1.913e-05 | 1.763e-05 | 0.92 | 1.19e-05 | 1.69e-07 | 2.48e-07 |
| 32 | 2.159e-05 | 1.608e-07 | 1.725e-06 | 1.035e-08 | 2.821e-07 | 3.797e-07 | 2.143e-05 | 1.987e-05 | 0.93 | 1.40e-05 | 9.19e-08 | 1.12e-07 |
| 36 | 6.160e-05 | 1.075e-06 | 5.378e-06 | 7.341e-08 | 1.008e-06 | 1.323e-06 | 6.053e-05 | 5.622e-05 | 0.93 | 4.37e-05 | 6.59e-07 | 8.25e-07 |

**Guide C2: noisy |psi0| + the same A2 signs** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; mean over seeds)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.141e-04 | 1.122e-04 | 1.757e-05 | 1.728e-05 | 5.100e-06 | 6.601e-06 | 1.945e-06 | 9.657e-05 | 49.64 | 8.85e-07 | 2.16e-07 | 2.83e-07 |
| 20 | 1.204e-04 | 1.148e-04 | 1.645e-05 | 1.577e-05 | 4.428e-06 | 5.689e-06 | 5.601e-06 | 1.039e-04 | 18.55 | 3.04e-06 | 6.81e-08 | 4.43e-08 |
| 24 | 1.369e-04 | 1.140e-04 | 1.667e-05 | 1.364e-05 | 3.900e-06 | 5.130e-06 | 2.284e-05 | 1.202e-04 | 5.26 | 1.38e-05 | 2.65e-07 | 4.31e-07 |
| 28 | 1.318e-04 | 1.126e-04 | 1.343e-05 | 1.128e-05 | 2.713e-06 | 3.603e-06 | 1.913e-05 | 1.183e-04 | 6.19 | 1.19e-05 | 2.18e-07 | 2.50e-07 |
| 32 | 1.329e-04 | 1.116e-04 | 1.329e-05 | 1.138e-05 | 2.576e-06 | 3.459e-06 | 2.130e-05 | 1.196e-04 | 5.62 | 1.40e-05 | 1.12e-07 | 1.18e-07 |
| 36 | 1.693e-04 | 1.088e-04 | 1.466e-05 | 9.202e-06 | 2.540e-06 | 3.420e-06 | 6.055e-05 | 1.547e-04 | 2.55 | 4.37e-05 | 6.77e-07 | 7.77e-07 |

**Guide J: |psi0| e^(kappa n_NN) + exact signs** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; single deterministic guide)

| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1.002e-04 | 1.002e-04 | 1.176e-05 | 1.176e-05 | 2.584e-06 | 3.447e-06 | 0.000e+00 | 8.843e-05 | nan | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 20 | 9.398e-05 | 9.398e-05 | 1.118e-05 | 1.121e-05 | 2.462e-06 | 3.293e-06 | 2.877e-09 | 8.280e-05 | 28778.33 | 0.00e+00 | 6.39e-09 | 1.48e-08 |
| 24 | 9.348e-05 | 9.348e-05 | 1.100e-05 | 1.101e-05 | 2.370e-06 | 3.187e-06 | 8.412e-10 | 8.248e-05 | 98048.54 | 0.00e+00 | 1.51e-09 | 4.52e-09 |
| 28 | 9.325e-05 | 9.325e-05 | 1.095e-05 | 1.097e-05 | 2.365e-06 | 3.176e-06 | 1.481e-09 | 8.230e-05 | 55574.55 | 0.00e+00 | 1.62e-09 | 3.32e-08 |
| 32 | 9.267e-05 | 9.267e-05 | 1.146e-05 | 1.150e-05 | 3.797e-06 | 4.457e-06 | 1.140e-09 | 8.121e-05 | 71219.30 | 1.11e-16 | 4.64e-09 | 3.89e-08 |
| 36 | 8.936e-05 | 8.936e-05 | 1.078e-05 | 1.079e-05 | 3.085e-06 | 3.770e-06 | 6.960e-10 | 7.858e-05 | 112913.81 | 0.00e+00 | 1.83e-09 | 3.96e-08 |

**Fractional gain (gain / start error)**: K, L1 (and L2)

| guide | quantity | N=16 | N=20 | N=24 | N=28 | N=32 | N=36 |
|---|---|---|---|---|---|---|---|
| A | K | 0.952 | 0.959 | 0.936 | 0.947 | 0.873 | 0.844 |
| A | L1 | 0.796 | 0.802 | 0.785 | 0.789 | 0.754 | 0.728 |
| A | L2 | 0.931 | 0.933 | 0.920 | 0.928 | 0.907 | 0.889 |
| B | K | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B | L1 | 0.617 | 0.547 | 0.536 | 0.515 | 0.623 | 0.559 |
| B | L2 | 0.869 | 0.770 | 0.768 | 0.743 | 0.834 | 0.772 |
| Cx | K | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cx | L1 | 0.846 | 0.862 | 0.880 | 0.900 | 0.898 | 0.915 |
| Cx | L2 | 0.956 | 0.963 | 0.972 | 0.980 | 0.980 | 0.986 |
| Cm | K | 0.942 | 0.948 | 0.927 | 0.937 | 0.857 | 0.829 |
| Cm | L1 | 0.793 | 0.799 | 0.780 | 0.782 | 0.742 | 0.713 |
| Cm | L2 | 0.929 | 0.931 | 0.918 | 0.922 | 0.898 | 0.879 |
| A2 | K | 1.000 | 0.996 | 0.991 | 0.984 | 0.993 | 0.983 |
| A2 | L1 | 0.806 | 0.899 | 0.886 | 0.907 | 0.920 | 0.913 |
| A2 | L2 | 0.941 | 0.977 | 0.975 | 0.981 | 0.987 | 0.984 |
| C2 | K | 0.017 | 0.047 | 0.167 | 0.145 | 0.160 | 0.358 |
| C2 | L1 | 0.846 | 0.863 | 0.878 | 0.898 | 0.900 | 0.913 |
| C2 | L2 | 0.955 | 0.963 | 0.972 | 0.979 | 0.981 | 0.985 |
| J | K | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| J | L1 | 0.883 | 0.881 | 0.882 | 0.883 | 0.876 | 0.879 |
| J | L2 | 0.974 | 0.974 | 0.975 | 0.975 | 0.959 | 0.965 |

**Power-law fits gain(N) ~ c N^(-p)** (all six sizes; p = 0: size-intensive, p = 1: gain ~ 1/N)

| guide | p(K) | p(L1) | p(L2) | p(K+L1) |
|---|---|---|---|---|
| A | +0.81 | +0.77 | +0.71 | +0.68 |
| B | n/a (gain 0) | +0.40 | +0.44 | +0.40 |
| Cx | n/a (gain 0) | -0.05 | +0.01 | -0.05 |
| Cm | +0.81 | +0.78 | +0.72 | +0.67 |
| A2 | -4.19 | -4.34 | -4.26 | -4.21 |
| C2 | -3.80 | -0.48 | -0.42 | -0.51 |
| J | n/a (gain 0) | +0.12 | +0.13 | +0.12 |
