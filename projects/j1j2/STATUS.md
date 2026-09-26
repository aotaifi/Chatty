# J1-J2 research status

## Goal
Find an exact or genuinely scalable way around the sign problem for the square-lattice spin-1/2 SU(2) J1-J2 Heisenberg model at J2/J1 = 1/2.

## Current constructive route
Iterate:
1. choose a compact guide/sign model s_theta(x);
2. solve amplitudes with a sign-free fixed-node projector for those fixed signs;
3. use sampled amplitudes/energies to improve the sign model;
4. rerun fixed node with the updated signs.

## Current bottleneck
Determine whether the sampled loop scales polynomially or merely hides exponential cost.

## Fixed-node scaling test
For each system size N, determine the walker population Nw*(N) and projection length needed to reach fixed error in energy per site.
Track population-control bias, local-energy variance, autocorrelation/projection time, and sign-model error separately.

## Immediate next step
Build a reproducible small-system benchmark that sweeps Nw and projection length for fixed guide signs, then repeats after sign updates.

## Stop criteria
SUCCESS: evidence and mechanism for polynomially scaling exact/scalable reconstruction.
FAILURE: a decisive exponential bottleneck in sampling, projection, or sign reconstruction that survives guide/model improvements.
