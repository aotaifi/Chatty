# STATUS

## Current state
The requested mechanism sweep is complete on exact finite clusters.

## Decisive result
The one-step Krylov sign miracle is **not generic frustration simplification**. It is a good-gauge / large-basin phenomenon in the effective fixed-amplitude sign-Ising problem.

- 4x4 square J1-J2 exact sweep completed at J2/J1 = 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0.
- Primary threshold is chosen label-free by minimizing fixed-amplitude energy within the one-step family.
- At J2/J1=0.5: O_S 0.974538 -> 0.999319; energy error/site 1.156e-2 -> 5.51e-4; P_stable=0.999658.
- Marshall fails at large J2 as a coordinate, not merely as a threshold selector.
- A stripe/J2-adapted baseline restores the one-step structure: at J2/J1=1, O_S 0.955819 -> 0.994884 and P_stable=0.997445.
- Only after that comparison, the 6x3 triangular-lattice Heisenberg model was tested. The one-step coordinate fails decisively: oracle O_S <= 0.0604 across tested two-color gauges, despite P_stable reaching ~0.95.

## Interpretation
Required ingredients are now:
1. a baseline gauge in the correct broad sign basin;
2. residual corrections approximately ordered by the scalar r0;
3. a physical label-free threshold objective that selects the same basin as the oracle.

Triangular frustration violates (1)-(2) and exposes many locally stable but globally wrong sign basins.

## Files
- MECHANISM_VERDICT_2026-09-28.md
- experiments/square_exact_energyopt.py
- experiments/triangular_exact_test.py
- results/square_exact_energyopt.{json,csv}
- results/triangular_exact_test.json

## New decisive result: iterated contraction
With exact ground-state amplitudes held fixed, repeated label-free fixed-amplitude-energy Krylov sign updates converge to the **exact ground-state signs** on all three representative square-lattice tests:
- J2/J1=0.5, Marshall: exact after 3 updates.
- J2/J1=0.6, Marshall: exact after 4 updates.
- J2/J1=1.0, stripe baseline: exact after 4 updates.

At every step, the energy-selected threshold attains the same next-step sign overlap as the hidden-sign oracle threshold on the same r_k coordinate.

This upgrades the mechanism from “one unusually good step” to **finite-cluster basin contraction**: once the starting gauge is in the correct broad sign basin, recomputing r_k progressively exposes the remaining defects.

See `ITERATED_KRYLOV_CONVERGENCE_2026-09-28.md`.

## Next question
The key unresolved issue is now scaling/generalization: can this contraction be predicted or proved from a sign-Ising frustration/loop-flux criterion, and does it survive when the amplitudes and threshold objective are only approximate rather than exact?
