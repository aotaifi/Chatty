# Approximate-amplitude robustness — 2026-09-28

## Question
The exact-cluster mechanism tests hold a(x)=|psi_*(x)| fixed. Does projected-Krylov sign contraction survive when the modulus is only approximate?

## Controlled structured mismatch
For a target square-lattice J1-J2 Hamiltonian, use as fixed modulus the exact ground-state modulus from a *different* J2 value. This gives a physically structured amplitude error rather than arbitrary noise.

The target Hamiltonian, Marshall starting signs, label-free threshold-energy minimization, and exact reference signs for scoring are unchanged.

Amplitude quality is measured by
- fidelity F_a = |<a_trial|a_target>|^2;
- probability total variation TV = 1/2 sum_x |a_trial(x)^2-a_target(x)^2|.

## Target J2/J1 = 0.5
| amplitude source J2 | F_a | TV | terminal sign overlap |
|---:|---:|---:|---:|
| 0.0 | 0.7132 | 0.4466 | 0.97454 |
| 0.2 | 0.8064 | 0.3601 | 0.97453 |
| 0.4 | 0.9404 | 0.1792 | 0.99683 |
| 0.5 | 1.0000 | 0 | **1.00000** |
| 0.6 | 0.8078 | 0.2923 | 0.96692 |
| 0.8 | 0.1711 | 0.7885 | 0.85713 |
| 1.0 | 0.0927 | 0.8642 | 0.65865 |

Only the exact target modulus reaches the exact target signs in this sweep. A nearby physical modulus can still produce very high sign overlap.

## Target J2/J1 = 1.0
| amplitude source J2 | F_a | TV | terminal sign overlap |
|---:|---:|---:|---:|
| 0.0 | 0.1026 | 0.8639 | 0.16337 |
| 0.2 | 0.0955 | 0.8687 | 0.02605 |
| 0.4 | 0.0828 | 0.8717 | 0.52121 |
| 0.5 | 0.0927 | 0.8642 | 0.67785 |
| 0.6 | 0.2510 | 0.7630 | 0.99660 |
| 0.8 | 0.9690 | 0.1191 | 0.99981 |
| 1.0 | 1.0000 | 0 | **1.00000** |

Again the sign map is remarkably forgiving in overlap — even a quite different J2=0.6 modulus produces O_S~0.9966 — but the fixed point is biased unless the modulus is exact.

## Interpretation
The exact sign-attractor statement depends materially on the modulus.

For exact a=|psi_*|:
- the true signs are a global minimum of the fixed-amplitude energy;
- repeated projected-Krylov updates reach them in the tested square cases.

For approximate a:
- the fixed-amplitude sign objective itself is changed;
- the true sign pattern need not be the exact minimizer/fixed point;
- the dynamics can converge to a nearby biased sign structure.

Therefore the cheap sign update is not the dominant fundamental obstacle in the tested setting. The remaining algorithmic difficulty is supplying amplitudes accurate enough that the fixed-amplitude sign landscape has the desired target basin/minimum.

This is a mechanism result only; no amplitude-update loop is introduced here, to avoid duplicating the separate fixed-node/amplitude project.

## Files
- experiments/amplitude_mismatch_square.py
- results/amplitude_mismatch_square.json
