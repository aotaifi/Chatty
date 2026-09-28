# STATUS

## Current state
The original question — why one Krylov/local-energy sign step works so well at J2/J1=1/2 — is empirically resolved on exact finite clusters and now has a sharper dynamical interpretation.

## Core mechanism
With fixed amplitudes a(x), write psi_s=a s and r_s=(H psi_s)/psi_s. The sign of (t-H)psi_s is

T_t[s] = s sign(t-r_s).

Choosing t by minimizing the fixed-amplitude energy over this threshold family gives an energy-non-increasing nonlinear sign map because the family always contains the current s.

With exact ground-state amplitudes, the exact signs are a fixed global minimum and a strong attractor.

## Square-lattice results
- J2/J1=0.5 Marshall: exact signs after 3 updates.
- J2/J1=0.6 Marshall: exact after 4.
- J2/J1=0.8 Marshall: exact after 10 despite one-step failure.
- J2/J1=1.0 Marshall: exact after 9 despite one-step failure.
- J2/J1=0.8 and 1.0 stripe: exact after 4.

The good gauge explains why the first step is spectacular, but is not required for eventual convergence on the tested 4x4 square cluster.

Random-corruption tests show a very large empirical ground-state basin: all 32/32 sampled square starts converged through q=0.42 wrong physical probability weight (O_initial about 0.16).

## Triangular basin structure
On the 6x3 triangular Heisenberg cluster, all three simple two-color starts are now explained by exact symmetry-sector obstructions:

- maxcut_x: wrong translation character Tx=-1 versus ground Tx=+1;
- parity_xy: wrong translation character Tx=-1 versus ground Tx=+1;
- parity_y: wrong inversion character under P:(x,y)->(-x,2-y), with start/fixed point P=+1 but ground P=-1.

The projected-Krylov map preserves such characters exactly whenever [P,H]=0 and the fixed amplitude modulus is P-invariant.

Random perturbations around the exact triangular signs still contract strongly back to them, so the triangular failure is not absence of a ground-state attractor.

## New refinement: symmetry-protected basin has finite thickness
Breaking parity_y's wrong inversion symmetry by an arbitrarily small perturbation does not immediately free the dynamics:
- q <= 0.01 perturbed physical weight: 0/12 recoveries;
- q=0.02: 2/12;
- q=0.05: 5/12;
- q=0.10: 8/12.

Thus an exact wrong symmetry sector forms the invariant core of a **finite nonlinear attraction basin**.

## Current verdict
The one-step miracle is the first strong contraction step of projected-Krylov sign descent. The sign dynamics has multiple macroscopic basins:
1. exact symmetry sectors provide rigorous invariant cores;
2. nonlinear attraction regions extend around those cores;
3. the ground-state basin is large on both tested square and triangular clusters.

## Primary current files
- BASIN_DYNAMICS_VERDICT_2026-09-28.md
- SYMMETRY_BASIN_INVARIANT_2026-09-28.md
- TRIANGULAR_PARITY_BASIN_2026-09-28.md

## Next question
The conceptual mechanism is now substantially closed. The next high-value question is algorithmic/scaling: does the same basin contraction survive when a(x) and the threshold-energy objective are only approximate/sampled rather than exact?
