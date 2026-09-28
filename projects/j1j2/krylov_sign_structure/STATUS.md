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

## Approximate-amplitude robustness
A structured mismatch test now removes the exact-modulus assumption in a controlled way by using the exact modulus from a neighboring J2 as the fixed trial modulus for the target Hamiltonian.

Results:
- target J2=0.5, amplitude from J2=0.4: F_a=0.9404 -> final O_S=0.99683, but not exact;
- target J2=1.0, amplitude from J2=0.8: F_a=0.9690 -> final O_S=0.99981, but not exact;
- poorer amplitudes degrade the terminal sign structure, though surprisingly high sign overlap can survive substantial modulus mismatch.

Thus the exact-sign attractor is tied to the exact modulus. With approximate amplitudes, the same sign dynamics remains useful but converges to the minimum/fixed point of a biased fixed-amplitude objective.

## Updated next question
For this theory/mechanism subproject, the core picture is now closed enough to park. The next step belongs to the separate algorithmic amplitude/sign loop: determine whether an efficient approximate-amplitude solver can keep the modulus inside the regime where the cheap sign update remains accurate as system size grows.

## Closed FN <-> current-sign Krylov loop launched — 2026-09-28

A key distinction from the older 4x4 loop was identified: the previous "energy-opt" implementation rebuilt signs relative to the fixed Marshall coordinate r_M each iteration. It did **not** iterate the current-sign map

r_k = (H a_k s_k)/(a_k s_k),
s_{k+1}=s_k sign(t_k-r_k).

A new exact 4x4 benchmark now implements the genuinely recursive current-sign map, with the threshold chosen label-free by exact fixed-amplitude energy minimization over grouped r_k values.

Loop:
1. current guide (a_k,s_k) defines the standard lattice fixed-node Hamiltonian;
2. exact FN ground state supplies positive amplitudes a_{k+1};
3. current-sign projected-Krylov update produces s_{k+1};
4. repeat.

Target exact ground-state information is used only after each step for diagnostics E-E0, sign overlap, and amplitude fidelity; it never enters the FN solve or threshold choice.

Two J2/J1=0.5 runs are being made:
- best-case control: exact target modulus used only as the *initial* amplitude, with Marshall signs;
- no-target-oracle bootstrap: J2=0 Marshall ground-state modulus used as the initial amplitude, with Marshall signs.

The decisive question is whether the coupled FN/Krylov map converges to the exact target pair, cycles, or settles at a self-consistent biased fixed point.

## Recursive FN/current-sign Krylov result — 2026-09-28

The genuinely recursive map has now passed the best-case exact 4x4 control at J2/J1=0.5.

Unlike the older loop, the sign step is built from the **current** sign pattern:
r_k=(H a_{k+1}s_k)/(a_{k+1}s_k), followed by an exact label-free energy-optimal threshold.

Starting from the exact target modulus with Marshall signs, the coupled FN/Krylov loop reaches O_S=1 exactly at iteration 27. At iteration 30 F_a=0.99999647 and the guide-energy error is only 8.67e-6 total.

Starting instead from the J2=0 modulus plus Marshall signs, with no target information in any update, the loop reaches O_S=0.99999823 by iteration 20 and remains on the same residual sign pattern through iteration 30 while amplitudes continue improving to F_a=0.9997867. This is the same residual pattern the control eventually escaped from.

A 100-iteration no-oracle extension is running under job 20260928-112353-51474 to decide whether the bootstrap also reaches the exact sign pattern.
