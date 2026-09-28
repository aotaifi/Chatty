# Krylov Sign Structure

## Goal
Understand why a single global Krylov/local-energy coordinate reconstructs the physically important frustrated ground-state signs so well, and determine whether this reflects a general structural property of frustrated quantum magnets rather than a special feature of the J1-J2 point.

## Core observation
With amplitudes a(x) and a baseline sign s_0(x), define
r_0(x) = (H psi_0)(x) / psi_0(x),  psi_0(x)=a(x)s_0(x).
A one-step Krylov sign rule
s_1(x)=s_0(x) sign[t-r_0(x)]
nearly reproduces the high-weight ViT sign structure on J1-J2 at J2/J1=1/2.

## Main hypothesis
r_0(x) is the amplitude-weighted local field of the fixed-amplitude sign-Ising problem. One Krylov step may therefore act like one global zero-temperature field update. If the effective sign-Ising landscape is weakly frustrated / non-glassy around the relevant state, one sweep may land close to a stable minimum.

## Measured quantities
- O_S^(0) = |sum_x p_x s_0(x) s_*(x)|, with p_x proportional to a_x^2: physical-weight sign overlap of the baseline with the reference ground-state signs.
- O_S^(1): the same overlap after one Krylov threshold update.
- Fixed-amplitude energy error = [E(a,s)-E(a,s_*)]/N; amplitudes are held fixed, so only the sign structure is tested.
- P_stable^(1): physical-weight fraction stable against a single sign flip under the fixed-amplitude sign-Ising local field.

## Falsification sequence
1. Sweep J2/J1 = 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0 with the same construction.
2. Track baseline sign overlap O_S^(0), one-step overlap O_S^(1), fixed-amplitude energy error, and post-update local-field stability P_stable^(1).
3. Near J2/J1 ~ 1 compare the usual Marshall baseline with a J2-adapted bipartite baseline.
4. Only after the internal sweep, test an external strongly frustrated model, with triangular-lattice Heisenberg as the first target.
5. Distinguish clearly between a special J1-J2 mechanism, a large basin-of-attraction phenomenon, and a generic frustration principle.

## Stop conditions
SUCCESS: derive a structural criterion explaining when one global local-field/Krylov update reconstructs the sign sector, and show it survives beyond the J1-J2 point.
FAILURE: identify a controlled frustration regime/model where the first-step rule breaks and isolate which assumption fails.

## Current mechanism verdict (2026-09-28)
The one-step rule is the first move of a **projected-Krylov sign descent**. With fixed amplitudes, the threshold family is the sign of `(t-H)psi`; choosing the threshold by fixed-amplitude energy minimization makes the update energy non-increasing.

With exact ground-state amplitudes, repeated updates reach the exact signs on every tested 4x4 square J1-J2 start, including Marshall at J2/J1=0.8 and 1.0 where the first step fails badly. A good gauge therefore explains why **one step** is spectacular, but is not required for eventual convergence on this finite square cluster.

The 6x3 triangular model reveals the real caveat: simple two-color gauges flow to competing wrong fixed points, while random perturbations around the exact signs still contract strongly back to them. Basin membership is therefore structural, not determined by overlap alone.

We also identified one exact basin invariant: symmetry character. If the current fixed-amplitude wavefunction is in a symmetry sector of H, the projected-Krylov threshold map preserves that sector. On the triangular cluster the exact ground state has Tx=+1 while maxcut_x and parity_xy have Tx=-1, so those starts are rigorously unable to reach the ground signs. Square Marshall/stripe share the ground-state translation sector, so this obstruction is absent.

Two further tests sharpen this. First, iid random square-lattice sign starts with only O_S~0.04-0.06 still reach the exact signs in 37-61% of 64 trials when exact amplitudes are fixed. Second, the mechanism survives substantial smooth amplitude distortion: using a_gamma proportional to |psi_GS|^gamma with gamma=0.75 still gives exact recovery at J2/J1=0.5 and 0.6 and near-perfect recovery at J2/J1=1, while gamma=0.5 largely destroys it.

Analytically, in the exact ground-state sign gauge the local-energy displacement r_x-E0 is exactly the amplitude-weighted boundary load of the current wrong-sign domain. This identifies the iteration as a threshold dynamics on defect boundaries, constrained by exact symmetry sectors.

See `BASIN_DYNAMICS_VERDICT_2026-09-28.md` and `SYMMETRY_BASIN_INVARIANT_2026-09-28.md` for the current authoritative interpretation. `MECHANISM_VERDICT_2026-09-28.md` records the earlier one-step stage and is partly superseded.

## Rule
This subproject is theory/mechanism only. Do not duplicate the fixed-node loop engineering being run elsewhere.
