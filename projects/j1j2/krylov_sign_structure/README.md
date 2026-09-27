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
- (O_S^{(0)}=|sum_x p_x,s_0(x)s_*(x)|), with (p_x=a_x^2/sum a^2): physical-weight sign overlap of the baseline with the reference ground-state signs.
- (O_S^{(1)}=|sum_x p_x,s_1(x)s_*(x)|): the same overlap after one Krylov threshold update.
- Fixed-amplitude energy error: ([E(a,s)-E(a,s_*)]/N); amplitudes are held fixed, so only the sign structure is tested.
- (P_{
m stable}^{(1)}=sum_x p_x,mathbf 1[s_1(x)h_x^{(1)}le0]), where (h_x^{(1)}=sum_{y
e x}H_{xy}a_y s_1(y)): physical-weight fraction stable against a single sign flip.

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
The sweep falsifies a generic-frustration principle. The one-step rule is instead a **good-gauge / large-basin phenomenon**: Marshall works through the low/intermediate-J2 regime, a stripe/J2-adapted gauge restores the structure near J2/J1~1, while the triangular-lattice antiferromagnet defeats the scalar r0 threshold even with an oracle threshold. High local-field stability by itself is not sufficient; the triangular test reaches (P_{
m stable}^{(1)}approx0.95) while having essentially zero sign overlap.

See `MECHANISM_VERDICT_2026-09-28.md` for the decisive tables and interpretation.

## Rule
This subproject is theory/mechanism only. Do not duplicate the fixed-node loop engineering being run elsewhere.
