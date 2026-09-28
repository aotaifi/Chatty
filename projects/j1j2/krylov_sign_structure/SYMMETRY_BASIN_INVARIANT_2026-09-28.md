# Symmetry-sector basin invariant — 2026-09-28

## Exact statement

Let P be a configuration-space symmetry satisfying [P,H]=0. Suppose the fixed positive amplitude vector a is P-invariant, P a = a, and the current wavefunction psi_s=a s is a P eigenstate,

P psi_s = chi psi_s,

with real character chi=+1 or -1.

Define the local-energy coordinate

r_s(x) = (H psi_s)(x) / psi_s(x).

Because H commutes with P and psi_s has character chi, numerator and denominator transform by the same chi. Therefore

r_s(Px)=r_s(x).

Hence every threshold mask m_t(x)=sign[t-r_s(x)] is P-invariant. The projected Krylov update

psi' = a s m_t

therefore satisfies

P psi' = chi psi'.

**Conclusion:** the projected-Krylov sign iteration exactly preserves any symmetry character carried by the current fixed-amplitude wavefunction, provided the amplitude modulus is invariant under that symmetry.

If the initial state and target ground state have different characters, no number of iterations can connect them.

## Numerical audit

The exact amplitude modulus is translation-invariant to about 1e-11 or better on all audited clusters.

### Triangular 6x3
Exact ground-state signs:
- Tx = +1, residual about 1e-12
- Ty = +1, residual about 1e-12

maxcut_x gauge:
- Tx = -1, residual about 1e-12
- Ty = +1

parity_xy gauge:
- Tx = -1, residual about 1e-12

Thus maxcut_x and parity_xy are **rigorously symmetry-sector forbidden** from converging to the exact ground state under this iteration. Their exactly vanishing overlap and wrong fixed points are not merely numerical metastability.

The parity_y gauge is not a Ty eigenstate and is not explained by this simple translation-character invariant; its wrong basin remains a separate structured-basin problem.

### Square 4x4, J2/J1=0.8 and 1.0
Exact ground state, Marshall gauge, and stripe_x gauge all have
- Tx = +1
- Ty = +1
with residuals around 1e-11 or smaller.

Therefore the translation-sector obstruction present for the triangular maxcut_x gauge is absent. This is consistent with the observed recovery of even the badly overlapping Marshall starts:
- J2/J1=0.8: exact after 10 updates;
- J2/J1=1.0: exact after 9 updates.

## Why random corruption can recover

A generic random sign corruption destroys exact translation-eigenstate structure rather than locking the state into the wrong character. It therefore need not remain in a symmetry-forbidden basin. This explains why random triangular corruptions with very small initial overlap can still flow back to the exact signs while the highly structured maxcut_x gauge cannot.

## Conceptual consequence

The basin structure has at least two layers:

1. **Exact invariant sectors** fixed by symmetries of H and the amplitude modulus. Wrong symmetry character is an absolute obstruction.
2. **Within a compatible symmetry sector**, the energy-decreasing threshold dynamics may still have multiple attraction basins / fixed points. The triangular parity_y case is evidence for this second layer.

This gives the first analytic predictor of basin membership found in the subproject.

## Full symmetry closure of parity_y
A full audit of the 72 affine automorphisms of the periodic 6x3 triangular bond graph finds an additional exact obstruction for the previously unexplained parity_y start.

For the inversion-like symmetry

P:(x,y) -> (-x,2-y),

the exact ground state has character P=-1, while parity_y and its converged wrong fixed point both have P=+1. The fixed modulus is P-invariant to numerical residual ~1e-10.

Hence parity_y is also rigorously sector-forbidden from reaching the ground-state signs. All three simple triangular structured gauges tested so far are now explained by exact symmetry mismatch.

A follow-up perturbation test shows that the invariant sector is surrounded by a finite nonlinear basin: weakly breaking P does not immediately cause escape. See `TRIANGULAR_PARITY_BASIN_2026-09-28.md`.
