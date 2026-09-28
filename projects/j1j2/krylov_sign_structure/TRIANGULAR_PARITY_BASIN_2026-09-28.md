# Triangular parity-y basin closure — 2026-09-28

## Question
The simple translation audit explained the triangular maxcut_x and parity_xy failures, but not parity_y. Is parity_y trapped by another exact symmetry sector, or by a genuinely symmetry-compatible local basin?

## Full finite-cluster symmetry audit
We enumerated all affine automorphisms of the periodic 6x3 triangular nearest-neighbor graph of the form

(x,y) -> (a x + b y + u mod 6, c x + d y + v mod 3),

retaining only bijections that preserve the bond graph. There are 72 such symmetries, and the exact ground-state modulus is invariant under all of them to numerical residual ~1e-10 or better.

The parity_y start and its converged wrong fixed point are exact eigenstates of the inversion-like symmetry

P: (x,y) -> (-x, 2-y)

(and its x-translated equivalents).

Characters:
- exact ground state: P = -1;
- parity_y start: P = +1;
- parity_y wrong fixed point: P = +1.

Therefore parity_y is **exactly symmetry-forbidden** from reaching the ground-state signs under the projected-Krylov map. This closes the previously unexplained structured triangular basin.

The earlier translation-only statement was incomplete: parity_y is not a Ty eigenstate, but it is in the wrong sector of this inversion symmetry.

## Symmetry-breaking escape test
To test whether symmetry mismatch is the whole practical obstruction, we randomly flipped a fraction q of the physical probability weight of the parity_y sign pattern, thereby destroying its exact P=+1 character, and reran the same label-free fixed-amplitude projected-Krylov descent.

12 trials per q, max 60 updates:

| q perturbed | exact-GS recovery |
|---:|---:|
| 0.0001 | 0/12 |
| 0.0005 | 0/12 |
| 0.001 | 0/12 |
| 0.005 | 0/12 |
| 0.01 | 0/12 |
| 0.02 | 2/12 |
| 0.05 | 5/12 |
| 0.10 | 8/12 |

Thus merely breaking the exact symmetry by an infinitesimal amount is **not sufficient**. The wrong symmetry eigenstate sits inside a finite attraction basin of the nonlinear threshold dynamics.

## Current interpretation
There are now two distinct notions:

1. **Exact invariant sector:** if a start is a symmetry eigenstate with the wrong character, convergence to the ground state is impossible.
2. **Finite nonlinear basin around that sector:** after symmetry is weakly broken, the flow can still return toward the wrong basin. A sufficiently large perturbation can cross into the ground-state basin.

This is consistent with the earlier random-corruption experiment around the exact ground-state signs: the ground state itself also has a very large attraction basin.

The projected-Krylov dynamics therefore has multiple macroscopic basins, with symmetry characters providing exact invariant cores inside them.

## Files
- experiments/triangular_full_symmetry_audit.py
- results/triangular_full_symmetry_audit.json
- experiments/triangular_symmetry_break_escape.py
- results/triangular_symmetry_break_escape.json
