# STATUS

## Current state
The original conceptual question — why the one-step Krylov sign update works so well at J2/J1=1/2 — is empirically resolved on exact finite clusters.

## Current mechanism
With fixed amplitudes a(x), define psi_s=a s and r_s=(H psi_s)/psi_s. The sign of the Krylov vector (t-H)psi_s is exactly

T_t[s] = s sign(t-r_s).

Choosing t by minimizing the fixed-amplitude energy over this threshold family gives an energy-non-increasing nonlinear sign map, because the family always contains the current s.

With exact ground-state amplitudes, the exact signs s_* are a fixed point and global minimum. Numerically they are a strong attractor.

## Decisive results
- J2/J1=0.5 Marshall: O_S 0.974538 -> 0.999319 -> 0.99999823 -> 1; exact after 3 updates.
- J2/J1=0.6 Marshall: exact after 4.
- J2/J1=0.8 Marshall, despite one-step failure: exact after 10.
- J2/J1=1.0 Marshall, despite one-step failure: exact after 9.
- J2/J1=0.8 and 1.0 stripe: exact after 4.

Thus a good gauge makes the first step exceptionally strong, but is not required for eventual convergence on the tested 4x4 square cluster.

Random-corruption tests show a very large empirical ground-state basin: all 32/32 sampled square starts converged through q=0.42 wrong physical probability weight (O_initial about 0.16).

## Triangular result
On the 6x3 triangular Heisenberg model, three simple two-color gauges flow to wrong fixed points with essentially zero ground-state sign overlap, even under repeated updates.

But random perturbations around the exact triangular signs recover robustly: all 8/8 sampled starts converged through q=0.45, and 4/8 at q=0.49.

Therefore the triangular obstruction is **competing structured basins**, not absence of local contraction around the exact signs.

The earlier triangular one-step oracle <=0.0604 was obtained by allowing an ordering that could split degenerate r values. The corrected grouped-threshold dynamics is the authoritative result.

## Current verdict
The one-step “miracle” is the first strong contraction step of a projected-Krylov sign descent. The unresolved theory problem is no longer why the first step works, but what structural invariant distinguishes the ground-state attraction basin from wrong threshold-stable basins.

## Primary files
- BASIN_DYNAMICS_VERDICT_2026-09-28.md
- ITERATED_KRYLOV_CONVERGENCE_2026-09-28.md
- experiments/basin_radius_exact.py
- experiments/basin_radius_refined.py
- experiments/wrong_gauge_iterated.py
- experiments/triangular_iterated_exact.py
- experiments/triangular_basin_exact.py
- results/basin_radius_exact.{json,csv}
- results/basin_radius_refined.{json,csv}
- results/wrong_gauge_iterated.json
- results/triangular_iterated_exact.json
- results/triangular_basin_exact.json

## New analytic basin criterion
A symmetry-sector invariant is now proved and numerically verified. If P commutes with H, the fixed modulus is P-invariant, and psi_s=a s is a P eigenstate with character chi, then r_s is P-invariant and every threshold update preserves chi.

This explains the best triangular obstruction exactly:
- triangular ground state: Tx=+1;
- maxcut_x and parity_xy starts: Tx=-1;
so those starts can never converge to the ground signs.
By contrast square ground state, Marshall, and stripe all have Tx=Ty=+1, consistent with eventual recovery.

See `SYMMETRY_BASIN_INVARIANT_2026-09-28.md`.

## New mechanism refinements
Two follow-ups sharpen the picture further.

1. **iid random square starts:** with exact amplitudes, completely random initial signs still reach the exact signs in 25/64 trials at J2/J1=0.5, 24/64 at 0.6, and 39/64 at 1.0. Thus appreciable initial sign overlap is not required.

2. **Amplitude robustness:** replacing a_* by a_*^gamma shows that gamma=0.75 still gives exact recovery at J2/J1=0.5 and 0.6 and O_S=0.99999914 at J2/J1=1, whereas gamma=0.5 largely destroys the correction. The amplitudes need not be exact, but their weighted profile matters strongly.

There is also an exact identity in the ground-state sign gauge:
a_x^2[r_x(s)-E0] = -2 sum_{y!=x:z_y!=z_x} J_xy,
with off-diagonal J_xy=H_xy a_x a_y s_x^* s_y^*. Thus r-E0 is precisely the weighted boundary load of the current wrong-sign domain.

## Next question
The remaining theory problem is now very specific: derive a sufficient **boundary-load contraction criterion**, compatible with the symmetry-sector invariant, that separates the square ground-state basin from wrong fixed points such as triangular parity_y. A separate practical question is how the threshold-energy minimization behaves when estimated stochastically rather than exactly.
