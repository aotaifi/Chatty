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

## Next question
Formalize a basin criterion in terms of symmetry / weighted sign flux / effective sign-Ising frustration, then test whether it predicts the triangular wrong basins and the square global-like basin without reference signs.
