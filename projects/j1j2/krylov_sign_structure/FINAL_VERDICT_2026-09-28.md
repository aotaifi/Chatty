# Final verdict — projected-Krylov sign mechanism campaign
**Date:** 2026-09-28

## Verdict
The mechanism question is closed on the controlled finite-size benchmark.

For the periodic 4x4 spin-1/2 square-lattice J1-J2 Heisenberg model at J2/J1=0.5, the fully recursive loop

\[
(a_k,s_k)
\;\xrightarrow{\text{standard lattice fixed node}}\;
a_{k+1}
\;\xrightarrow{\text{current-sign projected Krylov}}\;
s_{k+1}
\]

was run with

\[
r_k(x)=\frac{H[a_{k+1}s_k](x)}{a_{k+1}(x)s_k(x)}
\]

and with the threshold chosen label-free by minimizing the fixed-amplitude variational energy over the grouped threshold family

\[
s_{k+1}(x)=s_k(x)\,\mathrm{sign}[t_k-r_k(x)].
\]

The exact target ground state was used only afterward for diagnostics.

## Decisive no-target-oracle run

Initialization:
- target Hamiltonian: J2/J1=0.5;
- starting signs: Marshall;
- starting amplitudes: exact ground-state modulus of the unfrustrated J2=0 Hamiltonian;
- no target-J2 ground-state amplitudes or signs enter any update.

Result:
- iteration 1: O_S ~= 0.974537, F_a ~= 0.8690;
- iteration 2: O_S ~= 0.995945;
- iteration 8: O_S ~= 0.999532;
- iteration 15: O_S ~= 0.9999358;
- iteration 20: O_S ~= 0.999998229;
- iterations 20-99: the loop remains on one residual sign pattern while FN continues improving the amplitudes;
- iteration 100: the final tiny sign sector flips and the exact target sign pattern is obtained.

At iteration 100:

\[
O_S = 1.0000000000,
\]

\[
F_a = 0.9999946178,
\]

\[
E_{\mathrm{guide}}-E_0 = 2.6044\times10^{-5}
\]

in total energy.

The final sign hash matches the exact-sign hash previously found in the best-case control.

## Best-case control

Starting instead from the exact target modulus with Marshall signs, the same recursive current-sign loop reaches the exact target signs at iteration 27.

At iteration 30:
- O_S = 1;
- F_a = 0.99999647;
- E-E0 = 8.67e-6 total.

This establishes that the long plateau on the residual sign pattern is not a true fixed sign obstruction. FN amplitude improvement eventually changes the projected-Krylov threshold ordering enough to trigger the final chamber-crossing sign update.

## What was wrong with the older 4x4 loop?

The earlier loop that stalled near O_S~0.9994 did not iterate the current sign state. It repeatedly reconstructed a correction relative to the fixed Marshall coordinate r_M.

That is a materially different nonlinear map.

The new current-sign recursion removes that recurrence and reaches the exact signs.

## Mechanistic interpretation

The campaign supports the following finite-size picture:

1. **FN amplitude step:** for fixed signs, the sign-free fixed-node solve improves the amplitude guide within the current sign chamber.
2. **Projected-Krylov sign step:** using those improved amplitudes, the current local-energy coordinate provides a low-dimensional chamber-crossing rule.
3. **Bootstrap:** better amplitudes sharpen the sign update; better signs reduce FN bias.
4. **Plateaus are allowed:** the sign pattern can remain unchanged for many FN iterations while the amplitudes move.
5. **Escape:** once the amplitudes cross the threshold at which another grouped-r sign pattern has lower fixed-amplitude energy, the Krylov step makes a discrete sign jump.
6. Repeating this process can reach the exact ground-state sign chamber.

## Exact structural results retained from the campaign

- The update is exactly the sign of one Krylov vector \((t-H)\psi\) followed by projection back to the chosen modulus.
- Fixed-amplitude energy-optimal thresholding is energy non-increasing because the current sign pattern is always contained in the threshold family.
- With exact target amplitudes, the exact target signs are a global minimum/fixed point.
- Exact symmetry characters of H are preserved whenever the fixed modulus is symmetry invariant, yielding rigorous disconnected sign sectors.
- Those invariant sectors can be surrounded by finite nonlinear attraction basins.
- Approximate amplitudes can still yield extremely accurate sign structures, but bias the fixed-amplitude sign objective.

## Scientific claim supported

**Finite-size claim:**

> A standard lattice fixed-node amplitude solve alternated with a current-sign, label-free projected-Krylov threshold update forms a self-correcting loop on the tested 4x4 J1-J2 benchmark and can reach the exact ground-state sign structure without target-ground-state information entering the updates.

This is stronger than the earlier one-step sign-reconstruction claim.

## What is NOT established

This does **not** solve the thermodynamic sign problem.

Still open:
- whether the number of outer FN/Krylov iterations remains polynomial with L;
- whether the FN projection length and walker population remain polynomial;
- whether a compact callable approximation to the FN amplitudes can be learned with polynomial samples;
- whether the threshold objective can be estimated with sufficiently low variance at large L;
- whether finite-size symmetry/basin barriers become increasingly difficult;
- whether convergence persists for generic starts/models rather than this tested route.

## Project transition

The mechanism subproject is therefore **CLOSED / ACCEPTED at finite size**.

The active question moves to the parent scaling project:

\[
\boxed{\text{Can sampled/compact FN amplitudes preserve this self-correcting loop at 6x6, 8x8, ... with polynomial resources?}}
\]

The existing 6x6 work has already shown that Krylov-guided FN materially improves over Marshall FN. The current parent bottleneck is the FN mixed-distribution -> callable-amplitude handoff and its population/sample scaling.

## Reproducibility
- experiments/closed_fn_krylov_exact4x4.py
- results/closed_fn_krylov_4x4_J2p5_exactinit.json
- results/closed_fn_krylov_4x4_J2p5_J2zero_init.json
- results/closed_fn_krylov_4x4_J2p5_J2zero_init_100.json
- CAMPAIGN_SUMMARY_2026-09-28.md
