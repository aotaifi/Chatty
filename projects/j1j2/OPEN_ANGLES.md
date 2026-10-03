# Open angles — parked side projects

These are intentionally **not part of the currently running main FN/Krylov scaling campaign**. Pick them up one at a time in separate chats when useful.

1. **Why one Krylov step works so well — RESOLVED on 6x6 (2026-09-30)**
   Mechanism: the amplitude-weighted off-diagonal Marshall field measures the competition between J1 and frustrating J2 hopping and removes ~95% of physical wrong-sign mass in K1. Afterward the useful residual signal is ultra-rare and naive K2 has poor precision. See `krylov_sign_structure/WHY_ONE_KRYLOV_STEP_VERDICT_2026-09-30.md`. Cross-size/model generalization remains under angles 3 and 5.

2. **Raw multi-Krylov continuation — DEPRIORITIZED/FALSIFIED as production route on 6x6**
   Direct K1 -> true K2 carries the Krylov-distorted modulus and worsens fixed-amplitude energy. Keep only as a diagnostic, not the scalable iteration.

9. **FN-refresh -> projected sign-step mechanism — RESOLVED conceptually (2026-09-30)**
   The correct iteration is `s_k -> FN amplitude refresh a_{k+1} -> recompute local-energy field -> projected K1-like sign step s_{k+1}`, not raw K1->K2->K3. Exact 4x4 recursive tests converge to the exact node when the refreshed amplitude is sufficiently accurate; 6x6 reaches a practical fixed point after one refresh. On 8x8 the current coarse four-bin FN handoff changes only ~5.2e-5 physical sign mass and is slightly harmful relative to the hidden ViT diagnostic, so the unresolved issue is no longer the sign-step mechanism but the fidelity/scaling of the FN -> callable-amplitude handoff. No claim of a general monotonic theorem `better amplitude => better signs` is made.

3. **Frustration dependence / K1 reconstructibility — RESOLVED conceptually on exact 4x4 (2026-09-30)**
   K1 is not uniquely special at J2=0.5. Away from the frustration crossover, one projected step removes most wrong-sign mass when started from the appropriate parent sign basin (Marshall on the low-J2 side, stripe on the high-J2 side). Near J2~0.62-0.65, one-step compression can worsen overlap even though the iterated exact-amplitude map still converges; Marshall leaves the attraction basin between J2=0.650 and 0.655 while stripe remains convergent. Thus reconstructibility is governed primarily by sign-basin membership/distance, not J2 alone. See `krylov_sign_structure/frustration_dependence/VERDICT_2026-09-30.md`. 8x8 size dependence belongs in the main scaling loop.

4. **Node-quality diagnostics without exact ground truth**
   Develop/use FN energy, fixed-amplitude energy, K1-K2 sign changes, and stability under amplitude improvement.

5. **Generalization beyond square-lattice J1-J2**
   Test triangular and other frustrated magnets for the same sign-reconstruction principle.

6. **Finite-temperature / CTQMC analogue — PARKED at amplitude-relearning gate (2026-10-03)**
   The adaptive K1/K2 sign-update mechanism is numerically strong on the exact 20-site benchmark through beta=3, but long-beta scaling requires periodic compression of the positive amplitude. Final factorized nonlinear overlap-SR discriminator 3565142 failed promotion: only ~0.61% blind Hamiltonian-edge RMSE improvement after 20 steps while physical fidelity worsened. Do not build a finer CTQMC framework or run another finite-T compressor search yet. Reopen only when the shared FN amplitude-learning branch has a materially better scalable callable-amplitude handoff. See `results/finite_tau_amplitude_relearning_subangle_closure_2026-10-03.md`.

7. **Representation / Z2-holonomy connection**
   Understand whether Krylov reconstruction is effectively learning or correcting the sign-flux obstruction found in the no-go work.

8. **Compression of the amplitude correction**
   Understand why a tiny representation (currently four static-r0 bins) captures the useful FN correction and when this representation fails.

## Current main campaign — do not mix with the above

The active campaign is the FN -> amplitude handoff -> Krylov loop and its size/resource scaling. Current focus: 8x8 Krylov-FN vs matched Marshall-FN, plus the already-submitted true-K2 diagnostic as a node-quality side measurement.
