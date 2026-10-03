# ViT 6x6 oracle-boundary diagnostic — 2026-09-27

Purpose: test whether finite-shell boundary bias, rather than Ising optimization, causes the catastrophic hidden-sign reconstruction failures.

Fixed inputs:
- checkpoint `vit_J2=0.50_N=6x6_k=0.mpack`
- K=50 protected connected cores
- 2 Hamiltonian-neighbor extensions
- paper-style global cutoff 1e-4, preserving all core-core bonds
- six saved deterministic cores: `vit6x6_saved_cores.npz`
- saved-core SHA-256: `1761bb5e1ea5c6626141242b84ba4ffffa2a908f195b33f22869d5c21f4cdf65`
- solver: source-faithful Westerhout `greedySolve` reconstruction from public Haskell source, including final sequential local optimization

Three variants on each identical induced graph:
1. `free`: zero exterior field (standard truncated-cluster objective)
2. `cutfield`: exact hidden-ViT exterior signs used only to generate fields from omitted cross-boundary bonds that individually survive the 1e-4 coupling cutoff
3. `allfield`: exact hidden-ViT exterior signs used only to generate fields from all omitted Hamiltonian cross-boundary bonds

The hidden signs are oracle diagnostics only and are never supplied as labels to the free-boundary reconstruction.

## Deterministic free-boundary baseline
Using the saved cores and source-faithful greedy solver, the weighted protected-core overlaps are:
- core0: 0.984853 (n=14,394)
- core1: 0.836906 (n=12,290)
- core2: 0.904856 (n=19,935)
- core3: 0.988315 (n=22,317)
- core4: 0.993982 (n=14,048)
- core5: 0.756225 (n=13,224)

The expensive oracle-boundary calculation is therefore targeted first at core5 (worst) and core1 (second worst), rather than evaluating all six exterior shells.

## Oracle result: core5 (worst free-boundary core)
- free boundary: O = 0.756225
- cutoff-consistent exterior field (only cross-boundary bonds >=1e-4 of the strongest retained coupling): O = 0.756225
- full exterior field from all omitted Hamiltonian cross-boundary bonds: O = 0.982377
- retained graph: 13,224 states, 34,835 retained internal edges
- exterior: 479,264 unique states, 911,458 cross-boundary Hamiltonian edges
- L1 norm of the full exterior field in the normalized Ising units: 2073.896

This is a causal rescue of the bad core by the omitted environment. Crucially, the 1e-4 per-edge cutoff field gives no improvement, while the sum of all omitted weak bonds nearly restores the hidden ViT sign sector. Therefore the failure is not just a poor local optimizer: a very large number of individually sub-threshold exterior couplings collectively generate an important boundary field. Any scalable replacement must estimate this aggregate boundary influence without explicitly materializing the full third shell.

## Oracle result: core1 (second-worst free-boundary core)
- free boundary: O = 0.836906
- cutoff-consistent exterior field: O = 0.960595
- full exterior field: O = 0.995735
- retained graph: 12,290 states, 34,644 retained internal edges
- exterior: 465,294 unique states, 858,400 cross-boundary Hamiltonian edges
- L1 norm of the full exterior field: 1016.485

The second bad deterministic core is also rescued. Unlike core5, the >=1e-4 cross-boundary subset already carries substantial useful information, but the full weak-coupling sum improves further. Across both targeted failures, restoring exterior influence changes the qualitative sign solution in the correct direction. This establishes finite-shell boundary truncation as a causal failure mode of the local-cluster reconstruction on 6x6.

## Constructive follow-up: non-oracle cavity boundary
For retained signs s_i and omitted one-shell states y, define positive fixed-amplitude couplings W_iy = H_iy |a_i a_y|. Ignoring couplings among omitted states for one cavity step, the cross-boundary energy is minimized exactly by
`s_y = -sign(sum_i W_iy s_i)`.
The induced field back on retained state i is `b_i = 2 sum_y W_iy s_y`. Alternate this exact exterior update with local minimization of the retained Ising graph under b_i. This uses only amplitudes and Hamiltonian connectivity; hidden ViT signs are used only afterward to score overlap. The work per boundary update is O(number of cross edges), approximately O(|R| N) for the Heisenberg configuration graph. Core5 is the first test.

### Core5 cavity result
- free-boundary overlap: 0.756225
- fixed Marshall-sign exterior boundary: 0.392353 (worse)
- amplitude-only cavity iteration: 0.756 -> 0.746 -> 0.873 by iteration 1, then converges at 0.869007 after iteration 9
- the cavity objective decreases monotonically from -2892.60 to -2932.70
- cached one-shell data: 13,224 retained states, 479,264 exterior states, 911,458 cross edges (`boundary_cache_core5.npz`, ~18 MB)

Thus a self-consistent aggregate boundary computed without hidden signs substantially repairs the bad core, proving that usable information exists in amplitudes/connectivity alone, but the simple block-cavity approximation remains well below the oracle full-boundary overlap 0.982377. Marshall boundary failure shows that the needed exterior sign structure is not captured by the bipartite baseline.

Next improvement tested: integrate the one-shell exterior spins out analytically. For fixed retained signs, each exterior spin minimizes independently, giving the effective energy `E_eff(s_R)=E_R(s_R)-2 sum_y |sum_i W_iy s_i|`. Optimize this nonlocal effective energy directly via retained-spin flips, allowing the exterior to reoptimize after every flip rather than only after block updates.

### Integrated one-shell verdict — FAILURE as a standalone objective
On core5, direct descent of E_eff from the free-boundary solution improves overlap from 0.756225 to 0.887393. However, different legitimate initializations find substantially lower E_eff with much worse hidden-sign overlap. Among the tested starts, the lowest effective energy is E_eff=-3550.0665 with O=0.446898, whereas the good free-start basin stops at E_eff=-3478.7010 with O=0.887393. Other lower-energy minima have O≈0.22–0.75. Therefore the one-shell integrated objective is not merely hard to optimize: its energetically preferred sectors can be physically wrong relative to the hidden ViT sign structure. Deeper exterior correlations are essential.

This falsifies the independent-one-shell/cavity closure as the final scalable boundary treatment. The remaining constructive direction is a global sign/boundary model evaluated on every Hamiltonian neighbor, so the omitted environment is represented without a finite-shell free boundary.
