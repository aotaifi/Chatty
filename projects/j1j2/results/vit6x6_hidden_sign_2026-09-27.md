# 6x6 ViT hidden-sign benchmark — 2026-09-27

Checkpoint: cqsl/nqsmagic `vit_J2=0.50_N=6x6_k=0.mpack`
Git blob SHA: `f8c35470b9ea4d402eabe45b930497891139aee1`
Size: 3,018,031 bytes
Model: 6x6 ViT, 154,980 parameters.

## Phase sanity
Weighted non-binary phase leakage on sampled states: ~6.8e-4, so the checkpoint is effectively a real ± sign wavefunction in the important-weight region.

## Isolated connected cores
Flattened seeds use p(x) proportional to |psi|^0.1.
1-hop medians for K=50,100,200,400: 0.871, 0.708, 0.747, 0.926; catastrophic seeds remain.
2-hop, K=50: overlaps = 0.891, 0.166, 0.981, 0.9999, 0.9990, 0.705; median 0.936, worst 0.166; median extension size ~74,078 states.
Naive overlapping-core majority consensus also failed on some seeds (e.g. 0.330).
Failures are not confined to low-amplitude tails: some bad K=100 cores have poor overlap even in the highest-amplitude 10%.

## Paper-style sparsification
2-hop, K=50, preserve core-core bonds and prune weak extension bonds.
cutoff 1e-3: 0.667, 0.126, 0.988, 0.985, 0.969, 0.096.
cutoff 1e-4 (paper default): 0.328, 0.949, 0.972, 0.824, 0.966, 0.924; median ~0.937, worst 0.328.
Conclusion so far: paper-style sparsification improves typical cases but does not remove seed-dependent catastrophic failures with the current custom greedy/local-flip solver.

## Source-faithful Westerhout greedy solver
Reimplemented `greedySolve` directly from Westerhout's public Haskell source: largest-coupling-first cluster merging, cluster-aware sign choice when adding a spin, then sequential local optimization. On six K=50, 2-hop, cutoff=1e-4 instances the weighted core overlaps were 0.864, 0.696, 0.983, 0.984, 0.991, 0.296. Therefore the catastrophic tail is not caused by the earlier approximate greedy solver.

## Boundary-bias diagnostic
For the first four matching deterministic instances, compare the source-faithful reconstructed signs with the hidden ViT signs under the truncated Ising objective. In every tested instance the reconstructed signs have lower energy on the whole finite graph, so a stronger annealer cannot simply restore the hidden signs by optimizing the same truncated objective. More importantly, for bad-overlap instances the sign of the preference reverses on edges touching the protected core: examples are O=0.864 with whole-graph E_hidden-E_rec=+5.185 but core-touching delta=-0.0128, and O=0.696 with +17.375 globally but -0.0325 on core-touching edges. A good instance O=0.983 has essentially zero core disagreement (~8.9e-5). This identifies finite-shell/boundary bias as the leading failure mechanism: the outer truncated shell selects a different Ising minimum and drags the protected core away from the hidden sign structure.

Two later diagnostic cores generated after a fresh process also showed the same qualitative pattern, but they are not numerically combined with the six-instance benchmark because the stochastic core-growth RNG restarted. Next tests use saved deterministic cores so every boundary treatment is compared on exactly identical configuration graphs.

## Next test
Generate and persist six deterministic K=50 cores. On each exact same core compare (a) source-faithful free-boundary solve and (b) an oracle-boundary solve in which omitted Hamiltonian neighbors contribute the exact hidden-ViT exterior Ising field. The hidden signs are used only for this causal diagnostic. If oracle boundary fields rescue bad cores, replace the oracle by a self-consistent/global predicted boundary model; do not spend further effort on larger free-boundary clusters.
