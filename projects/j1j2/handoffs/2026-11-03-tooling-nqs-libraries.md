# 2026-11-03 tooling angle: should we reuse NQS libraries? No.

What was asked: the PI asked whether NetKet, Quantax, jVMC or similar should replace parts of our custom JAX pipeline.
Full record: PLAN_tooling.md (ranked table, benchmark B1, answer); inventory with file:line evidence: angles/tooling/INVENTORY.md.

What was built
- A venv with NetKet 3.23, quantax 0.4 (Ao Chen; JAX + Equinox, MinSR, fermion mean-field models), lrux 0.2 (Chen & Roth,
  arXiv:2602.05255, low-rank det/Pfaffian updates in JAX), jVMC 1.5.8: /scratch/a/A.Otaifi/venvs/tooling (wheels only).
- Benchmark B1 (angles/tooling/bench/b1_lib.py): all one-swap ratios of one pair-product determinant, ours vs quantax/lrux
  styles, CPU and A40, plus a drift test of a persistent inverse on near-node 8x8 configurations.

Result (A40, n=50, time relative to ours): quantax E_L path 1.7-1.8x slower; persistent inverse + our ratio table 0.62-0.69x,
but near-node drift needs a refresh every 2 steps at worst (median 8-26), which leaves <= 1.45x on the plain PP step and
~1.1x on the Jacobi / two-PP guides. Pre-registered rule: keep ours.

Rejected and why
- Library MinSR/SR (quantax solve_equation, NetKet SRt): the same damped Cholesky we already do; our fit loss is custom.
- NetKet FullSumState: no symmetry-reduced sectors; 6x6 Sz=0 (9.1e9 states) exceeds its int32 index.
- GCNN / symmetric architectures: not size-portable (TRIED_IDEAS K8), and we use a published ViT and our PP.
- Lattice FN-GFMC: none of the libraries has any.

Three things to understand
1. All-neighbour ratios from an inverse cost O(n) each only with precomputed products (T = f_DD G); lrux's general formula
   pays a matrix-vector product per neighbour, so for d ~ 4n neighbours it loses to our three GEMMs.
2. A persistent (Woodbury-updated) inverse loses ~eps x cond(F) per step; near nodes cond(F) reaches 1e8, so it must be
   refreshed often, and the refresh is the cost it was meant to save.
3. The real levers for the 10x10 loop guide are ours, not a library's: the two-swap kernel (71% of the Jacobi step) and the
   symmetry-image count (1600 vs 200).
