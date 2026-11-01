# Lead's blind ranking (written 2026-11-01 before reading Astra's answer cj-a65b62dbf0ac)
Criterion: P(aug 6x6 < 4.0e-6 AND walker <= 20x plain at 10x10).
1. (d) two-body Jastrow e^{J2} on B (O(1) two-swap ratios with cached fields; cost class unchanged, ~5.4x at 6x6, falling ~1+c/N):
   20%. Mechanism: Jastrow carries bulk density-density correlations the PP misses, so it raises the per-round storage r of the bulk.
   Prediction: aug 4.0-4.2e-6. Kill: round from theta_4 with Jastrow, bare drop <= 0.5e-6. Cost ~1.5 H100-h.
2. (c) second explicit Jacobi level: accuracy likely (aug ~3.0-3.5e-6, 60%), but the walker cost is three-hop
   (rank-6 three-swap ratios, ~d^3 of them), which I derive at >= 50x at 6x6 and not falling with N: 15% on the joint criterion.
   Kill: measured walker factor > 20x at 10x10 (CPU timing, ~2 CPU-h).
3. (a) active RBM bulk: Test N says RBM features miss the round direction (eps^2 0.95-0.98). Its two-swap ratios are O(M) each,
   so the head costs ~1+32 alpha by op count, which does not fall with N: 10%. Prediction: aug 4.2-4.4e-6.
4. (b) backflow PP: rank 12-14 per swap gives 36.5x at 6x6 (measured T2), and the window rank does not shrink with N: 5%.
