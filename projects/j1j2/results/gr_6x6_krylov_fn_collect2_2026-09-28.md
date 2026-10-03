# 6x6 Krylov-FN four-bin g(r) collection — 2026-09-28

Two independent direct-ViT Krylov-FN replicas completed successfully at M=32, beta_target=2.0, burn_beta=0.4, tau_max=0.025.

Replica 9001:
- Emean = -18.15081797
- tail8 = -18.14134301
- mixed samples = 2336, unique = 791
- g = [-0.0747573, -0.1241628, +0.0023822, +0.1950386]

Replica 9002:
- Emean = -18.05314756
- tail8 = -18.07979698
- mixed samples = 2176, unique = 754
- g = [-0.0725455, +0.0614802, -0.0617888, +0.0730463]

Combined:
- Nmixed = 4512, unique = 1545
- counts = [1041,1083,1095,1293]
- g = [-0.0753858, -0.0317579, -0.0298131, +0.1362909]

Replica difference:
- delta g = [-0.00221, -0.18564, +0.06417, +0.12199]

Interpretation:
- The lowest-r bin is highly reproducible (~ -0.073).
- The middle/high bins are not reproducible at M=32; differences are comparable to or larger than the inferred correction.
- The two FN projected energies themselves differ by ~0.098 total energy, confirming substantial population/trajectory noise.
- Therefore the criterion for freezing the four-bin amplitude correction is NOT met.
- This does not yet falsify the four-bin representation; it shows M=32 direct-ViT FN sampling is too noisy for a stable amplitude handoff.

Current next-step options:
1. Increase walker population / independent replicas to reduce FN mixed-distribution noise before fitting g(r).
2. Introduce a cheaper callable amplitude guide so larger walker populations become affordable, while validating against direct-ViT on the same states.
Do not launch the second 6x6 FN iteration with the current combined g.
