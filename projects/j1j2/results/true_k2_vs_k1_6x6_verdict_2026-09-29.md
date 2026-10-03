# True K2 vs K1 node test — 6x6 verdict — 2026-09-29

True second Krylov step:
psi1=(T1-H)psi0, r1=(H psi1)/psi1, psi2=(T2-H)psi1.

T2 was selected label-free from the alpha=1.2 sample reweighted to |psi1|^alpha.

Observed physical sign-change fractions:
- T2=-14.7772: 0.78%
- T2=-15.1381: 1.56%
- T2=-15.8102: 1.95%

These K2 changes move away from the hidden ViT sign pattern, but ViT was treated only as a diagnostic.

Decisive same-amplitude energy comparison on the same 256 physical configurations:
- T2=-14.7772: E[K2]-E[K1] = +0.64584 +/- 0.32244  (2.00 sigma)
- T2=-15.1381: E[K2]-E[K1] = +0.90285 +/- 0.37750  (2.39 sigma)
- T2=-15.8102: E[K2]-E[K1] = +0.96024 +/- 0.38078  (2.52 sigma)

Verdict:
Within this direct two-step Krylov construction, K2 is worse than K1 on 6x6: it changes only ~1-2% of physical configurations, but those changes raise the fixed-amplitude energy. Therefore K1 appears to be a local optimum/fixed point for this Krylov continuation, at least within the tested threshold family. Do not use naive K2 as the next production node update.

This does not prove the K1 node is exact; it rules out this straightforward second-Krylov continuation as an improvement route.
