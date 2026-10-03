# 8x8 Krylov-FN scaling verdict — 2026-09-30

## Jobs
- Krylov-FN: Slurm 3471544, COMPLETED 00:13:24 on gpu1019.
- Marshall-FN control: Slurm 3471545, COMPLETED 00:02:58 on gpu1019.
- 6x6 amplitude/node audit: Slurm 3479622, COMPLETED 00:01:09 on gpu1019.

## 8x8 matched FN comparison
Both used M=128, beta_target=1.2, two independent replicas.

Krylov-FN replica means:
- seed 10501: -31.8870157191
- seed 10502: -31.8838325000
- mean: -31.8854241096
- between-replica SE: 0.00159161
- replica difference: 0.00318322

Marshall-FN replica means:
- seed 11501: -31.7038264769
- seed 11502: -31.6511272585
- mean: -31.6774768677
- between-replica SE: 0.02634961
- replica difference: 0.05269922

Difference:
E_Krylov - E_Marshall = -0.20794724 +/- 0.02639763
(using between-replica spread; ~7.9 sigma as a two-replica diagnostic, not a precision asymptotic error estimate).

Verdict:
The one-step K1 sign guide plus fixed-node projection survives from 6x6 to 8x8 at M=128. It is both substantially lower in projected energy than Marshall and much more replica-stable.

## First 8x8 density-ratio correction
Four-bin fits:
- rep1 = [0.20594036, 0.02587822, -0.13992589, -0.09165562]
- rep2 = [0.07177718, 0.00104940, -0.03851878, -0.03409760]
- combined g0^(8) = [0.14059278, 0.01262784, -0.08930914, -0.06367385]
- pair-noise estimates = [0.09486770, 0.01755663, 0.07170566, 0.04069966]

The first-bin signal is only ~1.5 pair-noise units and no bin exceeds 2. Therefore g0^(8) is not a precision amplitude reconstruction merely because the FN energies are stable.

## 6x6 amplitude/node audit (3479622)
Using the previous four-bin a1 correction, the fixed-sign local-energy dispersion proxy did not improve:
- train SD: 0.41725 -> 0.43251 (x1.0366)
- validation SD: 0.57529 -> 0.58862 (x1.0232)

This is supporting evidence that the four-bin correction is not a faithful callable FN amplitude representation. It is not as decisive as an exact 4x4 Delta_FN test because this proxy is not itself the exact projected-amplitude error.

A translation-invariant correction network trained at fixed a1/K1 never produced nonzero held-out sign-flip mass in three seeds; all validation selection chose the initial checkpoint. Treat this as no positive evidence for a richer nodal correction, not as a proof of nodal optimality, because hard-sign/STE optimization can get stuck.

## 8x8 refresh implementation follow-up
A later closed-loop implementation using g0^(8) found:
- threshold shift T0=-28.37108 -> T1=-28.00271
- reweighted training sign-change mass = 0.00279486 (3 changed sampled states)

The full second FN pass then blew up the nested callable-amplitude/sign caches:
- by beta~0.44: r0 cache ~14.7 million states, ~625.7 million ViT evaluations
- H100 job was OOM-killed despite 128 GB.

A separate node diagnostic on the three changed states found physical changed mass 5.23e-5; relative to the hidden ViT sign diagnostic, one change helped and two harmed, slightly reducing overlap (0.998253 -> 0.998155). Hidden ViT signs are diagnostics only, but there is no evidence that this tiny projected node update is useful.

## Bottom line
1. K1 itself scales positively to 8x8.
2. M=128 is sufficient for a stable first FN energy comparison at 8x8.
3. The current four-bin g amplitude handoff should not be treated as a scalable closed-loop representation.
4. Repeated g -> recompute-local-energy -> FN creates a severe nested-call/cache prefactor at 8x8 even though the node changes only at ~10^-3 mass.
5. Further amplitude work should move to a representation that directly produces callable FN amplitudes (frozen deep-feature residual or full-network FN-informed retraining), rather than another static four-bin layer.
