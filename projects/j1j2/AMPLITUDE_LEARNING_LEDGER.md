# Amplitude-learning ledger (compiled 2026-10-03)

Every attempt to turn fixed-node (FN) information into a callable positive amplitude
(a model evaluable on any configuration), compiled from STATUS.md, FAILED_ROUTES.md,
results/*.md and experiment headers. E_FN = fixed-node energy; a_k, s_k = current guide.

| date | lattice | target | model / optimizer | data | outcome | verdict | why | key file |
|---|---|---|---|---|---|---|---|---|
| 09-28 | 4x4 | 4-bin density ratio g(r), eta=0.5 | 4-bin correction | 1e4+1e4 samples from the exact q, f; 12 iterations | O=0.99926 held; fidelity 0.9994-0.9999 | pass (ideal samples) | — | results/fn_density_ratio_handoff_4x4_2026-09-28.md |
| 09-28 | 4x4 | same, from real GFMC | 4-bin | GFMC M=32/128 | bins differ 0.07-0.12 between replicas and from exact | fail | correlated populations, low effective sample size | STATUS "Controlled 4x4 diagnosis" |
| 09-28/29 | 6x6 | 4-bin g(r) | 4-bin | GFMC M=32 -> 128 | M=128: one handoff, then sign change 0 and next correction < 0.36 sigma | partial (fixed point after 1 iteration) | replica-dependent mixed distributions | results/fn_krylov_closedloop_6x6_verdict_2026-09-29.md |
| 09-29/30 | 8x8 | 4-bin g0 | 4-bin | 2 replicas, M=128 | no bin > 2 sigma; recursive cache 14.7M states, OOM | fail | noise + cache wall | results/8x8_krylov_fn_scaling_verdict_2026-09-30.md |
| 09-30 | 8x8 | density-ratio classifier | CNN residual | replica walkers | cross-replica AUC 0.64; E_FN -27.42 vs -31.885 | fail | moves 16-45% of the node | results/learned_fn8_classifier_verdict_2026-09-30.md |
| 09-30 | 8x8 | classifier, warm-started ViT (full/head/last block) | ViT, Adam | ~5.8k per replica | val AUC 0.45-0.53 | fail | no transferable signal | experiments/warmstart_vit_*_gate_8x8.py |
| 09-30 | 4x4 | MLE a^2 ∝ a phi_FN | small MLP | 1e4 exact mixed samples | fidelity 0.968 -> 0.950 | fail | net too small for the guide | results/fn_mle_halfstep_4x4_2026-09-30.md |
| 10-01 | 8x8 | walker MLE | full ViT, Adam | GFMC M=128 x2 | held-out log-lik -0.25..-0.29; replica gradient cosine -0.853 | fail | optimizer geometry | results/fnmle_vit8_gate_2026-10-01.md |
| 10-01 | 8x8 | walker MLE + SR (lambda=1) | ViT, SR, eta=0.01 | GFMC M=128 x2 | held-out +0.0156; E_FN -31.9012 (vs -31.8854); independent rebuild -31.839 | not reproducible | — | results/fnmle_sr_closedloop_8x8_iter1_2026-10-01.md |
| 10-01 | 8x8 | MLE + SR, iteration 2 | ViT, SR | GFMC | E_FN -31.815; frozen-s1 -31.828; smaller steps all worse | fail | >99% of one-hop neighbour ratios unsupported by samples | STATUS 10-01 |
| 10-01 | 8x8 | fixed-sign physical energy | ViT, SR | 256 states | held-out energy better, frozen-s1 E_FN +0.051 | fail | variational descent != FN descent (H_FN depends on guide) | results/fixedsign_energy_sr8_3509464 |
| 10-01 | 4x4 | exact E_FN Hellmann-Feynman force | natural gradient | exact | 1.74x descent vs Euclidean | pass | — | results/efn_hf_sr_exact4x4.json |
| 10-01 | 4x4 | forward-walking E_FN force | — | GFMC M=20000 | cosine 0.41-0.62 | fail | noise | results/forward_walking_efn_gradient_exact4x4.json |
| 10-01/02 | 4x4/8x8 | mixed-walker E_FN force + QGT | ViT, SR | exact 4x4: cosine 0.97; walkers: ~0.03 | 8x8 surrogate gate passed, frozen-s1 E_FN +0.046 | fail | surrogate != E_FN | STATUS 10-02 |
| 10-02/03 | 4x4 | frozen-H_FN Rayleigh quotient, a = a_k exp(r_theta) | residual ViT, Adam | exact | 81-98% of rebuilt E_FN gain | partial | Adam | results/frozen_hfn_factorized_sr_exact4x4_2026-10-03.md |
| 10-03 | 4x4 | same | residual ViT (155k params), SR | exact | 99.7% of rebuilt E_FN gain | **pass (exact)** | — | same |
| 10-03 | 4x4 | pointwise log-ratio / Hamiltonian-edge differences | residual ViT, Adam | exact | 45.9% / 65.9% (best checkpoint) | partial | Adam drift, no early-stopping rule | jobs 3565720/1 |
| 10-03 | 8x8 | frozen-H_FN SR | residual ViT, SR | 3 x 256-512 a1^2 samples | 2 accepted steps lowered train/val frozen energy, then OOM | partial (not physically tested) | implementation OOM | frozen_hfn_sr8_pilot_chunked.py |

## Pattern
- Exact 4x4 with full-space information works (best: frozen-H_FN + SR, 99.7%).
- Real GFMC walkers: correlated, replica-dependent; estimators noise-dominated.
- 8x8: Euclidean/Adam gradients anti-aligned across replicas (SR needed); >99% of the neighbour ratios the sign step needs are never sampled; surrogates that pass held-out gates still raise E_FN.
- Only the frozen-H_FN objective is consistent with FN; untested end-to-end at 8x8.
- **Never done:** a network trained on sampled data, plugged into the 4x4 loop and iterated.

## Live / unresolved (2026-10-03)
3565752 (8x8 chunked frozen-H_FN SR retry); 3554536, 3554836, 3554843, 3554893, 3554908 (8x8 pilots, outcome not recorded);
3560026/3560054 (4x4 Adam ITE); 3566702/3566703 (split-amplitude K1 A/B). Paderborn GPUs down 2026-10-05..09.

## Plan: 4x4 learning ladder (scored against ED)
1. exact target + network + SR — passes (99.7%).
2. samples from the exact FN distribution + network + SR, iterated in the loop.
3. real FN walkers + network + SR, iterated.
