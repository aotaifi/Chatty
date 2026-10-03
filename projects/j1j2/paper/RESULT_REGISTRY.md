# Paper result registry

This file separates manuscript-ready results from live or superseded project branches.
Figures and claims in the paper should be sourced only from entries marked STABLE unless
explicitly labeled provisional.

## STABLE: exact / controlled sign reconstruction

| Result | Source | Paper use |
|---|---|---|
| Exact 4x4 K1 sweep vs J2 | ../krylov_sign_structure/results/square_exact_energyopt.csv | Main Fig. 1(b,c) |
| 4x4 r(x) distribution and threshold at J2=0.5 | data/k1_threshold_4x4_J2p5.npz (scripts/k1_threshold_4x4.py) | Main Fig. 1(a) |
| Exact N=20 K1 check at J2/J1=0.5 | ../krylov_sign_structure/results/groundstate_k1_20site_exact_J2p5.json | Main Fig. 1(d) |
| Exact 4x4 current-sign FN/K1 loop | ../krylov_sign_structure/results/closed_fn_krylov_4x4_J2p5_J2zero_init_100.json | Main Fig. 2 |
| Exact N=20 current-sign FN/K1 loop | ../krylov_sign_structure/results/closed_fn_krylov_20site_J2p5_J2zero_init.json | Main Fig. 2 |
| Exact-init 4x4 control | ../krylov_sign_structure/results/closed_fn_krylov_4x4_J2p5_exactinit.json | Control / possible supplement |
| 6x6 K1 mechanism | ../krylov_sign_structure/WHY_ONE_KRYLOV_STEP_VERDICT_2026-09-30.md | Main Fig. 3(a) |
| 6x6 K1 vs Marshall vs ViT fixed-amplitude energy | data/energy_krylov_vs_vit_6x6_3479622.npz | Main Fig. 3(b); quote paired K-M = -4.79(125)e-3/site, not the difference of means |
| 6x6 naive K2 falsification | ../results/true_k2_6x6_3471990/true_k2_6x6.npz | Appendix |

## STABLE: fixed-node scaling

| Result | Source | Paper use |
|---|---|---|
| 6x6 K1-FN vs Marshall-FN | ../results/krylov_fn_6x6_matched_verdict_2026-09-28.md | Main Sec. III C |
| 6x6 practical FN/K1 fixed point | ../results/fn_krylov_closedloop_6x6_verdict_2026-09-29.md | Main Sec. III C |
| 8x8 K1-FN replicas | data/fn8_krylov_M128_3471544.npz | Main Fig. 4 |
| 8x8 Marshall-FN replicas | data/fn8_marshall_M128_3471545.npz | Main Fig. 4 |

## STABLE: amplitude handoff and SR geometry

| Result | Source | Paper use |
|---|---|---|
| Exact 4x4 MLE half-step identity | data/fn_mle_halfstep_exact4x4.out and ../results/fn_mle_halfstep_4x4_2026-09-30.md | Main Fig. 5(a) |
| Failed Adam/full-batch Euclidean MLE | ../results/fnmle_vit8_gate_2026-10-01.md | Appendix |
| SR cross-replica direction, lambda=1 | data/fnmle8_sr_lambda1_linesearch_3506041.json | Main Fig. 5(b) |
| Iteration-2 SR training diagnostics | data/fnmle_sr_iter2_train_3506071.json | Provisional; not used in current main text |

## EXTERNAL BENCHMARK

The same 8x8 periodic J2/J1=0.5 geometry is tabulated in Qian and Qin,
Chinese Physics Letters 40, 057102 (2023), DOI 10.1088/0256-307X/40/5/057102.
The literature values are frozen in data/literature_8x8_pbc_J2p5.json.

This benchmark is directly comparable in Hamiltonian, size, periodic boundary conditions,
and energy normalization. It includes CNN, RBM+PP, DMRG, VMC, and FAMPS results.

## PROVISIONAL / LIVE

| Result | Status | Rule |
|---|---|---|
| 8x8 SR iter-1 FN mean -31.90123 | replicated inconsistently by a near-identical checkpoint | Do not claim robust energy lowering yet |
| 8x8 independent SR rebuild mean -31.83896 | real reproducibility warning | Keep visible in diagnostics |
| Iteration-2 FN mean -31.81537 | stable failure of repeated update | Main limitation |
| Frozen-s1 a2 diagnostic | replicas -31.84646 and -31.80947 (mean -31.82797) | Amplitude update causes the iteration-2 loss |
| Exact 4x4 frozen-H_FN factorized SR | recovers 99.7% of rebuilt E_FN gain (3551953/4) | Quoted in Sec. III E as representability evidence |
| 8x8 frozen-H_FN SR and split-amplitude K1 A/B | live as of 2026-10-03 | Not in paper |

## SUPERSEDED / FORBIDDEN FOR MAIN FIGURES

- krylov_sign_structure/results/square_exact_sweep.csv:
  older threshold construction; superseded by square_exact_energyopt.csv.
- Old 4x4 Marshall-referenced FN/K1 recurrence near O_S ~ 0.9994:
  superseded by the current-sign recursive loop reaching O_S=1.
- Early statement that four-bin g(r) itself was falsified at 6x6:
  later audits identified correlated/population-biased FN mixed samples.
- Classifier threshold/node results using physical a^2 weights in the alpha=1.2
  threshold fit: invalid measure and must not be resurrected.

## Error-bar language

- Exact diagonalization: exact up to numerical eigensolver precision.
- Variational Monte Carlo / sampled fixed-amplitude energies: report estimator uncertainty
  exactly as defined by the generating analysis.
- Two-replica FN error bars: call them between-replica spread proxies, never rigorous
  asymptotic statistical uncertainties.
- Literature values: reproduce the uncertainty notation of the cited source.
