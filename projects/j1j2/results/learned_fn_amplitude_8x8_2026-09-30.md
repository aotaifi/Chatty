# Learned callable FN amplitude — 8x8, 2026-09-30

## Motivation
The static four-bin (g(r)) handoff is closed because it is both noisy at 8x8 and creates a
recursive local-energy/cache explosion when used inside the next K1/FN iteration.

The replacement is a flat learned positive amplitude:
[
log a_{\rm learn}(x)=log a_{0}(x)+\eta,g_\theta(x),
]
where (a_0) is the original ViT amplitude and (g_\theta) is a small translation-invariant
CNN trained as a density-ratio classifier between FN mixed walkers and (a_0^2) samples.
## Replica-transfer gate
Two independent M=128 8x8 K1-FN populations were used only as separate train/test domains.

A linear translation-invariant correlation model transfers only weakly:
- rep1 -> rep2 AUC ~0.535
- rep2 -> rep1 AUC ~0.531.

The nonlinear periodic CNN transfers materially better:
- rep1 -> rep2 best AUC = 0.6412
- rep2 -> rep1 best AUC = 0.6361.

Thus the replicas share a nonlinear amplitude-correction signal not captured by the old scalar
four-bin coordinate.
## Calibration and final callable model
Independent cross-replica calibration shows raw classifier logits are overconfident:
- rep1 -> rep2 optimal multiplier eta = 0.3207
- rep2 -> rep1 optimal multiplier eta = 0.3986.

The production residual uses their mean, eta = 0.35967, clipped below 1.
Final pooled model:
- checkpoint: `results/fn_residual_cnn_8x8.mpack`
- backend: `experiments/learned_fn8_callable_amplitude_adapter.py`
- handoff bundle: `results/learned_fn_handoff_8x8.npz`
- direct callable API: `experiments/callable_fn_loop.py`.

Backend smoke test passed: finite and deterministic; repeated-call max difference 0.
## Threshold / pool reweighting
The original threshold sample is from (a_0^{1.2}), so learned-amplitude physical weights are
(w_1=w_0exp(2\eta g_\theta)). Train and validation threshold samples are combined with
sample-count weighting.

Effective sample sizes:
- threshold train alone: 197.2
- threshold validation alone: 205.9
- combined threshold sample: 358.0
- physical initialization pool after learned reweighting: 2338.3.

## Production test
Corrected H100 Slurm job: **3494473**.
The earlier 3494470 submission was intentionally cancelled before use so the combined threshold
sample could replace the train-only threshold inference.

Job 3494473 runs two independent M=128 replicas (seeds 12001, 12002), beta_target=1.2,
burn_beta=0.4, tau_max=0.025. It will report:
1. learned-amplitude K1 threshold;
2. learned-vs-original K1 weighted node-change mass;
3. two FN replica energies and between-replica spread;
4. amplitude evaluation count / scaling behavior.

Baseline for comparison: original 8x8 K1-FN mean E=-31.88542411 with two-replica SE 0.00159161.
