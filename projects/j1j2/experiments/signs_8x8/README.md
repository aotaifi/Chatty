# Stored Krylov signs at 8x8 (pre-registration)

Periodic 8x8 J1-J2, J2/J1 = 0.5. Written 2026-10-12 BEFORE any training-data, referee or training run of this study.
Context: RESEARCH_MAP 2 (stored signs: composite net + one exact hop), results/sign_learnability/README.md (rules),
results/stored_signs_6x6/, experiments/loop_8x8/ (8x8 ViT guide, referee; reused read-only).

## Question
Does the stored-sign machinery (composite sign net + ONE exact Krylov hop) work at 8x8, at fixed amplitude
a = |psi_P| (D4 x flip projected 8x8 ViT), so that the loop can update signs at sizes without exact data?
Score: variational energy <H_ph> of (a, s), paired between sign functions on the same samples.

## Measured before this file (label-free, existing referee data only)
`s8_thresh.py` on the five loop_8x8 referee batches (15360 samples x ~ a^2, two-hop guide data;
`results/signs_8x8/step1_threshold_existing_referee.json`):
- One Krylov step on the ViT's own sign, s1 = s_V sgn(T - r_V), r_V = E_L of (a, s_V):
  pooled energy-optimal T1 = -20.059 (total-energy units; E ~ -31.93), in-sample gain -2.1e-6/site,
  cross-fitted (leave-one-batch-out) gain **+2.1(2.6)e-6/site**: no resolvable gain.
- T is unstable under half-sample splits (-3.9 ... -66.7): the energy curve is flat within noise.
- At T1 no a^2 sample flips; 0.34% of their neighbours flip. On tempered (beta = 1/2) samples 1.2% of x and 3.9% of
  neighbours flip (loop_8x8 pool).
- Consequence: from the ViT sign the Krylov step is energetically a null step at 8x8 (as at 6x6: 1.32e-4 -> 1.27e-4).
  The ViT-sign arm below is therefore a precision/no-harm test. A second, discriminating arm starts from the Marshall
  sign (sign-poor guide, the regime where the sign step matters; reviewer 3c "sign-poor guides at 8x8+").

## Method (fixed here)
- Guide amplitude a = |psi_P|, psi_P = sum of the 16 D4 x flip images of the complex ViT (loop_8x8 `l8_core.Guide`,
  fp32, highest matmul precision). ViT sign s_V(x) = sign cos(arg psi_P(x) - phi), phi = 0.056006. Marshall sign M.
- **Data** (`s8_twohop.py`): Metropolis chains at a^(2 beta) of psi_P; for every sample x the guide on the full
  two-hop set {x} u N1 u N2 (~1e4 configurations). From it any sign b gives r_b on x and all its neighbours, and the
  exact local energy of (a, b) and of (a, b sgn(T - r_b)) at x (no bond subsampling).
  - TR: training data, beta = 1/2, 1024 chains x 4 samples (burn 40, thin 2 sweeps). Chains with id % 5 == 0 are held
    out (20%). Training states = samples + ALL valid one-hop neighbours (~146 per sample).
  - RA: selection referee, beta = 1, 1024 chains x 3 (burn 40, thin 2), own seed. Used ONLY to choose thresholds.
  - RB: evaluation referee, beta = 1, 2 x (1024 chains x 3), own seeds. All energies below are measured on RB.
- **Arm V (primary, as specified).** Base sign s_V, T1 = -20.059 (fixed above, from independent data).
  Labels on TR states: y = M s_V sgn(T1 - r_V) (the composite sign relative to Marshall). The exact one-hop Krylov sign
  is K1_V = s_V sgn(T1 - r_V).
- **Arm M (secondary, discriminating).** Base sign M; T_M = energy-optimal on RA (label-free). Labels y = sgn(T_M - r_M);
  K1_M = M sgn(T_M - r_M).
- **Net** (`s8_net.py`): translation-invariant periodic CNN, 64 channels x 6 layers, s_net = M sgn(logit), logit
  averaged over the 16 D4 x flip images (the composite targets are exactly invariant). BCE, AdamW lr 2e-3 warmup-cosine,
  wd 1e-4, batch 1024, 6e4 steps, one random image per sample. Final checkpoint (held-out error logged 20 times).
- **Hop.** s_net + hop = s_net sgn(T2 - r_net), r_net = E_L of (a, s_net), T2 energy-optimal on RA (label-free).
- **Measurements (RB, paired, SE from per-chain means over 2048 independent chains):**
  - <H_ph>(a, s) for s in {s_V, K1_V, s_net^V, s_net^V + hop}; and {M, K1_M, s_net^M, s_net^M + hop} (+ s_V).
  - Disagreement fraction of s_net with K1 on held-out TR samples x and on their neighbours (and on RB samples).
  - Cost per sign evaluation (network calls and measured seconds) vs the recursive alternative (exact chain).
  - RA numbers are reported as secondary (thresholds selected there).

## Pass / fail (pre-registered)
Delta_K1 = <H>(s_net + hop) - <H>(K1), Delta_V = <H>(s_net + hop) - <H>(s_V), per site, on RB.
- **Arm V PASSES** if (i) Delta_K1 + 2 SE < 1e-5 (matches or beats the exact one-hop Krylov sign within 1e-5/site)
  and (ii) Delta_V < 2 SE (not significantly worse than the ViT sign).
- **Arm M PASSES** if Delta_K1 + 2 SE < 1e-5 (with K1 = K1_M). Also reported: the fraction of the gap
  <H>(M) - <H>(s_V) closed by s_net^M + hop and by K1_M.
- Power requirement: SE(Delta_K1) <= 3e-6/site. If larger, the arm is "inconclusive (cause: referee SE)" unless the
  budget allows more RB chains (decision on SE only, never on the value).
- s_net alone is reported but does not enter the criterion.
- Otherwise "inconclusive (cause: X)", diagnosed from held-out label error, error location (|r - T| margins,
  amplitude decades), training curves. "Failed" only with a conceptual limit.

## Budget
<= 12 GPU-h (ws1 A40 slices, jax 0.8.2 venv; Paderborn GPU partition drained on 2026-10-12). Estimated: TR 2.3 h,
RA 1.7 h, RB 3.4 h, training 2 x ~1 h, net evaluation on the two-hop sets ~1 h.

## Files
- `s8_thresh.py`: step-1 threshold analysis on the existing loop_8x8 referee data (CPU).
- `s8_twohop.py`: two-hop data (GPU).  `s8_post.py`: labels (train) and per-sample/edge quantities (ref).
- `s8_q.py`: numpy helpers (Marshall, r_b, local energies, E(T) curves).  `s8_net.py`: sign net.
- `s8_train.py`: training.  `s8_energy.py`: thresholds, energies, disagreement, table.  `s8_figure.py`: figure.
- `s8.sbatch`: job script. Run data: ws1 `/project/theorie/a/A.Otaifi/chatty_signs8/runs/`.
