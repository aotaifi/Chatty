# Stored Krylov signs at 8x8: composite sign net + one exact hop (J2/J1 = 0.5)

Pre-registration: `experiments/signs_8x8/README.md` (commit f516c94, amendment 1 in 7835f48, both before any label,
threshold or energy from the new data). Code: `experiments/signs_8x8/`. Run data: ws1
`/project/theorie/a/A.Otaifi/chatty_signs8/` (two-hop chunks ~0.5 GB per run; `ana/` holds the processed arrays).

## Verdict
- **Arm M (from the Marshall sign): PASS.** The stored composite net plus one exact hop beats the exact one-hop
  Krylov sign by **-5.2(1.1)e-4 per site** (bound -3.1e-4 < +1e-5). It closes 98.5% of the Marshall -> ViT-sign
  energy gap; the exact one-hop step closes 88%. On the evaluation samples it reproduces the ViT sign on every x.
  The machinery works at 8x8 where the signs matter.
- **Arm V (from the ViT sign): inconclusive (cause: referee SE).** (net + hop) - K1 = **+3.5(3.6)e-6 per site**,
  so the 2-sigma bound is 1.08e-5, just above the 1e-5 pass line. The SE (3.6e-6) misses the pre-registered
  3e-6 power requirement. Criterion (ii) holds: (net + hop) - ViT sign = -2.8(3.3)e-5.
  - The paired difference is heavy-tailed. One of the 6144 samples has a high local energy (r_V(x) = -16.8 > T1),
    so K1 flips x itself. That sample alone carries -0.186 per site for K1 - ViT and +0.016 for (net + hop) - K1.
  - Without it (diagnostic, not pre-registered): (net + hop) - K1 = +0.9(2.5)e-6, and K1 - ViT = -1.2(4.6)e-6.
  - At the ViT sign, one Krylov step is a null step at 8x8: also on the 15360 old referee samples (cross-fitted
    +2.1(2.6)e-6) and on RA. Arm V therefore tests only that the net + hop does no harm. It does no measurable harm.
  - Resolving 1e-5 against this tail needs either more samples (SE ~ 1/sqrt(n)) or a tail-aware referee.
- Not a conceptual limit: the stored nets are data-limited (2048 tempered samples; held-out error flat, training
  error 4x lower). The one exact hop repairs this, as at 6x6 and N <= 36.

## Table (RB: 6144 samples x ~ |psi_P|^2, 2048 independent chains; exact local energies over all ~145 neighbours)
Paired differences per site, with SE from per-chain means. Thresholds: T1 = -20.059 (step 1, old referee data),
T_M = -27.615, T2(net V) = -17.81, T2(net M) = -15.31 (energy-optimal on RA, label-free).

| sign s (amplitude a = projected ViT) | E/N | vs ViT sign | vs exact K1 of its arm |
|---|---|---|---|
| ViT sign (binarised) | -0.498785(52) | 0 | |
| **arm V** exact one-hop Krylov K1_V | -0.498816(37) | -3.1(3.1)e-5 | 0 |
| stored net s_net^V | -0.498799(39) | -1.5(2.3)e-5 | +1.67(1.01)e-5 |
| **s_net^V + one exact hop** | -0.498813(38) | -2.8(3.3)e-5 | **+3.5(3.6)e-6** |
| **arm M** Marshall | -0.493748(238) | +5.04(0.24)e-3 | |
| exact one-hop Krylov K1_M | -0.498186(129) | +5.98(1.18)e-4 | 0 |
| stored net s_net^M | -0.498012(201) | +7.72(1.95)e-4 | +1.74(1.39)e-4 |
| **s_net^M + one exact hop** | -0.498711(71) | +7.4(3.0)e-5 | **-5.24(1.06)e-4** |

- RA (1536 samples, where the thresholds were chosen; secondary) gives the same picture:
  (net + hop) - K1 = -5.5(3.5)e-5 (V) and -4.8(1.9)e-4 (M).
- Absolute E/N carries the guide's sampling noise (5e-5). Compare via the paired columns. The ViT-sign E/N here,
  -0.498785(52), agrees with loop_8x8's -0.498856(29) at 1.2 sigma.

## Disagreement with the exact one-hop Krylov sign
Held-out tempered states (beta = 1/2, 20% of chains: 410 samples + 58.5k neighbours). The "no-flip" column is the
error of simply keeping the base sign.

| arm | samples x: net vs K1 (no-flip) | neighbours: net vs K1 (no-flip) | training error |
|---|---|---|---|
| V | 1.2% (1.5%) | 3.5% (3.9%) | 0.9% |
| M | 3.9% (25.6%) | 6.7% (29.9%) | 3.7% |

- On a^2 samples (RB x): net vs K1 disagree on 0/6144 samples (V) and 4/6144 (M).
- The errors sit in the low-amplitude tail. In the lowest 10% of log a the error is 19% (V) and 23% (M); it is
  < 0.2% in the top half.
- The errors also sit near the threshold: 35-47% for |r - T| < 1.

## Cost per sign evaluation (measured)
| sign | network calls per configuration x | seconds per x |
|---|---|---|
| stored net s_net | 16 CNN passes (D4 x flip average; 186k parameters) | 5.6e-5 (1.8e4 configurations/s, RTX 2080 Ti) |
| s_net + one exact hop | a and s_net on {x} u N1(x): 146 projected ViT evaluations + 146 x 16 CNN passes | ~0.04 (ViT 4.5e3/s on 2080 Ti) |
| exact recursive K1 (one step) | 146 projected ViT evaluations | ~0.03 |
| exact recursive K2 (what net + hop replaces) | ~1.0e4 (two-hop set, measured 10119 per sample) | ~2.2 |
| exact recursive K3 | ~3-4e5 (three-hop set of random configurations) | ~70-90 |

- A projected ViT evaluation is 16 ViT passes. The local energy of a sign needs the sign on N1(x) as well.
  - For s_net + hop that means a and s_net on the two-hop set (~1e4 per sample).
  - For the recursive chain of the same depth it means the three-hop set (~4e5).
- So the stored net makes each Krylov step cost a fixed one hop instead of a growing recursion. This is ~50x per
  sign already at depth 2, and the gap grows geometrically with depth.

## How it plugs into the loop
- The planned amplitude target is a clipped Lanczos factor 1 + alpha (E_L - E). Its sign,
  sgn(1 + alpha (E_L - E)), is exactly a Krylov threshold step on r = E_L with T = E - 1/alpha (alpha < 0).
- So the same machinery applies:
  - store the composite s * sgn(1 + alpha (E_L - E)) relative to Marshall in a sign net, trained on tempered
    samples + all neighbours;
  - evaluate it with one exact hop, which re-applies the next threshold step at fixed cost.
- Arm M shows this works when the step changes the energy (from a poor sign). Arm V shows it is harmless, within
  the resolution, when the step is a null step (near the ViT).

## Method checks
- **Labels.** Labels and energies come from the guide on the full two-hop set: no bond subsampling.
  - Check: E_L recomputed from the edge arrays equals r_V(x) to 4e-14.
  - All targets are D4 x flip invariant; the net is averaged over the 16 images.
- **Training.** AdamW, 6e4 steps, batch 1024, warmup-cosine 2e-3. BCE is still falling at the end
  (V 0.051, M 0.114). Held-out disagreement is flat over the last 2-3e4 steps (V 0.0365, M 0.068).
  This is a generalisation limit at 2048 samples (training error 0.9%/3.7%), not an optimisation limit.
  More tempered samples, not more steps, is the lever. The 6x6 study used 4e4 samples.
- **Thresholds.**
  - T_M is stable under RA half splits (-27.6 ... -29.2) and T2(net M) is stable too (-12.9 ... -16.8).
  - T1 and T2(net V) are unstable (-17 ... -66 and -18 ... -53), because the energy curve is flat (null step).
  - T1 re-chosen on RB would be -22.7.
- **Referee.** 2 x 1024 chains x 3 samples (burn 40, thin 2 sweeps). The SE comes from per-chain means over 2048
  independent chains. The thresholds were chosen on independent RA data, so the RB numbers are free of selection
  bias.
- **Implementation.**
  - Flax nn.Conv with XLA autotuning OFF (the flag inherited from loop_8x8) runs 9x slower: 134 vs 14 ms per step
    on an a40-8gb slice (`bench_*.json`).
  - Arm V trained with it off (2.1 h); arm M and all evaluations ran with it on. The function is the same, only
    the speed differs.
  - A gather-matmul forward was checked identical to 6e-6.

## Cost (sacct)
| stage | device | device-h |
|---|---|---|
| TR (2048 tempered samples, two-hop) | a40-16gb slice | 2.75 |
| RA (1536) | a40-16gb slice | 2.48 |
| RB1 / RB2 (3072 each) | RTX 2080 Ti / a40-24gb slice | 2.10 / 1.83 |
| training V / M (+ one cancelled and two OOM-failed M attempts) | 2080 Ti | 2.15 / 0.24 (+0.31) |
| nets on the two-hop sets (RA, RB1, RB2) | 2080 Ti | 0.49 + 0.98 + 1.03 |
| smoke tests, benchmarks | various | 0.05 |
| **total** | | **14.4 device-h** (over the 12 GPU-h budget). In A40 equivalents (16gb = 1/3, 24gb = 1/2) ~10. |

The overrun has two causes: a40-16gb slices run at half the 24gb throughput (2.1e3 projected ViT evaluations/s),
and arm V was trained with autotuning off.

## Files
- `energy_RB_RA.json`: all energies, paired differences, disagreements, thresholds, half splits.
- `threshold_TM_RA.json`: the Marshall-step threshold.
- `step1_threshold_existing_referee.json`: step 1 on the old referee data.
- `train_V.json`, `train_M.json`: training curves, errors by margin and by amplitude.
- `bench_*.json`, `twohop_meta_*.json`.
- `fig_signs_8x8.{png,pdf}`:
  - (a) arm V paired differences, with the +1e-5 pass bound;
  - (b) arm M energies above the ViT sign;
  - (c) held-out disagreement with K1 during training (dotted: no-flip predictor).
