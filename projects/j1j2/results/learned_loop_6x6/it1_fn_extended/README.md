# Learned loop 6x6, iteration 1 guide (state_it1b.json): extended fixed-node run

64 FN populations of the it1b guide (J2/J1 = 0.5, 6x6), 16 from the original run and 48 new.
Same settings for all: M = 128 walkers, beta = 1.2, burn beta = 0.4, tau_max = 0.025, float32 ViT with full fp32 matmuls,
`ll6_fn.py` and `state_it1b.json` unchanged, seeds 42001-42016 (old) and 42017-42064 (new, all distinct).
All numbers are energy per site. E0 = -0.5038097, ViT (VMC) = -0.503654(21).

## Result (n = 64)
| estimator | E_FN/site | error | E_FN - ViT | sigma |
|---|---|---|---|---|
| mean | -0.503708 | 2.9e-5 (SE) | -5.4(3.6)e-5 | -1.5 |
| median | -0.503654 | 4.2e-5 (bootstrap) | -0.0(4.7)e-5 | 0.0 |
| 10% trimmed mean | -0.503681 | 2.7e-5 (bootstrap) | -2.7(3.4)e-5 | -0.8 |

- Old 16 only: -0.503667(63). New 48 only: -0.503721(33). Consistent.
- Population SD 2.3e-4, skew -1.44 (Shapiro p = 6e-5): the distribution has a long low tail.
- 17 of 64 populations (27%) lie below E0. Lowest -0.504643 (E0 - 8.3e-4), next -0.504349, -0.504177. The upper tail ends at -0.503323. Dropping the single lowest point moves the mean to -0.503693(26).
  Since the exact fixed-node energy is >= E0, these are finite-population (M = 128) fluctuations, and the below-E0 fraction shows the per-population noise (2.3e-4) is larger than the gap to E0 (1.6e-4). The mean could still carry a small population-control bias that this run cannot quantify. The robust estimators (median, trimmed mean) sit 2.7e-5 to 5.4e-5 above the mean.
- Batch means of 8 populations (seeds 42001.., 42009.., ...): -0.503672, -0.503662, -0.503671, -0.503737, -0.503721, -0.503851, -0.503578, -0.503770. The one RTX 2080 Ti batch (42049-42056, -0.503578) agrees with the A40 batches within the batch scatter.

## Verdict
Tie with the ViT. The best (mean) estimate is 1.5 sigma below it, and the median and trimmed mean are even closer; none reaches 2 sigma.
Caveat: the ViT number is its VMC energy. FN on the ViT's own guide (ViT signs) would be at or below the VMC value, and no paired FN run on the ViT guide exists here, so this is not a paired comparison. The only paired result is the earlier <H>_new - E_ViT = +0.4(3.5)e-5.

## Jobs and cost
New populations, `ll6_gpu.sbatch ll6_fn.py state_it1b.json it1bxK 128 1.2 0.4 0.025 <8 seeds>`, run dir on ws1 `~/chatty_ll6/runs/fn_it1bxK_<jobid>`:
| job | seeds | GPU | elapsed |
|---|---|---|---|
| 16843596 (x0) | 42017-42024 | A40 | 1:17:53 |
| 16843597 (x1) | 42025-42032 | A40 | 1:34:36 |
| 16843598 (x2) | 42033-42040 | A40 | 1:34:05 |
| 16843599 (x3) | 42041-42048 | A40 | 1:34:50 |
| 16844931 (x4) | 42049-42056 | RTX 2080 Ti | 2:07:05 |
| 16843601 (x5) | 42057-42064 | A40 | 1:35:47 |

GPU time: 7.6 A40-h plus 2.1 RTX 2080 Ti-h (9.7 GPU-h). Job 16843600 (x4, A40) was cancelled while pending and resubmitted on the 2080 Ti as 16844931.
Old populations: jobs 16818408 (it1ba) and 16818411 (it1bb), seeds 42001-42016.

## Files
- `summary.json`: all statistics and per-seed E_FN/site.
- `analyze.py`: reproduces summary.json (`python analyze.py ../runs`).
- Per-job raw output in `../runs/fn_it1bx*_<jobid>/`, logs in `../logs/fn_it1bx*`.
