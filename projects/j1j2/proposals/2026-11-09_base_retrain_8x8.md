# Proposal 2026-11-09: retrain the 8x8 base network (M1, RBM+PP) to published quality (plan + cost probe; no production run)
Gap: M1 VMC -0.498671(3) (rbmpp8/rb8_prod eval) vs Nomura-Imada -0.498886(1) (2005.14142 Table 1): 2.15e-4/site; Varbench -0.498963(2).

## M1 as run (Paderborn 3831844, /scratch/a/A.Otaifi/rbmpp8/rb8_prod; code b906f5f, rbmpp_lib 5fa08acc, rbmpp_train 475c175f)
Launcher experiments/rbmpp_6x6/m13_8x8/{launch_rbmpp8.sh,rbmpp8_prod.sbatch}, spec specs/train_prod4500.json (= rb8_prod/spec.json).
Full SR (dense S, Cholesky in diag-rescaled basis), 8192 samples/step, lr 0.05->0.005 cosine, relative shift 1e-3 -> 1e-2 (RISING),
max |d|_S 0.3, 4500 steps (halved from 9000 for the express budget), 3.04 s/step, 3.8 H100-h, one seed (101), from scratch.
Not plateaued: 500-step block means of train.json -0.4986609, -0.4986625, -0.4986667 (steps 3000-4500, SE 2.4e-6); linear
slope over the last 3000 steps -1.5e-8/step (-4.5e-5 in total). Gate 1 of its brief was never run (ANALYSIS.md: Bash denied).

## Ours vs published (8x8, J2 = 0.5)
| item | M1 (rb8_prod/spec.json, train.json) | Nomura-Imada 8x8 benchmark |
| projection | K=0 translations x spin parity, 128 images | K=0, spin parity, + simplified point group {E, R_pi/2}: 256 images (App. A "Calculation conditions", App. B) |
| total-spin projection | none | none for RBM+PP (only mVMC PP-only had it, App. A) |
| RBM | M=16 complex b, W (alpha 0.25), no visible bias | same: 16 hidden, complex (Sec. III.1, App. A) |
| pairing f | real, full 64x64, no sublattice | real, no sublattice for the benchmark (App. B; production runs used 4x4 sublattice) |
| optimiser | SR, uniform relative shift 1e-3 -> 1e-2 | SR, eps1 (relative) RBM 1e-2..1e2 -> 1e-5..1e-4, f +1e-2; eps2 (abs, x max S_mm) 1e-3 -> 1e-7..1e-6 (1709.06475 App. C.1, cited as Ref. 15) |
| init | b, W U[-0.05,0.05]; f U[-fmax,fmax], fmax ~ R^-2 | same (App. A) |
| seeds | one | >= 3 independent optimisations, lowest kept (App. A; 1709.06475 App. C.1) |
| samples/step, steps | 8192, 4500 | not stated in either paper |

## Ranked differences (what can explain 2.15e-4)
1. Point-group projection: MEASURED +1.0e-4. Same rb8_prod params, 'simple' projection, no retraining: -0.4987725(83) (probe a0,
   single chain, 20 x 8192 samples) vs -0.498671(3): half the gap at zero training cost. Cost x2 per step (images 256/128; measured 6.02 vs 3.04 s).
2. Unconverged optimisation + over-damped SR: the trace still fell 4.5e-5 over the last 3000 steps; our late shift (1e-2 on every
   parameter) is 100-1000x the published RBM eps1. Probe a (lr 0.02, shift 1e-2) moves |d|_S 0.004-0.005 per step and gains nothing
   resolvable in 100 steps (blocks of 20: -0.4987821(96) ... -0.4987895(82); slope +8(16)e-8/step). Expected remainder: the 1.1e-4 to -0.498886.
3. One seed vs best of >= 3: size unknown at 8x8 (not measured here).
4. Varbench -0.498963 vs paper -0.498886 (7.7e-5): consistent with a full (C4v, 1024-image) projection there (PLAN_pfaffian.md l.1316 says so;
   Varbench file not opened). Full projection = 4x 'simple' per step (inferred from linear image scaling; probe stage b did not finish
   thermalisation within 480 s, so not measured).

## Probe (Paderborn 4191128, express, gpu1011, 25.5 min = 0.43 H100-h, MaxRSS 2.9 GB; /scratch/a/A.Otaifi/rbmpp8/rb8_pgprobe)
Code = rb8_prod copies (sha above); specs experiments/rbmpp_6x6/m13_8x8_retrain/specs/; stages a0=0 a=0 b=124 (timeout).
'simple': 6.02 s/step at 8192 samples (thermalisation 3 s/sweep, ~half of each step). 'full': not measured (>480 s before step 1).

## Plan (needs the lead's choice of SR shift; the probe shows M1's late SR setting is too damped to move)
S0 (1.5 H100-h): warm start from rb8_prod with 'simple', lr 0.02, 300 steps each at shift 1e-2 / 1e-3 / 1e-4 (constant); pick the
   arm with the lowest last-100-step mean that stays finite. Prediction: 1e-3 or 1e-4 gains >= 2e-5 in 300 steps; if none gains
   >= 2e-5, the shift is not the limiting term and the lead steps back before S1.
S1 production: 'simple', the S0 shift with decay to 0.1x, lr 0.02 -> 0.005 cosine, 3000 steps (range 2000-6000: M1 needed 4500 from
   scratch and had not plateaued; the warm start removes the first ~500 steps), 8192 samples, ckpt every 100.
S2 reproducibility: a second seed from the same warm start (tests optimiser noise, NOT an independent basin: same start).
Eval: 2 independent chains x 1.6e6 samples with 'simple' and with 'full' (eval only), quotable.py series mode.
## Cost (H100-h; 6.02 s/step measured)
S0 1.5 | S1 5.0 (3.3-10) | S2 5.0 (3.3-10) | eval 'simple' ~0.6 + 'full' ~2.4 (inferred x4) | total ~14.5 (range 11-24).
Express left this month ~5.4 H100-h (1800 - 1452 - 26 min used, sacctmgr 2026-11-09): S1/S2 need cont/normal QoS or December express.
Downstream: 'simple' doubles FN walker cost (projection dominates; M1 p=1 FN was 7.9 H100-h -> ~16).
## Convergence evidence required
- Plateau: over the last 20% of steps, the 100-step block means show no slope beyond 2 sigma and drift < 1e-5/site.
- S1 and S2 final eval energies agree within 2 sigma (combined); both eval-chain pairs QUOTABLE.
- Target: E('simple') <= -0.498866 (published within 2e-5).
## Kill
No improvement > 5e-5 below the step-0 'simple' energy -0.4987725 (i.e. no 200-step mean below -0.4988225) within the first 20% of
S1 (600 steps, ~1 H100-h): stop, keep the projection-only gain (+1.0e-4, free), report to the lead.
