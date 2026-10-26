# Test L8 (one loop round at 8x8 with walkers): status 2026-10-25 (builder worker), READY TO SUBMIT, nothing in production submitted

## Built (all under experiments/gfmc_fn_lattice/)
- `gfmc_fn_lib.GFMC.keep` + `gfmc_run.py --dump`: per block OUT/dump/blk_NNNNN.npz (pre-resampling X int8, logw, EL); RNG stream unchanged.
- `l8/l8_lib.py`, `l8/l8_refit.py` (stage 2): R(theta) = Q^2 Ninv, Q = E_pi(v/g)/E_pi(v0/g) from walkers (per-block normalised weights),
  Ninv = <v0|v0>/<v|v> from VMC of psi_theta; SR ascent on R; held-out = last 20% of blocks per chain; exit 3 weight-health STOP, 4 no rise.
  SR details changed from the STATUS design after the 4x4 tests (both measured): (a) a metric from the VMC samples alone left the walker
  directions uncontrolled (rank <= #samples): one step blew R up by 1e24 at 4x4 and to 1e-4 at 8x8 (local test). Now S = (S_v + S_w)/2 with S_w
  from 1024 walkers resampled by |w~|, diagonal floor 1e-2 x median, shift 0.1, |d|_S <= 0.02 and a realised trust region (centred RMS of
  Delta log|v| <= 0.04, else halve). (b) The VMC term of the gradient was the larger noise source (cos with the exact gradient 0.55 from one
  sweep): 8 VMC sweeps of 1024 chains per step.
- `l8/l8_vmc_prefix.py` (control i, stage-3 half), `l8/l8_check4x4.py` (controls ii-iv + i at 4x4), `l8/l8_decide.py` (decision gate),
  `l8/pilot_l8.sbatch` + `l8/submit_pilot.sh`, `l8/launch_l8.sh` (production D -> F -> V -> G, afterok chain, watchers, ANALYSIS_BRIEF.md per run dir).
- `m13b_8x8/watch_gfmc8.sh`: optional GF_DONE_PAT (verdict line; default GFMC_DONE unchanged); test_watcher8.sh all branches PASS again.

## Controls (4x4: /scratch/a/A.Otaifi/l8/check4x4/run2.log, guide = check_general 30-iteration complex RBM+PP, F(v0) = 0.9885; L8_CHECK4X4 ALL PASS)
- (iv) gfmc_run.py --dump vs no dump (seed 7, 50 blocks): blocks.npz arrays (all but wall-clock sec) and final walkers + RNG state bitwise identical.
  Broken (dump drawing one extra random number per block): 39/40 blocks differ.  PASS.
- (ii) 2 chains x 300 blocks (M 256, dtau 0.1, production gfmc_run.py, RBMPPGuideG p=1, exact consts): estimate vs exact (z from bin jackknife):
  theta0 (a = v0): E_pi(a/g) 0.99160(26) vs 0.99197 (z -1.4); theta0+0.1 xi: E_pi 0.98424(40) vs 0.98487 (-1.6), Q 0.99258(29) vs 0.99284 (-0.9),
  R 0.99048(86) vs 0.99137 (-1.0); theta0+0.3 xi: E_pi 1.05016(67) vs 1.04995 (+0.3), Q 1.05906(64) vs 1.05845 (+0.9), R 0.9663(20) vs 0.9672 (-0.5).  PASS.
- (iii) broken sign a/|g|: estimator level R 0.9149 vs correct 0.9905 (theta0+0.1 xi), 0.6225 vs 0.9663 (+0.3 xi). Refit level (12 SR steps, same
  seeds): exact R of the result 1.00614 (correct) vs 0.94350 (broken); the broken run's own held-out "R" ran away to 619.  PASS.
  Caveat: held-out R estimate of the correct refit 1.0134(28) vs exact 1.0061 (z +2.6); a debug repeat gave 1.0101(29) vs 1.0062 (+1.3): the held-out
  estimate reads HIGH by ~1.5-2.5 sigma (adjacent held-out blocks are correlated with training blocks, tau_int ~1-2 blocks). The decision rests on E_FN.
- (i) zero-step refit: 4x4 params bitwise = params0, R = 1.0 exactly; broken (1 step): params differ, R 1.00226.  8x8 (pilot job 3922180):
  zero-step params_final arrays = rb8_prod; l8_vmc_prefix on them reproduces rb8/V BITWISE: stage-A Ec -31.916815001782673 (rel 0), B0 E_ref
  -31.919916101661535 (rel 0), per-group sums of the first 2 sweeps of seed 1 (rel 0, bitwise), consts from V's stored sums alpha -0.1754318714549749,
  E -31.915135844116154 (rel 0 vs consts.json).  Broken (f[0,1] x (1+1e-7)): Ec rel 1.26e-5 -> FAIL as it must.  PASS.

## 8x8 pilot (Paderborn 3922180, express, 9.2 min = 0.15 H100-h; remote /scratch/hpc-prf-ppdmrg/hpcalot/l8/runs/pilot_3922180, local /scratch/a/A.Otaifi/l8/pilot/3922180)
- Resume of a copy of rb8/p1_s1/ckpt.npz (block 1080) with --dump, 20 blocks: 13.1 s/block median (13.0 without dump), dump 21.9 kB/block (438 kB / 20).
- Weight health of w~ = walker weight x v/g on 5120 walkers: n_eff/n 0.998, largest share 4.1e-4 (limits 0.1, 1e-2); h < 0 fraction 2e-4, min|h| 0.45.
  The feared 1/h weights do not appear: at 8x8 the walkers never sit near h = 0.
- Refit pilot (8 steps, 16 train / 4 held-out blocks, ONE chain, 4 VMC sweeps per step; before the 8-sweep default): held-out R 1.0117(25), 1.0149(56),
  1.0131(85), 1.0414(120), 1.0267(140), 1.0157(180), 1.0172(250), 0.9945(260) at steps 1..8; train R rose to 1.43.  Held-out rose beyond its error at
  steps 1-4 (4.7, 2.7, 1.5, 3.4 sigma) and then fell back: overfitting of 6176 parameters to 4096 training walkers.  2.7 s per SR step.

## Production (launch_l8.sh, QOS cont default, DRY by default; GO=1 submits D_s1, D_s2 -> F -> V -> g1/p1_s1, g1/p1_s2 with afterok)
- D: 400 blocks per chain, 1.46 h per chain against 3 h.  F: 20 steps (fixed budget), 2 x 320 training blocks (40x the pilot), <= 0.7 h against 3 h
  (upper bound: walker pass scaled linearly).  V: 0.95 h against 3 h (rb8/V).  G: 1080 blocks (930 + 150 burn) per chain, 3.9 h against 8 h.
  Total ~12.5 H100-h of 15.
- Risk (measured, not resolved): the pilot overfit within 4-5 steps on 16 blocks; production has 40x the data, but whether 20 steps stay on the rising
  part is unknown. F saves params every 2 steps and the curve is in refit.json; its gate (held-out R > 1 by 3 sigma at the end) blocks stage 3 otherwise.
  The lead may prefer fewer steps (STEPS_F) before submitting; the gate's held-out R reads ~2 sigma high (4x4 caveat above).
- Decision after G: python3 l8/l8_decide.py /scratch/a/A.Otaifi/l8/prod/g1 (L8_VERDICT GAIN | NO_GAIN | LOSS | UNDECIDED | NOT_QUOTABLE vs gfmc_m13b/prod/rb8).
- Not checked: the production F step time with 164k walkers (bounded, not measured); l8_decide.py runs only on real chains (syntax-checked only).

## Revision 2026-10-25 22:10 (lead review): chain-split held-out, early stopping, kill-on-invalid-dep, g_0 = -0.498946 +- 4.6e-6
- l8_refit.py: --holdout_chain 1 (default; train on the first chain, hold out the whole second; --holdout_chain -1 = old split), params saved every
  step, params_final.npz = step with the largest held-out R (--select best; refit.json final.selected_step), held-out R there re-evaluated with a fresh
  VMC set and gated (exit codes unchanged).  launch_l8.sh: V and G carry --kill-on-invalid-dep=yes (flag accepted by sbatch --test-only on Paderborn).
  l8_decide.py: g_0 asserted = -0.498946 with error max(inverse-variance, quotable) = 4.6e-6; thresholds GAIN < -3, |z| <= 2 NO_GAIN, LOSS > +3 sigma.
- 4x4 rerun (/scratch/a/A.Otaifi/l8/check4x4/run3.log, 20 steps, train c1 / held-out c2; ALL PASS): selected step 16 of 20 (curve turned down after it);
  held-out R re-evaluated there 1.0068(20) vs exact 1.00450: z +1.15.  The in-curve value at the selected step was 1.0126(36): picking the maximum of the
  noisy curve reads ~+2 sigma high; the re-evaluation removes most of it.  Exact R per step: rises to 1.0050 at step 5, plateau 1.0039-1.0048 to step 16,
  1.0027-1.0033 at steps 17-20.  Broken a/|g|: exact R 0.8807.

## F 3923271 NO_RISE and diagnosis 2026-10-26 (l8_diag.py, Paderborn 3973532 express, part (a) only, ~2 min)
- F: chain-split held-out R fell from step 1 (0.9984(7) vs train 1.0041(7)); early stopping selected step 0; V/G killed by Slurm.
- Cross-chain S-metric cosine of the per-chain natural gradients at v_0 (disjoint VMC halves, jackknife over 20 block groups, 400 blocks/chain):
  shift 0.1: +0.033(29) (+1.2 sigma); shift 1: -0.006(41); shift 10: -0.029(99).  Same-chain halves (chain 1): +0.025(26), +0.009(35), -0.028(96).
  Predicted held-out change g2.d1 (lr 1): +3.4(1.4)e-2 at shift 0.1, +4.7(6.9)e-3 at 1, -1.7(250)e-5 at 10.
- 4x4 positive control (run3 dumps, 250 blocks/chain, /scratch/a/A.Otaifi/l8/diag4x4): +0.351(87) (+4.0 sigma) at shift 0.1, +0.458(137) at 1, +0.59(27) at 10.
- Branch: DATA-STARVED (cosine within 2 sigma of 0 at every shift); (b) not run.  Data factor for a 3-sigma cosine at shift 0.1: (3/1.16)^2 = 6.6x
  (noise ~ 1/sqrt(blocks)) -> 2660 blocks per chain, +2260 per chain = 16.4 H100-h more dump at 13.1 s/block; if instead cos grows linearly with data
  (noise-dominated regime), 2.6x -> 1030 per chain, +4.6 H100-h.  Shifts 1 and 10 give no signal to extrapolate from.
- Rerun with 8x the VMC samples (Paderborn 3995351, express, 1:49 elapsed, MaxRSS 20.8 GB; 128 sweeps x 1024 chains = 131k samples):
  cross-chain cosine -0.010(35) / +0.007(44) / -0.029(123) and g2.d1 -2.0(5.2)e-3 / -2.0(25)e-4 / -6(830)e-6 for shift 0.1 / 1 / 10;
  same-chain halves +0.021(37) / +0.033(38) / -0.037(102).  Pre-registered: g2.d1 within 2 sigma of 0 -> UNRESOLVED; L8 closes.
  (The first diag's g2.d1 +3.4(1.4)e-2 at shift 0.1 did not survive more VMC samples.)
