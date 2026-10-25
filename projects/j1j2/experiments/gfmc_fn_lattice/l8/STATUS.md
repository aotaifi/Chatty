# Test L8 (one loop round at 8x8 with walkers): worker status 2026-10-25, stopped at migration

Built: nothing yet (no code, no launcher). No job submitted; no running job ids.
Controls passed: none yet (none run).

Findings from reading (verified on disk / Paderborn, 2026-10-25):
- M1.3b p=1 chains did NOT save configurations: paderborn:/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8/runs/rb8/p1_s{1,2}/ hold
  only ckpt.npz (final walkers + RNG), blocks.npz (energies), progress.txt. So stage 1 needs walkers; cheapest route: resume
  p1_s1/p1_s2 from ckpt.npz (already equilibrated, no burn) with a new --dump option. g_0 consts: paderborn:.../gfmc8/runs/rb8/V/consts.json
  (alpha -0.17543187, E -31.91513584, w = -1).
- Note: the run family on Paderborn is `rb8`, not `rb8_prod` (gfmc8/runs/rb8/...); params are rbmpp8/runs/rb8_prod/params_final.npz.

Design worked out (not yet coded):
1. Walker dump: GFMC.block keeps pre-resampling X and logw (attribute only, RNG stream unchanged -> regression check: bitwise identical
   blocks.npz on a 4x4 chain old vs new); gfmc_run.py --dump writes per block X (int8) and logw. Guide values at X (Re log psi_P0,
   c_g = Re[w e^{i Im l0}(1+alpha(E_L0-E))]) computed post hoc in the refit job.
2. Fidelity with v = Re[w psi_P^theta]: phi_0 norm is constant, so maximise R(theta) = F(v)/F(v_0) =
   [E_pi(v/g)/E_pi(v_0/g)]^2 * E_{|v|^2}[(v_0/v)^2], pi = g_0 phi_0 (walker weights), v_0 = rb8_prod. R(v_0) = 1.
   Gradient: grad log F = 2( sum_pi w~ (D - D_bar_v) ), w~ = self-normalised walker weights x v/g, D = grad log|v|, D_bar_v = VMC mean of D
   (centred / control-variate form). SR: S = Cov_{|v|^2}(D) as in rbmpp_train.sr_solve (~6176 params, dense Cholesky).
   Held-out: last 20% of blocks of each chain; R on held-out set must rise.
3. Stage 3: vmc_moments.py on the new params (alpha by Ritz) + 2 p1 chains, M 256, dtau 0.1 as M1.3b.
Sizing estimate (not measured): stage 1 ~2.5 H100-h (2 x 300 blocks), refit ~1-1.5, V 1.1, g_1 chains ~7: ~12 of 15 H100-h.
4x4 control plan: check_general.py machinery (exact basis, fn_exact) + TableGuide walkers with dump; compare E_pi(a/g) to exact
<a|phi>/<g|phi>; broken: weights a/|g| must lower R.
Remaining: everything in the brief (code, controls i-iii, 8x8 pilot, launchers, watchers, ANALYSIS_BRIEF.md).
Queue note (from M1 LxL staging): cont start estimate was 2026-11-07; re-check with sbatch --test-only.
