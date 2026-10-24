# M1.3 analysis brief: 8x8 RBM+PP VMC, translations x spin flip (128 images), J2/J1 = 0.5, PBC

Run directory (local copy, filled by the watcher): /scratch/a/A.Otaifi/rbmpp8/rb8_prod  (remote: Paderborn /scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/rb8_prod).

Question: the 8x8 RBM+PP (Nomura-Imada 2005.14142, M = 16 complex RBM, real f, full N x N f) trained from scratch by SR VMC with only
translations x spin-flip projection: what is its VMC energy per site, and how far is it from the published full-projection RBM+PP value
-0.498886(1)? (The difference is the accuracy cost of dropping the point group, which is what makes walkers affordable: README
compress_6x6 "Results: M1.1 + M1.2".) Reference at 6x6, same parameters: translations -0.502023(6) vs simple (144 images) -0.503727(5) vs
full -0.503758(3), exact -0.50380965 (A params rpT4_sr2_17164867; runs ctl1_trans6_long, ctl1_simple6, rpT4eval_17168763), i.e. the
point group costs 1.7e-3 at 6x6; the 8x8 shortfall against -0.498886 should be read against that.

Files (per run directory):
- train.json  hist[it] = {E (per site), err, var, acc, dnorm, lr, shift, t_step, wall}; 9000 steps; params_XXXXX.npz every 100 steps, params_final.npz.
- eval/E_s31.npy, eval/E_s32.npy: two independent evaluation chains (different seeds, fresh random starts, 100 thermalisation sweeps,
  own RNG), one entry per sweep = mean over 1024 chains of Re E_loc / N, in time order; eval/eval.json (chain-mean errors, Var/N, acceptance).
- run_metadata.txt (code sha256), VERDICT_RUN (last line "DONE_RBMPP8 exit=0 ..."), rb8prod-<job>.out (PROGRESS lines).

Gate: (1) training convergence, judged on the energy-vs-step curve, not on a rule: block-average train.json E in blocks of 500 steps over the
last 3000 steps, with the standard error of each block mean from the 8192-chain error of each step (err) and from the scatter between
steps; report the blocks and the slope over the last 3000 steps; lr decays from 0.05 to 0.005 by schedule, so a late drift means the cosine
tail was too short, not that the run failed. (2) the final number:
  python3 /project/theorie/a/A.Otaifi/tools/quotable.py series eval/E_s31.npy eval/E_s32.npy --name "E/N 8x8 RBM+PP translations" --plot analysis.png
Quote only the verdict line it prints. The final energy is the eval chains' value, not the training trace (training E is the biased
mean of the SR iterates and uses 8192 chains of one run).

Next step per verdict (the lead decides; the worker reports):
- QUOTABLE and the last-3000-step blocks agree within 3 sigma (no trend) -> report E, the gap to -0.498886 (E - (-0.498886), per site and
  as a fraction of E0 relative to the 6x6 point-group cost), and the rms Var(E_loc)/N; M1.3b (the translation-projected guide in the 8x8
  FN-GFMC walker) can use params_final.npz.
- QUOTABLE but the energy still drifts over the last 3000 steps beyond 3 sigma -> not converged: report the trend; the continuation is
  "same RUNNAME, steps raised, resume true" (the schedule is a function of the step, so raising steps moves lr; say so), after the lead decides.
- NOT QUOTABLE (chains disagree / drift in the eval series / too few effective samples) -> report which check failed; do not quote a value;
  the plot is the deliverable. Chains agreeing within errors but gap to -0.498886 > 5e-3 -> the translations-only ansatz is much worse
  than the full projection at 8x8; report, do not tune.
- STALLED / FAILED / TIMEOUT / DISAPPEARED -> read the .out and run_metadata.txt; a TIMEOUT resumes with the same RUNNAME (state.npz holds
  parameters, chain configurations and RNG key every 100 steps; resume is bit-identical, tested at L=4 on a local GPU).
