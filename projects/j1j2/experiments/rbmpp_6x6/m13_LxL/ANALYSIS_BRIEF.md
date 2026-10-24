# M1 analysis brief (template, @L@ filled by launch_rbmpp_L.sh): @L@x@L@ RBM+PP VMC, translations x spin flip (2N images), J2/J1 = 0.5, PBC

Run directory (local copy, filled by the watcher): /scratch/a/A.Otaifi/rbmppL/prod/<RUNNAME>  (remote: Paderborn /scratch/hpc-prf-ppdmrg/hpcalot/rbmppL/runs/<RUNNAME>).

Question: the @L@x@L@ RBM+PP (Nomura-Imada arXiv:2005.14142, M = 16 complex RBM, real f, full N x N f) trained from scratch by SR VMC with only
translations x spin-flip projection (same recipe as the 8x8 run rb8_prod, README compress_6x6 "Results: M1.3a"): what is its VMC energy per site,
and how far is it from the published RBM+PP? The 8x8 reference of this recipe: -0.498671(3) against the published full-projection -0.498886(1), gap 2.15e-4.
References per site (arXiv ids): L = 10: RBM+PP -0.497629(1) (2005.14142, full projection), ViT -0.497634(1) (Rende et al. 2310.05715, Table I), record
ResNet2+MinSR -0.4976921(4) and ZVE -0.497715(9) (Chen-Heyl 2302.01941), Hu p = 0 / 1 / 2 -0.49521 / -0.49718 / -0.49755 and variance-extrapolated -0.49781(2)
(Hu et al. 1304.2630, Tables I-IV). L = 12: RBM+PP -0.496791(4) (2005.14142, as quoted in Table I of Roth et al. 2211.07749), GCNN -0.496769(9) (Roth et al.
2211.07749); no Hu, no DMRG-torus and no tabulated transformer / ResNet number exists for 12x12 (Chen-Heyl and Rende 2606.02794 computed it but give V-scores / figures only).

Files (per run directory):
- train.json  hist[it] = {E (per site), err, var, acc, dnorm, lr, shift, t_step, wall}; params_XXXXX.npz every 100 steps, params_final.npz.
- eval/E_s31.npy, eval/E_s32.npy: two independent evaluation chains (different seeds, fresh random starts, 100 thermalisation sweeps, own RNG), one entry per sweep =
  mean over the chains of Re E_loc / N, in time order; eval/eval.json (chain-mean errors, Var/N, acceptance).
- run_metadata.txt (code sha256), VERDICT_RUN (last line "DONE_RBMPP8 exit=0 ..."), rbLprod-<job>.out (PROGRESS lines).

Gate: (1) training convergence on the energy-vs-step curve: block-average train.json E in blocks of 500 steps over the last 3000 steps with the standard error of each block
mean from the per-step err and from the scatter between steps; report the blocks and the slope; lr decays 0.05 -> 0.005 (cosine), so a late drift means the tail was too
short, not that the run failed. No early stop on noise. (2) the final number:
  python3 /project/theorie/a/A.Otaifi/tools/quotable.py series eval/E_s31.npy eval/E_s32.npy --name "E/N @L@x@L@ RBM+PP translations" --plot analysis.png
Quote only the verdict line. The final energy is the eval chains' value, not the training trace.

Next step per verdict (the lead decides; the worker reports):
- QUOTABLE and the last-3000-step blocks agree within 3 sigma -> report E, the gap to the RBM+PP reference above (E - ref per site), rms Var(E_loc)/N; the GFMC / VMC-moments stage
  (gfmc_fn_lattice/m13b_LxL/launch_m13b_L.sh) can use params_final.npz.
- QUOTABLE but still drifting over the last 3000 steps beyond 3 sigma -> not converged: report the trend; continuation = same RUNNAME, steps raised, resume true (the schedule is a
  function of the step: raising steps moves lr; say so), after the lead decides.
- NOT QUOTABLE -> report which check failed; no value; the plot is the deliverable.
- STALLED / FAILED / TIMEOUT / DISAPPEARED -> read the .out and run_metadata.txt; a TIMEOUT resumes with the same RUNNAME (state.npz holds parameters, chain configurations and RNG key every
  100 steps; resume is bit-identical, tested at L = 4 on a local GPU). Memory: the Jacobian chunk is 256 (L = 10) / 128 (L = 12); an out-of-memory error means lower "chunk" in the spec.
