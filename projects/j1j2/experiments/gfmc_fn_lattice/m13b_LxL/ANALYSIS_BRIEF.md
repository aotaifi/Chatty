# M1 LxL analysis brief (L = 10, 12; template, launch_m13b_L.sh writes a filled copy per job): J1-J2 (J2/J1 = 0.5, PBC) fixed-node GFMC, translations x spin-flip RBM+PP guide

Same question and gates as ../m13b_8x8/ANALYSIS_BRIEF.md (read it; the 8x8 result is in README compress_6x6 "Results: M1.3b"), with the L = 10 / 12 references:
- L = 10 (per site): RBM+PP -0.497629(1) (Nomura-Imada arXiv:2005.14142), ViT -0.497634(1) (Rende et al. arXiv:2310.05715, Table I), record ResNet2+MinSR -0.4976921(4) and
  ZVE -0.497715(9) (Chen-Heyl arXiv:2302.01941), Hu VMC p = 0 / 1 / 2 -0.49521 / -0.49718 / -0.49755, variance-extrapolated -0.49781(2) (Hu et al. arXiv:1304.2630, Tables I-IV),
  DMRG torus (Gong et al. arXiv:1311.5962) is not converged at 10x10 (truncation error 8e-5 at 32000 states), extrapolation -0.4988 without error bar: not a reference.
- L = 12 (per site): RBM+PP -0.496791(4) (arXiv:2005.14142, quoted in Table I of Roth et al. arXiv:2211.07749), GCNN -0.496769(9) (Roth et al. arXiv:2211.07749, Table I).
  Chen-Heyl (2302.01941) and the scaling-law paper (Rende et al. arXiv:2606.02794) computed 12x12 but tabulate no energy; Hu 2013 and the DMRG torus paper have no 12x12 entry
  (results/literature/J1J2_BENCHMARKS_COST.md section 1.4, checked there against the PDFs for Roth Table I; the other entries are not re-read here).
Run order: [LSZ=L] launch_m13b_L.sh V PARAMS -> (afterok) A, B, X.  Gate before A/B: V's vmc_s1/vmc_s2 agree (quotable.py series E0_s1.npy E0_s2.npy; same for E1), check_h2 within 2 jackknife
errors of 1, ratio_vs_fastupdate_maxrel < 1e-8, convergence block of vmc_pooled.json stable.
Gate for GFMC: python3 gfmc_L_analyze.py ROOT --L L  (ROOT = /scratch/a/A.Otaifi/gfmc_m13b_L/L<L>/<TAG>): per-chain blocking error, tau_int, drift over fifths, chain difference,
quotable.py verdict on the pair, table E_FN vs VMC vs references with GPU-h per number. Next per verdict as in the 8x8 brief (QUOTABLE + chains within 3 sigma + no drift ->
report; drift or difference > 3 sigma -> continue with RESUME=1 before any number; NOT QUOTABLE -> report the failed check, no number).
Lesson of 8x8: the p = 1 chains were NOT QUOTABLE after 420 blocks (binning error still rising): size p = 1 for the quoted error with margin, and look at the binning curve before quoting.
Sigma_walker of the production guide is not known until the first progress lines: error scales with sigma_walker^2; if the first chain's blocks give a different sigma_walker than the sizing
assumed, rescale the remaining chains (cost x f^2).
