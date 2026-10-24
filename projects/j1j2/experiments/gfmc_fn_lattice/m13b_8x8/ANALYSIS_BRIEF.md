# M1.3b analysis brief (template; launch_m13b.sh writes a filled copy per job): 8x8 J1-J2 (J2/J1 = 0.5, PBC) fixed-node GFMC, translations x spin-flip RBM+PP guide

Question: the fixed-node energy E_FN per site at 8x8 for the guide p = 0 (pure Re[w psi_P]) and p = 1 (plus one signed Lanczos hop), from two independent
chains each (own init walkers and RNG), against (a) the VMC <H> of the SAME guides (p = 0: E; p = 1: E + alpha var, Sorella style; vmc_moments.py,
stage V) and (b) the references -0.498886 (published RBM+PP, full projection), -0.49906 (published, extrapolated), -0.4992 (DMRG). Cost: GPU-h per number
(sum of block seconds of the chains; V's seconds for the VMC numbers).  Parameters: params_final.npz of the 8x8 translations training (job 3831844,
run rb8_prod), given to launch_m13b.sh as an argument.
Run order: launch_m13b.sh V PARAMS  -> (afterok) A, B, X.  Gate before A/B: V's vmc_s1/vmc_s2 agree (quotable.py series E0_s1.npy E0_s2.npy; same for E1),
check_h2 within 2 jackknife errors of 1, convergence block of vmc_pooled.json (E1 from 1/4, 1/2, all sweeps) stable to its error.
Gate for GFMC: python3 gfmc8_analyze.py ROOT (ROOT = /scratch/a/A.Otaifi/gfmc_m13b/prod/<TAG>): per chain blocking error, tau_int, drift over fifths, the two
chains' difference, quotable.py verdict on the pair, table E_FN vs VMC vs references with GPU-h.
Next per verdict: QUOTABLE, chain difference < 3 sigma, drift in fifths < 3 sigma -> report E_FN(p), E_VMC(p) - E_FN(p), gap to the three references, GPU-h.
Drift or chain difference > 3 sigma -> relaxation or trapping: continue (RESUME=1) before any number. NOT QUOTABLE -> report the failed check, no number.
Bias: stage X chains (p = 0: dtau 0.2, 0.4, M 256, 4096; p = 1: M 64) against the reference point (A chains: M 1024 / 256, dtau 0.1) before quoting an extrapolation.
Known caveats: (1) VMC moments mu3, mu4 (hence var_1, alpha) are heavy-tailed (rare configurations carry up to half of mu4 in a 4x4 test) and biased low at
~30k samples; E and E_1 are robust (4x4 exact-distribution test: bias +0.6 and +1.3 sigma of the mean). The convergence block shows the trend. (2) The old gfmc6_run
RBMPPGuide assumes psi_P real; gfmc_run uses RBMPPGuideG (phase kept), tested at 4x4 with complex RBM and at 6x6 (identical rows).
Pilot numbers (pilot parameters, single short chains): see launch_m13b.sh SIZING block.
