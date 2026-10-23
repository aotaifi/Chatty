# ANALYSIS_BRIEF (template for every production chain of M1.1; launch_m11.sh writes a filled copy next to each chain)
Question: does continuous-time lattice fixed-node GFMC, guide = RBM+PP A (rpT4_sr2_17164867/params_final.npz) + one signed Lanczos
hop (alpha = -0.18337958, E = -0.503762233419 per site), reproduce the exact E_FN = 5.5738332e-06 per site above
E0 = -0.503809653891 within 2 sigma, with the resampling-interval (dtau) and population (1/M) biases measured, two seeds?
Files per chain (Paderborn chatty_gfmc/runs/<name>, pulled to /scratch/a/A.Otaifi/gfmc_m11/prod/<name>): progress.txt (last line GFMC_DONE),
blocks.npz (emix, egrow, ess, elvar, hops, sec, evals per block), result.json. ckpt.npz stays on Paderborn (resume).
Gate: python3 gfmc6_analyze.py /scratch/a/A.Otaifi/gfmc_m11/prod  -> per-chain blocking error, tau_int, drift in fifths,
seed difference at the reference point, dtau and 1/M slopes, FINAL line PASS/FAIL (2 sigma).  Statistics follow the qmc-statistics
rules: seeds are independent chains (own Metropolis init and RNG, nothing shared); run tools/quotable.py on the two reference chains before quoting a number.
Next: FINAL PASS -> report to the lead (M1.1 done; M1.2: cost per walker step at 8x8/14x14).  FAIL, or seeds differ > 3 sigma, or drift in
fifths > 3 sigma -> stop, do not extrapolate; extend the chains (RESUME=1) and report the slopes.  GFMC_DONE with completed False just means
--maxsec was hit: relaunch with RESUME=1.
Verified before production: 4x4 check (check_4x4.py), exact 6x6 guide check (verify job 3825550: alpha, <H>, amplitude 1.8e-10, exact E_FN).
