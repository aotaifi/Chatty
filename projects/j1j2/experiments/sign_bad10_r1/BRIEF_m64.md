# T-BAD10 round 1, stage 2: population-bias check, ONE e_0 chain at M=64 (written at submission)
Question: is e_0 (Marshall x |ViT|, FN c=0) at M=16 biased by population control? Same code (r1/code/gfmc_sign.py), guide, dtau 0.1, burn 30; M=64, 300 post-burn blocks (~20 s/block, ~1.9 A40-h), seed 61, start = shipped samples.
Files: m64/c1/E_mix.npy (post-burn), m64/gate.json, m64/all.log, slurm_<job>.out ends M64_DONE. Reference: e0ext/c1, c2 (M16).
Gate: python3 code/m64_gate.py e0ext/c1 e0ext/c2 m64/c1 m64/gate.json  (inside the sbatch; exit 3 on fail, so afterok stops vmc): PASS iff |e_0(M64) - e_0(M16)| < 2e-4 and >= 256 blocks. Also run quotable.py on the M64 chain together with a
 second chain is NOT possible (single chain); its error is the binning error (quotable.series_chain). z = diff/hypot(errs) is printed: at the expected errors (1.5e-4 and 0.8e-4) the 2e-4 cut is 1.2 sigma, so a bias-free pair fails ~24% of the time.
Branches: PASS -> vmc. FAIL -> the chain stops before vmc; report both values, errors, z. z < 2: probably noise (lead decides to continue); z > 3: population bias at M16, the M16 arms are not comparable: report.
UPDATE (lead, before GO): gate = FAIL iff |e_0(M64) - e_0(M16)| > 2e-4 AND |z| > 2 (significant bias above 2e-4); else PASS and gate.json carries upper_bound_2sigma = |d| + 2 sigma_d. Synthetic tests: bias-free pair PASS, |d| 1.5e-4 with z 0.9 PASS, +6e-4 shift FAIL rc 3.

## RESTART STAGE A (e0m64c2, written at resubmission): e_0 chain 2 at M=64 (m64/c2, seed 62, same code file/guide/dtau/burn, 300 post-burn blocks)
The round-1 VMC gate stopped the first chain (job 17997722); decision (takeover): grade BOTH arms at M=64; m64/c1 (job 17997721, -0.494113(86)) is e_0 chain 1, m64/c2 is chain 2.
Files: m64/c2/E_mix.npy, m64/all_c2.log, slurm_<job>.out ends E0M64C2_DONE. Gate: python3 /project/theorie/a/A.Otaifi/tools/quotable.py series m64/c1/E_mix.npy m64/c2/E_mix.npy --name e0_M64 --plot m64/e0_M64_quotable.png ; LOOK at the raw curve.
Branches: QUOTABLE -> e_0(M64) = verdict value; stage vmc_exact proceeds (afterok). NOT QUOTABLE -> report (physics/bias, qmc-statistics), the chain continues (g_1 not quoted if e_0 stays NOT QUOTABLE). The M16-vs-M64 bound B64 uses m64/gate.json (job 17997721).
