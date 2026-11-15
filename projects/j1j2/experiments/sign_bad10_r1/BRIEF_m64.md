# T-BAD10 round 1, stage 2: population-bias check, ONE e_0 chain at M=64 (written at submission)
Question: is e_0 (Marshall x |ViT|, FN c=0) at M=16 biased by population control? Same code (r1/code/gfmc_sign.py), guide, dtau 0.1, burn 30; M=64, 300 post-burn blocks (~20 s/block, ~1.9 A40-h), seed 61, start = shipped samples.
Files: m64/c1/E_mix.npy (post-burn), m64/gate.json, m64/all.log, slurm_<job>.out ends M64_DONE. Reference: e0ext/c1, c2 (M16).
Gate: python3 code/m64_gate.py e0ext/c1 e0ext/c2 m64/c1 m64/gate.json  (inside the sbatch; exit 3 on fail, so afterok stops vmc): PASS iff |e_0(M64) - e_0(M16)| < 2e-4 and >= 256 blocks. Also run quotable.py on the M64 chain together with a
 second chain is NOT possible (single chain); its error is the binning error (quotable.series_chain). z = diff/hypot(errs) is printed: at the expected errors (1.5e-4 and 0.8e-4) the 2e-4 cut is 1.2 sigma, so a bias-free pair fails ~24% of the time.
Branches: PASS -> vmc. FAIL -> the chain stops before vmc; report both values, errors, z. z < 2: probably noise (lead decides to continue); z > 3: population bias at M16, the M16 arms are not comparable: report.
