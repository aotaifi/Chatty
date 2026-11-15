# T-BAD10 round 1, stage 1: e_0 extension (Marshall x |ViT|, M16, dtau 0.1), written at submission
Question: e_0 quotable, >= 300 post-burn blocks per chain, sigma ~1.5e-4 suffices for g_1 ~ 1e-3 (PLAN_sign ROUND 1 SIZED).
Files: e0ext/c1/E_mix.npy, e0ext/c2/E_mix.npy (post-burn, time order; prefix = step-0 series step0/prod/e0_c{1,2}/E_mix.npy), e0ext/all.log (RESUME_PROOF lines), slurm_<job>.out ends E0EXT_DONE.
Gate: python3 /project/theorie/a/A.Otaifi/tools/quotable.py series e0ext/c1/E_mix.npy e0ext/c2/E_mix.npy --name e0 --plot e0ext/e0_quotable.png ; LOOK at the raw curve; also confirm 'RESUME_PROOF PASS' for both chains in all.log.
Branches: QUOTABLE -> e_0 = verdict value (stage 5 reads it). NOT QUOTABLE with >= 300 blocks -> fifths/drift or chain disagreement = physics/bias (qmc-statistics): report, stage 5 still runs (g_1 not quoted if e_0 stays NOT QUOTABLE).
 RESUME_PROOF FAIL -> the chain is a cold restart: stop the chain, report.
