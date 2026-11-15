# T-BAD10 round 1, stage 4: fit n_1 (64x10 CNN, screen-arm-d recipe, 30 steps/root) + learning curve 1000/2000/3200 train roots, written at submission
Question: held-out label disagreement and E_L error vs data (curve) and the stored net n_1 for the FN grade.
Files: fit/fit_{1000,2000,3200}.json (disagree, on_flips/on_keeps, by_decade, EL{mean_abs_err, rms, Marshall base}), net_*.pkl (sha in json), final_net.txt (= largest), all.log, slurm_<job>.out ends FIT_DONE.
Gate: none (no MC). Read: 2000 -> 3200 change of held-out disagreement / E_L error: < 10% relative = data adequate (8x8: 4k->8k 3%); train error vs held-out (train error rising with n = capacity-limited, as at 8x8).
 Compare with 8x8 round 1 (held-out disagreement 0.103, E_L err 3.0 total, Marshall 4.4) as context only.
Branches: always continue to stage 5 (afterok). Still improving > 10% from 2000 -> 3200 -> FN grade is data-limited: note in the report (T-BAD FINAL: 'data-limited -> no verdict, double data').
