# T-BAD10 round 1, stage 3: labels K[Marshall] on 4000 tempered roots, written at submission
Question: complete one-swap neighbourhood labels y = sgn[1 + alpha (E_L(|ViT| M; z) - E)] (K[M] = M y) at 10x10; 1000 chains x 4 samples, held-out iff chain % 5 == 0 (800 roots held out).
Files: labels/chunk_*.npz (L1 packed spins, lp, r float32, ELM, f, y), samples.npz, sampler_meta.json (mean log|psi| per sample time: equilibration look), controls.json, labels_done.json, all.log, slurm_<job>.out ends LABELS_DONE.
Gate: controls.json verdict PASS (C1 20 states vs direct recomputation + corrupted copy detected; C2 wrong-alpha arms change labels; C3 Marshall-in-place generic path 0 mismatches); the stage exits 3 otherwise (fit does not start). No quotable.py (labels).
 Read: minus_share and flip fraction by decade (fit stage reports), sampler_meta mean_lp_by_sample drifting? (non-stationary sampler = burn too short).
Branches: PASS -> fit (afterok). FAIL after chunk 0 -> stage aborted early (R1_LABELS_ABORT); report. TIMEOUT -> resubmit the same sbatch (chunks restart).
UPDATE (lead change 5): SAMPLER GATE before any labelling: sampler_meta.json gate{log_psi2, diag_energy: window means A (first half of [burn/2, end]) vs B (second half), per-chain differences, z over the chains}, passed = |z| < 3 for both.
 Fails -> sampler_meta.json passed=false, 'R1_LABELS_ABORT sampler not equilibrated', exit 4 (fit/fn do not start). Broken check in smoke: Neel start, burn 0 -> exit 4 (S3s). Also runs when samples.npz is reloaded.
 With 1000 chains the test is sensitive; a genuine small residual drift of the shipped |ViT|^2 -> |ViT| relaxation could trip it: on FAIL report the drift sizes (diff vs chain_sd) before any decision.

UPDATE (restart): constants are read from vmc_exact/consts_alpha0_E0.json (+ .sha256) = the EXACT-estimator VMC stage, not r1/vmc.
