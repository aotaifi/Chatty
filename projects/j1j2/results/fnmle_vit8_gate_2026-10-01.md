# 8x8 full-network FN-walker MLE gate — 2026-10-01

Three independent executions of the warm-started 8x8 ViT MLE update completed:
- H100 3497176 (batch 512)
- A40 3497177 (batch 512)
- quick A40 3497195 (batch 256)

All three fail the held-out-replica acceptance gate at the first optimizer step. Representative
held-out log-likelihood gains at step 1 are -0.2524 (H100), -0.2878 (A40), and -0.2594
(quick A40). In every run the best checkpoint is step 0 with best_val_ll_gain=0.

The first Adam step at LR=2e-5 is already macroscopically large in function space:
delta-log-amplitude RMS on the guide pool is about 0.33-0.36 and guide-pool importance
ESS falls from 4096 to roughly 805-993. Subsequent steps worsen both train and held-out
likelihood. Therefore the current optimizer configuration is rejected.

This does not falsify the MLE half-step identity, which passed the exact 4x4 test. NetKet
Spin(1/2) samples were explicitly checked to use +/-1, matching the FN walker encoding, so
there is no sampler-input convention mismatch.

Next decisive diagnostic: full-batch empirical KL gradient at theta0 using all rep1 FN walkers
and the stored guide a0^2 sample, followed by a line search against independent rep2.
H100 Slurm 3500546. If infinitesimal steps improve held-out likelihood, replace Adam with a
trust-region/small-step optimizer. If even the infinitesimal full-batch direction fails, the
finite-population MLE estimator itself is not reliable at the current walker/sample quality.
