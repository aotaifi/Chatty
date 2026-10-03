# Frozen-H_FN factorized residual NQS — exact 4x4 SR verdict (2026-10-03)

Late-guide exact 4x4 test of the directly-callable amplitude refresh
a_theta(x) = a_k(x) exp[r_theta(x)]
with signs/nodes held fixed and the Rayleigh quotient of frozen H_FN[a_k,s_k] minimized by exact SR.

Implementation
- D = 12,870
- translational 2D ViT residual, 154,780 parameters
- exact frozen-H_FN energy and exact Fisher/QGT
- memory-safe state chunking, chunk=512
- trust RMS(delta log a)=0.002
- 30 accepted SR steps
- full-space non-chunked JAX version OOMed only because lowering captured ~13.9 GB constants; this was an implementation issue, not a physics change.

Results
- 3551953, shift=0.1:
  frozen gain fraction = 0.9988307021
  rebuilt-E_FN gain fraction = 0.9970657510
  phi_FN fidelity = 0.9999999650
  phi-weighted log-ratio RMS = 1.8547e-4
- 3551954, shift=0.01:
  frozen gain fraction = 0.9985525139
  rebuilt-E_FN gain fraction = 0.9968015942
  phi_FN fidelity = 0.9999999488
  phi-weighted log-ratio RMS = 2.2486e-4

Verdict
The earlier late-stage direct-head ceiling (~27–38% retained refresh gain) is not a fundamental representability limit. The factorized node-preserving parameterization plus frozen-operator SR can represent essentially the full exact FN amplitude refresh at 4x4.

Project consequence
Keep signs analytic/on-the-fly via one-shell K1/Jacobi neighbour vote and learn only the positive directly-callable amplitude. No sign-network distillation is required. Remaining question is finite-sample/scaling of the frozen-H_FN SR estimator.

8x8 promotion
Pilot 3554536 was launched from the successful a1,s1 checkpoint with ntrain=256, nval=256, fisher_n=256. Gate requires both train and independent validation frozen-H_FN energy to decrease and reweighting ESS >80% before any physical FN run.
