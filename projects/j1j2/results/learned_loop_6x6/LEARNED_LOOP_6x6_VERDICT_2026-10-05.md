# Learned FN/Krylov loop on 6x6 (J2/J1 = 0.5), iteration 1 (2026-10-05)

Code: `experiments/learned_loop_6x6/` (ll6_core, ll6_iter, ll6_fn, ll6_ed, ll6_vmcpair, ll6_tune, vit_dt).
Run dir: ws1 `~/chatty_ll6/runs/` (A40, venv `~/chatty_ll6/venv_gpu`, jax 0.10.2 CUDA). Per-site energies. E0 = -0.5038097, ViT = -0.503654(21).

## Validation
- Float32 ViT with FULL fp32 matmuls: log|psi| error 1.2e-5 rms vs the fp64 original. The jax GPU default (TF32) gives 0.027 rms and was unusable.
- FN code reproduces fn_bound_check seeds 9501/9502/21001/21002 to 1e-6. ED scorer reproduces w(K1) = 1.125e-3 (450/400k).
- Baseline E_FN[|ViT|, K1 signs] = -0.503100(40) from 76 populations. The quoted -0.503288(90) came from 8 populations and was low by about 2 sigma.

## Iteration 1 (job it1b 16818096)
- Training: fine-tuned Re log psi of the ViT on the frozen H_FN. SR with shift 10 and trust 0.02; 4096 fresh training and 8192 validation samples per step. 6 steps accepted, rejected at step 7. Sum of per-step validation decreases: -1.9e-4.
- Independent held-out paired dL: -0.7(6.8)e-5 on samples from |a1|^2 and -5.8(26)e-5 on samples from |a0|^2. Both are unresolved. RMS(delta log a) = 0.018.
- Amplitude-only FN: E_FN(a1, s0) - E_FN(base) = -1.1(0.7)e-4 (paired, 32 populations).
- Sign step: T = -6.37, label-free. The half-sample estimates -1.5 and -12.3 show T is unstable. Held-out dH = -2.39(0.49)e-4.
- <H>_new guide - E_ViT = +0.4(3.5)e-5 (paired), so <H>_new = -0.50365(4). This is below E_FN[old] by 5.5e-4.
- True sign error w_s: 8.0(2.0)e-5. Compare K1 1.13e-3, ViT 1.3e-4, and K2 at fixed ViT amplitude 2.4(0.35)e-4.
- std(log a - log|psi0|) went from 0.0478 to 0.0610. The learned amplitude moved away from the exact amplitude.
- E_FN = -0.503667(63) from 16 populations; the median is -0.50357. One population fell to -0.50435, below E0. E_FN - ViT = -1.3(6.7)e-5.
- Control, K2 sign step with no learning (job it1 16818020, T = -1.70): E_FN = -0.503596(52) from 10 populations; <H> - ViT = +5.1(4.8)e-5.

## Verdict
After one iteration the loop ties the ViT and does not beat it. Nearly all of the gain over the K1 guide comes from the second current-sign Krylov step. The amplitude step is at the noise floor: about 1e-4 in E_FN, which is not resolved. It does improve the signs that the next Krylov step produces (w_s 2.4e-4 -> 8e-5).

## Cost and why it stops here
- Iteration 1 cost about 5.2 GPU-h: SR 0.35, sign step 0.6, FN 16 populations x 11 min, ED 200k samples 0.75 h. Total use was about 11.5 GPU-h.
- Every sign step adds one hop to the sign recursion. Shell sizes per state are 1.3e3, 4.9e4 and 1.2e6 states for 1, 2 and 3 hops.
- In iteration 2, one SR step took 38 min (4.5e8 evals per 6144 samples). The job (16818473) was cancelled after 1 step.
- The K3 sign step and FN on the K3 guide need about 1.2e6 amplitude evaluations per sample or walker. Estimated cost of a complete iteration 2 is 50 GPU-h or more.
