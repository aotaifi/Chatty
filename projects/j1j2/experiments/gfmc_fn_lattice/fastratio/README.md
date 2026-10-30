# fastratio: determinant-contraction bond ratios for the FN-GFMC walker code

Base = the code that was LAUNCHED for rb8 p1_s1 (md5 gfmc_run.py 0f858d33, gfmc_fn_lib.py 40a1c409, rbmpp_lib.py de68e845), committed unchanged first.
Spec: /scratch/a/A.Otaifi/ansatz_runs/gpu_profile/DECISION.md (+ ASTRA_ANSWER.md).

Flag: `gfmc_run.py ... --ratio ref|fast` (default ref = production behaviour, byte-identical output lines).  `make_fns(lat, dtype, use_rbm, ratio=...)`.
fast: per image P = A G, Q = G B, S = P B (A = f[D,D], B = f[U,U], C = f[D,U], G = F^-1, U/D sorted), psi_PP(z')/psi_PP(z) = -(P*Q + G*(C - S))[b,a],
b = position of the down site q in D, a = position of the up site p in U.  Replaces V and the two bond reductions (rbmpp_lib.py img_quant, lines 183-190 of the launched file).
Fallback (fast only): an image with non-finite log Psi / ratio or 1-norm condition estimate ||F||_1 ||G||_1 > kappa_max (1e12) flags the configuration; the guide
re-evaluates flagged configurations with eloc_full_direct (all neighbours by their own slogdets), outside the jitted batch (RBMPPGuideG._chunked).

Tests (CPU or GPU, f64): check_agree.py (item 1, also `--break sign|noCS`), truth_pairs.py (50-digit truth), acc_fast.py + acc_fast_compare.py (near-node pairs of gpu_profile/acc_f64),
test_fallback.py (singular images), bench_gpu.py + job_speed.sh + speed.sbatch + gate_speed.py (matched-input timing, walker equivalence), watch_local.sh (job watcher).
