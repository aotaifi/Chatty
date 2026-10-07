# Stall diagnosis 6x6 (J1-J2, J2/J1 = 0.5, periodic): progress log

Question: why does the FN/Krylov loop stall at the ViT level on 6x6? Compression (ViT cannot represent a better
amplitude), information (FN information per iteration), or optimizer/noise (sampled updates)?
Code: `experiments/stall_6x6/`. ws1 run dir `~/chatty_stall6/` (code, logs); large data and runs on
`/project/theorie/a/A.Otaifi/chatty_stall6/` (csr, data, runs).
Budget: <= 15 GPU-h + 100 CPU-h. All energies per site; E0 = -0.50380965.

## Method (exact, no sampling in any energy)
- Everything lives in the fully symmetric sector (k=0, A1, flip+; D = 15,804,956 orbit reps). A symmetric state is a
  per-configuration log-amplitude la_r and a sign s_r on the reps; its energy, the Krylov sign step (energy-optimal
  threshold) and the lattice-FN ground state are computed exactly, as in the CPU exact loop
  (`krylov_sign_structure/experiments/closed_fn_krylov_sym6x6.py`).
- GPU implementation (JAX float64): H = diag(sqrt n) C diag(1/sqrt n) with integer codes C (8 c in int16), upper
  triangle only (H symmetric) -> 4.2 GB on the device, fits an RTX 2080 Ti. FN ground state by block-1 LOBPCG with
  the same diagonal preconditioner as the CPU code.
- A network state is the network evaluated on the reps (one evaluation per orbit). Supervised fits use exact data:
  reps drawn from p_r^beta (p = sector weight of the target), mapped to a uniformly random orbit element (random
  space-group element x spin flip, so the ViT is trained to be point-group/flip symmetric), plus K random one-hop
  neighbours with exact target values (canonical-rep lookup on the GPU).

## Log
- 11:20 Setup. CSR build (CPU, 16 cores, 4 min, 1.1 CPU-h): D = 15,804,956, nnz = 1.185e9, E0 from the stored ED
  vector -0.5038096538908782 (table -0.50380965389088), codes 8c exact (max rounding 9e-16), range -72..120.
- 11:24 All full A40s are allocated to multi-day jobs (cip-cl-nv01, kng-cl-nv01/02); the first GPU job (16856428)
  sat in the queue -> cancelled; code rewritten for an 11 GB RTX 2080 Ti (upper-triangular coded H, 4.2 GB).
- 11:27-11:33 **Home quota hit**: the 7.2 GB CSR in `~/chatty_stall6/csr` pushed `/home` over quota (writes failed
  for ~6 min, also for other jobs writing to home). Moved to `/project/theorie/a/A.Otaifi/chatty_stall6/` (306/475 GB);
  home writable again at 11:34. All large files of this task now live on /project.
- 11:37 Unit tests (ws1 login CPU, `test_st6_small.py`): synthetic 400-dim sector vs dense numpy: matvec 4e-15,
  FN ground state 6e-14 (vector 1e-6), Krylov threshold step = brute force over all cuts (same signs). Real 6x6
  table: canonical-rep roundtrip exact on 5000 random orbit images; lookups of configs and one-hop neighbours equal
  to the independent numpy `Psi0` class. `test_st6_fit_cpu.py`: all model variants build and train
  (ViT 154,980 params; ViT d96/h12 392,976 = 2.5x). Adam at lr 1e-4 moves the warm ViT by 1.6 rms in log a in 4
  steps -> warm-start fine-tuning needs lr ~1e-6..1e-5 (scan in the first GPU job).
- 11:45 GPU job 16856662 (2080 Ti): validation vs CPU exact loop + ViT baseline/oracles + lr scan. Queued (all
  2080 Ti / A40 busy).
