# Compute scout for j1j2 JAX fp64 MC, 2026-11-12

Probe: 8x8 'full' VMC eval, s307 params_final (sha b86a0f36), rbmpp_eval.py 1bfd6574 / rbmpp_lib.py 5fa08acc, 1024 chains,
therm 2, jax 0.8.2 venv, 15-min cap. s/sweep = delta(s)/delta(sweeps) between PROGRESS lines. Run dirs: /scratch/a/A.Otaifi/rbmpp8/scout/<tag>/.
H100 reference 2.7 s (lead's number, not re-measured); A40 full 14.6 s (rb8_retrain/s307/ev_full_a40probe, jobs 17903151/17903352).

| device | partition / gres | exist | free now / test-only start | memory | s/sweep (1024 ch) | ratio to H100 | notes |
|---|---|---|---|---|---|---|---|
| H100 NVL | inter `h100nvl` | 7 (usm-cl-nv01) | 7/7 allocated; start 2026-11-23 04:14; cap 4/user | 94 GB | not measured (pending, cancelled 17930399) | 1 (ref 2.7) | allowed |
| H200 NVL | inter `h200nvl` | 5 (kng-cl-nv03) | 5/5 allocated; start 2026-11-22 21:22; cap 2/user | 141 GB | not measured (pending, cancelled 17930400) | ~1 expected | allowed |
| A100 | inter `a100` | 2 (kng-cl-nv02) | 2/2 allocated; start 2026-11-23 03:45 | 40/80 GB (unchecked) | not measured (pending, cancelled 17930401) | - | allowed |
| A40 full | cip, inter `a40` | 9+9+8 (cip-cl-nv01, kng-cl-nv01/02) | all 26 allocated (inter start 2026-11-23); cip cap 3 GPUs/user | 46 GB | 14.6 (ev_full_a40probe) | 5.4x slower | |
| A40-24gb slice | cip `a40-24gb` | 8 | 4 idle at 10:27, started at once | 24 GB | 15.5 (17930395, a40-24gb/) | 5.7x | jax 0.8.2 only; shared vGPU |
| A40-16gb slice | cip `a40-16gb` | 9 | idle nodes, started at once | 16 GB | 17.2 (17930463, a40-16gb/, chunk 16) | 6.4x | jax 0.8.2 only; needs ~10 GiB device |
| A40-8gb slice | cip `a40-8gb` | 6 | idle | 8 GB | FAILS: OOM, 9.89 GiB alloc even at chunk 8 (17930431) | - | not usable for 8x8 |
| Quadro P5000 | inter `p5000` | 4 (met-cl-vis02) | 3 free, started at once | 16 GB | no 20-sweep PROGRESS line in 15 min (17930465), so > ~45 s/sweep | > 17x slower | fp64 weak; not usable |
| RTX 2080 Ti | inter `rtx2080ti` | 4 (met-cl-vis01) | 2 of 4 free but job stayed PENDING (Priority); start 2026-11-23 14:11 | 11 GB | not measured (pending, cancelled 17930429) | - | 8x8 needs ~10 GiB device: likely OOM or marginal |
| V100 | inter `v100` | 8 (th-cl-nv01/02) | all allocated; start 2026-11-24 19:26 | 16/32 GB | not measured (pending, cancelled 17930430) | - | earlier failure was cuDNN conv autotune (sm_70, ViT code); our RBM code has no conv, untested |
| CPU Xeon W-2245, 6 threads | th-ws, th-ws-7010m52 | 4 ws (m42/47/52/53) | started (node woke from power-save) | 24 GB | no PROGRESS in 11 min (17930538), so > ~220 s/sweep | > 80x slower | lower bound only; 6 threads |
| CPU th-cl-262h, 32 thr | cluster, th-cl-262h04n1 | 23 nodes | PENDING Priority, pinned-node test-only 2027-05-22 | 1-2 TB | not measured (cancelled 17930435) | - | nodes at load 12-105 of 128 |

## Notes
- Nothing beyond A40-class was measurable today: every H100/H200/A100/A40/V100 GPU was allocated, and the pending probes were cancelled at wrap-up.
- Free A40 slices are the only extra GPU capacity that starts at once: 24gb 5.7x and 16gb 6.4x slower than an H100, about equal to a full A40 (the slice is a time-shared vGPU, so throughput can vary with neighbours).
- The 8x8 'full' eval needs ~10 GiB on device regardless of chunk (chunk 64 asks 10.4 GiB, chunk 8 still 9.89 GiB), so it fits 16 GB cards and up and not the 8 GB slice or probably the 2080 Ti.
- Two printing/chunk changes vs the A40 probe: a40-16gb and p5000 used chunk 16 (a40-24gb chunk 64); the a40-8gb, 2080 Ti, V100 and CPU copies print every 5 or 1 sweep (rbmpp_eval.py sha 4389b186 for a40-8gb, 126a56e8 for cpu_ws). Numerics unchanged.
- Use (a) VMC evals: Paderborn/theorie H100/H200 when free, otherwise A40-24gb/16gb slices in parallel (up to the cip cap of 3 GPUs/user, shared with any full A40); (b) FN-GFMC chains of the same kernel class: same ranking; P5000 and the 8gb slice are out; (c) CPU-only pool/label work: workstations and cluster CPUs, not for this eval kernel (>80x slower than an H100 on 6 threads). The 32-thread 262h figure is missing.
- Not checked: H100/H200 speed on the venv, the V100 failure on this code, A100 memory size, CPU at 32 threads.
- Caps: inter QoS 512 CPUs, 8 jobs, gres h100nvl 4, h200nvl 2; cip 3 GPUs, 4 jobs. MaxRSS of probes 0.65-1.1 GB (GPU), 5.8 GB (CPU ws).
