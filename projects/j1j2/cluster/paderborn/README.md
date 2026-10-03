# Paderborn production backend

Purpose: production execution of the J1-J2 fixed-node / Krylov campaign.

Mac-side source of truth:
- /Users/aliotaifi/Chatty/projects/j1j2
- /Users/aliotaifi/j1j2_vit_bench

Paderborn compute mirror:
- /pc2/users/h/hpcalot/chatty/j1j2

Policy:
- develop and interpret results on the Mac;
- stage only minimal source + active inputs to Paderborn;
- run GPU-heavy production there;
- copy decisive outputs back to the Mac;
- keep generated outputs under Paderborn chatty/j1j2/results.

Backend benchmark, 2026-09-28, steady 4096-state ViT evaluation:
- Mac M2 CPU: ~2.61k states/s
- Paderborn A40: ~32.64k states/s
- Paderborn H100: ~782.36k states/s
Checksums matched.

Current production job:
- Slurm job 3466194
- partition gpu_h100
- 1 H100, 16 CPUs, 64 GB RAM
- script: closedloop_iter2_h100.sbatch
- Paderborn results root: /pc2/users/h/hpcalot/chatty/j1j2/results
- native Slurm END/FAIL email enabled to the Chatty notification address.
