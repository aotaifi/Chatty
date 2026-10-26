# hop-aware SR pilot (Paderborn job @JID@, gpu_h100, QoS cont): remote /scratch/hpc-prf-ppdmrg/hpcalot/hopsr/runs/pilot_@JID@
Question: cost per SR step of the sampled hop-aware trainer (hopsr_train.py) at 4x4 (t4: RBM-only, 16k samples), 8x8 (p8: rb8_prod,
8192 chains, translations) and 6x6 (p6: rpT4_sr2, full projection, 4096 chains); MaxRSS and GPU memory per stage; per-step E_1/N and its
jackknife error (the noise that sizes a fine-tune). No physics verdict; not a production run.
Files: progress.txt (stage lines with exit code, MaxRSS_kB, gpu_mem_peak_MiB, TRAIN_DONE), {t4,p8,p6}/progress.txt (PROGRESS it lines with
t_step and the split sample/x/y/red/solve), {t4,p8,p6}/train.json.
No MC chains for quotable.py (cost pilot): gate = grep -h "stage .* exit\|PILOT_DONE" progress.txt; and grep -h "PROGRESS it" */progress.txt.
Next: all three exit 0 -> fill the cost table (median t_step of steps 2-3, H100-h per 100 steps, MaxRSS) and size the fine-tunes (worker).
A stage exit != 0 -> read its .log; OOM -> lower jchunk/chunk and re-pilot that stage only.
