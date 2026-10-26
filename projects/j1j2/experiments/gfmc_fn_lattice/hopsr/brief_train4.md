# 4x4 hop-aware SR training test (Paderborn job @JID@, gpu_h100): remote /scratch/hpc-prf-ppdmrg/hpcalot/hopsr/runs/train4_@JID@
Question (deliverable 2): does the sampled hop-aware SR trainer, started at the E_v optimum of the RBM-only class (w30), reach an exact
E_FN(a*h_a)/N <= -0.5285009557 (within 20% of the gap to the exact optimum -0.5285525325; start -0.5282946487)?  Runs A, B (E1 objective,
two learning rates) and C (control: objective E0, must NOT gain).  The exact E_FN is computed on ws1 after the job (l8_loop4x4.py evalfn on
{A,B,C}/params_*.npz), not by this job.
Files: progress.txt (run lines with exit, MaxRSS, TRAIN_DONE; heartbeats), {A,B,C}/progress.txt (PROGRESS it lines: E1/N +- jackknife),
{A,B,C}/train.json, {A,B,C}/params_XXXXX.npz.
No independent MC chains to pool (one optimisation trajectory per run): quotable.py not applicable; gate = grep -h "run . exit\|TRAIN4_DONE" progress.txt.
Next: all exit 0 -> worker evaluates exact E_FN on the checkpoints (evalfn) and applies the 20%-of-gap rule. A run exit != 0 -> read its log.
