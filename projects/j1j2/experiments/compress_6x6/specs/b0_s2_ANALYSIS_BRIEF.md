# B0 analysis brief, SEED @SEED@ (local copy of the Paderborn H100 run /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/@NAME@; job on gpu_h100; Amendment B0-2; seed 1 = /project/theorie/a/A.Otaifi/chatty_compress6/runs/b0_ppbf_simple, job 17445407; training projection: simple)
Question (README compress_6x6, "Test B0", pre-registered 2026-10-22): the SAME network as the O2b loop (PP + 3x3 backflow, cx_t1s 'ppbf'),
trained by plain SR VMC from the rpT1_pp start (u1 = u2 = 0), then read out exactly: E_FN with sign s_2 at zero hop and one hop (alpha_V),
and with the network's own sign (zero hop, one hop).  Is the standard pipeline (VMC + one hop + FN) on this network within 1.2x of the loop?
Files: /project/theorie/a/A.Otaifi/chatty_compress6/runs/@NAME@/cx_b0.json  control {s2 {EFN_zero_hop, EFN_one_hop}, own {...}, net_vs_table, ok}; b0_s2 {EFN_zero_hop, EFN_one_hop, one_hop{alpha_V,...}, disagree};
  b0_own {same, own sign; disagree = |psi0|^2 weights of sign mismatch with s0 and s_2, raw and mod global sign}; table_b0.npz (lP, sP);
  /project/theorie/a/A.Otaifi/chatty_compress6/runs/@NAME@/train/train.json  hist per step (E, err, var, t_step ...), plateau {diff, se_diff, ok}, stop ('converged' | 'budget out' | ...),
  steps_done, tail_mean_100; params_XXXXX.npz every 50 steps, params_final.npz; progress.txt (train/ and run level).
  E_loop = best E_FN(g_k) in /project/theorie/a/A.Otaifi/chatty_compress6/runs/o2b_17431886/cx_o2b.json (rounds[k].EFN; 9.26e-6 at k=3 when this brief was written, run still going).
Gate: no Markov-chain estimator enters the decision (E_FN is an exact solve of the table).  The VMC energy trace is a Markov-chain estimate; read the
  plateau record in train.json (se_diff = statistical error of the 100-step-window difference; the pre-registered tolerance is 1e-6 per site and is
  below se_diff at 2048 chains, so 'converged' can be a coincidence: judge the plateau from the energy-vs-step curve, not from the flag).  Command:
  python3 -c "import json;d=json.load(open('/project/theorie/a/A.Otaifi/chatty_compress6/runs/@NAME@/cx_b0.json'));print(d['control']['ok'],d['b0_s2']['EFN_zero_hop'],d['b0_s2']['EFN_one_hop'],d['b0_own']['EFN_one_hop'])"
Control (must hold, else the readout is not trusted): s_2 zero-hop 5.3677e-5 and one-hop 2.3692e-5 on the unmodified start (rel 1e-3; cx_b0 exits 2 otherwise);
  the network at the start (u = 0) agrees with the table (control.net_vs_table).
Next per verdict (README, E_B0 = B0 one hop with sign s_2, E_loop = O2b best E_FN(g_k)):
  - E_B0 <= 1.2 x E_loop: no accuracy edge on this network. The fit-floor lever is deprioritised; the loop's case is cost only, and the next step is
    GPU-h to reach equal E_FN (loop vs SR), which O4 needs anyway.
  - E_B0 >= 2 x E_loop: the FN targets carry what energy minimisation does not find; the edge is real. Next: the fit-floor lever (5x5 backflow fitted
    to the saved round_k/target_la.npy, tax compared with 5.93e-6).
  - In between: a modest edge; run both, the fit-floor test first (cheaper).
  - B0 not converged (train.json stop != converged, or the energy curve still falling): E_B0 is an upper bound; only the "edge is real" branch can be
    read off it, and only if a longer run is not projected (from the energy-vs-step curve) to close the gap.
The worker reports the numbers (and the extra own-sign numbers: zero hop, one hop, disagreement weights) to the lead only; the lead decides (projects/j1j2/PLAN.md).
Amendment B0-2 (README): the original seed 1 (17445407, lr 0.05) is VOID; this run trains lr @LR@ -> 0.002 (cosine, 4300 steps), plateau stop DISABLED (plateau_min_step 100000), seed @SEED@.
  The two production runs are b0_ppbf_simple_s1h (seed 1) and b0_ppbf_simple_s2 (seed 2), same code (sha256), start and specs but the seed.  Compare them in the same directory
  /project/theorie/a/A.Otaifi/chatty_compress6/runs/: E_FN(s2) zero and one hop, own-sign numbers, train.json tail_mean_100 and the energy-vs-step curve against the start
  (-0.503708 +- 1e-5 from the first 10-20 steps; training must have gone BELOW it, else the readout is not a trained-network readout).  stop must be 'steps exhausted' (4300), not 'budget out'.
  A disagreement of the two seeds' one-hop E_FN beyond the training noise (scatter of the last 100 steps) is a FINDING: report both numbers, do not average it away (qmc-statistics).
  The worker reports to the lead only; the README decision rule uses both seeds.
