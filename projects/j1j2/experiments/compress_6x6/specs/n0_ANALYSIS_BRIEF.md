# N0 analysis brief (local copy of Paderborn H100 run @REM@; local dir @LOC@)
Question (README compress_6x6 "Test N0", pre-registered 2026-10-23): is the tail (bonds carrying the d^2 of the violating-bond error of a_2 = O2b round 1) the near-node
cancellation region, c(x) = |sum_y H_xy phi(y)| / sum_y |H_xy phi(y)| small?  No training; w-sampled violating bonds, 2^20, seed 7777.
File: @LOC@/cx_n0.json: controls {A_bruteforce_c (rel 1e-10), B_unsigned (c = 1), C_sampled_bonds}, sets {top_0.1pct, top_1pct, reference_all_w_sampled}, decision {ratio, branch}, stop (only on CONTROL FAILED).
Gate: exact computation, no Monte Carlo chains (quotable.py n/a).  python3 -c "import json;d=json.load(open('@LOC@/cx_n0.json'));print(d.get('stop'),d['controls'],d['decision'])"
Branches (median min(c) of top 0.1% over the reference median): <= 1/3 near-node picture holds (proceed with the signed guide); >= 2/3 picture dead; between: report.
The worker reports the numbers to the lead only; the lead decides.
