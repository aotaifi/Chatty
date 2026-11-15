# T-BAD10 round 1 restart, stage E: FN c=0 of (|ViT|, n_1) at M=64, 2 chains (written at resubmission)
Decision (takeover): both arms graded at M=64. e_0 = m64/c1 (17997721) + m64/c2; e_1 = fn_net64/c1, c2 (seeds 71/72, burn 30, 300 post-burn blocks, ~31 s/block expected, maxsec 12500 s each, limit 7.5 h).
Files: fn_net64/c1/E_mix.npy, c2/E_mix.npy, fn_summary.json, quotable_{e0,e1}.png, all.log, slurm_<job>.out ends FN64_DONE.
Gate: python3 /project/theorie/a/A.Otaifi/tools/quotable.py series fn_net64/c1/E_mix.npy fn_net64/c2/E_mix.npy --name e1 --plot fn_net64/e1_quotable.png ; LOOK at the raw curve; then python3 code/analyze_fn10.py --e0 m64/c1 m64/c2 --e1 fn_net64/c1 fn_net64/c2 --out fn_net64 --bias_json m64/gate.json
Halves rule (analyze_fn10.py): NOT QUOTABLE if any chain |z| > 3 or chi2 p < 0.01 over all chains of both arms; every z and the chi2 printed.
Sigma sizing: step 0 gave sigma ~1.2-1.5e-4 for M16 pairs of ~250 blocks; at M=64 the per-block sd falls ~1/2: 2 chains x 300 blocks -> sigma(e_0) ~ 6e-5 (m64/c1 alone: 8.6e-5), e_1 at or below that; sigma(g_1) well under the 3e-4 bar.
READING RULE (replaces B_M): population-bias bound at M=64 B64 = (|dM| + 2 sigma_d)/3 with dM = e_0(M64) - e_0(M16), sigma_d from m64/gate.json (bias scales as 1/M); printed as 'BIAS B64 ... BRANCH:'.
 RESOLVED if g_1 - B64 > 2 sigma(g_1); INCONCLUSIVE if g_1 > 2 sigma but g_1 - B64 <= 2 sigma (only the bias bound fails); KILL INPUT only if g_1 itself is not beyond 2 sigma(g_1) with data adequate (fit stage: 2000->3200 change < 10%) -> diagnosis + Astra before any verdict.
 e_1 or e_0 NOT QUOTABLE -> plot only, say so. T-BAD10 predictions: g_1 >= 1e-3 (70%); contraction gap_1/gap_0 worse than 8x8's 0.18 (55%).
