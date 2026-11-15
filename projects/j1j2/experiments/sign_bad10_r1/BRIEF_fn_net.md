# T-BAD10 round 1, stage 5: FN c=0 of (|ViT|, n_1), 2 chains, M16, written at submission
Question: e_1 and g_1 = e_0 - e_1 (T-BAD10: predicted g_1 >= 1e-3, 70%; contraction gap_1/gap_0 worse than 8x8's 0.18, 55%).
Files: fn_net/c1/E_mix.npy, c2/E_mix.npy (post-burn), fn_net/fn_summary.json + quotable_{e0,e1}.png (from analyze_fn10.py), all.log, slurm_<job>.out ends FN_R1_DONE. e_0 = e0ext/c{1,2}.
Gate: python3 /project/theorie/a/A.Otaifi/tools/quotable.py series fn_net/c1/E_mix.npy fn_net/c2/E_mix.npy --name e1 --plot fn_net/e1_quotable.png ; LOOK at the raw curve; then python3 code/analyze_fn10.py --e0 e0ext/c1 e0ext/c2 --e1 fn_net/c1 fn_net/c2 --out fn_net
Branches: both arms QUOTABLE, g_1 > 2 sigma(g_1) -> round 1 USEFUL; report e_1, g_1, gap_1/gap_0 (gap to e_ViT = -0.497682(4), step-0 control). g_1 not beyond 2 sigma with data adequate (fit stage: 2000->3200 change < 10%) -> KILL of 'the loop improves the sign every round at 10x10':
 diagnosis (label error of n_1 vs exact K[M] on held-out; exact one-step leverage) + Astra before any verdict. e_1 NOT QUOTABLE -> plot only, say so. sigma per arm target <= 1e-4.
