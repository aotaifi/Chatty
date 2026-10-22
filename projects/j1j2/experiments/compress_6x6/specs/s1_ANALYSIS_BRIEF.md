# S1 analysis brief (run @RUN@)
Question (README compress_6x6, "Test S1: the ceiling of the Pfaffian's own sign", pre-registered 2026-10-22 18:30, not edited here): O2a's fit-free hop+FN chain
(g_k = |a_k||1 + alpha_V (E_L - E)|, a_{k+1} = FN solution of g_k, alpha_V by the O1 rule, no training) from the VMC PP table, but with the PP's OWN sign
(sP of sym_tables_PP.npz, oracle-free) instead of s_2. Up to 15 rounds, stop when E_FN changes < 2% per round. The ceiling = the self-consistent E_FN of s_PP.
Files: @RUN@/cx_s1.json  (sign_name, disagree{w_ne_s0_modsign, w_ne_s2_modsign}, EFN_zero_PP, control_zero, control_ref_k, rounds[k]{alpha_V, EFN, c, ratio, aitken,
  expelled{w_psi0, w_psi0_sign_right, w_psi0_sign_wrong}}, aitken, E_last, stop, done); @RUN@/progress.txt; spec_s1.json; code/.
Controls (must hold; cx_s1 exits 2 otherwise): zero-hop 4.773828e-5 and round-0 E_FN 1.869427e-5 (b0 control job 17444507). Round 0 with the s2 chain reproduced O2a (2.36919e-5, 8.106143e-6) in the pre-staging control job.
Gate: exact solves, no Monte Carlo chains: quotable.py does not apply (say so); read the numbers directly:
  python3 -c "import json;d=json.load(open('@RUN@/cx_s1.json'));print(d['stop'],d['aitken'],d['E_last']);[print(r['k'],r['alpha_V'],r['EFN'],r.get('c'),r.get('ratio'),r.get('aitken')) for r in d['rounds']]"
Decision (copied from the README; ceiling = the converged E_FN of the own-sign chain, Aitken beside it; if the 2% stop was not reached, say so and give the Aitken value and the last E_FN):
  - Ceiling(s_PP) <= 3e-6: the oracle-free Pfaffian sign is good enough for the loop at 6x6. O3 is redesigned as "carry the network's own sign" (the loop trains
    sign and amplitude together, like VMC), and O2b is re-run with s_PP (one change). This would remove the sign oracle.
  - 3e-6 < ceiling <= 1e-5: the PP sign is a usable start for O3; O3 flips from s_PP, not from the ViT's s_1.
  - Ceiling > 1e-5: the advantage is amplitude-sign consistency only; O3 as planned.
Lead's prediction before the run: ceiling 3-8e-6 (about 60% in the middle branch). For reference, the s_2 ceiling is 9.15e-7 and O2a's chain reached 1.69e-6 at k=5.
The worker reports the numbers to the lead only; the lead decides.
