# Learned FN/Krylov loop 6x6, iterations 2-3 with stored signs (progress log)

Goal: beat the ViT on 6x6 (J2/J1 = 0.5): E_FN/site below -0.503654(21) by > 2 sigma (RBM+PP -0.503765, E0 -0.5038097).
Code: `experiments/learned_loop_6x6_it2/`; run dir ws1 `~/chatty_ll6it2/` (reuses venv/data of `~/chatty_ll6`, K1 stored net of `~/chatty_ss6`).

- 2026-10-06 10:30: read handoff, CLAUDE_TAKE, ledger, it1 verdict, rung-2 verdict, ss6 K1 progress. Facts that set the design:
  it1 threshold used only 2048 |a1|^2 samples with ~1 flipped sample (flip frac 5e-4), hence T unstable;
  it1 SR local data cost 41M ViT evals/step because the recursive s^(1) needs level 2 (3315 states/sample);
  with a fully stored sign the same step needs level 1 (86 states/sample), ~40x cheaper.
  FN on a level-2 guide (K1) costs ~30 s/population (M=128, beta 1.2), so 64+ populations are cheap once signs are stored.
- Notation: G1 = (a1, s1) iteration-1 guide. Three versions of s1: (i) recursive s^(2) [K1 recursion with |ViT|, T_FN; hop with a1, T1=-6.3718];
  (ii) stored K1 net (ss6 L b0.3) + exact hop with a1, T1; (iii) fully stored: K1 net x net2 (net2 distilled from (ii)).
- 2026-10-06 10:55: code written (it2_core, it2_distill, it2_amp, it2_sign, it2_fn, it2_pipe, it2_gpu.sbatch). Smoke tests on rtx2080ti
  (16843870 OOM in 65536-batch sign-net eval -> bucketed eval <= 16384; 16844147 slice bug; 16844844 distill+amp OK, sign ED slice bug; 16844882 sign+FN rerun).
- 2026-10-06 10:57: distillation of s1 submitted: 16844883 d2main (A40; beta 0.3, 1024x40 samples + neighbours, L 48x6, 20 ep, ED 4e5 + recursive s^(2) on 5e4, paired <H> vs hop and vs recursive),
  16844884 d2big (2080Ti; 2x training samples, no recursive checks). Why the variant: ss6 K2 at fixed ViT amplitude misses 44% of the (rare) flips on validation.
- 2026-10-06 11:15: **design change** from the ss6 K2 result (16843547, fixed |ViT|): stacked stored K2 net w_s 5.75(38)e-4 vs
  stored K1 net + exact hop 6.0(1.2)e-5 (exact recursive K2 9(3)e-5). A stacked net trained on the hop factor misses the rare
  flips that matter (held-out b1 disagreement only 0.04%, yet 10x the ED error). Hence: the LAST sign step must always be an
  exact hop; a stored net is only the base under the hop (a lossy base is repaired by the hop, as K1-net + hop shows).
  New iteration-2 plan: SR on frozen H_FN[a1, K1net + hop(a1, T1)] (level-2 local data, ~like it1 but never level 3);
  distil s1 -> D2 = K1net x net2 (jobs above, now only as the base for the next hop); sign step s2 = D2 x sgn(T2 - r[a2, D2]).
- 2026-10-06 11:15: FN reference, ViT guide (|ViT|, binarised ViT phase), M=128 beta 1.2: -0.503612(28) from 16 populations,
  5 s/population on a 2080Ti, population SD 1.1e-4/site (Krylov guides: ~3e-4). Running: 256 more (fnvitB 16844901) and
  M=512 beta 2.4 (16844902) to fix the protocol bias; FN of G1 = (a1, K1net + hop) 32 populations (16844893, ~60 s/pop).
