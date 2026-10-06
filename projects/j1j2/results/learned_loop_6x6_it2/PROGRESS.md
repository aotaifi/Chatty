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
- 2026-10-06 11:45: FN references (it2_fn, M=128 beta 1.2 burn 0.4 unless noted; per site):
  ViT guide 256 pops (vitB) -0.503676(10); ViT guide M=512 beta 2.4 burn 0.8, 32 pops -0.503654(10) -> protocol bias of M=128 ~ -2e-5 (1.5 sigma);
  G1 = (a1, K1net + hop(a1, T1)) 32 pops -0.503615(33), median -0.503632, 3/32 below E0 -> iteration-1 guide in stored+hop form is
  6(3)e-5 ABOVE the ViT guide under the same protocol. SR jobs: a2main 16844974 (A40, 8192+8192/step, pending), a2small
  (2080Ti, 3072+4096/step; first try 16844975 OOM in cuBLAS autotune -> 16845080 with XLA autotune off). fnK2vit 16845042:
  FN of (|ViT|, K1net + hop(|ViT|, T=-7.74)), the no-learning guide with ED w_s 6e-5.
- 2026-10-06 12:10: CRN pairing (same seeds) of FN populations between guides gives corr ~0.2 only (K2vit vs ViT, 16 seeds): not useful;
  matched-protocol unpaired comparison + many populations instead. Partial fnK2vit (18 pops) -0.503652(50).
  Submitted oracle FN diagnostics (16845135): (|psi0|, sgn psi0) sanity -> E0; (|ViT|, sgn psi0) = floor of any sign improvement at
  the ViT amplitude; (|psi0|, ViT sign) = cost of the ViT signs alone. Decides whether signs or amplitude limit E_FN near the ViT.
  K3 sign step at fixed |ViT| on the ss6 stacked K2 base (16845101).
- 2026-10-06 12:20: **distillation d2big (16844884, 2080Ti, 1.37 GPU-h: labels 0.71, train 0.30)**: 6.5e6 training states (82k tempered b0.3 samples + nbrs), flip frac 11%,
  val error 4.6% but 43% on flips. ED (4e5): stored D2 = K1net x net2 w_s 4.90(35)e-4; hop form (ii) K1net + hop(a1,T1) w_s 6.5(1.3)e-5
  (recursive s^(2), it1: 8.0(2.0)e-5 on the first 2e5 of the same samples; (ii) 6.5e-5 there too). Variational energy on 16384 |a1|^2 samples:
  <H>_{a1,(ii)} = -0.503590(42); stored D2 +2.30(63)e-4 higher. Same pattern as ss6 K2: the stacked net is a lossy base, the hop form is
  as good as the recursion. D2 is used only as the base of the next hop. Submitted K3 at a1 on base D2 (16845259, A40).
- 2026-10-06 12:45: **ORACLE FN DIAGNOSTICS (16845135, M=128 beta 1.2, 32 pops each; decisive):**
  | guide (amplitude, sign) | E_FN/site |
  |---|---|
  | (|psi0|, sgn psi0) sanity | -0.5038096539 exactly, zero variance (= E0: FN code validated) |
  | (|ViT|, sgn psi0) exact signs | -0.503666(52) |
  | (|psi0|, ViT sign) | -0.503731(15) |
  | (|ViT|, ViT sign) ViT guide, 256 pops | -0.503676(10) |
  | (|ViT|, K1net + hop T=-7.74) K2vit, 48 pops | -0.503657(25) |
  | (a1, K1net + hop(a1,T1)) G1, 32 pops | -0.503615(33) |
  With the ViT amplitude, even EXACT signs do not lower the lattice-FN energy below the ViT guide (+1.0(5.3)e-5). The exact amplitude with
  the ViT signs is 5.5(1.8)e-5 lower. Lattice FN moves frustrated (sign-violating) edges to the diagonal weighted by a(y)/a(x), so near the
  ViT the remaining FN error is set by the AMPLITUDE on violating edges; sign steps cannot beat the ViT, only a better amplitude can.
  Follow-up (16845413): 96 more pops of (|ViT|, exact) and (a1, exact) = amplitude quality of iteration 1 measured with oracle signs.
- 2026-10-06 13:00: (|ViT|, exact signs) 96 more pops: -0.503687(23) (128 pops total ~ -0.503682(21)) = ViT guide -0.503676(10) within 0.3 sigma. Confirmed.
- 2026-10-06 13:00: d2main (16844883, A40) ED on 5e4: recursive s^(2) w_s 1.0(0.45)e-4 = hop form (ii) 1.0e-4 on the same samples (identical errors) -> (ii) is a faithful replacement of the recursive iteration-1 sign.
- 2026-10-06 13:00: **amplitude SR, iteration 2** on frozen H_FN[a1, (ii)]:
  a2main (16844974, A40, 8192+8192 fresh/step): steps 1-3 all rejected by the independent validation gate (dL/site +4e-5, +1.2e-4, +1e-4, SE 1-2e-4 at full step; halves +1e-6..+4e-5) -> stop, theta unchanged, 0.45 GPU-h. Local data 210-270 s/step (level 2).
  a2small (16845080, 2080Ti, 3072+4096/step): 6 of 8 steps accepted so far, val dL per step -1e-5..-9e-5 with SE 1-5e-5 (cum -1.65e-4/site is selection-biased).
  The frozen-H_FN objective of G1 has no resolvable descent: its maximum gain is <H>_G1 - E_FN[G1] = -0.503590(42) - (-0.503615(33)) = 2.5(5.3)e-5/site.
- 2026-10-06 13:00: exact 6x6 loop (closed_fn_krylov_6x6/sym6x6_plain_history.jsonl) near eps_FN ~3e-4: E_FN - E0 contracts by ~11%/iteration,
  i.e. ~1.5e-5/site per EXACT iteration at the ViT level (E_FN - E0 = 1.3e-4). Beating the ViT by 2 sigma (~7e-5) needs ~5 exact iterations
  (2-3 with Anderson); one learned iteration resolves only ~1e-4.
- 2026-10-06 13:20: (a1, exact signs) 96 pops: -0.503652(29) vs (|ViT|, exact) -0.503682(21): the iteration-1 learned amplitude is NOT better than |ViT|
  under FN with oracle signs (+3.0(3.5)e-5; std(log a - log|psi0|) 0.061 vs 0.048).
- 2026-10-06 13:20: sign steps on stored bases (robust T on 32768 samples, held-out 16384):
  K3a1 = D2 + hop(a1, T=-3.55) (16845259, A40 0.33 GPU-h): ED w_s 4.25(1.0)e-5 (best so far; ViT 1.3e-4); held-out dH = -3.98(1.05)e-4 vs base D2;
    <H> - E_ViT(VMC) = +2.2(1.3)e-5 -> <H> = -0.503632(25). T: raw -5.14, smooth -3.55, halves -6.5/-5.1 (raw), bootstrap 68% [-6.4,-5.1];
    held-out dH differs by < 1e-5 between T = -6.37, -5.14, -3.55: the curve is flat, so the T instability of it1 is energetically harmless.
  K3vit = stored K2(ss6) + hop(|ViT|, T=-2.3) (16845101, 2080Ti): held-out dH -5.07(1.12)e-4 vs base; <H> - E_ViT = +2.9(1.8)e-5.
  FN 64 pops each submitted: K3a1 16845762/3, K3vit 16845764/5.
- 2026-10-06 13:30: a2small final (1.19 GPU-h on 2080Ti; 6 accepted SR steps, stop after rejections at trust 0.0025), independent
  check on fresh 16384 + 16384 samples: dL(frozen H_FN)/site = -7.4(5.3)e-5 on a2^2 samples, -0.2(1.8)e-5 on a1^2 samples;
  RMS(delta log a) = 0.006. Unresolved -> the amplitude step of iteration 2 is at the noise floor (as in iteration 1); the
  per-step validation sum -1.65e-4 was selection bias of the gate. a2 = params_a2small.npy is used for completeness.
  K3vit ED: w_s 4.25(1.0)e-5 (same as K3a1). Submitted sign step G2 = D2 + hop(a2) (16845923) and oracle (a2, exact signs) (16845924).
- 2026-10-06 14:10: coordinator: it1 recursive guide, 64 pops M=128: -0.503708(29), median -0.503654(42), 17/64 below E0 (skew -1.44) ->
  M=128 population-control bias. Submitted (a) ViT guide with the SAME seeds as the K3 runs (73001-064, 74001-064) M=128 and
  M=512 beta 2.4 seeds 75001-032 (16847800); (b) K3a1 at M=512 beta 2.4 burn 0.8, seeds 75001-032 (16847801/2).
- 2026-10-06 15:00: FN M=128 beta 1.2, 64 pops each: **K3vit = (|ViT|, stored K2 + hop(|ViT|, T=-2.3)) -0.503725(21)**, median -0.503706, 17/64 below E0;
  K3a1 = (a1, D2 + hop(a1, T=-3.55)) -0.503629(18), median -0.503644, 6/64 below E0. ViT guide same protocol (272 pops) -0.503672(10), 40/272 below E0.
  K3vit - ViT(VMC) = -7.1(3.0)e-5 (2.4 sigma), K3vit - ViT(FN, M128) = -5.3(2.3)e-5. Below-E0 fraction 27% vs 15% for the ViT guide -> must be checked
  at M=512 before any claim. Submitted K3vit M=512 beta 2.4, 44 pops (16848xxx).
- 2026-10-06 15:15: **iteration-2 guide G2 = (a2, D2 + hop(a2, T=-1.95))** (sG2 16845923, 2080Ti 1.13 GPU-h): ED w_s 4.5(1.1)e-5; <H>_G2 - E_ViT = +1.0(1.1)e-5
  (paired, 16384 held-out) -> <H>_G2 = -0.503644(24). Held-out dH of the sign step is the same within 4e-7/site for T = -1.95, -3.55, -6.37
  (T bootstrap 68% [-3.6, -1.6] smoothed): the curve is flat, so T is ill-determined but irrelevant. std(log a2 - log|psi0|) 0.062 (ViT 0.048).
  FN 64 pops submitted (16848136/7, seeds 73001-064 = paired with vitP).
  Paired ViT guide (same seeds 73001-064, 74001-064) M=128: -0.503666(15), 25/128 below E0. ViT M=512 beta 2.4: 32+32 pops -0.503654(10), -0.503677(11).
- 2026-10-06 15:20: coordinator note (4x4 net-start: learned refresh realises 0.3-0.5 of the frozen-H_FN gain; SR stopping at first rejection).
  it2_amp already continues after rejections (trust halved, stop after 3 consecutive or trust < 0.0025): a2small accepted 6 of 9 steps.
  On 6x6 the limit is not the optimiser but the available gain: <H>_G1 - E_FN[G1] = 2.5(5.3)e-5/site, and the ideal loop contracts by
  ~0.85-0.89/iteration (4x4 net-start, 6x6 exact loop) = ~1.5e-5/site per iteration at the ViT level, below the FN referee resolution (~2e-5).
  A stronger optimiser cannot make one iteration visible here; no further amplitude variants run.
- 2026-10-06 16:00: **FINAL**: G2 (iteration 2) E_FN -0.503687(23) (64 pops, median -0.503661, 16/64 < E0), = ViT within errors.
  K3vit M=128 -0.503725(21) (17/64 < E0) does NOT survive M=512: -0.503699(29) (42 pops, one at -0.50477), same-seed paired vs ViT guide
  M=512 +0.6(1.8)e-5. K3a1 M=512 vs ViT +0.1(1.8)e-5. (a2, exact signs) -0.503649(23). Verdict: TIE; iteration 3 not run (no resolvable
  amplitude gain; oracle shows amplitude limits). See IT2_VERDICT_2026-10-06.md, fig_it2_6x6.pdf, summary_fn.json.
