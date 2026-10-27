# Future angles (user ideas, 2026-10-06) — to discuss later

1. **Bad start.** Can the loop start from a very bad guide, e.g. an NQS with large excited-state overlap or a pure excited state, and still reach the ground state?
   First take: FN is imaginary-time projection inside the guide's sign class, so excited components in the same symmetry sector should be projected out (the exact loop already works from the poor J2=0 amplitude + Marshall). A guide in a different symmetry sector should stay trapped there. Testable exactly on 4x4/20 sites.
   > DONE 2026-10-27 (index; correction to this stale list): done 2026-10-06 as angle 1 (bad start), see CLAUDE_TAKE.md takeover log and RESEARCH_MAP section 2 'Bad starts' (bad_start/): any in-sector GS content reaches the GS, other symmetry sectors trap.
2. **Sign net as a step function.** The sign is +-1, a discontinuous function: is learning it a bottleneck?
   First take: it is trained as a classifier (logit, BCE), so the step shape is natural; the hard part is the decision boundary and where the training data is. Evidence: 4x4 plateau is a training-distribution limit, not capacity; 6x6 K1 large nets match/beat the recursion; the exact hop repairs net errors. Open: boundary complexity vs N (8x8).
   > DONE 2026-10-27 (index): done 2026-10-06 as angle 2 (sign net as step function), TRIED_IDEAS A11: not a bottleneck (exact GS sign learnable at N=36, one exact hop removes 96-99% of net error); boundary complexity at 8x8 remains the open part (RESEARCH_MAP 3s / sign_learnability).
3. **Amplitude net on H_FN.** Is learning the FN amplitude equally hard?
   First take: yes, this is the main bottleneck (see AMPLITUDE_LEARNING_LEDGER.md): 6x6 iteration-1 amplitude gain at the noise floor (~1e-4), 4x4 sampled loop limited by a sampling floor, 8x8 updates failed.
   > ANSWERED/CLOSED 2026-10-27 (index): answered in AMPLITUDE_LEARNING_LEDGER.md; the H_FN refit loop was tried 10-01..10-04 (rung 2 results/learning_ladder_rung2/RUNG2_VERDICT_2026-10-04.md, 4x4 sampled loop stable 150 its, error ~N^-0.75) and CLOSED 2026-10-26 (L8, docs/reports/2026-10-26_l8_loop_round_8x8.md; PLAN.md 'Open question').
