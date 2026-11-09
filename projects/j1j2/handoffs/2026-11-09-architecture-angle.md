# Handoff 2026-11-09: architecture angle (which network can carry the loop at 14x14) — closed

## What was done
- Two conditions written as equations (PLAN_architecture.md):
  - C1, cost: a Jacobi step per walker move needs ~d^2 two-swap ratios, so the per-move cost must stay a constant factor over a plain step.
  - C2, reach: a refit must store the round's correction, measured by its differences across blocked bonds (bond metric, board 5).
- Candidates checked:
  - Gated CNN hybrid on small |B|^2 fails C1: 6.2% of two-swap endpoints lie in decades <= -7 at 8x8 (arch_tail).
  - Local window add-on (CPS / plaquette / CNN-Jastrow) fails C2: the correction's bond difference is not local, explained share <= 0.13 (arch_hloc).
  - Pfaffian sums: up-up and down-down blocks sit at a bilinear saddle (derived, not run).
  - Backflow and the two-body Jastrow fail on earlier measurements.
  - Tied symmetric PP sum: no cost gain (Astra).
  - Dimer-subspace effective model: about 1e-2 per site off in the literature (1301.4492).
- Exact identity, reviewed on 4x4: H acting on a pair product is a sum of pair products with two sites relabelled. One Lanczos step of a
  translation-symmetric PP is 1+4m projected PPs. Each of those still needs all N translation images, so the cost is no better than
  general f plus one Lanczos step.
- Side finding: pp2_lemma's dedup keys are one uint64 and collide above 64 sites (handed to the Pfaffian angle).
- Queued with the lead: round 6 with a third PP at 6x6, testing why the stored share falls from 0.49 to 0.25.

## Why / alternatives rejected
- Every rejected class failed a stated equation, with a number from a run or a reviewed derivation, not a judgement call.
- The angle lead's (my) claim "5 PPs = 5x cost" was wrong and was retracted after Astra's review.

## Three things to understand
1. Only ratios psi(x')/psi(x) are needed. A swap changes one row and one column of M, so a ratio is a 2x2 (4x4 for two swaps)
   determinant from M^-1. That is why determinant networks pass C1 and generic networks (~2N per ratio) do not.
2. Fixed node with psi_1 = (1 + alpha(H - E))B as the guide uses a once-improved node. psi_1 is evaluated at every configuration
   (computed on the fly) but never re-applied, so the node does not keep improving during the run (that would be released node).
3. The failures sit in different terms: a gate on |B|^2 is extensive while the failure is local (C1); the round correction is non-local (C2).
   So sums of pair products remain the only cheap route.

Files: PLAN_architecture.md; TRIED_IDEAS AR1-AR5; /scratch/a/A.Otaifi/ansatz_runs/arch_{tail,hloc,paper,sym6}; /scratch/a/A.Otaifi/astra/arch_tied/ANSWER.md.
Still running at writing: arch_sym6 (6x6 symmetric-f accuracy, PI's request; for the record, not a gate).
