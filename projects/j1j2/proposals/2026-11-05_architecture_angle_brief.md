# Angle brief: an amplitude network built for the Jacobi loop (lead, 2026-11-05; start after M5' reads out)

You are an angle lead (root CLAUDE.md §0). Plan file: projects/j1j2/PLAN_architecture.md (create it). Report to the project lead
when the angle closes or a result changes the loop's plan. Read first: projects/j1j2/CLAUDE.md (agreement 1-8), PLAN.md
("What we understand", Queue line "Amplitude network for the 14x14 loop"), PLAN_pfaffian.md (Θ v2, Test N, R5/R6, M5'), board 20-23.

## Question
Which network class can carry the loop's amplitude at 14x14? Required, both:
(C1) a two-swap amplitude ratio at a constant factor over a plain walker step (low-rank or local update), because one Jacobi step needs
     ~(2N)^2 ratios per walker move (board 23; pair products measured 5-11x, generic nets ~2N x, CNN-MPS ~2N x:
     proposals/2026-11-05_cnn_mps_twoswap_cost.md);
(C2) the round's correction Delta* = c f_J[B] within reach of a refit (PP tangent space holds 0-4%, Θ v2; finite-weight RBM features
     eps^2 0.95, Test N; a second PP stores r 0.49 vs 0.05, R5).
Note the shape of the target: f_J(x) is a ratio of sums over CONFIGURATION-space neighbours (one swap away), not a real-space
aggregation. Real-space aggregators (CNN, ViT, RBM) change everywhere under one swap; algebraic objects (determinants, Pfaffians)
update by matrix lemmas.

## Gate before starting
M5' (PP+RBM, parameter-matched control; readout ~11-05 15:30, PLAN_pfaffian.md). "Structure" -> this angle is motivated as written.
"Parameter count" -> narrow the angle to cost-efficient parameter growth (C1 only).

## Method (paper first; CPU only; ~2 days)
1. Write C1 and C2 as equations (cost per two-swap ratio; tangent/refit reach of Delta* in the bond metric).
2. Candidates, each checked on paper against C1 and C2 before any code:
   - sums of K pair products with multi-determinant fast updates (quantum-chemistry table method; Clark, Morales et al.,
     J. Chem. Phys. 135, 244105 (2011): cite only after reading);
   - a dressing that changes only the swapped sites' pair amplitudes (Queue 'local dressing'), keeping the update low-rank;
   - Pfaffian sums; anything the mathematics of f_J suggests (do not start from published NQS architectures; two lines on any
     published method's assumption and whether our problem breaks it).
3. Best candidate: exact 6x6 test on existing round-5 tables (no GPU): share of Delta* within reach (Θ v2 metric, calibrated as in
   PLAN_pfaffian.md) and measured two-swap cost vs plain.
Prediction (lead): K > 2 pair products keep raising the stored share (~55%); something genuinely new beats that (~25%).
Kill: no candidate passes C1 and C2 on paper. Second opinion (Astra, CLAUDE.md point 8) before any GPU proposal.

## Added 2026-11-08 (PI + lead): gated bulk/tail hybrid as a first-class candidate
G(x) = B_PP(x) exp[w(x) g(x)], w(x) a smooth switch on |B_PP(x)|^2 in the deep decades (free from B_PP), g a CNN or other expressive net
evaluated only where w > 0. Cost: g only on the fraction of the (2N)^2 Jacobi endpoints in the tail (measure it on pool8 draws first).
Motivation: 8x8 R1 fails on ~0.8% deep x next to much larger blocked neighbours (PLAN_pfaffian R1 outlier diagnosis); literature has no gated
|psi| hybrid (proposals/2026-11-08_tail_fitting_literature.md). Gate: build only if the R1 (a)/(b) separator says CLASS (b); if sampling (a),
oversampling fixes it and the hybrid is not needed. Paper work (cost count, switch smoothness vs the Jacobi factor, tail training data) may
start now; no run before the (a)/(b) verdict. M5' result: structure matters (two PPs >> PP+RBM at equal parameters).
