# Next major calculation, final after two review rounds (2026-10-23; supersedes v1)
Reviewer (independent Opus): round 1 reject v1 (its bar passed at round 0); round 2 accept with changes (~75%).

## What changed the plan
- Fair competitor at 6x6 = VMC RBM+PP (A) + p Lanczos steps + FN, own sign (results/rbmpp_6x6/runs/table_A.json):
  p=1 E_FN 5.57e-6; p=2 <H> 2.49e-6; p=3 <H> 1.07e-6. Our fitted loop (1 hop, oracle sign) 8.4e-6. No loop edge shown.
- The loop needs its refit to improve the network beyond VMC at equal hops; the two-hop refit tax (1.9e-6, PP+bf) is
  already close to E_FN(K_2(A)); reviewer puts that at <= 15%, lead at ~25%.
- The goal's number is GPU-h at 8x8-14x14; no published 8x8 cost exists (J1J2_BENCHMARKS_COST.md).

## M1 (the next major calculation): cost-accuracy at 8x8 for RBM+PP + p Lanczos + FN-GFMC
0. Literature first (CLAUDE.md §4): published Lanczos + FN numbers at 8x8+ (Hu-Becca-Parola-Sorella 2013, Becca/Sorella
   group, Nomura-Imada 2005.14142); M1 only adds what is missing (FN-GFMC on RBM+PP, the GPU-h axis).
1. GFMC validation at 6x6: walkers reproduce E_FN(A + 1 hop) = 5.57e-6 (time-step and population-control bias
   measured, two seeds) before any 8x8 number.
2. Cost per walker step with 1- and 2-hop RBM+PP guides at 8x8 and 14x14 (< 0.5 H100-h).
3. 8x8: RBM+PP trained by VMC on H100, then p = 0, 1, 2 Lanczos + FN-GFMC; energies against the 8x8 bar (RBM+PP VMC
   -0.498886, rel 3.5e-4) on one GPU-h axis that counts training.
These walkers are O4's infrastructure, so the work carries over if the loop later passes.

## Gate for the loop (C), cheap, in parallel
- A2 null control with the REAL fit residual (a_2 - phi_1) spliced into tail-only and bulk-only decades; tax_2/tax_1 each.
- N0 control: the top-d^2 set of a smooth perturbation unrelated to the fit; does it also land on low cancellation c?
- A3 fit-free two-hop chain from A (own sign): gain per round.
- One two-hop refit tax measured on RBM+PP (~1 H100-h).
- Run C only if A3's gain per round > 2 x that tax. C's PASS: E_FN(K_2(a_4)) <= 0.5 x E_FN(K_2(A)) AND below
  E_FN(K_3(A)), baseline a_0 = A itself, same FN solver and coefficient rule in every arm, fixed steps per round with a
  falling held-out loss, im_weight guard (stop > 1e-3), two seeds; sign update only after a fixed-sign pass.

## Parallel angles
- ansatz-pioneer: an ansatz that stores the near-node region (what the refit cannot); this is now central.
- conceptual: physics of the blocks with the PI.
