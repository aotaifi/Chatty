# PLAN_conceptual (angle lead, started 2026-10-22; project lead's PLAN.md points here)
Angle: the physics behind the loop's five blocks (guide amplitude, one hop, lattice FN, refit, sign step), discussed
with the PI. Deliverable: conceptual/MEMO.md (< 2 pages): per block the physics picture, <= 5 ranked hypotheses
(confidence, kill test, cost, the existing number each rests on). Ends when the PI agrees it is useful.
Budget: ~2 days of discussion; only small exact diagnostics on existing 6x6 tables (<= a few slice-h, via workers).
No production runs; do not touch O2b 17431886 or B0.

## Working picture (hypotheses, not claims)
- Tail = nodal region: near-node amplitudes are cancellations psi(x) = sum_y H_xy psi(y)/(E - H_xx); frustration
  (stripe-tilted, Neel-disordered configurations) is what makes the cancellation and the non-Marshall sign.
  Rests on T0-phys (t0phys_17410698): J1 Marshall violation 0.40 on X_0.1 vs 0.013 bulk; M_stripe^2/M_Neel^2 5.7x.
- The hop computes that cancellation sum explicitly, which a smooth network cannot; the refit throws it away again
  (fit tax ~6e-6/round, O2b 17431886 rounds 0-2).

## Open questions for the discussion
1. Is the O2b fit tax concentrated on the same frustrated, Marshall-violating bonds (fit floor and sign problem = one
   object: node placement)?
2. How good is the trained Pfaffian's own sign sign(Pf F) on W_2 and the tail (a sign prior for O3)?
3. Does the FN target carry information VMC energy training does not reach (B0 decides; what physics predicts).

## Diagnostics (none submitted yet)

## Status
- 2026-10-22: plan file created; greeting sent to the PI.
