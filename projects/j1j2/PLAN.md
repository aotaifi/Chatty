# PLAN (J1-J2 FN/Krylov loop) - current state only; history in delta_today.md, experiments/compress_6x6/README.md, RESEARCH_MAP.md
Goal (REFOCUSED by the PI 2026-10-26, after L8): network + p Lanczos hops + lattice fixed-node GFMC as an add-on to any trained
NQS, cheaper in GPU-h than brute-force NQS at equal accuracy at 10x10-14x14, for frustrated magnets where QMC has a sign problem.
The refit loop (store each round in the network) is closed: L8, docs/reports/2026-10-26_l8_loop_round_8x8.md.

## Where we stand (2026-10-25; energies per site; details and paths in README compress_6x6 "Results: ..." sections)
- Physics: the energy-relevant error sits on ~0.1% of sign-violating bonds, at near-node cancellation configurations on
  Marshall-violating J1 bonds (N0, not by construction: G-N0c). A hop computes them exactly (G-A2: tail part 0.33 after
  two hops, bulk part 0.86). Networks cannot learn them in any tested objective (F1, F2, T1s).
- Loop at 6x6: without storage the oracle-free signed two-hop chain from RBM+PP goes 1.95e-6 -> 3.0e-9 in 6 rounds
  (G-A3). With RBM+PP storage one refit loses 1.66e-6 = 1.02 x the round gain (G-tax): GATE SHUT, loop C not run.
- CORRECTION 2026-10-23: the fair competitor (VMC RBM+PP + same Lanczos steps + FN, own sign: 5.57e-6 at one step)
  beats the old fitted loop (8.4e-6). The loop's accuracy edge at 6x6 is not demonstrated.
- 8x8 (M1, translation x flip RBM+PP, results/m1_8x8/): VMC p=0 -0.498674(12), p=1 -0.498923(17); FN-GFMC p=0
  -0.4987565(28), p=1 -0.498946(5) (both QUOTABLE, 2 chains; p=0 dtau/M scans flat; p=1 M=64 check 0.1 sigma).
  p=1 FN is 6.2e-5 below published full-projection RBM+PP (-0.498886) and 2.5e-4 above DMRG (-0.4992). Cost: training
  ~4.1, VMC 1.1, FN p=0 0.4, FN p=1 7.9 H100-h. Projection dominates walker cost (8x8 one hop 4.0 s/walker-tau full,
  0.41 translations).
- One Lanczos step removes 84% of the error at 6x6, 47% at 8x8 (fades with N).
- L8 CLOSED 2026-10-26 (docs/reports/2026-10-26_l8_loop_round_8x8.md): a same-class refit at the VMC energy optimum cannot store
  the hop (fidelity gradient toward the one-hop state = alpha x dE = 0, exact 4x4 rel 5e-9; exact refit R 1.000009). Only the FN gain
  beyond the hop is storable (38% at 4x4, RBM-only), and at 8x8 it is 2.3e-5/site in total and below walker noise with 400
  blocks/chain (cross-chain cosine -0.010(35) with 131k VMC samples). The loop with same-class refit has no route to a
  constant-cost gain per round. PI decision 2026-10-26: refocus on the add-on (Goal above).

## Running (re-arm waits after a clear: until grep -q "VERDICT job <id>" delta_today.md)
| job | run dir | next step per verdict |
|---|---|---|
| 10x10 RBM+PP training, Paderborn 3921516 (cont, 4096 chains, 4500 steps, ~19 H100-h) | Paderborn /scratch/hpc-prf-ppdmrg/hpcalot/rbmppL/runs/rbL10_prod; local /scratch/a/A.Otaifi/rbmppL/prod/rbL10_prod (ANALYSIS_BRIEF.md) | converged -> VMC p=0/1, FN p=0, FN p=1 via experiments/gfmc_fn_lattice/m13b_LxL/launch_m13b_L.sh (LSZ=10, cont); drifting -> resume same RUN |
| 12x12 RBM+PP training, Paderborn 3921519 (cont, 4096 chains, ~66 H100-h) | .../rbL12_prod | converged -> VMC p=0/1, FN p=0; FN p=1 (39 H100-h) only if 10x10 shows a trend worth it |
Both trainings RUNNING since ~2026-10-26 02:00 (cont).

## Budget (takeover decision 2026-10-25)
<= 130 H100-h of the project allocation hpc-prf-ppdmrg for 10x10/12x12 + L8; no single job > 70 H100-h. Express
(personal, 30 H100-h/month) exhausted for October (29.6 used). cont/lowcont estimates say start ~Nov 7 but jobs do run.

## Angles
- ansatz-pioneer (PLAN_ansatz.md): reopened 2026-10-26 with the PI's loop bar (judge a candidate by its per-round refit
  tax inside the loop, gate tax_2 < 0.82e-6 at 6x6); R1 4x6 PASS, R2 pre-registered (commit 2fa6644). The L8 identity (any
  class at its own energy optimum has zero fidelity gradient toward its one-hop state) sent to the angle at the PI's request 2026-10-26.
- conceptual (PLAN_conceptual.md): waiting for the PI's go on the hop-sign check.

## Dead ends (number, run)
- One loop round at 8x8 with same-class refit (L8): hop part has zero fidelity gradient at the energy optimum (exact 4x4,
  /scratch/a/A.Otaifi/l8/idtest4x4); beyond-hop part 2.3e-5/site, unresolved (diag 3973532, 3995351); refit F 3923271 held-out R falls.
- Fitting the tail better, any objective: L2 (9.2e-5, refresh), decade-stratified (T1s/T1s-2 r(-11) 0.07), 13x steps
  (F1 A -6%), 5x5 capacity (F1 B -18%), the FN-relevant bond-ratio objective (F2-1: held-out L_v flat).
- Loop with RBM+PP storage at 6x6: tax_2 1.657e-6 vs round gain 1.629e-6 (runs/gate_tax).
- B0 PP+3x3bf VMC: no gain over PP in 400 SR steps (Paderborn 3806793).
- gamma-FN: 26% of the gap (g0_17411369). ViT refit: 9.2e-5. Tail-trained ViT + hop 1.52e-5.
- Constant-cost storage of loop steps: 7 designs (RESEARCH_MAP 2026-10-13); learned bond field 1.92e-5.

## Cockpit
- STATUS (in progress, high): Refocused on 'trained network + correction steps + walker method' as an add-on; at 8x8 it already beats the best published network energy of its type.
- NEXT (in progress, high): 10x10 and 12x12 network trainings are running; then measure the add-on's energy gain and GPU cost there against published networks.
