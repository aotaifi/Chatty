# PLAN (J1-J2 FN/Krylov loop) - current state only; history in delta_today.md, experiments/compress_6x6/README.md, RESEARCH_MAP.md
Goal (REFOCUSED by the PI 2026-10-26, after L8): network + p Lanczos hops + lattice fixed-node GFMC as an add-on to any trained
NQS, cheaper in GPU-h than brute-force NQS at equal accuracy at 10x10-14x14, for frustrated magnets where QMC has a sign problem.
The refit loop (store each round in the network) is closed: L8, docs/reports/2026-10-26_l8_loop_round_8x8.md.
> CORRECTION 2026-10-27 (index): the 10x10-14x14 add-on test is NOT done: the 10x10/12x12 trainings were cancelled and the Paderborn budget revoked 2026-10-26 (delta_today.md 19:10; checkpoints under rbL{10,12}_prod/backup_paderborn), re-opened only for the LP test (see 'Budget'). The Goal is a target, not a result; the deliverable at 10x10-14x14 remains pending, subject to the lead's budget (TRIED_IDEAS.md Contradictions item 12).

## Where we stand (2026-10-25; energies per site; details and paths in README compress_6x6 "Results: ..." sections)
- Physics: the energy-relevant error sits on ~0.1% of sign-violating bonds, at near-node cancellation configurations on
  Marshall-violating J1 bonds (N0, not by construction: G-N0c). A hop computes them exactly (G-A2: tail part 0.33 after
  two hops, bulk part 0.86). Networks cannot learn them in any tested objective (F1, F2, T1s); none of those tested a log-free pair loss (see 'Open: storage near the nodes').
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

## Open question: can the loop be rescued? (PI 2026-10-26: 'I was hoping to rescue our loop')
- Derived: refitting to the one-hop state is one SR step (gradient = alpha x energy gradient at any point), so a same-class loop
  gains beyond VMC only from the FN projector's information beyond the hop (2.3e-5/site at 8x8 after a hop; FN p=0 vs VMC p=0 8.3e-5).
- Test 1 DONE (exact 4x4, /scratch/a/A.Otaifi/l8/loop4x4): refit to FN(a) and node-weighted refit make E_FN WORSE every round
  (RBM -0.5275259 -> -0.5273929); with a hop the RBM class gains 27% of the add-on-to-E0 gap in 5 rounds, but sign-broken targets
  gain as much (control fails): the gain is E_FN(a*h_a) being non-stationary at the E_v optimum, not storage. Storage loop closed unless T-Lfix passes (STORAGE_PROBLEM.md).
- Test 2 (exact 4x4): training the network for E_VMC(a*h_a) closes 79%/69% of the add-on-to-E0 gap (loop 27%/9.9%), but it is
  CLOSED without a test (PI 2026-10-26): any explicit Krylov step over configurations needs the two-hop neighbourhood per sample
  (137x plain sampling at 8x8, growing ~N) while its gain per site fades. Pilots 4010838/4012781 cancelled before start.
> CORRECTION 2026-10-27 (index): 'closed without a test' is wrong as worded. An exact 4x4 test EXISTS and was run 2026-10-26: minimising E_VMC(a*h_a) closes 79% (RBM-only) / 69% (f-only) of the add-on-to-E0 gap, direct E_FN minimisation 86% / 75% (/scratch/a/A.Otaifi/l8/loop4x4/hop_rbmonly, hop_fonly; delta_today.md 2026-10-26 15:58 and 16:00, commit 2299a2c). What the PI closed on 2026-10-26 (delta_today 19:10) is any scaled test (6x6/8x8), on COST: O(N^2) per sample, two-hop neighbourhood per sample, 137x plain sampling at 8x8. Not run at 6x6/8x8; the sampled-SR pilots 4010838/4012781 were cancelled before start.
- Literature: the loop is self-healing DMC (Reboredo-Hood-Kent arXiv:0808.3417) on a lattice. Caffarel et al. arXiv:2609.01301: noise-free,
  the FN energy RISES to a wrong node for any finite basis; cure = node-weighted update. Test 1 got variant (iii), node-weighted fidelity.

## Open: storage near the nodes (PI + ansatz lead, 2026-10-27) - PI direction: resolve storage first
Details, cause equations, tried rows and status: STORAGE_PROBLEM.md. Three issues:
1. Weighting (CORRECTED 2026-10-27, independent review verified by the ansatz lead and the lead, c2le_gtax.py l.426-464): the G-tax
   copy loss was NOT psi^2-weighted; it draws configurations equally per decade and violating bonds proportional to |H| phi(y)/phi(x),
   already the FN-shaped small-error weight. So weighting does not explain the G-tax. The pair loss L_fix is the exact upper bound of
   E_FN at frozen phi, so storing with it converges to a stationary point of E_FN over the class: issues 1 and 3 are one issue.
2. Support: walkers do not reach the node; flatter importance g = |psi_T|^beta (TRIED_IDEAS F1, parked, never run).
3. Gradient denominator in E_FN training (C15 failed sampled at 8x8); the factorised a_k e^r is the known handle (B10/B11).
Running (ansatz lead): T-Lfix (STORAGE_PROBLEM.md, incl. 'Independent review 2026-10-27', 61e539c). Deciding number now
R_ceil = tax(network trained directly on exact E_FN) / tax(warm start), ceiling arm = stage (c) code (cfc2681), exact 4x4 running;
R_ceil >= 0.7 means the network class is the floor (an ansatz question: next exp(J) x sum_k Pf[M_k]); a pass at 4x4 and 6x6 reopens
the storage loop; kill keeps the add-on Goal. Pointers: equation board https://claude.ai/artifact/LwUGJoTxGJku3k6mTVgMVP;
../../handoffs/2026-10-27-fn-error-weights.md; PLAN_ansatz.md (angle record); local entry folder /project/theorie/a/A.Otaifi/j1j2/
(CLAUDE.md; PLAN.md and delta_today.md symlink here). Candidate architecture if the loss is not enough: exp(J) x sum_k Pf[M_k]
(lead + PI, 2026-10-27; not done for spin models per literature check).

## Step back 2026-10-27 (second, with TRIED_IDEAS.md) - SUPERSEDED by the PI's 'resolve storage first'
1. Verdict: REFRAME. Stop searching for a sign-repair method; write up what the project measured: three walls that stop learned sign
   repair for J1-J2 (support, precision, generalisation), plus the 8x8 add-on number -0.498946(5).
2. Thought: the PI's chicken-and-egg is already measured. The Hamiltonian labels a wrong sign at any configuration we check (Krylov
   verifier works, AVSL-09/12), but (a) FN walkers never visit the configurations worth fixing (0 of 198 at 20 sites, AVSL-08);
   (b) near nodes the label is a near-cancellation that networks cannot hold even with exact data (6x6 refit tax 1.02x);
   (c) no compact rule generalises (direct learned rules make E_FN worse, AVSL-10/12; profitable configurations grow 82 -> 1.6e7
   orbits from 16 to 36 sites, RESEARCH_MAP 3f).
3. Cheapest next: a paper skeleton from existing numbers (no compute; paper-writing skill), after the PI agrees.

## Step back 2026-10-27: proposal RETRACTED (already tried)
Training the network on its own lattice FN energy was done 2026-10-01/02 (STATUS.md 'Exact 4x4 E_FN Hellmann-Feynman', '8x8 mixed
E_FN force/QGT'): exact 4x4 force works; sampled 8x8 force passed its surrogate gate (3537066) but the physical FN energy got WORSE
(-31.8551 vs -31.9012, 3549023/24); finite-walker force SNR collapses near the fixed point (walkers for cosine 0.7: 8e4 -> 2.5e8).
Lesson: j1j2 records are split across PLAN.md, RESEARCH_MAP.md, STATUS.md, AMPLITUDE_LEARNING_LEDGER.md, FAILED_ROUTES.md and
delta_today.md; check all before proposing (a tried-ideas index is being built).

## Running
- Nothing on Paderborn. Cancelled 10x10/12x12 trainings: checkpoints in /scratch/a/A.Otaifi/rbmppL/prod/rbL{10,12}_prod/backup_paderborn.

## Budget
Paderborn re-opened by the PI for the LP test (2026-10-26, 'use wisely'): lead's ceiling 15 H100-h, single job <= 6 H100-h, cont.

## Angles
- ansatz-pioneer: CLOSED 2026-10-27 (its own report): the J2-undone pre-image passes the loop gate only with exact stored
  magnitudes (tax 2.4e-8); a baseline-size network fails by 4.5x (2.58e-5 / 7.4e-6). PLAN_ansatz.md holds the record.
- conceptual (PLAN_conceptual.md): waiting for the PI's go on the hop-sign check.

## Dead ends (number, run)
- Training the guide on its lattice FN energy (sampled mixed HF force + QGT, 8x8, 2026-10-02): physical E_FN worse by +0.046 total
  (3537066 -> 3549023/24; STATUS.md); exact 4x4 version works, the sampled force does not.
- Local product of Krylov factors (LP, per bond / site / plaquette): 6x6 E_FN-E0 3.5247e-5 vs G 5.5738e-6 (KILL); near-node pole
  |psi_LP/psi| 9.7e6 (local_product/stageA/6x6/lp6x6.json; report docs/reports/2026-10-27_local_product_krylov.md).
- Exact iterated refit loop at 4x4 (refit to FN(a), node-weighted): E_FN worse every round (l8/loop4x4); hop-aware training works
  at 4x4 (79% of the gap) but costs O(N^2) per sample on global guides (PI: closed without a test).
> CORRECTION 2026-10-27 (index): hop-aware training at 4x4 is a TESTED exact result (79% / 69% of the gap, /scratch/a/A.Otaifi/l8/loop4x4/hop_*), closed by the PI on cost (O(N^2) per sample on global guides), not 'without a test'; see the Test 2 correction above.
- One loop round at 8x8 with same-class refit (L8): hop part has zero fidelity gradient at the energy optimum (exact 4x4,
  /scratch/a/A.Otaifi/l8/idtest4x4); beyond-hop part 2.3e-5/site, unresolved (diag 3973532, 3995351); refit F 3923271 held-out R falls.
- Fitting the tail better, any objective: L2 (9.2e-5, refresh), decade-stratified (T1s/T1s-2 r(-11) 0.07), 13x steps
  (F1 A -6%), 5x5 capacity (F1 B -18%), the FN-relevant bond-ratio objective (F2-1: held-out L_v flat).
- Loop with RBM+PP storage at 6x6: tax_2 1.657e-6 vs round gain 1.629e-6 (runs/gate_tax).
- B0 PP+3x3bf VMC: no gain over PP in 400 SR steps (Paderborn 3806793).
- gamma-FN: 26% of the gap (g0_17411369). ViT refit: 9.2e-5. Tail-trained ViT + hop 1.52e-5.
- Constant-cost storage of loop steps: 7 designs (RESEARCH_MAP 2026-10-13); learned bond field 1.92e-5.

## Cockpit
- STATUS (in progress, high): The loop failed at the copying step; today's analysis traced it to how network fits weight errors next to the nodes, against how the fixed-node energy weights them.
- NEXT (in progress, high): A matched test of a fit weighted the way the fixed-node energy weights errors (4x4 now, 6x6 next) decides whether the storage loop reopens.
