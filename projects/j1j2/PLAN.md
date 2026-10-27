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

## ROADMAP (PI, 2026-10-27): back to basics, no brute-force testing
Rule: understand from the code, the records and existing data first. A run only when a derivation names the outcome that would
change a decision, and only with the PI's go. Owners: Pfaffian angle (PLAN_pfaffian.md) owns stage 1 with the PI; the ansatz
angle closes after handing over T-Lfix; the lead keeps PLAN.md and TRIED_IDEAS.md and starts nothing on its own.
**Stage 1 (now): copy the AMPLITUDE of the FN state** (H_FN ground state at fixed Krylov sign s2) - which network and training.
  NOT the problem of getting sign and amplitude of the physical H. Known:
  - Lattice E_FN is amplitude-dominated: at 6x6 the global step's amplitude with the old sign covers ~77% of its gain
    (3.52e-5 -> 1.23e-5 of 5.57e-6; local_product/stageA/6x6/lp6x6.json; also a207289).
  - Where the copy loses (6x6 O2b bond split, f5b7b24): 75% / 84% of the cost on pairs where the copy's SMALL side is far too small
    (cost-weighted median log error -6.0 / -10.6, large side right); a few deep spurious zeros carry it; splicing exact values below
    phi^2 1e-9 halves E_FN (2.98e-5 -> 1.45e-5).
  - Pfaffian angle hypothesis (opinion, ~65%): the zeros are the network's OWN untrained node. The copy is s2|A|, A a signed sum of
    dets; where sgn A != s2 the fit forces |A| -> 0. Derived from cx_t1s.py: the log loss is infinite at A = 0, so training can never
    flip sgn A; the own node stays at its warm start (PP sign, A20). Consistent with F1: longer training +6%, bigger window +18%.
  - Loss (T-Lfix 4x4, STORAGE_PROBLEM.md): FN pair loss L_fix vs log loss: zero-hop tax 6.5x lower (1.28e-5 vs 8.34e-5), after two
    Lanczos steps only 0.77x (inconclusive); the 4x4 tax is bulk-dominated. Ceiling arm invalid (amplitudes reached 1e15).
  - Knobs, derived before any run: log loss infinite and Jacobian d ln|A| = dA/A singular at the network's own node; learning
    rate = damped Gauss-Newton/minSR (lambda0 0.01), nothing more derivable; more parameters cannot move an own node that the loss
    cannot cross.
  Next: the q check on EXISTING data (Pfaffian angle): share q of the near-node cost on pairs with sgn[A(x)A(y)] != s2(x)s2(y),
  forward pass of saved round_1 theta, CPU minutes; q >= 0.5 confirms the own-node cause, q <= 0.1 kills it. Waits for the PI's go.
  FIRST DISCUSSION of the next lead session (PI, 2026-10-27): how to build a network with NO sign of its own (output strictly
  positive, sign only from Krylov), so the log loss never meets a zero of the network itself.
  Then design the amplitude network WITHOUT a node of its own: s2 * e^J, possibly with s2's node bonds at x as inputs (node-bond inputs untried).
  > Lead 2026-10-27 (records + code): a node-free copy EXISTS: the 6x6 ViT refit to phi*(s_2) (refresh_17409419, compress_6x6 README
  'Amplitude refresh') has complex log-psi output (vit_dt.py l.151), so |psi| has no generic zeros; E_FN 9.16e-5 vs O2b real-PP copy
  2.98e-5, failing at the same depths (phi^2 < 1e-9), but its fit stalled (28/200 steps) and its bulk is worse (dE_VMC 1.32e-4 vs PP 6.3e-5).
  Removing the own node is not shown sufficient, not refuted. Derived: H_FN is stoquastic in the s2 frame, so phi = s2 u with u > 0
  (no zeros needed). Proposed controlled test (after q): same PP, complex vs real parameters, |A| with sign s2 (cx_t1s.py l.178 is a
  real signed sum); complex A changes sign around 0 at finite log loss. Trap: from a real warm start d ln|A|/d Im f = 0, so Im f needs a
  small random start. Node-bond inputs set FN's own 1/(H_xx - E + V_sf) dip, but on the costly pairs ln phi - ln guide ~ 0: not the cost.
**Stage 2: division of labour.** Network = amplitude only, sign from Krylov (s2/K1). A Pfaffian for the amplitude only makes sense
  if its node equals s2's.
**Stage 3, only after stage 1: is the Krylov sign enough?** Prior: the energy error was amplitude, not sign.

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
- pfaffian-angle (PLAN_pfaffian.md, opened 2026-10-27 by the PI): can a pair-product network copy the FN state in rare configurations; paper discussion with the PI only, no runs yet.
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
- STATUS (in progress, high): Back to basics: the next question is which network copies the fixed-node state's amplitude; current evidence points to spurious zeros from the network's own sign.
- NEXT (open, high): A check on data we already have (minutes of CPU) tests whether those zeros come from the network's own sign; waits for the PI's go.
