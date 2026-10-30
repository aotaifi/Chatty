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

## OPEN QUESTION (the one, CLAUDE.md point 4; set by the PI 2026-10-29)
Can the stored network carry the local, Jacobi-shaped correction cheaply, so that the copy loses < ~1e-6/site per loop round
near the fixed point (today 9.4e-6)? The loop stays the method (PI): stored sign + amplitude keep the walker cost at plain-network
level while the gains add up over rounds; a step at walker time nests (400^k).
- Mechanism (measured, oracle s2): the deep third of the copy loss is a local eigen-equation violation (M-J, s_II 0.985-0.989);
  one round's correction is ~92% c x f_J, the copy's own Jacobi factor (Test L); the bulk two thirds are moderate errors.
- Owner: Pfaffian angle (network expert); prediction and ending test to be written by the angle before any run.
- Constraint: a feature that needs neighbour sums at walker time is not cheap (it nests like a step).
- Candidate loop (PI 2026-10-30, pending: multi-round progress, Test N, the sign stage): guide -> one signed Lanczos step (its sign
  flips = the Krylov step, its size ~ the Jacobi factor, Control S within 8%) -> FN -> label-free refit of the amplitude AND a
  stored sign network -> repeat. Condition (PI's 'iff'): one step must be cheap at the target size; board 23: true for
  determinant-type networks with fast updates (constant factor in N), false for backflow (~N^2) and generic NNs (~2N).
- Competitor bar (PI 2026-10-30, reference, NOT a kill at 6x6): RBM+PP VMC + one signed Lanczos step + FN, own sign, 5.6e-6/site
  (same walker-cost class as network + one Jacobi factor); two steps 1.95e-6 (nested, out at scale). Our best now: 8.3e-6 (one
  label-free round, plain PP, s1; s1 floor 2.2e-6). PI's hypothesis: the advantage is at large sizes, where k Lanczos steps nest
  (400^k) while the loop stores rounds in the network at one-level walker cost; 6x6 favours Lanczos. Report every result against
  this bar; the scale claim is tested only by an equal-cost comparison at >= 8x8.
- Candidate (angle, board 20-21): A_aug = A e^{c f_J} = A^(1-c) A'^c, A' one Jacobi step of the FN equation with the copy's own
  walls; O(N^3) per walker step via rank-2/4 updates (plain PP; backflow raises the rank; T2 measures, oracle sign).
- Astra xhigh (cj-44177b7587f9; lead's confidence ~45% before): nonlinear Jacobi over-relaxation in log amplitude, no SOR
  guarantee; c* not transferable (rescan each round). T3 (loop round 2) is a fair best-case test IF it beats an equal-budget
  control (refit the previous target, same augmentation and c-scan): otherwise fresh Jacobi improvement masks a bad refit.
  Most fragile at scale: labels from finite walkers near nodes with a deployable sign; cheapest falsification = 6x6
  finite-sample replay on exact tables (deployable sign, hidden labels, exact grading), CPU.

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
  q RESULT 2026-10-27 (Pfaffian angle, verified by the lead from /scratch/a/A.Otaifi/ansatz_runs/pfaffian_q/q_ownnode.json): q = 0.424(7)
  vs control 0.069(4): INCONCLUSIVE by rule; the network's own node crosses the costly bond on ~42% of the cost (6x control). The
  'frozen at warm start' derivation is WRONG as worded: O2b's sign differs from the plain PP at 47% of the costly small sides.
  Lead decision: complex vs real PP copy, pass tax_0 <= 0.75x AND S_x cut >= 2x S_rest cut; kill >= 0.95x or S_x cut <= S_rest;
  3 seeds, both converged, <= 6 A40-h (PLAN_pfaffian.md).
  TEST C RESULT 2026-10-28 (Pfaffian angle 332186c; 5.31 H100-h; verified by the lead from pfaffian_c/runs/c1_*/cx_c1.json):
  complex PP copy tax_0 1.107 / 0.912 / 0.786e-5 (converged, 3 seeds) vs converged real F1 A 1.80e-5 (real reruns 1.74-1.90e-5, capped
  at 324 steps, unconverged): ratio ~0.51; one-hop tax 3.75 / 3.23 / 2.85e-6 vs F1 A 5.60e-6. Registered verdict INCONCLUSIVE (real
  arm unconverged, seed scatter 0.088 > 0.05, cut rule unreachable once cut_R > 0.5: rule flaw). Next (lead): tax_2 of the complex
  copies directly; PASS mean <= 0.82e-6, PARTIAL < 1.63e-6, KILL >= 1.63e-6. Real rerun (3.3 H100-h) only if PASS.
  T2 RESULT 2026-10-28 (Pfaffian angle 1baa881; verified by the lead from pfaffian_c/runs/t2_cplx/s{1,2,3}/cx_h1.json, where the
  complex copy sits under the key 'F1B'): complex tax_2 1.244 / 1.070 / 0.925e-6 (mean 1.08e-6) vs real F1 A 1.846e-6: PARTIAL, below
  the round gain 1.629e-6 for the first time, above the gate 0.82e-6. Next (lead): bond split of the complex residual on CPU; 5x5
  complex (2 seeds) only if the residual is capacity-like. Loop C needs the PI (loop closed 10-26).
  PREDICTION for Test D (lead, committed cd68f9e 2026-10-28 09:51, written before reading any complex split file; split_s1 already on disk, unread).
  Pair cost of the copy, frozen phi: c_xy = |H_xy| phi(y)^2 (r - r*)^2 / r, r = |A(x)/A(y)|, r* = phi(x)/phi(y).
  Real O2b round_1 split (pfaffian_d/split/split_o2b_round_1.txt): class r < r*/3 = 0.752 of the cost, of which q = 0.424 on pairs
  where sgn A(x)A(y) != s2(x)s2(y) (own-node crossings, q_ownnode.json) -> crossings 0.319 of the total, non-crossing dips 0.433,
  other classes 0.248.
  M0 (complex A removes the crossings only) predicts tax_0 ratio complex/real = 0.68; test C measured 0.52(5) (seed mean
  0.935e-5 vs F1 A 1.80e-5): M0 is off by 3 sigma, so the complex copy also removed ~0.16 of the real cost outside the crossings.
  M1 (lead's mechanism): a real A is small on BOTH sides of its own nodal surface; complex A has no surface (zeros codim 2), so
  it removes the crossings and part of the non-crossing dips near them, and leaves the other classes unchanged.
  M1 predicts for the complex split: class r < r*/3 = (0.433 - 0.16)/0.52 = 0.52 of the cost (range 0.45-0.57 for ratio
  0.47-0.57), class 1/3-3 about 0.41. Test D verdict then INCONCLUSIVE (between 40% and 60%).
  Rival M2 (complex = more parameters, a uniform cut): class r < r*/3 stays ~0.75 -> RESIDUAL DIPS. A share >= 0.65 refutes M1;
  a share <= 0.40 refutes M1 too (the bulk class would have shrunk, which M1 says it does not).
  Caveats: shares come from the frozen-phi bound formula (17.7x loose in total); q and the split are for O2b round_1, the ratio
  is vs F1 A (O2b trained 13x longer, -6%); Test D splits tax_0, the open number is tax_2 (tax_2/tax_0 0.116 complex, 0.103 real).
  TEST D RESULT 2026-10-28 (Pfaffian angle 000fd73; verified by the lead from pfaffian_d/split/split_s{1,2,3}.{txt,json}):
  class r < r_true/3 = 17.8 / 19.6 / 23.1% of the complex copies' cost (mean 20%; O2b 75%), class 1/3-3 = 68 / 66 / 63%.
  MY PREDICTION (M1, 0.52) IS REFUTED (rule: <= 0.40 refutes); M2 (0.75) too. Failed term: I used shares of the frozen-phi
  bound c_xy as shares of the true tax. The bound over-weights deep dips (c ~ r_true^2/r as r -> 0, unbounded) while the true
  E_FN loss saturates (a dip only raises the wall |H_xy|/r at x): bound/true = 17.7 for O2b (dip-dominated), 6.05/6.87/7.81 for
  the complex copies (split_*.json ratio_cost_over_dE). So the dips were a smaller part of the real tax than 75%, and removing
  them took ~half the tax, not a quarter. Open: a saturating cost formula (true dE per pair) before any further split-based prediction.
  5x5 complex NOT run (lead): the only number we have is the real 3x3 -> 5x5 gain (F1 A -> F1 B one-hop tax -18%), which
  predicts complex tax_2 ~0.89e-6 vs gate 0.82e-6, a gap below one seed's scatter (~0.1e-6): two seeds could not decide.
  Budget 4.5 H100-h kept. Stage 1 continues as the ansatz-choice question (Pfaffian angle with the PI, msg f71485ee).
  LOOP C PREDICTION (lead, takeover 2026-10-28, derivation only, no run; board block 25; under worker review):
  Map with storage: E_{k+1} = r E_k + T (E = E_FN of the round's target, r = exact two-hop contraction, T = tax_2, constant).
  T constant per round is MEASURED for the real one-hop loop: O2b rounds 0-4 tax 6.01/5.93/5.69/6.35/5.88e-6 (README
  compress_6x6 'Results: O2b'); tax_2/tax_1 = 0.32-0.33 for every copy (t2_cplx cx_h1.json). Fixed point E* = T / (1 - r).
  Complex copies, O2b chain: T = 1.08e-6; r from phi_1: E_FN(K2(phi_1)) / E_FN(phi_1) = 3.24e-6 / 9.85e-6 = 0.33
  (G-A3 chain: r 0.16 -> 0.49 as E falls; range 0.26-0.5) -> E* = 1.6e-6 (range 1.5-2.2e-6).
  Rounds: E(phi_2) = 4.32e-6 (already measured: the complex copies' two-hop E_FN, 4.17-4.49e-6), then ~2.5-3.0e-6, ~1.9-2.4e-6.
  Competitor (no loop, RBM+PP VMC + two signed hops + FN): 1.95e-6 (G-A3 round 0). Prediction: loop C with complex PP+3x3bf
  copies gains NO accuracy edge at 6x6 (E*/competitor 0.8-1.1). Derived gate (replaces 'half the round gain'): a 2x edge needs
  E* <= 1e-6, i.e. T <= (1 - r) x 1e-6 = 0.5-0.7e-6, a further 1.5-2x cut of the storage tax.
  Caveats: competitor and loop are on different chains (RBM+PP own sign vs O2b with the s2 sign); r taken from one target;
  T assumed to transfer between targets (real copies: 1.66e-6 on the RBM+PP target vs 1.85e-6 on phi_1, so roughly).
  CORRECTION 2026-10-28 (worker review, reviews/2026-10-28_loop_fixed_point_review.md; load-bearing numbers re-read by the lead
  from compress_6x6 README 'Results: O2b' and 'Results: G-tax'): the LOOP C PREDICTION above is WRONG in two terms.
  (a) r had the wrong denominator: the recursion carries phi_1's FN energy 14.12e-6 (E_FN(g_1)), not its zero-hop 9.85e-6, so
  r = 3.24/14.12 = 0.23 on the s2 chain. (b) The competitor comparison was across chains. On the competitor's own (own-sign,
  RBM+PP) chain, G-tax gives round 1 = E_FN(K2(phi_0)) + T = 0.318e-6 + T: real copy 1.975e-6 vs no-loop 1.947e-6 (no edge,
  as measured); complex T 1.08e-6 would give 1.40e-6 = 0.72x the competitor (0.66-0.80 over the seeds). On the s2 chain the
  fixed-sign floor F = 0.915e-6 (itself unconverged) keeps E* >= F + T ~ 2e-6, which is why that chain showed no edge.
  Also: r is not a function of E (O2b rounds 3-4 same E, next target 12% lower): the loop has memory; linear map is approximate.
  DECISIVE OPEN NUMBER: tax_2 of a complex PP + 3x3 backflow copy of the own-sign target phi_0 = FN(K2(A)) (the G-tax target).
  Prediction (real copies transferred at 0.87-0.90x from phi_1 to phi_0): 0.94-0.97e-6 -> round 1 at 1.26-1.29e-6, 0.65x the
  competitor. Kill: tax_2 >= 1.63e-6 (no edge at round 1). 2x edge at round 1 needs tax_2 <= 0.66e-6. Cost: one complex copy
  per seed (test C rate ~1.5 H100-h) + T2 readout 0.2: 2 seeds ~3.2 of the 4.5 H100-h left. NOT submitted (takeover: no spend);
  PI decides at the check-in.
  WITHDRAWN 2026-10-28 12:20 (PI): the loop prediction, its correction and the proposed phi_0 tax run above all used tax_2 (two
  signed LANCZOS steps on the copy), i.e. the old pipeline. The agreed method (Stage 1/2) is: sign from the Krylov step s2, amplitude
  from the network, fixed node. Its storage loss is tax_0 = E_FN(s2|copy|) - E_FN(s2|phi_1|): complex 1.107/0.912/0.786e-5 (mean
  0.94e-5) vs real F1 A 1.80e-5, against the target's own 0.985e-5 (test C). The 6x6 table is a best case: at scale the network must
  be fitted to H_FN's ground state from walkers (board 14, open). No run proposed until the PI and lead agree on the pipeline.
  TEST E (CPU, existing data; pre-registered by the lead 2026-10-28 before any number; board blocks 5, 8):
  Question: does the complex copies' remaining cost sit where the fit (cx_c1fit.py: equal weight per phi^2 decade -5..-13 plus an
  e_x^2 term) puts little weight? Per decade band of x: fit weight share vs share of the pair cost
  c_xy = 2|H_xy| phi(x)phi(y)(cosh(e_x - e_y) - 1) of the complex copies (seeds 1-3); also configurations outside -5..-13 and the
  e_x^2 share of the converged fit loss.
  Prediction (lead, ~55%): the cost concentrates in middle bands where cost share >= 2x fit share. Kill of the weighting cause: cost
  and fit shares agree within 1.5x in every band holding >= 10% of the cost. Caveat: c_xy is the frozen-phi bound (6-8x the true
  cost for the complex copies); shares, not totals, are compared.
  TEST E RESULT 2026-10-28 (worker; verified by the lead from pfaffian_e/test_e.json; 151 s CPU): gate SUPPORTS the weighting
  cause, not as predicted (lead's 'middle bands' WRONG). Pair cost by band of the smaller-phi end, mean of 3 complex seeds:
  -11 10.9% (0.98x its fit share 1/9), -12 23.1% (2.08x), -13 27.4% (2.47x), below -13 30.4% with fit share 0.
  Configurations with phi^2 < 1e-13 are 67.6% of all configurations (phi^2 weight 7e-5) and are never drawn as x by the fit
  (cx_c1fit.py l.71, l.295), only as bond partners. Half-each assignment: no in-band 2x, 'below' 16.4%. e_x^2 share of the fit loss
  ~0.21-0.26 (estimated from records). Caveat: shares are of the frozen-phi bound, which over-counts deep regions; the true share of
  the unsampled region is not measured. Next (proposed, CPU): splice exact values (e = 0) into the configurations below 1e-13 and
  re-solve E_FN exactly: the true cost of that region.
  TEST F (splice, CPU; pre-registered by the lead 2026-10-28 before any number): true cost of the deep region. For each complex copy
  (seeds 1-3) set e = 0 (|A| = phi up to the offset) on (A) configurations with phi^2 < 1e-13 (never drawn by the fit) and (B) phi^2
  < 1e-11 (bands -12, -13 and below); re-solve E_FN exactly with sign s2 (same solver as split_*.json E_FN_copy). Reading: the
  drop of the copy loss tax_0 (0.94e-5 mean). Prediction (lead): A removes 10-25% of tax_0 (the bound says 16-30%, and it over-
  counts deep regions); B removes 35-60%. Kill of 'the untrained region matters': A < 5%.
  TEST F RESULT 2026-10-28 (worker; read by the lead from pfaffian_f/test_f.log; controls reproduce split_s*.json E_FN exactly):
  exact values spliced into (A) phi^2 < 1e-13 remove 10.3 / 10.7 / 10.8% of tax_0 (mean 10.6%), (B) phi^2 < 1e-11 remove 36.5 /
  34.9 / 33.7% (mean 35.0%); random bulk splice of A's size removes 0.002% (seed 1). Both inside the prediction (low ends).
  Reading: the fit's deep-band weighting explains at most ~1/3 of the copy loss; ~2/3 sits in phi^2 >= 1e-11, where the fit's
  weight is already >= the cost share (Test E). Scale: tax_0 0.94e-5 is about the whole gap of phi_1 above the s2 floor
  (9.85e-6 - ~0.9e-6 = 8.9e-6), so the copy must improve several-fold, not 1.5x, for the amplitude loop to gain.
  TEST G (pre-registered by the lead 2026-10-28 before any code or number; PI go 'submit the true energy fit'; board block 10):
  Mechanism: the band fit mis-weights (Test E/F: true cost 35% deep / 65% bulk; band fit draws no x below 1e-13 and spends ~21-26% of
  its loss on e_x^2, which the energy does not see). Fitting the TRUE E_FN (Hellmann-Feynman: gradient of L_chi* with chi* = exact
  ground state of H_FN[A], re-solved every few steps) gives every region its true weight.
  Setup: complex PP + 3x3 backflow, warm start = the converged band-fit copies c1_cplx_s1 and s2; same sign s2, target-free (phi only
  for readout); exact chi* solves; guards: fixed sum phi^2 ln|A| (E_FN is scale-invariant), cap |d ln|A|| per step; stop on E_FN flat.
  Prediction: tax_0 0.94e-5 (s1 1.107e-5, s2 0.911e-5) falls 20-40%, to 0.55-0.75e-5; E_FN decreases monotonically (exact gradient).
  Kill of 'the fit's weighting is a lever': drop < 10%. Follow-up readout: Test F splice on the new copies; deep share still ~35%
  -> the network cannot hold deep corrections (capacity, Pfaffian angle); clearly lower -> the fit captured them.
  Budget: <= 4.0 H100-h incl. rate probe, from the 4.5 left. Derivation reviewed by a worker before submission (CLAUDE.md point 6).
  TEST G AMENDMENT 2026-10-28 (after the derivation review, reviews/2026-10-28_true_EFN_gradient_review.md, before any number):
  scheme SOUND with conditions (conj gradient for complex parameters, blocked set from s2 only, kept-graph connectivity, float64,
  scale as pure gauge, cap blocked-bond differences or backtrack). Key point from the review: frozen-chi* L is a majoriser of E_FN
  (majorise-minimise, monotone), and with enough capacity the scheme converges to the SAME fixed point as the plain amplitude
  iteration guide <- |chi*| (self-consistent phi*), at its linear rate (~0.926 per solve, cx_s2ceil). So Test G measures how close a
  3x3 complex network gets to that fixed point. Readout changed to E_FN of the network itself (band-fit copy 1.93e-5 mean; phi_1 as
  guide 9.85e-6; s2 floor ~0.5-0.9e-6). Prediction restated: E_FN 1.93e-5 -> 1.54-1.74e-5 (tax_0 drop 20-40%); kill < 10%.
  Caveat from the review: Test E bound shares (additive) vs Test F splice response (non-additive) are not directly comparable.
  M-J RESULT 2026-10-28 (Pfaffian angle 86023ef; s_II verified by the lead from pfaffian_mj/mj_split.json): below phi^2 1e-11 the
  copy's error is 98.5-98.9% the LOCAL eigen-equation violation (II), not inherited from neighbours; deep copy too small by
  e^-0.13..e^-0.28 relative to its (correct) neighbours. Lead (takeover) approved the angle's CPU steps: (2) one oracle-free Jacobi
  step with the copy's own V_A and E_FN (cost class: one neighbour sum per amplitude, as a Krylov s1 step), (1) per-decade offset.
  NOTE 2026-10-28 (lead, records check): Test G's objective is TRIED_IDEAS B10/B11 (frozen-H_FN Rayleigh with a factorised residual;
  exact 4x4 99.7%, 8x8 sampled inconclusive, OOM), now at 6x6 with the complex PP class over the full space. Astra second opinion
  (angle, d25fbae, ansatz/REVIEW_astra_pfaffian_2026-10-28.md): expressibility vs learnability unresolved; G success = reducible in
  the class, G failure = no floor (both seeds share the F1 A real parent). Separator (teacher-student + constructive witness,
  ~1.5-2 H100-h) awaits the PI's GPU go.
  TEST G AMENDMENT 2 2026-10-28 (lead, after Astra xhigh, codex job cj-34203b7a8e09; lead's prior confidence in G ~65%): G becomes
  a single-seed GATE, <= 0.75 H100-h: one frozen-chi* block from c1_cplx_s1. Readout per round k: Delta_k = E_FN[A_k] - E_FN[chi_k]
  (exact one-round opportunity, chi_k = FN[A_k] as guide) and tau_k = E_FN[A_{k+1}] - E_FN[chi_k] (storage penalty); realised gain
  = Delta_k - tau_k. Continue (fund more rounds / a second seed) only if tau_0 <= 0.5 Delta_0; capture < 10% -> stop and diagnose.
  Reasons: a full G costs ~4.4 H100-h per seed in tables alone (532.6 s per table, 0.926 per solve); the teacher-student test T
  reuses the band sampler and a shared parent and decides neither branch (deferred). T and any budget increase stay with the PI.
  DELTA_0 RESULT 2026-10-28 (CPU worker, pfaffian_g/delta0/delta0.json; controls reproduce split_s1 and target E_FN exactly):
  one exact amplitude round (guide <- |FN state|, sign s2) from the complex copy s1: 2.092e-5 -> 1.310e-5, Delta_0 = 7.82e-6/site;
  from the exact target phi_1: 9.853e-6 -> 7.768e-6, Delta = 2.09e-6. Gate for G: tau_0 <= 3.9e-6. Test G's own prediction
  (tax drop 20-40% -> tau ~5.6-7.5e-6) predicts the gate FAILS (tau/Delta 0.7-1.0). Scale: near the target a round gains only
  ~2e-6, so a useful copy must lose <~1e-6, about 10x below today's 0.94e-5, not 2x.
  M-R RESULT 2026-10-28 (angle d45c681; verified by the lead from pfaffian_mr/mr_result.json): KILL. RBM residual storing the round's
  correction R* = ln(phi/|A|) (bond rms 0.0107): tax/gain 0.954 (alpha 4) / 0.941 (alpha 16); the stored R is ~6% of R*. The law
  tax/gain = eps^2 HOLDS (ratio 1.007 / 1.011; controls R=0 -> 1, R=R* -> 0, R=-R* -> 2.51): any storage proposal now only needs
  its eps. Angle's next (CPU, Test L): exact least squares of R* on features, 2-body vs the copy's own Jacobi factor f_J.
  STEP BACK 2026-10-28 20:27 (lead, takeover; second failed version of sampled inner steps in G, probes 4049890 + 4052565,
  0.12 H100-h; all physics controls passed, the optimiser did not: 1/20 steps significant, MM bound dE_FN +7.9e-8 +- 3.2e-7):
  1. Verdict: stop Test G (no gate, no more GPU); REFRAME proposed to the PI.
  2. Thought: by the review's majoriser result, the amplitude loop whose store step is the true-energy projection IS Test G's
     update, so with a FIXED network class its endpoint is a stationary point of min_theta E_FN[s2|A_theta|]; a log-copy store
     can only end higher. A fixed-class loop cannot beat the class's best fixed-node guide; it is an optimiser for it. Only a
     GROWING store (M-R chain, Jacobi round, L) can beat the class, at growing cost. Same SNR term as C15 (sampled HF force
     near a stationary point).
  3. Cheapest next test: none on GPU. CPU, existing nets: E_FN[s_k|A_VMC|] for k = 0,1,2 vs E_var of the same VMC net at 6x6
     (the add-on value the Goal names), ~1 h CPU, partly in the records. Astra consulted on the reframe (rule 8).
  ASTRA ON THE REFRAME 2026-10-28 (cj-e4f141f630b1; load-bearing items re-read by the lead):
  - Fixed-class bound correct (as an infimum): no fixed-class loop ends below inf_theta E_FN[s2|A_theta|]; the true-energy store
    makes the loop an optimiser for it (MM), not a route beyond it. 'Only a growing store beats it' is too strong (any change of
    class, e.g. a fixed-depth correction, also escapes).
  - LABEL ERROR (verified, experiments/compress_6x6/cx_refresh.py l.48): the 's2' used in all stage-1 tests is s1 flipped on the
    FN expelled set X (an oracle flip from an exact FN solve), NOT two Krylov steps. Every 'Krylov sign s2' in this plan, on the
    board and in today's explanations to the PI means this oracle sign. Deployable claims must exclude it.
  - Existing data (verified, results/lanczos_baseline_6x6/lanczos_baseline_6x6.json, ViT guide psi_P, per site): VMC 1.319e-4;
    FN with own sign 1.018e-4; FN with one Krylov sign step 9.43e-5 (-7%); FN with one signed Lanczos step 1.887e-5 (5.4x lower
    than one Krylov step). One Lanczos step costs one neighbour sum per configuration (as a Krylov step) when alpha is chosen by
    a scan like T; the amplitude change, not the sign, carries most of the gain.
  Lead's proposal to the PI (not acted on): the Goal's deliverable is network + one signed Lanczos step + FN as an add-on; stage 1
  (storage) becomes optional; the Krylov-only restriction of 2026-10-28 costs ~5x at 6x6 on this evidence.
  TEST L RESULT 2026-10-28 (angle 0618eaa; verified by the lead from pfaffian_l/l_result.json; ORACLE s2, copy s1, one round far
  from the fixed point): the round's exact FN correction R* = ln(chi/|A|) is, to 92%, one scalar times the copy's own Jacobi factor
  f_J = ln[sum_kept |H||A(z)| / ((H_xx + V_A - E')|A(x)|)]: eps^2 0.079 (f_J alone, coefficient 1.34) vs 0.973 for 2-body
  features (explains the M-R kill). Exact E_FN: A e^{1.33 f_J + small} tax/gain 0.048 (tax 3.75e-7/site); pure Jacobi A e^{f_J}
  0.257. Caveats: c = 1.34 fitted to the oracle R*; oracle sign. Next (angle, CPU): oracle-free c* by an exact line search on
  E_FN (pred. 1.2-1.5, tax/gain <= 0.08, kill > 0.2), then near the fixed point (gain 2.09e-6) against the < 1e-6 bar.
  ONE-STEP LOCAL CORRECTIONS 2026-10-28 (angle; verified by the lead from pfaffian_c_star/ext/cstar.json and
  pfaffian_s/cstar_lanczos.json; ORACLE s2, complex copy s1, E_FN - E0 per site): copy 2.092e-5; one exact FN round 1.310e-5;
  copy x e^{c f_J} (Jacobi, c* = 3.92 by exact line search) 8.03e-6; copy x |1 + alpha(E_L - E)| (Lanczos, alpha* = -0.10 by the
  same search; moment formula -0.098) 8.68e-6; s2 floor ~0.9e-6. One local-energy factor of either kind is worth ~1.7 exact FN
  rounds (64% / 59% of the way to the floor); Jacobi beats Lanczos by 0.66e-6 (8%, weak). The rest of the way needs nested steps
  (cost 400^k) or a stored multi-round correction ln(phi/A_0) (untested).
  TEST Z RESULT 2026-10-29 (angle; verified by the lead from pfaffian_z/z_result.json): at the ORACLE-FREE sign s1 (K1 sign of
  psi_P; s2 = s1 flipped on X, 9.28% of orbits, 0 mismatches): copy 2.503e-5, one exact FN round 1.608e-5 (gain 8.96e-6),
  A e^{3.9 f_J} 1.018e-5; s1 floor 2.05-2.5e-6. The copy, not the sign, binds (~4x the floor); conclusions unchanged by the
  oracle. Angle's T3' (complex RBM x PP, paired control, stall rule) predicts g2/g1 = 0.3-0.5; Astra consulted before any GPU.
  T3' DEFERRED 2026-10-29 (lead, after Astra cj-6e37615f546c): its stall rule mixes non-comparable gains (geometric convergence to
  the s1 floor would give g2/g1 ~0.35 and still be useful). First the CPU screen: phi_2 = FN(A_1) from Z's A_1, rescan c (with c=0),
  replay saved copy errors; judge by tau/Delta and the paired-control advantage. GPU (R1 first) only if the screen leaves headroom.
  T3' SCREEN RESULT 2026-10-29 (angle cb311ca; verified by the lead from pfaffian_screen/screen_result.json; sign s1):
  S1 one exact round from A_1 = A e^{3.9 f_J}: 1.018e-5 -> 7.98e-6 (Delta 2.21e-6). S2 zero copy error: phi_2 e^{c f_J} reaches
  5.60e-6 at c = 5, still falling: with a perfect copy round 2 keeps moving toward the s1 floor (~2.2e-6). S3 (replayed copy
  error, tau/Delta 19.9) INVALID: it transplanted the s2-target copy error onto an s1 target (error rms 1.06 on the flip set X vs
  0.010 off it). Lead: R1 approved (<= 1.5 H100-h), judged by the SAME rule on a valid measurement (tau/Delta <= 0.7 vs S2).
  TEST W (PI go 2026-10-29 'both go'; queued question 'labels from walkers'; pre-registered by the lead before any number; CPU):
  Can one round's gain survive labels estimated from FINITE samples instead of the exact table? Setup at sign s1 (Test Z): guide
  g = s1|A| (complex PP copy s1, E_FN 2.503e-5); exact next guide phi'' (E_FN 1.608e-5); Delta = 8.957e-6. Draw N independent
  samples from the mixed distribution n_x |g(x)| phi''(x) and N from n_x g(x)^2 (ideal walkers, optimistic: no correlation, no
  population bias); estimate h(x) = phi''/g from the count ratio on visited orbits, h = 1 (keep g) elsewhere, normalised; guide
  phi_hat = g h. Arms: (a) raw phi_hat; (b) phi_hat x e^{c f_J[phi_hat]} with c by exact line search. N = 1e5, 1e6, 1e7, 1e8.
  Readout: retained = (E_FN[g] - E_FN[phi_hat]) / Delta per N and arm; per-band share of what is lost.
  Prediction (lead, ~50%): arm (a) retains <= 30% at 1e6 and 30-70% at 1e8 (raw counts miss the near-node cost); arm (b) retains
  more than arm (a) at every N, because the Jacobi factor fills unvisited configurations from visited neighbours.
  Decision: (a) >= 50% at 1e6 -> walker labels are not the 6x6 bottleneck; (b) < 20% at 1e8 -> finite-sample labels cannot carry
  a round, and the estimator (not the network) is the next open problem. Raw labels without a network fit: tests label information,
  not generalisation.
  TEST K RESULT 2026-10-29 (angle a9cf9f4; sign s1): growing store of primitive PP Jacobi features KILLED on round 2. Round 1:
  f_J[A_s1] eps^2 0.078 (best 3: 0.0776). Round 2 correction ln(phi_2/|A_1|): single features 0.989-1.000, best 3 0.9705 (kill
  >= 0.70), tau/Delta 0.947; feature cosines 0.65-0.82. Reading: each round's correction is the CURRENT guide's own Jacobi factor;
  a dictionary of other networks' factors cannot hold it. K3 (running): R*_2 ~ c f_J[A_1]. If so, only nesting (cost) or a refit
  carries rounds forward: R1's refit quality decides.
  K3 RESULT 2026-10-29 (angle a6e123b; sign s1): round 2's correction R*_2 on the augmented guide's own factor f_J[A_1]: eps^2
  0.058 (coefficient 1.085; + primitive best 3: 0.056; cosines with primitive factors 0.01-0.09). Two nested levels
  A_1 e^{2.4 f_J[A_1]}: E_FN 7.22e-6 < phi_2's 7.98e-6. Mechanism, measured in two rounds: each round's FN correction is
  ~1.1-1.3 x the CURRENT guide's own Jacobi factor (eps^2 0.078, 0.058). Fixed dictionary cannot hold it (K); nesting holds it at
  one level per round (cost); a per-round refit of the base is the cheap route = what R1/R2 measure.
  TEST W RESULT 2026-10-29 (worker, ~1.9 h CPU; verified by the lead from pfaffian_w/w_result.json; sign s1): decision rule
  '(b) < 20% at 1e8' FIRES. Raw finite-sample labels (count ratio of mixed vs reference ideal samples, no network) destroy the
  round: retained (a) -1482 / -262 at N = 1e6 / 1e8 (seeds agree within 0.8%), (b) with the Jacobi factor -301 / -37; controls
  exact h -> 1.0000, shuffled h -> -12788. Mechanism (measured): the label is tiny, rms ln(phi''/g) = 0.0059 (phi''^2-weighted),
  while per-orbit count noise is 0.40 (1e6) / 0.13 (1e8): 20-70x the signal; coverage is not the limit (visited orbits hold 0.995
  of phi''^2 at 1e8); noise = signal only near N ~ 5e10. Prediction (a) <= 30% at 1e6 held; (a) 30-70% at 1e8 missed.
  Reading: pointwise walker labels cannot carry a round; any walker-trained store must average label noise over many
  configurations, or avoid labels. Note: g e^{3.9 f_J[g]} needs NO labels and retains 1.66 x Delta (Test Z).
  LABEL-FREE LOOP (lead's idea after Test W; Astra cj-ff745902d93d): refit the base each round to T = A e^{c f_J[A]} (exact at any
  sampled x, no labels; walkers only for E_FN and c). Astra: sound, convergence unproved; it is Kochkov-Clark SWO (1811.12423) with
  nonlinear Jacobi targets and an FN line search (known failures: non-monotone fits, trapping on unrepresentable intermediates,
  small amplitudes hard to sample). Fixed-class bound unchanged. ALSO: R1's augmented-energy gate is void (the unchanged guide plus
  its own factor already gives tau/Delta -0.658): R1 is judged by the BARE post-fit E_FN (angle told). Proposed paired label-free
  arm (same architecture/start/optimiser/seeds/budget; samples and weights without phi''): screen e(fit(T)) <= 1.761e-5 per site
  (tau_J/Delta_J <= 0.5), kill tau_J/Delta_J >= 1. Awaits the PI's go (loop change).
  NEXT TEST N (PI 'Go' 2026-10-29, after Astra cj-6168204a40dd; angle runs, CPU): can a cheap NONLINEAR feature bank hold the
  round correction? Frozen symmetric finite-weight RBM features + linear readout, exact least squares in the bond metric on round
  1, round 2 and the cumulative correction (s1), graded by exact E_FN. Prediction (Astra, low conf.) eps^2 <= 0.2; promote at bare
  tax/gain <= 0.3 and cost <= 2x base; kill eps^2 >= 0.7. Astra corrections: 'eps^2 ~0.95 for any class' NOT established (O2b
  round 1 T/g = 0.386; M-R/L/K = small-weight 2-body and one dictionary only); the 4x4-6x6 T(N) test is class-specific, deferred.
  L-SWO RESULT 2026-10-30 (angle 55910d9; Paderborn 4062263, 520 steps converged, 0.58 H100-h; e(fit) and tau/Delta verified by the
  lead from pfaffian_lswo/readout/readout.json; sign s1): BETWEEN. Label-free refit to T = A e^{3.9 f_J}: bare e(fit) 1.957e-5 (pass
  <= 1.761e-5, kill >= 2.503e-5; predicted 1.9-2.1e-5), tau_J/Delta_J 0.632: stores 37% of the Jacobi gain without any FN labels.
  Re-augmented: fit e^{4.0 f_J[fit]} 8.34e-6 vs A_s1 e^{3.9 f_J} 1.018e-5 = the augmented guide moved 1.84e-6 in one label-free
  round (perfect-copy reference S2: 5.60e-6, so ~40% of the ideal progress). Copy error vs T: 76% of pair cost moderate.
  LEAD'S DERIVATION 'label-free round = fixed-sign VMC step' (board 22) PARTLY WRONG (Astra cj-d0bc7446d51c): steps (1)-(3) hold
  (f_J ~ -(E_L - E')/K_A to 5.7% in the bond norm, not uniformly: r spans -0.82..29.9; E' = E_FN, not E_var). Step (4)-(5) fails:
  the fit weights are band-uniform over decades -5..-13 with blocked-bond terms (~120x more weight in decade -13 than |A|^2), so the
  fixed point is Cov_q(O, f_J) + sum_blocked (q b / Wbar)(O_x - O_y)(f_J,x - f_J,y) = 0: neither VMC nor E_FN stationarity. L-SWO may
  beat the VMC optimum in E_FN by fixing blocked-bond ratios in low-|A|^2 regions; not guaranteed. Decisive test (later): converged
  L-SWO vs SR-VMC, same class, sign s1, equal cost, same augmentation.
  CODE FINDING: cx_r1fit.py zero-initialises the RBM block, whose gradients then vanish; L-SWO's 148 RBM parameters stayed 0. L-SWO
  and R1 fit a plain complex PP (no RBM, no backflow), not 'RBM x PP'. Their pairing stays valid; labels corrected; fix for future runs.
  ASTRA CAMPAIGN AUDIT 2026-10-30 (cj-040b53cdc8c3): demonstrated so far = a compression mechanism, not an equal-cost advantage.
  Risks ranked: (1) no advantage over one-step Lanczos at equal total cost (8.34e-6 + 0.58 H100-h vs 5.57e-6); (2) deployable walker
  cost (backflow 36.5x; sign lookup excluded); (3) storage and sampling with size (the label-free fit still samples from enumerated
  target tables, cx_r1fit.py l.368; one round; RBM inactive); (4) sign choice (s1 was compared with the unstepped PP sign, not the
  competitor's signed-Lanczos sign); (5) stored-sign stage untested in this loop; (6) E', c noise. Structural: versus network + ONE
  Lanczos step both keep one correction level, so the loop has NO neighbour-depth advantage, only possibly better quality per
  training cost; its depth advantage is only against several nested steps. Next (lead, CPU): TEST X crossed signs, pre-registered:
  loop magnitude (L-SWO fit x e^{4.0 f_J}) with the competitor's signed-Lanczos sign, and the competitor's magnitude with s1; 2 exact
  FN solves. Prediction (Astra, low conf.): the competitor's sign lowers the loop's E_FN by >= 1e-6/site. If so, the sign choice is
  material to the 2.8e-6 deficit; if not and Lanczos still wins at common s1, the deficit is amplitude.
  (Old) Next: the q check on EXISTING data (Pfaffian angle): share q of the near-node cost on pairs with sgn[A(x)A(y)] != s2(x)s2(y),
  forward pass of saved round_1 theta, CPU minutes; q >= 0.5 confirms the own-node cause, q <= 0.1 kills it. Waits for the PI's go.
  (Done 2026-10-27/28: no-own-sign network = complex-parameter PP |A| x s2; tests q, C, T2 below; Test D running.)
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

## What we understand (CLAUDE.md point 7; one line each, board block, evidence)
- E_FN[A] = lowest eigenvalue of H_FN[A]; the copy enters only through V_A(x) = sum_blocked |H_xy| |A(y)|/|A(x)| (board 4).
- <chi|H_FN|chi> - <chi|H|chi> = sum over blocked bonds of a perfect square (the double sum counts each bond twice), >= 0 (board 5).
- Copy cost <= sum_blocked 2|H_xy| phi(x)phi(y)(cosh(e_x - e_y) - 1): only error DIFFERENCES across blocked bonds cost (board 5);
  it is a bound on E_FN[A] - <phi|H|phi>, equal to the copy's cost only if phi is self-consistent; bound/true 17.7 (real O2b),
  6-8 (complex) because the true FN state moves away from the copy's walls (pfaffian_d/split/*.json).
- phi_1 is the FN ground state of the previous guide, not of H_FN[phi_1] (E 14.12e-6 vs 9.85e-6 as its own guide); the amplitude
  loop at fixed sign converges to the self-consistent phi* (ground state of H_FN[phi*]), which is phi_G only if s2 is the exact
  sign; 6x6 floor with s2 0.9e-6 (compress_6x6 README).
- Complex parameters remove the network's own near-zeros: P(|A| < eps) ~ eps (real) vs eps^2 (complex) (board 6); Test D share
  r < r*/3 75% -> 20%, copy loss 1.80e-5 -> 0.94e-5 (pfaffian_c, pfaffian_d).
- The copy fit (cx_c1fit.py) draws equal numbers per phi^2 decade -5..-13 (our own design, T1s 2026-10-21) and never draws x below
  1e-13 (67.6% of configurations); it also penalises e_x^2, which the energy does not see (board 8; Test E).
- Exact values in phi^2 < 1e-11 remove 35% of the complex copy's loss, below 1e-13 11% (Test F, pfaffian_f): the deep bands
  matter but two thirds of the loss sits in the bulk, where the fit already weights enough.
- One exact amplitude round gains 7.8e-6 from the complex copy but only 2.1e-6 from the exact target phi_1 (delta0.json): near the
  fixed point the copy must lose <~1e-6, ~10x below today's 0.94e-5.
- Storing a round's correction with relative error eps keeps a fraction 1 - eps^2 of the round's gain (M-R, measured ratio
  1.01; pfaffian_mr/mr_result.json).
- One round's FN correction is ~92% one scalar times the copy's own Jacobi factor f_J (Test L, oracle sign, one round):
  computable on the fly from A with one neighbour sum, no training.
- One local-energy factor (Jacobi e^{c f_J} or Lanczos |1+alpha(E_L-E)|, scalar by line search) on the copy is worth ~1.7 exact FN
  rounds at oracle sign (8.0e-6 / 8.7e-6 from 2.09e-5; floor 0.9e-6).
- Each round's FN correction is ~1.1-1.3 x the current guide's own Jacobi factor (eps^2 0.078 round 1, 0.058 round 2; Tests L,
  K3, sign s1/s2): a fixed dictionary of other networks' factors cannot hold it (Test K).
- Pointwise labels from finite walker samples are 20-70x noisier than one round's correction (Test W): a round's label signal
  is rms 0.006 in ln phi; the Jacobi correction needs no labels.
- One LABEL-FREE loop round (refit to A e^{c f_J}, re-augment) moved the augmented guide 1.018e-5 -> 8.34e-6 (L-SWO, s1): the
  loop can progress without FN labels; storage keeps ~37-40% of the ideal step.
- Label-free storage beats exact labels (R1b vs L-SWO, both converged, complex PP RBM inactive, s1): bare 2.373e-5 vs 1.957e-5,
  augmented 9.64e-6 vs 8.34e-6; one stored round keeps 15% (labels) vs 37% (Jacobi target) (pfaffian_r1/readout_b, pfaffian_lswo/readout).
- Two label-free rounds keep improving (L-SWO round 2, unconverged, one seed): bare drop 1.26e-6 (s1) / 1.27e-6 (s_G); with the
  Lanczos sign s_G the augmented guide reaches 5.39e-6 at c*=4 (bracketed: c=3 5.92e-6, c=5 5.95e-6) vs competitor 5.574e-6: first guide at the bar,
  not yet quotable (pfaffian_lswo2/readout/sG/readout.json). Converged 2-seed rerun running.
- Sign and amplitude do not separate (TEST X, exact FN, pfaffian_x/x_result.json): loop |L| 8.34e-6 (s1) -> 7.39e-6 (s_G);
  competitor |G| 5.57e-6 (s_G) -> 1.00e-5 (s1). s_G is better for both magnitudes, but the gain depends on the magnitude
  (0.95e-6 for |L|, 4.45e-6 for |G|), and the magnitude ranking flips with the sign (s1: |L| better by 1.68e-6; s_G: |G| better
  by 1.82e-6). No additive sign/amplitude split of the 2.76e-6 deficit; at s_G the loop magnitude is 1.8e-6 worse.
  (Corrected 2026-10-30, sign angle + Astra cj-344a55c0d108: not "each magnitude is best with its own sign".)
- Cost of one Lanczos/Jacobi step per walker move (derived, board 23): needs ~d^2 = 4N^2 two-swap ratios. Determinant networks
  with fast updates: O(N^2 det4) per image, a CONSTANT factor over plain (measured 5.4x at 6x6); backflow grows ~N^2 (36.5x at 6x6);
  generic NNs pay a factor d ~ 2N (~400 at 14x14). So 'one step is cheap at any size' holds only for determinant-type networks.
- A Krylov step costs one H application per configuration; s2 costs neighbours of neighbours; recomputing the sign each round costs
  ~400^k at 14x14 (Queue).

## Queue (CLAUDE.md point 4: no run, no worker; the PI picks the next one when the open question closes)
- Walker-code speed (GPU profile 4075217, gpu_profile/PROFILE.md): 8x8 FN-GFMC is GPU-bound (sm 96.6%), 98.3% network compute,
  dominated by XLA reduce/transpose fusions, not matmul/solves; bigger GPU batches do not help; f32/tf32/bf16 not usable near nodes
  (f32 18 sign flips / 1.44M pairs). Next lever (hypothesis): find the line behind input_reduce_fusion_2 (12 of 25 s) and use
  fast determinant-ratio updates per hop; before the 10x10 stage. Repo gfmc_run.py/gfmc_fn_lib.py differ from launched M1 code.
  DECISION (worker + Astra cj-a27475bc4afa, gpu_profile/DECISION.md): BUILD the determinant-contraction rewrite (rbmpp_lib.py l.183-190
  bond sums over a 2 GiB V array = 49% of GPU time -> batched GEMMs), expected 2.4x (1.8-3.0) at 8x8 and 10x10, 3 days, gated on
  measured >= 2x at 10x10 + near-node accuracy; PI approved 2026-10-30. BUILT (ec94679, --ratio fast, default ref): accuracy no worse
  than ref vs 50-digit truth, walker energies equal to 8e-14; A40 speed 1.39x (8x8) / 1.17x (10x10), A40 f64 GEMMs weak; H100 bench
  4078411: kernel 2.95x (8x8) / 2.04x (10x10, random params) -> gate PASS (marginal at 10x10). Before production use: j1j2 has no ed_cases.json (root CLAUDE.md §4), write one.
- Sampling engineering for the 10x10 stage (PI list 2026-10-30): convergence watcher, resetting stuck chains, larger proposals for
  acceptance near nodes, chains per sample, NetKet / NVIDIA samplers vs our code; opens as an angle with the 10x10 test, first
  input = GPU profile (Paderborn 4075217). Optimizer fixed by the PI: SR / minSR, no optimizer comparison.
- Laws of learnability (PI 2026-10-30, parked for its own angle): why an architecture or input embedding learns a target better.
  Theory exists: NTK eigenspectrum vs target alignment (1806.07572, 2006.13198), spectral bias (1806.08734), embeddings reshape
  that spectrum (Fourier features 2006.10739); NQS sign generalization (1907.08186). Ids from memory, unread here. Our use: predict
  which network stores f_J (spectral-bias guess ~55%, delta 2026-10-29) before training it.
- Loop value vs guide quality and size (PI 2026-10-29): the loop settles at E* ~ floor + T/f (T copy tax per round, f fraction
  of the excess one round removes; O2b: E* ~9e-6 = tax/(1-r), measured). At 6x6 our guides already sit near E* (~2.8e-5 for
  the complex class at s1), so R1/L-SWO cannot gain; at 14x14 trained nets err 1e-4-1e-3/site, so the loop helps if T/site stays
  small. Next question after R1: (a) bad 6x6 start through 3-4 rounds, does it reach T/f; (b) T per site at 8x8. Expert angle
  asked for its view first (PI).
- [KILLED by Test K 2026-10-29] Growing store (Astra cj-6e37615f546c): retained dictionary A_K = |P_0| exp(sum_j c_j f_J[|P_j|]) of independent backflow-free PP
  Jacobi features, walker cost linear in K, cap K. Prediction: 3 features eps^2 <= 0.30 on round 2's correction; kill eps^2 >= 0.70
  or walker cost > 20x plain. After the T3' screen.
- Sign at scale: the network stores only the amplitude, so round k's Krylov sign needs k layers of neighbours (~400^k network
  evaluations per configuration at 14x14): a block for a loop that updates the sign. Options: fix the sign at s2 (current design,
  6x6 cap 0.9e-6), store the sign in a second network (tried: A9/A12/A13 work for the sign when trained on the composite sign with
  flattened |a|^(2 beta) samples; A8 naive fails, 'self-sealing'; A10 per-step factors fail; energy a tie at 6x6), or cache along
  walker paths (untried). Lead's view: two networks, to keep the amplitude network free of a node of its own. (PI + lead 2026-10-28)
- Krylov cost at scale: s2 needs neighbours of neighbours (~1e5 evaluations per configuration at 14x14, s1 ~400); choosing T needs
  E_FN per trial T, possibly from one walker run by correlated sampling (amplitude independent of T; unchecked). (PI + lead 2026-10-28)
- Lanczos alpha needs mu_3 (H applied twice): costly at scale; later, not now (PI 2026-10-28).

## Running (2026-10-30)
- Pfaffian angle: L-SWO round 2 (PI go), arms s1 and s_G, ~1.2 H100-h; kill: round 2 lowers bare B by <= 0.5e-6 -> change the network (Test N / Theta); pass -> round 3 and the scale question.
- Pfaffian angle (its own session, survives the lead's clear): R1 rerun on Paderborn H100 to convergence (job id in
  PLAN_pfaffian.md; A40 R1 17603167 = unconverged, superseded); Test N (nonlinear feature bank, CPU). Angle reports by message.
- Sign-storage angle (session local_3026e44f, started 2026-10-30 on the PI's "start the sign angle", detached; PLAN_sign.md): can a second network store each round's Lanczos sign; sent Test X + R1b results.

## Budget
PI 2026-10-29: GPU budget OPEN until further notice, "wisely spent, no grind": every run follows a written mechanism with a
predicted number (CLAUDE.md points 1, 8); sized from measured rates; reported per test.
PI 2026-10-27 ('go for Paderborn'): test C (complex vs real PP, Pfaffian angle) <= 10 H100-h, single job <= 6 H100-h; rate probe first.
Paderborn re-opened by the PI for the LP test (2026-10-26, 'use wisely'): lead's ceiling 15 H100-h, single job <= 6 H100-h, cont.

## Angles
- 2026-10-30 (PI): Pfaffian angle asked for a learnability prediction (NTK alignment at init vs measured storage of existing fits, CPU) before the next network decision.
- pfaffian-angle (session local_d60cd248, PLAN_pfaffian.md): the project's NETWORK EXPERT from 2026-10-28 (PI): capacity, shape,
  function class; open: can complex PP + 3x3 backflow hold the deep corrections (Tests E-G), and if not which network. Owned stage 1; lead holds authority over its decisions (PI 2026-10-27:
  'you are project lead, you decide'). RUNNING: Test D (CPU bond split of the complex copies' residual, rule in PLAN_pfaffian.md);
  it reports to the lead. Budget left 4.5 of 10 H100-h (Paderborn, PI 2026-10-27 'go for Paderborn', test C line only).
- how-we-work angle (session local_f1c3599f, PLAN_howwework.md once written; PI 2026-10-28): the PI's discussion of method and process
  with a Fable session; no runs; waits for the PI.
- ansatz angle (session local_5c701cde): CLOSED 2026-10-27 (no jobs); session can be archived. conceptual: archived.
- sign-gap angle (session local_6ef8717c, PLAN_signgap.md): CLOSED 2026-10-29, KILL at stage A (see Dead ends); no jobs.

## Dead ends (number, run)
- Nonlinear spin-pattern feature bank (Test N, finite-weight RBM features, M <= 128) for the round correction: eps^2 0.95-0.98,
  only 2-3% beyond 2-body; the correction is not a spin-pattern function at reachable order (pfaffian_n/grade_result.json).
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
- Sign gap delta = (E0 - E_B)/N as a size-converging correction (TRIED J8): fit N <= 32, holdout N = 34/36 misses 5.5e-3 / 1.1e-2
  per site at J2 0.5, same as fitting E0 directly (bar 1e-4); E_B/N is smooth (residual 2e-4), E0/N carries the cluster-shape
  scatter, so delta inherits it; same at J2 0.3 (signgap/A0_ed/energies.json, A1_fits/fits.out; lead re-derived the fits).

## Cockpit
- STATUS (in progress, high): Two label-free loop rounds with the Lanczos sign bring the 6x6 guide to the one-Lanczos-step competitor (5.39e-6 vs 5.57e-6 per site), not yet confirmed (one unconverged fit).
- NEXT (in progress, high): Confirm the crossing with two converged fits, then run round 3 to see whether the gain per round holds.
