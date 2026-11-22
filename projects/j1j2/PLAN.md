# PLAN (J1-J2 FN/Krylov loop) - current state only; history in delta_today.md, experiments/compress_6x6/README.md, RESEARCH_MAP.md
## PI'S CURRENT TASK (verbatim spec; the lead reports "Spec / Ran" against it; a proxy never closes it)
Fit the add-on r on the FROZEN network to the fixed-node correction (target l + c f_J: the Lanczos factor plus c ~ 1.3 Jacobi steps), paper first,
then existing data, then a run. Paper: done 11-17 (frozen base -> no 1/B in the gradient; the 1/B blow-up of a log fit of B itself is DERIVED,
never to be re-tested by a run, PI 11-18). Existing data: done ('ARM H DATA STEP': target ~0 on the bulk, O(1-7) excursions on 20% of orbits
holding 2e-5 of the Born weight). Run: arm H only, 6x6, frozen A_0 x trained ViT add-on, 2 seeds, <= 4 A40-h, graded by retention_N
(STORAGE_PROBLEM 'CONSOLIDATION TEST 6x6 SPEC' grade + 'SCOPE'); pass >= 0.5, kill both < 0.25. FIXED ANSATZ (PI 11-18: frozen pair-product guide x add-on e^r; the ViT as the add-on is the LEAD's choice, 11-17 arm H design, open to change): cx_c1fit log loss on r, same grade at every size; no new knobs per run (6x6 guide's backflow is the recorded exception). Budget raised by the PI 11-18 00:55 (est. 18 A40-h, ceiling 40). Grader break PASS (A1 0.9999, A0 0.0001). Open on paper for 8x8: drawing training rows (the target's weight 2e-5 is never drawn from |B|^2; 6x6 uses enumerated decades). DONE 11-19 02:30 (lead): fix D retention_N 0.8956 / 0.8929 (mean 0.894, BETWEEN 0.844-0.9; broken 0.0065; STORAGE 'FIX D ... RESULT'); round-1 add-on = lin_s1 (runs/j1j2/consol_6x6/ratnet/lin_s1, ratnet_lib 4bd298f). RUNNING 11-19 15:10 (lead): fixed-input add-on fails round 2 (STORAGE 'CAPACITY CALIBRATION, RESULT': same cubic keeps 0.949 of round 1, q_2 0.59 of round 2); paper check 'REFRESH THE BASE EACH ROUND' (spec P's loss still had 1/T^2, corrected), 4x4 known case + data check on CPU running, GPU refit only if both pass. Was 11:40: N(T_2) 2.567e-6 (pred 2.576e-6); round-2 bar q_2 >= 0.85 (Astra); word-moment cubic fits q_2 L1 0.32 / L2 0.59 / L2+ 0.75 (STORAGE 'WORD-MOMENT FIT, RESULT'); CPU capacity calibration on round 1 running (STORAGE 'CAPACITY CALIBRATION', rule there); no GPU staging. Was: round 2 paper check written (STORAGE 'ROUND 2 OF THE COPY', NOT DONE until the known case passes); worker builds the teacher T_2 = M[chi_1] on CPU (known case lA_1 -> lA_2, broken E'), out /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round2/teacher; N(T_2) <= 3.0e-6 -> stage runs/j1j2/consol_6x6/round2/r2_{s1,s2,shuf} (lead reads launchers, then submits); > 3.5e-6 -> locate the rows where the copy's ratio errors enter E_L, no fit. Run: PASS q_2 >= 0.85, KILL < 0.70. Was: NEXT (critical path, paper check first): round 2 on 6x6 with the best round-1 add-on: G_1 = A_0 e^{r_1} (table), target ln G_1 + l[G_1] + 1.307 f_J[G_1] via neighbour lookups as rangelimit_6x6/rounds23, fit r_2 on frozen G_1 with G_1's ratios as inputs, grade retention_2 against N(A_2) 2.416e-6. DONE 11-19 00:10 (lead): log-domain pooling (lse) two-seed mean retention_N 0.844 (0.8590 / 0.8288; broken 0.0072; sum pooling 0.717); BETWEEN the lead's 0.9/0.75 lines. ~70% of the remaining held-out loss is in 154 deep rows with a ratio >= 20 (STORAGE 'LOG-DOMAIN POOLING, RESULT'). DONE 20:50 (lead): PI TASK PASSED at 6x6, one round: ratnet two-seed mean retention_N 0.717 (s1 0.7285 job 18045653, s2 0.7054 job 18049191; broken 0.008). Ceiling: the one-hop formula's held-out loss is 25x below ratnet's (2.32e-4 vs 5.77e-3), and the formula is a function of ratnet's inputs, so the gap is architecture. Diagnosis: sum pooling cannot carry ln sum_b |h_b| rho_b over e^{+-30}. Next: log-domain pooling (STORAGE 'LOG-DOMAIN POOLING', NOT DONE: known case + data check running on CPU), then 2 seeds + broken, pass >= 0.9. DONE 19:45 (lead): RATIO-INPUT add-on (ratnet: per-bond MLP on A_0's one-hop ratios rho_b, sign, bond type, spins; masked sum; MLP; output init 0; Adam 55000 steps) seed 1 retention_N 0.7285 (job 18045653, runs/j1j2/consol_6x6/ratnet/ratnet), broken control (inputs from a same-decade partner row) 0.0080 (18045654); spins-only ViT on the same loss: GN 0.056, Adam 0.057 (adam/full 18045200). Next: seed 2 (task needs two-seed mean >= 0.5) + ceiling check (held-out loss of the explicit one-hop formula, 0.946 retention) to tell fit gap from input gap. Open: at 8x8 with the ppbf guide the inputs cost ~NB A_0 evaluations per amplitude. DONE 15:30 (lead): arm H as specified FAILED its grade: retention_N seed 1 0.056 (400 steps), seed 2 -0.011 (150 steps, cancelled under takeover); held-out flat in both. Ridge scan: steps safe at lam 1000 but empty (-6e-7/step). Diagnosis: the spins-only add-on cannot rebuild A_0's one-hop ratios. Explicit one-hop add-on 0.94 l + 0.72 f_J[A_0] (no training) retains 0.946 (exact grade, job 18042890), certificate fails by 8%. Running: round-2 fixed-input check (STORAGE 'MANY ROUNDS'), pass >= 0.8 -> design ratio-input add-on; <= 0.5 -> inputs must refresh per round. Was: RUNNING 08:15: arm H as Hm_s1 18041502 (cip-cl-h01g05n1), Hm_s2 18041503 (h01g06n1); commit 0d8fb71, sym mean64, chunk_rows 2052, a40-24gb, --mem 41472M, -t 10h; watchers PPID 1, worker holds VERDICT waits; branches in runs/j1j2/consol_6x6/Hm_sN/ANALYSIS_BRIEF.md (PASS >= 0.5 mean; KILL both < 0.25 + convergence stop; step cap + falling loss -> resume; loss down >= 20% with retention < 0.25 -> bound-as-loss paper check; OOM / > 150 s/step -> FAILED). Old held jobs 18039662/18039663 cancelled 08:15 (PI OK). Release conditions below all met 08:15 (STORAGE_PROBLEM step-rejection RESULT DONE, DATA CHECK DONE, probe 18041302). Was HELD 06:00 (lead, PI: order paper -> disk data -> GPU was violated at 02:25): 18039662/18039663 on scontrol hold (cip full A40, est. start 11-19 06:30). Release only when (1) paper check: ARM H block in STORAGE_PROBLEM DONE, step-rejection block still NOT DONE; (2) disk check: first-order term of the FN bound on 6x6 tables (consol_6x6/first_order/, worker) decides log loss vs bound-as-loss; (3) jac chunking (same maths, equivalence test) to run on idle a40-24gb today (24gb probe 18040993 OOM 14.22 GiB). If the loss changes -> cancel (PI OK) and restage. Was: RUNNING 02:25: arm H submitted as staged (spec head, tau 0.05/0.01, commit e3dea56, code sha 40fd57a5): cip A40 jobs 18039662 (s1), 18039663 (s2), est. start 07:03, watchers PPID 1, VERDICT waits armed; branches in H_sN/ANALYSIS_BRIEF.md (PASS >= 0.5 mean; KILL both < 0.25 with flat held-out; step cap with falling loss -> resume; >150 s/step or OOM -> Paderborn H100 same settings, PI yes). tau_adapt CPU test failed its rule (B 256 cannot rank step rules) -> kept off, uncommitted. STATUS 02:00 STEP BACK (second failed head fix): 1. Verdict: switch the fix from the head to the step rule; keep the spec head (image sum). 2. Thought: the body is nonlinear anyway, so a linear head only fixes step 1; LM damping cannot shrink the step (lam ~ tr/R with 60 active directions) and tau 0.05 is inactive at the start, while a standard trust-region radius update (Nocedal-Wright Alg. 4.1, from knowledge) shrinks on rejection and grows on success. 3. Cheapest test: CPU B 256, 30 steps x 3 arms (adaptive tau from 0.005, fixed 0.005, staged 0.05), ~30 min; pass = held-out -5% and below both controls and tau >= 0.02 by step 30. Passed over: logmean/logmean64 heads (depart from the spec architecture; logmean failed the parabola rule 22.5%). Status 11-18 00:30 (fresh lead): output gate w_out CONFIRMED (lead design, with the ViT add-on); grader break A1 0.9999 PASS, A0 running; sizing from the learning need (warm ViT fit t1s2_vit_17418858 flattened only at ~180 steps): cap 400 steps, convergence stop from 200, checkpoint/resume every 50; cip full A40, 2 jobs x <= 20 h (PI cap was 4 A40-h: cost growth to be stated to the PI). Staging worker on runs/j1j2/consol_6x6/BRIEF_STAGE_H.md (branches per verdict there; NOT CONVERGED -> resume, never a kill). Nothing submitted yet.
Side checks (CPU, running detached on th-ws-7010m40/m41/m42, report only if they change arm H): rangelimit_6x6/moving/ (TEACHER CHECK A1),
rangelimit_6x6/node_geom/ (NODE-GEOMETRY CHECK).

## THE DESIGN (PI + lead 2026-11-17; read first, always together with 'Goal' and '## Motivation' below)
Why: beat (plain trained network + p Lanczos steps + fixed node) at equal total cost, training included. The loop stores what extra Lanczos
steps would give, so each round costs one level; p >= 2 steps at the end is the competitor, not an add-on.
Notation: psi_k = S_k A_k; A_k = |B_0| e^{r_0 + ... + r_{k-1}} (B_0 trained once, then frozen; r_j one small 'head' per round).
Round k:
 1. Lanczos on the fly: psi'_k = (1 + alpha_k (H - E_k)) S_k A_k, alpha_k re-optimised for the current state (not fixed). Not stored.
    (Theorist 11-17: a fixed alpha needs 0 < -alpha <= 1/(E_max - E_var), else a 2-cycle between non-eigenstates exists; small fixed alpha
    is safe but gains ~1/N of the optimal step.)
 2. Fixed node with guide psi'_k, sign AND amplitude -> phi_k. Cost: one level (two-hop ratios; 5.4x for the PP at 6x6). Not stored.
 3. Sign: S_{k+1} = sgn psi'_k, stored in the sign network (sign angle). The guide uses the exact one-level sign, so the stored copy serves only as the
    next round's neighbour shadow; judge it by its error at first-shell neighbours: <= 1-2% needed (false flips on keeps dominate: 1% random -> q 0.90,
    5% -> -0.9, one-round-stale -> -10.6(55)). CF copy: 8.5% wrong near the next flips (1.9% average), held-out q_4 -0.3(13), 12 flips, unresolved.
    Limit: generalisation near the last round's changes (PLAN_sign.md STEP 8, 8b).
 4. Amplitude (11-17, PI away, goal run): B10 was NOT run: rung 2b failed (an RBM head stores 2-3% of the correction) and the sampled B10
    gradient is unsampleable at 6x6 (M_req 5.2e9). Rung 2 found the correction is an explicit formula (exact 6x6, fixed sign s2):
    A_{k+1} = A_k |1 + alpha(E_L - E)| e^{c f_J[G]}, c ~ 1.3, keeps 96-98% of each round's FN gain over 3 rounds. PROPOSAL v2 (Astra-reviewed,
    for the PI): docs/theory/2026-11-17_next_method_proposal.md: freeze the network, use the formula as teacher (alpha, c by energy search),
    consolidate a full network with a frozen-relative pair loss, accept only if E_H[chi] < E_FN[g] (budget ~5% of a round's gain). Next:
    rung-2 check with moving signs and energy-searched c (CPU); a 6x6 consolidation run only on the PI's 'now test it'.
    LATE 11-17 (PI 'yes test'): the 6x6 consolidation run was NOT staged: it repeats L-SWO (37%/round, class cannot absorb the correction,
    Theta v2), and the relative loss does not fix 8x8 (|dB|/|B| >= 4 at the worst R1' rows). Verdict on copying: capacity-limited at 6x6,
    coordinate-limited at 8x8 (STORAGE_PROBLEM 'CONSOLIDATION TEST HELD'). Running: NODE-GEOMETRY CHECK (CPU, sign-only features).
    REVERSED same evening (PI): 'capacity-limited' not established (October used the log loss with its 1/B); the 6x6 test is staged with
    arms P (relative loss), C (log loss) and H (frozen base x trained ViT add-on e^r on the formula), STORAGE_PROBLEM 'HOLD REVERSED'.
    Superseded: head r_k by frozen-H_FN Rayleigh (B10 consensus 11-17); head r_k ~ ln|phi_k/A_k| by log loss.
 5. A_{k+1} = A_k e^{r_k}. Optional, every few rounds: merge into one network by the relative-square loss.
End: one final Lanczos step + fixed node for the reported energy; never more.
On paper (STORAGE_PROBLEM.md CURRENT, 'LOOP FIXED POINTS'): with a perfect copy the FN energy falls every round and resting points are exact
eigenstates; with copy error eps_k, descent survives iff eps_k <= Delta_k = E_FN[psi'_k] - E_var[phi_k].
Open questions we work on (PI 11-17, in this order):
 (i) NOW, on paper: Jacobi labels. f_J = ln R only if the neighbours are right, and c ~ 4 is unstable (period-2; stable needs
     0 < c(1 - Vs/K) < 2), so the descent guarantee does NOT yet cover step 4. Repair: damping and the error of the one-step estimate.
     STEP BACK 11-17 (PI): 1. Verdict: continue, reframed: step 4 is 'estimate phi/psi_T for a fitted power iteration', asked before in three
     fields. 2. Thought: c f_J is Davidson's diagonal correction in log form [std, Davidson 1975], (I - P) r = f_J is the Jacobi-Davidson
     correction equation [std, Sleijpen-van der Vorst 1996], and walker forward walking / the mixed distribution gives phi/psi_T exactly with
     noise [std, Liu-Kalos-Chester 1974; Pilati 1907.00907; SHDMC 0808.3417]; fitting each iterate can diverge unless the fit is a
     non-expansion [std, Gordon 1995 'averagers'; 2303.08902]. 3. Cheapest decisive check (only on 'now test it'): 6x6 exact tables,
     exact ln(phi/A) vs c f_J vs (I - P)^-1 f_J, CPU minutes.
 (ii) INVESTIGATED 11-17, revisit later (PI). Ground state vs excited eigenstate (theorist subagent, docs/theory/2026-11-17_loop_convergence.md; lead re-ran check3, check6):
     design B with optimal alpha and a perfect copy descends strictly off eigenstates (no cycles); low excited eigenstates are fixed points
     but unstable (linearised FN map R W has exactly n eigenvalues > 1); NOT from every start: a 6-config start with ground-state overlap
     -0.214 lands exactly on psi_1 in one round (1e-10 noise escapes in 10). Convergence to some eigenstate needs one missing lemma (orbit
     avoids guides with a zero next to one-signed neighbours, or a disconnected kept graph). Certificate: any round with E_FN < E_1 (sector).
     Fixed alpha needs 0 < -alpha <= 1/(E_max - E_var), else a 2-cycle exists. Imperfect copy: limsup(E_var - E_0) <= eps/(1 - rho).
 (iii) Sign copy: the PI with the sign angle (its error enters the same Delta_k budget).
 (iv) Priority C, later research question: training cost as a function of copy quality, counted against the competitor; not computable
      before tests.

NAME CORRECTION 2026-11-17 (PI): our 'Pfaffian' / 'PP' is the Gutzwiller-projected SINGLET pair product, psi_PP = eps det f[Up, Dn]
(Nomura-Imada 2005.14142), the special case of a Pfaffian with only up-down pairing; a spin exchange changes one row AND one column (rank-2,
2x2-determinant ratio). The 8x8 base is B = sum_g phi_RBM(z_g)[det f_1 + lam det f_2] over 256 images, f, b, W complex, so B is complex
(ansatz/code/simple/pp2_lemma.py l.1-12). Angle, directory and file names keep 'pfaffian'. Definitions: STORAGE_PROBLEM.md 'Notation'.
Goal (PI 2026-11-04, restating the loop of 2026-10-30): THE LOOP: guide -> signed Lanczos step -> fixed node -> store the amplitude
in the network AND store the sign -> repeat, at one-level walker cost per round; beat network + Lanczos steps + FN at equal cost
at >= 8x8, usable on any trained NQS at 10x10-14x14. (Old 2026-10-26 wording 'add-on without loop' superseded.)
The refit loop (store each round in the network) is closed: L8, docs/reports/2026-10-26_l8_loop_round_8x8.md.
> CORRECTION 2026-10-27 (index): the 10x10-14x14 add-on test is NOT done: the 10x10/12x12 trainings were cancelled and the Paderborn budget revoked 2026-10-26 (delta_today.md 19:10; checkpoints under rbL{10,12}_prod/backup_paderborn), re-opened only for the LP test (see 'Budget'). The Goal is a target, not a result; the deliverable at 10x10-14x14 remains pending, subject to the lead's budget (TRIED_IDEAS.md Contradictions item 12).

## Motivation: the questions the project must answer (PI 2026-11-12; steer by them, not a task list; lead's current one-line take)
1. How does it work? Trained net guides FN; one Lanczos step corrects; the correction is stored in the net (Jacobi layer); repeat.
2. How much better than a plain NQS? Old 8x8 net: VMC -0.498674 -> p=1 FN -0.498946 (~half the error); strong net s306: p=1 pending (headline B arm).
3. Our net vs a plain NQS? Same published class (RBM+PP); the difference is its use: FN guide + stored Jacobi corrections, not the final state.
4. How hard is the sign/node to learn? Hard: error on ~0.1% of bonds at the node; 12 trained 8x8 signs worse than untrained; residual flip mask on a strong base unmeasured (T-STRONG).
5. Scaling? Little known: one Lanczos step removes 84% (6x6) -> 47% (8x8); nothing above 8x8 except 10x10 ViT FN p=0 (+5e-6 only).
6. Do we escape a wrong state? Partly: T-BAD from Marshall gained 2.6e-3, 5e-4, 5e-5 over 3 rounds, stalls 4.5e-5 above the net's own sign; FN alone cannot move a node.
7. Problems? Copy weighs |psi|^2 where FN needs the near-node bonds (cousin of the SR 'wrong samples' issue); measurement noise ~ the effects (1e-5); Lanczos step cost ~137x plain sampling at 8x8.
8. Worth it for a trained NQS without retraining? FN p=0 alone small and shrinking with net quality (8.2e-5 old, 4.0e-5 strong 8x8, 5e-6 10x10 ViT); the p=1 step is the main win; strong-net p=1 pending.

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
  p=1 FN is 6.2e-5 below published full-projection RBM+PP (-0.498886) and 2.5e-4 above DMRG (-0.4992).
  > CORRECTION 2026-11-02: Varbench (arXiv 2302.04919) lists RBM+PP K=0 at -0.498963(2) (via ketworks; tools/ketboard.py --L 8 --bc pbc):
  >   our p=1 FN -0.498946(5) is 1.7e-5 ABOVE the best published 8x8 value, not below it (Varbench file itself not yet opened). Cost: training
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

## OPEN QUESTION (PI 2026-11-02) -- ANSWERED 2026-11-03: NO at p=0. FN p=0 from the ViT -0.497682(4) vs VMC -0.497677(5): gain 5(6)e-6, 3 sigma
above the bar (docs/reports/2026-11-03_addon_10x10_vit_fn.md); the ViT's error is in its sign, which p=0 cannot change. Original question: does the add-on improve a strong published 10x10 network below the
best published bound -0.4976939(2) (arXiv 2603.14425)? Guide: published ViT nqs-models/j1j2_square_10x10_05 (-0.497676).
Mechanism (hypothesis): FN p=0 removes the guide's amplitude error except across sign-violating bonds; at 8x8 FN p=0 gained 8.3e-5
over VMC p=0 (RBM+PP). Prediction (unvalidated): FN p=0 from the ViT lands -0.49770 to -0.49775. Kill: not below -0.4976939 by
2 sigma at two-chain error <= 5e-6. A0.1-A0.3 DONE: our JAX copy (f7cdb0c) VMC -0.497677(5) two chains (vit10_runs/a03), f32h + f64 recheck of |r|<1e-3 passes a
held-out FN-precision test. A0.4 (FN p=0 driver, validation) running. Old step list: A0.2: reproduce its VMC energy + H100
throughput; A0.3: FN p=0 pilot sized from A0.2. C (Pfaffian angle): two-PP sum + Jacobi level at 6x6, ~12 H100-h.

## OPEN QUESTION: CLOSED 2026-11-01 (answer below); the PI picks the next one from the Queue / scale designs
ANSWER: the stored network cannot carry the Jacobi correction in its tangent space (Theta v2: 0-4%); with the Jacobi factor kept as an
explicit one-level layer and label-free refits under the Lanczos sign, the loop banks ~3 rounds and saturates (round 4 drop 0.34e-6,
kill fired). Best one-level guide at 6x6: 4.37e-6/site (aug, theta_4 averaged; pfaffian_r4) vs competitor 5.57e-6 at the same sign;
deployable only with a stored sign (sign angle: XL-class stored sign costs 3.6e-7 inside this guide; V1 decides whether to store).
Report: docs/reports/2026-10-31_label_free_loop_6x6.md.

## (closed) OPEN QUESTION (the one, CLAUDE.md point 4; set by the PI 2026-10-29)
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
CURRENT 2026-11-17 (theorist mode, project CLAUDE.md point 9): the open question is on paper, in STORAGE_PROBLEM.md 'CURRENT 2026-11-17'.
PI's reformulation: freeze B_k, store B_{k+1} = B_k e^{r_theta}, plain regression r_theta ~ c f_J (|f_J| <= 0.14 at q.9999); no run until the PI says 'test it'.

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

## Step back 2026-11-01 (stored sign; PI request)
1. Verdict: continue, change the TRAINING before the class: the deployable PP phase is under-trained and trained on a misaligned
   objective (uniform BCE vs FN tax weighting blocked pairs; 60% of its wrong-sign weight lies outside its training set); no
   representability barrier shown (Astra cj-ffe80dbfda01, EVIDENCE.md in /scratch/a/A.Otaifi/astra/signstep).
2. What changed it: Astra's exact-table checks; lead's blind guess (compute s_G on the fly, LEAD_BLIND_GUESS.md) is untested,
   Astra finds selective recomputation ~6.4x shell work at 14x14 but did not test full s_G with cheap-sign walls.
3. Next (sign angle): CPU check of on-the-fly s_G with sgn(psi_P) walls (pred 5.0-6.0e-6), then edge-weighted PP vs matched
   restart (<= 1.5 H100-h; kill: no gain over restart or tax >= 0.93e-6).

## Step back 2026-11-03 (f_J defect; PI request)
1. Verdict: a refinement, confirmed for all 8 quoted guides (audit 17688846, pfaffian_fjaudit/fj_audit.json: capped e moves <= 1.5e-10,
   one no-kept row at weight 1.5e-19, largest orbit share 0.0093 = a bulk orbit). The defect bites only after sign changes (sign angle). f_J is one Jacobi step of the FN equation; on rows with no kept
   bond the correct value is -infinity (isolated row, phi' = 0), not 0; on tiny-B rows the global over-relaxation c > 1 amplifies huge
   local corrections (nonlinear SOR instability). Fix: node or floor on isolated rows, per-row relaxation c_eff = c min(1, f0/|f_J|).
2. What changed it: the sign angle's collapse case (one orbit 1-1e-9 of the weight); reported loop energies remain exact FN bounds.
3. Test: audit + fix on all quoted guides (Pfaffian angle, CPU); kill "refinement" if fixed guides move > 0.2e-6 or the two-PP gain
   shrinks > 20%.

LESSONS: projects/j1j2/LESSONS.md (PI 2026-11-05; overrides root CLAUDE.md §8 for j1j2): read before designing any test; add a row when a lesson costs a run.

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
- Two label-free rounds under the Lanczos sign beat one Lanczos step at 6x6 (R2G-conv, 2 seeds, exact FN): augmented guide
  5.207e-6 / 5.290e-6 vs competitor 5.574e-6 (same sign s_G; spread 0.08e-6 < margin 0.28e-6; pfaffian_r2g/readout/g{1,2}).
  Caveats: seeds share warm start and target; s_G is the competitor's sign; walker step ~5.4x plain; 6x6 only; equal-cost not shown.
- Round 3 under s_G keeps gaining (single-iterate r 0.145 -> 0.11, but averaged r_3 0.178 and seed spread make the decline NOT
  established, q 0.63-0.95; bare 1.418 -> 1.313e-5 at step 1200; official aug 4.786e-6 at c* 3.9;
  Polyak average of tail checkpoints 1.252e-5 bare / 4.644e-6 aug, better than every single iterate; pfaffian_r3avg). At a fixed target E_FN walks +-0.2e-6 per 150 fit steps (copy loss flat there): fit-to-fit
  scatter of that size limits any single-fit number (pfaffian_r3, pfaffian_r3c).
- Deployable loop guide = S B e^{c f_J[B;S]} with a STORED sign S (sign angle 2026-11-01): computing the Lanczos sign at walker time
  and feeding it into the Jacobi walls costs d^3 (three swaps), so s_G must be learned offline. The plain Lanczos magnitude fails with a
  stored sign (R0: 1.50e-6/site from rho = 2|alpha|kappa - 1); the Jacobi magnitude does not carry that term.
- The loop's advantage needs a stored sign (sign V1): with the free one-level sign sgn(psi_P) the best loop guide is 8.76e-6, above the
  competitor (5.57e-6); s_G is worth 4.1e-6 (sign_v1). S2a: an XL CNN storing s_G costs 4.2-4.3e-7 (two seeds;
  sign_s2/v2_S.json), stored loop 5.07e-6 < 5.57e-6. DEPLOYABLE sign (complex-PP phase, det4 updates):
  tax 1.63e-6 -> loop 6.27e-6, ABOVE the competitor (S2b, unconverged; S2b' to plateau running).
- A sum of two pair-products lifts the stall (round 5, 2 seeds, pfaffian_r5): stores r 0.49 vs 0.046 for one PP; augmented guide
  3.26-3.29e-6 vs competitor 5.57e-6 (same sign). Round 6: r 0.25, aug 2.81e-6 (pfaffian_r6). Walker step 10.9x plain PP at 10x10.
  STRUCTURE, not parameter count (M5' PP+RBM same budget, 2026-11-05/06; lead-verified readouts): r_5 0.122 / 0.127 at 1500 / 2100 steps vs two-PP 0.493; aug 4.01e-6 vs 3.26e-6; E_FN plateaued, extension stopped by the lead.
- A final signed Lanczos step on the STORED network beats the Jacobi head (T-B2P1, 2026-11-05; lead verified sign_b2p1/*.json; exact FN, 6x6,
  moment alpha, c = 0): s_G|B_6| + step 2.646e-6 (< aug 2.814e-6; ~ published CNN-MPS VMC 2.654e-6), s_G|B_5| + step 3.211e-6,
  sgn(psi_P)|B_6| + step 3.705e-6 vs competitor 5.574e-6; sgn(psi_P) + head (c* 4), no step 7.202e-6. Mechanism of step > head NOT understood.
- 8x8 loop, one round (2026-11-09; lead verified headline_p1/prodA ANALYSIS): R1 at fixed sgn(M1) gains 4.65e-5 bare (-0.498803 vs -0.498755);
  with a final p=1 step: B_1 -0.4989545(34) vs B_0 -0.4989390(45) (same code), A - B -1.55e-5(5.6) 2.7 sigma; vs M1 p=1 -8.5e-6 (1.4 sigma):
  registered KILL fires (not 3 sigma). Mechanism (measured): the step gains 1.83e-4 on B_0 but 1.52e-4 on B_1, so ~1/3 of R1 survives:
  refit (Jacobi) and Lanczos step correct largely the same error. Caveat: R0 (same code) -0.498968(6) is 3.8 sigma from arm B; block errors on
  heavy-tailed p=1 energies may be underestimated (B chi2/dof 0.06); pooling R0 gives A - B ~ -0.5e-5.
- Storage survives sampling at 6x6 (V, 2026-11-04; lead read pfaffian_v/readout/readout.json, Paderborn express fit 4109778, 1500 steps,
  control theta_4 exact): two-PP round 5 fitted from 4e5 sampled configurations stores r_V 0.464 vs 0.493 from the exact table (ratio 0.94;
  bare 8.561e-6 vs 8.332e-6). PASS (>= 0.35). Gap 0.029 ~ one fit-to-fit scatter unit (+-0.2e-6 in E). Aug (c* 4.0) 3.346e-6 vs exact-table 3.258e-6 (+0.088e-6, < fit scatter; readout/aug/readout_aug.json, lead read). c4 replica, new pool (pfaffian_v_c4/readout/readout.json, lead read): r_V2 0.4397, aug 3.390e-6; two pools: r_V/r_5 = 0.916 +- 0.024.
- Deployable stored sign on the two-PP loop guides (lead read sign_comb/r6grade/v1_sG.json, v2_S_a1.json; exact FN, 6x6): S_a1 tax 0.88e-6 on
  R6 theta_6 (aug 3.70e-6 vs s_G 2.81e-6) and 0.86e-6 on R5 (4.12 vs 3.26e-6): stored-sign loop 1.88e-6 below the competitor 5.574e-6 at R6.
  Caveats: one sign seed (seed 0 was 0.6e-6 worse on the 1-PP guide); exact tables; competitor on RBM+PP base (Queue: bar).
- At 6x6 the second sign step gains < 0.5e-6/site from all four learned starts (re-graded with fixed f_J, sign_fix/fix_B{1,2}.json:
  B1 -4.3e-7, B2 -7.6e-8, B2 refreshed +3.9e-7; first steps +1.1e-5 / +5.3e-6; top orbit share <= 9.3e-3). Larger L untested.
- At 6x6 the amplitude, not the sign, limits the loop (pfaffian_floor): s_G floor <= 0.83e-6 (Aitken 0.39e-6) vs best guide 4.37e-6;
  the exact ED sign improves theta_4's augmented guide only to 4.15e-6. Even the exact sign blocks 44% of bonds (16.8% of the
  |H|psi0 psi0 weight, mostly J2), so FN error there is purely amplitude-ratio error; psi0 itself is the fixed point (7e-16).
- The round correction is not in the complex PP's tangent space, so a local refit cannot absorb it (not slow learning) (Test Theta v2, held-out, calibrated: C_GN
  -0.02..0.02, C_200 0.00..0.04; gate C_exact 0.082 vs held-out 0.065-0.069; pfaffian_theta2): the base cannot absorb it by a local
  refit, so the explicit Jacobi factor carries it; the nonlinear label-free refit still banks 11-18% per round (consistent with K, K3, N).
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
- 10x10 stage (PI agreed 2026-11-11): gated on the 8x8 retrained-base rounds accumulating. If they do: (1) pp2_lemma multi-word key
  (Pfaffian angle, queued), (2) 10x10 speed test < 1 H100-h, (3) budget decision with the PI (one round + headline ~83/248/500 H100-h;
  base rbL10_prod params_04150, 3.6e-4 above published; proposals/2026-11-11_10x10_prep.md). If not: 10x10 not worth the spend.
STEP BACK 2026-11-12 (PI: 'am I jumping ahead to the full loop?'):
1. Verdict: REFINE FIRST, full loop after. Order: (a) the ruler: paired FN estimator (sign angle, Astra cj-c9a7d4cfa462, 6x6 gate, ETA ~11-14);
   (b) the size half: R1-R3 at own sign (does the Jacobi copy accumulate at 8x8 after p=1?); (c) the sign half: T-STRONG stage 0 (false-flip rate
   of a residual mask on s306). Full loop once (a) passes and (b), (c) are each a readable yes.
2. Thought: every deciding difference is ~1e-5 while unpaired errors are 4e-6 (p=0) to 1.3e-5 (sign FN) and the base itself settled by ~1e-5;
   a full-loop round mixes three unknowns (sign storage, Jacobi copy, overlap with p=1), so a miss could not name the failing term (agreement point 1).
   Unasked: if a round is ~ one stored Lanczos step, the honest equal-cost bar is B_k + p=1 vs B_0 + p=(k+1) at p=1's cost.
3. Cheapest next test: the paired estimator's exact-6x6 gate (<= 48 core-h CPU + build); it sets the resolution of every later grade.
   AMENDED same day (sign angle's view, lead agrees): not serialized. R1-R3 runs now unpaired (p=0 grades ~1.2 H100-h/arm for sigma(diff) 2.5e-6);
   early rho gate after estimator stage 1: ~0.25 A40-h coupled 8x8 probe (s306 vs s306 x random sparse flip mask + A=B arm), stop the build if rho < 0.5;
   guesses rho: sign-only mask 0.6-0.9, size round 0.3-0.6, p=1 pairs < 0.3. If low: grade bigger effects (cumulative sign rounds, weaker start, 10x10).
STEP BACK 2026-11-11 (big picture, PI request, not a failure):
1. Verdict: CONTINUE R1-R3 as registered; move the base's FN arm (B_0' FN p=0, then p=1 in the headline) forward: it answers 'below every published
   8x8 number with the given sign' (Varbench -0.498963(2), plain VMC) and is the bar every round must beat. Name the sign question as the gate of 'better every round'.
2. Thought: a fixed-sign amplitude round and the Lanczos step both apply H once, so the round's value vs B_0'+p=1 is at most ~ one extra Lanczos step
   (old base: 1/3 survived). Lasting gains need the class to grow per round (CIPSI-DMC, Caffarel et al. 1607.06742, abstract only: nodes improve
   by adding selected terms each iteration, = our fresh PP per round / K=3) and a sign that ends BELOW the network's own; T-BAD so far climbs
   toward sgn(M1) from Marshall (4.5e-5 above after 3 rounds, contraction x6 -> x2), and 12 trained 8x8 signs were worse than the untrained one.
3. Cheapest next test: B_0' FN p=0 at the 'simple' projection, ~1 H100-h (unsized; Pfaffian angle sizes it, reusable as the headline's base arm).
   Prediction: -0.49894..-0.49897 (old base FN p=0 gained 8.3e-5 over VMC; gain shrinks with a better base). Nobody has published NQS-guided lattice FN for J1-J2 at 0.5.
STEP BACK 2026-11-10 (8x8 base retrain; second failed version: shift scan then lr scan, 280-step warm starts, 2.4 H100-h):
1. Verdict: SWITCH from short warm-start tuning scans to one standard production retrain; short scans cannot separate "slow" from "stuck".
2. Thought: shift and lr are not the limit (gains 1.5-2.3e-5 +- 1e-5 per 280 steps in every arm, single chains = noise level); the loop needs a
   PP-class base (the refit stores into PPs), so a published ViT/CNN is no substitute; the 0.1e-3 projection gain is free and already measured.
   SUBMITTED 2026-11-10: Paderborn 4191676 (s306) / 4191677 (s307), default QoS -> cont, est. start ~11-15, ~10 h each; kill at step 1000
   (no 200-step mean < -0.4988225); watchers write VERDICT lines to delta_today.md; run dirs /scratch/a/A.Otaifi/rbmpp8/rb8_retrain/<seed>.
3. Next: production retrain, simple projection, lr 0.05 -> 0.005 decay, 16384 samples, ~3000 steps, 2 seeds, ~10-15 H100-h on December express;
   meanwhile check for released published RBM+PP 8x8 weights (minutes). Kill: no 200-step mean below -0.4988225 within 600 steps.
- NOW (lead decision 2026-11-09, PI "you are lead and decide"): retrain the 8x8 base to published quality BEFORE more loop rounds.
  M1 VMC -0.498674(12) is 2.1e-4 above Nomura-Imada RBM+PP -0.498886(1) (2005.14142); the loop must be tested on a strong guide (Goal: add-on
  to strong networks). Step 1 (worker): ours-vs-published setup table, probe, plan (proposals/2026-11-09_base_retrain_8x8.md). Then rerun
  R1 + headline on the new base. Overlap analysis (CPU) runs in parallel.
- K = 3 at round 6 (architecture angle, PI "yes" to queue, 2026-11-09): why does the two-PP stored share fall 0.49 (R5) -> 0.25 (R6)?
  Hypothesis: P_2 spends its freedom on the round it first stores. Test: R6 from theta_5 with a fresh P_3, same T_6/budget, exact 6x6, vs K=2
  r 0.25 / aug 2.81e-6 (pfaffian_r6). Pred ~0.45 (55%%) -> one PP per round, cost ~linear in rounds; <= 0.30 -> decline in the fit, not the class.
  Cost 3-7 H100-h (normal QoS or December express). Lead: worth it (tests the only cheap route left); after the 8x8 headline.
- Sign tax over many rounds (sign angle 2026-11-04; the PI's 14x14 worry, errors accumulating over sign updates): in the 8x8 loop build,
  re-store the sign after every sign update and grade the tax per round; flat = OK, growing = the sign must be re-trained from scratch
  or the class changed. Lead: include as a per-round readout of the 8x8 loop (no extra run), not a separate test.
- Competitor bar CORRECTED 2026-11-14 (write-up angle caught it, lead verified in both papers): the strongest published 6x6 network is the ORIGINAL CNN-MPS D=20, -0.50380737(5) = 2.28e-6/site above our E0 (2603.14425 text, Fig. 2); the line below is LaQX's re-implementation (2611.04333 Sec. 4.2), not the bar. Our K=3 guide 2.546e-6 is 0.27e-6 ABOVE it.
- Competitor bar (LaQX review 2026-11-04, PLAN_tooling.md): published plain VMC CNN-MPS at 6x6 is -0.5038070(2), 2.65e-6/site above our E0 -18.137147540071613/36
  (arXiv 2611.04333 SciPost TeX l.763; CORRECTED 2026-11-04 by the Opus review, lead re-derived: 3.2e-6 was vs the paper's rounded ED), BELOW our best loop guide
  (2.81e-6 exact sign; 3.56e-6 stored sign) and
  well below our competitor 5.57e-6 (RBM+PP base). Any loop-advantage claim must face strongest-network + Lanczos + FN at equal walker
  cost (CNN-MPS has no fast ratio path: generic-NN step cost ~2N). Lead view: not a kill, a sharper bar; decide after V.
- Amplitude network for the 14x14 loop (PI + lead 2026-11-05): spec = a two-swap ratio at ~constant x a plain step (low-rank or local update)
  AND the Jacobi correction within reach of a refit. PP: cheap, holds 25-49%/round. CNN: plain re-eval ~2N x, but finite light cone ->
  cached local update possibly constant (worker reading LaQX cnn_mps.py: proposals/2026-11-05_cnn_mps_twoswap_cost.md, future input).
  ViT: global attention, every layer changes all tokens -> no cheap update; as a node sign with PP walls FAILS (T-HYB, A45: hybrid tax 0.998e-6 > PP 0.883e-6). Own architecture = this spec
  (the earlier network-angle attempt lacked it); exploratory, parked.
- Local dressing that keeps cheap determinant updates (PI 2026-11-03): CNN-MPS (2603.14425) gains from a local disentangler (backflow);
  our backflow PP killed the cheap Jacobi step (~800x). Question: a dressing that changes only the swapped sites' pair amplitudes, so a
  two-swap stays a low-rank update. Paper-only look by the Pfaffian angle after the B design.
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
  input = GPU profile (Paderborn 4075217). Optimizer fixed by the PI: SR / minSR, no optimizer comparison. If 10x10 VMC/FN chains
  stick or disagree: try NetKet parallel tempering first (tooling venv /scratch/a/A.Otaifi/venvs/tooling; PLAN_tooling.md).
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

## Running (2026-11-16 09:45; every job has a watcher writing VERDICT lines to delta_today.md)
- 2026-11-20 (3rd): worker exact A_3..A_5 (ref2hop/exact) + arm R round 5; then compare copy vs exact per round.
- 2026-11-20 (2nd, done: R PASS eps 0.170/0.253, N(chi_4^R) 1.9482e-6): worker arm R (refreshed one-hop reference) + round 4 for B and R, same dir (STORAGE "REFRESHED ONE-HOP REFERENCE"); KILL eps_3(R) >= 0.37e-6 -> stay with B; else round 4 decides the trend.
- 2026-11-20 (done, B eps 0.209/0.393 BETWEEN): worker CPU arms A/B/K/S in /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop (STORAGE "FIXED-REFERENCE TWO-HOP INPUT"); PASS eps_3 <= 0.30e-6 -> GPU ratnet with fixed two-hop inputs, warm-started; KILL eps_3 >= 0.45e-6 -> step back (lever is not neighbour spread); K misses N(T_2) by >1% -> fix features first. Ratnet r2_s1 done (N 3.3865e-6, refit from scratch, off-path).
- 8x8 base s306 (sha 45fa54e6) / s307 (b86a0f36): full projection -0.4989092(14) / -0.4989096(14); base FN p=0 -0.4988780(27) (4222088; may be
  ~1e-5 high, not reused in round comparisons). Bars: Nomura-Imada -0.498886(1), Varbench -0.498963(2).
- s306 p=1 constants DONE (theorie a40-24gb 17931202, check.txt PASS): alpha -0.17605(337e-5), E -31.926142, w -1; B0 control z +0.3/+1.2;
  runs/j1j2/headline_s306/inputs/consts_s306_p1.json sha 54add61c (copied to Paderborn, same sha). The ONE set; full-A40 copy 17930563 HELD (never to run).
- HEADLINE B ARM DONE 2026-11-15 03:41 (Paderborn 4222359, started early; runs/j1j2/headline_s306/fn/prod/{ANALYSIS.md,raw_E_HS.png}): NOT QUOTABLE.
  Chains (E/site) -0.4989834(58) -0.4989707(51) -0.4989853(58) -0.4989603(45), chi2/dof 5.7 (3 dof). Lead's check: fifths trendless in every chain, binning
  flat ~5e-6 to bin 40 (bin 80: 8.7/3.8/4.9/1.8e-6), lb_med / deep-decade share / elvar do not track the offset. So a chain-level offset up to 2.5e-5 that
  persists 40 tau: a mode slower than the run or start-dependent bias, cause NOT understood. Scatter-only reading -0.498975(6) (Varbench -0.498963(2):
  -1.2e-5, 2.1 sigma), labelled not quotable. Paired grades (A and B share walkers) are protected; absolute FN numbers from this engine are not, at 1e-5.
  Next (PI decides; not on the critical path): extend the same 4 chains +400 blocks (~12 H100-h, normal QoS). Slow mode -> chi2/dof falls toward 1;
  start-dependent trap -> offsets persist. Write-up told not to quote it. Heavy-tail E_L hypothesis dead (top-5 blocks removed: <= 4e-6); k0 fifths span
  4.3e-5 vs 9.6e-6 expected, so a slow mode exists inside a chain too.
- R1' (open question's size half; PLAN_pfaffian 'R1-R3 ON THE RETRAINED 8x8 BASE'): warm start (ii) = B_0' + lam s307's PP at 0.05 (held-out
  tangent share +0.32 vs (i) -0.40); grade = R1' + warm-start arm (no fit) + base re-run, burn 150, halves check, one job; loop gain = R1' - warm.
  SEALED, corrected twice BEFORE data (66550e6; 97c4ba9 full held-out): (1'') warm vs base -1.2(1.4)e-6; (2'') loop R1' - warm -4.6..-8.3e-5
  (extrapolation). Labels 17985658 gate PASS (bond ESS min 59171, pool sha 2141b1e3). FIT DONE 2026-11-15 07:35: Paderborn 4255548, 1500/1500, theta_tail
  sha 15cff596 (slice 17990426 cancelled by lead, vGPU time-sliced to 237 s/step). Held-out (424585 rows, paired, bootstrap): step 150 D -0.0154(85),
  step 300 -0.0013(62) (stop rule not fired), end +0.0011(110) UNDECIDABLE -> grade decides (end rule 8267af0). Trimmed (5 near-node rows fixed at
  step 150) end +0.0167(23): derived R1' - warm ~ -1.4e-5 (unvalidated), 3.5-6x short of (2''). Near-node sensitivity d ln|P_1| ~ 1/P_1 confirmed
  (tangent 26-73x), transient not runaway. GRADED 2026-11-15 12:06 (Paderborn 4271870 + 4271928, 3 arms x 8 chains; lead re-derived from E_mix.npy; report
  docs/reports/2026-11-15_r1prime_grade_8x8.md): LOOP GAIN E(R1') - E(warm) = -1.20(23)e-5/site, 5.2 sigma, 8/8 chain diffs negative; warm - base
  -0.04(31)e-5 (= sealed 1''). Sealed (2'') MISSED at 0.14-0.26x; the trimmed held-out proxy predicted it (0.5 sigma). Missed term: stored fraction f 0.145
  vs 0.51-0.90 allowed; cause NOT understood -> Pfaffian angle diagnoses f on paper before R2'. R2' early kill |gain_2| < 6.0e-6.
  f DIAGNOSIS 11-16 (PLAN_pfaffian d157ddc, 1744f4c, 78722a6): full-train f 0.096(29) ~ held-out 0.13 (not generalisation); GN tangent share C_full NOT
  identifiable (held-out C <= 0.011; in-sample = optimism, shuffled control 0.11-0.39); loss-weighted ESS 2463 (theta_0) / 640 (theta_end) rows vs P 18466.
  Loss-ESS vs pool size: -8..-2 near-linear, decade -9 saturates only at theta_end (exponent 0.78 -> -0.7): the FIT creates the heavy near-node rows
  (d ln|P_1| ~ 1/P_1) that then own loss and gradient (hypothesis, tested by this scaling). NEXT (Pfaffian angle, paper, no run): predicted f and the
  -8..-2 cost for a bounded per-row loss (asinh(B/s), B/(|B|+eps)) and a per-row |d ln B| trust region; to the lead before any refit.
  REMEDY CHOSEN 11-16 10:35 (PLAN_pfaffian 863fc55, diag_ho/remedy_counterfactual_tr.json, verified): counterfactual on 373k train rows, bond f
  0.020 realised -> 0.267 with per-configuration trust region |dlnB| <= 0.5 (no loss at -8..-2 except -5 0.30 -> 0.25); asinh 0.05-0.06; B/(|B|+eps) ~0.
  Function-space idealisation, untested as a parametric fit. NEXT: R1'' = R1' pool + warm start + trust region only (G known). Fit kill: held-out
  bond f < 0.12 -> no grade, diagnose. Grade R1'' vs warm, 8 chains: pred -3e-5 (-2..-5e-5); confirmed iff E(R1'') - E(R1') < 0 beyond 2 sigma,
  else to the lead. Then R2' from the better round-1 net, same trust region, G_2 measured first. Launchers to the lead.
  Kill for R2': loop gain < 0.5 x previous. Fresh PP per round from R2' on (K=3 6x6: r 0.49 vs 0.25); its source is open (s307 now used).
  R1'' PROBE RUNNING: Paderborn 4299397 (submitted by the lead 16:42, runs/j1j2/r1pp_fit/probe), 60 steps; OK on (a)-(e) -> R1'' fit (LEAD_OK in r1pp_fit/pdb); STALL -> q9999 backtrack + per-row clip, CPU first; BOUND EXCEEDED or held-out rise >= R1''s -> lead.
  PROBE 4299397 = STALL 11-16 16:57 (TRIED_IDEAS P37): accepted 4/60; off-batch q999 0.36-0.40 OK; held-out bond 0.510 -> 0.680 after 2 accepted steps
  (R1' mechanism again). Third failure of the batch-enforced trust region. Candidates: (A) active set via full-pool sweeps (angle, ~50%; lead ~35%:
  nodes move, violators grow with the cumulative step; kill: > 1000 full-pool violators at step 10 or growing); (C) lead: freeze theta0, fresh PP
  inside a bounded ratio B = B0 exp(tau tanh(lambda P_n/B0)): bound by construction, nodes fixed, non-local (escapes AR2). Astra review
  cj-9ad7e6a1e027 pending; decision after it.
  DECISION 11-16 17:20: (C') clipped deployed guide ln|B~| = ln|B0| + clip(ln|B_R1'| - ln|B0|, +-0.5) on R1''s endpoint, no refit (Astra cj-9ad7e6a1e027
  reproduced train bond f 0.0197 -> 0.2667; registration PLAN_pfaffian 'REGISTRATION R1'-CLIP'). CPU held-out gate f >= 0.12 (pred 0.23) -> grade
  (~8 H100-h, pred -3e-5 vs warm, confirmed iff below R1' beyond 2 sigma) -> lead reads the launcher. Below 0.12 -> stop. (A) dropped (Astra: |m| = 3
  persists, 3.8x cost); (C) = R2''s trainable form if (C') confirms. R2' on hold.
  CLIP GRADE 11-17: Paderborn 4306152/4306153 PENDING (cont QoS; express is out of GPU-minutes), start est. 19:00, 4.47 H100-h, runs/j1j2/r1p_clip_grade,
  watchers 1897015/1897330 (setsid, write VERDICT to delta_today.md). Lead 11-17: patches the step, not the loss (d ln|B| = dB/B at nodes the fit moves);
  recommended cancelling; the PI's OK is needed and was not given (asked 11-17 12:25). Branches (registered 11-16 22:46): CONFIRMED -> R2' design to the PI
  on paper (theorist mode, no run); NOT CONFIRMED -> diagnosis on paper only.
  STEP-BACK 11-16 15:25 (second failed controller: soft hinge overshoot, then Jd bound refuted, PLAN_pfaffian 'KILL TEST RESULT', P36):
  1. Verdict: continue with backtracking on the actual per-row m (TRPO line search, 1502.05477). 2. Thought: near a node the
  quadratic term of dB/B carries the same 1/|B| as the linear one, so only the true m can be bounded; the stall risk comes from a hard max over
  18466 rows (PPO clips per row instead). 3. Next test: 60-step H100 probe, 0.32 h; it goes to the fit only if (a) off-batch q999 |m| <= 0.6,
  (b) s/step x1500 fits with 1.2x margin, (c) median halvings <= 2, (d) held-out bond loss stays below R1''s rise 0.53 -> ~2.0 at steps 20-60,
  (e) dry diff clean. Stall branch: backtrack on q9999 + per-row gradient clip, CPU step first.
  For the PI next message (lead's view, ~40%, off path until the probe reads): three controllers have now failed on the same ~80 near-node rows of
  18466; if the probe stalls, the fix is more likely in what the copy loss asks of those rows (their energy weight) than in a fourth controller.
  Test: their |psi|^2-weighted share of the FN energy change, CPU minutes.
- Sign angle (PLAN_sign.md): paired FN estimator PASSES at 8x8 (rho 0.978 at pF 1e-5, 0.79 at 1e-4, as sealed). Symmetry branch closed
  (TRIED_IDEAS A51: D4 label obstruction <= 0.1% of the stored-sign error). T-STRONG blocked by the mask (no capture at FP/TP <= 1); ceiling first:
  G(-5) DONE: KILL FIRED, Delta +6.4(3.5)e-6 (Astra cj-8a965a47de4e: stands for storing Lanczos labels; NOT a ceiling over masks). T-VARDEC (17994730)
  branch C: trial-energy cost +2.44(25)e-5, 98.6% window flips; F' gain -0.56(14)e-6 (1/9 of the bar). Stored masks at 8x8 closed OPERATIONALLY.
  T-BAD10 (10x10 bad-start sign loop, PLAN_sign e15d1af/c72dd75): round 1 labels 18007712 RUNNING, fit 18007713, fn64 18007714 (afterok);
  alpha DECLARED -0.170 (exact two-hop too heavy-tailed: -0.163(65)); both FN arms at M=64 (M16 population bias ~few 1e-4); resolved iff
  g_1 - B64 > 2 sigma, bias-limited -> inconclusive (not a kill). T-TAN: fixed CNN class NOT shown limiting (C 0.196 -> 0.142; collapse in -9..-5).
  T-SEP exact arm unaffordable (~125 A40-h) -> 6x6, deferred; grade4 = 8x8 T-BAD round 4 paired n_4 vs n_3 (~6.4 A40-h, approved 11-16,
  pred g_4 1.0e-5 [0,3e-5]; cip 18007736 RUNNING; runs/j1j2/sign_bad/r4/grade4). G_max parked. PLAN_sign "G(-5) RESULT", "T-VARDEC RESULT".
  CONTROL C DONE: loop beats SR of the same sign class by 1.058(18)e-3/site at matched compute (8x8, from Marshall).
- Write-up angle (local_71c014c5, PLAN_writeup.md): progress deck v1 published; 6x6 bar corrected to 2.28e-6 (2603.14425).
- Closed 11-12: lead's second-order (Hessian-direction) storage hypothesis DEAD (TRIED_IDEAS P32); pairing-resonance hypothesis leans false (two-PP overlap 0.991).
- Workflow angle (local_c7d4e44d, PLAN_workflow.md): rule-conflict review, reports to the PI. Records inventory: docs/consolidation_2026-11-12.md (nothing moved; PI reading).
- Channels: all angles by direct session message (PI 2026-11-17); the MESSAGE_BOARD_* files are archives. /scratch 888 GiB.

- JOINT LOOP (PI 11-16, the project's main test): directly at 8x8, then 10x10; 6x6 dropped (storage is size-dependent: fresh PP 0.49 at 6x6 vs ~0.1 at 8x8).
  Gates: R1'' confirms the remedy (held-out bond f >= 0.12, E(R1'') < E(R1') beyond 2 sigma) AND 8x8 sign round 4 not worse than round 3 (gate re-read 11-16 16:45: grade4 g_4 = -0.7(1.9)e-5, quotable 4 chains chi2/dof 1.88,
  runs/j1j2/sign_bad/r4/grade4/ANALYSIS.md, lead re-ran quotable: same; g_4 > -2 sigma holds; a sign loop at ~0 gain under a fixed amplitude is
  the joint loop's premise, not a block; resolving 1e-5 would cost ~25 A40-h, not spent). Grading design input: paired walkers decouple after
  ~170 blocks (rho 0.88/0.54/0.80/0.60), so joint-loop grades use many short chains restarted in the coupled regime. Round = Lanczos step from
  sign-net x amplitude-net guide, FN, store sign (CNN) + amplitude (PP refit, trust region, fresh PP); 3 rounds; kill gain_k < 0.5 gain_{k-1}.
  Cost ~10-15 H100-h/round (rough, to be sized from R1' + sign-round rates). 10x10 only after 2 steady 8x8 rounds (needs a 10x10 amplitude base).
- T-BAD10 CAPPED at round 1 (PI approved 11-16, lead's recommendation): no 10x10 sign round 2 until the 8x8 sign slowdown (x5.4, x6.2, x2.1) is explained; sign angle told.
  Design + prediction to the PI before any run.

## Budget
PI 2026-10-31: takeover H100 ceiling (4 H100-h) lifted, "use it"; budget open as below.
PI 2026-10-29: GPU budget OPEN until further notice, "wisely spent, no grind": every run follows a written mechanism with a
predicted number (CLAUDE.md points 1, 8); sized from measured rates; reported per test.
PI 2026-10-27 ('go for Paderborn'): test C (complex vs real PP, Pfaffian angle) <= 10 H100-h, single job <= 6 H100-h; rate probe first.
Paderborn re-opened by the PI for the LP test (2026-10-26, 'use wisely'): lead's ceiling 15 H100-h, single job <= 6 H100-h, cont.

## Angles
- architecture angle (session local_8584f371, PLAN_architecture.md): CLOSED 2026-11-09: no candidate passes C1+C2 (gated hybrid fails C1, local add-ons fail H-loc, tied PP sum no cost gain); more PPs the only cheap route; TRIED_IDEAS AR1-AR4.
- tooling angle (session local_e7c8b147, PLAN_tooling.md): CLOSED 2026-11-03, reuse no library; levers are our two-swap kernel and image count.
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
- LOG LOSS NEAR NODES (PI 2026-11-17, firm for every new amplitude fit): no new fit with a log loss on ln|B| unless it is shown log-free near nodes. Cause: d ln|B| = dB/B
  blows up at the network's own nodes (PLAN.md 104-112, Oct); same term killed D5 and G18 (TRIED_IDEAS). R1' at 8x8 still used it (lead approved; drift):
  5 near-node rows at 26-73x the decade -9 tangent norm (PLAN_pfaffian 2577). CORRECTION 11-17 (Astra cj-5ca32ee74fb4, /scratch/a/A.Otaifi/astra/lfix_review/ASTRA_lfix.md): L_fix is NOT log-free: l_xy = 2h t_x t_y (cosh delta - 1), gradient carries O_x = d ln a_x and scales 1/eps^2 at a spurious node (log loss: ln(eps)/eps). Tested only at 4x4 (tax 6.5x lower at 0 hops, 0.77x after 2). Frozen base x bounded factor (B10 worked exact 4x4, B11 8x8 never finished) ranked first.
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
- Pretrained learned correction map (small exact clusters -> 10x10 ViT, replacing f_J; PI + Claude + Astra cj-cc17b9b84ecf,
  2026-11-03, no run): the multi-round target does not exist at an approximate sign (H_FN[a]a = H^s a, so a fixed point is an
  eigenstate of H); the one-round residual beyond c* f_J is ~8% of a round and f_J is already worth ~1.7 rounds, so the ceiling is
  below the gap that matters (4.37e-6 vs floor 0.83e-6 is storage). Closed by the PI: not worth it.
- Relaxing the FN constraint (released node, FN-SR / gamma-FN): released node costs Delta_B 0.385/site (DESIGN_scalable_loop:504);
  gamma-FN 26% of the gap (above); Sorella's group moved on to VMC + Lanczos. Not novel; closed by the PI 2026-11-03.

- Write-up angle (PI 2026-11-13): `PLAN_writeup.md`, session local_71c014c5; progress deck on both lines first, then the paper; numbers signed off by direct session message (board archived 11-17).

## Cockpit
- STATUS (in progress, high): The network copies were never trained to convergence (fixed step size leaves the final weights noisy), and the copy error tracks that training loss; per-round share kept at round 6 is 25%.
- NEXT (in progress, high): Refit round 6 with a decaying step size, 20k and 80k steps (CPU, 1.5 h and 6 h): pass if the share kept rises above 60%.
