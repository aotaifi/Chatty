# PLAN_pfaffian (angle lead "pfaffian angle", started 2026-10-27 by the ansatz angle; PI asked for it)

STEP BACK 2026-10-28 (PI asked; Astra xhigh second opinion running, codex job cj-d28d6046530c):
1. Verdict: continue, but on the right question: expressibility vs learnability is decided by the lead's full-space
   Test G (floor) against the current copies; my J/R/M-J tests only describe the error.
2. Thought that changed it (own guess, unpublished): the deep offset may be one scalar, a miscalibrated power,
   |A| ~ phi^gamma with gamma slightly > 1 (copy "sharper" than phi), from an objective that fixes e's level per decade
   while the energy sees only differences across bonds (a Dirichlet form of e). Then e_x - e_y = (gamma-1) ln r* on
   blocked bonds, and a one-parameter remap |A|^(1/gamma) would remove it. Literature: from memory "Sobolev training"
   (fit derivatives/differences directly), arXiv id not verified (one search, nothing found).
3. Cheapest next test: free, from test R's arrays: c-weighted regression of e on ln phi (slope = gamma - 1) and R after
   removing the linear trend. Prediction: |slope| 0.01-0.05 and detrended R <= 0.3; slope ~ 0 (|slope| < 0.005) kills it.

## Question
Board numbering changed 2026-10-28 (version 32): old 23->4, 24->6, 8/18/19->7, 20->11, 21->12; block numbers below
before this line are the OLD ones (full old board: version 29).
Can a Pfaffian-type network copy the fixed-node (FN) state accurately in the rare configurations,
where today's copies fail? Words: configuration x, neighbour y (one swap), amplitude psi(x),
true ratio r* = phi(x)/phi(y), the copy's ratio r.

## Facts this rests on (verified 2026-10-27; STORAGE_PROBLEM.md, TRIED_IDEAS.md)
- Best-case copy: O2b round 1, Chatty/runs/compress6/runs/o2b_17431886, PP + 3x3 linear backflow, 1944 parameters
  (f 1260 used + u1,u2 648), symmetrised over 576 images, no RBM (code/cx_t1s.py l.38, l.154-176; board block 18).
  Log loss on the exact FN state; stopped at 80 steps, not converged.
- Its error: 75% of the cost from pairs where r < r*/3 (copy's small side too small), r* 0.01-1, phi^2 1e-9..1e-12.
- Failures to quote: B31 (bigger backflow 5x5 does not cut the tax: 5.60e-6 vs 4.58e-6 vs 5.93e-6), G4/C11 (PP+bf by
  plain VMC: no gain in 400 SR steps), G3 (fixed PP killed in T0; backflow route inconclusive, no tail-weighted
  training), B29 (tail-weighted fits: none below 1e-5), G11 (projected BCS + visons killed at 4x6),
  D13 (residual over PP is not local at J2 = 0.5). Pass: G14/G1 7.06e-6 with oracle sign (one seed).
- Published: RBM+PP (Nomura-Imada arXiv:2005.14142); neural backflow (Luo-Clark 2019).

## What we understand (one line each; evidence)
- At 8x8 the STORED SIGN dominates the augmented guide's walker cost: XL CNN 0.261 of 0.378 s/walker-step (69%; A40, fp32 CNN vs f64 lemma; 28.8M sign calls at x, y and two-swap endpoints). On-the-fly s_G is 223x dearer. At scale the sign, not the amplitude ratios, is the expensive part unless the CNN shrinks (pp2_lemma_bench/a40_17693959/bench.json).
- Round stop = fixed step budget + Polyak averaging of the tail iterates: at a fixed target E_FN walks 0.1-0.2e-6 per 150 steps (noise in copy-loss-flat directions); the 6-checkpoint mean beat the best iterate by 0.35e-6 (1.252e-5 vs 1.287e-5; pfaffian_r3avg/readout/readout.json, P19).
- The PP's linear tangent space at B_1 holds 0-4% of the round residual (Θ v2, calibrated: C_exact 0.082 vs held-out 0.069/0.065; P16); the nonlinear refit still stores r = 0.11-0.18 of each round's Jacobi gain under s_G (P15-P19). Whether r declines per round is NOT established (q 0.63-0.95 single-iterate; averaged r_3 0.178 >= r_2 0.145).

## Scope
Paper discussion with the PI first; no run until he agrees. Candidates: PP + linear backflow (what we used),
neural backflow, sum of Pfaffians e^J sum_k Pf[M_k], plus RBM/Jastrow factor.
Waits on: ansatz angle's 4x4 ceiling test (R_ceil) and guide-vs-FN-vs-copy split (results into STORAGE_PROBLEM.md).
Budget: none spent; to set with the PI when a test is proposed.

## Hypothesis on the table (2026-10-27, not yet agreed with the PI; board block 19; ~35%, raised to ~65% same day)
O2b copy = s2(x) |A(x)|, A = signed sum of 576 dets, trained on log|A| only, so A's own sign is uncontrolled.
Where sgn[A(x)A(y)] != s2(x)s2(y), A's own node sits at x and |A(x)| is near 0, so r << r*.
Test: q = share of the r < r*/3 cost on such pairs (forward pass of O2b round_1 theta.npy on a few thousand
cost-sampled pairs from /scratch/a/A.Otaifi/ansatz_runs/bond_split; minutes of CPU).
Supported if q >= 0.5; refuted if q <= 0.1; in between, inconclusive.
If supported: wider or deeper backflow does not help while the network has a node of its own. Training the sign is
OUT (PI 2026-10-27: that is the hard NQS problem the project avoids; sign from Krylov, size from FN). Remedies that keep
the division of labour (board block 21): (a) size from a positive net s2 e^J; (b) same, with the node bonds of s2 at x
as input (they set the FN dip, block 3). TRIED_IDEAS/ledger have no row for (b). If refuted: the size response itself is wrong, and the next question
is whether F should respond nonlinearly to z (neural backflow).
Support (ansatz angle's guide split, verified from guide_split_round_{1,3}.json, class r < r*/3): the copy's small side is
far below phi, e_x = ln|copy| - ln|phi| cost-weighted mean -6.2 / -8.9, median -6.0 / -10.6 (rounds 1 / 3); large side right
(e_y +0.02). Spurious deep near-zeros where phi has none, as an own node predicts. Not yet the direct check (sgn A vs s2).
Derived (cx_t1s.py, log loss on ln|A|): the loss is infinite where A(x) = 0, so small-step training never flips sgn A at a
sampled x. CORRECTED by the q test: relative to the plain PP, O2b flipped sgn A at 47% of the costly small sides, so the
node is NOT frozen at the plain PP's sign (those rare x are mostly not in the training sample, where the barrier applies).
Evidence against "window too small" at 6x6: F1 (delta_today 2026-10-23): converged 3x3 5.60e-6, converged 5x5 4.58e-6
(25 of 36 sites seen), O2b 5.93e-6.

## Test q (pre-registered 2026-10-27 before any number; lead decided to run it, PI delegated)
Per orbit rep: t(x) = sgn A(x) * s2(x), A = signed projected sum of O2b round_1 (theta.npy), s2 = refresh_17409419/s2.npy.
t is orbit-invariant if A and s2 sit in the same symmetry sector (checked: t constant over images of 20 random x).
On violating pairs (s2 x s2 = +1), the network's own node crosses bond {x,y} iff t(x) t(y) = -1.
- Primary: q = cost share of class r < r*/3 with t(x)t(y) = -1 (cost as bond_split.py cost_pass; pairs sampled
  proportional to cost, n = 5000 with replacement; error from a binomial over samples).
- Control (predicted difference): q_ref = same share in class r*/3 < r < 3r*. The hypothesis predicts q >> q_ref.
- Secondary: q_any = cost share where the small side has ANY neighbour z with t(z) != t(xs); share with t(xs) = -1;
  is sgn A on those configurations equal to the warm-start PP's own sign (rpT1_pp f, u = 0)?
- Code-ran check: ln|A| from theta minus la_table is one constant (spread < 1e-6) on the sample; e_x medians reproduce
  guide_split_round_1.json (-6.0 cost-weighted).
- Verdict: q >= 0.5 and q >= 3 q_ref supports the own-node cause; q <= 0.1 refutes it; otherwise inconclusive.
Next per verdict (lead, 2026-10-27, PLAN.md ffd8e9d): supports -> pre-register complex vs real PP parameters
(same O2b PP, |A| with sign s2, equal parameters and steps; Im f needs a small random start because d ln|A|/d Im f = 0
at a real warm start; kill: complex tax_0 > 0.5 x real tax_0), size on ws1/A40 and submit. Refutes -> drop the own-node
and complex routes, report to the lead.

## Result q (2026-10-27, ws1 CPU, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_q/q_ownnode.json; script ansatz/code/q_ownnode.py)
INCONCLUSIVE by the rule: q = 0.424 +- 0.007, q_ref = 0.069 +- 0.004 (ratio 6.1; q < 0.5).
Checks passed: t orbit-invariant (20 reps); ln|A| - la_table spread 8.9e-9 (broken arm 19.7); total cost 3.528e-4/site = bond_split.
Reading: A's own node crosses the bond on 42% of the costly-class cost (= ~32% of all pair cost), 6x the accurate class.
A real contributor, not the majority. Large side has t = -1 on 99.9% (global sign convention), so q = share with sgn A(xs)
flipped relative to the bulk. q_any (any neighbour across A's node) does not discriminate: 0.996 vs 0.909.
Not done: e_x split by t(xs) (t per sample was not saved); one sample seed.
Consequence for the complex-PP test: removing own-node zeros can at most remove ~1/3 of the frozen-phi pair cost, so the
lead's kill (complex tax_0 > 0.5 x real) would fire even if the mechanism is real; needs re-sizing before it is run.

## Test C: complex vs real PP (pre-registered 2026-10-27 before any number; lead's go and amended rule, PLAN.md)
Question: does removing the network's own zeros (complex parameters: |A| = 0 needs Re A = Im A = 0, generically never on
a lattice) cut the copy's error, specifically on the bonds the real network's own node crosses?
- Model: O2b's PP + 3x3 linear backflow, target phi_1 = O2b round_1 target_la, sign s2 for the copy, log loss, damped
  GN/minSR, same fit code and spec as F1 A (cx_f1fit.py, Chatty/runs/compress6/runs/f1_A/round_1/spec_fit.json).
  Complex arm: f, u1, u2 complex (Re and Im as separate real parameters, 3888), A = signed sum over 576 images.
- Start for both arms: F1 A's converged real theta (f1_A/round_1/theta.npy; tax 5.60e-6, E_FN zero hop 2.79e-5).
  Real arm: continue training (Im fixed at 0) to the same flatness criterion. Complex arm: Im = sigma * N(0,1) with
  sigma = 0.01 x rms(Re theta) (at Im = 0, d ln|A| / d Im = 0, so a random start is required). Seeds 1, 2, 3 per arm
  (sample seed; also the Im draw in the complex arm).
- Stop: F1's criterion (held-out change < flat_tol 0.02 over the 50-step window, min 50 steps), same for both arms.
- Readout per run: tax_0 = E_FN(copy, zero hop) - E_FN(phi_1) (9.85e-6), and the one-hop tax as F1 reports it.
  Mechanism: S_x = pairs of class r < r*/3 (from F1 A's real copy) with t(x)t(y) = -1 (t = sgn A x s2, from F1 A),
  S_rest = all other violating pairs; pair cost (bond_split.py cost_pass) of each arm on S_x and on S_rest, fixed sets.
  cut_X = 1 - C_cplx(S_x)/C_real(S_x), cut_R = 1 - C_cplx(S_rest)/C_real(S_rest) (seed means).
- Pass: complex tax_0 <= 0.75 x real tax_0 AND cut_X >= 2 cut_R.
  Kill: complex tax_0 >= 0.95 x real tax_0, OR cut_X <= cut_R. Otherwise inconclusive, cause stated.
  Seed scatter: std over seeds of tax_0(cplx)/tax_0(real) must be <= 0.05, else inconclusive.
- Code check before production: with Im = 0 exactly, the complex code reproduces the real code's ln|A| to 1e-10 on
  1000 reps; a broken arm (Im start with sigma = 1) must change it.

Budget (PI via the lead, 2026-10-27): <= 10 H100-h on Paderborn gpu_h100, account hpc-prf-ppdmrg, single job <= 6 H100-h.
Order: CPU checks pass -> one-step rate probe -> resize if complex step > 6x real -> lead reads the launcher -> submit.

Amendment before any production number (2026-10-28): F1 A's la_table is not on local disk, so S_x and S_rest come from
the real arm's seed-1 final copy (la_table + t_real), which starts from F1 A and only re-confirms flatness.
CPU checks passed (ws1, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/checks.json; code ansatz/code/cx_c1*.py, c1_mech.py):
Im = 0 reproduces real ln|A| to 5.2e-13; broken arm (sigma 1) shifts it by 39; Im Jacobian columns exactly 0 at Im = 0;
Jacobian vs finite differences 2.2e-7 relative; dry fits exit 0 (peak 13.7 / 19.2 GB). CPU step cost complex/real 1.37
(B = 512). Readout, not a criterion: sigma 0.01 already lifts |A| at the costly small sides by median e^2.29 at step 0
(held-out loss 8.7e-2 vs 1.2e-2 real), so the complex arm starts higher and must recover; sigma kept as registered.

## Running
Rate probe job 4040973 done (H100, B 8208, ~0.02 H100-h): real 3.67 s/step, complex 6.68 s/step (ratio 1.82); held-out
loss at step 0 real 0.034, complex 0.089. Production launcher ansatz/code/submit_c1_all.sh (-> launch_c1.sh, c1.sbatch,
c1_pdb_watch.sh) read by the angle lead; sizes proposed real 00:45 (fit cap 1200 s), complex 02:25 (fit cap 7200 s,
max_steps 1100), worst case 9.5 H100-h; approved by the project lead 2026-10-28.
## Result C (2026-10-28; verified by the angle lead from each cx_c1.json and c1_mech_out.json)
Jobs real 4041269/4041272/4041274, complex 4041271/4041273/4041275; 5.31 H100-h incl. probe (budget 10).
| arm | steps | converged | tax_0 | one-hop tax |
| real s1/s2/s3 | 324 each | NO (1200 s cap; held-out change 0.21/0.42/0.035) | 1.90/1.84/1.74e-5 | 6.03/5.69/5.23e-6 |
| complex s1/s2/s3 | 280/720/710 | yes | 1.107/0.912/0.786e-5 | 3.75/3.23/2.85e-6 |
| F1 A (start, converged, 1070 steps) | | yes | 1.80e-5 | 5.60e-6 |
tax_0 ratio complex/real 0.51 (per seed 0.61/0.50/0.43, std 0.088 > 0.05). Every complex seed is below every real seed and F1 A.
Mechanism (sets from real s1): cut_X 0.973, cut_R 0.832; remaining cost complex/real 0.027 on S_x vs 0.168 on S_rest.
VERDICT: INCONCLUSIVE by the registered rule (real arm unconverged; seed scatter 0.088 > 0.05; cut_X < 2 cut_R).
Causes: (1) my sizing: restarting from F1 A re-anneals tau from 0.05, so the real arm drifts instead of settling; (2) the
cut rule saturates: with cut_R = 0.83, cut_X >= 2 cut_R is unreachable (cut <= 1); on remaining-cost ratios S_x is 6x more
specific (post hoc, not a registered criterion); (3) S_x from real s1 only, while each real seed's own node moves
(real C(S_x) 1.15e-2/4.2e-3/2.4e-3), so other seeds' node pairs sit in S_rest.
Derived, unvalidated: H1 measured tax_2/tax_1 = 0.32 for real PP copies; if it holds for complex, tax_2 ~ 1.0e-6 vs the
loop gate 0.82e-6 and round gain 1.63e-6 (C8). Not measured.
Lead 2026-10-28: the cut rule is a flaw of the rule (unreachable once cut_R > 0.5), recorded as such, not as a fail. Real-arm
rerun + per-seed mechanism sets funded only if test T2 passes.

## Test T2: tax_2 of the complex copies (pre-registered 2026-10-28 before any number; lead's go and rule)
tax_2 = E_FN(2 hops of copy) - E_FN(2 hops of phi_1), as cx_h1.py computes it (summary tax_2; code sha b555f9ba, codeH1 on
Paderborn, unchanged). Three cx_h1 runs in one H100 job, each with o2b_table = O2b a_2, f1_table_A = F1 A, f1_table_B =
complex copy s_k la_table (runs c1_cplx_s{1,2,3}/round_1). Controls per run: phi1, o2b_a2 and F1A reproduce h1_full
(F1A tax_2 1.846e-6, O2b 1.903e-6) to rel < 1e-6; otherwise stop.
Reference: F1 A converged real copy tax_2 = 1.846e-6 (h1_full/cx_h1.json). Gate numbers: 6x6 G-tax round gain 1.629e-6,
gate 0.82e-6.
- PASS: mean complex tax_2 <= 0.82e-6 and every seed <= 1.63e-6.
- PARTIAL: mean in (0.82e-6, 1.63e-6): net gain per round, below the gate.
- KILL: mean >= 1.63e-6.
Budget: ~0.3 H100-h (H1 took 222 s for four cases). RUNNING: job 4042525 (Paderborn H100, 40 min wall), launcher launch_t2.sh read by the angle lead, watcher t2_pdb_watch.sh; local /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/t2_cplx.

## Result T2 (2026-10-28, job 4042525, 0.19 H100-h; verified by the angle lead from s{1,2,3}/cx_h1.json)
| copy | tax_1 | tax_2 | tax_2/tax_1 |
| complex s1 / s2 / s3 | 3.754 / 3.231 / 2.845e-6 | 1.244 / 1.070 / 0.925e-6 | 0.331 / 0.331 / 0.325 |
| F1 A real (control, reproduced) | 5.596e-6 | 1.846e-6 | 0.330 |
| O2b real (control, reproduced) | 5.930e-6 | 1.903e-6 | 0.321 |
Mean complex tax_2 1.080e-6 (0.58 x F1 A), max 1.244e-6. VERDICT: PARTIAL (net gain per round, below the gate 0.82e-6).
Derived: net gain per loop round = 1.629e-6 - tax_2 ~ +0.55e-6 (frozen round gain from C8's RBM+PP loop; not measured for this copy).
Controls: phi1 1.99e-11, o2b_a2 <= 9.1e-12, tax_1 = cx_c1.json to 1e-10. F1A tax_2 reproduces to 6.3e-6 relative (1.2e-11
absolute), over my registered 1e-6: the tolerance was set too tight for a difference of two E_FN values, not a physics miss.
H100-h used in total: 5.50 of 10. Key map: t2_cplx/README.md (slot "F1B" = complex copy, not the real 5x5).

## Test D: where the complex copies' remaining tax sits (pre-registered 2026-10-28 before any number; lead's rule)
Rerun the O2b bond split (ansatz/code/bond_split.py) and the guide split (bond_guide.py) on the three complex copies'
la_tables (pfaffian_c/runs/c1_cplx_s{1,2,3}/round_1/la_table.npy), same target phi_1 and sign s2, CPU only.
- CAPACITY (5x5 is the lever): class r < r*/3 below 40% of the cost, error spread over moderate ratios with |e_x| < 1
  (cost-weighted median).
- RESIDUAL DIPS (near-zeros of |A| where Re A and Im A are both small): class r < r*/3 still >= 60% of the cost, with
  cost-weighted median e_x <= -3.
- Otherwise inconclusive; state which class dominates. Judged on the seed mean, with all three seeds reported.
Next: CAPACITY -> complex 5x5, 2 seeds, if it fits the remaining 4.5 H100-h (else report). RESIDUAL DIPS -> report.
Loop C with complex copies: the PI closed the loop 2026-10-26; the lead raises it with the PI once a copy reaches the gate.

## Result D (2026-10-28, ws1 CPU 2.6 h, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/; shares and e_x verified by the angle lead)
Checks: O2b split reproduced (3.5284e-4/site, <1/3 share 0.752); zero-hop E_FN differences equal tax_0 to 0.03%.
| copy | pair cost/site | <1/3 | 1/3-3 | 3-30 | >30 | <1/3 median e_x |
| O2b real | 3.53e-4 | 75.2% | 21.3% | 1.8% | 1.7% | -6.0 |
| complex s1/s2/s3 | 6.70/6.26/6.14e-5 | 17.8/19.6/23.1% | 68.4/66.1/63.4% | ~12% | ~1.6% | -1.91/-1.79/-1.91 |
VERDICT: INCONCLUSIVE by the rule; dominant class 1/3-3 (66%, copy within a factor 3 of the true ratio). CAPACITY clause 1
met (<1/3 at 20% < 40%); clause 2 as applied (median e_x of the <1/3 class, -1.87) fails |e_x| < 1. The clause did not name
the class: my error in writing it. Far from RESIDUAL DIPS. Reading: complex parameters removed the deep dips (<1/3 median
e_x -6.0 -> -1.9, pair cost 5.6x lower); what remains is mostly moderate error spread over ratios 0.01-1.
Retracted: my "capacity-like" reading names no term (projects/j1j2/CLAUDE.md point 2). Lead's prediction M1 (share 0.52) refuted
by 0.20; his failed term: frozen-phi bound shares read as true-tax shares (bound/true 17.7 for O2b vs ~6.9 complex).
Lead 2026-10-28: no 5x5 run (the only number, real 3x3 -> 5x5 = -18%, predicts ~0.89e-6 vs gate 0.82e-6, inside seed
scatter); keep the 4.5 H100-h.

## Open question (2026-10-28, discussion with the PI, no run): which function class does our problem demand for the copy?
Board 23: E_FN depends on the copy only through V_A(x) = sum_blocked |H_xy| |A(y)/A(x)|. Board 24: the target is strictly
positive in the s2 gauge (Perron-Frobenius on the stoquastic, connected s2 H_FN s2; measured: ln phi finite on all 15.8M
orbits, min -34.5), so the class must be zero-free; a real pair product is not (zeros on a 1-equation set), a complex one
generically is (2 equations). Needed next, with the PI: the physics of the remaining moderate error (where s2's blocked
bonds sit; which configurations), and which zero-free class gets r within a factor 3 at r* 0.01-1, with a mechanism.
Tried and failed, do not re-propose without a new term: learned bond field (E9, C16: plateau 1.92e-5, non-local).

## Dead ends
none yet

## Cockpit
- STATUS (in progress, high): Branch 2 at 8x8: round-1 refit done; its first energy grade (no augmentation, same sign as the production network) is running tonight.
- NEXT (in progress, high): Read that grade against the production network on the same code; if the refit is lower, run the one-Lanczos-step headline (about 8 GPU-hours, needs the PI).
- NEXT (open, medium): Find why the refit leaves the heaviest deep bonds unfitted (sampling or network class) before a second round.

## Role from 2026-10-28 (lead, at the PI's request): network expert. Open question: can complex PP + 3x3 backflow hold
the deep (near-node) corrections, and if not, which network can? Inputs (lead, verified): Test E (67.6% of configurations,
phi^2 < 1e-13, never trained; 21-26% of the fit loss is the e_x^2 term), Test F (exact splice below 1e-11 removes 35% of
the complex copies' tax_0, below 1e-13 11%, random bulk 0%), Test G (refit on true E_FN) running in the lead's session.

## Mechanism M-J (2026-10-28, board 14; not yet run, waits for the PI)
FN eigen-equation at x (phi = ground state of H_FN[g], E its energy, s2 frame): phi(x) = sum_kept |H_xz| phi(z) / (H_xx + V_g(x) - E).
Exact split of the copy error: e_x = (I) ln sum_kept w_xz e^{e_z} [inherited from neighbours] + (II) rho_A(x) [local
violation of the eigen-equation by A at x]. One Jacobi step A' = sum_kept |H||A(z)| / (H_xx + V_g - E) removes (II) exactly
and is positive (no node of its own); the old Lanczos hop removed part of it (G-A2: 2/3 of the tail tax).
Prediction (from G-A2, before any number): s_II = sum c_x rho_A^2 / sum c_x e_x^2 over phi^2 < 1e-11 is 0.5-0.8 for the
three complex copies; s_II < 0.3 refutes M-J (then the deep error is inherited from neighbours, a bulk problem).
Check: identity e_x = (I) + (II) to 1e-10 on all orbits (code check); a broken arm (V_g -> 0) must break it.
Cost: CPU, existing tables (target_la, s2, guide g_1 table, nb_J1/nb_J2, complex la_tables), est. 1-3 h on ws1.
If supported: the network fix is a local eigen-equation structure (Jacobi layer: O(N) neighbour sum per amplitude,
the same cost class the PI closed C14 on; or a cheap surrogate). If refuted: deep error follows bulk error.Guide for M-J (lead 2026-10-28): g_1 = hop(a_1) is not on disk; rebuild with o2b_17431886/code/cx_o2b.py hop() l.74-100,
a_1 = round_0/la_table.npy, alpha_V = -0.11787844532393324 (cx_o2b.json rounds[1].hop_a); control E_FN(g_1) =
1.411883720998159e-05/site to 1e-9 rel, then FN(g_1) = round_1/target_la.npy. The identity uses V = V_{g_1}, the wall of the
HOP guide (not a plain s2|A| guide as in the agreed pipeline); the split itself is exact either way, but the s_II number
describes copies of a hop-guide target.
PI 2026-10-28: go for the M-J CPU split; GPU hours only after clearing them with the PI (CPU is free).

## Result M-J (2026-10-28, ws1 CPU 29 min; /scratch/a/A.Otaifi/ansatz_runs/pfaffian_mj/mj_split.json; verified by the angle lead)
Controls: E_FN(g_1) to 2.8e-11 rel; FN(g_1) = target to 1.6e-10 rms; broken arm (V_g = 0) rho_phi up to 16.3.
rho_phi and the identity hold to 1e-11 for phi^2 >= 1e-11 but only to 7e-7 deep (LOBPCG residual is absolute): misses my
1e-10 tolerance, 1e5 below the signal (|rho| ~ 0.1); tolerance was mis-set, not a physics miss.
| band phi^2 | share of orbits | s_II (s1/s2/s3) | mean e | mean (I) | mean rho |
| < 1e-11 | 92% | 0.985/0.987/0.989 | -0.133/-0.123/-0.279 | ~1e-3 | = mean e |
| 1e-11..1e-9 | 7% | 0.974/0.981/0.983 | -0.10 (s1) | ~1e-3 | ~ mean e |
| >= 1e-9 | 0.7% | 0.47/0.44/0.46 | ~0 | ~0 | ~0 |
VERDICT: outside the predicted band on the high side (0.99 vs 0.5-0.8); M-J not refuted. Flaw in my metric: e and (I)
shift with the copy's normalisation (A -> cA), rho does not; with the bulk-anchored normalisation (sum n A^2 = 1),
(I) ~ 0. Reading: the deep error is LOCAL (rho), one-signed and smooth in depth: the copy is too small by e^-0.13..e^-0.28
below phi^2 1e-11, ~0 above 1e-9; its neighbours are right. That pattern fits the training (no samples below 1e-13,
Test E) more than a capacity limit; not yet separated.

## Tests J and R (pre-registered 2026-10-28 before any number; lead's go, CPU only, no GPU until the PI is back)
Copies: complex s1..s3 (pfaffian_c/runs/c1_cplx_s{k}/round_1/la_table.npy), normalised sum n A^2 = 1.
J (Jacobi step, oracle-free): A'(x) = sum_kept |H_xz| |A(z)| / (H_xx + V_A(x) - E_FN[A]), V_A and E_FN[A] from the copy
only (no phi); kept/blocked by s2. Denominator is > 0 by construction: E_FN[A] = lambda_min(H_FN[A]) <= every diagonal
entry (Rayleigh quotient on a basis vector); count of <= 0 must be 0 (report it).
- Arm (a): A' only where the copy's own |A|^2 < 1e-11, A elsewhere. Arm (b): A' everywhere.
- Broken control: arm (a) with the denominator replaced by its c-weighted mean over the band (local diagonal removed).
- Readout: tax_0 = E_FN - E0 per site - 9.85e-6, cut = 1 - tax_0(A')/tax_0(A), per seed.
- Prediction: arm (a) cut 25-40% (the exact splice below 1e-11 removed 35%, Test F; Jacobi removes rho, and (I) ~ 0 there);
  arm (b) cut >= arm (a); broken control cut < arm (a) / 2. Kill: arm (a) cut < 10%.
- Cost at scale [derived, to be checked against code]: one A'(x) needs |A| at all neighbours of x (<= 2N bonds, 144 at
  6x6): one local-energy-type sum. In FN every walker step needs A' at all neighbours y of x, so neighbours of neighbours:
  the same class as evaluating the Krylov sign s'(y) = s(y) sgn(T - E_L(y)) on the fly (D6 cached global Krylov step).
R (offset test): m_d = c-weighted mean of rho_A in phi^2 decade d (diagnostic, uses phi); R = sum_deep c (e - m_d)^2 /
sum_deep c e^2 over phi^2 < 1e-11. Prediction R <= 0.3 (deep error = smooth depth offset). Broken control: m_d with the
decade labels randomly permuted, R_broken must be >= 0.8. If R <= 0.3: the network needs its depth as input, e.g. a
monotone remap of its own output, A'' = A exp(h(ln|A|)).
R addendum (pre-registered 2026-10-28 before R's numbers; PI asked: expressibility or learnability?): from m_d per decade
and the fit's train / held-out orbit masks (cx_lib Lab.HO; trained decades -5..-13):
- coverage (learnability) predicts |m_d| for d <= -14 (never sampled) >= 2 x |m_d| for -13 <= d <= -9 (sampled);
  a ratio < 1.3 (flat across the training edge) says the offset is not caused by missing samples.
- within sampled decades, mean rho on held-out orbits vs trained orbits: equal within 0.02 = no overfitting (the offset is
  systematic, from the objective or the class); held-out more negative by > 0.05 = memorisation.
The separating test (class's best fit on the true E_FN over all configurations) is the lead's Test G, if it covers the
full space; two seeds agreeing needed before calling its residual the expressibility floor.
Lead 2026-10-28: Test G covers the full space (exact sums or x ~ n chi*^2 over all 15.8M orbits); its residual counts as the expressibility floor only if E_FN is flat to < 1e-8/site over 3 re-solves and two seeds from different warm starts agree.

## Astra second opinion (2026-10-28, ansatz/REVIEW_astra_pfaffian_2026-10-28.md) and what I take from it
Conceded (my errors): (1) the real-vs-complex zero-set argument (board 6) is a heuristic, not a bound: on a finite set a
sign change between neighbours does not force a small modulus; q shows association, not cause. (2) "one-signed deep offset"
is wrong as stated: the cost-weighted mean of e is negative, but by orbit count only 43-44% of deep orbits are negative.
(3) s_II is not an orthogonal variance share (cross term) and depends on normalisation. (4) R's held-out readout is
contaminated: bond partners y in training are not restricted to training orbits (cx_c1fit.py bonds_of, checked).
(5) C_phi (frozen-phi bound) is not the tax and not its decomposition (R_s(phi) - E_FN(phi) = 1.63e-6/site offset).
Agreed: no test so far separates expressibility from learnability; Test G success would show reducibility, its failure no floor.
Astra's separator (GPU, needs the PI): (a) constructive witness: same class, fit C_phi with importance-corrected sampling
covering every blocked edge; tax_0 < 2e-6/site and C_phi down 10x refutes a floor near today's value; failure inconclusive.
(b) teacher-student control: target generated by known parameters of this same class (expressible by construction),
same training; failure to recover it = learnability problem. Est. 1.5-2 H100-h + 1 CPU-h; Astra forecasts 20-40% tax cut.
Expressibility failure can only be certified by a lower bound over the class; no test provides one.
Class suggestion: positive guide-relative Jastrow A = g exp(R_theta) (factorised residual, B10/B11 exist), O(NM) per
amplitude; a Jacobi layer is O(N^2) per walker step. Refs (Astra [Checked], not read by me): Gispen-Lamacraft
arXiv:2012.07063 (same log eigen-equation), Carleo-Troyer arXiv:1606.02318.
Concessions accepted by the lead 2026-10-28.

## Separator tests (pre-registered 2026-10-28 before any number; GPU waits for the PI, lead puts it to him at 19:30)
(a) TEACHER-STUDENT (learnability). Teacher: the complex copy s3's own table A_T (pfaffian_c/runs/c1_cplx_s3/round_1/
la_table.npy), exactly representable by this class (theta_T exists), same dynamic range as phi (it copies phi).
Student: identical to test C complex seed 1 (cx_c1fit, band sampler, log loss, F1 A real theta + Im 0.01, flatness rule,
max_steps 1100), target A_T instead of phi; sign s2 for both.
Readout: C_T = frozen-target pair cost of the student against A_T (bond_split cost_pass with A_T as target), compared
with C_phi of test C seed 1 against phi (6.70e-5/site, Test D); ratio rho_a = C_T / C_phi. Secondary: |E_FN[student] -
E_FN[A_T]| per site vs test C seed 1's tax_0 1.107e-5; deep-band e statistics.
- Learnability dominant: rho_a >= 0.5 (training cannot reproduce even an expressible target of this shape).
- Target-specific / expressibility: rho_a <= 0.1 (training reproduces an expressible target; phi's error is phi's).
- Between: mixed.  My prediction: rho_a = 0.4 (0.2-0.7). Kill of the learnability lean: rho_a <= 0.1.
- Caveat written now: student and teacher share the F1 A parent, so the test is generous to training; a failure to
  recover is strong evidence, a success is weak. Second-basin variant (if funded): student from the plain PP
  (rpT1 f, u = 0) + Im 0.01.
- Cost (measured, test C complex: 6.68 s/step on H100, 280-720 steps, 41-91 min wall incl. table and E_FN): <= 1.5
  H100-h for one seed; a second seed only if rho_a lands within 1.5x of 0.1 or 0.5.
(b) CONSTRUCTIVE WITNESS (Astra): mostly covered by the lead's full-space Test G (same class, true E_FN, all edges); a
separate C_phi fit with pair sampling proportional to |H| phi_x phi_y would add ~1-1.5 H100-h. Prediction if run:
tax_0 6-7.5e-6 from s3's 7.86e-6 (20-40% cut, as Astra); refutes a floor near today's value only if tax_0 < 2e-6 and
C_phi down 10x. Recommendation: do not fund (b) separately; read Test G with this criterion.

## Bar (lead, 2026-10-28, pfaffian_g/delta0/delta0.json, exact): one exact amplitude round gains 7.82e-6/site from the
complex copy s1 and only 2.09e-6 from phi_1; near the fixed point a useful copy must lose < ~1e-6 per round, 10x below
today's tax_0 0.94e-5. A 2x improvement of the absolute copy is not enough.

## Proposal M-R: store the round's correction, not the amplitude (2026-10-28, board 15; not run; my confidence ~60%)
|A_{k+1}| = |A_k| e^{R_theta}, target R* = ln(phi_k/|A_k|). Storing nothing (R = 0) loses exactly the round's gain;
in the quadratic regime with an error shaped like R*, tax/gain ~ eps^2, eps = relative error of the stored correction.
The bar needs eps < 0.7 (tax < 1e-6 at gain 2.09e-6); the absolute copy instead needs ~10x lower absolute error.
Records: B22 (residual re-base, old pipeline with one-hop features) kept 85% of the table loop's gain at it 6-8 (eps ~0.39);
it was set aside only because it lost to equal-hop Lanczos, a comparison the agreed pipeline (no Lanczos) no longer makes.
B10 (factorised residual a_k exp(r), exact 4x4) 99.7%. Risks: chain depth grows by one net per round (C16: constant-cost
re-basing failed); at scale R* must come from walkers (board 11).
Test (CPU, free): R* = ln(FN(A_s1)/|A_s1|) from delta0's exact FN state; fit a small symmetric positive residual (RBM-type,
~1-2k parameters) with the same sampled GN; readouts: tax = E_FN[A_s1 e^R] - E_FN[FN(A_s1)], gain 7.82e-6, and
eps_C^2 = C_phi(e)/C_phi(R*).
- Prediction: tax <= 0.3 x gain (<= 2.3e-6), and tax/gain = eps_C^2 within a factor 2.
- Kill: tax >= 0.7 x gain, or the quadratic law off by > 3x.
Decision (mine, PI gave me the final say): this addresses the 10x bar directly and is CPU-only; it goes before the GPU
teacher-student test, which can only explain a 2x-scale error. Teacher-student stays pre-registered, not requested.
M-R additions (lead 2026-10-28, pre-registered before any M-R number):
- Error shape: with the bond form D(a,b) = sum_blocked |H| phi'(x) phi'(y) (a_x - a_y)(b_x - b_y), split e = -eps_par (R* - mean)
  + e_perp with D(e_perp, R*) = 0. Report eps_par^2 and D(e_perp,e_perp)/D(R*,R*). Expected: tax/gain ~ eps_par^2 +
  perp share; if the perp share dominates, a pass need not carry to the next round (reported as such).
- Version tested: (a) keep the chain, one round. Cost [derived]: one symmetric RBM residual (alpha 16) = 576 images x
  36 x 16 ~ 3e5 flops per amplitude vs the PP base 576 dets of 18x18 ~ 3.4e6, so ~10 stacked residuals (~10 rounds at
  21% of the excess per round, lead's number) cost ~1 base evaluation: chain doubles the cost, does not change its class.
  (b) folding (one net for ln(phi/A_0)) is not tested now; its readout would be fit quality of ln(phi/A_0) vs ln phi at equal capacity.

## Result M-R (2026-10-28, ws1 CPU 1.9 h, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_mr/mr_result.json; verified by the angle lead)
Controls: E_FN[A_s1], E_FN[phi'] reproduce delta0 (rel 0, 7.5e-12); gain 7.825e-6; R = 0 -> tax/gain 1.000; R = R* -> 0;
R = -R* -> 2.509. R* is small: rms bond difference 0.0107; D(R*,R*) = 1.76e-4 = C(R=0) (deep in the quadratic regime).
| alpha | params | steps | tax/gain | eps_C^2 | ratio | eps_par | perp_share |
| 4 | 148 | 100 | 0.954 | 0.948 | 1.007 | 0.941 | 0.061 |
| 16 | 592 | 100 | 0.941 | 0.930 | 1.011 | 0.928 | 0.069 |
VERDICT: KILL by the registered rule (tax >= 0.7 x gain); the quadratic law holds (ratio 1.01). The stored R is ~6-7%
of R* (eps_par 0.93-0.94); the fits stopped at the flat rule while still creeping (LM at minimum damping, ~1%/step).
Per projects/j1j2/CLAUDE.md point 2, "optimiser stalled" is not a cause. Candidate term [derived, to test]: at small
weights an RBM residual is sum ln cosh(b + W z) ~ const + quadratic in z, i.e. a symmetric 2-body Jastrow; if R* has
only a small projection on 2-body functions, the RBM is stuck there by expressibility, not by the optimiser.
What R* should be [derived from board 14, first order in R*]: R*(x) ~ f_J(x) = ln[ sum_kept |H_xz||A(z)| /
((H_xx + V_A(x) - E') |A(x)|) ], the one-Jacobi-step factor of the copy itself (E' = E_FN[A]), a local-energy-type
quantity of A, not a 2-body function of the spins. Test J arm (b) (running) is exactly A e^{f_J}.

## Test L (pre-registered 2026-10-28 before any number; CPU): what function class is R*? Convex, no optimiser.
Exact D-norm least squares of R* on nested feature sets (all symmetric, per orbit rep), eps^2 = D(e,e)/D(R*,R*) at the
optimal coefficients (normal equations with D computed over all blocked pairs):
 L1: 2-body z_i z_j summed per symmetry class of displacement (9 classes);  L2: L1 + n_blocked J1, n_blocked J2 (from s2);
 L3: L2 + H_xx;  L4: f_J alone (+ constant);  L5: L3 + f_J.
Predictions: L1 eps^2 >= 0.85 (explains the RBM's 0.93-0.95); L4 eps^2 <= 0.3 (R* is the Jacobi factor);
L5 <= L4. Kill of "R* = Jacobi factor": L4 eps^2 >= 0.7. For the best set, one exact E_FN: tax/gain ~ eps^2 (within 2x).
Intent if L4 passes (stated 2026-10-28 before L's numbers, lead asked): (a) with exactly ONE Jacobi round, fused with the
Krylov sign. [derived] In the s2 frame E_L(y) = H_yy + V_A(y) - K_A(y), K_A(y) = sum_kept |H_yz||A(z)|/|A(y)|; the sign
s2(y) = s(y) sgn(T - E_L(y)) needs E_L at every neighbour y of a walker, and f_J(y) = ln K_A(y) - ln(H_yy + V_A(y) - E')
uses the same two sums. So one Jacobi factor adds no cost class beyond the on-the-fly Krylov sign (D6 caching applies
to both). k > 1 rounds nest (~400^k at 14x14) and are out. One round reaches at most one exact round's gain: from phi_1
21% of the excess (lead, delta0), so (a) alone does not reach the s2 floor; further rounds need (b) folding, i.e. the
copy problem again for ln(phi/A_0). A pass of L is therefore a method for one round and a diagnosis beyond it.
CORRECTION (lead 2026-10-28, verified by him; Astra flagged it): "s2" in every stage-1 table is NOT two Krylov steps; it is
s1 flipped on the FN expelled set X from an exact FN solve (experiments/compress_6x6/cx_refresh.py l.48): an ORACLE sign.
All results of this angle (q, C, T2, D, M-J, M-R, J, L) are for an oracle sign. My cost argument "one Jacobi factor fused
with the Krylov sign" holds for a deployable on-the-fly Krylov sign, not for this s2; its cost class is that of whatever
deployable sign replaces it. Lead's reframe to the PI (PLAN.md ca56a72): deliverable = network + one Lanczos step + FN
(ViT guide 6x6: FN+1 Krylov sign 9.43e-5, FN+1 signed Lanczos 1.89e-5, VMC 1.32e-4). The Jacobi factor f_J and the Lanczos
factor 1 + alpha (E_L - E) are the same family (local-energy corrections of the amplitude, same two-hop cost class); Jacobi
is positive, Lanczos can change sign.
Test L addendum (lead's request; pre-registered 2026-10-28 before any L number). Sign: ORACLE s2 throughout; the table
compares amplitude parts only (at a deployable sign the Lanczos factor's own sign flips would also matter).
Extra feature f_L(x) = ln|1 + alpha (E_L(x) - E)|, psi = s2|A_s1|, E = <H>_psi, alpha = argmin of the exact <H> of the hop
state (as cx_o2b.py hop(), bounded). Fits: L6 = f_L alone; L7 = f_J + f_L. Exact E_FN for A e^{f_J} and A|1 + alpha(E_L - E)|.
Prediction [derived]: with E_L = H_xx + V_A - K_A (s2 frame), f_J = -ln(1 + (E_L - E')/K_A(x)) ~ -(E_L - E')/K_A(x): the
same direction as the Lanczos factor but with a configuration-dependent scale 1/K_A(x) instead of a constant alpha.
K_A varies across configurations, so f_J fits R* better: eps^2(L4) <= eps^2(L6) - 0.1 (absolute). |difference| < 0.05
counts as no difference; between 0.05 and 0.1 as weak. Kill of this prediction: eps^2(L6) < eps^2(L4) - 0.05.

## Result L (2026-10-28, ws1 CPU 2.6 h, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_l/l_result.json; ORACLE sign s2; verified by the angle lead)
Controls: D_RR reproduced (1.761485e-4); {R*} -> eps^2 0; f_J permuted -> 0.99998; normal equations = direct pass to 1e-15;
denominator min 9.14 > 0. (G near-singular for L1-L3/L5: exact collinear columns; eps^2 unaffected, coefficients not unique.)
| set | eps^2 |  | copy (exact E_FN, oracle s2) | tax/site | tax/gain |
| L1 2-body | 0.973 |  | A e^{1.33 f_J + small} (L5 fit) | 3.75e-7 | 0.048 |
| L2/L3 +n_blocked,+H_xx | 0.944 |  | A e^{f_J} (pure Jacobi, no fit) | 2.01e-6 | 0.257 |
| L4 f_J alone (coef 1.342) | 0.079 |  | (gain 7.825e-6; bar near fixed point < 1e-6) | | |
| L5 L3 + f_J | 0.068 |  | | | |
All pre-registered predictions met (L1 >= 0.85; L4 <= 0.3; L5 <= L4; tax/gain = eps^2 within 2x: 0.70 and 1.85).
Reading: the round's FN correction is, to 92-93%, one scalar times the copy's own Jacobi factor f_J; 2-body / blocked-count /
Ising features carry ~3-6%. This explains M-R's kill (an RBM residual at small weights is 2-body: L1 0.973 vs RBM 0.93-0.95).
The fitted scalar (1.34) uses the oracle R*; pure Jacobi (scalar 1) loses 26% of the gain.
Caveats: oracle sign s2; one copy (s1), one round, from far above the fixed point (gain 7.8e-6, not 2.1e-6).
Next (pre-registered now, before any number): oracle-free scalar. c* = argmin_c E_FN[A e^{c f_J}] (exact 1-D line search,
CPU, ~6 solves). Prediction: c* in 1.2-1.5 and tax/gain <= 0.08; kill: tax/gain > 0.2 at c*. Then repeat from phi_1's
neighbourhood (near the fixed point, gain 2.09e-6) to check the bar < 1e-6 where it matters.

## Result c* (2026-10-28, ws1 m53 CPU 10 min, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c_star/cstar.json; ORACLE s2; verified)
E_FN[A_s1 e^{c f_J}] - E0 per site: c 0: 2.092e-5; 1: 1.511e-5; 2: 1.106e-5; 2.5: 0.968e-5 (still falling; range edge).
Prediction "c* in 1.2-1.5" REFUTED; kill (tax/gain > 0.2) not triggered: at c = 2.5 the copy beats the exact one-round
FN state phi' (1.310e-5) and matches phi_1 (0.985e-5). <H> also falls (3.09e-5 -> 1.16e-5). Checks c = 0, 1 reproduce to
2e-7, 3e-6 (LOBPCG residual ~1e-6; my 1e-8 gate was too tight).
Mechanism of the miss [derived]: f_J ~ -(E_L - E')/K_A is a descent direction with a local step 1/K_A (an imaginary-
time / Jacobi step); the FN round's correction R* is a short step along it (fit 1.34), while minimising E_FN picks a
longer step: steepest descent with a line search. My prediction confused "matching one round" with "the best step".
Next (pre-registered before any number): extend to c in [2.5, 10]. Prediction: interior minimum c* in 3-6, with
E_FN(c*) between the s2 floor (~0.9e-6, compress_6x6 README) and 0.97e-5; I expect 4-7e-6. Kill of "one fused factor
reaches far": E_FN(c*) > 8e-6.
Control S (lead's request; pre-registered 2026-10-28 before any number; ORACLE s2): scanned Lanczos factor on the same copy,
A_s1 |1 + alpha (E_L - E)|, psi = s2|A_s1|, E = <H>_psi, alpha by the same exact line search on E_FN (scan alpha in
[-0.6, 0], refine to 0.01), same budget (~8 solves) as the Jacobi scan c in [0, 10].
Prediction [derived]: Jacobi wins. Both factors agree to first order (f_J ~ -(E_L - E')/K_A vs alpha (E_L - E)), but
|1 + alpha (E_L - E)| vanishes where E_L - E = -1/alpha, i.e. it creates zeros of its own at long steps (board 6), while
e^{c f_J} stays positive; the local scale 1/K_A also fits R* better (L addendum prediction). Margin: E_FN(best Jacobi) <=
E_FN(best Lanczos) - 1e-6 per site counts as a win; |difference| < 0.5e-6 is a tie. Kill: Lanczos lower by > 0.5e-6.
Result c* extension (2026-10-28, m53 12 min, pfaffian_c_star/ext/cstar.json; ORACLE s2; verified): interior minimum
c* = 3.92, E_FN - E0 = 8.026e-6/site (c 4: 8.031e-6; 6: 1.156e-5; 8: 2.158e-5; 10: 3.816e-5). <H> 1.131e-5 at c*.
Large factors (c*|f_J| > 5) on 6e-5 of orbits, 1e-11 of psi^2: no dominance by a few configurations.
VERDICT by the rule: c* in 3-6 MET; E_FN 4-7e-6 MISSED; kill "> 8e-6" FIRES by 0.3% (8.026e-6). So "one fused factor
reaches far" is killed as registered. What it does reach: from 2.092e-5 to 0.803e-5, 64% of the way to the s2 floor
(~0.9e-6), vs 37% for one exact FN round from the same copy (1.310e-5): about 1.7 rounds' worth, not the 10 rounds needed.
Result S (2026-10-28, ws1 m42 22 min, pfaffian_s/cstar_lanczos.json + extra/; ORACLE s2; verified): scanned Lanczos factor
A_s1|1 + alpha(E_L - E)|: alpha* = -0.10 (moment formula -0.098 agrees), E_FN - E0 = 8.683e-6/site (alpha -0.08: 9.08e-6,
-0.12: 9.00e-6, -0.2: 1.68e-5, -0.6: 1.84e-4). 150-190k orbits have |factor| < 0.05 (own near-zeros), yet it works.
Best Jacobi 8.026e-6 vs best Lanczos 8.683e-6: Jacobi lower by 0.66e-6 (8%). VERDICT: between "tie" (< 0.5e-6) and
"win" (>= 1e-6): weak Jacobi advantage, as registered; not a kill.
Reading: at the oracle sign, the amplitude part of the reframe (network + one Lanczos step + FN) is within 8% of the best
single local-energy factor we found; its moment alpha is already the E_FN-optimal step. One local factor of either kind
is worth ~1.7 exact FN rounds from this copy, not the ~10 to the s2 floor.

## OPEN QUESTION (PI, 2026-10-29, PLAN.md): can the stored network carry the Jacobi-shaped correction cheaply, so that the copy
loses < ~1e-6/site per round near the fixed point? Proposal OQ (board 20; written before any run; ORACLE s2 throughout).
Mechanism [derived]: for a pair product A = sum_g det M_g(x), a one-swap ratio is a rank-2 update of the same inverse
G_g(x) = M_g(x)^-1 (matrix determinant lemma), and a two-swap ratio a rank-4 update of it. So one Jacobi level
A_aug = A e^{c f_J[A]} lives inside the stored network: its ratios at all neighbours of a walker need n^3 + N^2 4x4 dets per
image vs n^3 + N 2x2 dets for plain A: ~15x plain at 6x6, ~3x at 14x14 (n = N/2), same O(N^3) class. Two levels would need
N^3 6x6 dets (~1e3-1e4 x): out. Over rounds nothing nests: each round refits the BASE PP to phi_k and recomputes f_J of the
new base; the stored object per round is (PP_k, c_k).
Where I disagree with the stated constraint: "a feature that needs neighbour sums at walker time is not cheap" holds for a
generic network, not for one level on a determinant/Pfaffian base; the low-rank update is the reason this family fits.
Measured, round 1: bare copy 2.092e-5, its target phi_1 0.985e-5, A_aug (c* 3.92) 0.803e-5: the stored augmented copy beats
its own target by 1.8e-6 (far from the fixed point).
T1 (CPU, now): seeds s2, s3. Prediction: both augmented copies beat phi_1 (E_FN < 0.985e-5), c* in 3-5. Kill: either > 1.1e-5.
T2 (CPU, now): f_J through the determinant lemma on the complex PP (all 576 images) for 1000 random x equals the table f_J to
1e-8; time per walker step (all neighbour ratios of A_aug) vs plain at 6x6. Prediction: ratio <= 20. Kill of "cheap": > 50.
T3 (ending test, GPU, needs the PI): round 2. phi_2 = FN(A_aug of s1) (exact, CPU); refit the bare complex PP to phi_2 as in
test C (1 seed, <= 1.5 H100-h); c* by line search (CPU). Pass: E_FN(A_aug,2) - E_FN(phi_2) < 1e-6/site AND E_FN(A_aug,2) <
E_FN(A_aug,1) - 0.5e-6 (the loop progresses). Kill: loss >= 2e-6 or no progress. Prediction: loss -1e-6..0, progress 1-2e-6.
Second arm of T3 (lead/PI): the refit with bands extended to phi^2 ~ 1e-30; ceiling ~11% of the bare-copy loss (Test F);
readout also the bulk e rms for phi^2 >= 1e-9 against the standard-band arm (shared-capacity trade-off). +<= 1.5 H100-h.
PI 2026-10-29 (via the lead, his words "I agree you can go ahead"): T3 and the deep-band arm approved, <= 1.5 H100-h each,
<= 3.0 total of the 4.3 left, CONDITIONAL on T2 passing (rank low, exact to 1e-8, timing as predicted); if T2 fails, report,
run nothing. Launcher read by me against this pre-registration; job ids recorded here.
At scale without the table (PI's question), per ingredient of A e^{c f_J}:
- f_J(x): from A's own ratios at x (determinant lemma), no table.
- E' = E_FN[A]: the FN-GFMC energy of guide A (mixed estimator), a standard output of the run.
- c: 1-D line search on the sampled FN energy of A e^{c f_J} (a few FN runs), or on VMC <H> as a cheaper proxy (6x6:
  <H>-optimal c ~3.4-3.9 vs E_FN-optimal 3.92); risk: noise in a flat minimum.
- base refit labels phi_k: NOT available without the table; at scale they must come from walkers (board 11), the open
  storage problem itself. This proposal does not solve that part.
- sign s2: oracle here; at scale a deployable sign, cost class of that sign.
T3 amendments (Astra xhigh review cj-44177b7587f9 via the lead, 2026-10-29; written before any T1/T2/T3 number is read):
- Control T3c: refit the PREVIOUS target phi_1 again with T3's exact budget, steps, warm start and seed (paired), then the same
  augmentation and c-scan. Excludes "fresh Jacobi improvement compensating an inaccurate refit" being read as stored FN information.
- T3 counts only if E_FN(A_aug, T3) <= E_FN(A_aug, T3c) - m, with margin m = max(1e-6, 2 x the seed scatter (std) of the
  augmented E_FN over s1-s3) per site; the scatter comes from T1 (s2, s3) plus s1's 0.803e-5.
- Report the bare refit E_FN and the augmented E_FN separately; a negative "copy loss" is not fidelity.
- c is rescanned in every round and reported; c* = 3.92 is not transferable.
- T2's timing is labelled "6x6, oracle sign"; it cannot establish the 14x14 deployable cost.
- Order: T3 and T3c first (T3c budget ~1.3-1.5 H100-h awaits the PI); the deep-band arm waits for the PI's answer.
PI 2026-10-29 (via the lead, his words: "GPU budget is agreed open from now on until further notice but wisely spent, no grind as we agreed, just sensible testing grounded in reasoning."): T3c approved; deep-band arm mine to schedule after T3/T3c only if its question still matters; every GPU job needs a written mechanism + predicted number first, sized from measured rates, cost recorded here; spend reported to the lead with each verdict.

## Result T2 (2026-10-29, ws1 m42 CPU, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_t2/t2_result.json; 6x6, oracle sign; verified)
Exactness PASS: lemma vs direct ratios 1.1e-11 (backflow) / 1.0e-11 (plain); f_J from lemma vs table 8.3e-14; two-swap
ratios 4.1e-10 / 1.9e-11; broken arms fail (rank-2 assumed for backflow: >= 0.13).
Rank: plain PP exactly 2 per swap, 4 per two-swap. Backflow PP: 12 (J1) / 14 (J2) per swap (every site whose 3x3 window
holds a swapped site), two-swap 16-18 = full rank at n = 18: the lemma saves nothing.
Cost per walker step, aug vs plain (one core; oracle-sign orbit lookup 0.36 s excluded, it would dominate):
plain PP 5.4x (op model 10.6; needs a small C kernel, numpy 55x); backflow PP 36.5x (op model 36.7).
VERDICT: the cost claim HOLDS for the plain PP (rank 2/4, 5.4x <= 20) and FAILS its prediction for the backflow copy s1
(36.5x > 20, below the 50 kill). T3 as registered (backflow base) does not run (PI's condition: T2 must pass).
Mechanism of the miss: backflow makes every F entry within the 3x3 window of a swapped site change, so rank ~ window size.
Redesign [proposal, not run]: keep the base low-rank: A = e^{J_RBM(x)} x PP (complex RBM+PP, no backflow; Nomura-Imada form,
our G2 guide). RBM ratios cost O(M) per swap (cached angles), the PP part stays rank 2/4, so A_aug stays ~5x plain. Copy
quality of that class is comparable on record (C8 RBM+PP storage tax_2 1.657e-6 vs F1 A PP+bf 1.846e-6, different setups).

## Test Z: the same numbers at an ORACLE-FREE sign (pre-registered 2026-10-29 before any number; lead's order; CPU only)
Sign: s1 = one Krylov (clipped-Lanczos) step of psi_P, as computed in experiments/compress_6x6/cx_refresh.py l.30-38
(the parent of the oracle s2 = s1 flipped on the FN expelled set X, l.46-48). Records: its self-consistent ceiling (exact
amplitude floor) 2.509e-6/site at 40 solves, Aitken 2.05e-6 (results/compress_6x6/README "Ceiling ... s_1"); own-PP sign
ceiling 7.0e-6 (A20) is worse, so s1 is the best oracle-free sign on record.
(a) floor at s1: taken from the record (2.05-2.5e-6); not re-run unless (b)/(c) land within 2x of it.
(b) Delta_0 at s1: E_FN[A_s1 copy, sign s1] and E_FN of its exact FN state phi'' (one round): prediction E_FN[A] 2.2-2.6e-5,
    gain 7-9e-6 (as at s2: 2.092e-5 -> 1.310e-5).
(c) c* scan of A e^{c f_J} at s1 (f_J recomputed with s1's blocked bonds and E' = E_FN[A, s1]): prediction c* 3-5,
    E_FN(c*) 0.9-1.1e-5 (s2: 0.803e-5 at 3.92, plus the ~1.2-1.6e-6 higher floor).
Decision this informs: if E_FN(c*) at s1 sits >= 3x above the s1 floor (> 6e-6), the copy, not the sign, is still what
binds, and the per-round bar stays (~1e-6 until within ~2e-6 of the floor). If (b)/(c) land within 2x of 2.5e-6, the sign
floor binds and T3' must aim at the sign, not the copy. Kill of "copy binds": E_FN(c*) < 5e-6 at s1.

## Result Z (2026-10-29, ws1 m42 CPU 22 min, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/z_result.json; sign s1, ORACLE-FREE; verified)
s1 reproduced: s2 = s1 flipped on X exactly (9.28% of orbits, 0 mismatches); E_FN[|psi_P|, s1] 9.54e-5 = record.
| E_FN - E0 per site, sign s1 | value | prediction |
| copy A_s1 | 2.503e-5 | 2.2-2.6e-5 |
| its one exact FN round phi'' | 1.608e-5 (gain 8.96e-6) | gain 7-9e-6 |
| A e^{c f_J}, c* = 3.9 | 1.018e-5 | c* 3-5, 0.9-1.1e-5 |
| s1 floor (record) | 2.05-2.5e-6 | |
All predictions met. VERDICT: the copy binds (1.02e-5 is ~4x the s1 floor; kill "< 5e-6" not met). The oracle-free sign
moves every number by ~2e-6 (the floor difference) and changes no conclusion. One orbit (n = 12, no kept hops) has f_J = -inf
at s1, set to 0. From here on T3' uses the oracle-free sign s1.

## T3' (pre-registered 2026-10-29; GPU only after the lead's Astra consult; sign s1 throughout)
Class: B = e^{R(x)} x PP(x), PP complex (f 36x36), R a real symmetric RBM (alpha 4, ln cosh, 576 images), NO backflow:
one-swap ratios = RBM ratio (cached angles) x rank-2 PP ratio, two-swap = RBM x rank-4 (T2: 5.4x plain at 6x6).
Augmented copy A = B e^{c f_J[B]}, c rescanned each round on exact E_FN (CPU).
Steps (each fit as test C: cx_c1fit pipeline, log loss, band sampler, flatness rule, max_steps 1100, same seed):
 R1: fit B_1 to phi'' (= FN(A_s1, s1), E 1.608e-5); A_1 = B_1 e^{c1 f_J[B_1]}.
 R2: phi_2 = FN(A_1, s1) (CPU); fit B_2 to phi_2 (warm start B_1); A_2 = B_2 e^{c2 f_J[B_2]}.
 Control R2c: fit B_c to phi'' again with R2's budget, steps, warm start (B_1) and seed; A_c = B_c e^{cc f_J[B_c]}.
Readouts (bare and augmented separately): E_FN[B_k], E_FN[A_k], c_k, E_FN[phi_k].
Predictions: E_FN[B_1] 1.8-3.0e-5 (no backflow; unverified class); E_FN[A_1] 0.9-1.3e-5; E_FN[A_c] within 1e-6 of E_FN[A_1];
E_FN[A_2] 0.5-0.9e-5. Round-1 gain g1 = 2.503e-5 - E_FN[A_1]; round-2 gain over control g2 = E_FN[A_c] - E_FN[A_2].
Pass: g2 >= m (m = max(1e-6, 2 x seed std of augmented E_FN; 3 seeds of test C/T1 give it)) AND E_FN[A_2] - E_FN[phi_2]
< 1e-6. Stall rule (lead): stop the loop if g2 < 0.5 x g1. My prediction: g2/g1 = 0.3-0.5, i.e. the stall rule likely
fires: each round removes a fixed fraction of the excess, the fixed-class bound one level up. Kill: g2 <= 0 (no progress).
Cost (measured rate test C 6.68 s/step complex PP+bf; no backflow is cheaper): 3 fits <= 1.5 H100-h each, <= 4.5 H100-h,
plus a code change (complex RBM x PP model block in a copy of cx_c1fit.py; CPU-checked as test C was).

## Lead decision 2026-10-29 after Astra (codex cj-6e37615f546c): T3' GPU DEFERRED; exact-table screen first (CPU, s1).
Stall rule g2 < 0.5 g1 WITHDRAWN (Astra derived: g1, g2 are not comparable successive gains; geometric convergence to the
s1 floor would give g2/g1 ~0.35 while useful). New criteria, pre-registered before any screen number:
 Delta = E_FN[A_1] - E_FN[phi_2] (the exact round-2 gain), tau = E_FN[A_2] - E_FN[phi_2] (storage loss after augmentation).
 Trained round useful: tau/Delta <= 0.5 AND paired-control advantage E_FN[A_c] - E_FN[A_2] >= m. Weak: 0.5 < tau/Delta <= 1.
 Storage costs more than the round: tau/Delta > 1. GPU (R1 only) released if the replay screen gives tau/Delta <= 0.7.
## Screen (pre-registered; s1, CPU, no training)
S1: phi_2 = FN(A_1), A_1 = A_s1 e^{3.9 f_J[A_s1]} (Z). Prediction: E_FN[phi_2] 0.70-0.80e-5; Delta 2.2-3.2e-6.
S2 (zero copy error): phi_2 e^{c f_J[phi_2]}, c scan incl. 0. Prediction: c* 2-4, min E_FN 0.45-0.60e-5 (Astra's
    geometric estimate 4.9-5.1e-6). A reference, not a bound on a trained round.
S3 (storage-error replay): B_2 = phi_2 e^{e_1}, e_1 = ln|A_s1| - ln phi_1 (round-1 copy error, normalised), then
    B_2 e^{c f_J[B_2]}, c scan incl. 0. Prediction: E_FN[B_2] - E_FN[phi_2] 1.0-1.5e-5; min augmented 0.70-0.95e-5;
    tau/Delta 0-0.7.
Queue (no run): Astra's retained dictionary A_K = |P_0| exp(sum_j c_j f_J[|P_j|]), backflow-free PP features; prediction
3 features eps^2 <= 0.30 on round 2's correction; kill eps^2 >= 0.70 or walker cost > 20x plain.

## Result screen (2026-10-29, ws1 m42 CPU 48 min, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_screen/screen_result.json; sign s1)
S1: E_FN[A_1] = 1.018474e-5 (= Z); E_FN[phi_2] = 7.978e-6; Delta = 2.207e-6 (prediction 2.2-3.2e-6: met, lower edge).
S2 (zero copy error): phi_2 e^{c f_J}: c 0: 7.98e-6; 2: 6.45e-6; 4: 5.69e-6; 5: 5.605e-6 (range edge, still falling).
    Min <= 5.605e-6: met (0.45-0.60e-5); c* 2-4: missed (>= 5). With a perfect copy round 2 gains another 2.4e-6+.
S3 (replay): E_FN[B_2] = 1.294e-4 (prediction phi_2 + 1.0-1.5e-5: missed 10x); best augmented 5.18e-5 at c 0.7;
    tau/Delta = 19.9 (prediction 0-0.7: missed). By the registered rule the GPU R1 release is NOT met.
Diagnosis of the S3 miss [measured, angle lead, from e1.npy, s1, s2]: e_1 is the round-1 copy's error against phi_1, an
s2-based target. On the flip set X (9.3% of orbits, 8e-7 of phi_2^2) e_1 has n-mean 0.59 and phi_2-weighted rms 1.06,
vs 0.11 and 0.010 elsewhere: the replayed field carries the structure where phi_1 and phi_2 differ by construction (the
sign change), so S3 measured a transplant mismatch, not the storage error a copy of phi_2 at s1 would make. My
pre-registration of S3 was the flaw (no s1-fitted copy exists to replay).
Lead 2026-10-29: S3 is recorded as INVALID (neither failed nor passed): it transplanted a copy error made against an
s2-based target onto an s1 target (rms 1.06 on X vs 0.010 off it). S2 is the reference: with a perfect copy round 2 still
gains >= 2.4e-6 at s1.

## R1 (approved by the lead 2026-10-29, <= 1.5 H100-h; pre-registered before submission; sign s1 throughout)
Fit B_1 = e^{R} x PP (complex PP f, real symmetric RBM alpha 4, no backflow) to phi'' = FN(A_s1, s1) (E_FN[phi''] =
1.608e-5), with the test-C pipeline switched to sign s1 everywhere (target, bad-bond sampling); warm start: PP f from the
complex copy s1's f block, RBM weights 0, Im of f as in s1's theta. Readouts on CPU afterwards (sign s1): E_FN[B_1] (bare),
c scan of A_1' = B_1 e^{c f_J[B_1]} (c 0..6, refined to 0.1), and the zero-error reference phi'' e^{c f_J[phi'']} (c scan).
Criterion (unchanged from "Lead decision 2026-10-29"): tau = E_FN[A_1'] - E_FN[phi''], Delta = E_FN[A_s1] - E_FN[phi'']
= 8.957e-6; R2/R2c released only if tau/Delta <= 0.7. Also reported (fidelity, not the gate): tau_ref = E_FN[A_1'] -
min_c E_FN[phi'' e^{c f_J[phi'']}] (augmentation on both sides), and tau_ref/Delta.
Predictions: E_FN[B_1] 2.0-3.0e-5 (class untested); E_FN[A_1'] 0.9-1.3e-5, so tau/Delta < 0 (gate passes; a negative
tau is not fidelity); zero-error reference 0.75-0.95e-5; tau_ref/Delta 0.1-0.5. Kill of the class: E_FN[B_1] > 4e-5 or
tau_ref/Delta > 1.
Cost: fit capped at 4200 s (step rate measured in the job; complex PP+bf was 6.68 s/step) + table; one H100 job <= 1.5 h.
RUNNING R1: Paderborn job 4058165 (gpu_h100, wall 01:30, fit_sec 3600, max_steps 1100; launcher launch_r1.sh / r1.sbatch read by the angle lead; code af662d3; watcher r1_pdb_watch.sh); run dir /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run. Cost <= 1.5 H100-h.

## Test K: growing store screen (PI "both go" via the lead; pre-registered 2026-10-29 before any number; CPU, sign s1)
Dictionary features f_J[|P_j|] at s1 (each with its own E'_j = E_FN[P_j, s1]) from different pair products: complex copies
s1, s2, s3 (all three share the F1 A parent), the plain PP start rpT1_pp (sym_tables_PP lP); real F1 A only if its
la_table is reachable (not local). Exact D-norm least squares (Test L machinery, D weighted by the target), greedy 1-2-3.
 K1: round-1 correction R*_1 = ln(phi''/|A_s1|). Prediction: f_J[A_s1] alone eps^2 0.05-0.15 (Test L at s2: 0.079);
     best 3 <= 0.10; other features add <= 0.03 (shared parent).
 K2 (Astra's actual target): round-2 correction R*_2 = ln(phi_2/|A_1|), A_1 = A_s1 e^{3.9 f_J[A_s1]}, phi_2 from the screen
     (pfaffian_screen/ln_phi2.npy). Prediction: best 3 eps^2 0.4-0.7 (near the kill: the features are near-collinear).
     Queue's rule: pass eps^2 <= 0.30; kill >= 0.70.
Independence readout: D-norm cosines between the features (prediction: s1-s2-s3 >= 0.9; rpT1_pp lower).
Exact E_FN (s1) for the best 3-feature store in each of K1, K2: least-squares c_j times one scalar s, s by line search
(about 5 solves each). For K2 also tau = E_FN - E_FN[phi_2] and tau/Delta (Delta 2.207e-6), next to S2's zero-error 5.6e-6.

## Result K (2026-10-29, ws1 m42 CPU 52 min, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_k/k_result.json; sign s1; verified)
Controls: {R*} -> 0; permuted feature -> 1.0000; f_J[A_s1] = Z's to 1.8e-15. Features: copies s1, s2, s3, rpT1_pp.
K1 (round-1 correction): single eps^2 A_s1 0.078 / s2 0.375 / s3 0.422 / rpT1 0.510; best 3 0.0776 (others add 1e-4).
    Predictions met. Best-3 store with line-searched scale: E_FN 1.018e-5 (same as A e^{3.9 f_J}).
K2 (round-2 correction ln(phi_2/|A_1|)): single eps^2 1.000 / 0.996 / 0.994 / 0.989; best 3 0.9705.
    KILL (>= 0.70); prediction 0.4-0.7 missed. Best-3 store on A_1: E_FN 1.0069e-5; tau/Delta 0.947.
Cosines between features 0.65-0.82 (prediction s1-s3 >= 0.9 missed): not collinear, yet they do not span R*_2.
Reading: round 1's correction is the current guide's own Jacobi factor (0.078); round 2's correction is orthogonal to the
Jacobi factors of the PRIMITIVE networks (eps^2 ~ 1). Hypothesis (to test, CPU): R*_2 ~ c f_J[A_1], the Jacobi factor of
the AUGMENTED guide itself, i.e. each round's correction is the current guide's own Jacobi factor; a store of primitive
features cannot hold it, only nesting (cost) or a fitted network can.
Test K3 (pre-registered now): eps^2 of R*_2 on f_J[A_1] alone (f_J of the augmented table, E' = E_FN[A_1] = 1.0185e-5).
Prediction eps^2 <= 0.15; kill of the hypothesis >= 0.5.
RUNNING R1 on theorie A40 (lead-approved parallel copy): job 17600579 (cip,inter, full A40, wall 09:00, fit_sec 25200 = the only non-path spec change; same code af662d3, seed 1); run dir /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run_a40; launcher launch_r1_a40.sh read by the angle lead; watcher r1_a40_watch.sh. Paderborn 4058165 stays queued (ETA 2026-10-31 17:30); no cancel without the PI's pre-approval. Cost ~8 A40-h. /scratch at 948.9/972.8 G (CRIT): R1 output ~130 MB.

## Result K3 (2026-10-29, ws1 m42 CPU, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_k/k3_result.json; sign s1; verified)
R*_2 on f_J[A_1] alone: eps^2 0.058 (coef 1.085); + K2's best 3: 0.056. Cosine of f_J[A_1] with the primitive features
0.01-0.09. Controls: R* -> 0, permuted -> 1.0000. PASS (prediction <= 0.15, kill >= 0.5).
E_FN[A_1 e^{c f_J[A_1]}]: min 7.219e-6 at c 2.4 (two nested Jacobi levels), below E_FN[phi_2] 7.978e-6.
UNDERSTOOD (mechanism, both rounds): each round's FN correction is ~1.1-1.3 x the CURRENT guide's own Jacobi factor
(eps^2 0.078 round 1, 0.058 round 2), and it is nearly orthogonal to the Jacobi factors of other networks. Consequences:
a store of fixed primitive features cannot hold round k's correction (Test K); nesting the factor holds it exactly but
costs a level per round (T2: two levels ~1e3-1e4 x plain). The only cheap route is to REFIT the base network each round
to the augmented target (fold), which is what R1/R2 measure.
PI 2026-10-29 via the lead, his words: "cancel yes once the first one starts, you are lead". Race watcher (r1_race_watch.sh, setsid) cancels the copy that has not started once the other shows a logged fit step; both in one window -> keep Paderborn 4058165, cancel A40 17600579. Log: /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/race.log and a [pfaffian R1 race] line in delta_today.md.

## R1 readout AMENDED (2026-10-29, before any R1 number; Astra cj-ff745902d93d via the lead; sign s1)
Flaw found by Astra: the unchanged copy plus its own Jacobi factor already gives tau/Delta = -0.658, so the augmented gate
passes without storing anything. New deciding number: the BARE post-fit E_FN of the refit, e(B).
Storage recovered r = (e(copy) - e(B)) / (e(copy) - e(phi'')) = (2.503e-5 - e(B)) / 0.895e-5.
Pass: r >= 0.5 (e(B) <= 2.055e-5). Kill: r <= 0 (e(B) >= 2.503e-5). Between: weak.
Prediction [derived from the class's measured absolute tax ~0.9-1.1e-5 vs a gap of 0.895e-5]: r = -0.2..0.3, i.e. the
bare refit will probably NOT store the round (the class's own copy error is as large as one round's gain).
Augmented numbers (B e^{c f_J[B]}, c rescanned, and the copy at the same depth) are reported separately, never as the gate.
Correction to "Result K3": R1 fits phi'', not an augmented target ("refit the base network each round to the augmented
target" there should read "to the round's FN state").

## Label-free arm L-SWO (prepared 2026-10-29; NOT to run until the PI's answer via the lead; sign s1)
Same architecture (complex RBM x PP, no backflow), warm start, optimiser, seed and budget as R1, but the target is
T = A e^{c f_J[A]} with c = 3.9 (Z; e(T) = 1.018e-5), computable per configuration from A's own neighbour ratios, with the
samples and weights generated from T (not phi''). This is supervised wavefunction optimisation (Kochkov-Clark,
arXiv:1811.12423; failure modes in Astra's note cj-ff745902d93d).
Delta_J = e(A) - e(T) = 1.485e-5; tau_J = e(fit(T)) - e(T). Screen pass: tau_J/Delta_J <= 0.5 (e(fit) <= 1.761e-5).
Kill: tau_J/Delta_J >= 1 (e(fit) >= 2.503e-5). Prediction: e(fit) 1.9-2.1e-5 (tau_J/Delta_J 0.6-0.75: between), from the
same class tax. Cost: one fit, <= 1.5 H100-h (or ~8 A40-h).

## Expert view on the loop's resting point (2026-10-29, PI's question via the lead; no run; sign s1)
Model (PI/lead): E_{k+1} - F = (1 - f)(E_k - F) + T  =>  E* = F + T/f. With f 0.39, T 1e-5, F 0.2e-5: E* ~ 2.8e-5; our copy
sits at 2.5e-5, so R1/L-SWO start at the resting point and are predicted not to move. I agree with the picture (~70%) as
the leading-order description, with two corrections that both RAISE E* or slow the approach:
1. T is not a constant. [measured identity, M-R: tax/gain = eps^2 to 1%] A warm-started refit loses T = T_abs + eps^2 g_k:
   T_abs = the class's misfit of the FN-state family (what does not change round to round), eps^2 g_k = the misfit of this
   round's correction g_k. For this class eps^2 on the correction is ~0.95 (M-R RBM, 2-body; L: 2-body 0.97), so the
   correction is essentially not stored: T ~ T_abs + 0.95 g_k. Then E_{k+1} - F ~ (1 - 0.05 f)(E_k - F) + T_abs: the
   loop barely contracts at all unless the class can represent the Jacobi-shaped correction (eps^2 small).
2. f falls with depth. [measured, s1] from the copy: 8.96e-6 of 2.28e-5 excess (f 0.39); from A_1: 2.21e-6 of 7.9e-6
   (0.28); [s2] from phi_1: 0.21. The ceiling iteration record (compress_6x6, s_1) shows ratio 0.53 -> 0.65 per 5 solves:
   sub-geometric. E* with constant f is an underestimate.
So the loop's value is to repair guides worse than E*, and its reach is set by T_abs and eps^2 of the class, not by f alone.
Scaling of T with N (what is known): no clean exact-size series of T in the records. Proxy: the class's variational error
per site grows with N: RBM+PP 4.5e-5 at 6x6 (I1: -0.503765 vs exact -0.50380965) vs ~2-3e-4 at 8x8 (I2: -0.498886 vs
DMRG -0.4992(1), var-extrap -0.49906): x5-7 from 36 to 64 sites. If T_abs tracks it at the 6x6 ratio (T ~1e-5 vs 4.5e-5,
~0.2) then T_abs(8x8) ~5e-5/site, against a measured FN gain at 8x8 of 8.3e-5/site (I3: VMC p=0 -0.498674 -> FN
-0.4987565): the loop's resting point would sit ~1.5e-4 above the sign floor at 8x8. [derived, unvalidated proxy]
Proposals (nothing runs before the PI agrees):
(a) Bad-start descent, 6x6, s1: start = plain complex PP (rpT1_pp, E_FN[s1] 5.72e-5), 4 exact-label rounds (FN exact on
    CPU, refit warm-started, same class as R1). Fixed-T model predicts 4.56 -> 3.86 -> 3.43 -> ~3.1e-5 (E* 2.8e-5);
    my corrected model (eps^2 ~0.95) predicts much smaller steps: <= 0.3e-5 per round after round 1. Kill of "the loop
    repairs bad guides": round 1 bare refit E_FN >= 5.0e-5. Cost: 4 refits x <= 1.5 H100-h (+ queue) + CPU FN.
(b) T(N) on exact sizes instead of 8x8 (8x8 T needs walker labels, the open board-11 problem): same class at 4x4, 4x6
    (exact tables exist), 6x6 (R1): T per site and eps^2 of the round correction at each N, one round from the class's own
    variational guide. Prediction: T/site grows with N (x2-4 from 16 to 36 sites); eps^2 of the correction stays >= 0.9 for
    a 2-body-like class. Kill of "T grows": T/site flat within 30% over 16-36 sites. Cost: CPU for 4x4/4x6 fits (small
    nets, exact), 6x6 from R1.
Recommendation: (b) first: cheap, and it decides whether the loop can matter at 10x10+ at all; (a) only if (b) shows T
grows slower than the FN gain.
AMENDMENT to "Expert view" (2026-10-29, Astra cj-6168204a40dd via the lead; I accept both):
- My "eps^2 ~0.95 for this class" is NOT established. M-R/L/K cover only small-weight two-body residuals and one
  primitive dictionary. Counter-record: O2b round 1 had T/g = 0.386 ((23.692 - 15.586 + 6.013)e-6 = 14.119e-6 tax, g as
  recorded). Corrected statement: eps^2 of the round correction is measured high for 2-body/small-weight residuals and
  primitive-PP dictionaries; for other classes it is open.
- My "f falls with depth" mixed guides and signs; along one guide family (O2b) g/(E - F) ROSE 0.684 -> 0.805. Withdrawn.
- Proposal (b) is deferred as a class-specific diagnostic; the next test is Test N (PI "Go").

## Test N: capacity of finite-weight RBM features (pre-registered 2026-10-29 before any number; CPU, sign s1)
Bank: frozen symmetric features h_j(x) = sum over the 576 images g of ln cosh(b_j + sum_i W_j(i) z_{g(i)}), W_j(i) ~
N(0, sigma^2/36), b_j ~ N(0, sigma^2), sigma in {1, 2} (nonlinear regime; field std ~sigma), 3 bank seeds. Nested sizes
M = 16, 32, 64, 128 (walker-cost cap: 576 x 36 x M flops <= the PP base's ~576 x 18^3, so M <= ~160; with cached angles
ratios cost O(M) per swap). Linear readout, exact D-metric least squares (L/K machinery, D weighted by the target).
Targets: (i) R*_1 = ln(phi''/|A_s1|); (ii) R*_2 = ln(phi_2/|A_1|) (K3); (iii) cumulative ln(phi_2/|A_s1|).
Controls: {R*} -> 0; permuted bank -> ~1; the bank plus L1's 2-body features reported next to L1 alone.
Grade: for the best bank on each target, exact E_FN[s1] of base x e^{s sum c_j h_j}, s by line search; bare-guide
tax/gain = (E_FN - E_FN[target]) / (E_FN[base] - E_FN[target]).
Predictions: Astra (low confidence): eps^2 <= 0.2 on (i) and (ii). Mine: eps^2 0.6-0.85 at M = 128 on (i) and (ii)
(the correction is a local-energy quantity of the guide, f_J, not a spin-pattern function; L: 2-body 0.97), (iii) similar.
Promote: tax/gain <= 0.3 at cost <= 2x base. Kill this family: eps^2 >= 0.7 at the cap, or tax/gain >= 1.
RUNNING L-SWO (PI 'Yes' via lead): paderborn=4062263 a40=17601653  ; launchers launch_lswo.sh / launch_lswo_a40.sh read by the angle lead (only target_la differs from R1); race watcher lswo_race_watch.sh (first logged step wins, else keep Paderborn). Cost <= 1.5 H100-h or ~8 A40-h.
R1 status 2026-10-29 23:24: first A40 copy 17600579 FAILED at startup (ModuleNotFoundError st6_sector; fixed by PYTHONPATH); resubmitted A40 17603167 STARTED (step 10) -> race watcher cancelled Paderborn 4058165 (PI pre-approval via lead). Paderborn codeR1 got st6_sector.py/vit_dt.py (sha-checked) for L-SWO 4062263. L-SWO: Paderborn 4062263 + A40 17603168 pending, race watcher re-armed.
L-SWO status 2026-10-30 00:35: Paderborn 4062263 STARTED (step 30) -> race watcher cancelled A40 17603168 (PI pre-approval). Running: R1 A40 17603167, L-SWO Paderborn 4062263.
L-SWO fit done (Paderborn 4062263, 2026-10-30 01:05): 520 steps, converged (held-out change 0.0147 < 0.02), 2.99 s/step, 0.58 H100-h; la_table pfaffian_lswo/run/round_1/la_table.npy (sha b65ab273). CPU readout e(fit) running. R1 A40 17603167 running.

## Result L-SWO (2026-10-30; fit Paderborn 4062263, 520 steps, converged, 0.58 H100-h; CPU readout
/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/readout/readout.json; sign s1; verified)
e(fit) = 1.957e-5/site: BETWEEN (pass <= 1.761e-5, kill >= 2.503e-5); prediction 1.9-2.1e-5 MET; tau_J/Delta_J = 0.632
(prediction 0.6-0.75 met). The label-free fit recovers (2.503 - 1.957)/(2.503 - 1.018) = 37% of the Jacobi gain, without phi''.
Copy error vs T: pair cost 5.28e-5/site, 76% in the moderate class (r within 3x), deep band 72% of cost, median e +0.05.
Augmented, NOT the gate: fit e^{4.0 f_J[fit]} 8.34e-6 vs A_s1 e^{3.9 f_J} 1.018e-5: one label-free round + augmentation
lowered the augmented guide by 1.84e-6. <H>(fit) 2.83e-5.
Reading: the class stores ~37% of a round's Jacobi improvement label-free; bare e(fit) 1.96e-5 is above one exact FN round
from the copy (phi'' 1.608e-5) but well below the copy (2.503e-5).

## CORRECTION 2026-10-30 (Astra cj-d0bc7446d51c via the lead; verified by the angle lead): the RBM was INACTIVE
cx_r1fit.py l.196/260 start the RBM block at W = 0, b = 0. For R = sum ln cosh(b + W z) both derivatives are tanh(.) x (z or 1)
= 0 there, a stationary point, so the block never moves: L-SWO's saved theta has the last 148 entries exactly 0 (max |.| 0.0).
My spec ("RBM weights W = 0, b = 0") caused it; the CPU Jacobian check ran at W, b != 0 and could not catch it.
RELABEL: R1 (A40 17603167) and L-SWO (4062263) both fit "complex PP, no backflow, RBM inactive" (2592 active parameters).
They remain a valid PAIRED comparison of that class; every number above under "R1"/"L-SWO" refers to it.
Fix for any future run: small random RBM init (W ~ N(0, 0.01^2), b ~ N(0, 0.01^2)) with a CPU check that the RBM Jacobian
columns are nonzero at the start.
Astra, also recorded: the fit's convergence flag is a 1.47% held-out-loss plateau, not a vanishing gradient; its fixed point
is neither VMC nor E_FN stationarity (band-uniform weights put ~120x more weight on decade -13 than |A|^2; exact condition
in the note).

## Competitor bar (PI via the lead 2026-10-30; a REFERENCE in every readout, not a kill)
RBM+PP VMC + one signed Lanczos step + FN, own sign, 6x6: 5.6e-6/site; same walker-cost class as network + one Jacobi factor.
Our best so far: 8.34e-6 (one label-free round, complex PP no backflow RBM inactive, + e^{4.0 f_J}, sign s1) = 1.5x the bar.
Every augmented E_FN is reported against it with equal-cost accounting: factor evaluations per walker step (one level =
~5.4x plain at 6x6 for a plain PP, T2) and the training cost of label-free rounds (L-SWO: 0.58 H100-h per round).
The scale claim (loop stores rounds at one-level walker cost while k Lanczos steps nest) is testable only by an equal-cost
comparison at >= 8x8; 6x6 favours Lanczos.
R1 A40 17603167 done 2026-10-30 06:53: 935 steps, NOT converged (budget; held-out change 0.0426 > 0.02), 26.8 s/step,
7.6 h wall (~7.6 A40-h); la_table run_a40/round_1/la_table.npy (sha fae7af85). CPU readout running.

## R1b: R1 rerun on Paderborn H100 to convergence (PI via the lead 2026-10-30; pre-registered before submission)
Same class as L-SWO (complex PP, no backflow, RBM inactive), same spec, seed 1, from scratch (same warm start as R1, not
the unconverged A40 theta), same convergence rule (flatness: held-out change < 0.02 over the window), which stopped L-SWO
at 520 steps. Only the caps change: max_steps 3000, fit_sec 9000 (measured ~3 s/step on H100), hard wall 03:00:00
(<= 3 H100-h). Readout and gate as "R1 readout AMENDED" (bare e(B), r = (2.503377e-5 - e(B))/0.895711e-5, pass >= 0.5,
kill <= 0; prediction r -0.2..0.3) plus the competitor-bar reference. The A40 R1 result is marked "unconverged,
superseded" once R1b reads out. Default from now: Paderborn H100; a theorie A40 race only if Paderborn's estimated start
is > ~1 day away. The fixed RBM init is used only in new arms.
RUNNING R1b: Paderborn job 4070227 (wall 03:00, fit_sec 9000, max_steps 3000; launcher launch_r1b.sh read by the angle lead; watcher r1b_pdb_watch.sh); run dir /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run_b. Cost <= 3 H100-h.

## Test Θ: a learnability law from the network's tangent kernel (PI via the lead 2026-10-30; pre-registered; CPU, no training)
Linearised fit: ln|A_{θ+δ}| ~ ln|A_θ| + J δ, J = d ln|A| / dθ per configuration. In the fit's own metric the storable part of a
target correction R is its projection onto span(J); finite training reaches only the top eigen-directions of the Jacobian Gram
(spectral bias; kernel learnability: Canatar-Bordelon-Pehlevan arXiv:2006.13198, Rahaman et al. arXiv:1806.08734, abstracts
only). Computed on blocked pairs sampled by the D weights (as Test K), metric D:
 G = sum_pairs w (J_x - J_y)(J_x - J_y)^T  (P x P),  v_i = eigenvectors,  C(k) = sum_{i <= k} (proj of R on v_i)^2 / D(R,R).
 C(P) = full linearised storable fraction; 1 - C(P) = linear eps^2.
Candidates (existing checkpoints): (a) complex PP+bf copy s1 (theta, 3888); (b) L-SWO fit (complex PP, RBM inactive, 2592);
(c) R1 A40 fit (same class, unconverged); (d) R1b when done; (e) same class with the FIXED RBM init (W, b ~ N(0, 0.01^2));
(f) small-weight RBM residual of M-R (alpha 16; its kernel is the 2-body span, cross-check against L1's 0.973).
Targets: R*_1 = ln(phi''/|A_s1|), R*_2 = ln(phi_2/|A_1|), f_J[A_s1]; all at s1.
Diagnostic it adds: C(P) low -> the class lacks the correction's directions (expressibility of the correction);
C(P) high but C(k ~ steps) low -> spectral bias (learnability). Prior: lead ~55% spectral bias; mine ~40% (I expect C(P)
itself low for f_J-shaped targets, as L1/K showed for 2-body and primitive features).
Ending test: rank the candidates with a measured storage on a common target by C(k = 200), and compare with the measured
order. The measured points that exist: M-R RBM eps^2 0.93-0.95 (R*_1 at s2); L-SWO kept 37% of the Jacobi gain; R1/R1b r;
Test N bank eps^2 (if it finishes). The law WORKS if the C-ranking reproduces the measured order on >= 3 comparable
points with no inversion; it is DROPPED on an inversion between points whose measured gap is > 2x their uncertainty.
Ranking stability: Spearman correlation of the per-direction weights of R*_1 vs R*_2 under each kernel; >= 0.7 = transfers.
Cost: J at ~16k sampled configurations x <= 3888 parameters per candidate on CPU (JAX), eigen-decomposition P x P; ~1 h per
candidate on one ws node. Runs after R1b reads out and Test N's capacity question is settled (lead's order).

## Result R1 (A40 17603167; NOT converged, budget-limited at 935 steps; class complex PP, no backflow, RBM inactive; sign s1;
/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/readout/readout.json; verified)
e(B) = 2.391e-5/site, r = 0.126: BETWEEN (prediction -0.2..0.3 met). Bond split vs phi'': 1.99e-5/site, 86% moderate class.
Not the gate: B e^{4.0 f_J[B]} 9.72e-6; zero-error reference phi'' e^{5.3 f_J} 7.18e-6; tau_ref/Delta = +0.28.
Competitor bar (reference): 5.6e-6; best augmented so far 8.34e-6 (L-SWO). To be marked superseded once R1b (converged,
4070227) reads out. Note: L-SWO (label-free, converged) e 1.957e-5 is below R1 (exact labels, unconverged) 2.391e-5; the
pair is not comparable until R1b converges.
R1b fit done (Paderborn 4070227, 2026-10-30 08:20): 1390 steps, CONVERGED (held-out change 0.0043 < 0.02 from step 1110), 2.99 s/step, 1.30 H100-h; la_table run_b/round_1/la_table.npy (sha f3c19d8a). CPU readout running. The A40 R1 result is now 'unconverged, superseded' pending this readout.

## Result R1b (Paderborn 4070227, CONVERGED 1390 steps, 1.30 H100-h; class complex PP, no backflow, RBM inactive; sign s1;
/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/readout_b/readout.json; verified). R1 (A40) is now "unconverged, superseded".
e(B) = 2.373e-5/site, r = 0.146: BETWEEN (prediction -0.2..0.3 met). The converged exact-label refit stores 15% of one
FN round's gain; converging moved r from 0.126 to 0.146. Pair cost vs phi'' 2.04e-5/site, 86% moderate class.
Not the gate: augmented 9.64e-6 (c* 4.0); zero-error reference 7.18e-6; tau_ref/Delta +0.275. Competitor bar 5.6e-6.
PAIRED with L-SWO (same class, both converged): label-free fit to T = A e^{3.9 f_J} reaches e 1.957e-5, better than the
exact-label fit to phi'' (2.373e-5) by 0.42e-5; augmented 8.34e-6 vs 9.64e-6. Exact labels do not beat the label-free
Jacobi target in this class.
Reading: in this class one stored round keeps ~15% (exact labels) to ~37% (Jacobi target) of the available improvement; the
loop's per-round storage loss is most of the gain (T/g ~ 0.85 for exact labels). Consistent with L/K: the class lacks the
Jacobi-shaped direction. Test Θ (kernel) is the next step to say whether that is expressibility or spectral bias.

## Network questions from the PI (2026-10-30, on paper; [lit] = literature, [mem] = id from memory, not re-checked;
[judg] = my judgment). Search budget: 3 arXiv queries; only arXiv:2302.01941 (Chen-Heyl, minSR) verified by title.
0. Premise correction: our storage fits are NOT first-order: cx_*fit.py uses damped Gauss-Newton (minSR in sample space
   when rows <= params), i.e. second order in the fit's own metric (docstring "Optimiser damped Gauss-Newton").
1. Initialisation. [judg, derived] Under GN/SR the update solves (J^T J + lambda) d = J^T r, so a block whose Jacobian
   columns vanish never moves, whatever the loss curvature: ln cosh at W = b = 0 (all derivatives tanh(0) = 0), and Im f at
   a real warm start (d ln|A| / d Im f = 0), are frozen points for any GN-type optimiser. Rule: every block must start
   with nonzero Jacobian columns; check rank(J_block) at step 0 on CPU. Because SR/GN are invariant to linear
   rescaling of parameters, the scale matters little once it is nonzero: RBM W, b ~ N(0, 0.01^2..0.05^2), Im f ~ 1% of
   |Re f|, PP f from the previous round (or a mean-field pairing). [mem] Nomura et al. arXiv:1709.06475 and Nomura-Imada
   arXiv:2005.14142 start the PP from a mean-field (Hartree-Fock/BCS-type) pairing and the RBM small and random; the
   assumption is that the PP carries the sign/node structure and the RBM a small correction (not checked in the text).
2. Optimiser. [judg] For a least-squares storage fit, Gauss-Newton in the loss's own metric IS the natural method; SR with
   the |psi|^2 QGT would weight by the model's own distribution (the bulk), the opposite of what storage needs (the tail).
   So: keep GN/minSR, but in the fit's metric (as now). [mem] Fidelity-based fits with natural-gradient preconditioning:
   projected time-dependent VMC (Sinibaldi et al., arXiv:2305.14294), SWO (Kochkov-Clark arXiv:1811.12423); not re-read.
   Spectral bias under GN: preconditioning by (J^T J + lambda)^-1 equalises the learning speed of all kernel directions
   with eigenvalue >> lambda; only directions below the damping stay slow. So Θ must report, besides plain C(k), the GN
   version C_GN(lambda) = share of R on kernel eigen-directions with eigenvalue > lambda_used (our lam_rel x tr(J^TJ)/P),
   and C(P). Expectation [judg]: under GN, "spectral bias" reduces to "below the damping"; if C(P) is low, the class lacks
   the direction (expressibility of the correction) and no optimiser fixes it.
3. Parameter noise / Langevin SR. [mem] Noise provably escapes strict saddles (Ge et al. arXiv:1503.02101) but not true
   minima; Langevin dynamics samples rather than optimises. [judg] For the zero-init case the cure is the init (item 1),
   not noise. Our fits are not noise- or plateau-limited in the usual sense (acceptance ~1.0, held-out plateau, M-R's
   quadratic law): L/K/K3 point to a missing direction, which noise cannot create. I would not add noise.
4. Architecture for holding the Jacobi-shaped correction at constant-factor walker cost (ranked):
   (1) ~45%: backflow-free pair product with an explicit Jacobi head, guide = B e^{c f_J[B]}; each round refits B to the
       previous augmented guide (L-SWO: 37% of the Jacobi gain kept, guide 8.34e-6). Mechanism: the round correction IS
       the guide's own f_J (L 0.078, K3 0.058); one swap is rank 2 and two swaps rank 4 of the same inverse, so the head
       costs n^3 + N^2 4x4 dets vs n^3 + N 2x2: ratio ~ 1 + c/N (5.4x at 6x6 measured, T2; falling with N). Kill: a
       second label-free round does not lower the bare B by > 0.5e-6 (the refit stops absorbing), or op-count > 20x
       plain at 10x10.
   (2) ~35%: (1) plus a two-body Jastrow e^{J2} (or an RBM with the fixed init) on B for the bulk; Jastrow ratios for one
       and two swaps are O(1) with cached fields, so the cost class is unchanged. Kill: no gain in bare e(B) over (1) at
       equal steps (bulk not the limiter).
   (3) ~20%: richer backflow (neural or windowed) inside the determinant. Every swap then changes ~window entries
       (T2: rank 12-14 for a 3x3 window, two-swap full rank), so the Jacobi head costs ~window-fold more (36.5x measured)
       and the advantage over Lanczos-type steps disappears. Kill: any measured head cost > 20x plain at the target size.
   Note [judg]: the determinant structure is what makes the head cheap; an RBM-only base has O(M) per swap but N^2 two-swap
   configurations, a ratio ~N that grows with size.

## L-SWO round 2 (PI GO via the lead 2026-10-30; pre-registered before any number; class as L-SWO: complex PP, no backflow,
RBM inactive; Paderborn H100; label-free; ~1.2 H100-h)
Base B_1 = L-SWO fit (pfaffian_lswo/run/round_1/la_table.npy, theta.npy), e(B_1, s1) = 1.957e-5; f_J[B_1] at s1 =
pfaffian_lswo/readout/fJ_fit.npy; augmented guide A_1' = B_1 e^{4.0 f_J} = 8.34e-6 (s1).
ARM s1: target T2 = A_1' (ln T2 = ln B_1 + 4.0 f_J[B_1], normalised); fit B_2 warm-started from B_1's theta (same spec,
seed 1, flatness rule, max_steps 3000, fit_sec 9000), sign s1. Delta_J2 = e(B_1) - e(T2) = 1.12e-5.
 Prediction: bare e(B_2) 1.45-1.75e-5 (keeps ~20-45% of Delta_J2, as round 1's 37%); augmented B_2 e^{c f_J[B_2]}:
 c* 3.5-4.5, 6.5-7.8e-6. KILL (candidate (1)): e(B_1) - e(B_2) <= 0.5e-6.
ARM s_G (single round 2 from B_1 under the competitor's signed-Lanczos sign s_G, rebuilt as in ansatz/code/test_x.py and
saved): f_J^G[B_1] with s_G's blocked bonds and E'_G = E_FN[B_1, s_G]; c_G by an exact line search under s_G (CPU);
target T_G = B_1 e^{c_G f_J^G}; fit as arm s1 but sign_path = s_G. Confound (stated now): B_1 was trained under s1, so this
arm is "round 1 under s1, round 2 under s_G"; its gain mixes the sign switch with the round. Separation: the zero-training
start point E_FN[B_1 e^{c f_J^G[B_1]}, s_G] (Test X gave 7.39e-6 at c 4.0 with s1's f_J) is reported, and the round is
judged on bare B_2^G vs bare B_1, both at s_G.
 Prediction: e(B_1, s_G) 1.7-1.9e-5; e(T_G) 7.0-7.5e-6; bare e(B_2^G) 1.3-1.6e-5; augmented 5.8-7.0e-6.
 KILL: e(B_1, s_G) - e(B_2^G, s_G) <= 0.5e-6. Competitor bar 5.6e-6 (s_G) as the reference for the augmented number.
Cost: 2 fits x ~0.6 H100-h (L-SWO took 0.58) + CPU prep and readouts.
Round-2 prep (2026-10-30, CPU, pfaffian_lswo2/prep_result.json; verified by the worker's controls): E_FN[T2, s1] 8.3376e-6;
s_G rebuilt (controls rel <= 1e-11; 9.92% of orbits differ from s1; sha 94eb19c9). Arm s_G start points (predictions
MISSED low, i.e. better): E_FN[B_1, s_G] = 1.545e-5 (predicted 1.7-1.9e-5); c_G = 4.0, e(T_G) = B_1 e^{4.0 f_J^G[B_1]}
at s_G = 6.069e-6 (predicted 7.0-7.5e-6): without any new training, 0.47e-6 above the competitor bar 5.6e-6 at the same sign.
Queues (--test-only, 13:35): Paderborn ~50 h, theorie A40 ~38 h -> both copies of each arm submitted, race rule as before
(A40 copy capped at ~840 steps; L-SWO round 1 converged at 520).
RUNNING L-SWO round 2 (race per arm, PI rule): s1 4077957 2026-10-30T15:14:11+01:00 sG 4077958 2026-10-30T15:14:14+01:00 s1-a40 17614777 2026-10-30T15:14:29+01:00 sG-a40 17614778 2026-10-30T15:14:30+01:00 ; launchers launch_lswo2.sh / launch_lswo2_a40.sh read by the angle lead; race watchers lswo2_race_watch.sh.

## Result N (2026-10-30, ws1 CPU ~4 h/bank after a thread-oversubscription restart; /scratch/a/A.Otaifi/ansatz_runs/pfaffian_n/;
sign s1; verified grade_result.json): KILLED. Finite-weight symmetric RBM feature banks (sigma 1, 2; 3 seeds; M <= 128):
eps^2 at M = 128: (i) R*_1 0.954-0.959, (ii) R*_2 0.972-0.975, (iii) cumulative 0.984-0.986; seed scatter <= 0.004;
L1 (2-body) alone 0.977 / 0.983 / 0.993: the bank adds 2-3% over 2-body. Controls: R* -> 0; permuted bank -> 0.999.
Graded exact E_FN: tax/gain 0.958 / 0.971 / 0.978 (kill >= 1 nearly; eps^2 >= 0.7 fires). Predictions: Astra (<= 0.2)
refuted; mine (0.6-0.85) also too optimistic. The correction is not a spin-pattern function at any order this bank reaches;
consistent with L/K/K3 (it is the guide's own f_J). This feature family is closed.

## Result Θ (2026-10-30, ws1 CPU ~5 h, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_theta/; sign s1): INVALID as designed
Contract estimator (32768 D-sampled pairs, 2 seeds): C(P) 0.63-0.86 for every pair-product candidate, C(200) 0.16-0.38,
C_GN 0.17-0.67; M-R RBM C(P) 0.06-0.12. Read literally that is "spectral bias". It is not trustworthy: [measured, the
worker's held-out diagnostic] coefficients fitted on one pair sample and scored on the other give C(P) < 0 for all PP
candidates (min-norm -0.8..-24; GN-damped -0.01..-1.7), and for the M-R RBM in-sample 0.11 vs held-out 0.055 and exact L1
0.023 (5x overstatement). With rank 2376-3494 against 32768 heavy-tailed pairs the in-sample projection overfits. My
pre-registration's flaw: no held-out score and too few pairs for P ~ 3000. Transfer Spearman 0.11-0.23 (far below 0.7),
also in-sample. Controls pass (J v_1 -> 1.000000; noise in-sample 0.04-0.13, held-out <= 0; J vs finite differences 4e-8).
The learnability law is NOT judged. A valid version needs a held-out C(P) at N_p >> P^2/... in practice ~1M pairs per
candidate (~2M J evaluations at ~80 conf/s ~ 7 h CPU per candidate) or exact sums. Proposal, not run: L-SWO (b) and
fixed-init (e) only, N_p = 2 x 524288, held-out C(P) and C_GN; cost ~2 x 7 h CPU.
Θ v1 status: INVALID (design fault: no held-out score, too few pairs); it is not a result.

## Test Θ v2 (lead GO 2026-10-30; pre-registered before any number; CPU only; sign s1)
Candidates: (b) L-SWO fit; (e) (b) with the fixed RBM init (W, b ~ N(0, 0.01^2), seed 0); calibration (f) M-R RBM alpha 16.
Pairs: two independent D-sampled sets A, B of 524288 blocked pairs each (target-phi weights as in v1).
Estimators (all with coefficients fitted on one set and scored on the other, both directions A->B and B->A):
 C_ho(P)  : ridge with lambda = 1e-6 tr(G)/P (near min-norm);  C_ho(GN): ridge with lambda = lam_rel tr(G)/P (the fit's
 final damping, b: 0.0625; e: 1e-2); C_ho(200): least squares on the top-200 eigenvectors of G_train.
 C_ho = 1 - D_test(R - J c_train) / D_test(R). In-sample values are reported but marked "overfit, not quotable".
Calibration gate (must pass before any PP number counts): C_ho(P) of (f) on R*_1 within 0.02 of the exact L1 value 0.023.
Targets: R*_1 primary; R*_2 and f_J[A_s1] secondary.
Verdict (R*_1, both directions): C_ho(P) <= 0.1 -> missing direction (expressibility of the correction);
 C_ho(P) >= 0.5 AND (C_ho(200) or C_ho(GN)) <= C_ho(P)/2 -> slow learning (spectral bias); else inconclusive.
Prediction (mine, ~70%): missing direction for (b) (C_ho(P) <= 0.1); (e) at most 0.05 higher. Lead/Astra prior: spectral bias.
Cost: ~2M J evaluations per candidate at ~80 conf/s ~ 7 h, (b) and (e) in parallel on two ws nodes, (f) minutes.

## Result Θ v2 (2026-10-30/31, ws1 CPU 3.4 h per PP candidate, /scratch/a/A.Otaifi/ansatz_runs/pfaffian_theta2/; sign s1)
CALIBRATION GATE FAILED as registered: (f) M-R RBM on R*_1 C_ho(P) = 0.069 / 0.065 against 0.023 +- 0.02 (C_ho(GN) 0.024 /
0.022 does match). So by the contract no PP number counts. My reading of the failure [derived, unvalidated]: a held-out
score of a fitted model is biased LOW, not high, so 0.067 > 0.023 says the reference is mis-specified, not the sample:
the alpha-16 RBM's linear span (592 columns, 1- and 2-body terms with finite b) is larger than L1's 9 symmetric 2-body
features, so 0.023 was the wrong yardstick for (f). Needs the lead's ruling.
Held-out on R*_1 (A->B / B->A): b C_ho(P) -0.48/-0.37, C_ho(GN) -0.009/0.016, C_ho(200) 0.007/0.021;
 e -0.55/-0.38, -0.023/-0.006, 0.001/0.041. f_J and R*_2 the same picture (GN, 200 in -0.11..0.10).
Provisional reading (if the gate is re-ruled): the tangent space of the trained PP class holds ~0-4% of the round
correction under any regularised fit (C_GN, C_200), the near-min-norm fit overfits (negative), and the fixed RBM init adds
nothing: "missing direction" at the linear level (my prediction), not spectral bias. Caveat: this is the linearised class at
the trained point; the trained fits (R1b, L-SWO) stored 15-37% by moving nonlinearly.
Θ v2 status (lead's ruling 2026-10-31): "calibration gate FAILED; provisional reading 'missing direction' (PP C_ho(GN)
-0.02..0.02, C_ho(200) 0.00..0.04), not established". The gate is NOT re-ruled.

## Θ v2 calibration check (pre-registered 2026-10-31 before any number; CPU; sign s1)
New reference: C_exact = the exact D-norm projection share of R*_1 onto the alpha-16 M-R RBM's own 592 Jacobian columns
(d R_theta / d theta at theta_a16), with D summed over ALL blocked pairs at s1 (phi'' weights), C_exact = b^T G^+ b / D_RR
(ridge 1e-10 tr(G)/P for numerical rank). Gate passes only if BOTH: |C_ho(P)_f - C_exact| <= 0.02 for both directions
(0.069, 0.065), AND C_exact >= max(C_ho(P)_f) (my bias argument: a held-out score is biased low; if this fails the argument
is wrong too). Pass -> the v2 PP numbers count under v2's verdict rule with no rerun. Fail -> Θ closed as unmeasurable
with this estimator. Prediction: C_exact 0.07-0.12.
L-SWO round 2 fits (2026-10-30/31): A40 copies won both races at 15:24 (Paderborn 4077957/58 cancelled before start); s1 17614777 and sG 17614778: 935 steps each, NOT converged (A40 budget 25200 s; held-out change 0.0086 / 0.0748 vs 0.02; sG held-out loss ended above its start), 7.64 A40-h each; la_tables run_s1_a40 / run_sG_a40 (sha 283576f9 / 69b22a28). CPU readout running.

## Result L-SWO round 2 (2026-10-31; A40 fits 17614777 / 17614778, 935 steps, NOT converged; class complex PP, no backflow,
RBM inactive; /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/readout/{s1,sG}/readout.json; verified)
| arm (sign) | bare before | bare after | drop | kill <= 0.5e-6 | augmented min (c*) | prediction aug |
| s1  | 1.957e-5 | 1.831e-5 | 1.26e-6 | not killed | 7.74e-6 (4.0) | 6.5-7.8e-6: inside |
| s_G | 1.545e-5 | 1.418e-5 | 1.27e-6 | not killed | 5.391e-6 (4.0) | 5.8-7.0e-6: beaten |
Bare predictions: s1 1.45-1.75e-5 MISSED high (kept ~11% of Delta_J2 = 1.12e-5); s_G 1.3-1.6e-5 inside.
Competitor bar at s_G (RBM+PP VMC + one signed Lanczos step + FN; Test X x_result.json 5.574e-6): our augmented round-2
guide 5.391e-6 is 0.18e-6 BELOW it. Caveats: one seed, fits not converged (budget-limited), round 1 under s1 / round 2 under
s_G (stated confound), 6x6 only, s_G is the competitor's own sign. Equal-cost: walker cost one Jacobi level ~5.4x plain
(T2, plain PP) vs one Lanczos step (same two-hop class); training 2 label-free rounds (0.58 H100-h + 7.6 A40-h) vs the
competitor's VMC training (not counted here).
Candidate (1) (backflow-free PP + Jacobi head, refit per round) SURVIVES its kill in both arms.

## R2G-conv: round 2 under s_G to convergence, 2 seeds (lead/PI takeover approval 2026-10-31; pre-registered before any number)
Same as the s_G arm (target T_G = B_1 e^{4.0 f_J^G[B_1]}, warm B_1 theta, class unchanged, sign s_G), seeds 1 and 2 (sampler),
convergence rule as R1b (flatness < 0.02, max_steps 3000). Paderborn H100 cont (fit_sec 9000, wall 03:00) raced against a
theorie A40 copy per seed (fit_sec 43200, wall 13:00, so ~1600 steps can converge); first logged step wins, loser cancelled
(PI pre-approval). Readout as round 2 (bare e at s_G, drop vs 1.5453e-5; augmented c scan 0..7 with 3.5/4.5 filled).
Predictions: converged bare e(B_2^G) 1.36-1.42e-5 per seed; augmented minima 5.2-5.5e-6; seed spread <= 0.15e-6.
QUOTABLE RULE for the bar crossing: BOTH seeds' augmented minima < 5.574e-6 AND |seed difference| < 5.574e-6 - max(minima).
Kill of candidate (1) at convergence: bare drop <= 0.5e-6 in either seed.
Then (only after this reads out): round 3 from the better converged seed under s_G, <= 1 H100-h. Lead's stretch ceiling
4 H100-h total. Full-Lanczos-factor arm deferred (budget).
Equal-cost note (training): competitor RBM+PP 6x6 guide training 14.3 GPU-h (RTX 2080 Ti / A40 slices) of which ~6.5 h on
plateaued runs (results/rbmpp_6x6/README.md section 4). Ours on top of a PP start that came from that project (rpT1_pp):
F1 A real fit ~1.2 H100-h, complex copy (test C) ~0.7 H100-h, L-SWO round 1 0.58 H100-h, round 2 7.6 A40-h (unconverged).
RUNNING R2G-conv: s1 4079208 2026-10-30T23:20:50+01:00 s2 4079209 2026-10-30T23:20:52+01:00 s1-a40 17623357 2026-10-30T23:20:57+01:00 s2-a40 17623358 2026-10-30T23:20:57+01:00 ; launchers launch_r2g.sh / launch_r2g_a40.sh read by the angle lead; race watchers r2g_race_watch.sh per seed.

## Result Θ v2 calibration (2026-10-31; ws1 m42 CPU 2.7 h; /scratch/a/A.Otaifi/ansatz_runs/pfaffian_theta2/calib/calib_result.json; verified)
C_exact = 0.0821 (ridge 1e-10; 1e-12 0.0822, 1e-8 0.0780) on all 519,978,046 blocked pairs; prediction 0.07-0.12: inside.
|C_ho(P)_f - C_exact| = 0.013 / 0.017 <= 0.02 and C_exact >= 0.069: GATE PASSES. Controls: L1 eps^2 0.977183 vs pfaffian_n
0.9771831; permuted L1 C 3.5e-5 (fails as designed); D_RR equal to pfaffian_n to 4e-12; J vs finite differences 4.4e-8
(perturbed-theta control 6.7e-3).
=> Θ v2 PP numbers count. VERDICT (linear tangent space of the complex PP at B_1, s1): MISSING DIRECTION for both
candidates b and e: held-out C(GN) -0.023..0.016, C(200) 0.001..0.041, so the tangent space holds 0-4% of the round
residual R*_1. Scope: linearised at the trained point, not the nonlinear refit; C_ho(P) < 0 is overfit of the near-min-norm fit.
Reading: storage of the Jacobi correction by local refit is not available to this class; the explicit Jacobi head carries it
(consistent with K/K3/N: round correction orthogonal to features, eps^2 0.95-0.98).
R2G-conv fits done (2026-10-31): Paderborn won both races (A40 17623357/58 cancelled at step 60). Seed 1 job 4079208: 2926 steps,
NOT converged (out of 9000 s budget; flatness 0.0088 from step 2340), 2.57 H100-h. Seed 2 job 4079209: 950 steps, converged
(flatness 0.0070), 0.89 H100-h. Held-out loss ROSE in both (5.30->5.54e-3, 5.56->6.95e-3): watch whether bare e still drops.
Stretch spend 3.46 of 4 H100-h, so round 3 (<= 1 H100-h) would exceed the ceiling by ~0.5 h: needs the lead's OK.
CPU readout running (r2g_readout.py, readout/{g1,g2}).

## Result R2G-conv (2026-10-31; H100 fits 4079208 / 4079209; CPU readout /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g/readout/{g1,g2}/readout.json; verified)
| seed | fit | bare e | drop vs 1.5453e-5 | aug min (c*) |
| 1 | 2926 steps, NOT converged (budget), sha 9be6a663 | 1.3839e-5 | 1.61e-6 | 5.207e-6 (3.98) |
| 2 | 950 steps, converged, sha b00dd2da | 1.4090e-5 | 1.36e-6 | 5.290e-6 (3.99) |
All three predictions hit (bare 1.36-1.42e-5; aug 5.2-5.5e-6; spread 0.083e-6 <= 0.15e-6). Kill not fired.
QUOTABLE RULE MET: both minima < 5.574e-6; |diff| 0.083e-6 < 5.574e-6 - 5.290e-6 = 0.284e-6. Bar margin 0.28-0.37e-6 (5-7%).
Caveats (stand): seeds share warm start B_1 and target T_G, so the spread measures the optimiser path only, not independence;
s_G is the competitor's sign; walker cost ~5.4x plain PP vs one Lanczos step; 6x6 only. Held-out copy loss ROSE in both fits
while E_FN fell: the copy loss does not track the FN energy (the known weighting mismatch), so convergence by loss is a weak criterion.
Code check: r2g_readout.py reproduces the round-2 sG bare e to 4e-11 relative. Spend 3.46 H100-h of the 4 H100-h stretch.
Round 3 (pre-registration: from the converged seed 2) waits on the lead's OK for ~0.5 H100-h above the ceiling.

## Round 3 under s_G (lead GO 2026-10-31: theorie A40 only, no Paderborn copy, H100 ceiling fixed; pre-registered before any number)
Start: B_2 = R2G seed 2 (converged, la sha b00dd2da, theta run_s2/round_1/theta.npy). Target T_3 = B_2 e^{4.0 f_J^G[B_2]}
(c 4.0 = c* 3.99), built as lswo2_prep.py built T_G; control: E_FN[T_3, s_G] must equal the g2 readout's c=4 point 5.2902e-6
(rel 1e-6). Class and spec as R2G seed 2 (complex PP, RBM inactive, seed 2), sign s_G.
Mechanism (eq. 20 loop): each round stores a fraction r of the available Jacobi gain g = e(B) - e(B e^{c f_J}) in the base, and
the augmented guide improves by about half the bare drop. Measured under s_G round 2: r = 1.36/9.38 = 0.145 (seed 2), aug/bare
drop ratio 0.53-0.57; under s1 r fell 0.37 -> 0.11 from round 1 to 2. Θ v2 (P16): tangent space holds 0-4%, so storage is
nonlinear and r should keep falling. Round 3: g = 1.409e-5 - 0.529e-5 = 0.880e-5; r 0.08-0.16.
PREDICTIONS: bare e(B_3) 1.27-1.34e-5 (drop 0.7-1.4e-6); augmented min 4.6-5.0e-6 at c* 3.5-4.5. The term that is wrong if
it misses: r (storage fraction per round).
STALL RULE: bare drop <= 0.5e-6 (r <= 0.057) -> the loop stalls at this class; the next step is then the class, not more rounds.
STOP RULE (E_FN, not loss): an in-loop E_FN stop is too costly on A40 (26.8 s/step; one table build 2085 s plus a ~15 min CPU
solve per readout). Fixed budget instead: exactly 1500 steps (min_steps = max_steps = 1500, so loss flatness cannot stop it),
fit_sec 41000, wall 13 h; theta saved at steps 900 and 1200 (small spec option in cx_r1fit.py). Afterwards one short A40 job
builds the two checkpoint tables; CPU readout of e at 900, 1200, 1500 (bare) and the augmented scan at 1500.
Converged iff |e(1500) - e(1200)| < 0.05e-6; otherwise "not converged", reported with the slope per 300 steps.
Cost: ~12 h + ~1.2 h A40, ~1 h CPU. Queue: submit even if ETA > 1 day.
Hardware change (lead 2026-10-31, PI lifted the H100 ceiling): round 3 also staged on Paderborn H100 (same spec, fit_sec 9000, wall 3 h) and raced against the A40 copy; first copy to log a step wins. Pre-registration unchanged.
RUNNING round 3 (race): Paderborn fit 4079680 + tabs 4079681 (afterok); A40 fit 17625929 + tabs 17625930 (afterok); A40 fit_sec 42500; launchers launch_r3_pdb.sh / launch_r3_a40.sh read by the angle lead; ids in pfaffian_r3/job_ids.txt.

## Result round 3, partial (2026-10-31; Paderborn fit 4079680 won the race, 1500 steps, 1.33 H100-h; tables 4079681;
readout /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3/readout/{r3_900,r3_1200,r3_1500}/readout.json; verified)
| step | bare e (s_G) | drop vs 1.4090e-5 | held-out copy loss | |dtheta| over previous 300 |
| 900  | 1.3390e-5 | 0.70e-6 | 7.3e-3 | |
| 1200 | 1.3133e-5 | 0.96e-6 | 8.2e-3 | 0.137 |
| 1500 | 1.2311e-5 | 1.78e-6 | 1.65e-2 | 1.33 |
E_FN convergence rule FAILS: e(1500) - e(1200) = -0.82e-6 (limit 0.05e-6). NOT converged; not stalled.
The bare prediction 1.27-1.34e-5 held at 900 and 1200. Between 1200 and 1500, though, the damping lam_rel collapsed from 7.8e-3
to 2.4e-4. Theta then moved 10x further, the held-out copy loss doubled, and E_FN still fell 0.82e-6.
Hypothesis (not established): the late drop is undamped Gauss-Newton steps that move away from the target T_3 in directions the
copy loss penalises but E_FN rewards. This is the copy-loss / E_FN mismatch again, so the stored-fraction model (r) does not
describe this segment. Kill test: E_FN at intermediate checkpoints (e.g. 1300, 1400) plus a continuation run. If E_FN reverses
upward while the loss keeps rising, the drop was a lucky walk; a monotone fall would mean the loss is a poor proxy and E_FN
should drive the stop or the step.
Augmented scan at 1500 (incomplete): c 3.0 4.951e-6, 3.5 4.614e-6, still falling; minimum pending (Slurm CPU readout).
Round 3 augmented scan at 1500 (Slurm CPU 17629941, readout/r3_1500/readout.json, done True, verified): minimum 4.520e-6 at c* 3.9
(parabola 3.94, 4.5195e-6). This is BELOW the prediction 4.6-5.0e-6, and 1.05e-6 below the competitor's 5.574e-6. NOT QUOTABLE:
one seed, not converged by the E_FN rule. Spend: 1.33 H100-h plus 14 min CPU.
Proposed next (awaiting the lead): continue from the step-1500 theta for 600 steps with checkpoints every 150 (~0.5 H100-h plus
4 tables plus ~1 h CPU). Prediction (55%): E_FN keeps falling, more slowly, and flattens at 1.18-1.22e-5. Kill of "systematic
drop": E_FN rises by > 0.1e-6 at any checkpoint, which would make the late drop a walk; round 3 then counts at e(1200).

## R3-cont: continuation + damping control (lead GO 2026-10-31; pre-registered before any number)
Mechanism under test (hypothesis H_u): the Gauss-Newton step is dtheta = (S + lam I)^{-1} g, where S is the copy-loss Gram
matrix. As lam -> 0 the step is dominated by the small-sigma eigenvectors of S. The copy loss is nearly blind to those
directions, but E_FN (through V_A on blocked pairs) is not, so the 1200->1500 drop came from the collapsed damping.
Arms (Paderborn H100, run in parallel; same class, target T_3, sign s_G; tau = tau_min = 0.01, no anneal; min = max steps):
 A: warm theta(1500), seed 2, lam0 = the fit's final lam_rel, default floor 1e-6; 600 steps; theta saved every 150
    (global 1650/1800/1950/2100).
 B: warm theta(1200), seed 1, lam0 = lam floor = 7.8e-3 (no collapse); 900 steps; theta saved every 150
    (global 1350...2100).
Logged at every held-out evaluation, next to the held-out loss: the TRAINING copy loss on one fixed batch drawn from the
training orbits (B_ho configurations, separate key). Code change: a spec key lam_floor (default 1e-6 = old LAM_MIN) and the
fixed training batch; nothing else.
Readout: bare e at every checkpoint (one CPU job, K/Vm built once); augmented c scan at the two arm ends.
Definitions: "falls" = e(end) - e(start) <= -0.2e-6 with no checkpoint rising > 0.1e-6; "flat" = |e(end) - e(start)| < 0.2e-6;
a rise > 0.1e-6 at any checkpoint = the kill rule.
Verdicts:
 - A falls and B is flat or falls at most at the old slow rate (<= 0.26e-6 per 300 steps): H_u supported; the next step is
   the step/stop design, worked out on the board first.
 - Both fall comparably (B reaches within 0.1e-6 of A at an equal global step): just more training; round 3 counts at the
   arm that meets the E_FN convergence rule.
 - A rises > 0.1e-6 at a checkpoint and B is flat: the drop was a walk; round 3 counts at e(1200) = 1.313e-5.
 - Overfit flag: if in A the training loss falls while the held-out loss rises, the A-arm fall is labelled
   "overfit-driven, unconfirmed" and needs a seed replicate before it counts.
Predictions: A (55%) falls, more slowly, to 1.18-1.22e-5 at global 2100. B falls slowly to 1.24-1.29e-5 at 2100
(H_u true) and stays above A. Training loss in A: falls while held-out rises (60%).
Cost ~1.25 H100-h fits + ~0.4 H100-h for 8 table builds + ~2 h CPU.
R3-cont staging notes (verified by the angle lead): A lam0 = 0.03125 (the code's lam_rel at step 1500; the collapse low was 2.44e-4 at steps 1301-1400 and the default floor never bound); B lam0 = lam_floor = 0.0078125; cx_r1fit.py sha 41f766cc (lam_floor + fixed training batch L_tr; the broken-version check fails as it should). Submitting via launch_r3c_pdb.sh; ids go to pfaffian_r3c/job_ids.txt.
RUNNING R3-cont: A fit 4081019 + tabs 4081020; B fit 4081021 + tabs 4081022 (Paderborn); CPU readout submitted later by r3c_chain.sh.

## Result R3-cont (2026-10-31; Paderborn 4081019-22, CPU 17630898; pfaffian_r3c/readout/readout.json; verified; 1.94 H100-h)
Bare e per site (s_G) at each global step:
| step | 1200 | 1350 | 1500 | 1650 | 1800 | 1950 | 2100 |
| round-3 fit | 1.3133 | - | 1.2311 | | | | |
| A (from 1500, seed 2) | | | 1.2311 | 1.2341 | 1.2401 | 1.2467 | 1.2592 |
| B (from 1200, seed 1, lam floor 7.8e-3) | 1.3133 | 1.3015 | 1.2872 | 1.2895 | 1.2901 | 1.3084 | 1.2885 |
(all e-5). Augmented minima: A end 4.614e-6 (c* 4.0), B end 4.708e-6 (c* 3.9); round 3 at 1500 was 4.520e-6.
Gate: UNCLASSIFIED; KILL RULE fired (A +0.125e-6 at 2100, B +0.18e-6 at 1950). Overfit flag false (L_tr and L_ho flat in A).
H_u is NOT tested: A's lam never re-collapsed (median 0.031). The step-1500 low was not a stable gain: continuing from it lost
0.28e-6 monotonically.
=> By the pre-registered kill rule, round 3 counts at e(1200) = 1.3133e-5 (drop 0.96e-6, r = 0.96/8.80 = 0.11, inside the
predicted 0.08-0.16; r per round under s_G 0.145 -> 0.11).
STEP BACK (second failed version of the stop-rule design):
1. Verdict: the loop works and r falls about 25% per round. The E_FN stop rule as designed is ill-posed.
2. The thought that changed it: along a fit at a FIXED target, E_FN walks by 0.1-0.2e-6 per 150 steps (B: +0.18 then -0.20), which
   is 2-4x the 0.05e-6 tolerance. Optimiser noise lives in directions the copy loss is flat in (S has small sigma there) but E_FN
   is not. This is the same mismatch as before, now acting as noise rather than drift. The textbook remedy for stochastic-
   iterate noise in flat directions is iterate averaging (Polyak-Juditsky 1992, SIAM J. Control Optim. 30, 838; from memory,
   not re-read).
3. Cheapest next test: Polyak average of B's six saved thetas (global 1350-2100, one basin, one run): 1 table (~3 min H100)
   + 1 CPU solve. Prediction: e(theta_bar) <= 1.285e-5, i.e. below B's checkpoint mean 1.293e-5 and at or below its best,
   1.287e-5. Miss = averaging does not remove the walk; then the stop is a fixed step budget plus the min over saved checkpoints
   (a selection, labelled as such).
Also now (CPU, free): the augmented scan at theta(1200) gives the official round-3 augmented number. Prediction 4.6-4.9e-6
(all 3 trajectory points measured so far are 4.52-4.71e-6).
RUNNING averaging test: Paderborn table 4081131 (theta_bar sha 1ec8f715) + theorie CPU readout 17632500 (controls, aug at theta(1200), then theta_bar); pfaffian_r3avg/.

## Result averaging test (2026-10-31; Paderborn 4081131, CPU 17632500; pfaffian_r3avg/readout/readout.json; verified; ~0.1 H100-h)
| guide (s_G) | bare e | aug min (c*) |
| theta(1200), round 3 by the kill rule | 1.3133e-5 | 4.786e-6 (3.9) |
| B checkpoints 1350-2100 | mean 1.293e-5, best 1.287e-5 | B end 4.708e-6 |
| theta_bar = mean of B's 6 thetas | 1.2522e-5 | 4.644e-6 (4.0) |
Controls: B final rel 0, tab_1200 rel 4.5e-11. Predictions: e(theta_bar) <= 1.285e-5 HIT, 0.35e-6 below B's best checkpoint;
aug(1200) 4.6-4.9e-6 HIT.
Mechanism confirmed at this level: E_FN's walk along the fit is iterate noise in copy-loss-flat directions, and E_FN is convex
enough there that the average beats every iterate. Polyak-Juditsky averaging of the tail is the stop for future rounds.
Official round 3 (s_G): bare 1.313e-5 (r 0.11), augmented 4.786e-6 = 0.79e-6 below the competitor's 5.574e-6 (one seed).
With averaging: 4.644e-6 (one run's checkpoints).

## Round 4 (lead GO 2026-10-31; ~1.5 H100-h; pre-registered before any number; NOT YET STAGED: the fresh session starts here)
Protocol: averaged. Start theta_bar (pfaffian_r3avg/theta_bar.npy, sha 1ec8f715; table pfaffian_r3avg/tabs/tab_bar). Target
T_4 = B_bar e^{c f_J^G[B_bar]} at c = 4.0 (c* of theta_bar), built as r3_prep.py built T_3. Control: E_FN[T_4, s_G] equals
theta_bar's c=4 point 4.6438e-6 (rel 1e-6). Fit: class and spec as round 3 (sign s_G, seed 2, tau 0.01 after the anneal as
round 3); fixed 1500 steps; save theta every 150 from step 600; theta_4 = mean of the saved thetas at 600..1500 (7 iterates).
Paderborn H100 (codeR3c, cx_r1fit.py sha 41f766cc), with an A40 race copy only if the Paderborn ETA is > 1 day.
Readout (CPU Slurm, K/Vm once): bare e and the augmented scan for theta_4; bare e at each checkpoint for the record.
Predictions: r_4 = (1.2522e-5 - e(theta_4)) / 0.7878e-5 = 0.11-0.15; bare 1.13-1.17e-5; aug 4.0-4.3e-6.
Kills: r_4 >= 0.178 kills the DECLINE (H_q); a bare drop <= 0.5e-6 kills PROGRESS. q = r_4 / 0.178 is the number the
lead uses to decide whether the loop is worth scaling. Templates: launch_r3c_pdb.sh, r3c_bare.py, r3avg_bare.py, r3_prep.py.
RUNNING round 4 (race): Paderborn fit 4085199 + tabs 4085200; A40 fit 17639808 + tabs 17639809; readout chained by r4_chain.sh; target T_4 control rel 4.3e-11 (pfaffian_r4/prep_result.json); launchers launch_r4_pdb.sh / launch_r4_a40.sh read by the angle lead. ETAs A40 ~2 Nov 20:34, Paderborn ~6 Nov.

## Result round 4 (2026-11-01; Paderborn 4085199/4085200 won the race, A40 17639808/09 cancelled at 13:04; CPU 17640414;
pfaffian_r4/readout/readout.json; verified; ~1.8 H100-h)
theta_4 = mean of the 7 iterates at 600..1500: bare 1.2184e-5 (theta_bar 1.2522e-5), drop 0.337e-6, r_4 = 0.043;
augmented 4.367e-6 (c* 4.0). Iterates 1.240-1.266e-5 (scatter 0.25e-6); the average beats every one of them again.
Predictions MISSED: r_4 0.11-0.15, bare 1.13-1.17e-5. Augmented 4.0-4.3e-6: just above the range.
VERDICT: the PROGRESS kill fired (drop <= 0.5e-6); the decline kill did not (r_4 < 0.178).
Matched protocol: round 3's averaged r = 0.178 included a one-off averaging bonus (single-iterate to average, 0.35e-6, about
0.04 of g). From one average to the next, r_4 = 0.043 against single-iterate r_3 = 0.109, so q is about 0.3-0.4, not 0.75.
The term that missed is r, the per-round storage of the nonlinear refit, which saturates by round 4 at this class
(complex PP, no backflow, RBM inactive).
Loop limit with q = 0.35: bare about 1.20e-5, augmented about 4.3e-6. By the pre-registered stall rule the next step is
the class, not more rounds.

## PROPOSAL for the PI: what comes after the stall (2026-11-01; paper only, nothing runs without the PI)
The stall: the refit's per-round storage r fell 0.145 -> 0.109 -> 0.043, and the one-level augmented guide sits at 4.37e-6.
To get below 4.0e-6 with one more matched round, a class needs r > ~0.09 (Astra's eq. 2, assuming the augmented guide gains
half the bare drop; unestablished).
Ranking criterion: P(augmented 6x6 < 4.0e-6 AND walker cost <= 20x plain PP at 10x10). Astra (gpt-6-astra xhigh, blind to my
ranking): ansatz/REVIEW_astra_after_stall_2026-11-01.md. My blind ranking, written before reading it:
ansatz/lead_blind_ranking_after_stall_2026-11-01.md.
| option | mechanism (term changed) | walker / plain at 10x10 (derived, Astra eq. 7-9) | prediction | kill test, cost | Astra | me (blind -> after) |
| (d) B = abs(P_1 + lambda P_2), two jointly refit PPs, one Jacobi level | new interference direction Re P_2/(P_1+lambda P_2) outside the PP tangent space; rank 2/4 per component kept | 11x | r 0.16, aug 3.7e-6 | two matched seed pairs vs single-PP continuation from B_4, fixed steps + averaging: r <= 0.05 and aug >= 4.2e-6 in both kills it; ~12 H100-h | 35% | not considered -> 25% |
| (a) active RBM (m = 4 prototypes per image, nonzero init), joint refit; small fields ~ a two-body Jastrow | bulk ratio corrections in K_B and V_B; the explicit level keeps the hard part | 7.7x (chi = 8 assumed) | r 0.08, aug 4.05e-6 | active vs inactive from B_4, 2 seeds: r <= 0.05 and aug gain < 0.1e-6 kills it; ~8 H100-h | 20% | 20% (Jastrow) / 10% (RBM) -> 20% |
| (c) second explicit Jacobi level G_2 = G_1 e^{c2 f_J[G_1]} | adds exactly the missing direction (K3), with r = 0 | ~650x (three-shell endpoints) | aug ~3.5e-6 (s1 analogue: 10.2 -> 7.2e-6) | exact c2 scan on existing tables, <= 16 CPU core-h; cost already fails | <1% | 15% -> <1% |
| (b) 3x3 backflow PP | pairing depends on the spin environment | ~4300x | r 0.15, aug 3.8e-6 (weak) | 10x10 kernel timing, 2 CPU core-h; fails at once | <1% | 5% -> <1% |
Records corrected by Astra:
- T2's 36.5x was backflow-augmented over backflow-bare. Against plain PP it is ~800x by op count (2842x in raw timing, with
  unmatched walker counts).
- My blind cost for (a), "1 + 32 alpha", wrongly assumed M = alpha N hidden units per image. The code has m = 4 prototypes per
  image, which gives 7.7x at 10x10.
- The plan's "falling ~1 + c/N" for the plain head is replaced by Astra's model: 5.5x at 10x10 and 3.8x at 14x14 (derived).
My recommendation, at about 45% that it crosses 4.0e-6: (d), then (a) as its cheaper fallback.
- (c) is not deployable, but its exact c2 scan (CPU only, free) gives the accuracy ceiling of explicit levels. I would run it
  only as a reference line for the paper.
- (d)'s collision risk is that the second PP lands in the same tangent neighbourhood (P16). G11 (a rank-20 vison/BCS sum,
  killed at 4x6) is the closest dead end. (d) differs in that it is refit to the Jacobi target rather than a variational energy.
Cost of the PI's choice: (d) ~12 H100-h, (a) ~8 H100-h, (c) reference 16 CPU core-h. Each would be pre-registered as above
before it runs.

## s_G floor (lead request 2026-11-02; CPU only; pre-registered before any number)
Floor(s) = E_FN at the self-consistent amplitude for a fixed sign s: phi <- the FN ground state of H_FN[phi, s], iterated.
This is the best any amplitude copy can reach with that sign. Method identical to the records (plain iteration from |psi0|,
Aitken on the last solves; experiments/compress_6x6/cx_veto.py / cx_s2ceil.py), run for s_G, s1 and s2 with one code.
Controls: s1 2.51e-6 at 40 solves (Aitken 2.05e-6); s2 ~0.9e-6 (cx_s2ceil.json).
PREDICTION: floor(s_G) 1.2-2.0e-6 per site. Then today's best guide (theta_4 augmented, 4.367e-6) sits 2.4-3.2e-6 above it,
so the amplitude, not the sign, still holds 55-75% of the guide's error at s_G.
Additions (lead/PI 2026-11-02, same tables and solver; one-line predictions written before any number):
- (A) Exact ED sign s0 plus the ED amplitude |psi0|: a control. It must give E0, i.e. e < 1e-10 per site (the solver
  residual is ~1e-6, so the energy error is quadratic).
- (B) s0 blocked bonds (s0 x H x s0 > 0): they exist, because frustration leaves positive sign-frame elements even at the
  exact sign. I predict 10-30% of J2 bonds and < 5% of J1 bonds by count, with a |H| psi0 psi0-weighted share of 1-5%.
  If there are none, (C) must be exact.
- (C) s0 plus the theta_4 amplitude: bare 0.8-1.2e-5; augmented (c rescanned on s0's walls) 1.5-3.0e-6, well below s_G's
  4.37e-6, because the walls move to where the exact node is.
- (D) Floor at s0 = 0 to solver precision (psi0 is the fixed point). Floors s_G 1.2-2.0e-6 (mine, as above); the lead's
  unmeasured guess is ~0.3e-6.

## Two-PP class (d) (PI chose "A first, C in parallel", 2026-11-02; this angle owns C; ~12 H100-h; pre-registered before any run; board block 27)
Mechanism: the stall is r -> 0 because the round residual lies outside the single PP's tangent space (Θ v2: C 0-4%; r 0.145
-> 0.109 -> 0.043). B = |P_1 + lambda P_2| adds d ln B / d lambda = Re P_2/(P_1 + lambda P_2) and P_2's own tangent space, a
configuration-dependent interference that one PP cannot make. Term changed: the tangent space O_x in the refit (board 22 (4),
27). Each P_j keeps rank-2/4 lemma updates, so the walker cost is ~2x one PP.
Setup: P_1 = theta_4's PP (pfaffian_r4 tabs theta_4.npy); P_2 = round-1 L-SWO PP (pfaffian_lswo/run/round_1/theta.npy, a
different trained PP, finite separation); lambda complex, init 0.1; RBM inactive as before. Target T_5 = theta_4
e^{4.0 f_J^G[theta_4]} (E_FN 4.367e-6), sign s_G. Fit as round 4: damped GN, fixed 1500 steps, theta = mean of 7
checkpoints 600..1500. Arms: two-PP seeds 1 and 2; CONTROL = single-PP continuation from theta_4 to the same T_5, seeds 1
and 2. r_5 = (1.2184e-5 - e(theta_bar)) / 0.7817e-5.
Predictions: two-PP r_5 0.06-0.16 (bare 1.09-1.17e-5), aug 3.4-4.1e-6 (Astra: r 0.16, aug 3.74e-6). Control r_5 0.01-0.04
(bare 1.19-1.21e-5), aug 4.20-4.35e-6.
KILL (d): both two-PP seeds r_5 <= 0.05 AND aug >= 4.2e-6. NOT ATTRIBUTABLE: two-PP r_5 minus control r_5 < 0.03 in either
seed (the gain is just one more round).
COST (must be measured before any accuracy claim counts): CPU timing at 10x10 with the T2 convention (all Hamiltonian-
neighbour ratios per walker step, lemma updates, random complex F, compiled kernel). Arms: plain one PP; one PP + one
Jacobi level; two PPs + one Jacobi level. Prediction: two-PP augmented over plain one-PP 6-12x (Astra's model 11.0x;
T2 measured/model 0.51 at 6x6); one-PP augmented 3-6x. KILL of deployment: > 20x. Also re-time 6x6 to anchor against T2's 5.4x.
Order: the CPU timing and the two-PP code (CPU checks) run in parallel; the GPU fits are submitted only after the timing
passes. Fallback (a), an active RBM, only if (d) is killed.

## Result s_G floor + exact-sign additions (2026-11-02; CPU 17646674-76, 17652669; pfaffian_floor/{sG,s1,s2,s0}/floor.json; verified)
Self-consistent floor, iterated as in the records from the ViT-ratio start (not |psi0|, corrected), 40 solves (e-6 per site):
| sign | solve 10 | 20 | 30 | 40 | Aitken |
| s_G | 7.79 | 2.67 | 1.338 | 0.827 | 0.39 |
| s1 | 10.39 | 4.62 | 3.085 | 2.509 | 2.05 |
| s2 (oracle) | 7.84 | 2.78 | 1.436 | 0.915 | 0.48 |
Controls: s1 and s2 reproduce the records to 1e-9 relative. The iteration is still converging (ratio 0.926), so 0.83e-6 is an
upper bound.
Prediction floor(s_G) 1.2-2.0e-6 MISSED low; the lead's guess of ~0.3e-6 was closer. s_G's floor is below the oracle s2's.
Gap of today's best guide (theta_4 augmented 4.367e-6) to the s_G floor: >= 3.54e-6, i.e. the amplitude holds >= 81% of
the error (91% with Aitken). Prediction 55-75% MISSED (low).
Exact ED sign s0:
- (A) |psi0| gives e = 7e-16 (PASS). (D) psi0 is the fixed point (PASS).
- (B) blocked bonds at s0, which my prediction missed:
  - by count: J1 29%, J2 58%, all 44%;
  - by |H| psi0 psi0 weight: 16.8% of all bonds, 89% of it on J2.
  - At the exact sign FN still removes many hops, and the FN error is entirely amplitude-ratio error on those bonds.
- (C) theta_4 at s0: bare 1.241e-5; augmented 4.147e-6 at c* 4.0. That is only 0.22e-6 below s_G (4.367e-6); the prediction
  1.5-3.0e-6 MISSED.
=> At this amplitude quality the sign is nearly irrelevant: even the exact sign gains 5%. The amplitude copy is the
bottleneck, which supports putting the effort into C (the amplitude class).
Result COST (2026-11-02; ws1 m42, one core, C kernel; pfaffian_cost/run/walker_cost_L{6,10}_time.json; verified):
| ratio to the plain one-PP walker step | 6x6 | 10x10 |
| one PP + one Jacobi level (J) | 4.78x | 5.50x |
| two PPs + one Jacobi level (D) | 9.33x | 10.91x |
Exactness: all ratios match direct determinants to 1.4e-12; broken versions fail. 6x6 J anchors T2 (4.78 vs 5.4). At 10x10,
D/P matches Astra's op model to 2%. VERDICT: PASSES (predicted 6-12x; kill > 20x). J inside its 3-6x prediction.
Convention (stated): T2's, where every image's inverse is recomputed per step. With persistent rank-2 inverse updates the
plain step gets much cheaper and both ratios rise, for ours AND for any one-Lanczos-step guide, which has the same d^2
two-swap class (board 23). The comparison with the competitor is convention-independent; the absolute "x plain" is not.
AMENDMENT before any fit (2026-11-02, from the staging CPU check (d)): the lambda direction is NOT new. Since
ln|B|(alpha f_1, lambda) = 18 ln|alpha| + ln|B|(f_1, lambda alpha^-18), the scale and phase of f_1 reproduce d/d lambda (cosine
with span(P_1, const) = 1.0; identity to 8e-13). The new directions come only from P_2's own block, weighted by
lambda P_2/(P_1 + lambda P_2): median |lambda P_2/P_1| = 0.095 at the warm start.
Rank of the tangent space on 12288 orbits: P_1 + const 2379; + lambda 2379; + P_2 4755.
The mechanism stands with this correction (the tangent space doubles). Predictions and kills are unchanged.
CPU checks: lambda = 0 reproduces the single PP to 3.6e-15; Jacobian vs FD 1.3e-9; symmetry 3.6e-15; every broken version
fails. Target control E_FN[T_5] rel 4.5e-11. Code shas: cx_r1fit.py 8f89a324, rbmpp_lib.py 97eade56; old arms
byte-identical. Cost gate passed (above).
RUNNING round 5 (two-PP + control, 2 seeds each): Paderborn copies (ids in pfaffian_r5) raced against theorie A40 copies 17659082-89 (A40 test-only ETA 2026-11-04; the H200s were all allocated); race watchers per arm; readout chained by r5_chain.sh.

## Result round 5, two-PP class (d) (2026-11-03; readout 17672082, pfaffian_r5/readout/readout.json; verified; 6.8 H100-h + 12.5 A40-h)
| arm (from theta_4 to T_5, s_G) | bare e | r_5 | aug min (c*) |
| two-PP seed 1 | 8.332e-6 | 0.493 | 3.258e-6 (4.0) |
| two-PP seed 2 | 8.347e-6 | 0.491 | 3.291e-6 (4.0) |
| control single PP seed 1 (A40) | 1.1827e-5 | 0.046 | 4.127e-6 (3.9) |
| control single PP seed 2 | 1.1814e-5 | 0.047 | 4.146e-6 (3.9) |
Control: e(theta_4) reproduced exactly. All fits ran 1500 steps; shas match between cluster and local.
VERDICT: KILL (d) not fired. ATTRIBUTABLE: r_5(two-PP) - r_5(control) = +0.447 / +0.444 (rule >= 0.03).
Predictions: two-PP r_5 0.06-0.16 MISSED HIGH (0.49); aug 3.4-4.1e-6 MISSED LOW (3.26-3.29e-6). Control r_5 0.01-0.04 and
aug 4.20-4.35e-6 just outside (0.046, 4.13-4.15e-6).
The term that was wrong: I sized the storage by the lambda direction plus a 0.1-weighted P_2 block. The P_2 block (tangent
rank 2379 -> 4755) stores half of the round's Jacobi gain in one round, about 10x the single PP at round 4.
Caveats:
- The two-PP class has about 2x the parameters, so "a second PP" and "more parameters" are not separated (no equal-capacity
  control exists).
- The two seeds share the warm start and target, so their spread (0.002 in r) measures the optimiser path only.
- The cost gate passed, measured: 10.9x plain at 10x10 against 5.5x for a single PP.
Against the competitor (5.574e-6, same sign): the augmented two-PP guide is 3.26-3.29e-6, i.e. 41% lower. Floor(s_G) <= 0.83e-6.

## Round 6 two-PP + parameter-matched control (lead GO 2026-11-03, takeover; <= 6 H100-h with A40 racers; pre-registered before any run)
Arm R6, two-PP round 6: from seed 1's theta_5 (P_1, P_2, lambda warm). Target T_6 = B_5 e^{4.0 f_J^G[B_5]}; control
E_FN[T_6] = 3.2584e-6. Seed 1, 1500 steps, mean of 7 checkpoints from 600 to 1500. r_6 = (8.3319e-6 - e(theta_6)) / 5.0735e-6.
Prediction: r_6 0.2-0.4 (bare 6.3-7.3e-6), aug 2.5-2.9e-6. KILL: r_6 <= 0.1 (the second PP saturates like the first).
Arm M5, parameter-matched single class at round 5: one PP plus an ACTIVE RBM bulk, as in the rbmpp kind with rbm_alpha = 72
(72 x 37 = 2664 real RBM parameters), init W, b ~ N(0, 0.01^2). That makes PP 2592 + RBM 2664 = 5256 parameters, against
5334 for two-PP. It starts from theta_4's PP with target T_5, the same as the round-5 arms, seed 1, 1500 steps + averaging.
The CPU checks must show nonzero RBM Jacobian columns, and that a sigma -> 0 init reproduces theta_4.
Prediction: r_5(M5) 0.05-0.15, aug 3.9-4.2e-6. Test N (RBM banks miss the round direction, eps^2 0.95-0.98) and Astra's
r ~ 0.08 for an active RBM both point the same way.
RULE: "structure matters" iff r_5(two-PP, 0.49) - r_5(M5) > 0.15, i.e. r_5(M5) < 0.34. Otherwise the round-5 gain is
attributed to parameter count. M5 is a capacity control, not a deployable guide: with m = 72 per image its walker cost
exceeds 20x by Astra's model.
AMENDMENT before any M5 run (2026-11-03): rbm_init_sigma 0.01 shifts ln|A| by rms 1.25 (576 images x 72 units; the
starting held-out loss 4.23 against 5.7e-3), so M5 would partly measure recovery from noise. Use sigma = 1e-3 instead (rms
0.0125, measured). Prediction and rule unchanged. Code: cx_r1fit.py f1374bed (rbm_init_sigma option; old arms
byte-identical). T_6 control rel 3.0e-11.
AMENDMENT 2 before any M5 run (2026-11-03): at sigma 1e-3 the RBM tangent block has rank 251 of 2664, against 2521 at 0.01
(tanh is nearly linear, so the start is Jastrow-like). M5 would then match two-PP in parameter count but not in tangent rank.
Selection rule, fixed now: the smallest sigma in {2e-3, 3e-3, 5e-3} with RBM rank >= 2000 AND step-0 held-out loss <= 1.14e-2.
If none qualifies, sigma 1e-3 with the label "rank-deficient start".
Sigma sweep (pfaffian_r6/cpu/sigma_sweep.txt): RBM rank 251 / 1248 / 2423 / 2521 at sigma 1e-3 / 2e-3 / 3e-3 / 5e-3, with
start held-out loss 6.0e-3 / 1.22e-2 / 3.96e-2 / 0.27. No sigma passed, so M5 runs at 1e-3 labelled "rank-deficient start".
Reading fixed now: if r_5(M5) < 0.34, "structure matters" holds only against a Jastrow-like start (weak). The follow-up would
then be M5 at sigma 3e-3 (rank 2423), ~2 H100-h, proposed to the lead after the readout.
RUNNING round 6: R6 Paderborn fit 4103501 + table 4103502 (won the race; A40 17673215/16 cancelled); M5 Paderborn 4103519
+ 4103520 raced against A40 17673270 + 17673271; readout chained by r6_chain.sh on m51.
LEAD DECISION (takeover, 2026-11-03): M5 at sigma 1e-3 is cancelled (4103519/4103520, 17673270/17673271), because a rank-251
control adds about a tenth of the directions the second PP adds (2376) and cannot separate structure from count. It is
replaced by M5 at sigma 3e-3 (RBM rank 2423), same spec and budget. r_5 is measured from M5's own start energy:
r_5(own) = (e_start - e(theta_5^M5)) / (e_start - 4.367e-6), with e_start from a table-only job on the initial theta.
The rule is unchanged: structure matters iff 0.4929 - r_5(own) > 0.15. My prediction for r_5(own): 0.10-0.25.
RUNNING M5 (sigma 3e-3): Paderborn fit 4103524 + table 4103525 (tab_t5 + tab_t0 for e_start), raced against A40 17673289 + 17673290; the cancelled sigma 1e-3 dirs are kept under pfaffian_r6/m5_s1em3_cancelled.

## Test W: where does the second PP store the round correction? (lead/PI 2026-11-03; CPU only; pre-registered before any number)
Hypothesis (the lead's, labelled as one): B = P_1 + lambda P_2 reshapes the amplitude where |P_1| is small, near the nodes,
where the Jacobi correction lives; one PP cannot (Θ).
Quantities, with Delta*(x) = ln T_5(x) - ln B_4(x) the target correction (c f_J) and Delta(x) = ln B_5(x) - ln B_4(x) the stored
one. Global constants are removed by the n|B_4|^2-weighted mean. Decades d = floor(log10 of the per-configuration |B_4|^2).
For two-PP s1 and the control ctl_s2, per decade:
- (i) the site-wise stored fraction kappa_d = <Delta Delta*>_d / <Delta* Delta*>_d, n-weighted;
- (ii) the same on blocked-bond differences (Delta_x - Delta_y) against (Delta*_x - Delta*_y), weighted by
  |H_xy| B_4(x) B_4(y) (board block 5), with x the smaller side;
- (iii) each decade's share of the total blocked-bond energy weight of Delta*.
(2) On a stratified sample (<= 20k orbits per decade, CPU jax), rho(x) = |lambda P_2(x) / P_1(x)| for theta_5 s1, and its
per-decade Spearman correlation with Delta(x).
Prediction (the lead's; I share it at ~60%): the two-PP's extra stored fraction (kappa(two-PP) - kappa(control), on (ii))
sits in decades < -9, with |difference| < 0.1 in decades >= -7; rho grows toward low decades (median rho higher at -11 than
at -6), and rho correlates positively with Delta in the low decades.
KILL of "near-node reshaping": the bond-weighted extra storage summed over decades >= -9 exceeds that over < -9. The
mechanism is then "more directions", and M5 should match.
Result Test W (2026-11-03; CPU 17681251; pfaffian_w/w_result.json; verified)
Per-decade bond-weighted stored fraction kappa (two-PP s1 / single-PP control s2 / decade's share of the Delta* bond weight):
-14: 0.46/0.03/0.11; -13: 0.50/0.04/0.17; -12: 0.45/0.05/0.19; -11: 0.36/0.06/0.18; -10: 0.30/0.05/0.15; -9: 0.26/0.03/0.09;
-8: 0.25/0.01/0.04; -7: 0.25/0.00/0.01; -6: 0.24/0.01/0.002.
Controls: Delta = Delta* gives kappa 1 everywhere; a permuted Delta* gives ~0 for decades <= -10 (noisy in the sparse decades
>= -9).
rho = |lambda P_2/P_1|: median 0.67 and flat from decade -6 to -11 (10-90% 0.57-0.81); it rises only below -14 (1.29 at -21),
where ~1% of the weight sits. Spearman(rho, Delta) is +0.08..+0.17 in decades -23..-9.
VERDICT: the KILL formally does not fire (0.31 < -9 vs 0.03 >= -9), but only because 85% of the weight lies below -9.
Per decade the extra storage is spread everywhere: 0.24-0.47 in every decade from -14 to -6, against the prediction of
< 0.1 for decades >= -7. rho is flat, not node-peaked.
So "near-node reshaping" is NOT supported. The second PP acts as a smooth multiplicative factor |1 + lambda P_2/P_1|
(~1 +- 0.67 with a varying phase) over the whole bulk, i.e. it adds directions everywhere. The parameter-matched M5 is
now the deciding control.

## f_J defect audit (lead 2026-11-03, from the sign angle's T-SPLIT; CPU; pre-registered before any number)
Defect: fJ_of sets non-finite rows (SK = 0: no kept bond; den <= 0) to 0, with at most 3 allowed. On tiny-|A| rows,
c f_J = c ln(A'/A) is unbounded for c > 1 (the guide is A^(1-c) A'^c, which grows like A^(1-c) as A -> 0).
Our quoted E_FN values are exact for the guides used, so they stay valid upper bounds; the risk is sampling at scale.
Principled value for no-kept rows (derived): the Jacobi update gives A'(x) = 0 there, so f_J = -infinity. In H_FN such a row
has no kept hop, so it forms its own component with eigenvalue H_xx + V_A(x) > E and the FN ground state vanishes on it.
-infinity is therefore exact, and a walker never enters x (guide ratio 0). No floor is needed in principle; a tabulated
solver needs only a finite stand-in (e.g. ln A - 69).
The unbounded positive side is the real hazard: c > 1 extrapolates beyond the one Jacobi step. A bounded alternative, if the
audit flags anything: cap c f_J at a quantile, or use c_eff(x) = min(c, f_max / f_J(x)).
Prediction: complex PPs have no exact zeros (P(|A| < eps) ~ eps^2), so for our guides the largest single-orbit share of the
augmented weight is < 1e-4, the 0.9999-capped E_FN differs by < 0.02e-6, and the no-kept rows are <= 3 per guide with
negligible weight. Flags at > 1e-3 share or > 0.05e-6.
FIX (PI go via the lead, 2026-11-03; pre-registered before any number):
(a) No-kept rows: (a1) a node: amplitude 0, finite stand-in lF + c f_J -> lF - 69; (a2) a floor: f_J = f_min, the 1e-4
    quantile of the finite f_J.
(b) Per-row relaxation instead of global over-relaxation: (b1) c_eff(x) = c min(1, f0/|f_J(x)|); (b2) a smooth cap,
    c f0 tanh(f_J/f0). Here f0 is the n-weighted 0.9999 quantile of |f_J|. Rationale: c > 1 is Jacobi over-relaxation
    (JOR), stable only for small spectral steps, and a per-row limiter is the usual trust-region remedy (Saad, Iterative
    Methods for Sparse Linear Systems, 2003, ch. 4, relaxation; cited from memory, not re-read).
For each fixed variant, c is re-scanned around c* (3, 3.5, 4, 4.5, 5).
Predictions (the lead's and mine): on every quoted guide the fix moves the augmented minimum by < 0.05e-6, and the two-PP
gain (control minus two-PP, round 5) changes by < 5%.
KILL of "refinement": any fixed guide moves by > 0.2e-6, or the two-PP gain shrinks by > 20%; then we step back.
Code: ansatz/code/fj_fix.py (shared fJ_of_fixed + a single-orbit weight check that asserts max share < 1e-3) goes into the
readouts and to the sign angle.

## B design: the label-free loop at 8x8, with no exact tables (paper only, 2026-11-03; for Astra's blind review; no runs until the f_J fix, R6 and M5 read out)
Anchors (8x8, E per site; fact sheet from the records):
- Production RBM+PP: rbmpp8/rb8_prod, f 64x64 real, complex RBM alpha 0.25, translations only, 6176 parameters,
  3.8 H100-h; VMC -0.498671(3).
- FN p=0 with that guide: -0.4987565(28), 0.40 H100-h for 2 chains.
- Network + one signed Lanczos step + FN (M1 p=1): -0.498946(5), 7.9 H100-h.
- Varbench RBM+PP with full projection (2302.04919): -0.498963(2).
- E0 is not known: Hu extrapolation -0.49906(1); ours -0.498992(16).
- So we grade absolute energies against p=1 and Varbench, and quote no "error vs E0".
6x6 law carried over (measured):
- c* = 3.9-4.0 for every guide and sign.
- Augmented/bare ratio 0.36-0.41 at round 0-1.
- One Jacobi level on the network's own copy is about equal to one Lanczos step: round 1 under s_G 6.07e-6 vs 5.57e-6.
- Each label-free round then gains. The two-PP round stored r = 0.49 against 0.05 for a single PP (P22).
- The amplitude carries >= 81% of the guide's error at s_G (P21).
(1) BASE. B_0 = P_1 + lambda P_2.
- P_1 = the production PP with the RBM (warm, rb8_prod).
- P_2 = a second PP warm-started from P_1 plus noise at finite separation, or from a second production seed if one exists.
  The 6x6 lesson: P_2 must not sit in P_1's tangent neighbourhood (Astra, P16); lambda is gauge-redundant (Amendment, round 5).
- Guide G = s_G(x) B(x) e^{c f_J[B](x)}, with c fixed at 4.0 by the 6x6 law and the FIXED f_J (no-kept rows set to a node;
  per-row relaxation cap f0; fj_fix.py). The single-orbit weight check is built in.
(2) LABEL-FREE FIT FROM SAMPLES.
- Round target T_k = G_k (B_k e^{4 f_J[B_k]}); the labels ln T(x) are deterministic given B_k, s_G and E'. Only the coverage
  of configuration space is sampled.
- Pool: S samples x plus K = 2 blocked partners y each, from a tempered ladder |B|^(2 beta), beta in {1, 1/2, 1/4}.
  The pool is reweighted to band-uniform weights over |B|^2 decades, as the 6x6 fits used. Decades are defined relative
  to the median of ln|B|^2 under beta = 1, because the norm is unknown.
- Coverage need, from Test W: decades below -9 carry 85% of the round's bond weight at 6x6 and get 0.25-0.5 stored per
  decade. The sampler must put >= 4e4 samples into each of the ~9 occupied decades, so S ~ 4e5.
- Generalisation noise (derived): for a least-squares fit with P parameters and S samples, the held-out excess is about
  P/S of the residual. Two-PP + RBM has P ~ 1.8e4 real, so P/S ~ 0.045 at S = 4e5: r shrinks by ~5%, small against 6x6's
  r = 0.49 per round.
- The iterate-averaging stop (P19) handles the optimiser noise as at 6x6.
- Label cost (derived from measured anchors): f_J(x) needs one walker-step's ratios. With the lemma this is 5.5x one
  evaluation (measured, 10x10 CPU kernel).
  - 4e5 x 3 configurations x (5.5 lemma evaluations, plus d + 1 = 129 sign-CNN calls) gives 1.5e8 CNN calls at 3.3e-5 s:
    ~1.4 GPU-h.
  - The lemma part runs on the CPU C kernel (walker_cost.py extended to 8x8): ~170 core-h, free.
- Fit: two-PP damped GN on the pool, 1500 steps; ~6 s/step on an H100 (2x the 3.04 s RBM+PP SR step) gives ~2.5 H100-h
  per round.
(3) SIGN. s_G = sign of (1 + alpha(H - E)) psi_RBM+PP: one signed Lanczos step on the production guide, the same sign as M1 p=1.
- The d^3 issue: computed on the fly, f_J(y) at every walker neighbour y needs s_G(z) at y's neighbours z, and each s_G(z)
  needs E_L(z), i.e. B at w ~ z. That is three swaps from the walker, d^3 ~ 128^3 = 2e6 evaluations per walker step at 8x8.
  Impossible, so s_G must be STORED.
- Store it in a CNN trained on s_G labels from tempered samples. At 6x6 the XL net on s_G gave w_T 1.07e-6 and a FN tax
  of 4.25e-7, all its errors in the low-amplitude tail. There is no 8x8 XL; the 8x8 L-size ground-state net reaches
  0.9-1.2% configuration error.
- Cost: labels from 2e5 samples x 129 evaluations, about 1 H100-h; training an XL at 8x8 about 1 H100-h.
- Acceptance: the |B|^2-weighted wrong fraction on held-out samples must be <= 1e-4 (estimated with s_G computed directly
  on those samples).
- Then the walker cost is one-level: d^2 lemma ratios plus d^2 CNN calls per step.
(4) GRADING. FN-GFMC with gfmc_guide.py, extended by a new guide class: augmented two-PP plus stored sign plus fixed f_J,
  with the lemma evaluator.
- Today gfmc_guide.py evaluates full configurations with no fast updates, which is the 128x-per-hop class of p=1.
- 2 chains; error target 1e-5 per point first (about 1/4 of p=1's cost, so ~2-6 H100-h per point, derived), tightened to
  5e-6 only for points within 3 sigma of a bar.
- Equivalence test before use: at c = 0 with the network's own sign, the new class must reproduce the p=0 value
  -0.4987565(28) within 2 sigma.
(5) ROUNDS, PREDICTIONS, KILLS, COST.
- V (the gate, at 6x6 first). Run the whole SAMPLED pipeline at 6x6, where the tables exist: pool, CNN sign, fit, then
  exact grading. Start from theta_4 with two-PP, as round 5 did.
  - Prediction: r_sampled >= 0.35 against the exact-table r = 0.49.
  - KILL of "the loop survives sampling": r_sampled < 0.25.
  - Cost ~3 H100-h plus CPU.
- R0 at 8x8. Augmented production guide (single PP, no refit), stored s_G, c = 4.
  - Prediction: E = -0.49894(1), i.e. p=1 within 1.5e-5, since one Jacobi level ~ one Lanczos step at 6x6.
  - KILL (machinery at scale): E above -0.49892, worse than p=1 by > 2.6e-5. Then stop and diagnose the CNN sign error and
    f_J at scale.
  - Cost: CNN ~2 H100-h, grading ~4 H100-h.
- R1 at 8x8. Two-PP label-free fit from the R0 guide's target.
  - Predictions: bare FN p=0 of B_1 -0.49880..-0.49885 (r 0.2-0.5 of g ~ 1.9e-4); augmented -0.49896..-0.49900.
  - Bars: beat p=1 by > 3 sigma; beat Varbench -0.498963(2) by > 3 sigma (the headline, at ~40% by my estimate).
  - KILL (the loop at scale): the bare FN drop is < 1e-5 (r < 0.05).
  - Cost: labels ~1.5 GPU-h + 170 core-h; fit ~2.5 H100-h; grading of bare + augmented ~6-10 H100-h.
- Total to the R1 verdict: ~20-25 H100-h plus ~3 A40-h, plus build time.
Build items (unknowns, with times; one worker each):
- (i) a batched JAX lemma evaluator for the two-PP at 8x8, 1-2 days; unknown: GPU efficiency of rank-4 dets over d^2
  endpoints;
- (ii) the s_G sign CNN at 8x8, 0.5-1 day;
- (iii) the sampled-pool fit driver, a cx_r1fit variant reading a pool instead of tables, 1-2 days;
- (iv) the gfmc_guide.py augmented guide class plus its equivalence test, 1 day.
- V needs (ii) and (iii) at 6x6 only; R0 needs (i), (ii) and (iv).
Scope line: passed over
- backflow (cost ~800x, P-proposal);
- a second explicit Jacobi level (650x);
- on-the-fly s_G (d^3).
B BUILD started (2026-11-03, lead/PI: B in parallel): (i) pp2_lemma.py, (ii) sign8.py, (iii) pool_fit.py + the staged V launcher, (iv) gfmc_aug.py; workers, code + CPU checks only. Next: V at 6x6 and 8x8 R0 in parallel after Astra's review (cj-c9d47cfcaf6d) and the f_J fix; 8x8 R1 only once V passes.

## B design revised after Astra's review (ansatz/REVIEW_astra_Bdesign_2026-11-03.md, cj-c9d47cfcaf6d; lead decision 2026-11-03)
Astra puts R1 at ~15%. Two points I accept:
- "P/S small -> ~5% loss" is unsupported: in Θ v2 a fit with P/S 0.005 scored 0.447 on its training pairs and -0.477
  held out.
- At S = 4e5 the deep decade -14 gets 19k of the 40k samples it needs, and it carries 11% of the bond weight.
Astra also measured that round 5 turned 0.288 of its bare gain into augmented gain, which puts R1 at -0.498951..-0.498967
(my -0.49896..-0.49900 was optimistic).
V (narrowed; runs after the f_J fix):
- One finite-pool replay of round 5: theta_4 -> T_5 (fixed-f_J target), two-PP, exact s_G, exact FN grading.
- Pool: S = 4e5 x-draws, i.i.d. from |B_4|^(2 beta) with beta in {1, 1/2, 1/4} in equal parts (exact sampling from
  the table, so only the coverage is tested). Each x gets K = 2 blocked partners with P(y|x) proportional to
  |H_xy| |B_4(y)/B_4(x)| over x's blocked y. Weights are reweighted to band-uniform decades.
- Fit as round 5 (1500 steps, tail average). About 3 H100-h plus CPU.
- PREDICTION (updated after Astra): r_sampled 0.15-0.35, central 0.25. KILL: r_sampled < 0.25. My odds of passing are
  ~50%.
- A second independent pool runs only if V is promising (r >= 0.25).
8x8 build now: only the full-guide benchmark (two-PP + Jacobi with the rank-4 lemma + sign CNN, realistic batch,
symmetry projection, f64/f32, memory), <= 1 GPU-h. Items (ii) labels/training and (iv) GFMC class are paused until V.
Controls added before any 8x8 run (Astra):
- (c1) an ENERGY-based sign-storage gate at 8x8: FN-GFMC of the production guide with the stored CNN sign against the
  directly computed s_G, on the same amplitude, at c = 0. Pass if they differ by < 1e-5 per site.
- (c2) E' and the cap f0 determined from samples (E' from p=0 GFMC; f0 as the sampled 0.9999 quantile of |f_J|),
  with the 6x6 sampled-vs-exact f0 check.
- (c3) an exact c > 0 validation of the 8x8 evaluator code at 6x6: the augmented GFMC energy against the exact table
  value (theta_4, 4.367e-6).
- (c4) independent pools and fits (two seeds of the pool, not only of the optimiser).
- (c5) same-sign bare/augmented comparisons: every bare and augmented 8x8 number under the same stored sign. p=0 with the
  network's own sign is quoted separately.
Note for the paused GFMC build (iv): gfmc_guide.py's guide is Re B with sign sgn Re B, but our 6x6 guides are s |B|. Equivalence at c = 0 therefore needs |Re B|. Decision deferred to the restart: grade the 8x8 guides as s_G |B| e^{c f_J} and state the convention.
Correction to "the d^3 issue" (2026-11-03, from the sign8 builder; not yet measured in the guide): the production 8x8 code has
a fast-update E_L (eloc_full, 0.09-0.13 ms per configuration on an H100, gfmc_m13b/prod/rb8/V/timing8.json). On-the-fly
s_G(z) therefore costs one fast E_L per endpoint, i.e. d^2 E_L calls per walker step: the same count as d^2 stored-CNN
calls, not d^3 full evaluations. Storing the sign may be unnecessary. The 8x8 benchmark times both options.
Also: s_G labels for CNN training cost ~26 s per 2e5 samples, not ~1 H100-h. M1 p=1 constants (consts.json): alpha
-0.17543, E_total -31.9151, w = -1. sign8.py (XL/L CNN, sign_fn, save/load) is built; its label/training code is paused.
Result 8x8 guide benchmark (2026-11-04; A40 f64, run 17693959; pp2_lemma_bench/a40_17693959/bench.json; verified):
- Evaluator exactness: pp2_lemma.py is exact against direct determinants to 2.8e-12 (one swap) and 2.3e-12 (two swaps);
  1.9e-9 on 3 of 1.1e5 two-swap ratios of theta_5, bounded by eps x cancellation x cond(M), with cond up to 1e6 at 8x8.
  It reproduces the pp2 table to 1.9e-13, the 8x8 production RBM+PP to 6e-12, and fJ_of to 1.6e-14. Every broken version
  fails.
- Full augmented two-PP guide with the XL sign CNN, 8x8, M = 1024: 0.378 s per walker step, made up of 0.0020 plain,
  0.114 lemma and 0.261 CNN (28.8M CNN calls). 26.9 hops per walker-tau gives 10.2 s/walker-tau on the A40, against p=1's
  0.514 s/walker-tau on an H100. The H100 figure for our guide is NOT measured: the A40 is weak in f64 (the lemma part),
  while the CNN part is fp32.
- f32 is unusable (r2 error 9e10), because the inverse has cond up to 1e6.
- Sign on the fly from E_L at the endpoints: 2.05e-3 s per endpoint (A40 f64), 223x the CNN, so the STORED sign stays.
- Savings not built yet: deduplicated endpoints (~1.9x) and less padding (~1.3x); an L-size CNN (~4x fewer parameters).
- J/P here = 185, because the plain step uses persistent fast updates (0.002 s). This is the convention caveat recorded with
  the CPU cost gate.
Consequence for B: 8x8 grading at a 1e-5 error target was budgeted at 2-6 H100-h per point. If the H100 is ~3-5x faster
than the A40 for this guide, it is ~2-6x p=1's cost per walker-tau, i.e. ~4-15 H100-h per point at 1e-5 before the
savings. One short H100 benchmark (~0.2 H100-h) settles it.
B design amendment (lead, 2026-11-04):
- Grading must reach ~3e-6 per point, not 1e-5. Astra's R1 range spans 1.6e-5 and contains Varbench. That is ~11x the
  1e-5 cost: ~45-165 H100-h per graded point before savings.
- The savings (endpoint dedup ~1.9x, padding ~1.3x, an L-size CNN ~4x) become a PRECONDITION for any 8x8 grading, not an
  option.
- The H100 benchmark is held until V reads out.
Result f_J audit + fix (2026-11-04; CPU 17688846 and 17690117; pfaffian_fjaudit/{fj_audit.json, fix/fj_fix.json}; verified):
- No defect on any quoted guide. The control reproduces every augmented minimum to <= 1.7e-10.
- No-kept rows: one on L-SWO r1 only (n = 12, base weight 1.5e-19). Setting it to a node changes e by 6e-16.
- Max single-orbit share of the augmented weight: 0.0093 on every guide. That equals the base guide's own share
  (0.0094; one high-multiplicity bulk orbit, n = 144, c f_J ~ 0), so it is not an f_J artifact.
- c* f_J: max 5.0-6.8; n-weighted q.9999 2.3-2.6.
- Capping at q.9999 changes E_FN by <= 1.5e-10.
- Fix variants: a1, a2, b1 and b2 (per-row cap) move every minimum by <= 1.2e-8, and the minimum stays at c = 4.0.
- Two-PP gain changes by +0.07% (b2) and ~0 (b1).
- Predictions (< 0.05e-6 move; < 5% gain change) HIT. The KILL of "refinement" did not fire.
- Our 6x6 guides never had the sign angle's blow-up: complex PPs have no near-zeros with large f_J.
Kept for scale: fj_fix.py mode b1 (no-kept -> node, c_eff = c min(1, f0/|f_J|)). Its weight check must be RELATIVE: raise
if the augmented max share exceeds 10x the base guide's max share (the absolute 1e-3 fires on the base guide's own bulk
orbit at 6x6).
V staged and submitted (2026-11-04).
- Pool: 4e5 i.i.d. draws over orbits from n|B_4|^(2 beta) (1/3 each for beta = 1, 1/2, 1/4), with K = 2 blocked partners
  each, P(y|x) proportional to |H_xy||B_4(y)/B_4(x)| (importance ratio kept). Band-uniform weights per decade.
- Realised counts in the 9 decades: 5k-57k; ESS 2.2k-39k. Decade -6 is short of the 4e4 target (30k), and so are the top
  two decades, which have little configuration mass.
- Checks: (a) pool = table reproduces cx_r1fit's loss, gradient and GN step to 4e-14; (b) the pool statistics match the
  exact multinomial (max |z| 2.2); (c) it fits from a perturbed start. Each check has a broken version, and it fails.
- Target T_5^b1 (fixed f_J; 1650 rows capped). Its E_FN differs from the old T_5 by 8e-12.
  Confound, stated: the exact-table r_5 = 0.493 was fitted to the OLD T_5. The kill and pass thresholds are absolute, so
  they are unaffected.
- Found: cx_r1fit's acceptance test let a NEGATED first step through as "noise", while the held-out loss rose. This
  weakness is in the table fit as well. The tail averaging masks it, but it is recorded.
- Run: Paderborn, with an A40 racer (Paderborn ETA more than 1 day). Readout: v_bare.py; PASS r_V >= 0.35, KILL r_V < 0.25.

## Queue (after V): strict step acceptance (lead 2026-11-04; paper only until V reads out)
Hypothesis: cx_r1fit's acceptance test admits bad steps as "noise". Each admitted step is a random kick along directions
where the copy loss is flat but E_FN is not. Their sum is the E_FN walk (0.1-0.2e-6 per 150 steps, R3-cont arm B) and
could feed the round-3 damping collapse.
Test: rerun arm B's segment, theta(1200) + 900 steps (seed 1, lambda floor 7.8e-3, checkpoints every 150), under a strict rule:
accept a step only if the held-out loss does not rise by more than its own noise, the standard error of L_ho from
the batch scatter.
Predictions: the walk amplitude (rms of the checkpoint-to-checkpoint E_FN change) falls from 0.13e-6 to <= 0.05e-6;
the averaging bonus (best checkpoint e minus e(theta_bar)) shrinks from 0.35e-6 to <= 0.1e-6; the final bare e is within
0.1e-6 of arm B's average, 1.252e-5.
Kill: walk amplitude >= 0.1e-6 under the strict rule. The walk then has another source, e.g. sampling noise in the batch.
Cost ~0.8 H100-h + tables + CPU.
RUNNING V: Paderborn fit 4107178 + tab 4107179, raced against A40 17695322 + 17695323; readout chained by v_chain.sh.

## Result round 6 + M5 (2026-11-04; readout 17698157, pfaffian_r6/readout/readout.json; verified; ~4.2 H100-h)
| arm | bare e | r | aug min (c*) |
| R6 two-PP round 6 (from theta_5 s1) | 7.061e-6 | r_6 0.2505 | 2.814e-6 (3.9) |
| M5 PP + active RBM, 5256 params, round 5 from theta_4 | 1.1134e-5 | r_5 vs theta_4 0.134; r_5_own 0.954 | 3.986e-6 (3.9) |
| (two-PP round 5, for comparison) | 8.332e-6 | 0.493 | 3.258e-6 |
Controls: theta_4 rel 0; theta_5 rel 4.7e-11.
R6 predictions HIT (r 0.2-0.4; aug 2.5-2.9e-6); no KILL. Per-round storage q = 0.25/0.49 = 0.51.
Augmented guide after round 6: 2.81e-6, i.e. 50% of the competitor's 5.574e-6. Floor(s_G) <= 0.83e-6.
M5 against the pre-registered rule (verdict AS REGISTERED: PARAMETER COUNT; see M5 RECORD below): the gate on r_5_own reads PARAMETER COUNT (0.4929 - 0.954 < 0.15). But e_start =
1.51e-4, 12x theta_4's, because the sigma 3e-3 RBM init scrambled the guide. r_5_own's denominator is therefore ~97% the
self-inflicted init gap, and the metric measures noise removal, not round storage. The rule was ill-conditioned for this start;
I do not read it as evidence either way.
What the arm does show, at equal target, steps and parameter count:
- M5 ends at bare 1.113e-5 and aug 3.99e-6, against two-PP 8.33e-6 and 3.26e-6 (0.73e-6 worse augmented).
- Against theta_4 it stored 0.134, against 0.49.
- So the two-PP reaches lower energies from the same budget, but a fair "structure vs count" verdict needs a control
  that starts AT theta_4's energy.
Proposed, cost ~1.6 H100-h:
- M5' = the same PP + RBM, first relaxed to the c = 0 target (B_4 itself) until e returns to <= 1.25e-5, then the round-5
  fit to T_5. Rule: structure matters iff r_5(two-PP) - r_5(M5', measured from its relaxed start) > 0.15.
M5 RECORD (lead ruling 2026-11-04, no re-ruling after the data): the registered verdict is "PARAMETER COUNT" (r_5_own
0.954), recorded AS REGISTERED. Reason the metric was broken: the init scrambled the start 12x (e_start 1.51e-4), so the
denominator is dominated by noise removal. Non-registered observation beside it: at equal budget the two-PP is better by
0.73e-6 augmented (3.26e-6 vs 3.99e-6).

## M5' (new pre-registered test, lead GO 2026-11-04; ~2 H100-h; before any run)
Arm: the M5 class (PP + RBM alpha 72, 5256 parameters, rbm_init_sigma 3e-3, seed 72), in two phases.
- RELAX: fit to the c = 0 target (theta_4's own table) for 1000 steps, then build a table and solve exactly. Proceed only
  if e_rel <= 1.25e-5; otherwise relax 1000 more steps (at most twice).
- ROUND: from the relaxed theta, fit to T_5 (fixed-f_J version, as V) for 1500 steps with tail averaging, as round 5.
Readouts:
- (1) r_5(M5') = (e_rel - e_end) / (e_rel - 4.367e-6). Structure matters iff 0.4929 - r_5(M5') > 0.15.
- (2) Start-independent: structure matters iff aug_min(M5') - 3.258e-6 > 0.3e-6.
Predictions: e_rel 1.21-1.25e-5; r_5(M5') 0.10-0.20; aug 3.8-4.1e-6; both readouts give "structure matters" (~70%).
If they disagree, I report both as registered and claim nothing.
RACE RULE (lead/PI 2026-11-04, replaces "the first copy to log a step wins"): the winner is the copy with the earlier
EXPECTED FINISH, i.e. now + remaining steps x measured s/step (+ table time), decided once both copies have logged steps.
The old rule cancelled V's H100 copy, ~9x faster, because the A40 logged its first step earlier. New script:
race_watch_eta.sh, for future launches.
V: a fresh Paderborn copy goes in alongside the running A40 copy 17695322/17695323, with the same pool (9146690d) and the
same pool_fit (66799f1c); new run dirs. Whichever copy finishes the readout first counts, and the other is NOT cancelled
without telling the lead. Expected ~2.4 H100-h.
V copies: Paderborn pdb2 fit 4109778 + tab 4109779 (squeue start est. 11-07); A40 17695322/23 at step 760, 53.3 s/step, finish est. 11-04 ~20:30 + 80 min table. v_chain2.sh takes the first completed copy and cancels nothing that is still running. m51 rebooted at 08:30 and killed the VPN and its watchers; re-armed.

## Result V gate (2026-11-04/05; Paderborn pdb2 fit 4109778 (express) won; readout pfaffian_v/readout/readout.json; verified)
Finite pool of 4e5 i.i.d. tempered draws, two-PP from theta_4 to T_5^b1, exact s_G and exact grading: bare e(theta_V) =
8.561e-6, r_V = 0.4636. The exact-table fit gave r_5 = 0.4929 (e 8.332e-6), a ratio of 0.94.
PASS (>= 0.35). Prediction 0.15-0.35 MISSED HIGH: Astra's P/S and coverage concerns did not materialise at S = 4e5 for this
class and round.
Augmented grading: CPU readout running (R5's c scan, old and b1 f_J), against the exact-table 3.258e-6.
Next: 8x8 work waits for the PI's go with preconditions c1-c5 and the savings (dedup, padding, smaller CNN). The A40
duplicate 17695322 keeps running until the PI decides.

## STATE AT MIGRATION (2026-11-04 ~13:30; angle session cleared)
Running, no VERDICT yet (re-arm waits):
- M5' ROUND (pfaffian_m5p; ANALYSIS_BRIEF.md has the gates): A40 fit 17720568 + tab 17720569 RUNNING, raced against
  Paderborn 4110546 + 4110547 PENDING. Relax rel1 is accepted: e_rel = 1.21843e-5 (relax_accepted.json, theta_rel sha
  bff7cdc4), i.e. back to theta_4's 1.21845e-5. After the tables: CPU readout m5p_read.py, then m5p_gate.py with both
  registered readouts. Its worker dies with this clear, so the next lead runs the gate if the chain does not.
- V A40 duplicate 17695322/17695323: kept running until the PI decides on cancelling it; V already PASSED via pdb2.
- V readout job 17721219 (pf_vrd) still running (bare done); the theta_V AUGMENTED grading (R5's c scan, old + b1 f_J,
  control = pp2_s1 3.2584e-6 at c* 4.0) was delegated just before the clear; check
  pfaffian_v/readout/aug/ (CPU job 17722344, watcher armed; output readout_aug.json).
Waiting on the PI: 8x8 go (B design, revised; c1-c5 and savings are preconditions; grading target 3e-6 per point,
~45-165 H100-h before savings); cancel of the A40 V duplicate.
Queued after these: the strict step-acceptance test (section "Queue (after V)").

## 8x8 BUILD, route 2 (PI go via the lead 2026-11-04 ~14:00; CPU and code only; NO GPU until Astra re-reviews after 11-05 09:55)
Basis: blind Opus review reviews/2026-11-04_v_go_review_opus.md (V PASS confirmed; storage holds on configurations never in the pool).
Control changes (lead): c1 tolerance tightened from 1e-5 to <= 3e-6 per site. Necessary before any 8x8 grading: c1, c3, c5; c4 for the
"0.94" ratio; c2 nice-to-have. Grading convention fixed: guide = s |B| e^{c f_J} (not gfmc_guide's Re B); the 8x8 c = 0 equivalence
against p=0 -0.4987565(28) needs |Re B| (amp mode 're') and is quoted as a convention check only.
Pieces (one worker each, disjoint files; each "done" is a test that can fail, with a broken version that fails):
- (1a) pool8.py: 8x8 tempered MCMC pool sampler of |B|^(2 beta) (exchange moves, Sz = 0, PT across beta rungs), output thinned to
  ~independent draws for pool_fit's draw_fn hook. DONE = at 6x6 on B_4 the sampled per-decade occupation and <ln|B|^2> per beta match the
  exact orbit multinomial (tau-corrected |z| <= 3, chi2/dof <= 2), and a broken acceptance fails it; at 8x8 on the production guide:
  tau_int per beta, decade occupancy, CPU s/sweep, and the S (and core-h) needed for ESS/P >= 3 in every fitted decade (review: S ~ 1.5-2e6).
- (1b) pool_fit.py 8x8 model: the two-PP (+ RBM) at L = 8 (P_1 warm from production rb8_prod, P_2 complex) and the 8x8 labels_fn
  (ln T via pp2_lemma.fJ_fixed, mode 'fix'). DONE = logA of the fit model equals pp2_lemma.logB on the same parameters to <= 1e-10;
  gradient vs finite differences <= 1e-6 rel; a GN step on a small synthetic 8x8 pool lowers the loss; labels_fn equals a direct
  slow-path f_J on >= 50 configurations to <= 1e-9; each with a broken version that fails.
- (2) gfmc_aug.py: the AugGuide class G = s |B| e^{c f_J} (fixed f_J, b1 cap, relative weight check) on the pp2_lemma evaluator, plus c3:
  6x6 FN-GFMC of theta_4 with exact s_G at c = 4 against the exact table value 4.367085e-6 (capped), and at c = 0 against 1.218448e-5.
  DONE = both within 2 sigma with sigma <= 5e-7 per site; broken versions (f_J sign flipped; s_G replaced by the network's own sign)
  miss by > 5 sigma, predicted on paper first.
- (3) savings in pp2_lemma.py: endpoint dedup and less padding (bit-for-bit or <= 1e-12 against the current path on 8x8 production and
  theta_5 configurations), the L-size CNN wired as the guide's sign (sign8 'L'); CPU timing ratios. GPU timing waits for Astra.
  The L-CNN's ACCURACY is the sign angle's T-LCNN precheck (stop if R6 tax > 1.5e-6), not this piece.
Also (lead): A40 V duplicate 17695322 matches pdb2 to step 1010; lead proposed cancel + reuse the slot for a new-pool replica (c4).
Do nothing on it until the lead confirms.
M5' note (2026-11-04 14:xx, CPU check on theta_rel.npy): after the relax the RBM block is still its random init (rms 3.0e-3; moved
1.06e-4 rms, corr 0.9994 with the seed-72 init; W full rank 36). The PP alone restored theta_4's energy (the c = 0 target needs no
RBM). M5' therefore starts with live but untrained RBM capacity, which is what the control intends. At readout, also record the RBM
block's movement over the round (rms change vs PP's) so that "parameter count" is not read off an RBM that never moved.
M5' VALIDITY GATE (lead 2026-11-04, pre-registered before reading any round checkpoint): M5' is a VALID parameter-count control only if,
by step 300 of the round (or the nearest saved theta), the RBM block's rms change from theta_rel is >= 1.06e-3 (10x the relax's 1.06e-4)
OR the RBM's share of the update norm |dtheta_RBM|^2/|dtheta|^2 is >= its parameter share 2664/5256 = 0.507. Otherwise INVALID (an
effective one-PP run; suspected cause: the small-init RBM gradient, as in the zero-init RBM finding of 2026-10-30) and the lead is told before it runs on.
M5' validity, analog evidence (2026-11-04, CPU, before the gate's own checkpoint): the earlier M5 arm (same class, same seed-72 init, same fitter, T_5,
1350 steps) has its RBM rms change from init 4.13e-4 at step 600 and 4.12-4.14e-4 through 1350 (frozen); PP moved 7.4e-3 rms; RBM share of the
squared update ~0.3% against 0.507. Mechanism: dR/dW = sum_g tanh(b + W.z_g) z_g ~ O(sigma sqrt N) ~ 2e-2, so the RBM columns are ~1e-2..1e-1 of the PP's
and the relative damping s^2/(s^2 + lam) suppresses them. Expected verdict: INVALID. Paderborn twin 4110546 on hold; A40 runs to step 600 (~17:35),
where the registered gate is read. Proposed M5'': larger RBM init (sigma ~0.05-0.1) + relax, RBM column norms matched to the two-PP arm's P_2 block
by a CPU check before submission; the fitter unchanged.
c3 amended (2026-11-04, after the gfmc_aug worker's cost probe): GFMC driven by pp2_lemma.aug_ratios on CPU costs 48 core-s per walker-tau at 6x6,
so sigma 5e-7 would need ~1.2e5 core-h. Already shown: AugGuide rows = exact table rows to 1.2e-14 at c = 0, 4, -4 (a 1%-flipped sign misses by 0.49);
exact capped target in 'fix' mode 4.367227e-6 (job 17725880, gfmc_aug/pred_exact). c3 is now two parts: (i) FN-GFMC with the exact table guides
(c = 4, c = 0), 2 chains, sigma <= 5e-7, within 2 sigma of 4.367227e-6 / 1.218448e-5, with c = -4 (3.79734e-5) as the broken version, plus one dtau/M
bias check; (ii) same-seed trajectory identity, AugGuide vs the table provider, ~200 walker-tau at c = 4, <= 1e-10 rel (broken: c = 3.9). The two
parts together give "GFMC with the evaluator = exact FN".
Result V augmented (2026-11-04; CPU 17722344, pfaffian_v/readout/aug/readout_aug.json; control pp2_s1 reproduced to 9e-11; verified): theta_V aug min
3.3462e-6 at c* 4.0 (f_J old and b1 identical to 2e-14), +0.088e-6 above the exact-table round 5 (3.2584e-6). Augmented gain from theta_4 (4.367e-6):
1.021e-6 vs 1.109e-6, ratio 0.92 (bare 0.94). The sampled pool costs ~8% of the round's gain, augmented or bare. Deterministic exact readout (no error bar).
Result (3) savings (2026-11-04; CPU, pfaffian_8x8build/savings/; code 7c4dddf, default path 'old' bit-equal to the frozen module; verified check_*.json):
- dedup vs old: 8x8 production rel 3.6e-15, two-PP rel 3.6e-15 (abs 6.8e-12 at |ratio| ~3.6e3; the registered abs 1e-12 was ill-posed for ratios up
  to 6.6e5, so READ AS REGISTERED: abs MISSED, rel 1e-13 PASS), 6x6 theta_5 exact 0.0, L-CNN 0 sign flips; every broken key fails.
- CPU speed (job 17730108): padding 1.2x, endpoint dedup 2.5x on the lemma; CNN calls 2.7x fewer (28.2k -> 10.1k per walker step). Whole step
  with the L-CNN 2.7x. Estimated savings were 1.9x / 1.3x / -; measured are ~1.3x better on dedup, as predicted on padding. GPU per-move cost unmeasured.
Result (1b) pool_fit 8x8 model (2026-11-04; CPU jobs 17729764-69, 17732278; pfaffian_8x8build/poolfit8/; verified check flags; committed 389be08 after the a,d recheck on 7c4dddf passed under old and dedup paths, job 17734779):
- arm 'pp8': complex f_1, f_2 (64x64), complex RBM M = 16, complex lam; 18466 real parameters; P_1 warm from rb8_prod params_final.npz (sha 0520c5be).
- (a) logA = pp2_lemma.logB to 1.1e-14 (and = rbmpp_lib production to 8.1e-12 at lam = 0); (b) gradient vs FD 2.4e-9..8.1e-8; (c) GN step lowers
  held-out 3.137 -> 2.892 and train 0.1505 -> 0.1170; (d) labels_fn = slow path to 1.1e-14 (37 of 60 capped). Each broken version fails
  (cut-lam on the loss gradient only by 2.2e-3). 6x6 checks a, b, c unchanged.
- Open: (c2) with REAL 8x8 labels at c = 4 one GN step lowers train 1.6% but raises held-out 0.4%, inside heavy-tail noise (per-entry loss mean 7,
  max 350-430 from single importance weights). The partner importance weights rho will need a tail check at real pool sizes (pool8).
- Cost: CPU GN step at B = 8208 ~11 min (derived) -> 1500 steps infeasible on CPU; H100 estimate 3-12 s/step (derived, unmeasured). Pool labels
  ~50 core-h per 4e5 draws (extrapolated).
Result M5' validity gate (2026-11-04 ~17:45; A40 17720568, round/fit_th/round_1/theta_step600.npy sha 8fc0b310, nearest saved theta to step 300; CPU):
RBM rms change from theta_rel 4.80e-5 (threshold 1.06e-3), PP 1.37e-3; RBM share of |dtheta|^2 0.0013 (threshold 0.507). VERDICT: INVALID as
registered. M5' is an effective one-PP run and cannot separate structure from parameter count. Paderborn twin 4110546 held; A40 continues
until the lead rules (cancel recommended; M5'' as proposed above).
Result (1a) pool8 sampler (2026-11-04; CPU 111 core-h; pfaffian_8x8build/pool8/results.json, RUN_LOG.md; verified results.json; code committed):
- Kernel: one antiparallel J1/J2 exchange, acceptance min(1, |B|^(2 beta) n_anti(x)/n_anti(y)); PT between rungs; draw_fn(beta, S, rng) ->
  (uint64 words, ln|B|, int64 min-image ids), thinned to ceil(2 max tau).
- (a) AS REGISTERED (|z| <= 3, chi2/dof <= 2): table evaluator, first set 64 chains FAIL (max |z| 3.37, one overflow bin, 5 seen vs 13.7 expected);
  the worker then switched low-count bins to Poisson tails (post hoc, not credited); fresh 128-chain replicate PASS under the original rule (2.15,
  1.19). PP2 evaluator: set 1 FAIL (3.50), replicate PASS (2.08), pooled 96 chains FAIL marginally (3.23, chi2/dof 1.45). PP2 lnB = table lnB to
  1.1e-13 on 17280 configs, so the two runs have the same transition kernel; I read the PP2 miss as a fluctuation (~33 bins per run) but it is
  NOT closed (~30 core-h would). Broken versions fail on the table evaluator: Hastings factor dropped |z| 18, |B|^beta 951, PT always-accept 379.
- (b) 8x8 production (params_final sha 0520c5be): tau_int 1.2 / 0.7 / 0.6 sweeps at beta 1 / 0.5 / 0.25; PT swap 0.10 / 0.30; hot vs random starts
  agree (max |z| 2.2 of 30); quotable.py on <2lnB> at beta 1: QUOTABLE -66.96(12). CPU 0.33 s per sweep per walker. Slow modes > 1500 sweeps not excluded.
- (c) Sizing (derived): the LIMITING decade is the TOP one (+2), not the deep ones; beta 0.5 already covers decade -6 (occupancy 0.15). ESS/P >= 3
  needs S 3.3e6 on equal thirds {1, .5, .25} (920 core-h), 1.56e6 on {1, .5} optimised (383 core-h independent chains), 1.16e6 on {1.2, 1, .5} (273).
  Plus labels (~50 core-h per 4e5, from 1b).
- W^B_x alone is benign (ESS fraction 0.72-0.96 per decade; max W/Wbar 7.6). So the heavy entries in 1b's (c2) come from the T/B factor
  exp[(lnT_y - lnB_y) - (lnT_x - lnB_x)] (c f_J differences) or from the Marshall stub sign (my inference; with the stub, f_J is not the real one).
Open design question (mine, before any 8x8 pool): which RELATIVE decades to fit. +2..-6 is inherited from 6x6 (absolute -5..-14 there, where
Test W put 85% of the round's bond weight below -9). At 8x8 the decade range must follow where the round's bond weight sits, measured with the
real s_G and f_J on a small pool, not assumed. Its answer moves S: if the deep decades matter, beta 0.25 comes back.

## Lead/PI decisions 2026-11-04 evening (PI "1. agree", via the lead)
- Cancel M5' (A40 17720568/69, Paderborn 4110546/47 held) and the V A40 duplicate (17695322/23): DONE (run by the lead, sacct CANCELLED; watchers stopped). Earlier my cancel was blocked by the
  session's permission classifier (authorisation came through a peer); waits on the PI running scancel or granting it.
- c4: V replica with a NEW POOL SEED on the A40 only (same pool_fit code 66799f1c, same fit spec and fit seed; only the pool seed changes).
  Readouts: r_V2 (v_bare.py), augmented min (aug_readout.py). Registered use: V's PASS stands iff r_V2 >= 0.35 as well; the sampled/exact ratio
  is then quoted as the mean of the two pools with their half-difference as the pool-to-pool scatter. No kill beyond that.
- M5'' (GO after Astra and after the CPU check): PP + RBM alpha 72 (5256 parameters), RBM init sigma chosen by the CPU check, relax to B_4
  (e_rel <= 1.25e-5), then the round to T_5^b1 with the same fitter, steps and tail average as M5'.
  CPU check (before any GPU): at the round-5 start, the RBM Jacobian column norms (median, of the copy-loss residual) are within 3x of the
  P_2 block's in the two-PP arm, and the first GN step's RBM share of |dtheta|^2 is >= 0.3. Smallest passing sigma in {0.02, 0.05, 0.1}.
  Validity gate during the run: at the first saved theta (step 600) the RBM share of |dtheta|^2 since theta_rel is >= 0.3. Else INVALID, stop.
  Readouts as M5': (1) structure iff 0.4929 - r_5(M5'') > 0.15; (2) structure iff aug_min(M5'') - 3.2584e-6 > 0.3e-6.
  Prediction (lead, ~65%): "structure". Mechanism: the RBM factor is a spin-pattern function and Test N showed the round correction is not one
  (eps^2 0.95-0.98 even with finite-weight RBM features), while Test W showed the second PP stores 0.24-0.47 in every decade.
  Refutation: M5'' within both margins, which would make the gain a parameter count. Disagreeing readouts: report both, claim nothing.
pool8 closure (lead 2026-11-04: close before pool8 feeds any 8x8 fit; pre-registered before any number)
Mechanism: the chain's transition kernel P(x->y) = (1/n_anti(x)) min(1, |B(y)/B(x)|^(2 beta) n_anti(x)/n_anti(y)) depends on the evaluator only
through lnB, and PP2 lnB = table lnB to 1.1e-13, so P_PP2 = P_table. A real PP2-only bias must therefore come from a code path the table run
does not share (walker init, the in-sampler tau pilot and thinning, batching, RNG streams, the PT swap bookkeeping), not from the kernel.
Steps: (1) a code-path diff, table vs PP2 mode, listing each branch that differs (no run); (2) a fresh 96-chain PP2 replicate, new seeds, same T.
Predictions if statistics: the flagged bin (beta 1, decade -2; 0.1563 vs 0.1508) |z| < 2 in the replicate; all-bin max |z| <= 3 (a perfect
sampler exceeds 3 in one of ~33 bins ~9% of the time, so "max |z| < 2 over all bins" is NOT the bar: P ~ 0.2 even if exact); chi2/dof <= 1.5;
all 192 PP2 chains pooled max |z| <= 3. KILL ("PP2-path bias"): the flagged bin again |z| >= 2 with the same sign, or pooled max |z| > 3 with
the same bin; then the differing branch from (1) is the named term and is fixed before use. ~30 core-h.
8x8 decade range (lead: my call; pre-registered before any number)
Quantity: Test W (iii) at 8x8 for round 1 (R0 guide = production rb8_prod, B = P_1): per relative decade d of |B|^2 (median of 2lnB under beta = 1
as zero), the share of the total blocked-bond weight sum_{x,y blocked} |H_xy| |B(x) B(y)| (Delta*_x - Delta*_y)^2 with Delta* = c f_J[B] (c = 4,
mode 'fix', s_G computed directly: sign of (1 + alpha (H - E)) psi with M1's alpha -0.17543, E_total -31.9151, w = -1; E' = p=0 FN -0.4987565 x 64;
f0 = the sampled 0.9999 quantile of |f_J|, which is c2). Estimated from pool8 draws at beta {1, .5, .25, .125} reweighted to |B|^2.
Rule: fit the smallest contiguous range of relative decades holding >= 95% of that weight. Error: decade shares to <= 20% relative where >= 2%.
Prediction (mine): at 6x6 the weight peaked 5-6 decades below the beta = 1 median (absolute -12/-13 with the median near -7), and the band widens
~1.33x at 8x8, so the peak sits near relative -7/-8 and the 95% range reaches about -10/-11. Then +2..-6 misses ~30-40% of the weight, beta 0.25
(or lower) is needed again, and S grows from the deep end. Miss: the 95% range ends at -6 or above (then the inherited range stands and pool8's
sizing holds). Cost: f_J at 8x8 with on-the-fly s_G is d^2 E_L calls per configuration; the worker times a pilot and sizes it, <= 300 core-h.
c4 RUNNING (2026-11-04 18:39): pfaffian_v_c4/, pool seed 2 (sha 8a9b80e6; spec diff = seed only; check (b) PASS, max |z| 2.85 over 41 cells);
A40 fit 17741531 -> tab 17741532 -> bare 17741533 -> aug 17741534 (afterok); 53.2 s/step, readouts ~11-05 18:35 / 19:40. Check (a) job 17741259 pending
read. Next per verdict: r_V2 >= 0.35 -> V PASS stands, quote ratio mean +- half-difference; r_V2 < 0.35 -> V's PASS does not hold across pools, report to the lead.
Per-round stored-sign tax (lead 2026-11-04, from the sign angle / PI's 14x14 question; a readout of the loop, not a run):
- Sign class at 8x8 = the sign angle's 1-PP complex phase (2 seeds + margin mix); the L CNN is the accuracy reference only.
- Each round k: re-store the sign after the sign update, then read the tax. Flat = fine; growing = re-train from scratch or change the class.
- Cost point (mine): the energy tax E_FN(stored S_k) - E_FN(direct s_G,k) needs a direct-s_G grading each round, and direct s_G at the d^2
  endpoints costs 223x the CNN (A40 benchmark), so a per-round ENERGY tax at 8x8 is not affordable. Proposed readout: per round the sign
  angle's w_T/N (MCMC on |v|^(2 beta), cheap; tax_site ~= kbar w_T/N with kbar 17.8 measured at 6x6), and the energy tax only at R0 (= c1) and
  the last round. Interface: gfmc_aug takes any sign_fn; a 1-PP phase sign can use pp2_lemma's ratio machinery (one more PP inverse per image),
  so its endpoint cost is ~ one extra lemma evaluation, not CNN-like.
Condition (lead, accepted): the 17.8 factor is a 6x6 calibration. At 8x8 measure kbar = tax_site / (w_T/N) at R0 and at the last round (where the
energy tax is graded anyway). If the two differ by > 30%, or either differs from 17.8 by > 2x, the per-round w_T/N proxy is NOT trusted and the
per-round trigger needs another readout (decided then).
Savings item (4): the 1-PP phase sign at the endpoints through pp2_lemma's determinant-ratio path. Exactness: sign identical and margin Re[.]
equal to the direct-phase evaluation to <= 1e-12 RELATIVE (the absolute bar was ill-posed for large ratios in (3)), on 6x6 S_a1 and on an 8x8
1-PP, with a broken version (wrong image parity) that fails; CPU timing vs the direct phase.
Result (2) c3 (2026-11-04; pfaffian_8x8build/gfmc_aug/c3_table/gate_i.json, c3_ident/gate_ii.json; verified; gfmc_aug.py committed, sha acd2f6e4):
- (i) table-guide FN-GFMC, 4 chains, M 1024, 5.1e6 walker-tau per c, quotable.py QUOTABLE: c = 0 1.274(47)e-5 vs exact 1.218448e-5 (1.18 sigma);
  c = 4 fix 4.34(20)e-6 vs 4.367227e-6 (0.15 sigma). PASS (2 sigma, sigma <= 5e-7). Broken c = -4 misses by 30.0(5.7)e-6 = 5.3 sigma
  (predicted +33.6e-6). M 256 at c = 0: +0.98(65)e-6; M 256 - M 1024 = 0.42(80)e-6, so the population bias is below sigma. Both c = 0 deviations
  are positive (+0.55, +0.98 sigma-units 1.2, 1.5): not significant; dtau not varied.
- (ii) same-seed identity, aug_ratios vs table provider, c = 4, 32 blocks: hops and walker hashes identical, E_mix max rel 6.7e-15. Broken
  c = 3.9 moves E_mix by 2.0e-5 rel. So GFMC with the evaluator = the table GFMC of (i). c3 PASSES.
- Cost note for 8x8: the 6x6 table-guide GFMC needed 5.1e6 walker-tau per c for sigma 2-5e-7 per site.
Result pool8 closure (2026-11-04/05; job 17741261; pool8/closure_pp2/{closure,pooled192}.json; verified): STATISTICS, as registered. Code-path
diff: no table/PP2 branch changes the stationary law. Fresh 96-chain PP2 replicate: flagged bin z -0.37 (sign reversed), max |z| 2.71, chi2/dof
1.35; pooled 192 chains max |z| 2.93. KILL did not fire. 44.6 core-h.
Gap found: the production entry point draw_fn uses the in-sampler tau pilot + thinning (Pool8Sampler._measure), which no exactness test has run.
Test (pre-registered): draw_fn itself at 6x6 (table and PP2 evaluators), >= 96 independent samplers, same bins and Gaussian rule; pass max |z|
<= 3 and chi2/dof <= 1.5; broken version: keep the pilot's first-sweep (unequilibrated) draws, which must fail. The informed move stays unused.
Result M5'' init check (2026-11-04 22:00; pfaffian_m5pp/cpu_check/summary.json; harness = cx_r1fit first step to 5.6e-15; verified): NO sigma passes,
and MY MECHANISM WAS WRONG. I said the RBM columns are O(sigma sqrt N) ~ 2e-2, small, and damped away. Measured at theta_4: RBM median column
norm 2.53 at sigma 3e-3, i.e. 31x P_2's (0.081) and 2.8x P_1's (0.90); damping factor at that column 0.99. I left out the 576-image sum.
sigma 0.02/0.05/0.1 scramble the guide (e_start 4.6e-2 to 6.5e-2, ~4000x theta_4) and pass the share only because of that.
What follows: a block with LARGE columns moves FEW parameters per unit of function change (dtheta ~ J^+ r), so the parameter-space share
|dtheta_RBM|^2/|dtheta|^2 is not a measure of how much the block is used. It penalises the RBM (large columns) and would also fail the two-PP's
own P_2, whose parameter-space share at the first step is 0.205 < 0.507. The M5' gate (58e8c0e) was therefore ill-posed: its verdict INVALID is
recorded AS REGISTERED, but it does not show that the RBM was unused.
Next (pre-registered before any number): the FUNCTION-space share, S_fn = sum_w |dlnA_RBM|^2 / sum_w |dlnA_total|^2 over the copy-loss value
rows (pool weights), with dlnA_RBM = R(x; W_end) - R(x; W_start) and dlnA_total the full model change, for M5 (init -> steps 600, 1350) and M5'
(theta_rel -> step 600); and the same for P_2 in the two-PP round 5 (P_2 term vs total) as the reference. Prediction (mine, from column norm x
parameter change): S_fn(M5') ~ 0.01, far below P_2's. Kill of "M5/M5' were effectively one-PP": S_fn(RBM) >= 0.5 x S_fn(P_2). If it fires,
M5' was a valid control and should be rerun as is (sigma 3e-3); if not, the stall is undiagnosed and M5'' waits for a mechanism. ~5 core-h.
S_fn amendment (lead 2026-11-04, before any number): metric = the copy loss's own (band-uniform decade weights + blocked-bond term with lam_b),
reported in total and per decade (value and bond rows separately), for RBM (M5, M5') and P_2 (two-PP round 5) over the same stretches.
If the kill fires: log "M5' validity gate ill-posed; cancel 2026-11-04 premature" beside the INVALID record and rerun M5' as it was
(sigma 3e-3, A40, ~12 h; inside the lead's takeover ceiling, no Astra wait).
Result draw_fn test (2026-11-05; job 17748174; pool8/drawfn/*.json; verified): draw_fn with pilot + auto thin PASSES at 6x6 on both evaluators
(table 96 samplers max |z| 2.71, chi2/dof 1.41; PP2 96 samplers max |z| 1.40, chi2/dof 0.53, low: its errors look overestimated, so it is the
weaker of the two). Broken version (pilot's first sweeps leaked): table 4% leak FAILS only on chi2/dof (1.54); PP2 10% leak not detected.
Reason, measured: the 6x6 chain forgets a random start in 2-3 sweeps, so the leak moves <2lnB>(beta 1) by only ~0.02. The test resolves
a beta-1 <2lnB> shift of ~0.03 (table statistics). Read: draw_fn is exact to that resolution; the leak failure mode is self-limiting at
tau ~1 sweep (8x8 tau 1.2). pool8 is cleared to feed an 8x8 fit; the informed move stays disabled.
Result S_fn (2026-11-04 23:00; jobs 17748167-69; pfaffian_m5pp/cpu_check/summary_sfn.json, gate_sfn.txt; fit's own centred metric, one 8208-row
first-step batch per arm; block + rest = total to 5e-15; verified):
| arm, stretch | S_fn | block norm^2 | total norm^2 | S_cross |
| M5 RBM, init -> 600 | 0.964(4) | 3.05e-2 | 3.16e-2 | +0.001 |
| M5' RBM, theta_rel -> 600 | 0.46(6) | 3.96e-4 | 8.68e-4 | -0.63 |
| two-PP P_2, start -> 600 | 4.6(7) | 1.96e-2 | 4.27e-3 | -8.1 |
AS REGISTERED: kill ratio S_fn(RBM)/S_fn(P_2) = 0.10 (M5'), 0.21-0.24 (M5); no decade >= 0.5. The kill does NOT fire; "M5/M5' effectively
one-PP" is not refuted on the registered metric.
But the registered metric is ill-posed a second time: S_fn is not a share when blocks cancel (P_2's change largely cancels P_1's: S_cross -8,
S_fn 4.6), so the denominator is inflated. Non-registered, cancellation-free reading (projection share <d_block, d_total>/|d_total|^2,
derived from the same numbers): M5' RBM 0.14, two-PP P_2 0.54. So the RBM is used, about 4x less than P_2. M5' also moved the whole
function 5x less than two-PP by step 600 (8.7e-4 vs 4.3e-3). M5's RBM change (0.96) mostly re-fits its own init noise (start 12x theta_4).
My reading: M5' was not an idle-RBM run. It is a PP + RBM run whose fit makes less progress, which is part of the answer to "structure vs
count", not a defect of the control. My prediction (0.01) missed; I have no mechanism for the slower progress.

## M5' rerun (lead decision 2026-11-04 ~23:30; A40, inside the lead's takeover ceiling; pre-registered before submission)
CORRECTION to "Result M5' validity gate" (34570b4) and its cancel: the gate was ill-posed (parameter-space share penalises large-column
blocks; the two-PP's own P_2 scores 0.205 on it). The RBM was active: projection share 0.14 vs P_2's 0.54 (S_fn result above). The cancel of
A40 17720568/69 and Paderborn 4110546/47 on 2026-11-04 was PREMATURE.
Arm: exactly M5' as before: theta_rel.npy (sha bff7cdc4, e_rel 1.218426e-5), PP + RBM alpha 72, sigma 3e-3 seed 72 lineage, cx_r1fit f1374bed, round
spec = pfaffian_m5p/round spec (target T_5^b1, 1500 steps, save_theta_at 600..1350, tail average), new run dir pfaffian_m5p2.
Readouts at 1500, as registered: (1) structure iff 0.4929 - r_5(M5') > 0.15, with r_5 = (e_rel - e_end)/(e_rel - 4.367093e-6); (2) structure iff
aug_min(M5') - 3.2584e-6 > 0.3e-6. Beside them, the projection share of the RBM over the round (fit metric, as the S_fn step).
Plateau condition (lead: class against class, not speed): plateau iff over steps 1200-1500 the held-out loss changes by < 1.5% (the fit's own
plateau convention) AND the exact E_FN of theta(1200) vs theta(1500) differs by < 0.1e-6. If not, extend from theta(1500) in 600-step segments
(same fitter; checkpoints every 150; table at each segment end) until both hold, at most 3000 extra steps; then read (1)/(2) again on the
tail average of the last 900 steps. Report the verdict at 1500 AND at plateau.
Prediction (lead, ~65%): "structure". Observation, not a cause: by step 600, M5' moved the function 5x less than the two-PP (8.7e-4 vs 4.3e-3).
Refutation: M5' within both margins at plateau. Disagreeing readouts: report both, claim nothing.
M5' rerun RUNNING (2026-11-04 23:02): pfaffian_m5p2/, A40 fit 17748827 -> tables 17748829 (tab_t5, tab_s1200, tab_s1500) -> CPU readout + plateau
(m5p2_chain.sh); reproduces the cancelled run exactly at steps 10-60 (repro_check.txt); VERDICT lines ~11-05 15:30. Scripts d1687fa.
Amendment before any number of this run: plateau condition (A) uses WINDOW means of the held-out loss (steps 1050-1200 vs 1350-1500), not two
single steps (the fixed held-out batch scatters 5.8e-3..1.0e-2 step to step). (B) unchanged. The RBM projection share over the round is read
by m5pp_sfn.py on the final theta after the readout (lead, CPU).
Result savings item (4) lemma-path sign (2026-11-05; pfaffian_8x8build/signlemma/; code 654efcf, default path bit-equal to 7c4dddf; verified):
- Signs IDENTICAL to the sign angle's own direct margin code on every valid endpoint: 6x6 S_a1 1.61M endpoints, 8x8 complex 1-PP 0.51M, all three
  eval paths; every broken version (parity, flip image, phase, Marshall, mask) gives 29k-2M mismatches.
- AS REGISTERED the margin bar (<= 1e-12 relative) is MISSED: max |dm| 7.6e-12 (6x6), 4.4e-11 (8x8); relative for |m| >= 0.1 5.6e-11 / 2.4e-10.
  Cause measured: the lemma path is the less accurate one (longdouble reference), errors ~1e3 eps kappa with image cancellation kappa 5e3 / 5e4,
  i.e. conditioning of the per-image inverse, same class as the known two-swap limit. Consequence: a sign can flip only where |m| < ~1e-10;
  none seen. f_J / aug outputs with the lemma sign equal the direct ones to <= 3.6e-15 (they depend on signs only).
- Speed (CPU, 8x8, M 64, dedup): lemma sign adds +61% to the amplitude step; ~93x faster per step than the direct phase sign_fn (240x on the sign part).
- Pre-existing: pp2_lemma_test.py (a) fails on the frozen 7c4dddf module too (two-swap rel 1-2e-9 vs 1e-10), the conditioning limit recorded 11-03.
Untested: a trained 8x8 phase PP (none exists yet); the tail beyond 24 walkers at 8x8; GPU.
Lead ruling on item (4): "1e-12 bar MISSED, GUARDED" (not passed). Guard before 8x8 production: wherever |m| < 1e-8 (relative, 100x above the
observed error) the lemma path falls back to the direct phase computation; the fallback rate is logged per run. Now: report the |m|
distribution on the 2.1M tested endpoints (fraction < 1e-8, < 1e-6). Fallback rate > 1% at 8x8 -> tell the lead (the 93x shrinks; re-size).
Guard result (2026-11-05; signlemma/guard/; code ebeb550; verified mdist_*.json): |m| < 1e-8 and < 1e-6 on 0 of 1.61M (6x6) and 0 of 1.86M (8x8)
endpoints; min |m| 1.5e-5 (6x6), 6.8e-6 (8x8). Guarded signs = direct signs everywhere; injected near-zero margins are caught (100/100, 20/20)
and wrong without the guard. Guard cost +0.7%, fallback rate 0 at 8x8 M 64. Item (4): 1e-12 bar missed, GUARDED; usable. Untested on a
trained 8x8 phase PP (its margins may sit closer to 0 than a random f's).
Result 8x8 decade range (2026-11-05; decades8/final/final_old.json, analyze_old.log; 6332 draws, 76 PT processes; verified): REGISTERED RANGE
= relative decades -2..-9 (0.958 of the round-1 Delta* blocked-bond weight; stable in 99.95% of resamples). Shares: -2 .038, -3 .084, -4 .138,
-5 .164, -6 .200(24), -7 .177, -8 .115, -9 .043; -1 .017, -10 .013. 2 chains agree (chi2/dof 0.90). The inherited +2..-6 holds only 0.648.
Controls: s_G = M1's own on 240/240 (alpha = 0 broken: 31/240 differ); 6x6 control reproduces Test W (iii) (max |z| 1.88), broken |z| up to 8e3.
Estimator change (pilot, before production): each pair seen from its larger-|B| end (the smaller-end estimator was heavy-tailed: 5 draws = 68%);
validated by the 6x6 control. Mode 'old' (uncapped) is primary.
My prediction (peak -7/-8, range to -10/-11) MISSED: peak at -6, range ends at -9. The registered miss (ends at -6 or above) did not fire.
Consequences: (1) the fitted 8x8 decades are -2..-9; pool8's sizing was for +2..-6 and must be redone (the top decades that limited it drop out;
-7..-9 come in, |B|^2 mass 4.4e-5 at -9, so beta <= 0.25 rungs return). (2) Labels must use the STORED sign: direct s_G cost 129 s per
configuration here (f_J with E_L at every endpoint), i.e. ~5e4 core-h for 1.5e6 labels. (3) f0 is unsettled: the beta = 1 sampled q.9999 is 0.143
(essentially the sample max), while the 6x6 fix used the n-weighted q.9999 (0.6236 for f_J), and a |B|^2-weighted cap moves Test W's shares by up to
72 sigma. Proposal: f0 = the n-weighted q.9999 at 8x8, sampled at beta = 0 (uniform over Sz = 0, i.e. n-weighted), validated at 6x6 against 0.6236.
Stored-sign labels check (lead 2026-11-05; 6x6 exact, CPU; pre-registered before any number)
Question: production labels use f_J computed with the STORED sign S (V used s_G). Does that change the round correction?
Quantity: Delta*_S = c f_J[B_4; S_a1] and Delta*_sG = c f_J[B_4; s_G] (c = 4, mode 'old' and 'fix' f0 0.6236), on the exact 6x6 tables. eps^2 =
|Delta*_S - Delta*_sG|^2 / |Delta*_sG|^2 in the Test W (ii) bond metric (|H_xy| B_4(x) B_4(y) on blocked bonds of s_G, x the smaller side,
global constant removed), per |B_4|^2 decade over the fitted decades and in total; also with S's blocked bonds.
Mechanism: f_J[B; S] differs from f_J[B; s_G] only on rows where S != s_G at x or at a neighbour (the kept/blocked split flips). S_a1's wrong
signs sit in the low-amplitude tail, where Delta* also lives, so eps^2 should grow toward deep decades.
Prediction (mine): total eps^2 0.03-0.3, central 0.1; the deepest fitted decades above the total.
Registered: eps^2 <= 0.1 -> stored-sign labels fine; eps^2 >= 0.3 -> they cost a large share of each round (M-R law: kept fraction ~ 1 - eps^2),
decide before 8x8 labelling; in between -> report and the lead decides. Broken version: S = s_G gives eps^2 = 0 exactly; S = Marshall sign must give
eps^2 = O(1). ~10 core-h.
Result stored-sign labels check (2026-11-05; job 17749959; pfaffian_8x8build/storedsign6/storedsign6_result.json; verified): total eps^2 = 0.171
(old) / 0.168 (fix) over decades -15..-5 on s_G's blocked bonds (0.180 / 0.175 on S's). In the registered "lead decides" band (0.1-0.3).
Per decade: -15 0.61, -14 0.50, -13 0.31, -12 0.13, -11 0.041, -10 0.022, -9 0.011, >= -8 < 0.005. S != s_G on 6.1% of orbits at -15, 2.1% at
-14, 0.35% at -13, ~0 above -11. Controls: S = s_G gives 0 exactly; Delta*_sG = ln T5 - ln B_4 to 1.4e-14; Test W (iii) shares to 6e-16;
Marshall gives 297. S_a1 identity confirmed (w_T 1.7149e-6 reproduced). Prediction (0.03-0.3, deep decades above the total) HIT.
Reading (mine): the stored sign's errors sit exactly in the decades that carry the round's deep weight (-13..-15 hold ~0.4 of it at 6x6), so ~17%
of each round's correction is mislabelled there. f0 note: the fix's f0 is the SIGNED n-weighted q.9999 of c f_J divided by c (fj_audit 'cfJ'),
not an |f_J| quantile (1.31).
Lead decision (2026-11-05): 8x8 R1 uses STORED-sign labels; expected loss ~17% of the round gain (eps^2 0.17 at 6x6), registered before the run.
When a trained 8x8 sign exists: eps^2 at 8x8 on the 6332 decades8 draws (direct-s_G f_J already stored); >= 0.3 -> fix the sign's deep tail
(sign angle), not the labels. The per-decade disagreement goes to the sign angle as its training target.
Result pool re-sizing + f0 (2026-11-05; decades8/sizing/sizing.json, beta0/b0_{6,8}/; code 40259d9):
- Pool for fitted decades -2..-9 (derived): rungs {1, .5, .25} with S_q 3.3e5 / 4.5e5 / 3.2e5 = 1.10e6 draws, ~231 core-h of sampling (halves 198 / 225);
  the optimum is flat ({1, .5, .125} 235; {1, .5} 243). Weak spot: beta = 1 efficiency at -5/-6 rests on 21/8 draws (those decades come from beta .5/.25).
- f0 (c2): definition = n-weighted signed q.9999 of f_J (0.6235649 exactly on theta_4's table). Sampled at beta = 0: 6x6 0.668(34), 1.3 sigma from
  0.6236 (reproduces); 8x8 1.22(8) (~1.2e4 draws, so this is near the sample max; rough). With f0 1.14-1.31 mode 'fix' leaves the range at -2..-9,
  shares within 0.001 of 'old'. c2 is settled for the range; the 8x8 E' check of c2 is still open.
8x8 R1 pool, x-draws (lead GO 2026-11-05; CPU ~231 core-h; pre-registered before any draw)
Guide: B_0 = P_1 = production rb8_prod params_final.npz (sha 0520c5be), the amplitude the round-1 pool samples; exists, no GPU. pool8 cleared
(closure STATISTICS 51d9de4; draw_fn test 5d3efec). Draw ONLY x (words, ln|B|, beta, chain id) now: partners need the stored sign (blocked =
S(x) S(y) > 0) and labels need S and f_J, so both wait for the trained 8x8 sign.
Plan: independent chains per rung, S_q = 3.3e5 / 4.5e5 / 3.2e5 at beta 1 / .5 / .25, draw_fn with pilot + auto thin, seeds recorded.
Checks on the output (pass before use): per-decade counts per rung within 20% of decades8 sizing.json's occupancy (decades -2..-9); the two seed
halves agree (max |z| <= 3 over decade fractions); median 2lnB at beta 1 within 0.3 of -67.03.
x-draw pilot (2026-11-05; job 17750526; pool_x/pilot/): draw_fn runs ONE shared PT ladder with one thin (3 sweeps, tau_max 1.0-1.4) and its sweep
count is set by the largest rung target, so the registered targets cost ~524 core-h at the measured 0.43 s/sweep/walker (0.30-0.51 by node),
not 231 (which assumed independent per-rung chains). Decision (mine, before any production draw): keep the tested draw_fn path unchanged
(a per-rung rewrite would need a new exactness test) and raise the cap to 560 core-h; every rung then gets 4.5e5 draws (beta 1 and .25 get a free
surplus). Gate amendment: cells whose sizing count is < 100 are tested at 3 sigma combined instead of the 20% band (the 20% band is
statistically impossible there); the literal 20% misses are printed beside.

## Lead decision after Astra (2026-11-05; cj-6a20013aa7ba, /scratch/a/A.Otaifi/astra/v_go/REVIEW.md, read)
GPU readiness work GO; production R0 / c1 / R1 grading NO-GO until (a)-(e) close:
(a) trained 8x8 sign: eps^2 < 0.3 on the 6332 decades8 draws, block errors + per decade, no-update control (waits for the sign angle's checkpoint).
(b) c1 as a PAIRED equivalence test (Astra: independent arms need sigma <= 0.72-1.08e-6 each; two at 3e-6 give sigma(tax) 4.2e-6). Design below.
(c) pool sized on BOND-weight ESS (root weights w_in mean_k rho), not configuration ESS (V's -6: 7.0k bond vs 23.2k configuration).
(d) labels: total bill first (direct s_G ~150 A40-h for 1.35e6 entries); decide after (a). RETRACTED from the registration: "eps^2 0.17 means ~17%
    of the round lost". That transferred the M-R storage law to a changed FN operator, which it does not cover. The stored-sign label effect is
    measured, not inferred.
(e) f0 at 8x8 rests on ~1.2 tail observations: bootstrap interval, or a less extreme quantile.
Also from Astra: close c2's E' part (E' must match the S|B| convention, not the Re B p=0 number) with a sensitivity check; c5 literally (same-sign
bare and augmented R0/R1); c is a 6x6 transfer, so R0 rescans c; the competitor bar is best trainable network + FN at equal TOTAL GPU-h.
c1 paired design (paper, before any run):
- Same amplitude |B| (production), c = 0, two sign arms: stored S and direct s_G. At c = 0 direct s_G costs one E_L per one-swap endpoint
  (d ~ 145 per walker step), one level, not d^3, so the s_G arm is affordable on GPU.
- Estimator: common random numbers. Both arms run the same walker population with the same RNG stream, and the reconfiguration uses the same
  uniforms (comb). The arms coincide until a walker meets a bond whose kept/blocked status differs (S != s_G at x or y), which is rare in the
  bulk. tax = mean over blocks of (E_S - E_sG) per paired block; error from the scatter of the paired differences over >= 2 independent pairs.
- Precision target: sigma(tax) <= 0.5e-6 (bounds a tax below 3e-6 at > 4 sigma, and resolves a ~1e-6 tax).
- Numbers come from a 6x6 pilot where the answer is exact: table-guide GFMC (c3 machinery) with S_a1 vs s_G on B_4 at c = 0; exact tax from the
  two table FN solves. Measure the variance ratio var(paired diff)/(var_S + var_sG), check the paired estimate against the exact tax within
  2 sigma, and show a broken version (independent seeds) gives the naive variance. The 8x8 cost then = per-arm variance from the H100 benchmark
  x the measured ratio (if the ratio transfers; re-measured on a short 8x8 pair before production).
Gate (a) protocol (lead / sign angle 2026-11-05, before any number): run the decades8 eps^2 check on 3-4 T-SIGN8 checkpoints (held-out BCE does
not track the tax: T-DEEP2 accuracy 0.979 -> 0.990 while the tax rose 0.27e-6). Split the 6332 draws into halves A/B by process block; choose the
checkpoint with the lowest eps^2 on A; the gate (eps^2 < 0.3) is judged on B, with block errors and per decade. No-update control as Astra asked.

## Two branches for R1 after T-SIGN8 (lead 2026-11-05: no trained 8x8 phase-PP checkpoint beats the no-update sign; T-IW is the last try)
Label constants, ONE set everywhere: M1's /scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json: alpha -0.1754318714549749, E -31.915135844116154,
w = -1 (decades8's; the sign angle's -0.18 / -31.915008 are NOT used in this angle).
Branch 1 (T-IW gets below d): stored-sign loop as planned; gate (a) on T-IW checkpoints with the A/B split protocol.
Branch 2 (T-IW fails): R1 at a FIXED sign, the amplitude-only loop. My reading of "sgn(M1)": the sign of the production network itself,
sgn Re[w psi_rb8_prod], frozen across rounds (the sign of M1 p=0, whose FN energy is -0.4987565(28)); NOT s_G (that would be M1 p=1's sign,
which cannot be evaluated at walker time with c > 0). Labels T = sgn(M1) |B_k| e^{4 f_J[B_k; sgn(M1)]}; for k >= 1 sgn(M1) at endpoints comes
from a second lemma evaluator of P_1 (PhaseSign-like). Competitor: M1 (or the best trainable network) + FN p=0 at the same sign, equal total
GPU-h. Gate (a) becomes information: eps^2 of sgn(M1)-labels vs s_G-labels on the decades8 draws = "value of the sign step not captured".
That number is computed now (CPU, below).
Branch 2 confirmed by the lead: sgn(M1) = sgn Re[w psi_rb8_prod], frozen (not s_G). BAR for branch 2 = M1 + FN p=0 at the same sign:
-0.4987565(28) per site, chains p0_s1 + p0_s2, 0.40 H100-h, chain diff -0.74 sigma, QUOTABLE (/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/analysis_final.json).
Result gates (c), (e), c2-E' (2026-11-05; pfaffian_8x8build/gates_cE/; control reproduces Astra's V bond ESS to 4e-16; f_J(y) evaluator = stored
to 2e-12; verified c_ess.json, c_size.json):
(c) FAILS for the registered pool. Bond/configuration ESS ratio per decade -2..-9: .89 .88 .84 .66 .70 | .076 .12 .15 (asymptote .060 at -7, 68%
.046-.097), and it still falls with the draw count at -7. Registered 4.5e5 per rung gives bond ESS 6.2e3 / 8.1e3 / 1.0e4 at -7/-8/-9, i.e. 0.34-0.56 P
(V had 1.3 P at its worst). Meeting 3 P at the 1:1:1 split needs 3.1-6.7e6 per rung; cheapest is ~8.6e6 more beta-0.25 draws, 1.2-2.7e3 core-h.
Mechanism (equation): rho = W^B_x exp[c (f_J(y) - f_J(x))]. In decades -7..-9, 3-10% of x have f_J(x) < -0.5 (down to ~ -1.1), so the factor
exp(-c f_J(x)) reaches e^4.4 ~ 80 at c = 4; those x carry 14-17% of a decade's root weight each (top-1). None in -2..-6. This is a property of
the loss's own bond weights |T(y)/T(x)|, not of the sampler: the round's correction is concentrated on few deep x where the target sits far
below B. It scales with c (another reason R0 rescans c).
(e) f0 = 1.22, 68% 1.09-1.31 (bootstrap, GPD tail fit 1.22(7)); q.999-based alternative 1.45 (95% 1.38-1.61). The range stays -2..-9 for all f0
in 1.05-1.61, shares move <= 0.0023. f0 is settled at 1.22 for 'fix'; it does not move anything measurable.
c2 E': label eps^2 = 9.9e-7 per (1e-4/site)^2, quadratic in the shift (9.9e-5 at 1e-3/site). The p=0 error 2.8e-6/site gives eps^2 ~1e-9; E'
needs to be known to ~1e-3/site. The Re B vs S|B| convention difference is below that (to check with the c=0 equivalence run). c2 E' CLOSED
for labels.
Pool expansion (lead GO 2026-11-05, staged, cap 2.7e3 core-h; pre-registered before any draw)
Stage 1: +2e6 beta = 0.25 x-draws (single-rung chains; a 1-rung ladder is the PT code with no swaps; first a 6x6 table exactness check of the
single-rung path, max |z| <= 3, chi2/dof <= 1.5, broken |B|^beta fails). Measure, with sgn(M1) (cheap, from the lemma ratios) as the working sign:
f_J(x) on all draws, partners K = 2 and their f_J, root weights, bond ESS per decade -7..-9 as a function of the draw count m (nested subsets).
Continue to 8.6e6 iff bond ESS grows ~linearly in the number of heavy x (f_J(x) < -0.5); STOP and tell the lead if the -7 ratio keeps falling with m
(tail heavier than a finite pool covers). Final gate on the built pool: bond ESS >= 3 P per decade, with the sign R1 actually uses.
Tilted chain, paper only (fallback): target |B|^(2 beta) W^B(x) e^{-c f_J(x)}. The x-side factor that makes the tail is then sampled, so the
remaining root-weight scatter is the y-side mean_k e^{c f_J(y)} and the in-decade band reweighting; if those behave as in decades -2..-6, bond/config
ESS rises from ~0.07 to ~0.7: ~10x fewer draws at -7..-9 (derived, unmeasured). Cost per proposal: f_J(y) needs one-swap ratios at y (one E_L-class
call): ~80 ms on CPU (informed-move timing) against ~5 ms for the plain move, i.e. ~16x. Net on CPU ~1.6x MORE expensive than brute force; on an
H100 (E_L ~0.1 ms batched) the per-proposal cost is negligible, so the tilt is the GPU route. Exactness test needed before use (6x6 table).
Stage-1 note (2026-11-05): pool8 cannot run a 1-rung ladder (median lookup needs a beta = 1 rung); authorised an opt-in med2 argument (default path bit-identical, checked), med2 = -67.03 at 8x8; the 1-rung path gets its own 6x6 exactness test before any draw.
Gate (a) first candidate (lead 2026-11-05): sign_scale/r8/ppw_s1, 1-PP complex phase, full space-group projection, one seed, trained on alpha -0.18
labels; w_T 5.87e-5 [5.23, 6.50]e-5 vs no-update d 1.63e-4 (sign_scale/r8/gradew/wT_ppw_s1.json). Gate (a) on it before the r8f checkpoints: eps^2 of
Delta*_S vs decades8's Delta*_sG (M1 consts.json constants), Test W (ii) bond metric, decades -2..-9, block errors; half A/B split as registered
(single checkpoint, so A and B are both reported; the gate reads B). Pass (eps^2_B < 0.3) -> branch 1 live; the stage-1 heavy-x set is then
recomputed under this sign.
Result branch-2 information (2026-11-05; jobs 17754200, 17755755; pfaffian_8x8build/eps_m1/; jackknife over 76 process blocks; controls: s_G path
reproduces stored fx/fy to 9e-16, shares to 3e-17, Marshall broken 4.5; verified): eps^2(sgn(M1) labels vs s_G labels) = 0.400(10) over -2..-9
(halves 0.394(12) / 0.406(15), QUOTABLE agreement). Per decade -2..-9: .036 .119 .247 .305 .456 .551 .616 .667. sgn(M1) != s_G on 0.5% (-6),
6.1% (-7), 10% (-8), 21% (-9) of draws. So the frozen production sign misses ~40% of the round's correction measure, concentrated deep; the
fixed-sign loop (branch 2) is an amplitude-only loop that does not carry the sign step. Information, not a gate.
Result gate (a) on ppw_s1 (2026-11-05; pfaffian_8x8build/gate_a/; export sha 930b61b6, theta aac1b0e6 verified; PhaseSign reproduces the export
0/2000, broken parity 2057; fallback 2 / 1.24e8; S = s_G gives 0; verified analyze_ppw_s1.log): FAIL. eps^2 half B 0.539(44) (A 0.564(71), total
0.551(42)) vs bar 0.3. The no-update step-0 PP gives 0.568(32): no measurable gain from training on this metric, although the sign angle's w_T
fell ~3x (5.87e-5 vs 1.63e-4). sgn(M1) gives 0.400(10), better than both. Per decade (ppw_s1): -2 .12, -3 .24, -4 .53, -5 .59, -6 .63, -7 .60,
-8 .63, -9 .70; S != s_G at x 4.9 / 9.0 / 16.1% at -7 / -8 / -9.
Branch 1 does not open on this checkpoint. Open question before reading it as a class failure: the sign was trained on alpha -0.18 s_G labels,
the reference is alpha -0.17543. If s_G itself moves between the two alphas where 1 + alpha (E_L - E) is near zero (deep tail), part of the
miss is a target mismatch. Control (pre-registered): eps^2 of s_G(-0.18)-labels vs s_G(-0.17543)-labels on a decade-stratified 1000-draw subset
of decades8 (-4..-9); >= 0.2 means the alpha mismatch alone is a large part of 0.54 and the sign must be retrained on M1's constants (or graded on
its own alpha) before a verdict on the class; < 0.05 means the mismatch is not the cause. ~40 core-h.
Also noted: w_T (|v'|^2-weighted wrong-sign weight) does not track eps^2 (bond metric with (Delta*_x - Delta*_y)^2): 80% of eps^2 comes from bonds
the sign wrongly unblocks, ~0.4% of the bond weight. A sign trained to minimise w_T is not trained on what the labels need.
Gate (a) round 2 (lead / sign angle 2026-11-05; registered before any number): candidates (i) r8f frozen-RBM checkpoints 3000 / 6000 / 9000 / 9900
(sign_scale/r8f/HANDOFF.md, stored_sign8.py; alpha -0.18 labels), (ii) T-IW frozen-RBM gamma arms trained on M1's alpha/E labels, as their exports
land (sign_scale/r8iw/HANDOFF.md). PP-full arms cancelled. Select the lowest eps^2 on half A across ALL candidates; gate it on half B (< 0.3).
Reference: step-0 = sgn(M1) per half, A 0.394(12), B 0.406(15) (this class starts there, so passing needs a ~25% cut). Branch 2 preparation
continues in parallel; R1 takes whichever branch is ready with its gates passed.
Result c1 paired pilot (2026-11-05; c1pair6/; gfmc_pair.py 1a892fa; exact tax job 17752394; verified exact_tax.json):
- Exact c = 0 tax of S_a1 on |B_4| (table FN solves): 1.32362e-6 per site (e 1.3508104e-5 vs 1.2184484e-5).
- Paired estimator, clock moves + per-walker stochastic-rounding reconfiguration (PK, 4 pairs, 1.23e6 walker-tau per arm): 1.36(81)e-6, z +0.04;
  QUOTABLE 1.34(79)e-6. Comb design as written (PC): 1.74(1.37)e-6. Independent seeds (IK): naive variance (ratio 1.09).
- Variance ratio var(diff)/(var_S + var_sG): PK 0.66(3), but 0.43 -> 0.94 over the run's fifths; walker coincidence 0.79 -> 0.10. The saving
  is only ~1.6x (3.2e6 vs 5.1e6 walker-tau per arm for sigma 0.5e-6).
- Mechanism: a walker that hops through a bond where S != s_G diverges and never re-merges, and branching lets diverged lineages take over;
  the comb spreads any weight difference to all later walkers (PC decouples within one fifth).
Reading: the estimator is UNBIASED (exact tax recovered) but common random numbers alone do not keep the arms coupled over a long run. The
remaining lever is short forks from a common population: many segments, each started from the same walkers, read while R is still ~0.4.
That needs the projection-relaxation time after a fork (bias of the forked arm) to be shorter than the decoupling time; not measured.
For 8x8 sizing: per-arm variance from the H100 benchmark x 0.66 (long runs) as the conservative factor; S != s_G is more frequent at 8x8, so
decoupling is likely faster there.
c1 sizing (lead 2026-11-05): 8x8 c1 is sized with NO pairing saving (ratio 1.0); any saving seen at 8x8 is a bonus. c1 matters only on branch 1 (T-BOND passes); branch 2 (fixed sgn(M1)) needs no c1. Short-fork work is dropped for now.

## BRANCH 2 STARTED (lead decision 2026-11-05 evening; Astra cj-a2cbf28c7311 rejected T-BOND's surrogate, puts the PP-phase class < 0.3 at ~20%)
Branch 1 reopens only if a T-IW checkpoint passes eps^2 on half A AND on FRESH process blocks (half B has been reused). T-IW finals are being graded
(ga2_g05/g10_19998, running).
Branch 2 = fixed-sign amplitude loop at sgn(M1) = sgn Re[w psi_rb8_prod], M1 consts.json constants everywhere.
Steps, each gated: (1) stage-1 pool passes bond ESS >= 3 P per decade -2..-9 with sgn(M1) (if the tail cannot be covered: tell the lead before
labelling); (2) labels on that pool (sgn(M1) is free from the lemma ratios); (3) fit R1: two-PP warm from rb8_prod (A40, or H100 inside the
lead's 15 H100-h ceiling); (4) no production grade until the H100 benchmark gives H100-h per point; anything above the ceiling goes to the PI.
Mechanism (why a fixed-sign loop can gain): lattice FN energy depends on the guide's amplitude ratios on sign-violating bonds, not only on its
signs. At 6x6 the c = 4 augmentation at fixed s_G moved E_FN from 1.218e-5 to 4.37e-6 per site with no sign change. So amplitude-only rounds
have headroom; how much at sgn(M1) is unknown. The target's correction measure differs from s_G's by eps^2 0.40.
PREDICTIONS (mine, registered before any 8x8 number; low confidence, ~50% for each range):
- R0-aug (B_0 = rb8_prod, sgn(M1), c rescan 2..5): E = -0.49882 .. -0.49886 per site (a gain of 0.6-1.0e-4 below p=0 -0.4987565(28)). My basis:
  the s_G-design R0 predicted about p=1 (-0.49894, 1.9e-4 gain); scaled by ~(1 - eps^2) ~ 0.6 for the lost deep correction, minus the
  stored-label loss.
- R1 bare (two-PP fit to T_1 under sgn(M1)): -0.49877 .. -0.49881 (a stored fraction r ~ 0.2-0.5 of the R0-aug gain).
- R1 aug: -0.49884 .. -0.49889.
KILLS: R0-aug not below p=0 by > 3 sigma -> the augmentation does not transfer to 8x8 at sgn(M1) (stop, diagnose c and f_J at scale). R1 bare
not below p=0 by > 3 sigma -> the fixed-sign loop does not store at 8x8.
BAR: M1 + FN p=0 at the same sign, -0.4987565(28), analysis_final.json. Same-sign comparisons only (c5).
Branch 2 bars (lead 2026-11-05): every branch-2 number is reported against BOTH (i) same-sign p=0 -0.4987565(28) (the mechanism test, as
registered) and (ii) M1 + one Lanczos step + FN p=1 -0.498946(5) (the competitor, which carries the sign step).
READ THIS BEFORE ANY BRANCH-2 RESULT: the predicted R1 aug range (-0.49884..-0.49889) sits 6-11e-5 ABOVE bar (ii). Branch 2 tests whether
amplitude rounds work at 8x8, not whether the loop wins there; a win needs the sign back. A pass on (i) is not a loop advantage.
Branch 2 final stage (lead, from a PI question, 2026-11-05; registered before any number): after the amplitude rounds converge at sgn(M1), run
M1's p=1 pipeline UNCHANGED with only the network swapped: psi_1 = (1 + alpha (E_L - E)) B_k, its own sign, alpha and E from the same VMC rule as
M1, then FN-GFMC. HEADLINE: B_k + p=1 + FN against M1 + p=1 + FN -0.498946(5) (results/m1_8x8), at equal TOTAL GPU-h including the loop
(pool, labels, fits, grading). The fixed-sign comparisons against p=0 stay as the mechanism test.
Mechanism: the Lanczos step's gain comes from the sign it adds plus the amplitude it starts from; a B_k with a better amplitude at a fixed sign
should carry its amplitude gain through the step, minus any interaction (B_k's own sign may differ from sgn(M1) where it was fitted against
sgn(M1) labels).
Prediction (mine, ~50%): B_k + p=1 below M1 p=1 by 1-3e-5 per site (-0.49896..-0.49898), i.e. about the R1 bare gain over p=0 carried through.
Kill: not below -0.498946 by > 3 sigma (sigma ~7e-6 combined) -> the amplitude gain does not survive the sign step. Also report it per GPU-h.
Cost ~8 H100-h at M1's rate; goes to the PI before submission (takeover ceiling).
Result M5' rerun at 1500 (2026-11-05; A40 17748827, tables 17748829, CPU 17761847; pfaffian_m5p2/readout_round/{readout,plateau}.json; gates rerun
by the lead: m5p2_gate.py final, m5p2_plateau_gate.py):
- r_5(M5') = 0.1218 (tail average; theta(1200) 0.054, theta(1500) 0.083); aug_min 4.0125e-6.
- Readout (1): 0.4929 - 0.1218 = +0.371 > 0.15 -> STRUCTURE. Readout (2): 4.0125e-6 - 3.2584e-6 = +0.754e-6 > 0.3e-6 -> STRUCTURE. They agree.
- Predictions (mine: r 0.10-0.20, aug 3.8-4.1e-6) HIT; the lead's "structure" (~65%) holds AT 1500.
- PLATEAU (amended rule): (A) window means 1050-1200 vs 1350-1500 change 0.12% (< 1.5%: PASS; the single-step 7.7% is the noisy form we
  replaced); (B) E_FN(1200) - E_FN(1500) = 2.24e-7 > 0.1e-6: FAIL. NOT PLATEAUED: E_FN is still falling (~0.75e-7 per 100 steps).
- What extension could change: readout (1) flips if r reaches 0.343 (e <= 9.50e-6, another 1.7e-6 drop; ~2300 steps at the current rate if it
  stayed linear); readout (2) flips if aug <= 3.56e-6. Per the registration the verdict is re-read at plateau; extension in 600-step segments
  (max 3000) goes to the lead first.
M5' record until plateau: "STRUCTURE at 1500, not plateaued".
Extension segment 1 (lead GO 2026-11-05; A40 only; registered before submission): warm from theta(1500) (final iterate), same fitter and spec,
600 steps (1501-2100), theta saved every 150; tables of theta(1800), theta(2100) and the tail average of the last 900 steps (theta 1200..2100,
saved iterates). Plateau re-read on 1800 vs 2100 (E_FN) and window means 1650-1800 vs 1950-2100 (L_ho). If not plateaued: segment 2 goes to the
lead. Prediction (linear extrapolation of the ~0.75e-7 per 100 steps fall): r at 2100 ~0.17-0.18, aug ~3.9e-6, so both readouts still
STRUCTURE. Caveat: the fit restarts its optimiser state (trust radius, damping) from the spec at step 1501, a discontinuity that the
1500-step registration did not have; it is recorded beside the result.
Result alpha-mismatch control (2026-11-05; job 17761852; alpha_ctl/alpha_ctl_result.json; control 17761264 exact; verified analyze.log): eps^2 of
s_G(-0.18, E -31.915008) labels vs s_G(M1) labels = 0.024(4) over -4..-9 (halves 0.024 / 0.024; QUOTABLE A vs B). Per x-decade -9..-4: .16(7) .041
.032 .031 .012 .008. NOT THE CAUSE (registered < 0.05): the gate (a) miss of ppw_s1 (0.54) is not a label-target mismatch; it is the stored
sign's own deep errors.
CORRECTION (2026-11-05, found by the ext1 worker): at 1500 I reported plateau (A) as PASS (0.12%). That number is a 4-point window (steps
1170-1200 vs 1470-1500), not the registered 16-point windows 1050-1200 vs 1350-1500, which give 2.9% > 1.5%: (A) FAILS as registered. The
verdict "not plateaued" is unchanged (B failed too). The ext1 plateau gate prints both and reads the registered 16-point form.
Ext1 RUNNING (submitted 2026-11-05 16:20; A40 fit 17763914 PENDING, table 17763915, CPU chain m5p2x_chain.sh): warm theta(1500) sha 325a9482;
tail average = 7 iterates 1200..2100 at spacing 150 (incl. 1500, as at 1500); same seed, so ext1's batch stream repeats round 1's first 600 steps
(same pool, different theta; recorded). Expected VERDICT ~8.5 h after start (--test-only start 11-06 06:29).
Ext1 amendment (lead, before the fit started; 2026-11-05 ~17:05): fit_overrides.seed 1 -> 2 in ext1/fit_th/spec_r1.json (read at job start; job
17763914 still PENDING; seed-1 copy kept as spec_r1.seed1.json). In cx_r1fit the seed drives the training sampler, the held-out batch, the WBAR
estimate, the fixed batch TRB and the JAX key; model state unchanged. So ext1 sees new training draws; its held-out loss is on a new batch (plateau
(A) compares windows inside ext1 only) and WBAR is a fresh estimate of the same normalisation (recorded as a small change of the loss weights).
BRANCH 2 = OPTION (ii) (lead decision 2026-11-05, from T-B2P1, sign_b2p1/*.json, controls 7.061e-6 / 5.574e-6 reproduced). This supersedes the
fixed-sign grading plan above where they differ:
- Rounds: amplitude rounds at fixed sgn(M1); the Jacobi head (c f_J) is used ONLY to build each round's refit target, not in the graded guide.
- Final guide: one signed Lanczos step at c = 0 on sgn(M1)|B_k| (psi_1 = (1 + alpha (E_L - E)) sgn(M1)|B_k|, alpha/E by M1's VMC rule), then FN.
- Evidence at 6x6: sgn(psi_P) on B_6 + step 3.705e-6; same sign + head, no step 7.202e-6 (worse than the competitor 5.574e-6).
- HEADLINE: B_k + p=1 + FN vs M1 + p=1 + FN -0.498946(5), at equal total GPU-h including the loop.
- R0 CHECK: M1's p=1 reproduced through the new evaluator (B_0 = rb8_prod, sgn(M1), step, FN) within 2 sigma of -0.498946(5).
- Mechanism test (kept): R1 bare at sgn(M1) vs p=0 -0.4987565(28).
- HYPOTHESIS (from the 6x6 analogue (b): the step on the refit B gains 1.87e-6 of the competitor's 5.57e-6, ~34% of M1-p=1's error to E0):
  8x8 headline gain over M1 + p=1 = ~34% of M1-p=1's error to E0. E0 is not known at 8x8: Hu -0.49906(1) gives an error of 1.14e-4 and a gain of
  ~3.9e-5; ours -0.498992(16) gives 4.6e-5 and ~1.6e-5. So B_k + p=1 = -0.498962 .. -0.498985. Caveat: at 6x6 B was fit with s_G labels; at 8x8 it is
  fit under sgn(M1) (labels eps^2 0.40 off s_G's), so I expect the low end or below. Kill: not below -0.498946 by > 3 sigma.
Result gate (a) round 2 (2026-11-05; gate_a/round2/summary.json; 12 candidates, rfa evaluator; step-0 control reproduces sgn(M1) eps^2 0.400(10)
exactly through the new path; verified): FAILS. No candidate beats sgn(M1). Best on half A: T-IW gamma 0.5 step 19998 (M1 labels), A 0.610(33),
B 0.674(31), total 0.642(23). Others 0.71-1.47; within each arm eps^2 falls with training but from > 0.9 and never below 0.40.
Mechanism (measured): training lowers the fraction of x with S != s_G (19.8% -> 14-16%) but moves errors into the HIGH-amplitude decades:
best candidate -2..-5: .40 .42 .72 .68 against sgn(M1)'s .04 .12 .25 .31. The sign is trained on a configuration-weighted wrong-sign count
(w_T), while the labels need the bond metric |H| B_x B_y (Delta*_x - Delta*_y)^2, where a wrong high-amplitude bond costs far more. The two
objectives conflict: what lowers w_T raises eps^2. Branch 1 stays closed (needs a pass on A and on fresh blocks).
Result registered x-draw pool (2026-11-05; pool_x/prod, job 17750939; gate.out; diag/diag_halves.json; verified gate.out): 450,080 draws per rung
(beta 1 / .5 / .25), counts within 13.7% (limit 20%), median -66.903 (diff 0.127), Sz all 32, recompute 6e-13. AS REGISTERED: FAIL on halves only
(beta 1, decade -2: fA 0.1510 vs fB 0.1548, z -3.44 > 3). Diagnosis: statistics (permutation null p ~0.010-0.011 over the 24 cells; between-chain
variance ratio 1.08(11), no drift, flat block scaling, error already includes chain scatter; a slow mode would need ratio ~4). Decision (mine):
the pool is USED, with the registered FAIL beside it. A real occupancy shift of this size could not bias the fit: pool_fit self-normalises the
weights inside each (beta, decade) cell, so a cell's occupancy changes only its ESS, not its weights' mean. This pool is in any case only the
-2..-6 part; the deep decades come from the stage-1 expansion + importance resampling (bond-ESS gate).
Lead agreed (2026-11-05): pool used, no confirmation batch. Condition: if R1's per-decade held-out loss is an outlier in decade -2 (the beta-1 cell that failed halves), re-open it.
Result H100 benchmark (2026-11-05; Paderborn 4117903 + 4117939, 1.44 H100-h; h100bench/job{A,B}/summary_*.json; adapter gfmc_aug8.py 7d8719a,
6x6 identity with the c3 run: hops/hashes equal, E_mix 4.7e-15; verified summaries). M = 1024, f64, 27 hops per walker-tau:
| arm | s per walker step | s per walker-tau | variance per site^2 per walker-tau |
| c = 4, stored phase sign | 0.0127 | 0.348 | ~1.1e-6 (NOT QUOTABLE, 2 chains disagree) |
| c = 0, stored phase sign | 0.0010 | 0.027 | 6.5e-6 (QUOTABLE at M 256: E_mix -0.498716(13)) |
| c = 0, direct s_G | 0.0155 | 0.426 | not measured |
H100-h per point (solo M 1024; ~1.6x less with 6 concurrent chains): c = 0 5.4 (sigma 3e-6) / 49 (1e-6); c = 4 ~12 (5-18) / ~105.
The sign's own lemma dominates c = 4 (0.0077 of 0.0127 s/step); a two-PP amplitude costs about the same as one PP. Phase-sign fallback 9e-10.
Noted, not quoted: c = 0 with the stored phase sign sits ~4e-5 ABOVE p=0's own-sign value (the stored sign is worse, as gate (a) said); the c = 4
estimate -0.49890 is unquotable.
BRANCH 2 GRADING PLAN (option (ii); derived from these rates and M1's measured p=1 cost; for the lead/PI):
- At sgn(M1) the sign comes free from the amplitude's own ratios, so the bare c = 0 grade costs ~ the plain step (1.3e-4 s/step, ~8x below
  the c = 0 phase-sign arm): mechanism test R1 bare vs p=0 at sigma 3e-6 ~0.7 H100-h (M1's own p=0 took 0.40 H100-h for 2.8e-6).
- The p=1 guide on B_k needs E_L at every one-swap endpoint (the direct-s_G-like arm, 0.43 s/walker-tau, ~M1 p=1's 0.51): M1 took 7.9 H100-h
  for sigma 5e-6. R0 check (M1 p=1 through the new evaluator, 2 sigma at sigma 1e-5): ~2 H100-h. HEADLINE (B_k + p=1, sigma 5e-6): ~8 H100-h.
  sigma 3e-6 would be ~22 H100-h.
- R1 fit on H100: 3-12 s/step x 1500 steps ~1.3-5 H100-h (derived, unmeasured at 8x8). Labels with sgn(M1): lemma cost, small.
- Total to the headline: ~12-16 H100-h at sigma 5e-6. The predicted gain (1.6-3.9e-5) against M1's 5e-6 error is resolved at 2.5-6 sigma.
  The lead's ceiling is 15 H100-h, so the upper end goes to the PI.
Branch 2 GO (lead 2026-11-05, renewed takeover ceiling 40 H100-h for the first graded 8x8 point): sequence bond-ESS gate -> labels -> fit -> bare
c = 0 mechanism grade -> R0 check -> headline (headline only after the bare grade passes its registered kill). express_hpcalot first (check the
November remainder of 20), normal QoS after; 6 concurrent chains where it saves; H100-h logged per step against 40; any step > 1.5x its estimate
-> stop and tell the lead. Estimates: fit 1.3-5, bare grade 0.7, R0 check 2, headline 8 H100-h.
Headline precision (registered before running): sigma 5e-6 on B_k + p=1; the difference to M1 p=1 -0.498946(5) then has sigma ~7e-6, so the
predicted low-end gain 1.6e-5 is ~2.3 sigma and the high end 3.9e-5 ~5.6 sigma. A gain below 2.1e-5 (3 sigma) cannot be claimed at this sigma; it
would be reported as "consistent with a gain of x +- 7e-6", not as a pass.
Result branch-2 prep (2026-11-06 ~00:00; pfaffian_8x8_r1/prep/; code above):
- Labels under sgn(M1): = slow direct path to 2.0e-12 on 80 draws (16 deepest-f_J); partner masks 0/11320 differ; Marshall broken 1.55. CPU 0.7
  core-s per pool entry (~370 core-h for 1.87e6, derived). Observation: under sgn(M1) only 1 of 1500 deep draws has f_J(x) < -0.5 (the heavy-x
  tail of gate (c) was measured under s_G), so the bond-ESS deficit may be much smaller at sgn(M1): the stage-1 gate measures exactly this.
- Grading providers: BareM1Sign = M1 p=0 guide, 0/28880 sign differences, ratios to 2.3e-10, rates 1e-12; P1Guide = M1 RBMPPGuideG, 0/28880
  signs, ratios to 1.14e-9 (registered 1e-10 MISSED; cause measured: pp2_lemma's E_L at the worst endpoint is off brute force by 6.7e-9, M1's by
  6.3e-11; effect on the FN rates 9e-12, on d_FN 2.5e-12: immaterial). Broken versions fail. CPU per walker step: P1Guide 3.42 s vs M1 p=1 3.09 s.
- R1 fit: pool builder = pool_fit's; pool_fit's held-out median bug on filtered pools fixed in b2_prep.fit. Dry run: the default P_2 start
  (sep 0.3, lam 0.1) wrecks the start (held-out 8.1-8.9 vs B_0's 0.147), even at lam 1e-5, because a 30% perturbation of a 64x64 f changes
  |det| by orders of magnitude, so |lam P_2| >> |P_1|. Decision (mine): keep P_2 far from P_1's tangent space (sep 0.3; Astra P16) but set lam
  so that the pool-median |lam P_2 / P_1| = 0.05 at the start (6x6 round 5 ended at median 0.67); accept if the start loss is <= 1.2x B_0's.
R1 warm-start scan (2026-11-06; prep/t2/warmscan_lamt*.json): with sep 0.3, ln|P_2/P_1| has median 14.85 and 10-90% spread 11.55-18.22 (a random
30% perturbation of f gives a PP whose amplitude is unrelated to P_1, unlike 6x6's trained P_2). So a median-normalised lam leaves a heavy tail.
Start loss vs B_0's 0.147: t 0.05 -> 4.8x, 0.01 -> 1.75x, 0.003 -> 1.16x, 0.001 -> 1.04x. Dry runs fall from both registered targets and the
negated steps are rejected.
Rule (registered before the next scan): scan sep in {0.03, 0.1, 0.3} x median target t in {0.003, 0.01, 0.05}; among the pairs whose start loss
is <= 1.2x B_0's, take the largest t (tie: the larger sep). Then a 20-step CPU dry run must show P_2 USED: its projection share of the function
change in the fit metric >= 0.1 by step 20 (the M5' lesson). Otherwise take the next pair.
R1 warm start CHOSEN by the registered rule (2026-11-06; prep/warm/scan.json, dry_sep0.03_t0.05/share.json; verified): sep 0.03, median target
0.05 (lam 0.0397 on the toy pool), start held-out loss 0.159 = 1.08x B_0's. ln|P_2/P_1| quantiles q1/q50/q99 -1.18/0.23/2.80, so P_2 is O(1)
of P_1 across the pool, not a tangent-space copy (a linearisable perturbation would sit at ratio ~1 +- small). 20-step dry run: P_2 projection
share 0.178(30) at step 20 (accept >= 0.1; it scatters 0.02-0.31 over steps 5-20, so this is a reading, not a trend); held-out 0.159 -> 0.138 with
a minimum at step 7 and one spike (0.512 at step 16). Launcher override R1_SPEC_OVER='{"p2_sigma_rel": 0.03, "p2_lam_median_target": 0.05}';
lam is recomputed on the production pool.
Result M5' ext1 (2026-11-06; A40 17763914 / 17763915, CPU 17774657; ext1/readout/{readout,plateau}.json; gates rerun by the lead):
- Tail average 1200..2100: r_5(M5') 0.1265, aug_min 4.0076e-6 (c* 4.0). Readout (1) +0.366 > 0.15, (2) +0.749e-6 > 0.3e-6: STRUCTURE, both agree.
- Prediction (linear: r ~0.17-0.18, aug ~3.9e-6) MISSED: the E_FN fall decelerated. E_FN 1200 / 1500 / 1800 / 2100 = 1.1761 / 1.1537 /
  1.1500 / 1.1482e-5; change per 100 steps -7.5e-8, -1.2e-8, -6.1e-9.
- Plateau: (B) E_FN(1800) - E_FN(2100) = 1.8e-8 < 1e-7 PASS; (A) held-out window means change 16% (16-point) / 9.8% (4-point): FAIL. The held-out
  loss (new batch, seed 2) ROSE 8.6e-3 -> 1.2e-2 while E_FN stayed flat. NOT PLATEAUED as registered.
- Bound (derived): at the current decaying rate, E_FN falls at most ~2e-7 more over the remaining 2400 steps to the 3000 cap, r <= ~0.15;
  readout (1) needs r >= 0.343 (another 1.6e-6) to flip, readout (2) needs aug <= 3.56e-6. Neither is reachable unless the rate turns up.
  Recommendation to the lead: stop the extension and record "STRUCTURE (1500 and 2100), E_FN plateaued, held-out loss not" rather than spend
  another ~15 A40-h; the lead's call under the registered rule.
M5' FINAL (lead 2026-11-06): "M5' STRUCTURE at 1500 and 2100 (r_5 0.1265 vs 0.4929; aug 4.0076e-6 vs 3.2584e-6); E_FN plateaued; held-out loss
did not; extension stopped by the lead". DEVIATION from the registered extension rule, reason: (B) passes (1.8e-8 per 100 steps) and the
decaying-rate bound (r <= ~0.15 by 3000) makes both flips unreachable; the held-out rise is on the new seed-2 batch, which E_FN does not see.
TRIED_IDEAS rows P27 (M5') and P28 (8x8 stored sign classes, gate (a)) added. The A40 goes to branch 2.
Result pool expansion stage 1 (2026-11-06; pool_x2/, jobs 17755345 (1-rung exactness: max |z| 1.99, chi2/dof 0.76; broken 232), 17755902 (2e6
beta 0.25 draws, halves max |z| 0.95), 17770149 (f_J under sgn(M1), 0/3316 sign mismatches, = fJ_fixed to 9e-16); ess/ess_stage1.json;
verified): registered reading CONTINUE (the -7 ratio is flat, not falling). But the premise changed: UNDER sgn(M1) THE HEAVY-x TAIL IS RARE.
Bond/config ESS ratio at m = 2e6: -7 0.382(17), -8 0.233, -9 0.134 (still rising with m), against 0.06-0.15 under s_G (gate (c)). Heavy x
(f_J(x) < -0.5): 8 / 35 / 205 of 2e6 at -7/-8/-9 (gate (c) under s_G: 3-10% of deep draws). New-draw bond ESS 8.1e4 / 6.1e4 / 3.8e4; with the
registered pool ~8.7e4 / 6.9e4 / 4.8e4 against 5.4e4: only -9 is short (~0.9 of target), and its ratio rises faster than linear.
So the 8.6e6 continuation (sized under s_G) is NOT needed for branch 2. Decision (mine): stage 2 = +0.8e6 beta 0.25 draws (~130 core-h runtime),
then build the combined pool and run the FINAL gate (bond ESS >= 3 P in every decade -2..-9 under sgn(M1), Kish ESS on the actual combined
root weights, not summed). Mechanism check (pending, sg_check.py): same x under s_G vs sgn(M1) at -9, to confirm that the sign is what removes
the heavy tail. Core-h are counted as runtime (per core), the convention of the earlier caps; billed is ~2x.
OBSERVATION (for the sign angle's joint-network hypothesis (b); lead 2026-11-06): under the network's OWN sign sgn(M1) the heavy-x tail
(f_J(x) < -0.5) shrinks from 3-10% of deep draws (direct s_G, gate (c)) to ~1e-4 (248 of 2e6, stage 1). In f_J = ln SK - ln den, SK (the kept-bond
sum) stays large when the node is the amplitude's own: B's own nodes leave kept bonds around every x, while s_G's nodes cut through regions where
|B| has no node, leaving x with few kept bonds and a very negative f_J. Pending confirmation on the same x (sg_check.py, -9). The 8.6e6
continuation is cancelled; stage 2 (+0.8e6) and the final gate proceed.
Result FINAL POOL GATE (2026-11-06 ~12:30; pool_x2/comb/pool_s2b/pool.npz, 2,000,175 entries, sha 426487e9; final_gate.json; verified): PASS, thin.
Bond ESS (training entries, held-out excluded) -2..-9: 5.80 6.56 8.02 9.39 10.3 9.26 6.89 5.54 (x1e4); min 5.54e4 at -9 = 1.03 x 3P. Broken
(rho = W^B only) moves -9 to 9.49e4 (detected). The -9 margin is 3% against a ~20% uncertainty of its ratio: an ESS/P of ~3 there, not more.
Pool = registered pool_x (781,644) + stage 1 (-2..-6 labelled, -7..-9 converted, 4e-12) + stage 2 (-9 only, 111,598). Spec: med 2lnB
-67.03209 (decades8 origin; R1 must read it from the pool), K 2, sgn(M1), mode fix, c 4, f0 1.22, E' -31.920416. ~258 core-h (budget 250, slow nodes).
Mechanism CONFIRMED on the same x (100 decade -9 draws): heavy fraction 0.00 under sgn(M1) (min f_J -0.455) vs 0.09 under s_G (min -0.948),
median f_J -0.034 under both. The sign alone makes the heavy-x tail (OBSERVATION above).
R1 SUBMITTED (2026-11-08 17:38): Paderborn H100 4143668 (express_hpcalot, est. start 18:39) + theorie A40 race 17842947, race_watch_eta.sh (pid 3412767, PPID 1); pool_s2b sha 426487e9; warm sep 0.03, t 0.05, lam_init 0.0268 recomputed on the pool (toy 0.0397); stop rule 7.5 H100-h. (11-06: the submission was lost when the worker hit the weekly API limit and no wait was armed on its job id; my error.)
R1 readiness reading (lead 2026-11-08, before the check's numbers): report B_0, warm, theta_tail and theta losses on the same held-out batch
(2 seeds) with ratios to B_0. B_0 vs T is the "store nothing" baseline: theta above B_0 means the round stored less than no refit. If theta <
B_0, run the bare c = 0 FN grade even if the start ratio misses 1.2, LABELLED OUTSIDE REGISTRATION in that case.
R1 fit DONE: Paderborn 4143668, 1500 steps, 1.73 H100-h, held-out 0.262 -> 0.18; A40 copy cancelled by the race watcher (ETA rule, 28 h later).
R1 graded network (fixed before any energy): the REGISTERED one is theta_tail (tail average of the 7 iterates 600..1350 + final, as round 5 / V's
theta_bar); it is graded. The final iterate theta is graded beside it only if the paired per-entry check (16 x 1024 held-out entries, L_net - L_B0,
bootstrap over batches) shows it clearly better; no swap after energies. 2-batch check (startcheck.json) unreadable: B_0 0.251 vs 0.165 between
batches (heavy single entries).
Result R1 paired loss check (2026-11-08; r1/startcheck/perentry_50*.npz, paired16.json; 16 x 1024 held-out entries, bootstrap over batches):
| network | paired mean L - L_B0 | fraction of entries below B_0 |
| warm start | +0.055(23) | 0.42 |
| theta_tail (registered) | +0.059(123) | 0.77 |
| theta (final) | +0.053(123) | 0.77 |
B_0 mean 0.348. Start rule: ratio of 16-batch means 1.159 <= 1.2, PASS (marginal). theta_tail vs theta: +0.005(9), no difference, so theta_tail is
graded alone. The mean is NOT below B_0, but the median entry improves (median diff -0.0054) and excluding the top 0.1% of entries (16) the mean
diff is -0.114 (~-33% of B_0). The mean is carried by a handful of CATASTROPHIC entries: +1011 (the same entry drawn twice), +200, +169, +91, ...;
the top 20 carry 83% of all positive excess. So R1 improved ~all entries and blew up a few: a defect (likely ln|1 + lam P_2/P_1| exploding where
|P_2/P_1| is in its tail) that could matter for the FN grade if walkers reach those configurations. Diagnosis before grading: below.
R1 grade decision rule (lead, registered before the diagnosis numbers): the diagnosis gives the SIGN of the outlier errors (|B_1| too large: P_2
dominating; too small: P_1 + lam P_2 cancelling) and the |B|^2-weighted pool fraction of each region. If the fraction is < 1e-3: run the bare c = 0
FN grade OUTSIDE REGISTRATION, with a weight-concentration readout per block (max single-walker weight share, top-10 share, share of walker-tau in
the outlier decades); a grade dominated by a few walkers is "not converged" whatever its error bar. If the fraction is larger: cap or penalise
|lam P_2/P_1| (form registered first), refit, then grade.
Result R1 outlier diagnosis (2026-11-08; r1/outliers/; batches reproduced to 0.0; verified rep2.txt summary): MY GUESS WAS WRONG. Not P_2
dominating (ln|lam P_2/P_1| in -1.5..+1.74 at all 30 worst entries) and only one true cancellation. The excess sits at entries with very heavy
bond weight (sum of partner wb > 5: 0.8% of entries carry all of it; corr(sum wb, diff) 0.73), deep x (decades -9/-8, rung 0.25), heavy partner
ln|B_0(y)/B_0(x)| 2.4-5.6; bond part 98% of their loss. Sign: |B_1| too LARGE at x in 25/30 (+0.9..+3.8), mostly from the trained P_1 itself
(ln|P_1| - ln|B_0| has std 0.48 at decade -9 vs 0.10 at -2); too small in 5/30. B_0 is already bad there (mean loss 59 for sum wb > 10) and
theta_tail doubles it. Dropping those 136 entries the mean loss falls 0.150 -> 0.076 (-49%).
Region fractions: P_2-dominated 6e-4 uniform at x, 4-6e-4 band-weighted, 0/20000 under |B_0|^2 (95% bound 1.5e-4); cancellation 2-3e-4.
Both < 1e-3 -> by the registered rule the bare c = 0 grade runs OUTSIDE REGISTRATION, with the weight-concentration readout. Open (not
understood): why the fit raised P_1 at deep heavy-wb x instead of lowering it; that is the term to change if R2 runs.
Grade spec: B_1 = theta_tail, guide sgn(M1)|B_1| (gfmc_b2 BareM1Sign), c = 0, 6 chains on one H100, sigma <= 3e-6; plus B_0 through the SAME code
as the same-sign control (c5) to <= 3e-6, so the mechanism comparison is paired in code; bar (i) -0.4987565(28) quoted beside. Readout per block:
max walker weight share, top-10 share, walker-tau share in decades -7..-9. Budget <= 2 H100-h total; stop at 1.5x.
Lead's hypothesis (the heavy bond term makes the fit match e_x - e_y, and e_x follows the partner's error) TESTED on the 30 outliers
(r1/outliers/outliers_top30_typ30.csv, heavy partner = larger wb): median |e_x| 1.38, median e_y(heavy partner) 0.04, median |e_x - e_y| 1.42;
0/30 have |e_x - e_y| < 0.3 |e_x|. REFUTED: the heavy partners are fitted (e_y ~ 0) and x is not; the bond differences themselves are wrong, so
the fit genuinely fails those bonds (and FN sees bond differences, so this can cost energy). Open, for R2: why the GN fit leaves the
heaviest-weight bonds unfitted (candidates: they are 0.8% of entries, so rarely in a step's batch, and when drawn their weight is clipped by the
trust radius; or the P_1 class cannot move deep x without spoiling the bulk). Not tested.
Separator (a) sampling/clipping vs (b) class conflict, for R2 (lead 2026-11-08; registered now, runs AFTER the bare grade lands; CPU):
from theta_tail, 50 GN steps on a batch made only of the heavy entries (sum of partner wb > 5; the 136-entry class, training entries only), no
trust clipping (same damping), tracking L_heavy (that batch, plus held-out heavy entries) and L_bulk (a fixed 4096-entry held-out batch without
heavy entries). Readings: (a) if L_heavy falls >= 50% while L_bulk rises < 10% -> fix by oversampling heavy entries with corrected weights;
(b) if L_bulk rises >= 25% by the time L_heavy has fallen 50% -> a structural change for deep x (architecture angle); in between -> report both
curves. Broken control: the same 50 steps on a random bulk batch must not lower L_heavy by 50%.
Separator addition (lead, from proposals/2026-11-08_tail_fitting_literature.md, after Gravina et al. arXiv:2410.10720 Sec. 3.2-3.3; I have
not read the paper, the definition below is mine): per-sample gradient SNR of the copy loss at theta_tail, SNR = ||mean_i g_i||^2 / mean_i
||g_i - mean g||^2 over (i) the heavy entries (sum wb > 5, training) and (ii) a bulk set of the same size; read the ratio SNR_heavy / SNR_bulk.
Registered: ratio < 0.1 -> sampling/noise (supports (a)); ratio >= 1 with no progress under the unclipped GN steps -> class (supports (b)); in
between -> reported beside the GN-step separator, which stays primary. CPU, after the grade.

## STATE AT MIGRATION (2026-11-09 00:05; angle session cleared)
Running, no VERDICT yet (re-arm the wait):
- Bare c = 0 grade, OUTSIDE REGISTRATION: Paderborn 4165381 (express_hpcalot, 2:50 wall, started ~23:16, ~1.4 h expected), 6 chains: B_1 = R1
  theta_tail (sha cbeecc75) x3, B_0 = rb8_prod x3, same code (gfmc_b2 BareM1Sign). Run dir pfaffian_8x8_r1/grade_bare/prod; ANALYSIS_BRIEF.md has
  the gate command (scripts/analyze_grade.py prod --sigma 3e-6) and the branches; watcher watch_job.sh pid 222542 (PPID 1, m51). Probe 4164840
  took 0.137 H100-h. Its worker was let go; the next session runs the gate itself if the watcher's analysis fails (it often does: "command refused").
After the grade (all registered above): (1) read E_1, E_0, E_1 - E_0 vs bar (i) -0.4987565(28) and the concentration readout (wmax median > 0.2 =
not converged); mechanism kill: R1 bare not below p=0 by > 3 sigma. (2) If it passes: R0 check (M1 p=1 through the new evaluator) then the headline
B_k + p=1 (~8 H100-h; total within the lead's 40 H100-h ceiling for the first point; the 8 H100-h p=1 run was to go to the PI before submission).
(3) CPU separator (a)/(b) + gradient-SNR readout for R2 (sections "Separator ..."). H100-h spent on branch 2 so far: bench 1.44 + R1 fit 1.73 +
probe 0.14 + grade (running).
Closed this session: M5' = STRUCTURE (P27); 8x8 stored-sign classes fail gate (a) (P28); V reproduced (c4, ratio 0.916(24)); stored-sign labels
eps^2 0.17 (6x6); pool gate passed (pool_s2b sha 426487e9); heavy-x tail is a property of s_G, absent under the own sign (confirmed same-x).

## Result R1 bare c = 0 grade (2026-11-09 00:24; OUTSIDE REGISTRATION; Paderborn 4164840 probe + 4165381 prod, 1.26 H100-h; grade_bare/prod/{ANALYSIS.md,summary_grade.json,raw_*.png}; verified: chain means recomputed from E_mix_post.npy, raw diff curve flat)
| guide sgn(M1)|B|, c = 0, M 1024, 3 indep. chains each | E / site |
| B_1 = R1 theta_tail (cbeecc75) | -0.498803(32) |
| B_0 = rb8_prod, same code | -0.498755(39) |
| E_1 - E_0 (paired by chain; unpaired -4.82e-5(44)) | -4.95e-5 +- 7.6e-6 (6.5 sigma) |
E_0 - bar (i) = +1.2e-6 +- 4.0e-6 (driver reproduces p=0). E_1 - bar (i) = -4.65e-5 +- 4.3e-6 (11 sigma): MECHANISM KILL DOES NOT FIRE.
Inside my registered R1-bare range (-0.49877 .. -0.49881). Concentration: wmax share max 0.002 / 0.011, top-10 <= 0.012 / 0.020, walker-tau in
decades -7..-9 ~1e-3: the R1 loss outliers do not show in the grade. sigma 3.2e-6 / 3.9e-6 (target 3e-6, not extended).
Next (registered): R0 check (M1 p=1 via P1Guide through grade_bare.py, sigma 1e-5, ~2 H100-h) being staged in pfaffian_8x8_r1/r0_p1 (lead's GO);
then the headline B_1 + p=1 (~8 H100-h) goes to the PI before submission. H100-h on branch 2: 1.44 + 1.73 + 1.26 = 4.43 of 40.
Separator (a)/(b) + SNR scripts ready, smoke-tested only (r1/separator/README.md); the heavy class is 17462 training entries (1.09%), not 136
(that was the count in the 16x1024 batches); off the critical path now, for R2.
Result R0 check (2026-11-09 04:08; Paderborn 4171854, 2.18 H100-h, M 256 = M1's M, 6 chains x 80 post-burn blocks, burn 5 tau; r0_p1/prodM256/):
NOT QUOTABLE (quotable.py refuses 80 samples per chain; my sizing). Estimate E_p1 -0.498965(8) (chain scatter 5.2e-6, chain chi2/dof 0.4), vs M1 p=1
-0.498946(5): diff -1.9e-5 +- 9e-6 (~2-2.7 sigma below; M1's M=64 run -0.4989467(14) gives ~3.5 sigma). All 6 chains below M1; chain medians too
(~-1e-5). Rare low blocks (< -0.4995): 4 of 480 here (k4, k5) vs 2 of 1860 in M1 p1_s1/s2 at the same M, heavy-tailed E_L of psi_1 near its
nodes (1 + alpha(E_L - E) -> 0). Guide identity is at ratio level only (9e-11). Registered R0 bar (2 sigma) not met as an estimate; reading:
unresolved (statistics with heavy tails vs a walk difference). Test running: CPU trajectory identity, new driver vs M1's driver, same start/seed
(r0_p1/identity/). If identical -> the evaluator is M1's, the offset is statistics, and the headline should be PAIRED in code (B_1 + p=1 vs
B_0 + p=1, same driver, same starts) so it does not lean on M1's reference. If not identical -> find the term before any headline.
H100-h on branch 2: 4.43 + 2.18 = 6.61 of 40. R0 at 2.18 vs estimate 2; quotable extension (~1.4-2 H100-h) would take R0 past 1.5x -> lead's call.
Result R0 trajectory identity (2026-11-09; r0_p1/identity/{README.md,result.json}; ~1.5 CPU-h): grade_p1.py and M1's own driver gfmc_run.py
(sha acb9c83f, unmodified) give the SAME FN-GFMC walk from the same start and seed (M 16, 4 blocks): max |dE_mix/N| 4.9e-14, weights 6e-14,
branching identical; broken alpha = 0 differs by 8.6e-5..3.5e-3. Same E_mix estimator and population control (fixed-M systematic resampling).
R0 has no transient (burn tenths -0.49892..-0.49897; post-burn quarters -0.498998(22) -0.498940(11) -0.498980(14) -0.498952(15)).
READING: the evaluator is M1's to machine precision; the R0 offset -1.9e-5(9) is statistics on a heavy-tailed E_L (80-block chains), not code.
DEVIATION from registration: the R0 bar (E within 2 sigma of M1) is replaced by this stronger identity test; R0 is NOT extended.
HEADLINE PROPOSAL (for the lead -> PI; not submitted): paired in code, M 256, dtau 0.1: arm A = B_1 (theta_tail) + p=1 via P1Guide on the PP2
params, arm B = B_0 + p=1 (R0's 6 chains count toward it). Per-block chain std ~1.7e-4 (R0) -> ~1150 post-burn blocks per arm for sigma 5e-6,
difference sigma ~7e-6 (= registered headline precision). Cost: arm B +680 blocks ~2 H100-h; arm A 1150 blocks ~3.2 H100-h at B_0's rate, x1-2.5
for the two-PP (bare grade: B_1 2.5x slower per block); total ~5-10 H100-h + burns, ~7-12 H100-h. Branch-2 total would be ~14-19 of 40.
Also quoted against M1 p=1 -0.498946(5). Kill (registered): B_1 + p=1 not below M1 + p=1 by > 3 sigma; here also checked against arm B.
Lead (2026-11-09): headline production GO, PI covered; stop > 15 H100-h. Pending lead/PI: c = 0 control (B_1's class refit to B_0 itself, bare grade, ~2 H100-h) -- do not start until the lead confirms.

## R1 ATTRIBUTION CONTROL (c5-type), REGISTERED 2026-11-09 before any staging (lead + PI GO)
Question: is R1's bare gain (-4.7e-5 vs p=0, -4.95e-5(76) paired vs B_0) due to the Jacobi-augmented target, or to the refit itself
(two-PP class, warm start, fitter noise)? Run: identical to R1 except the target. Pool pool_s2b (sha 426487e9), warm start (sep 0.03, t 0.05,
lam_init as R1), same fitter/arm pp8, steps (1500), seed, stop rule, graded network chosen the same way (theta_tail, iterates 600..1350 + final).
Target T = B_0 (c = 0: labels are B_0's own ratios, no f_J). Then the same bare grade: sgn(M1)|B|, c = 0, 3 independent chains, M 1024, code
grade_bare/code/grade_bare.py (theta path swapped, sha pinned), sigma ~3e-6; paired against the existing B_0 chains of grade_bare/prod.
Predictions: lead E(B_1^{c=0}) = -0.498755 +- 1e-5. Mine: 0 .. +1e-5 above B_0 (fit noise in deep decades, ln|P_1| - ln|B_0| std ~0.5 at -9 in R1,
can only cost), ~70%.
Reading (gain = E_0 - E(B_1^{c=0}), sigma ~5e-6): gain >= 2.3e-5 -> R1's gain NOT attributable to the Jacobi target; gain < 1e-5 -> attributed
to the target; 1e-5..2.3e-5 -> partial, reported as such. A loss (E above B_0) also attributes, and sizes the fit-noise cost R1 carries.
Cost: fit ~1.7 H100-h (R1 took 1.73) + grade ~0.9-1.3 -> ~2.6-3 H100-h (lead's estimate 2; stop and tell the lead above 4.5). Express, submitted
after the headline probe, never displacing headline production.
Headline decisions (2026-11-09, angle lead, before the probe): arm A uses B_1's OWN alpha/E by M1's VMC rule (registered l.1972 'pipeline unchanged, network swapped'; l.1869 is the pool-label constants), arm B M1's consts.json (= that rule on B_0); check: rule on B_0 reproduces M1's alpha/E. Chains >= 256 post-burn blocks (quotable.py minimum), production 5 A x 260 + ~4 B x 260; arm B verdict from new chains only, R0's 6 pooled as secondary. KILL fires unless A < M1 and A < B, each by > 3 sigma.
Result headline probe + step constants (2026-11-09; headline_p1/; probe2 4182862 0.24 H100-h; VMC 4183654 + 4185784 ~1.3 H100-h):
arm A 89 s/block (3A+3B mixed), arm B 61 s/block (= R0). M1's VMC rule (vmc_moments.py 937f0ed8, imported unchanged) reproduces M1 on B_0:
alpha -0.17403(324e-5) vs -0.17543(186e-5), E -31.91394(142e-5) vs -31.91514(76e-5) (z 0.4/0.7; broken constants fail at z -4.3/7.0).
B_1's own constants: alpha -0.19357(317e-5) (10% larger, ~3.6 sigma), E -31.92084(57e-5), w = -0.8786 + 0.4775i (28.5 deg from M1's -1: M1's w
would have been wrong for arm A). check_h2 0.963(71). Production: arm A 5 x (260 + 50 burn) on express (20 h limit), arm B 4 x 310 on QoS cont
(express remainder ~9.2 h < projection 9.9-13.3). Projection 11.5-14.9 H100-h incl. spent 1.56 (stop at 15). c=0 control fit 4182798 sits in
cont with estimated start 2026-11-21 (Priority).
c=0 control placement (lead 2026-11-09): fit moves to theorie A40 (R1's b2r1a40 launcher, target only changed; s/step after 50 steps to be registered here); grade on A40 if <= 24 h projected, else Paderborn express next month. Paderborn cont job 4182798 stays pending until the PI approves its cancel in this session (est. start 11-21).
Code notes for > 8x8 (2026-11-09): pp2_lemma dedup keys are one uint64 (bit per site): wrong above 64 sites (14x14 collision shown, Astra cj-852432069699); needs a multi-word key + collision test vs direct evaluation. Speed lever: pp2_lemma recomputes inverses per call (no persistent update).

## K=3 ROUND-6 TEST at 6x6 (lead + PI GO "regular queue go", 2026-11-09; REGISTERED before any code or run)
Mechanism (architecture angle's wording, board block 27): P_2 spends its freedom on the round it first stores. (R5: fresh P_2 stored r_5 0.49;
R6, same P_2: r_6 0.25; q = 0.51.)
Question: does adding one fresh PP per round restore per-round storage? Run: repeat R6 exactly (from seed 1's theta_5, target T_6 =
B_5 e^{4.0 f_J^G[B_5]} target_T6_sG.npy, same fitter, 1500 steps, seed 1, mean of 7 checkpoints 600..1500, same budget) with a fresh third PP:
B = P_1 + lam P_2 + mu P_3, P_3 and mu initialised by the rule R5 used for P_2 (sep 0.03, t 0.05 -> mu_init recomputed on the same x).
Grade exactly as R6 (pfaffian_r6/readout): bare e, r_6 = (8.3319e-6 - e) / 5.0735e-6, aug c-scan min. Against K=2 R6: r 0.2505, aug 2.814e-6 (c 3.9).
CPU checks before submission: mu = 0 reproduces e(theta_5) (rel <= 1e-9, as R6's 4.7e-11); P_3 Jacobian columns nonzero (tangent rank rises);
a broken P_3 (columns zeroed) leaves the rank unchanged.
Predictions: architecture angle r_6(K=3) ~ 0.45 (~55%). Mine: 0.33-0.48 (~50%); the fresh PP restores most but not all of R5's 0.49,
since T_6's residual is harder (R5 two-PP stored 0.49 of a larger gap). Aug: 2.4-2.7e-6.
Reading: r_6(K=3) >= 0.40 -> one PP per round restores storage; <= 0.30 -> the decline sits in the fit, not the class; between -> partial.
Caveat registered now: K=3 has ~1.5x K=2's parameters; a pass does not separate "a fresh PP" from "more parameters" (M5 showed RBM count
does not substitute at round 5, which is the nearest control).
Placement: Paderborn regular QoS (cont/normal), NOT express (headline's). No race copy (a race needs a cancel, which needs the PI's word).
Cost 3-7 H100-h (R6 fit + tables; K=3 ~1.5x per step), inside the lead's ceiling; stop and tell the lead above 9.
K=3 placement (lead, pre-approved): Paderborn normal or cont if start <= 2 days, else theorie A40 (~22 h at R6's 53 s/step), concurrent with the c=0 control if a second A40 is free, else after it.

## Result HEADLINE 8x8 (2026-11-09 20:01; Paderborn 4186187 arm A + 4186886 arm B, express; headline_p1/prod{A,Bx}/; verified: chain means recomputed from E_mix_all (burn 50 dropped), running means flat, raw_AB_chains.png)
| p=1 + FN, M 256, dtau 0.1 | E / site | chains x blocks |
| A = B_1 (theta_tail) + own step (alpha -0.19357, w rot. 28.5 deg) | -0.4989545(34) | 5 x 260 |
| B = B_0 + M1 step (same code) | -0.4989390(45) | 4 x 260 |
| M1 p=1 (M1 code, reference) | -0.498946(5) | |
A - B = -1.55e-5 +- 5.6e-6 (2.7 sigma); A - M1 = -8.5e-6 +- 6.1e-6 (1.4 sigma). REGISTERED KILL FIRES (needs > 3 sigma on both).
Prediction (-0.498962 .. -0.498985) MISSED HIGH: the gain is 0.85-1.55e-5, below the 1.6-3.9e-5 range.
Equation for the miss: E(p=0) -> E(p=1) gains 1.83e-4 on B_0 (-0.4987565 -> -0.498939) but 1.52e-4 on B_1 (-0.498803 -> -0.4989545):
the step recovers 3.1e-5 less on B_1. So the Lanczos step and the refit correct largely the SAME error (R1's p=0 gain 4.65e-5 shrinks to
~1.55e-5 after the step: about 1/3 survives). The 6x6 analogue (34% of M1-p=1's error) assumed the two were additive; at 8x8 they overlap.
INCONSISTENCY (affects the errors): R0 (same code, constants, M; 6 x 80 blocks) -0.498968(6) vs arm B -0.498939(4.5): 2.9e-5 (3.8 sigma
naive). Starts are both beta = 1 pool draws; R0 had 4 low-tail blocks in 480 vs 1 in 1040. B agrees with M1 (+7e-6 +- 6.7e-6). Either the
block errors are underestimated (heavy-tailed E_L of psi_1; arm B chain chi2/dof 0.06 is also suspiciously small) or there is a start
effect; not resolved. Pooling R0 into B would give A - B ~ -0.5e-5. Either way no 3-sigma gain.
Cost: headline 11.7 H100-h (probes + VMC 1.56, A 6.7, B 3.45). Branch 2 total ~18.3 of the lead's 40.
K=3 AMENDMENT (2026-11-09, before submission; my error): the registration said "P_3 and mu by the rule R5 used for P_2 (sep 0.03, t 0.05)".
That was wrong: sep 0.03 / t 0.05 is the 8x8 R1 warm-start rule. R5's actual P_2 init was a DIFFERENT TRAINED PP at finite separation
(round-1 L-SWO PP) with lambda 0.1 (median |lambda P_2/P_1| 0.095). Corrected rule, as the lead specified ("as P_2 was in R5"): P_3 = a
trained 6x6 PP from outside theta_5's lineage (candidates: pfaffian_r2g run_s2, pfaffian_r1 run_b), the one whose separation from P_1
(theta_5) is closest to R5's P_2-vs-P_1 separation and that is not close to P_2; mu real, set so median |mu P_3/P_1| = 0.095 on the fit's
held-out x. Readings, predictions and placement unchanged. Placement: theorie A40 (Paderborn cont est. 11-15, lowcont/nocont December;
'normal' is not a QoS of this account), wall 47 h for a projected 30-36 h.
K=3 SUBMITTED (2026-11-09 ~21:30): theorie A40 fit 17871864 + table 17871865 (afterok; if the fit fails the table stays pending and needs a
PI-approved cancel), chain r6k3_chain.sh -> CPU readout; watchers 222392/222400/222401 (PPID 1); est. start 11-10 12:59, ~30-36 h.
P_3 = pfaffian_r1/run_b/round_1/theta.npy (sha f03e1325, RBM block dropped as R5's P_2). Separation rms(f - f_1)/rms(f_1): 0.250 from
theta_5's P_1, 0.173 from P_2 (R5 reference: L-SWO P_2 0.0865 from theta_4's P_1). The r2g / lswo2 PPs (0.09-0.10 from P_1) sit 0.05-0.06 from
P_2 (same family) and were excluded by "separation from P_2 > half of P_2's distance from P_1 (0.067)". So P_3 is ~2.9x further from P_1
than R5's P_2 was: noted as a deviation that cannot be avoided with the PPs on disk. mu_init 0.0698 (median |mu P_3/P_1| = 0.095). Start
held-out loss 3.905e-3 (K=2 R6 3.871e-3). Checks: mu=0 1.8e-14 / spliced e rel 8.3e-11; Jacobian vs FD 2.3e-9; rank 4757 -> 7133 (zeroed
4757); K=2 path bit-identical (sha 449d8057). Gate: pfaffian_r6k3/ANALYSIS_BRIEF.md; owner wait = pfaffian_r6k3/owner_wait.sh (re-run it after a clear).

## STATE (2026-11-09 21:30) — running, no VERDICT yet
- c=0 attribution control fit: theorie A40 17852869 (pending); duplicate Paderborn cont 4182798 pending (est. 11-21), cancel needs the PI's word.
- K=3 round-6 test: theorie A40 17871864 + 17871865 (above).
- Headline done (Result HEADLINE 8x8). Duplicate Paderborn cont 4186183 (arm B copy, est. 11-22) pending, cancel needs the PI's word.
- Open: why the p=1 step and the refit overlap (paper first; cheapest test VMC E_L moments of B_1 vs B_0 per decade, CPU); R0-vs-B 3.8 sigma.
c=0 control fit A40 17852869: 70.5 s/step (steps 1-50), 1500 steps = 29.4 h > fit_sec 86400 (copied from R1's A40 spec) -> stops at ~step 1225. Held-out c=0 loss 4.6e-2 -> 1.6e-7 by step 40 (the fit reproduces B_0). Decision (no cancel/restart): let it stop; theta_tail by hand over the existing iterates 600..1200 + final (deviation); identity check |Delta ln B| and bond differences vs B_0 on held-out + heavy entries (control: R1's theta_tail must show O(1)). If the bond differences are ~1e-4 or below, propose replacing the A40 grade (FP64-weak, long) with this bound.
c=0 CONTROL READOUT CHANGED (lead 2026-11-10, registered before the fit ends): the FN bare grade is replaced by an identity bound.
(1) MAX (plus median, 99%) of |Delta ln B| and of bond differences |Delta e_x - Delta e_y| (Delta = ln|B_net| - ln|B_0|) over held-out AND heavy-wb
(sum partner wb > 5) entries. (2) Energy bound, board-5 form: dE <= sum over blocked bonds of 2|H_xy| B_x B_y (cosh(de_x - de_y) - 1), estimated on
pool_s2b with its importance weights, per site; for the control AND for R1's theta_tail. Calibration: R1's value must come out of the order of its
measured 5e-5 (positive control); otherwise the bound is uncalibrated and the control is graded by FN after all. (3) theta_tail by hand over
the existing iterates (fit stopped by fit_sec at ~1225 of 1500), deviation recorded. Reading: control bound < 1e-6 per site -> R1's gain ATTRIBUTED
to the Jacobi target; otherwise as registered before (FN grade).

## R1-R3 ON THE RETRAINED 8x8 BASE (PI GO 2026-11-11; brief proposals/2026-11-11_R123_retrained_brief.md; REGISTERED before any run)
Base B_0' = rb8_retrain s306 params_final (sha 45fa54e6; real f 64x64 + complex RBM M 16; proj 'simple' = 64 translations x {E, R_pi/2} x flip
= 256 images; VMC -0.498838(2), replica s307 -0.498841(2); rb8_prod's VMC p=0 was -0.498674(12)). Sign fixed at the base's own sgn(Re w psi).
Rounds k = 1..3, branch-2 pipeline: pool from B_{k-1}, labels c = 4 f_J under the fixed sign, two-PP refit (R1's fitter and warm rule), bare c = 0
FN grade of B_k paired against B_0' through the same code. EARLY KILL: gain_k < 0.5 gain_{k-1}. Final headline (B_0' + p=1 vs B_3 + p=1, paired)
only if the rounds accumulate.
MECHANISM and predictions (mine, before any number):
- Round gain = stored fraction r x the round's Jacobi gain: dE_k = r_k G_k (6x6: r_5 0.49 fresh P_2, r_6 0.25 same P_2; q = 0.51).
- Round 1: G_1 is set by the base's sign-violating bond ratios. The better base (VMC 1.6e-4 below rb8_prod) has fewer bond-ratio errors, so I
  expect a smaller R1 gain than rb8_prod's -4.7e-5: R1' bare gain 1.5-3.5e-5 (~50%).
- Rounds 2-3 with the same two-PP class: the 6x6 storage ratio q = 0.51 sits ON the kill threshold (0.5). By that mechanism the kill fires or not
  at ~50/50, by design. The K=3 round-6 test (A40 17871864, running) measures whether one fresh PP per round restores storage (r >= 0.40).
  If it does, R2/R3 should add a fresh PP each round (K=3, K=4); if not (<= 0.30), the decline is in the fit and R2 is expected to hit the kill.
  So the class for R2 is decided by the K=3 readout, before R2's fit.
- Headline: the p=1 step overlapped with the refit at 8x8 (rb8_prod: 1/3 of the bare gain survived). If that carries over, a > 3-sigma final
  win needs a cumulative bare gain >= ~6e-5. I put it at ~25% (lead prior 35%).
FIRST CHECK (CPU): pipeline support for 'simple' (pp2_lemma has full/translations/none; pool_fit hard-codes 'translations' 128 images). Add
'simple' (rbmpp_lib.set_proj point indices 0, 2) to pp2_lemma, pool_fit (model, canonical keys, warm loader), gfmc_b2 guides, VMC evaluator;
identity test vs rbmpp_lib's evaluator on s306 (ln psi to 1e-12 on random Sz=0 configs; broken: drop the R_pi/2 images, must differ).
Cost x~2 per evaluation (256 vs 128 images). Pool re-sized by bond ESS under B_0' (gate as pool_s2b: bond ESS >= 3P per decade -2..-9).
Budget: ~10 H100-h to the round-3 bare grade (at risk with the 2x), ~20 for the headline, ceiling 40. December express + cont; A40 where sooner.
GPU steps go to the lead before submission.
Lead 2026-11-11 (board): 'simple' go at ~2x cost (ceiling 40 covers R1-R3 bare grades; headline cost back to the lead); kill stays 0.5; R2 class
from K=3 (>= 0.40 fresh PP per round; <= 0.30 run R2 as registered, expect kill). PI OK (lead session): production pp2_lemma multi-word dedup key
for > 64 sites + collision test (10x10, 14x14) + 8x8 bit-identity; low priority, after the 'simple' worker (both touch pp2_lemma).
Result 'simple' prep (2026-11-11; pfaffian_8x8_s306/prep/, code ansatz/code/simple/): (a) ln psi s306 vs training-code rbmpp_lib 2.4e-11 on 2000
configs (broken translations-only 7.2); (b) translations path bit-identical to originals; (c) BareM1Sign 14436 bonds / P1Guide 1718 bonds, 0 sign
diffs, rel <= 3.2e-10 (broken versions fail); (d) lambda = 0 warm start 1.3e-10 max (float64 slogdet floor; originals show 9e-11 on rb8_prod).
|B| is NOT C4-invariant under 'simple' (R_pi/2: max |d ln B| 6.6); dedup key stays translation x flip (largest valid). Cost x2.0 for labels, fit,
grade. Pool under B_0' (6000-draw pilot): decade origin med 2lnB -78.46; bond ESS >= 3P needs ~1.1e6 draws/rung at -9 (+-21%, 180x extrapolation),
3.7-5.7e5 at -2..-8. Decision: stage-1 = 3 rungs x 4.5e5 + 1e6 beta-0.25 deep expansion (pool_x2 conversion), gate; stop above 900 core-h.
Constants: w = -1 (Im ln psi = 0 mod pi on 2000 configs), E' = -0.498838 x 64.
s306 stage-1 pool: measured 1.79 core-h per 1000 draws per rung on cluster nodes (my 0.2-0.3 estimate was wrong); x3 3 rungs x 4.5e5 = 806 core-h released; beta-0.25 expansion resized to ~6.5e5 at thin 2 (~250 core-h), sized after the first 10 x3 tasks' deep bond ESS; stage-1 stop raised to 1100 core-h. Occupancy gate referenced to the pilot = self-consistency only.
Lead 2026-11-11: s306 pool ceiling 1.2e3 core-h (CPU, critical path); new j1j2 output under /project/theorie/a/A.Otaifi/runs/j1j2 (scratch near quota).
CORRECTION (2026-11-11): both "pending" Paderborn cont duplicates ran on 2026-11-10 (start estimates said 11-21/11-22). (i) 4186183 = prodB is
bit-identical to prodBx (same starts and seeds), so its auto-verdict "A - B -3.3 sigma" double-counts arm B and is WRONG; headline stays 2.7 sigma,
kill fires (README_DUPLICATE.md in headline_p1/prodB; gates must exclude prodB). (ii) 4182798 = the registered c=0 control fit, run to completion
(1500 steps, done, L_ho 0.046 -> 6.9e-9, 4.1 s/step, 1.74 H100-h cont): its round_1/theta_tail.npy is the registered control network; the A40 copy
(17852869, FAILED on fit_sec at ~1225) is superseded. Lesson: Paderborn cont start estimates were 11 days off; a duplicate staged with the same
seeds is not extra data.
Result R1 ATTRIBUTION CONTROL (2026-11-11; registered control = Paderborn cont 4182798 theta_tail sha f0ad918f, 1500 steps, L_ho 6.9e-9;
runs/j1j2/ctrl_c0_bound/{bound_R1,bound_ctrl,identity_R1,identity_ctrl}.json; verified totals): board-5 bound per site, pool_s2b weights,
covered mass 0.9996: R1 theta_tail 1.94e-4 +- 2.5e-5 (measured |E_1 - E_0| 4.95e-5: 3.9x, inside the 10x calibration limit, spread over
decades -1..-6); CONTROL 3.9e-12 +- 5.7e-13 (< 1e-6 by 5 orders). Bond differences control vs B_0, held-out median/99%/max 5.0e-6 / 1.4e-4 / 5.0e-3,
heavy 7.5e-5 / 9.0e-4 / 1.2e-2 (R1 heavy 0.52 / 2.2 / 4.2). READING (registered): ATTRIBUTED. The refit machinery without the Jacobi target
returns B_0 to ~1e-5 in bond ratios; R1's bare gain comes from the c = 4 target. Prediction (lead: B_0 +- 1e-5; mine: 0..+1e-5) HIT.
Caveat: the bound is an upper bound on |dE| (not the sign); the maxima are from 3200 + 600 sampled entries (lower bounds on the true maxima).
BASE ARM FN p=0 (lead, PI yes 11-11; REGISTERED 2026-11-12 before staging): B_0' = s306 ('simple', own sign sgn Re[w psi], w = -1), bare c = 0
FN-GFMC through the 'simple' copy of grade_bare (BareM1Sign), M 1024, dtau 0.1, >= 3 independent chains (distinct |B_0'|^2 starts from the s306
pool, seeds distinct from every earlier run), >= 256 post-burn blocks per chain, burn 50. Reused as the paired base arm of every R1-R3 bare grade.
Bar: Varbench -0.498963(2) (plain VMC). Lead prediction -0.49894..-0.49897. Mine: -0.49892..-0.49896 (FN p=0 sat 8.2e-5 below VMC p=0 for
rb8_prod: -0.498674 -> -0.4987565; s306 VMC -0.498838; a better guide gains less, so ~5-10e-5 -> -0.49889..-0.49894; I put the centre at
-0.49892, ~50%). Reading: below -0.498963 by >= 2 sigma at sigma <= 5e-6 -> 'below published'; else reported, p=1 decides at the headline.
Sizing from a measured rate; launcher + H100-h/A40-h to the lead before submission; counts in the R1-R3 40 H100-h ceiling; R1 has priority.
Lead 2026-11-12: (i) the sign angle builds a correlated-sampling PAIRED FN estimator (PLAN_sign T-STRONG): same-seed pairing in our GFMC decorrelates (T-NODE paired 3.8e-6 vs unpaired 3.6e-6), so the 11-10 headline's A-B 5.6e-6 was effectively unpaired; R1-R3/headline to use it once validated. (ii) Lead adopts my p=0 range -0.49892..-0.49896; if above -0.498963, next lever is the 'full'-projection guide (VMC ~7e-5 lower, ~4x images); rate full/simple to be priced by the base-arm worker.
Base arm staged (runs/j1j2/base_s306_p0): 4 chains x 650 blocks, M 1024, seeds 501-504, starts s306 stage-1 beta=1 out_3/15/32/39; ~0.85 H100-h (express 90 of 96 remaining min) or ~8-10 A40-h; CPU check 0 sign diffs / 5e-10; 'full'/'simple' cost 3.45x per walker-step (CPU). Waiting for the lead's OK.
Result BASE ARM FN p=0 (2026-11-12; Paderborn 4222088 express, 0.51 H100-h; runs/j1j2/base_s306_p0/prod/; verified chain means):
E_p0(s306) = -0.4988780(27) per site (4 chains x 600, QUOTABLE, chain chi2/dof 0.14), vs Varbench -0.498963(2): +8.5e-5 ABOVE (no). Only 4.0e-5
below s306's VMC -0.498838 (rb8_prod: 8.2e-5). My prediction (-0.49892..-0.49896) MISSED HIGH; the lead's adopted range too. Term that was wrong:
I scaled the FN-over-VMC gain by guide quality only mildly (5-10e-5); it halved. Variance 1.93e-6 per site^2 per walker-tau (rb8_prod 4.0e-6);
2.75 s/block at 4 concurrent. CAVEAT: all 4 chains have their second half lower than the first (-1.2, -1.4, -0.1, -1.3e-5; pooled -1.0e-5 +- ~5e-6,
~2 sigma, same sign 4/4): possible residual relaxation after a 5-tau burn; the converged value may be ~1e-5 lower (-0.49888x). Does not change
the reading. Next lever for the published bar (lead): 'full'-projection guide (VMC ~7e-5 lower; 3.45x cost) or p=1 at the headline.
REVISED R1' PREDICTION (2026-11-12, before any pool label or fit; after the base FN p=0): lead hypothesis: s306's remaining error is mostly
in its sign, which a fixed-sign round cannot touch. Mechanism for the revision: on rb8_prod the R1 bare gain was 4.7e-5 against an FN-over-VMC
gap of 8.2e-5 (ratio 0.57); the round stores part of the Jacobi correction, which approximates what FN does to the amplitudes near the fixed
nodes, so it scales with that gap. s306's gap is 4.0e-5 -> R1' bare gain ~0.57 x 4.0e-5 = 2.3e-5 at most; with the stored fraction lower for a
better-fitted base I predict 0.8-2.0e-5 (~50%). Kill (as for R1): R1' bare not below the base arm by > 3 sigma (sigma ~3-4e-6 paired-unpaired ->
needs >= ~1.1e-5). Disagreement recorded once: the base arm's halves drift -1.0e-5 (same sign 4/4, ~2 sigma); the lead reads no drift from the
fifths; R1'-R3' grades will use burn 150 blocks (15 tau) and a halves check.
HEADLINE B ARM (lead 2026-11-12): B_0' + p=1 FN ('simple', own constants by M1's VMC rule, seeds distinct), cont default QoS (express gone for
November), sized from a measured p=1 rate; launcher + H100-h to the lead first; R1 keeps priority.
R1' GRADE DESIGN (lead 2026-11-12, registered before the fit): base arm quoted as -0.4988780(27) with a possible bias up to ~1e-5 high (second
half -0.498883(~4e-6)); stays 'no'. Since ~1e-5 is the size of the expected R1' gain, the R1' grade must NOT be compared with the 50-burn base
arm: the base arm is RE-RUN with the same burn (150 blocks) and halves check IN THE SAME JOB as R1''s grade (~0.5 H100-h), new seeds and
starts; the 4222088 chains are not used in any R1'-R3' comparison. Same for R2'/R3' (base re-run in each job, or one long base arm shared if the
code and burn are identical).
Headline B arm staged (runs/j1j2/headline_s306): VMC-constants job ~1.85 H100-h (s306 has no p=1 constants) + FN p=1 4 x (150+400) blocks M 256 seeds 601-604, ~11.9 H100-h (78 s/block projected from CPU ratio); total ~13.7 (13.2-17.8); cont default QoS; waiting for the lead's OK. s306 pool x3 running (~78% at 2026-11-12 ~02:00).
s306 expansion sizing (2026-11-12; stage1/expansion_sizing.json, runs/j1j2/pfaffian_8x8_s306/ess_x3/): bond ESS ~ m^g, g 1.00 (-2..-7), 0.96 (-8), 0.91 (-9) (sublinear at depth, unlike pool_x2); full x3 projects -9 14.6e3, -8 30.8e3, -7 45.9e3 (-2..-6 pass); 3P everywhere (+20% at -9) needs ~2.4e6 beta-0.25 draws (fit; 1.7e6 linear); x3 measured ~594 core-h + expansion 702 (thin 2) = ~1.3e3 > 1.2e3; asked the lead for 1.35e3. Held 17903298 / 17901202 untouched.
s306 pool ceiling 1.35e3 core-h (lead 2026-11-12 08:50); expansion 2.4e6 beta-.25 thin 2 released; if -9 still misses 3P: report shortfall, no extension (deeper rung vs accept).

## STATE (2026-11-12 09:10) — running, no VERDICT yet
- s306 stage-1 pool (R1' critical path): x3 17901200/17901201 (theorie CPU, ~594 core-h) + expansion 17929342 (xe3, 40 x 60000 beta-0.25,
  thin 2, ~700 core-h); watcher watch_xe3.py (stage1/WATCH_XE3_VERDICT); end ~11-13 02:30-07:30. Held, untouched: 17903298, 17901202.
  Next: f_J labels (c = 4, sgn(B_0'), E' -0.498838 x 64) on the combined pool -> bond-ESS gate 3P per decade (-9 miss -> report, no extension)
  -> R1' fit (GPU, launcher to the lead) -> R1' grade with base re-run in the same job (burn 150, halves check).
- K=3 round-6 test (6x6): theorie A40 fit 17871864 (running since 11-10) + table 17871865 (afterok) + CPU readout chain; decides R2's class.
- Headline B arm (s306 + p=1): Paderborn cont VMC 4222358 -> FN 4222359 (afterok, 24 h limit); runs/j1j2/headline_s306/.
- Base arm FN p=0 s306 done: -0.4988780(27) (+8.5e-5 above Varbench; possible ~1e-5 high bias).
- Comms: session messages to the lead often fail (hook timeout); MESSAGE_BOARD_lead_pfaffian.md is the fallback (watch for '-> PFAFFIAN').
B arm (lead 2026-11-12 10:27, PI agreed): Paderborn VMC 4222358 cancelled by the lead; FN 4222359 held, dependency cleared. Constants come from the theorie A40 copy (diff sent to the lead); after its gate: install once, post sha, COPY consts_s306_p1.json to Paderborn headline_s306/inputs (the FN job reads it there), then scontrol release 4222359.
s306 x3 pool COMPLETE (17901200/17901201, 40/40 tasks, 450080 draws/rung, 604 core-h measured): gate PASS (halves max|z| 1.94, median -78.523 vs pilot -78.462, Sz 0 bad, recompute 0.0; occupancy = self-consistency only). A second, independent 10-task sizing (runs/j1j2/pfaffian_8x8_s306/ess10/) put -9 at 1.05-2.2e6 expansion draws (fit 1.43e6), inside the released 2.4e6 (17929342). Its watcher's delta VERDICT line (1808/1231 core-h) used stale rates and is superseded.
PI rule (11-12, via lead; LESSONS.md Operations): GPU placement chosen on the spot by fastest finish (an idle slower card beats a fast card later). s306 VMC moved to an idle a40-24gb slice; 17930563 to be HELD (not cancelled; cancels need the PI's word here).
R1' WARM-START RULE (lead 2026-11-12 13:35, registered before any R1' data): if the escape-time estimate (worker r1prime_prediction, part B) is ready before the R1' fit is submitted and shows the symmetric/small-lambda saddle taking a large fraction of 1500 steps (say > 30%), R1' is amended to the residual-fit warm start (P_2 = fit of P_1's residual at O(1) weight), registered before the fit; else the old rule (sep 0.03, t 0.05) stays. Part A gives what the old rule cost on rb8_prod.
R1' WARM-START DECISION (2026-11-12 ~17:00, before any R1' data; r1prime_prediction/B_main/saddle_first.json, job 17933867): linearised-GN
escape at s306's R1'-rule warm start (lam_init 0.00969, sep 0.0297) = 20-92 steps = 1.3-6.1% of 1500 (< 30%) -> the OLD RULE STAYS for R1'.
Curvatures: antisymmetric f1-f2 kappa 72-87 (~100-130 x damping 0.661), lam ~5e3 x damping (not slow); P_2's directions outside P_1's
tangent (Schur complement) mean 0.50 (0.76 x damping, 46 steps), along the warm noise 0.18 (92 steps). The binding fact is not the saddle:
94-96% of P_2's first GN update lies inside P_1's tangent (P_2 acts as an extra P_1 step), and the new directions carry 1.3% of the f2
gradient and 1.1% of the target loss (in-sample, ~4 rows/parameter, biased up). Caveats: GN is PSD (linear relaxation; second-order saddle
growth not modelled); R1's schedule raises lam_rel to 0.5-1 after ~step 300, so escape must happen early (it does, by this estimate).
RF warm start evaluated (r1prime_prediction/RF/rf_result.json, job 17933950) in the form 'lam = 1, f_2 = f_1 + 0.03 noise, 200-step f_2-only pre-fit': UNUSABLE (start loss 4.24/4.51 vs 0.569 for the R1' rule, since |P_2/P_1| ~ e^1.64 at lam = 1 breaks B ~ B_0; the linear pre-fit overfits: train 4.24 -> 1.96, held-out 4.51 -> 11.2). Not needed for R1' (rule kept). A corrected recipe (t = 0.05 lam, f_2 = f_1 + early-stopped f_2-block GN delta, accept iff held-out <= 0.569 and sep >= 0.1) is recorded, NOT evaluated; reconsidered before R2' after K=3.
Result K=3 ROUND-6 TEST (2026-11-12 17:11; exact 6x6 tables; fit A40 17871864, table 17871865, readout 17936914; pfaffian_r6k3/readout/;
controls theta_5 rel 0.0, theta sha local = cluster): bare e 5.8532e-6, r_6 = 0.4886 (K=2 0.2505; R5 fresh P_2 0.493), aug min 2.5456e-6
(c* 4.0; K=2 2.814e-6). READING (registered): RESTORES (>= 0.40): one fresh PP per round restores per-round storage. Predictions: architecture
angle ~0.45 (close), mine 0.33-0.48 (missed slightly high), aug 2.4-2.7e-6 HIT. Caveat (registered): K=3 has ~1.5x K=2's parameters; a fresh
PP vs more parameters not separated (M5 at round 5 is the nearest control: RBM count did not substitute). Fitted lam (0.145, -0.010),
mu (0.147, -0.002): both extra PPs end at comparable weight.
Consequence for R2'/R3' (registered rule): add a fresh PP per round. Note: the 6x6 fresh PPs were DISTINCT TRAINED PPs (sep 0.25); at 8x8 a
P_1 + noise start puts 94-96% of P_2's first update inside P_1's tangent. Proposal (mine) for R2''s fresh PP: s307 (the independently
trained replica of s306, same 'simple' class, VMC -0.498841) as P_3's start, mu by the median-0.05 rule; to be registered with its own
tangent-share check before R2''s fit.
R1' WARM START, second test (lead 2026-11-12 17:25, registered before the R1' launcher): predict r and bare gain for (i) P_1 + noise (registered) and (ii) P_2 = s307's PP (distinct trained optimum, RBM dropped, |lam P_2/P_1| median 0.05, phase so median Re(lam P_2/P_1) > 0); seal both. If (ii) is predicted clearly higher (outside A's calibration error), R1' uses (ii), registered before submission. Checks: s307 ln|P| same code path to 1e-10; lam = 0 reproduces B_0' exactly.
R1' PREDICTION, interim (2026-11-12 ~18:30; r1prime_prediction/B_main/tk_result.json): the linearised (deterministic GN) prediction FAILS its
own held-out test at s306 start (i) with the affordable fit set (49152 entries, ~8 rows/parameter, P 18467): held-out tangent share C
negative for every estimator (ridge 1e-6 -2.25 ... top-200 eigendirections -0.04), in-sample 0.96 (overfit); the linear curve predicts held-out
loss RISING 0.569 -> 1.5-2.5. Mechanism of the failure: the real fit is noise-dominated (fresh 8208-entry batches, ~1.3 rows/parameter,
trust-clipped; R1 had 97% 'noise' steps), not the population GN relaxation; a population estimate needs ~150 rows/parameter (Theta v2) =
~2.8M entries, the whole pool. Plan: A_main (rb8_prod, 65536 and 131072 entries) gives C_ho(N) at two N and the test vs R1's measured curve;
r_pred = (extrapolated population share) x (board-5 transfer factor calibrated on R1), LABELLED AN EXTRAPOLATION. Saddle numbers, dL/dlam and
the P_2-in-P_1 share are point quantities (G, b) and stay valid. s307 checks pass ((a) 4.8e-12, broken 6.3; (b) 0.0, broken 7.0); s307's f is
CLOSE to s306's (sep 0.147, overlap 0.989; P_2/P_1 same sign on 99.4% of held-out x), i.e. a nearby optimum, not a distinct one like the 6x6
fresh PPs (sep 0.25); lam 0.0372 real. Decision rule for R1' start unchanged: (ii) only if predicted clearly higher; with this prediction's
uncertainty the registered (i) is the default.
R1' WARM START FINAL (2026-11-12 ~19:00, before any R1' data; B_main + B2_main saddle_first.json, same 24576 entries): start (ii) P_2 = s307's f
is WEAKER than (i) on every valid first-order quantity: dL/dlam (Re) 0.052 vs 31.5 (log units 0.002 vs 0.30); new-direction (Schur) curvature
0.0097 vs 0.76 lam_d, escape 465 steps (31.0% of 1500, above the 30% rule) vs 46 (3.1%); eigenvalues above damping 4 vs 1469 of 8192;
new-direction target share 0.22% vs 1.1%; P_2's first update in P_1's tangent 98% vs 94-96%. Cause: s307's pair function is a near-copy of
s306's (overlap 0.989, sep 0.147, P_2/P_1 real), so lam P_2 is a rescaling of B absorbed by the offset c (corr -0.78), and Im lam / Im f_2
start at a GN frozen point. DECISION: R1' keeps the registered start (i). The "fresh PP" for R2' must be genuinely distinct (s307 is not);
to be designed before R2' (e.g. a PP trained on a different sector/projection, or the corrected residual-fit recipe), with these same point
quantities as the acceptance test.
MECHANISM HYPOTHESIS for two-PP storage (lead, ~40%; registered 2026-11-12 before any computation; Part C', 6x6 exact):
B = P_1 + lam P_2 with f_2 = f_1 + delta. The two-PP tangent at theta is span{J(f_1), J(f_1 + delta), d/dlam} with J = d ln P / d f, and
J(f_1 + delta) = J(f_1) + H(f_1)[delta, .] + O(delta^2), H = d^2 ln Pf = -(1/2) tr(A^-1 dA A^-1 dA) + tr(A^-1 d^2 A)/2. So the second PP adds
the Hessian-vector directions H[delta, e_k] (one config-space vector per parameter k, along the fit's own delta), which lie outside the single
PP's tangent (P16: single-PP held-out share 0-4%). Prediction: the round's stored fraction r ~ the share of the round correction inside
span{J(f_1), H(f_1)[delta_fit, .]} (and the exact two-PP tangent at the fitted point); at 6x6: R5 theta_5 (r 0.49) and K=3 theta_6 (r 0.49) vs
K=2 R6 (r 0.25). Kill: share at the fitted point < 0.2 when r = 0.49 (second-order directions do not explain storage). If it holds, the R2'
fresh-PP acceptance test becomes "does the start give second-order directions overlapping the target", not "is P_2 distinct".
Part C' method (2026-11-12): exact 6x6 shares cost ~100 core-h (15.8M orbits incl. partners); run as the SAMPLED Theta-v2 held-out estimator (P16 calibration 0.013-0.017 vs exact) at N/P >= 100, two orbit sets A->B / B->A, cap 12 core-h, after A_main. Kill unchanged.
Part C' sized by probe (17939519): 6x6 J costs 22/39 ms per config (576-image 'full' projection); full table at N/P 100 ~50 core-h: approved (cap 60), ordered R5 -> R6 K=2 -> K=3, stop early if R5 trips the kill; exact jvp for H[delta] + FD check.
R1' PREDICTION, calibration + CORRECTION (2026-11-12 ~21:15, timestamp corrected from '~22:30'; A_main 17933866, B_main, B2_main):
(1) The deterministic linear-GN curve fails calibration on R1: predicted held-out 0.262 -> 0.84-1.27, measured 0.262 -> 0.188 (fraction removed
0.281 +- 0.006). Not calibratable (N/P ~21; the real fit uses fresh batches from 1.6M entries). Step-0 loss reproduced exactly.
(2) Held-out tangent share (fit metric, ridge at lam_rel 0.5): R1 C(65k) -0.075, C(131k) +0.032 (HOB 0.101) -> extrapolated population share
0.14-0.17; measured fraction 0.281 -> calibration factor ~1.7-2.0 (nonlinear + minibatch part; two N, one seed).
(3) s306 at equal N (49k): start (i) noise: every estimator <= 0 (ridge 0.5 -0.40/-0.71, top-200 -0.04/-0.16); start (ii) s307: every
estimator > 0 (ridge 0.5 +0.32/+0.27, top-200 +0.21/+0.17); nested 24k -> 49k: (i) -0.71 -> -0.40, (ii) +0.18 -> +0.32; extrapolated: (i) ~0,
(ii) ~0.3-0.45. Start losses (held-out): (i) 0.569, (ii) 0.496: adding s307 at |lam P_2/P_1| = 0.05 removes 13% of the copy loss before any
step (error averaging: s307's errors partly independent of s306's). The earlier "(ii) weaker" reading used in-sample point quantities that
exclude the lam column; the held-out shares settle it. Caveats: s306 mini-pool 6000 entries/decade (train L0 0.12-0.34 vs held-out 0.50-0.60),
one seed, two N; B->A reverse negative for both (B only 8k).
R1' DESIGN FINAL (lead YES 2026-11-12 20:45; registered before any R1' data): warm start (ii) B = P_1 + lam P_2, P_2 = s307's f (RBM dropped),
|lam P_2/P_1| median 0.05, lam real (mean-phasor rule; lam 0.0372). Grade job: R1' arm + ATTRIBUTION ARM (the (ii) warm start itself, no fit)
+ base re-run, same code, burn 150, halves check. READING: the loop's gain is E(R1') - E(warm arm), NEVER E(R1') - E(base).
SEALED PREDICTIONS (formulas now; numbers inserted from the bound passes 17940673/17939321 before submission, no later):
(1) Warm arm vs base: dE_warm = -(fraction of held-out copy loss removed at the start) x G_s306, fraction = 1 - 0.496/0.569 = 0.128.
    Conversion (assumption: in the quadratic regime of board 5 the FN gain is proportional to the copy-loss fraction removed; R1 check:
    4.95e-5 / 0.281 = 1.76e-4 vs R1's board-5 bound 1.94e-4, consistent to 10%): G_rb8 = 1.76e-4 per site (measured gain / measured fraction),
    G_s306 = G_rb8 x proxy_s306 / proxy_rb8, proxy = board-5 bound of the target T vs the base on each pool.
(2) Loop storage R1' - warm: dE_loop = -(C_inf(ii) 0.3-0.45) x (R1 factor 1.7-2.0) x ... expressed as a copy-loss fraction removed by the
    fit beyond the start, times G_s306. LABELLED AN EXTRAPOLATION (one seed, two N, low-ESS mini-pool).
Note for R2': s307 is used; the fresh PP per round needs another source.
R1' SEALED NUMBERS (2026-11-12 ~21:40 (timestamp corrected; first written as 11-13 00:30 in error), before the pool labels, fit or grade; r1prime_prediction bound passes 17940673/17939321):
proxy (board-5 bound of T vs base, per site): rb8 1.905e-4(51), s306 1.011e-4(31) (control: R1 theta_tail bound reproduced to all digits).
G_rb8 = 4.95e-5/0.281 = 1.76e-4(27); G_s306 = G_rb8 x 0.531(22) = 0.935e-4(15).
(1) WARM ARM vs base: dE_warm = -0.128 x G_s306 = -1.2e-5 (range -1.0 .. -1.6e-5; -1.39e-5 on held-out set B).
(2) LOOP R1' - warm (EXTRAPOLATION: one seed, two N, low-ESS mini-pool, k from one R1 run): dE_loop = -f x 0.872 x G_s306 with
    f = min(1, C_inf(ii) 0.30-0.45 x k 1.7-2.0) = 0.51-0.90 -> -4.2 .. -7.3e-5 (with G's error -3.5 .. -8.5e-5).
TENSION, recorded before data: (1)+(2) = -5.4 .. -8.9e-5 is 3-4x my earlier R1' range 0.8-2.0e-5 (scaling heuristic: R1 gain = 0.57 x the
FN-over-VMC gap). That heuristic is not a bound (lattice FN at fixed sign depends on the guide's amplitudes on sign-violating bonds; nothing
caps it at the VMC-FN gap). The two routes name different terms: if R1' lands near 1-2e-5 total, the wrong term is G_s306 = G_rb8 x proxy
ratio (a bound ratio taken as a gain ratio); if near 5-9e-5, the gap-scaling heuristic is wrong. Proportionality on R1: copy-loss fraction
0.281 vs energy fraction 4.95e-5/proxy_rb8 = 0.260(40): ratio 1.08(17), consistent; board-5 residual fraction 0.128(100) only within its error.
Result PART C' (two-PP storage mechanism, 2026-11-12 22:31; SAMPLED Theta-v2, not exact; r1prime_prediction/Cprime/cprime_R5.json, reduce
17943843; 100-195 rows/parameter; calibration +-0.02): held-out share of the R5 correction at the fitted theta_5: (a) single PP -0.016, (b) +
H[delta] -0.001, random-delta control -0.049, (c) exact two-PP tangent +0.018; r_5 = 0.49. REGISTERED KILL FIRES (< 0.2): linear first- and
second-order directions AT THE FITTED POINT do not hold the stored correction. R6 not run. eps check: FD H[delta] -> exact jvp linearly.
DIAGNOSIS (before calling the hypothesis dead): the test measured tangents at the END point, while the hypothesis concerns the FINITE move
ln B(theta_5) - ln B(theta_4) = J Delta + (1/2) Delta.H.Delta + ...; storage may sit in the quadratic remainder of the finite move, which no
tangent at either end contains. Cheapest untried variant: decompose the actual move on held-out orbits into J(theta_4) Delta and the remainder,
and measure each part's share of the R5 correction (sampled, ~1 jvp per orbit, ~10 core-h). Not run; the mechanism of two-PP storage stays
NOT UNDERSTOOD (the tangent-share route, single PP 0-4% and now two-PP <= 0.02 at the end point, does not explain r 0.49).
Part C: fitted pairs overlap 0.997 (8x8) / 0.991 (6x6), same translation-averaged irreps (that average holds ~2% of ||f||^2), f_2 - f_1
d-enriched; |de_y - de_x| on R1 bonds follows the across-bond change of lam P_2/P_1 (Spearman 0.52 phase, 0.42 modulus).
Cost: r1prime_prediction ~95 core-h total (A/B ~59, C' ~36 stopped at R5). Reduce 17941240 pending DependencyNeverSatisfied (2 tasks died rc
132 on v2-only node th-cl-uv201): needs a PI-approved cancel.
Part C' answer (cprime.py L19/L58/L95/L65/L83/L94): the FULL R5 correction (ln T_5 - ln B(theta_4)) was projected, but J/H/two-PP tangent were evaluated at the FITTED theta_5, not the start. Start variant (shares at theta_4 + warm P_2) run at N/P 100 (~18 core-h; 10 core-h would only allow N/P 55, uncalibrated). Kill: start share < 0.2 with r 0.49 -> hypothesis dead.
Result PART C' START VARIANT (2026-11-12 23:05; SAMPLED Theta-v2, Cprime/cprime_R5start.json, jobs 17944127/17944128/17944129; full R5
correction at the round-5 warm start theta_4 + warm L-SWO P_2, lam 0.1, sep 0.085; 100-195 rows/parameter; +-0.02): best held-out share
(a) J(f_1) -0.03, (b) J(f_1) + H[delta_start] -0.04, (c) exact two-PP tangent +0.07 (other estimators -0.02..-0.11), random control <= -0.08;
in-sample 0.2-0.58 for every span incl. the control (overfit, nothing generalises); positive control: the same estimator reads +0.32 on s306
start (ii). REGISTERED KILL FIRES at the start too: the second-order (Hessian-vector) mechanism is DEAD as an explanation of two-PP storage.
What stands: r 0.49 is stored while no linear span at the start or at the fitted point holds > 0.07 of the correction on held-out orbits, so
the storage is a genuinely nonlinear (finite-move) effect of the fit, or the stored part is exactly the part a held-out tangent estimator
cannot see (high-dimensional, overfit-prone). Mechanism NOT UNDERSTOOD. Whole prediction step ~110 core-h. Note: the delta_today VERDICT line for
17944129 quotes the fitted-point numbers; the json is the source.
R1' SEAL ADDENDUM (2026-11-12 23:06, before any R1' data; not a change of the sealed numbers): the share-to-storage factor k is not a constant of the method: rb8_prod R1 k = 1.7-2.0 (population share 0.14-0.17 vs fraction 0.281); 6x6 R5 start share of the exact two-PP tangent +0.07(2) vs r 0.49 -> k ~ 7 (4-12). The R1' prediction uses rb8_prod's k; the R1' grade therefore also tests whether k transfers from rb8_prod to s306. If R1' - warm lands above the sealed range, the wrong term is the transferred k.
