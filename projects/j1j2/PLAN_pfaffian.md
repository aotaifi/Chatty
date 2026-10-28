# PLAN_pfaffian (angle lead "pfaffian angle", started 2026-10-27 by the ansatz angle; PI asked for it)

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
- STATUS (open, medium): Discussing with the PI whether a pair-product network can copy the fixed-node state in rare configurations.
- NEXT (open, medium): Agree on paper which Pfaffian variant to test and the test that would kill it.

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
