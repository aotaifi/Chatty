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
