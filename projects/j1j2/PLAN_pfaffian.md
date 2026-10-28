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
