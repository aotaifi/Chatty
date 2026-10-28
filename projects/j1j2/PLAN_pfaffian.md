# PLAN_pfaffian (angle lead "pfaffian angle", started 2026-10-27 by the ansatz angle; PI asked for it)

## Question
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
RUNNING (submitted 2026-10-28 ~02:50, Paderborn gpu_h100): real s1 4041269, s2 4041272, s3 4041274; complex s1 4041271,
s2 4041273, s3 4041275. Local dirs /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_{real,cplx}_s{1,2,3}; watchers
c1_pdb_watch.sh; worker waits for all six, pulls la_table/t_real, runs c1_mech.py. Next per verdict: PASS -> report to the
lead, design the size-only follow-up (board 21) against complex PP; KILL -> own node is not the lever, report; INCONCLUSIVE
-> state cause (unconverged run, scatter) and the cheapest fix. H100-h used: 0.02 + pending.

## Dead ends
none yet

## Cockpit
- STATUS (open, medium): Discussing with the PI whether a pair-product network can copy the fixed-node state in rare configurations.
- NEXT (open, medium): Agree on paper which Pfaffian variant to test and the test that would kill it.
