# Sign learnability vs N (FUTURE_ANGLES angle 2): progress log
Cluster run dir: ws1:~/chatty_signlearn/ (code/, data/N*/, runs/N*/, evals/N*/, logs/). Code: experiments/sign_learnability/.

## Design (2026-10-06)
Question: is learning the +-1 sign (a step function on configurations) a bottleneck for the stored-sign scheme, and how
does it scale with N? Exact amplitude a = |psi0| (ED), J2/J1 = 0.5, symmetric sector (k=0, A1, flip +), clusters
N = 16, 20, 24, 28, 32, 36 (same as sign_design_tests/b_depth; orbits D = 107, 2518, 15578, 3.6e5, 1.18e6, 1.58e7).
- Targets (exact on every orbit, stage A sl_prep.py): (i) Krylov step labels c_k = sgn[T_{k-1} - r_{k-1}], k = 1,2,3,
  from Marshall with the exact amplitude and exact s_{k-1} (energy-optimal T); (ii) the exact GS sign relative to
  Marshall, sigma0 = sgn(psi0) M. Both are exactly invariant under the full space group x spin flip.
- Learner (stage B sl_train.py, GPU): translation-invariant periodic CNN (gather 3x3 convolutions on the torus,
  residual GELU+LayerNorm, per-site head summed / sqrt N) = same family as stored_signs(_6x6); BCE on the logit
  (aux regression of asinh(T - r) as an ablation). Sizes XS 8x2, S 16x3, M 32x4, L 64x6 (channels x layers).
  Training orbits: i.i.d. tempered samples q(x) ~ |psi0(x)|^(2 beta), beta = 1, 0.5, 0.3, nested budgets
  n = 1e2..1e5 samples, + all one-hop (H-connected) orbits; uniform minibatches, random point-group/spin-flip
  augmentation; evaluation = logit averaged over point group x flip on EVERY orbit (exact errors, no sampling).
- Scoring (stage C sl_eval.py, CPU): w_lab (p0-weighted label error), w_net = w_s of stored sign (s_{k-1} * net, or
  Marshall * net) vs ED, w_hop/w_hop2 after one/two exact Krylov hops on top of the net (energy-optimal T),
  d_hop vs the exact chain s_{k+1}, repaired/residual/new error weight, decade-resolved errors (per-config
  |psi0|^2), margins delta = r - T and logit distributions of wrong vs all states, seen vs unseen orbits.

## Log
- 2026-10-06: stage A done for all N (job 16848168, <5 min each; exact chain reproduces b_depth: e.g. N28 w_s 4.87e-4,
  1.19e-5, 1.69e-7, 7.6e-10 for k=1..4). Orbit-level label balance is very different from p0-level: c_1 flips 32-62%
  of the ORBITS (N16..36) but only ~2% of the p0 weight; c_3 flips 1-17% of orbits at 1e-6..4e-5 weight.
  Coverage of the training set (beta 0.5, n=1e4 samples + neighbours): p0 weight seen = 1.0 (N<=24), 0.9994 (N28),
  0.991 (N32), 0.890 (N36); at N<=24 the training set is essentially the whole sector (memorisation regime).
- 2026-10-06: smoke tests OK (ws1 T1000 and 2080Ti job 16848176; N16 net+hop reproduces exact w 3.4e-4 -> 8.9e-7).
  Speed: M net 3.2 ms/step (2080Ti) / 4 ms (A40, incl. bookkeeping); full-sector symmetrised eval at N36 (15.8M
  orbits x 16 group copies) 180 s on A40. Main grid (61 runs per N, make_tasks.py): jobs 16848202 (N16-24),
  16848203 (N28,32) on 2080Ti; N36 on A40: 16848206/7 (ablations + capacity), 16848278-81 (fixed budget + sweep).
  Scoring: 16848249 (N28,32), 16848263 (N36), N16-24 on ws1.
- 2026-10-06: **N=28 complete (61 runs), first picture** (beta 0.5, n=1e4, M=28k params unless noted):
  | target | w_triv | w_lab (net) | w_net vs ED | w_hop | exact chain ref |
  |---|---|---|---|---|---|
  | c1 | 2.0e-2 | 1.2e-4 (L: 5.4e-5) | 4.6e-4 | 1.0e-5 | s1 4.9e-4, s2 1.19e-5 |
  | c2 | 4.9e-4 | 1.6e-4 (L: 8.4e-5) | 1.6e-4 | 3.8e-6 | s2 1.19e-5, s3 1.7e-7 |
  | c3 | 1.2e-5 | 1.2e-5 (= trivial) | 1.2e-5 | 1.5e-7 | s3 1.7e-7, s4 7.6e-10 |
  | gs | 2.0e-2 | 2.1e-6 (L: 1.2e-7) | 2.1e-6 | 7.2e-9 | 0 |
  | m2 = s2*M | 2.0e-2 | 1.2e-5 vs s2 | 7.8e-6 | 5.7e-8 | s2 1.19e-5 |
  | m3 = s3*M | 2.0e-2 | 2.2e-6 vs s3 | 2.1e-6 | 6.8e-9 | s3 1.7e-7 |
  (w_triv = weight of the -1 labels = Marshall error for c1/gs/m_k.) The exact GS sign (a step function too) is learned 1e4x below the Marshall
  error; the Krylov STEP FACTORS c_k get harder with k (c3 not learned at all); the full Krylov sign s_k*M is as easy
  as the GS sign. c1/c2 are capacity-limited (XS 0.7k / S 4.9k / M 28k / L 200k params: c1 w_lab 3.5e-3 / 6.1e-4 /
  1.3e-4 / 5.4e-5), not budget- or step-limited (n 1e3..1e5 flat; 4x longer training identical). One exact hop on top
  of the c1 net reproduces the exact 2-step chain (1.0e-5 vs 1.19e-5), but on top of c2 it stays 20x above s3.
  Neighbours are essential (gs n=1e4 without neighbours: 1.8e-4 instead of 2.1e-6); beta=1 is 8x worse than 0.5/0.3.
  -> added supplementary grid (make_tasks_supp.py: m2/m3 budget sweep + m2 L) jobs 16848334 (N<=32), 16848336 (N36).
- 2026-10-06: **N=32/36 first results: strong N dependence at fixed capacity.** (beta 0.5, n=1e4, M) w_lab/w_triv:
  gs 1.0e-4 (N28) -> 7.8e-3 (N32); c1 6e-3 -> 5.2e-2 (N32), 6.4e-2 (N36, aux run); c2 0.32 -> 0.96 -> 0.96 (not learned
  at N>=32); c3 = 1 (never). Capacity at N36 (n=3e4): gs XS/S/L w_lab 8.2e-3 / 8.5e-4 / 6.0e-6; c1 1.05e-2 / 2.5e-3 /
  5.7e-4 -> the exact sign is learnable with ~2e5 params, c1 saturates much more slowly. Decade-resolved: the
  wrong-label fraction is ~0 for per-configuration |psi0|^2 >~ 1e-9 and rises to 20-40% in the tail; as N grows the
  p0 weight moves into lower per-configuration decades (more configurations), i.e. into the hard region.
  c1 errors sit at the threshold: 57% (N28) / 31% (N32) of the error weight has |r - T| < 1, vs 0.7% / 3% of the flip
  weight. -> supplementary grid 2 (make_tasks_supp2.py): XL nets (128 ch x 6, 0.74M params) for gs/c1 at N24-36, L at
  n=1e5 and L with 4x training at N32/36. Jobs 16848417 (2080Ti, N24-32), 16848418/9 (A40, N36).
- 2026-10-06: N=32 grid complete, N=36 G1/G2 mostly complete. At fixed M capacity the error vs sample budget
  saturates by n ~ 1e3-3e3 samples for every N >= 24 (floor rises with N: gs 5e-6, 1e-4, 7e-3, 9e-3 rel. error at
  N 24, 28, 32, 36); c2 is not learned at N >= 32 (rel. 0.96), c3 never (1.00). One hop on the c1 net: w_hop / w_s(s2)
  = 1.0-1.6 (N28-36) -> still repairs; on c2: 50-140x above s3. BUT at N36 4x more optimizer steps (ep40) lowers gs
  1.7e-4 -> 4.5e-5 and c1 1.1e-3 -> 7.9e-4 (the N28 'ep40' control had only ~10% more steps because of min-steps 5000,
  so it did not test this). -> supplementary grid 3 (make_tasks_supp3.py): fixed 2e4 / 8e4 optimizer steps for
  gs, c1, m2 at N28-36 (jobs 16848682 2080Ti, 16848684 A40). N36 jobs moved partly to 2080Ti (16848573/4).
- 2026-10-06: steps matter: N28 M at n=1e4 with 5k / 2e4 / 8e4 Adam steps: gs rel. error 1.1e-4 / 1.9e-5 / 1.2e-5,
  c1 6.3e-3 / 2.8e-3 / 1.6e-3 (training error -> 0.1% / 1.6%). The default protocol (max(5000, 10 epochs)) is
  under-trained and gives N-dependent step counts (5k at N<=32, 6-15k at N36), which confounds the N comparison.
  -> **converged protocol** (make_tasks_conv.py): every run exactly 8e4 Adam steps, batch 1024: capacity S/M/L(/XL at
  N32, 36) for gs and c1 at n=3e4, budgets n=1e3/1e5 (+1e4 from supp3) for gs/c1 with M, targets c2/c3/m2 with M and
  c2 with L. Jobs 16848757-65 (A40 for N32/36, 2080Ti for N24/28); scoring 16848767 (N24-32, after 16848249), 16848263 (N36).
  The default-protocol grid stays as the 'fixed compute' reference.
- 2026-10-06: first 8e4-step results. M net, n=3e4, rel. error w_lab/w_triv: gs 1.1e-7 (N24), 4.2e-6 (N28), 8.8e-4 (N32),
  1.1e-3 (N36); c1 1.0e-6, 1.9e-3, 2.9e-2, 3.5e-2. Big jump N28 -> N32, small N32 -> N36. Caveat: at N<=28 the
  training set holds 58-97% of ALL orbits (memorisation regime), at N32 45%, at N36 9.5%. Seen/unseen split:
  per unit weight the unseen-orbit error is 10-100x the seen-orbit error (gs N36 M: 7.7e-6 seen, 5.1e-4 unseen),
  and at N36 both contribute comparably. Optimisation matters at every size: N36 L gs 1.47e4 -> 5.87e4 steps:
  rel 3.0e-4 -> 8.1e-5 (absolute w 1.6e-6). Started a shared worker pool (lock files in run dirs, sl_train.py) for
  the remaining 8e4-step runs: 16849286-90, 16849501.
- 2026-10-06/07: 8e4-step grid complete except the N36 XL nets. Results in README.md / tables_8e4.md. Highlights:
  gs relative error (M) 1.1e-7 / 4.2e-6 / 8.8e-4 / 1.1e-3 (N24/28/32/36); L 4.1e-8 / 1.0e-6 / 2.9e-5 / 6.3e-5; N32 XL
  6.9e-6 (w = 9.4e-8). c2 (L) 2.1e-4 / 9.1e-3 / 0.29 / 0.39; c3 (M) 3.1e-3 / 0.69 / 1.0 / 1.0, while m2 = s2*M (M)
  1.3e-6 / 3.5e-4 / 2.2e-3 / 3.7e-3 -> learn the composite sign, not the factor. m2 + one hop = exact s3
  (N36 6.7e-7 vs 6.6e-7); c2 + hop 2.8e-5. c1 net + hop matches/beats exact s2 at every N. Data: n=1e3 suffices
  at N>=32 (M); params for gs rel. 1e-4: <5k, ~7k, ~0.9e5, ~1.4e5. A40 pool workers cancelled (nothing left);
  remaining XL N36 runs on the 2080Ti pool workers.
- 2026-10-06 21:30: all 523 runs trained and scored; XL (0.74M, 8e4 steps): gs rel. 6.9e-6 (N32), 2.2e-5 (N36, w = 4.4e-7);
  c1 4.5e-3 (N32), 1.4e-2 (N36). Scoring jobs cancelled. Final summary.json / evals.tar.gz / figures / README written.
