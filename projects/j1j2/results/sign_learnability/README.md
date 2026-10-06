# Is learning the sign (a +-1 step function) a bottleneck? Scaling with N (FUTURE_ANGLES angle 2)

J1-J2, J2/J1 = 0.5, symmetric sector (k=0, A1, spin flip +), **exact ED amplitude** a = |psi0|, clusters
N = 16, 20, 24, 28, 32, 36 (orbits D = 107 ... 1.58e7; same clusters as `sign_design_tests/b_depth`).
Code: `experiments/sign_learnability/`. Dated log: `PROGRESS.md`. Run dir: `ws1:~/chatty_signlearn/`.

## Verdict
**The step-function shape is not the bottleneck. What a net can or cannot learn depends on which sign function it
is asked to learn, and the cost of learning it grows with N mainly through capacity and optimisation, not data.**

1. **The exact ground-state sign is learnable.** It is a +-1 function too, and a 0.2M-parameter periodic CNN
   learns it to w = 1.2e-6 at N=36 (L net; Marshall error 2.0e-2). An XL net (0.74M params) reaches w = 9.4e-8 at N=32 and 4.4e-7 at N=36.
   The errors sit only in the low-|psi0| tail (per-configuration |psi0|^2 below ~1e-9 at N=32-36). Configurations
   with large weight are never wrong. So the net learns where the sign changes; it does not just memorise values.
2. **Krylov step factors c_k with k >= 2 are the hard targets.** c_k = s_k/s_{k-1} encodes where the previous step
   was wrong, and that set sits right at the previous threshold. The full sign is easy at every depth:
   with an M net (28k params), n=3e4 and 8e4 steps, the relative error w_net/w_triv at N=36 is
   gs 1.1e-3, m2 = s2*M 3.7e-3, c1 3.5e-2, c2 0.86, c3 1.00 (c3 is not learned at all). With an L net: c2 0.39.
   **Practical rule for the stored-sign scheme:** train each new net on the composite sign s_hat_{k-1}*c_k
   (relative to Marshall). Do not stack factor nets. The composite plus one exact hop matches the exact chain
   one step further (N36: m2 + hop w = 6.7e-7, exact s3 = 6.6e-7). The factor c2 plus a hop is 40x worse
   (2.8e-5).
3. **For c1, the errors sit on the decision boundary.** 50-77% of the c1 error weight has |r0 - T0| < 1, a band
   that holds only 0.05-0.14% of the |psi0|^2 weight (median |r0 - T0| ~ 3). 51-63% of the error weight (N28-36) also has
   |logit| < 1, where only <= 0.1% of all weight lies. So the boundary is sharp almost everywhere, and the net
   is uncertain exactly where it is wrong. The exact-sign errors do not follow r0 - T0 (median |r0 - T0| ~ 10 at
   the errors); they follow amplitude, i.e. they sit in the tail.
4. **Scaling with N.** Fixed protocol: 8e4 Adam steps, n = 3e4 tempered samples + neighbours.
   - At **fixed capacity** the error grows steeply up to N=32 and then flattens. gs relative error with the M net:
     1.1e-7, 4.2e-6, 8.8e-4, 1.1e-3 (N = 24, 28, 32, 36). c1: 1.0e-6, 1.9e-3, 2.9e-2, 3.5e-2. N <= 28 is a
     memorisation regime: the training set holds 58-97% of all orbits there, against 45% at N32 and 9.5% at N36.
     So only the step from 32 to 36 measures real generalisation, and over that step the error grows 1.2-2.2x
     while the Hilbert space grows 13x.
   - **Parameters needed** for gs relative error <= 1e-4: < 5k (N24), ~7k (N28), ~0.9e5 (N32), ~1.4e5 (N36).
     For gs <= 1e-3: < 5k, < 5k, 26k, 30k. For c1 <= 3e-2: < 5k, < 5k, 27k, 48k. The error falls about
     1.5 decades per decade of parameters for gs at N = 32-36, but only ~0.35 for c1 (N36: S/M/L/XL
     8.3e-2 / 3.5e-2 / 2.0e-2 / 1.4e-2). c1 <= 1e-2 needs ~0.5M parameters at N32 (XL: 4.5e-3) and > 0.74M at N36.
   - **Data is not the limit.** With the M net, n = 1e3 tempered samples (+ neighbours) are already within 2x
     of the best error at N >= 32. Going from 1e4 to 1e5 samples changes nothing (gs N36: 1.2e-3 vs 1.2e-3).
     More data only helps where the net can memorise (N <= 28). One-hop neighbours are essential: without them
     the gs error is 3-80x worse. beta = 0.5 and 0.3 are equivalent; beta = 1 is 1.4-8x worse.
   - **Optimisation matters at every size.** The error keeps falling with steps (gs N32, M: 7.8e-3 / 2.0e-3 /
     8.4e-4 at 5e3 / 2e4 / 8e4 steps; gs N36, L: 3.0e-4 / 8.1e-5 / 6.3e-5 at 1.5e4 / 5.9e4 / 8e4 steps). Short
     schedules overstate the difficulty.
5. **One exact hop keeps repairing net errors as N grows.** w_hop/w_net is ~1-4e-2 for c1, c2 and m2, the same
   contraction as an exact Krylov step (s1 -> s2: 2.7e-2, 2.4e-2, 2.1e-2, 4.0e-2). For gs nets it is 1e-3 to 6e-3 (N28-36).
   - **c1:** the hop removes 96-99% of the error weight, and c1 net + hop matches or beats the exact two-step chain at
     every N (N36 M: 3.3e-5 vs 4.4e-5; N32: 1.2e-5 vs 1.4e-5). The repair hardly depends on c1's label accuracy: at
     N36 an L net with 0.6x the label error gives about the same w_hop (3.9e-5 vs 3.3e-5).
   - **gs:** gs net + one hop reaches w = 1.4e-7 (M) and 2.2e-9 (L) at N36.
   - **Caveat:** the repair works only if the stored sign is close to a Krylov iterate, which is what the
     composite target gives.

Bottom line for the stored-sign scheme: storing the sign in a net costs O(1) per evaluation and, with one exact
hop, keeps the accuracy of the exact chain up to N=36. The requirement is to learn the composite sign, not
the step factors. The capacity and training compute needed for a fixed accuracy grow with N. Between N=32 and
N=36 that growth is modest (about 1.5x the parameters for gs at 1e-4). The two large clusters are not enough to
fix a rate, and an 8x8 test is the natural next point. The FN amplitude (angle 3) remains the main bottleneck.

## Setup
- Targets, all exact on every orbit (`sl_prep.py`, CPU):
  - c_k = sgn[T_{k-1} - r_{k-1}] for k = 1, 2, 3, from Marshall with the exact amplitude, given the exact
    s_{k-1} and the energy-optimal T. The chain reproduces `b_depth_N*.json`.
  - gs = sgn(psi0) * Marshall.
  - m2, m3 = full Krylov signs s_k * Marshall (added after the first results).

  All targets are exactly invariant under the space group x spin flip.
- Learner (`sl_train.py`, GPU, JAX), the same family as `stored_signs` / `stored_signs_6x6`:
  - Translation-invariant periodic CNN: 3x3 gather convolutions on the torus, residual GELU + LayerNorm blocks,
    and a per-site head summed over sites and divided by sqrt N.
  - Sizes, as channels x layers: XS 8x2 (0.7k params), S 16x3 (4.9k), M 32x4 (28k), L 64x6 (186k),
    XL 128x6 (0.74M).
  - Loss: BCE on the logit. Aux regression of asinh(T - r) was tested and gives no gain (N28-36, within 10%).
  - Training set: i.i.d. tempered orbit samples q ~ |psi0(x)|^(2 beta) plus all H-connected orbits, uniform
    minibatches of 1024, random point-group element and spin flip per sample.
  - Evaluation: logit averaged over point group x flip on **every** orbit, so all errors are exact, with no
    sampling noise.
- Scoring (`sl_eval.py`, CPU):
  - w_lab is the |psi0|^2-weighted label error; w_triv is the error of the no-flip predictor.
  - w_net and w_hop are sign errors against ED: of the stored sign itself, and after one exact energy-optimal
    Krylov hop on top of it (also a second hop).
  - Also recorded: repaired / residual / new error weight, errors by decade of |psi0|^2, margins r - T,
    logit histograms, and seen vs unseen orbits.
- Protocols:
  - (i) "default": max(5000, 10 epochs) steps. This is the full grid of 61 runs per N, incl. beta, no-nbr, aux
    and seed ablations (`make_tasks.py`, `make_tasks_supp*.py`).
  - (ii) **8e4 steps** for the scaling statements (`make_tasks_conv.py`), added because (i) is under-trained
    and gives N-dependent step counts.

## Key numbers (8e4 steps, n = 3e4, M net; full tables in `tables_8e4.md`)
| N | gs w_net | gs+hop | m2 w_net | m2+hop | c1 label err | c1+hop | exact s2 | c2 rel. | c3 rel. |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 2.0e-9 | 1.3e-14 | 1.4e-5 | 1.0e-7 | 1.8e-8 | 1.4e-5 | 1.4e-5 | 2.2e-4 | 3.1e-3 |
| 28 | 8.2e-8 | 7.8e-11 | 7.3e-6 | 5.9e-8 | 3.8e-5 | 1.1e-5 | 1.2e-5 | 0.16 | 0.69 |
| 32 | 1.2e-5 | 3.9e-8 | 2.4e-5 | 1.4e-7 | 3.9e-4 | 1.2e-5 | 1.4e-5 | 0.81 | 1.00 |
| 36 | 2.2e-5 | 1.4e-7 | 5.2e-5 | 6.7e-7 | 6.8e-4 | 3.3e-5 | 4.4e-5 | 0.86 | 1.00 |

How to read the table:
- The w_net and +hop columns are sign errors against ED.
- "c1 label err" is the weight where the net's c1 disagrees with the exact c1.
- The exact chain's own errors, for comparison: s1 = 5.1e-4, 4.9e-4, 6.5e-4, 1.1e-3 and s3 = 1.1e-7, 1.7e-7,
  9.2e-8, 6.6e-7 (N = 24, 28, 32, 36).
- N <= 20 is learned exactly by every target and size >= S.

Capacity (gs relative error, n = 3e4, 8e4 steps): N32 S/M/L/XL 1.6e-2 / 8.8e-4 / 2.9e-5 / 6.9e-6; N36 S/M/L/XL
2.2e-2 / 1.1e-3 / 6.3e-5 / 2.2e-5 (XL w = 4.4e-7).

## Figures
- `fig_sign_learnability_scaling.{png,pdf}`:
  - (a) relative label error vs N for gs, m2, c1, c2, c3.
  - (b) w_hop/w_net vs N, with the exact-chain contraction for comparison.
  - (c) error vs tempered samples.
  - (d) error vs parameters (gs filled, c1 open, colour = N).
- `fig_sign_learnability_mechanism.{png,pdf}`:
  - (a) wrong-weight fraction per decade of per-configuration |psi0|^2.
  - (b) cumulative c1 error weight vs |r0 - T0| compared with all states.
  - (c) error vs Adam steps.

## Files
- `summary.json`: one scalar record per run (523 runs) plus decade/margin/logit histograms for the reference
  runs and the exact-chain metadata per N. `evals.tar.gz`: all per-run eval JSONs.
- `tables_8e4.md`: tables generated by `make_figures.py`.
- Cost: ~21 GPU-h (RTX 2080 Ti / A40) for 523 runs, of which the 8e4-step L/XL nets at N36 take 0.5-1.2 h
  each. Stage A (exact data) takes < 5 CPU-min per N. Scoring a run takes ~1 min at N36 (two exact hops on
  1.58e7 orbits). The scoring jobs held 16-core nodes for ~20 h in total, mostly while polling.
