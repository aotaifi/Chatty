# AVSL experiment 01 — discovery and verifiable *immediate* reward

Date: 2026-10-18. Status: exploratory 4x4 test; NOT an RL-trained neural network or a scaling result.

## Question
Can a cheap local-bond acquisition policy find more *frozen-amplitude* energy-lowering sign changes per unique guide evaluation than uniform candidate selection? Crucially, what if there is no immediately rewarding action to discover?

## Controlled setup
Periodic square 4x4 spin-1/2 J1-J2, J2/J1=0.5, Sz=0 dimension 12,870. Hamiltonian has 32 nearest-neighbor and 32 next-nearest-neighbor bonds. Marshall signs. A simple **callable analytic amplitude** a(x)=exp[-kappa H_xx] (not exact target/FN ED amplitudes) is evaluated only on demand during exploration. Symmetry orbits use 16 translations and global spin inversion (441 representatives). Seed configurations from a local Metropolis sampler targeting amplitude a (tempered distribution). One-hop frontier and *observed signed couplings* provide candidate-orbit priorities. The same candidate pool, 40 verified orbits, at most 12 greedily accepted orbit flips, and query-count accounting are used across arms. Every accepted orbit decision recomputes interacting marginal gains. Independent full Hamiltonian expectation checks are only post-hoc.

Arms: random priority, mean observed signed coupling priority, or maximum individual frustrating-bond priority. The energy gain `G=E_before-E_after` is positive for improvement. All stated gains are *unnormalized H numerator decreases* (normalization constant per guide); normalize by sum a(x)^2 for variational energy units. Sign-learning NN, learned amplitude, on-policy policy-gradient and new FN solves are **not included**.

## First run, BEFORE active RL training
- For kappa = 0.32, an independent complete enumeration of the 441 orbit representatives shows **0** favorable single-orbit flips. None of the three policies obtains a positive immediate reward over six seeds. This is a *reward sparsity / no-bootstrap* warning, not evidence that Marshall has exact ground-state signs.
- For deliberately steeper kappa = 0.9, 23 / 441 orbits have individually positive gains under the callable analytic guide. Twelve random-seed trials, 40 queried candidate orbits:
  - random: mean unnormalized gain **70.0740**, mean unique guide queries **4804.9**;
  - mean observed signed bond: mean gain **136.2153**, mean unique queries **4775.2**, better than random in **10/12** trials;
  - max one frustrating bond: gain **0**, mean unique queries **4613.2**.
- Analytic-guide squared norm at kappa 0.9 = **343068.1392**, so mean normalized energy improvement is **0.00020426** (random) versus **0.00039705** (observed-signed).
- All accepted changes were verified independently by direct full 4x4 Hamiltonian expectation differences; the discrepancy on trial 101 was at most **1.24e-7 in unnormalized energy numerator** (roundoff on subtraction of extensive sums). The 0.32 run has norm 28416.5252 and energy -6.45533472.
- Frontier exploration itself queried about 4,094 configurations at kappa 0.9 (one-third of all states), far too expensive to demonstrate good scaling even though the additional selected-orbit verification cost was bounded. This is a **pipeline sanity check**, not an efficient new exploration method.

## Interpretation
(A) Active reward prioritization can outperform random choice when profitable proposals already exist and a local proxy is informative. (B) A no-reward basin exists for this analytic guide under one-orbit greedy corrections; simply applying RL to this reward may get stuck even though other joint/sequential/FN-updated sign changes could help. (C) A single largest violated bond is not a reliable reward proxy: collective contributions matter. (D) The 4x4 toy does not yet exercise a sign-network policy or FN walkers and has not passed a meaningful large-size sampling efficiency test.

## Follow-up, only after reviewing this result
1. Use actual FN walkers (128) and an *on-demand learned* guide to propose a fixed-budget frontier, rather than global precomputed exact FN amplitudes.
2. Compare discovery-only (equal proposal/query) to discovery + verifier. Record rejected/false-positive corrections, include true-energy audits.
3. For the 0-reward basin, compare multi-orbit/short-horizon rewards, amplitude refresh, and a continuation curriculum to see whether longer-horizon outcomes escape the trap.
4. Only if candidate discovery becomes feasible, train a **policy over which candidates to query/verify** (contextual bandit). Reward net `E_before-E_after` should be measured on accepted *whole* updates, never blindly assigned to every constituent flip. Build a second network for the composite updated sign function, with replay and no-flip anchors.
5. Prohibit reward leakage: the action policy receives only cached queried features; its verifier must be counted and evaluated independently.

## Reproduction
`discovery_probe4.py` and `validate_probe4.py`; outputs `probe4_k09.json`, `probe4_k032.json`. Run `python discovery_probe4.py --kappa .9 --seeds 12 --n-seeds 180 --B 40 --out probe4_k09.json`; analogous with .32, 6 seeds. Validation uses full 4x4 enumeration ONLY after acquisition.
