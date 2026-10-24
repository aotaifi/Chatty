# Ansatz-pioneer angle (J1-J2, J2/J1 = 0.5): closed 2026-10-24
Files: projects/j1j2/PLAN_ansatz.md (pre-registrations, dead ends with run paths), projects/j1j2/ansatz/ANSATZ_MEMO.md
(result), ansatz/lit/{LANDSCAPE,NOVELTY}.md (literature), ansatz/REVIEW_adversarial.md, ansatz/code/ (all tools).

## What was built
- Exact small-lattice test benches: 4x4/4x6 Sz=0 (small_pretests.py, c2_t1_small.py), an exact full-sum variational
  optimiser at 4x6 in JAX (exact_vmc.py), 6x6 orbit-sector J1/J2 bond operators verified to 2.7e-16 (sector_ops.py,
  neighbour tables in /scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables), parton/vison spans (c3_*.py), parent-
  Hamiltonian inversion (c6_t0*.py), non-unitary similarity sign exponent (c6b_t0.py).
- Nine candidate ansatze, each with a pre-registered kill test; all dead or closed.

## Why
The PI asked for an ansatz that captures frustration physics (sign, near-node region, criticality), serves the FN
loop, and scales. The loop's bottleneck is storing the near-node tail (93% of the refit tax).

## Rejected, and why (deciding numbers in PLAN_ansatz.md "Dead ends")
- Global sign rules (Neel/stripe interference): violations are class-blind (R/R_null 0.99).
- J2 layers (add-on, trained, stored pre-image in the loop): the full-H Lanczos layer wins; the stored pre-image is
  node-free only at cond >= 30, where the two-hop tax is 5.7e-5 vs bar 1.66e-6.
- Z2 spin liquid + vison sums: random no-flux perturbations beat vison insertions 1.5-2.3x at equal rank (4x6).
- Plaquette frame, sign-free parent + J2 dressing, J2-gate similarity: no gain over Marshall/baselines.
- Coupled amplitude-sign output: ratios ~1 at 4x6, no pass in reach.

## Three things to understand
1. The Marshall-violation fraction rises with decreasing amplitude to ~1/2 in the deep tail: the node runs through
   the tail, and the sign there is set by cancellation, not by any rule.
2. Undoing one J2 exchange gives an exactly Marshall-positive pre-image (4x4 at moderate layers; 6x6 only near
   singular layers), but every static use of it failed: the sign problem lives in the dynamics, not in one state.
3. Black-box stored networks do capture the sign (CNN + MPS, arXiv:2603.14425, 2.3e-6/site at 6x6). The negative
   result is about the physics-structured forms tested here; the loop should try that network as its stored part.
