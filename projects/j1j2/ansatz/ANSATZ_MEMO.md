# Ansatz pioneer: what captures frustration in the J1-J2 ground state (J2/J1 = 0.5)
Angle lead, 2026-10-23/24; reopened by the PI 2026-10-26 and closed again 2026-10-27 (section "Reopened"). Revised after two
independent reviews (REVIEW_adversarial.md; Codex gpt-6-astra, four debate rounds, REVIEW_astra_2026-10-26.md) and their re-tests. Plan, pre-registrations and dead ends: ../PLAN_ansatz.md.

## Bottom line
None of the physics-structured ansatze tested here captures the sign of the frustrated ground state better than
the baselines: global sign rules, Z2 partons dressed by visons, rotated local frames, an unfrustrated parent,
stoquastic non-unitary similarities, and J2 layers (add-on, trained, or as a stored pre-image in the FN loop).
An independent review found three tests unfair (C3 vison sum, C6 parent, C5 coupled output); the fixed re-tests
confirmed C3 and C6 dead (vison span 1.55-2.29x worse than the repaired control; best sign-free parent 2.45x worse
than the baseline) and left C5 without a pass in reach (ratios 0.65 / 1.02 / 0.96, bar 0.5). Black-box learned signs do capture it (CNN + MPS,
2.3e-6/site at 6x6, arXiv:2603.14425), so the claim is about these structures, not about stored signs in general.
Exact facts that survive: (i) the Marshall-violating fraction rises to ~1/2 in the deep tail; (ii) a short
imaginary-time filter makes the state nearly sign-free at a strength that is the same at 4x4 and 4x6 (exp(+t H_J2),
t* in (0.14, 0.16]); it is not J2-specific (a J1 filter at matched fidelity is only 1.2-2.2x worse; the two agree to
first order, H_J2 psi0 = (E0 - H_J1) psi0). (iii) In the FN loop at 6x6 the exactly stored pre-image is excellent
(two-hop tax 2.4e-8 at t = 0.20, bar 1.66e-6), but a 2442-parameter positive network storing it is not (7.4e-6).

## Landscape (details and evidence: lit/LANDSCAPE.md; novelty checks: lit/NOVELTY.md)
| family | builds in | structurally cannot / measured | best 6x6 (per site above ED unless rel.) |
|---|---|---|---|
| Jastrow + Marshall | sign of the unfrustrated limit | any sign change; Marshall-sign accuracy 0.980 at 6x6 (2502.01264) | - |
| projected partons / RVB + Lanczos | spinons, PSG, Dirac nodes | nodes of one mean-field determinant | p=2: rel 4.7e-4 (1304.2630) |
| pair product + RBM | singlet pairing, projections | sign = Pfaffian x phase | rel 8.9e-5 (2005.14142) |
| CNN/ViT/ResNet NQS | symmetry, depth, attention; usually Marshall prior | Marshall trap; supervised sign accuracy falls with L | - |
| CNN + MPS (nuTNS) | CNN disentangler + MPS backflow | cost not stated | 2.3e-6 (2603.14425; checked in text) |
| Lanczos/Krylov-boosted | H-generated nodes | (zN)^p cost; NN fits of Lanczos states underfit at L >= 6 | RBM p=2 rel 6e-5 (2206.14307) |
Gaps that no family fills natively (searched): an explicit model of the nodal region; a non-Marshall sign rule at
J2 ~ 0.5; PSG inside NQS; sums over flux sectors.

## What the exact data say about the sign (all exact, J2/J1 = 0.5)
1. Violations are local and sit in the tail. Marshall-violation fraction by probability decade at 6x6: 0.001 (-7),
   0.11 (-10), 0.44 (-12), 0.54-0.57 (-13..-15). In the deep tail the sign is uncorrelated with Marshall.
2. No global sign rule carries information: the violation fraction is the same in every (Neel, stripe-x, stripe-y)
   parity class at every decade (R/R_null = 0.99). These are the only symmetric product rules.
3. Undoing the J2 exchange removes the violations. W_M(Phi)/W_M(psi0) at kernel cond 3: 0.015 (4x4), 0.13 (4x6),
   0.24 (6x6) for the J2 hop; J1 hops 2-3x weaker. The additive layer's grip fades with N (extensive local energy).
4. That pre-image is not the ground state of a few-body stoquastic Hamiltonian (4x6: min-variance stoquastic parent
   drops all same-sublattice XY, fidelity 0), and the similarity it defines is not stoquastic (sign exponent rises
   from 0.383 to 0.497/site where Phi is already exactly positive). The sign problem lives in the dynamics.

## Candidates (all pre-registered; numbers per site above E0)
| id | idea | novelty | decisive result | verdict |
|---|---|---|---|---|
| C1 | psi = s_N A + s_X B + s_Y C, positive parts | partial (2508.09870 uses the patterns as features) | violations class-blind, R/R_null 0.99 | dead |
| C2 add-on | (1 + a H_J2) on existing sign-free states | Sorella 2001 family | full-H layer better 4.5x/5.5x/1x (4x4) | dead |
| C2-T2 | trained base + J2 layer (4x6) | Sorella 2001 family | J2 layer worse than full-H: 1.58x (Jastrow), 1.26x (RBM) | dead |
| C3 | Z2 spin liquid + vison-pair sum | open part: learned local flux sums | 4x6, repaired base, rank 24: vison 6.4e-4 vs random no-flux 2.8-4.1e-4 | dead |
| C4 | positive state in the 2x2 crossed-plaquette frame | Herz et al. 2607.24679 (ladders) | no rotation beats Marshall (4x4) | dead |
| C5 | amplitude-sign coupled output e^f tanh(g) | generic | continued 3 seeds: ratios 0.65/1.02/0.96 vs standard (4x6) | closed |
| C6 | psi0 = (1 + a H_J2) GS(H_p), H_p stoquastic | novel construction | couplings optimised for dressed energy: 1.02e-2 vs baseline 4.17e-3 (4x4) | dead |
| C6b | 2-site J2 gates as non-unitary similarity | novel | sign exponent never below H's | dead |
| C2-L | store the Marshall-positive J2 pre-image in the FN loop | loop-specific | exact store 2.5e-7; fitted 5.7e-5 (positive RBM), 3.4e-3 (RBM+PP); bar 1.66e-6 | dead |
| C2-L-exp | same, exponential form exp(+t H_J2) | loop-specific | exact store 2.4e-8 (t 0.20); fitted positive RBM 7.4e-6; bar 1.66e-6 | dead (this network) |
| R3 | O(N) local rebuild, product of one-bond propagators | generic | E_FN 1.34x / 1.80x Marshall-FN at t 0.16 (4x4 / 4x6) | dead |

## Reopened 2026-10-26/27: judged inside the FN loop (PI's bar), sorted as he asked
| class | finding | number (exact; runs under /scratch/a/A.Otaifi/ansatz_runs/) |
|---|---|---|
| insight | the frustrated sign is recovered from a nearly sign-free state by a short filter, given exact magnitudes | 4x6: Marshall sign on exp(+0.16 H_J2) psi0, rebuilt: E_FN 1.45e-5 vs Marshall-FN 6.95e-3 (r2/); 6x6 loop target: exact store two-hop tax 2.4e-8 at t 0.20 (c2le_6x6/run_17510897) |
| hint | the filter strength needed does not grow from 16 to 24 sites | t* (0.14, 0.16] at 4x4 and 4x6; ratio curves overlay within 25% (recheck/); both lattices width 4, so not proof of locality |
| failure | J2-specific physics reading | matched-fidelity J1/J2 violating weight 1.2-2.2 (bar 3), first-order identity with J1 (r2/) |
| failure | storing the pre-image in a network of the baseline's size | positive RBM 2442 params, t 0.20: tax_1 2.58e-5, tax_2 7.40e-6 vs baseline 5.11e-6 / 1.657e-6 (c2le_6x6/run_17516512); oracle residual sign: same (7.99e-6) |
| failure | cheap rebuild at a configuration | sampled: C = E[eta^-2] 3.4 -> 5.0 (16 -> 24 sites; exponential vs linear undecided); local decoders: additive loses grip with N, independent-bond product divides by near-zero amplitudes (E_FN 1.3-1.8x Marshall-FN, r3/) |
Why the fit fails (hypothesis, not isolated): the rebuild is a cancellation near small amplitudes, so storage errors grow
there (centered log error by |phi_0|^2 decade, fit -> rebuilt: 0.139 -> 0.614 at -13, 0.263 -> 1.041 at -15); before hops
the decoded fit is 4.2x worse than the same error put directly on phi_0 (E_FN 1.68e-4 vs 4.02e-5), after two hops 8.5x
better. The fit was trained on Phi_t's own weights and signs, the baseline on phi_0's; which configurations set the tax is
not measured. Not shown: that the plateau (held-out flat from step ~100) is capacity rather than objective or optimiser.
Cost of the reopening: ~2.1 A40-h (theorie cip) plus ~1.5 CPU-h on ws1.

## Ranked recommendation (my view)
1. Keep the frustration computed in the loop (hops + FN); none of the structured signs tested here helps it. A stored
   sign-free pre-image passes the loop gate only if stored exactly (2.4e-8); at the baseline's network size it fails
   4.5x (C2-L-exp). The open question it leaves: can any network of affordable size store |Phi_t| near small
   amplitudes ~10x better than this one? Untested, and not cheap (the needed gain is 9-16x).
2. Improve the stored part with the literature's best storage (CNN + MPS backflow, arXiv:2603.14425, 2.3e-6 at 6x6)
   as the loop's refit network, and test it with the G-tax protocol (tax_2 < 0.82e-6).
3. Physics worth keeping (memo-level, not yet a paper claim): a short imaginary-time filter (J2 or J1, they agree
   to first order) makes the frustrated ground state nearly sign-free at a strength that did not grow from 16 to 24
   sites, and with exact magnitudes the Marshall sign then recovers the node to 1.45e-5/site. The sign is "a short
   filter deep"; the cost is in evaluating that filter at a configuration and in storing magnitudes near the node.
4. Do not spend more on structured signs of the kinds tested (rules, partons + visons, frames, parents, similarities).
