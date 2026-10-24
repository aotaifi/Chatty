# Ansatz pioneer: what captures frustration in the J1-J2 ground state (J2/J1 = 0.5)
Angle lead, 2026-10-23/24. Draft for adversarial review. Plan, pre-registrations and dead ends: ../PLAN_ansatz.md.

## Bottom line
In every exact test (4x4, 4x6, 6x6) the sign of the frustrated ground state is made by a local, dynamical
cancellation of J1 and J2 exchange paths, not by any static structure. Seven structural candidates were killed by
pre-registered tests: global sign rules, Z2 partons dressed by visons, rotated local frames, an unfrustrated parent,
stoquastic non-unitary similarities, and J2 layers used as add-on or as a trained variational layer.
The one robust positive fact is about the ground state alone. Undo the J2 exchange (an inverse J2 operator, additive
or a product of 2-site gates) and the pre-image is exactly Marshall-positive (4x4 both forms; 6x6 loop target with the
additive form at strong layers, cond >= 30).
So the frustration is "one J2 step" deep, but only as a computed operation, never as a stored or static one.
Storing that pre-image in the FN loop fails for a quantitative reason (C2-L): the pre-image is node-free only at a
strong layer (cond 30 at 6x6), and the layer then magnifies storage errors ~30x (two-hop tax 5.7e-5 vs bar 1.66e-6,
although the network fits the pre-image's tail 4.5x better than the loop fits its own target).
Recommendation: frustration's sign is a computed, not a stored, object. The right class is "stored smooth part +
computed cancellation" (hop/Krylov-dressed networks), which the FN loop already is; improve what is stored with the
best literature network rather than with a new sign structure.

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
| C3 | Z2 spin liquid + vison-pair sum | open part: learned local flux sums | 4x6 rank 20: vison 7.4e-4 vs random no-flux 4.1-5.7e-4 | dead |
| C4 | positive state in the 2x2 crossed-plaquette frame | Herz et al. 2607.24679 (ladders) | no rotation beats Marshall (4x4) | dead |
| C5 | amplitude-sign coupled output e^f tanh(g) | generic | ratios 0.73/1.30/0.97 vs standard (4x6) | parked |
| C6 | psi0 = (1 + a H_J2) GS(H_p), H_p stoquastic | novel construction | no stoquastic parent; dressing amplifies parent error | dead |
| C6b | 2-site J2 gates as non-unitary similarity | novel | sign exponent never below H's | dead |
| C2-L | store the Marshall-positive J2 pre-image in the FN loop | loop-specific | exact store 2.5e-7; fitted 5.7e-5 (positive RBM), 3.4e-3 (RBM+PP); bar 1.66e-6 | dead |

## Ranked recommendation (my view)
1. Keep the frustration computed: the loop's hops and FN are the only constructions that represent the near-node
   cancellation; no static sign structure tested here does (seven kills), and storing a sign-free pre-image trades the
   tail for error amplification (C2-L).
2. Improve the stored part with the literature's best storage (CNN + MPS backflow, arXiv:2603.14425, 2.3e-6 at 6x6)
   as the loop's refit network, and test it with the G-tax protocol (tax_2 < 0.82e-6).
3. Physics worth keeping (memo-level, not yet a paper claim): the J2 pre-image of the frustrated ground state is
   exactly Marshall-positive at 4x4 and at strong layers on 6x6, and the fraction of Marshall-violating weight rises
   to 1/2 in the deep tail. Both say the sign is "one J2 exchange deep" but only as a dynamical operation.
4. Do not spend more on static sign structure (rules, partons + visons, frames, parents, similarities).
