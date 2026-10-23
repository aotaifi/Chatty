# PLAN_ansatz (angle lead "ansatz pioneer", started 2026-10-23; project lead's PLAN.md points here)
Question: is there a variational ansatz for frustrated magnets that represents the frustration physics (sign structure,
near-node region, criticality) natively, which the literature has missed? Trade-off allowed: physics over scalability.
Ends: a candidate beats the best in-family baseline on a pre-registered exact 6x6 test (J2/J1 = 0.5), or ~5 working
days (by 2026-10-30). Deliverable: ansatz/ANSATZ_MEMO.md (< 3 pages). Report to the project lead at close or when a
finding changes PLAN.md. Since 2026-10-23 the PI lets this angle run as its own project (own workers, own angles).
Budget: <= 20 H100-h (normal QoS, Paderborn), workstation CPU freely; literature first.

## Working picture (hypothesis, not claim)
Each of the three symmetric product sign rules cures two of the three bond families and frustrates the third:
Neel s_N = (-1)^{n_up(x+y even)} cures J1x, J1y; stripe s_X = (-1)^{n_up(x odd)} cures J1x, J2; s_Y cures J1y, J2.
They are the only lattice-symmetric product rules on the square lattice, and s_N s_X s_Y = +1 at even N_up (every site
sits in an odd number of the three sets). Frustration = no rule cures all bonds. Exact data (T0-phys, t0phys_17410698):
the near-node set X_0.1 is stripe-tilted (M_S^2/M_N^2 5.7x bulk) and 40% of its J1 bonds violate Marshall.

## Candidates (novelty check: worker -> ansatz/lit/NOVELTY.md; landscape: ansatz/lit/LANDSCAPE.md)
| id | ansatz | physics it targets | first kill test | my prior |
|---|---|---|---|---|
| C1 | psi = s_N A + s_X B + s_Y C, A,B,C >= 0 smooth (node = level set of smooth functions) | Neel/stripe competition; the node as interference, learnable in log space | C1-T0 below | ~55% survives T0 |
| C2 | psi = sum_{y near x} K_theta(x,y) Phi(y), Phi >= 0 nodeless, K local signed (learned generalised hop) | cancellation built into the ansatz | representability: is K^-1 psi0 near-positive for a local K (e.g. 1 + a H_J2)? | ~35% |
| C3 | weighted sum of projected BCS states over Z2 flux (vison) configurations | spin liquid vs columnar VBS competition (vison fluctuations) | 4x4/6x6 overlap gain over one projected state at equal params | ~30% |
| C4 | positive amplitude in the 2x2 crossed-plaquette eigenframe | frustration absorbed by a local frame | sign-violating weight of psi0 in the plaquette frame at 4x4 vs Marshall | ~25% (likely anticipated: Torlai 2020, Levy-Clark 2021) |
Scaling caveat on C1 (stated before any test): a global sum of 3 components is a low-rank version of a product of local
interferences; at 6x6 (L ~ correlation length) global ~ local, so a 6x6 pass must be followed by a locality test.

## Pre-registered tests
### C1-T0: is the exact sign structure that of three-rule interference? (2026-10-23, before any run; exact, no training, CPU minutes)
Data: exact psi0 table (chatty_stall6/data/psi0_6x6_table.npz; per-rep amp, orbit size). Weight w = amp^2 * orbit.
Classes by (s_N, s_X, s_Y): K1 (+,+,+), K2 (+,-,-), K3 (-,+,-), K4 (-,-,+) (global sign of psi0 set by Marshall majority).
A symmetric C1 (B, C related by rotation) forces exactly one of K1, K2 to one sign; K3, K4 are always representable.
- Checks (must be able to fail): identity s_N s_X s_Y = +1 on all reps (broken control: s_X from columns x = 0 mod 3
  must violate it); class labels constant over orbits (random 10^4 reps x all 576 images).
- Measured: W_M = Marshall-violating weight; violation weight per class; U = min over {K1, K2} of the class's
  violation weight (forced sign = Marshall); R = U / W_M. Null (stripe rules carry no information): violations with
  s_N = +1 split evenly over K1, K2, so R_null = 0.5 W_M(s_N=+1) / W_M. Also by psi0 decade.
- Pass: R <= 0.2 R_null (stripe rules locate the violations). Kill: R >= 0.6 R_null. Between: inconclusive, read the
  per-class table. Pass leads to C1-T1 (fit A, B, C with small positive networks vs a same-size Marshall network
  and vs a same-size complex network, exact 6x6 overlap and E_FN with the ansatz's own sign).

## Running
| job | where | next step per verdict |
|---|---|---|
| lit: landscape map | worker (Opus) -> ansatz/lit/LANDSCAPE.md | read; update candidates/gaps |
| lit: novelty check C1-C4 | worker (Opus) -> ansatz/lit/NOVELTY.md | drop anticipated candidates |

## Dead ends
(none yet)

## Status
- 2026-10-23: plan created; greeting with blind-spot reading and C1-C3 sent to the PI; C4 added.

## Cockpit
- STATUS (in progress, medium): Mapping the ansatz literature and testing whether the exact 6x6 sign structure is an interference of the Neel and stripe sign rules.
- NEXT (open, medium): Run the sign-class test on the exact 6x6 state, then fit the three-component ansatz if it passes.
