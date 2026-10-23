# PLAN_ansatz (angle lead "ansatz pioneer", started 2026-10-23; project lead's PLAN.md points here)

## How to work (PI, via the project lead, 2026-10-23)
- Bold in ideas, prudent in compute: write what a test can teach and what kills the idea before any GPU time; a bold
  idea earns a cheap test first (4x4 exact, minutes), never a large run on hope.
- Step back on a countable event (second failed version of one idea, a spent test budget, five actions on one
  sub-problem without the key number moving): load the step-back skill.
- Consult: ask the PI what only he knows (which physics matters most, what counts as a breakthrough); literature for
  field knowledge; a worker without my view to break a claim; the ws1 research workspace for colleagues. Ask early.
- Breadth before depth: >= 3 live directions until tests kill them; drop a refuted idea in a sentence.
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

### C2-T0 and C4-T0: small-lattice representability (2026-10-23, before any run; exact 4x4 and 4x6 PBC, CPU minutes; code ansatz/code/small_pretests.py)
Reference: W_M(psi0) = Marshall-violating |psi0|^2 weight (6x6: 0.0196). Both reported in total and by decade.
- C2-T0: Phi_a = (1 + a H_J2)^{-1} psi0 (and with the J2 hop part V_J2 only), a scanned over the range where the
  kernel is well conditioned (cond <= 1e3). Is psi0 one local J2 kernel away from a Marshall-positive state?
  Pass: min_a W_M(Phi_a) <= 0.1 W_M(psi0) at both sizes. Kill: >= 0.5 W_M(psi0) at both sizes.
- C4-T0: psi0 in the 2x2 crossed-plaquette frame: start from the plaquette eigenbasis (Sz-resolved, best sign gauge),
  then optimise a translation-invariant Sz-conserving rotation (and a two-plaquette-sublattice variant) by gradient
  ascent of sum psi'|psi'| (= 1 - 2 W_neg). Pass: W_neg <= 0.1 W_M(psi0); kill: >= 0.5 W_M(psi0), at both sizes.
  Control: the same optimiser in the Ising frame with single-site rotations must not beat Marshall by > 10% (the
  Marshall rule is the best product rule; a big gain would mean the optimiser or the metric is wrong).

### C2-T0 at 6x6 (2026-10-23, after the 4x4 result and before any 6x6 run; code ansatz/code/c2_t0_6x6.py; <= 3 H100-h)
4x4 (pre_4x4.json): Phi = (1 + aK)^{-1} psi0 is EXACTLY Marshall-positive (W_M = 0, cond 4.6-17) for K = H_J2, V_J2 and
the control V_J1; V_J3 0.028 W_M. The effect is generic to local hop kernels, not J2-specific. Phi's support stops at
decade -6, psi0's reaches -10: the pre-image has no tail. 4x4 is anomalous, so 6x6 decides.
Kernel at 6x6: V = H - diag(H) (the full off-diagonal hop; the only one computable with the existing sector CSR).
Note: K = 1 + aH cannot work at all (psi0 is its eigenvector, pre-image = psi0), so the standard Lanczos step on a
sign-free state can never be exact, while a hop kernel can.
- Pass: min_a W_M(Phi_a) <= 0.1 W_M(psi0) = 1.96e-3. Kill: >= 0.5 W_M(psi0). Second readout: tail weight
  (per-config p < 1e-9) of Phi vs psi0; the "tail-free pre-image" claim needs <= 0.1x psi0's.
- Pass leads to C2-T1: fit a positive network to log Phi vs the same network to log|psi0| (equal parameters, same
  objective), then E_FN and energy of (1 + aV) Phi_fit vs Phi_fit-free baselines.

### C2-T0b at 6x6: is the pre-image effect specific to the frustrating J2 hop? (2026-10-23, before any run; <= 4 H100-h)
4x4 at moderate conditioning (cond <= 3, |a| rho(K) <= 1/2), pre_4x4.json: V_J2 cuts W_M 68x (1.9e-4 vs 1.27e-2),
V_J3 3.3x, V_J1 not at all (1.7e-2); V_J1 reaches 0 only at cond 19 (Perron dominance, trivial). So the honest
metric is W_M at fixed cond, and the kernel must be V_J2 alone.
Build: sector operators V_J1, V_J2 (bond-class neighbour tables in the 15.8M orbit basis); check: D0 + V_J1 + V_J2
equals sec.H on 3 random vectors to rel 1e-12 (fails on a deliberately mis-scaled J2).
- Pass: W_M(Phi_a; V_J2) at cond <= 3 is <= 0.1 W_M(psi0), AND V_J1 at cond <= 3 is >= 0.5 W_M(psi0) (specificity).
- Kill: V_J2 at cond <= 3 >= 0.5 W_M(psi0). Else inconclusive. Also report the tail weight of Phi (p < 1e-9).

### C3-T0: do vison (Z2 flux) fluctuations of a projected Z2 spin liquid carry the missing physics? (2026-10-23, before any run; exact 4x4, CPU; code ansatz/code/c3_t0.py)
Base: Gutzwiller-projected BCS, Hu-Becca-Parola-Sorella Z2 ansatz (arXiv:1304.2630): NN hopping t = 1, NN
d_{x2-y2} pairing D1, NNN d_xy pairing D2, chemical potential mu; spinon boundary conditions (P/A in x, y).
psi(x) = sign_order(x) det[f(u_a, d_b)], f from the BCS ground state. Optimise (D1, D2, mu) for max overlap per BC.
Spans compared by Ritz energy (generalised eigenproblem in the span) and infidelity of the best combination:
 (0) best single state; (S) + the other 3 BC sectors; (V) + vison-pair states (sign of t, D flipped on a bond string
 between two plaquettes, all separations/orientations, each symmetrised over translations); (R) control: + the same
 number K of states with random +-10% changes of t, D on the same bonds (no flux).
Checks: D2 = 0, mu = 0 gives W_M = 0 (bipartite pairing obeys Marshall); a 1-state span's Ritz energy = <H>.
- Pass: (V) lowers E_Ritz - E0 by >= 2x more than (R) at equal K. Kill: (V) <= (R). 4x4 is anomalous: a pass
  earns a 6x6 version, never a production run.
Third live direction (C5, test to be written after C2-T0b): amplitude-sign coupled output psi = s_N e^f tanh(g)
(node = smooth zero crossing) vs e^f x sign at equal capacity, exact 6x6 tail error and own-sign E_FN.

## Running
| job | where | next step per verdict |
|---|---|---|
| C2-T0 6x6 (kernel V = H - diag) | worker, Paderborn H100 | PASS/INCONCLUSIVE: wait for C2-T0b; KILL: V_H mixes J1 smoothing, still run C2-T0b |
| C2-T0b 6x6 (V_J2, V_J1 operators) | worker, Paderborn H100 | PASS -> C2-T1 full-sum VMC arms; KILL -> C2 dead, back to C3 |

## Dead ends
- C1 global sign-rule interference (C1-T0 KILL, 2026-10-23, /scratch/a/A.Otaifi/ansatz_runs/c1_t0/c1_t0.json, code
  ansatz/code/c1_t0.py): Marshall-violation fraction is the same in every (s_N, s_X, s_Y) class at every psi0 decade
  (e.g. decade -11: K1 0.262, K2 0.257, K3+K4 0.274); R/R_null = 0.99. Cause: the rule classes are global sublattice
  parities, the violations are local events. Checks: identity 0 violations (broken control 7.9M), orbit labels 0
  inconsistencies. Side fact: Marshall-violation fraction by decade rises 0.001 (-7), 0.11 (-10), 0.44 (-12), 0.54-0.57
  (-13..-15): in the deep tail the exact sign is uncorrelated with Marshall; the node runs through the tail.

## Status
- 2026-10-23: plan created; greeting with blind-spot reading and C1-C3 sent to the PI; C4 added.

## Cockpit
- STATUS (in progress, medium): Mapping the ansatz literature and testing whether the exact 6x6 sign structure is an interference of the Neel and stripe sign rules.
- NEXT (open, medium): Run the sign-class test on the exact 6x6 state, then fit the three-component ansatz if it passes.
