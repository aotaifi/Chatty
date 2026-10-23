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

## Breakthrough bar (lead's proposal to the PI, 2026-10-23; PI asked for my view)
A compact, interpretable ansatz built on one named mechanism (tens to hundreds of parameters) that reaches <= 2e-5
per site at exact 6x6 (within ~10x of the black-box record 2.3e-6, CNN-MPS arXiv:2603.14425), and whose mechanism
parameter says something checkable at larger L (e.g. vison fugacity: spin liquid vs columnar VBS). An energy record
alone is not the bar; a sign-structure law that does not lower the energy is not either.
PI 2026-10-23: also (a) make the loop's node -> network -> repeat edge prominent and (b) work because it encodes the
physics. Lead's reading: the stored part of the guide should be compact physics (few parameters, cheap refit) and the
near-node detail computed by FN + hop. Best fit: C3 as the loop's guide; first loop test = FN energy of its sign.
PI 2026-10-23: and it must scale to larger L without data-centre compute. C3 cost: one projected determinant O(N^3)
(fast updates O(N^2)); a short vison pair is a low-rank update of the pairing matrix (string of l bonds: rank ~2l),
so a dilute sum over short pairs costs O(N^2 l^2) per amplitude given F^{-1}; VMC/GFMC with such a guide is standard
to 18x18+ (Hu et al. 1304.2630 ran 18x18). To be measured, not assumed, if C3 survives.

## Working picture (hypothesis, not claim)
Each of the three symmetric product sign rules cures two of the three bond families and frustrates the third:
Neel s_N = (-1)^{n_up(x+y even)} cures J1x, J1y; stripe s_X = (-1)^{n_up(x odd)} cures J1x, J2; s_Y cures J1y, J2.
They are the only lattice-symmetric product rules on the square lattice, and s_N s_X s_Y = +1 at even N_up (every site
sits in an odd number of the three sets). Frustration = no rule cures all bonds. Exact data (T0-phys, t0phys_17410698):
the near-node set X_0.1 is stripe-tilted (M_S^2/M_N^2 5.7x bulk) and 40% of its J1 bonds violate Marshall.

## Step back 2026-10-23 night (trigger: second failed C2 version: specificity only quantitative at 4x6, add-on fails at 4x4)
1. Verdict: switch. C2 as an add-on layer is dead; C2 survives only as a jointly trained positive base + J2 layer,
   tested once with a shared exact optimiser that also serves C5. C3 (vison sum) becomes the lead if C3-T0-1 passes.
2. Thought: the sign-free pre-image Phi = (1 + a H_J2)^{-1} psi0 is far from |psi0| (a ~ -0.1 times the J2 diagonal,
   up to +9 on Neel-like configurations, reweights them ~10x), so its existence says nothing about add-on value; on
   existing bases the full-H Lanczos layer wins 4.5x/5.5x/1x (4x4, c2t1_4x4.json). Form-wise this is a non-diagonal
   similarity transform (cf. Gutzwiller-transcorrelated FCIQMC, Dobrautz-Luo-Alavi PRB 99, 075119 (2019), diagonal only).
3. Next: exact full-sum variational optimiser at 4x6 (2.7M configs; JAX on a ws1 GPU) with pluggable positive base
   (Jastrow, RBM), layers (none, 1 + aH, 1 + aH_J2) and output forms (C5); C2-T2 and C5-T0 on it; ~2 h build + <= 2 GPU-h.

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

#### C2-T0 small-lattice result (2026-10-23; ansatz_runs/small/pre_4x4.json, pre_4x6.json)
W_M(Phi_a)/W_M(psi0) at matched conditioning (best a with cond <= 3 / <= 5):
| kernel | 4x4 | 4x6 |
|---|---|---|
| H_J2 (hop + J2 diagonal) | 0.014 / 0 | 0.10 / 0.003 |
| V_J2 (J2 hop) | 0.015 / 0 | 0.13 / 0.13 (0.005 at cond 5.3) |
| V_J1 (control) | 1.3 / 0.9 | 0.45 / 0.14 |
| V_J3 (control, same sublattice, wrong geometry) | 0.30 / 0.05 | 0.59 / 0.42 |
| V = H - diag (full hop) | - | 0.95 / 0.93 |
Reading: J2 is the most efficient kernel (3.5x V_J1 at cond 3 on 4x6), but any positivity-preserving smoothing
removes violations at large enough cond, so specificity is quantitative. Phi is NOT tail-free at 4x6 (its weight in
decades -8..-13 is similar to psi0's); it is sign-free there. The full-hop kernel does nothing (J1 and J2 parts act
in opposite directions), so the running 6x6 C2-T0 (kernel V) is expected uninformative; C2-T0b now has V_J2, V_J1, H_J2.
C4 (plaquette frame) 4x4: no rotation beats Marshall (0.012725 vs 0.012731 from Marshall start; eigenbasis start
ends at 0.015-0.018): KILL at 4x4. The 4x6 confirmation was stopped (ws1 overloaded, low value).

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

#### C2-T0b result (6x6, theorie A40 job 17463609, ansatz_runs/c2_t0b_6x6/run_17463609/c2_t0b_6x6.json; verified by the lead): INCONCLUSIVE
W_M(Phi)/W_M(psi0) (better sign of a; J2 kernels need a < 0, a > 0 makes it 1.4-3.1x worse):
| cond | V_J2 | H_J2 | V_J1 |
|---|---|---|---|
| 3 | 0.24 | 0.18 | 0.47 |
| 5 | 0.082 | 0.034 | 0.22 |
| 10 | 0.016 | 0.0011 | 0.029 |
Size trend of V_J2 at cond 3: 0.015 (4x4), 0.13 (4x6), 0.24 (6x6): the additive layer's grip weakens with N, as the
extensive-local-energy argument predicts. Specificity over J1 is 2-2.5x, not qualitative. Phi is not tail-free
(tail weight p < 1e-9: psi0 0.078, Phi(cond 3) 0.08-0.09, rising to 0.30 at cond 30). Checks: D0 + V_J1 + V_J2 = H to
2.7e-16 (J2 x 1.01: 3.7e-3); tables match brute force; W_M(psi0) 0.0195582523. Tables: ansatz_runs/c2_t0b_6x6/tables/.

### C2-T1: does a J2 layer on a sign-free state beat the standard Lanczos layer? (2026-10-23, before any run; exact 6x6, no training; after C2-T0b delivers sector_ops.py; <= 2 H100-h)
psi_K(a) = (1 + a K) phi_+, phi_+ Marshall-signed and positive. Bases phi_+: (1) |ViT| (sym_tables.npz), (2) |PP|
(sym_tables_PP.npz), (3) Marshall lattice-FN ground state (sec.fn_solve with s_N, guide |PP|), (4) oracle |psi0|
(ceiling). Layers K: H (standard Lanczos step, the project's hop form), H_J2, V_J2, H_J1 (control).
Measured: E_var at the optimal a (closed form from 3 matvecs) and E_FN with the sign of psi_K(a*), per site above E0;
W_M of psi_K(a*). Check: K = H on base (4) must give E_var = E0 at a = 0 to 1e-10 (|psi0| with Marshall is not psi0,
so this check is instead: base psi0 itself with K = H gives a* = 0 and E0).
Scaling arms (added before any run): an additive layer acts through the EXTENSIVE local energy, 1 + a E_L^J2(x),
so in a large system its sign flips are set by a global fluctuation (the C1 lesson). The extensive form is a product
of local layers: psi = phi_+ prod_p (1 + a e_p(x)), e_p = (H_J2^(p) phi_+)(x)/phi_+(x) on plaquette p (its two J2
bonds; per-bond variant prod_b too). Same bases, E_var by a 1-D scan in a, E_FN at a*.
- Pass: for bases (1)-(3), H_J2 layer gives E_FN <= 0.5 x the H layer's E_FN, on at least 2 of 3 bases.
- Kill: H_J2 layer E_FN >= H layer's on 2 of 3 bases. Pass makes C2 a concrete proposal for the loop's hop
  (report to the project lead) and earns C2-T2 (trained phi_theta, full-sum VMC, vs same-size free-sign NQS).

### Shared tool + C2-T2 + C5-T0 (2026-10-23 night, before any code or run; exact 4x6, ansatz/code/exact_vmc.py; <= 4 GPU-h ws1)
Tool: exact full-sum variational optimiser on the 4x6 Sz=0 basis (2.7M configurations; psi0, E0/N -0.52252493537),
JAX on one GPU, H and H_J2 as per-bond gather matvecs, energy E = <psi|H|psi>/<psi|psi> with autodiff gradients,
L-BFGS (deterministic objective), 3 seeds per arm, curves of E vs step kept. Base f_theta: Jastrow over displacement
classes (J); translation-symmetric real RBM with alpha = 1, 2, 4 (R1, R2, R4). FN energy of the final state with its
own sign (scipy, as c2_t1_small.py). Checks: autodiff gradient = finite differences on 5 parameters (and fails with a
flipped sign in the energy); the 1-state energy equals a scipy evaluation; J2 = 0 Marshall-Jastrow lands within 1%
of E0(J2=0).
- C2-T2 (trained J2 layer): arms with the same base: s_N e^f; (1 + a H) s_N e^f; (1 + a H_J2) s_N e^f; a trained with
  theta. Pass: J2 layer E_var <= 0.5 x full-H layer for J and R2. Kill: J2 layer >= full-H layer for both.
- C5-T0 (amplitude-sign coupling): f, g of the same RBM size; standard s_N e^{f + i pi sigmoid(g)} vs coupled
  s_N e^f tanh(g). Pass: coupled E_var <= 0.5 x standard for 2 of 3 alphas and lower E_FN. Kill: coupled >= standard
  for all three. (Equal parameter counts; same optimiser and steps.)

### C2-L: does storing the sign-free J2 pre-image cut the loop's refit tax? (2026-10-23, before any run; loop lead's test design; exact 6x6; after G-tax lands and sector_ops.py H_J2 exists; <= 2 GPU-h)
Second opinion of the FN-loop lead (2026-10-23): for the loop, (B) = C2-stored-pre-image beats (A) = C3 about 60/40,
because 93% of the refit tax sits in the tail (G-A2) and the fit-free signed two-hop chain goes 1.95e-6 -> 3.0e-9 in
6 rounds (G-A3): whatever cuts the storage tax wins. My bet stays (A) for physics; the tests decide.
Setup = gate G-tax (runs/gate_tax): target phi_1 = FN solution of K_2(RBM+PP A), own sign.
- Phi = (1 + a H_J2)^{-1} phi_1, a = argmin W_M over the cond <= 5 grid; report W_M(Phi) and its decade table.
- Fit the G-tax network class to log|Phi| (Marshall sign), same fixed steps and time as G-tax.
- psi_fit = (1 + a H_J2) s_N e^{f_fit}. tax_B(k) = E_FN(K_k(psi_fit)) - E_FN(K_k(phi_1)), k = 1, 2 (the J2 layer costs
  ~half a full hop: 72 of 144 bonds).
- Pass: tax_B(1) (1.5 hops) < the G-tax baseline tax at K_2 (2 hops), AND tax_B(2) < 0.82e-6 (loop gate C bar).
  Kill: tax_B(2) >= G-tax baseline tax at K_2.
Baseline (loop lead, verified from raw JSON; runs/gate_tax, commit 52e7f76; gtax_full): RBM+PP (2480 params) refit
to phi_0 = FN(K_2(A)), 400 fixed steps, B = 8208: tax_1 5.107e-6, tax_2 1.657e-6 (E_FN(K_2 a_1) 1.9749e-6 vs
E_FN(K_2 phi_0) 3.1818e-7); one seed, held-out loss jumpy (0.03-0.12), so match steps and batch exactly.
So C2-L passes only if tax_B(1) < 1.657e-6 and tax_B(2) < 0.82e-6; kill if tax_B(2) >= 1.657e-6. Sign of the stored
Phi: report both sign(Phi) and s_N (oracle-free).

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
#### C3-T0 result (4x4, /scratch/a/A.Otaifi/ansatz_runs/c3_t0/c3_t0.json): INCONCLUSIVE by the rule (cause: rule and sector)
(0) best single Z2 state 5.07e-3/site; (S) 4 BC sectors 4.86e-3; (V) +20 vison-pair states 1.1e-5 (rank 23);
(R) +20 random no-flux states 4.9-5.6e-4 (rank 12-13). Lowering ratio V/R 1.1 (rule: pass >= 2) but the residual
differs 44x. Not interpretable: unequal rank (13 vs 23) in a 153-dim symmetric sector. Checks passed (Marshall at
D2 = 0: W_M 1e-62, broken sign 0.27-0.47; brute-force BCS fidelity 1.0).
#### Amendment C3-T0-1 (2026-10-23, before the run): 4x6, rank-matched control
4x6 PBC (symmetric sector ~1e4 dims, so 24 states cannot fit by counting). Control (R'): same strings, t and D
multiplied by random factors in [0.5, 1.5] (no flux), 3 seeds. Add states greedily (largest energy gain first) and
compare E_Ritz - E0 at EQUAL S-rank k = 4..rank_max. Pass: at the largest common rank, V residual <= 0.3 x R'
residual for all seeds. Kill: V residual >= 0.7 x R' for at least 2 of 3 seeds (clarified 2026-10-23 before any
4x6 output). Pass earns a 6x6 VMC-sampled version (design then). Runtime: only the AA sector is optimised (from the
4x4 optimum, <= 60 evaluations); other sectors at that point; Slurm CPU node, not ws1 (ws1 load 35-52 on 10 cores).
#### C3-FN (2026-10-23, before any run): is the vison-dressed spin liquid's NODE good? (exact 4x4 now, 4x6 after C3-T0-1)
Lattice FN energy (ten Haaf, guide = the state itself) with the sign of the best state of each span (0), (S), (V), (R/R')
at equal rank; references: Marshall sign E_FN, and E_FN of (1 + aH) applied to the best (S) state.
- Pass: E_FN(V) <= 0.3 x E_FN(R) and <= 0.3 x E_FN(Marshall). Kill: E_FN(V) >= E_FN(R).

Third live direction (C5, test to be written after C2-T0b): amplitude-sign coupled output psi = s_N e^f tanh(g)
(node = smooth zero crossing) vs e^f x sign at equal capacity, exact 6x6 tail error and own-sign E_FN.

## Running
| job | where | next step per verdict |
|---|---|---|
| exact 4x6 optimiser, C2-T2, C5-T0 | worker (Opus), ws1 GPU | per the rules above |
| C2-L (loop refit tax with stored pre-image) | worker, theorie A40 | per the rule above |
| C3-T0-1 4x6 (vison span, rank-matched) | moving to a Slurm CPU node (worker) | PASS -> 6x6 VMC design; KILL -> C3 dead |

## Dead ends
- C2 as an add-on layer (C2-T1 preview, 4x4, ansatz_runs/small/c2t1_4x4.json): E_FN with the H_J2 layer vs the
  full-H layer on |psi0|-Marshall 6.9e-3 vs 1.6e-3, Marshall-FN 3.7e-3 vs 6.6e-4, no-J2-hop GS 1.6e-2 vs 1.6e-2
  (product forms 1.4e-2). The J2 layer is not a useful add-on to existing sign-free states.
- C1 global sign-rule interference (C1-T0 KILL, 2026-10-23, /scratch/a/A.Otaifi/ansatz_runs/c1_t0/c1_t0.json, code
  ansatz/code/c1_t0.py): Marshall-violation fraction is the same in every (s_N, s_X, s_Y) class at every psi0 decade
  (e.g. decade -11: K1 0.262, K2 0.257, K3+K4 0.274); R/R_null = 0.99. Cause: the rule classes are global sublattice
  parities, the violations are local events. Checks: identity 0 violations (broken control 7.9M), orbit labels 0
  inconsistencies. Side fact: Marshall-violation fraction by decade rises 0.001 (-7), 0.11 (-10), 0.44 (-12), 0.54-0.57
  (-13..-15): in the deep tail the exact sign is uncorrelated with Marshall; the node runs through the tail.

## Status
- 2026-10-23: plan created; greeting with blind-spot reading and C1-C3 sent to the PI; C4 added.
- 2026-10-23 evening: C1 dead; C2 lead (4x4, 4x6); C3 inconclusive at 4x4, 4x6 running; asked the PI what counts
  as a breakthrough; Paderborn gpu_h100 queue est. 5 days (QoS normal not allowed there; default cont used).

## Cockpit
- STATUS (in progress, medium): The frustrated ground state looks like a sign-free state plus one local J2 cancellation layer on 4x4 and 4x6; the exact 6x6 test is queued.
- NEXT (open, medium): Test the J2 layer against the standard Lanczos layer on exact 6x6, and the spin-liquid vison sum at 4x6.
