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

## Step back 2026-10-24 early (trigger: second failed version of C6; six candidates killed, none passed)
1. Verdict: reframe toward what the data say, then finish. Static structure (global rules, partons + visons, local
   frames, unfrustrated parents, stoquastic similarities) fails every time; the sign of the frustrated state is made by
   a dynamical cancellation of exchange paths, which only a hop/Krylov construction computes. The answer to the
   angle's question is likely "stored smooth part + computed cancellation", and C2-L is the decisive test of the best
   stored part. Next deliverable: ANSATZ_MEMO.md with this ranked conclusion, whatever C2-L gives.
2. Thought: the one robust positive fact (an exactly Marshall-positive pre-image under J2 operators, additive or
   product, up to 6x6) is a statement about the ground state alone; every test that needed more (a useful layer, a
   stoquastic parent, a stoquastic similarity) failed, because the frustration is in the dynamics.
3. Next: C2-L-1 fits (running, <= 5 A40-h); draft the memo now; no new candidate without a cheap 4x4 kill test.

## Step back 2026-10-23 night (trigger: second failed C2 version: specificity only quantitative at 4x6, add-on fails at 4x4)
1. Verdict: switch. C2 as an add-on layer is dead; C2 survives only as a jointly trained positive base + J2 layer,
   tested once with a shared exact optimiser that also serves C5. C3 (vison sum) becomes the lead if C3-T0-1 passes.
2. Thought: the sign-free pre-image Phi = (1 + a H_J2)^{-1} psi0 is far from |psi0| (a ~ -0.1 times the J2 diagonal,
   up to +9 on Neel-like configurations, reweights them ~10x), so its existence says nothing about add-on value; on
   existing bases the full-H Lanczos layer wins 4.5x/5.5x/1x (4x4, c2t1_4x4.json). Form-wise this is a non-diagonal
   similarity transform (cf. Gutzwiller-transcorrelated FCIQMC, Dobrautz-Luo-Alavi PRB 99, 075119 (2019), diagonal only).
3. Next: exact full-sum variational optimiser at 4x6 (2.7M configs; JAX on a ws1 GPU) with pluggable positive base
   (Jastrow, RBM), layers (none, 1 + aH, 1 + aH_J2) and output forms (C5); C2-T2 and C5-T0 on it; ~2 h build + <= 2 GPU-h.

## Candidates and live directions (2026-10-23 night; dead ones are in Dead ends)
Live (>= 3, per the PI's guidance):
| id | idea | decisive test | status |
|---|---|---|---|
| C2-loc (untested beyond 4x4: the 4x6 preview stalled in ARPACK) | product of LOCAL J2 cancellation layers psi = phi_+ prod_p (1 + a e_p): scales (the additive layer's grip fades with N: 0.015 / 0.13 / 0.24 at 4x4 / 4x6 / 6x6) | 4x6 preview arms + C2-T2 | running (4x6 optimiser) |
| C5 (parked: inconclusive, ratios ~1) | amplitude-sign coupled storage: s_N e^f tanh(g); its sinh variant = s_N (P - Q), two positive parts (J1-driven minus J2-driven amplitude, from the eigen-equation) whose level set P = Q is the node | C5-T0 (4x6), then the G-tax refit tax | running (4x6 optimiser); sinh arm to add |
Reserve: MPS-augmented storage of the tail (CNN-MPS, arXiv:2603.14425) as the loop's refit network: tail rms and
refit tax vs RBM+PP on the G-tax target (literature method, not a new ansatz).
Dead: C1 (global sign-rule interference), C3 (vison-dressed Z2 spin liquid), C4 (plaquette frame), C2 as add-on.

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

#### C2-L first result (theorie A40 17466020, ansatz_runs/c2l_6x6/run_17466020): s_N arm fails before any fit
Control: E_FN(K_2 phi_0) 3.181795e-7 (G-tax 3.1818e-7). Phi at a = -0.0702 (H_J2, cond 5): W_M 5.76e-4 (phi_0 1.87e-2,
32x cut), tail 0.092 vs 0.078. Exact |Phi| with s_N, no fit: tax_B(1) 7.35e-5, tax_B(2) 2.93e-5, 18x above the kill
bar; the residual violating weight costs E_FN 4.5e-4 at zero hop. sign(Phi) arm: exact round trip tax 0, but sign(Phi)
is not available at use time without storing a non-Marshall sign (the problem C2 was meant to remove); its fit (400
steps at 16.7 s/step on A40 = 2.4 h, warm-start loss 14x worse than G-tax's) is not run.
#### Amendment C2-L-0 (2026-10-23, before the run): exact Phi with s_N at stronger layers, no fit (<= 0.5 GPU-h)
H_J2 and V_J2 at cond 5, 10, 30, 100 (a < 0): tax_B(1), tax_B(2) with exact |Phi| and s_N, W_M(Phi), tail weight.
- Kill C2-L: min over cond of tax_B(2) >= 1.657e-6. Continue: tax_B(2) < 0.82e-6 at some cond -> the 400-step fit at
  that cond (<= 3 GPU-h). Between: inconclusive, report.

#### C2-L-0 result (theorie A40 17466418, run_17466418/cx_gtax.json L0): CONTINUE by the rule
Exact |Phi|, s_N, no fit; tax_B(2): H_J2 cond 5/10/30/100: 2.93e-5 / 1.85e-6 / 2.5e-7 / 0 (W_M(Phi) 5.8e-4 / 1.7e-5 /
7.4e-8 / 0; tail 0.092 / 0.124 / 0.241 / 0.404 vs phi_0 0.078). V_J2 never below 1.38e-6. Cond 100 is trivial (Phi
dominated by H_J2's top eigenvector, positive in the Marshall frame); cond 30 is the test point.
#### Amendment C2-L-1 (2026-10-24, before the fit): fit at H_J2 cond 30 (a = -0.1028), two network arms
(i) RBM+PP (G-tax class, as pre-registered); (ii) a positive network of ~2480 parameters (RBM part only, no Pfaffian:
a Pfaffian's amplitude has its own nodes, a structural mismatch for a node-free Phi). Both: sign s_N, 400 fixed
steps, G-tax batch/tau/stratification, warm start: (i) A, (ii) A's RBM factor. Report fit rms by decade next to
G-tax's. Rule as C2-L (pass: tax_B(1) < 1.657e-6 and tax_B(2) < 0.82e-6; kill: tax_B(2) >= 1.657e-6), per arm.
<= 5 A40-h. A kill with the 400-step cap reads "inconclusive (cause: warm start)" only if the held-out loss is still
falling steeply at step 400 (slope over the last 50 steps > 1% of the loss per 10 steps).

### C6-T0: is the frustrated ground state a local dressing of an UNFRUSTRATED parent? (2026-10-24, before any run; exact 4x4 and 4x6, CPU; code ansatz/code/c6_t0.py)
Hypothesis (~20%): psi0 = (1 + a H_J2) GS(H_p) with H_p local and stoquastic in the Marshall frame (sign-free GFMC at
any L; frustration lives only in the explicit J2 dressing). Rests on: the exact pre-image Phi = (1 + a H_J2)^{-1} psi
is Marshall-positive at strong layers (6x6 loop target: W_M 7.4e-8 at cond 30, 0 at cond 100; C2-L-0).
Method (inverse problem, Chertkov-Clark PRX 8, 031029 (2018)): for Phi at H_J2 cond 5, 10, 30, the quantum covariance
matrix C_kl = <O_k O_l> - <O_k><O_l> in Phi over a local symmetric operator basis O_k: XY_d and ZZ_d bond sums for all
displacement classes with |d| <= 2 sqrt 2, plus the J1-plaquette ring exchange. Stoquastic constraint per operator from
the sign of its Marshall-frame off-diagonal elements (checked numerically). Parent = argmin c^T C c with the J1 XY
coefficient fixed to 1, (u) unconstrained, (s) stoquastic-constrained. Then GS(H_p) by Lanczos and the dressed state
(1 + a' H_J2) GS(H_p) with a' re-optimised.
Measured: relative variance of the parent on Phi, fidelity |<GS(H_p)|Phi>|^2, E_var - E0 per site of the dressed
state, and its E_FN. Baselines (sign-free routes at the same size): Marshall-FN E_FN (4x6: 6.95e-3/site), and
(1 + aH_J2) applied to the Marshall-FN ground state. Checks: parent search on psi0 itself, unconstrained, recovers
H_J1-J2 (J2/J1 = 0.5 to 1e-6); a stoquastic parent of Phi at cond 100 recovers ~-H_J2 dominance (sanity).
Novelty (worker, ansatz/lit/NOVELTY.md section C6): NOVEL as a construction; closest: transcorrelated FCIQMC
(Dobrautz-Luo-Alavi arXiv:1811.03607; diagonal similarity only, which can never cure non-stoquasticity), Chertkov-
Clark inversion (no stoquastic constraint). Risks it names: additive K gives a global sign threshold (my size trend);
a local parent may not exist (Brito-Marston, projected states, abstracts only). Scalable successor if C6-T0 passes
(C6b): K = prod_b (1 + a h_b) as a finite-depth circuit of 2-site non-unitary gates, so K^{-1} H K is quasi-local;
test whether it is (nearly) stoquastic in the Marshall frame, which would make projector QMC on it sign-free.
- Pass (4x6): stoquastic parent with dressed E_var - E0 <= 1e-3/site (7x below Marshall-FN). Kill: dressed state no
  better than (1 + aH_J2) on the Marshall-FN ground state. Pass earns a 6x6 version (sector machinery) and a GFMC plan.

### C6b-T0: does a product of 2-site J2 gates, used as a non-unitary similarity, ease the sign problem? (2026-10-24, before any run; exact 4x4, CPU minutes; code ansatz/code/c6b_t0.py)
K(a) = prod_b (1 + a h_b) over all J2 bonds b in a fixed order (h_b = J2 S_i.S_j), Ht = K^{-1} H K (same spectrum;
right ground vector Phi = K^{-1} psi0). Sign-problem exponent per site of a matrix M in the Marshall frame:
g(M) = (E0 - lambda_min(M_s))/N, M_s = M with every off-diagonal replaced by -|M_xy| (the stoquastised matrix whose
projector governs the average sign). Reference g(H). Scan a over values where every gate is invertible.
- Pass: min_a g(Ht(a)) <= 0.1 g(H). Kill: g(Ht(a)) >= 0.5 g(H) for all a. Also report W_M(Phi(a)) and the locality
  of Ht (off-diagonal weight versus Hamming distance), since a quasi-local Ht is what makes GFMC possible.
Check: a = 0 gives g(H); g(M) = 0 for M = H at J2 = 0 (Marshall-stoquastic).

### Amendments after the adversarial review (2026-10-24, ansatz/REVIEW_adversarial.md; before any rerun)
Review verdicts: C1, C2 add-on (run at 4x4, registered for 6x6), C2-T2, C2-L, C6b SOUND; C6 FLAWED; C3, C4, C5 WEAK.
- C6-T0b (fixes the min-variance criterion; exact 4x4, CPU minutes): optimise the stoquastic couplings (all of them
  free, normalised by |c| = 1, so -H_J2 is reachable) directly for the dressed state's E_var, (1 + a' H_J2) GS(H_p),
  Powell/Nelder-Mead from 5 starts incl. the C6-T0 parents and -H_J2 + H_J1. Pass: dressed E_var <= 0.5 x the
  dressed-Marshall-FN baseline (4.17e-3 at 4x4); kill: >= baseline. Pass earns the 4x6 version.
- C3-T0-2 (fixes the control's advantage; 4x6, <= 1 CPU-node-hour): converge the AA base (<= 400 evaluations), add
  the 3 parameter-derivative states of the base to (S) so both arms start from a repaired base, then V vs R' at equal
  rank with the greedy step that only takes new directions, to the largest common rank. Rule as C3-T0-1.
- C5-T0 continuation (<= 2 A40-h): continue all C5 arms from their checkpoints to |dE| < 1e-8/site per 500 steps or
  6000 steps, complete 3 seeds for R1, R2. Rule as C5-T0.
- C4 stays "dead at 4x4, untested at 4x6" (low value). C2-L kill stands; its mechanism is corrected: the positive
  RBM's fit is 2-2.4x worse than G-tax in the bulk (decades -8, -10) and better only at -12, -13; amplification by the
  layer and bulk misfit are not separated.

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

## Dead ends
- C2-L store the J2 pre-image in the FN loop (C2-L-1 KILL, theorie A40 17468620/21, ansatz_runs/c2l_6x6/run_*,
  verified): H_J2 cond 30, 400 fixed steps, two-hop tax (bar 1.657e-6, gate 0.82e-6): RBM+PP 3.41e-3 (cannot fit a
  node-free target: rms 0.13 already at decade -5), positive RBM (2442 params) 5.66e-5; exact store 2.50e-7. The
  positive RBM fits Phi's tail far better than G-tax fits phi_0 (rms at -13: 0.060 vs 0.273), but the layer magnifies
  the storage error ~30x. Trade-off with no window at 6x6: node-free Phi needs cond >= 30 (cond 5: exact-store tax
  already 2.9e-5); at cond 30 the fit's bulk misfit (2-2.4x G-tax's) and the layer's amplification are not separated.
  Held-out slopes 0.11% / 0.73% per 10 steps (< 1%).
- C6b product of 2-site J2 gates as a non-unitary similarity (C6b-T0 KILL, ansatz_runs/c6b/c6b_4x4.json; first run
  void: inverse gates in the wrong order, residual 0.3, kept as c6b_4x4_buggy_order.json): sign exponent per site
  g(Ht(a)) >= g(H) = 0.383 for every a (0.497 at a = -0.25 where Phi is already exactly Marshall-positive; 27 at
  a = -1.5); eigen-residual 5e-12. A sign-free ground state does not make the propagator sign-free.
- C6 unfrustrated parent + additive J2 dressing (C6-T0 KILL, cip 17470331/2, ansatz_runs/c6_t0): the min-variance
  stoquastic parent of Phi drops every same-sublattice XY term and its GS is magnetic (fidelity 0 at 4x6); dressed
  E_var 0.82/site vs 4.06e-3 baseline. At 4x4 a stoquastic parent's GS has fidelity 0.949 with Phi, yet the dressed
  state is worse than the baseline (1.36e-2 vs 4.17e-3): the dressing amplifies parent errors, so C6 needs fidelity
  -> 1 from a few-body sign-free parent, which the data rule out. The only good parent (unconstrained, cond 5,
  fidelity 0.958, dressed E_FN 2.8e-3 < 3.6e-3) carries non-stoquastic long-range XY.
- C2 as a trained variational layer (C2-T2 KILL, 4x6 exact optimiser, jobs 17469982/17469926, ansatz_runs/
  exact_vmc_4x6/verdict.json, verified): E_var with the H_J2 layer vs the full-H layer on the same jointly trained base:
  Jastrow 1.61e-2 vs 1.02e-2, RBM a=2 3.56e-3 vs 2.83e-3 (E_FN same direction). Check note: the J2 = 0 Marshall-
  Jastrow sits 1.16% above E0 against my 1% bar; scipy BFGS gives the same optimum, so the bar was wrong, not the tool.
- C5 amplitude-sign coupled output (C5-T0 INCONCLUSIVE, parked; jobs 17469983/4): coupled/standard E_var 0.73 (R1),
  1.30 (R2, one seed), 0.97 (R4, within seed spread 3e-4); no consistent advantage, pass needed <= 0.5.
- C3 vison-dressed Z2 spin liquid (C3-T0-1 KILL, C3-FN KILL at 4x6; theorie jobs 17463650, 17465189; ansatz_runs/
  c3_t0_4x6, c3_fn_4x6; verified VERDICT line): at equal rank 20, vison span E_var 7.36e-4 vs random no-flux
  perturbations of the same bonds 4.1/5.7/4.3e-4 (V/R' 1.3-1.8); E_FN 4.6e-4 vs 2.7-3.9e-4. Flux fluctuations carry
  less than amplitude fluctuations of the same bonds; the 4x4 gain (1.1e-5 vs 5e-4) was the 4x4 anomaly. Caveat:
  base AA optimisation stopped at 60 evaluations (overlap 0.889); d_xy on (1,1) not (2,2) bonds (Hu et al.).
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
- STATUS (in progress, medium): Eight structured ansatz ideas tested on exact lattices, five kills sound and three being re-tested after an adversarial review; none beats the baselines so far.
- NEXT (open, medium): Re-test the three weak kills (spin-liquid vison sum, unfrustrated parent, coupled output), then close the angle with the memo.
