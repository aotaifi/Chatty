# PLAN_ansatz (angle lead "ansatz pioneer", started 2026-10-23; project lead's PLAN.md points here)

## Bar under review (2026-10-26, project lead): PI refocused the main project on network + Lanczos hops + FN as an add-on,
without the refit (L8: at a network's own energy optimum the one-hop refit gradient is exactly zero). If the refit is gone,
the per-round refit tax below stops being the bar; candidate replacement: E_FN of (candidate guide + K hops) vs (plain network
+ K hops) at equal cost. To confirm with the PI.

## Bar for the loop (PI, 2026-10-26)
An ansatz need not carry the whole sign in one go: the FN loop corrects the sign round by round. Judge a candidate
by what it does inside the loop (per-round refit tax with the loop's own sign step, gate tax_2 < 0.82e-6, baseline
G-tax 1.657e-6), not by sign exactness. Tests C1, C2-T0, C4, C6, C6b used one-shot sign bars; C2-L stored the
pre-image with the plain Marshall sign (one-shot by choice) and was not tested with the loop carrying the residual sign.

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

## Step back 2026-10-26 (trigger: second failed version of the stored pre-image: C2-L additive by fit amplification, exp-J2 by R2 (c))
1. Verdict: switch the evaluator, not the stored object. R2 puts the whole difficulty in one place: evaluating exp(-t H_J2)
   at a walker. The stored pre-image itself is benign (fit errors amplified locally, A 1.15 -> 1.19; residual sign
   droppable at 1.45e-5/site vs Marshall-FN 6.95e-3 at 4x6), but its sampled reconstruction costs C = e^{0.067 N} (exponential by rule).
2. Thought: this is released-node over a fixed short time (sign problem ~ e^{0.42 N t}); the O(N) way out is the one the
   FN loop already uses for its hops: a deterministic local evaluator. exp(-t h_b) on one 2-site J2 bond is exact and local,
   so psi_L(y) = Phi~(y) prod_b g_b(y) (independent-bond factors) costs O(N) amplitude ratios with no sign problem; its sign is a
   product of local sign events, which is what the R1 overlay (dilute local defects) predicts. Same form as C2-loc, which
   was only ever tried on wrong bases (|ViT|, |PP|, no-J2-hop GS), never on the true pre-image.
3. Cheapest next test: R3, the local evaluator on the exact pre-image at 4x4 and 4x6, CPU < 1 h on ws1 (below).
Passed over: the 6x6 GPU run (waits on R3; it would only add a third ln C/N point); p-tVMC-style refits of exp(-t H_J2)
(they must store the final sign again: circular); Taylor truncation of exp(-t H_J2) (extensive generator, no convergence).

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

### Re-check R1 (2026-10-26, PI asked to look again; before the 4x6 run): does a LOCAL undoing of J2 scale?
Hypothesis: violations are local (C1-T0), so a local undoing of the J2 exchange should need an N-independent strength;
the additive layer's growing cond (C2-T0b size trend) is its non-locality. Forms (code ansatz/code/local_undo.py):
exponential Phi = exp(t H_J2) psi0 (psi0 = exp(-t H_J2) Phi: a sign-free state evolved for imaginary time t under
J2), gate product Phi = prod_b (1 + a h_b)^{-1} psi0, additive as reference. Strength* = smallest strength with
W_M(Phi) <= 1e-3 W_M(psi0). 4x4: t* in (0.1, 0.2], gates a* in (-0.25, -0.1], additive cond* in (3, 5]; LM overlap
at threshold 0.53 / 0.59 / 0.61 (psi0: 0.24). Also checked: at strong undoing Phi becomes the Lieb-Mattis Neel singlet
(overlap 0.96 at additive cond 30), the trivial regime the reviewer flagged.
- Pass (local form scales): t*(4x6) <= 1.3 t*(4x4) (same for gates), LM overlap at threshold <= 0.7 at both sizes.
  Then the exponential form at 6x6 (sector Krylov with the J2 table, GPU, <= 0.5 h).
- Kill: t* (or a*) grows >= 1.5x from 4x4 to 4x6, or LM overlap at threshold >= 0.9.

#### R1 results so far (2026-10-26; ansatz_runs/recheck/lu_4x4*.json, lu_4x6.json; exact; the 4x6 fine grid is running)
Strength* for W_M(Phi) <= 1e-3 W_M(psi0):
| form | 4x4 | 4x6 (coarse grid) |
|---|---|---|
| exponential exp(t H_J2), t* | (0.14, 0.16] | (0.1, 0.2] (t = 0.2: 1.7e-5) |
| gate product, per-gate a* (gate cond 1.08-1.10) | (-0.19, -0.16] | (-0.25, -0.1] (a = -0.25: 1.7e-8) |
| additive, cond* | (3, 4] | (5, 10] (cond 5: 1.6e-3) |
LM overlap at threshold falls with N for the local forms (exp t = 0.2: 0.53 -> 0.20; psi0 itself 0.24 -> 0.052): not
the trivial regime. Additive needs more with N. Provisional statement: psi0 = exp(-t H_J2) Phi_+ with Phi_+ sign-free
and t ~ 0.15-0.2/J1 independent of N (two sizes). Also (recheck_preimage.py, rc_4x4/4x6): undoing J2 pushes Phi toward
the Lieb-Mattis Neel singlet; at moderate strength Phi stays psi0-like and is smoother in the tail (4x4, rms
bond-to-bond jump of log|amp| at decade -10: Phi(cond 3) 1.6 vs psi0 3.8).
#### R1 verdict at 4x6 (2026-10-26, ansatz_runs/recheck/lu_4x6_fine.json, read by the lead): PASS
W_M(Phi)/W_M(psi0), exp form, t = 0.10/0.12/0.14/0.16: 4x4 5.4e-2/2.1e-2/6.8e-3/8.8e-4; 4x6 5.2e-2/1.9e-2/5.4e-3/8.5e-4
(curves overlay within 25% at every t; 4x4 hits exact 0 from t = 0.18, 4x6 1.1e-4, 1.7e-5 at 0.18, 0.20). t* in (0.14, 0.16]
at both sizes (ratio <= 1.14); gates a* in (-0.19, -0.16] at both; additive cond* (3,4] -> (5,6] (grows, as predicted).
LM overlap at t = 0.16: 0.47 / 0.16 (<= 0.7). Not yet checked: an exponential-form CONTROL (e.g. exp(-t H_J1) psi0, any
positivity-improving local evolution at matched overlap with psi0); asked of the second-opinion review first.
Second opinion: Codex gpt-6-astra xhigh, read-only, job cj-baa5ea3230cf (/scratch/a/A.Otaifi/codex_jobs/cj-baa5ea3230cf).
#### Second opinion (Astra, ansatz/REVIEW_astra_2026-10-26.md) and what I take from it
Verified by me: (i) exact identity H_J2 psi0 = (E0 - H_J1) psi0, so exp(+t H_J2) and exp(-t H_J1) change psi0 identically to
first order: J2-specificity is second order (its 4x4 matched-fidelity J1 control is only 2.2x worse: unverified);
(ii) global conditioning kappa(exp(-0.16 H_J2)) = exp(0.16 x bandwidth(H_J2)) = 4.95 / 10.4 / 28.7 (4x4 / 4x6 / 6x6;
4x4 bandwidth 10 from K_{4,4} by hand): grows exponentially with N at fixed t (worst case; fit-error amplification unmeasured);
(iii) W_M saturates at 1/2 for extensive defects, so the intensive gamma_M = -ln|1 - 2 W_M| / N is the size-safe metric.
Taken: the crux is reconstruction (evaluating exp(-t H_J2)|Phi> at a walker) and an explicit residual-sign rule, not W_M.
Disputed: the R1 overlay of the ratio curves (within 25% at every t) is itself evidence of dilute local defects, which
C1 alone did not give; the 4x4 loop test it proposes would pass trivially (Phi exactly Marshall at t >= 0.18), so use 4x6.
Kept from it: J2's real distinction from a J1 filter is structural: exp(-t H_J2) is stoquastic in the stripe frame s_X,
so reconstruction is a positive propagator acting on a signed boundary vector, while J1's inverse exp(+u H_J1) is not.

### R2 (2026-10-26, before any run): does the exp-J2 pre-image survive reconstruction and amplification? (exact 4x4 + 4x6, CPU, ws1; worker; code ansatz/code/r2_recon.py)
t in {0.12, 0.14, 0.16, 0.18, 0.20}; control exp(-u H_J1) psi0 at matched |<psi0|Phi>|^2. Absolute kernel T = exp(-t H~),
H~ = S_X H_J2 S_X (check: every off-diagonal <= 0; s_X = (-1)^{n_up on x odd}).
(a) Reproduce Astra's 4x4 numbers (t* 0.15918, u* 0.19172, W_M ratios 8.80e-4 vs 1.92e-3 at F = 0.907323; eta medians 0.694
    overall, 0.0412 for |psi0|^2 < 1e-7). Any miss > 2% is reported before (b)-(e) are read.
(b) gamma_M for psi0, Phi_t, Phi_J1. J2-specific (physics reading kept) only if W_M(J1)/W_M(J2) at matched fidelity >= 3
    at both sizes; else the reading is "generic Perron filter + stoquastic reconstruction frame".
(c) Reconstruction cost: eta_y = |psi0(y)| / (T |Phi_t|)(y); C = E_{|psi0|^2}[eta^-2] (samples per unit relative accuracy).
    Exponential signal: ln C / N at 4x6 >= 0.8 x its 4x4 value. Bounded signal: ln C at 4x6 <= 1.3 x 4x4. Between: inconclusive.
(d) Fit-error amplification: Phi -> Phi e^{eps g}, g a random symmetric 2-body Jastrow normalised to rms 1 under |Phi|^2,
    eps = 0.01, 0.03, 5 draws; A = (E(exp(-t H_J2) Phi e^{eps g}) - E0) / (E(psi0 e^{eps g}) - E0). Global (kills scaling):
    A(4x6)/A(4x4) >= 1.7 (0.8 x kappa ratio 2.1). Local: <= 1.2.
(e) Oracle-free residual sign (drop it): psi_rec = exp(-t H_J2)(s_N |Phi_t|); E_var and own-sign lattice-FN energy per site
    vs Marshall-FN (s_N |psi0|) at both sizes. Continue to a loop-native C2-L only if, at some t, E_FN(psi_rec) <= 0.3 x
    Marshall-FN at both sizes AND (c), (d) give the bounded/local signal. Otherwise: write the ranked memo update.

#### R2 result (ws1 CPU, /scratch/a/A.Otaifi/ansatz_runs/r2/r2_4x4.json, r2_4x6.json, code ansatz/code/r2_recon.py; raw JSON read by the lead)
(a) Astra's 4x4 numbers reproduced, all within 0.2%. At t = 0.16 (4x4 / 4x6):
| quantity | 4x4 | 4x6 | rule | verdict |
|---|---|---|---|---|
| W_M(J1)/W_M(J2) at matched fidelity, t 0.12-0.20 | 1.20-2.18 | 1.27-2.06 (0.39 at t 0.20) | >= 3 both | J2-specific reading FAILS |
| ln C / N (C = E[eta^-2] = sign-cost of sampled reconstruction) | 0.0760 | 0.0669 | ratio 0.88 >= 0.8 | EXPONENTIAL |
| ln C | 1.217 | 1.606 | ratio 1.32 > 1.3 | not bounded |
| A, eps 0.01, 5 draws (mean +- sd) | 1.15 +- 0.45 | 1.19 +- 0.32 | ratio 1.03 <= 1.2 | LOCAL |
| E_FN(psi_rec)/site, residual sign dropped | 6.6e-6 | 1.45e-5 (2.3e-7 at t 0.20) | <= 0.3 x Marshall-FN (7.3e-3, 6.95e-3) | PASS |
Rule: continue to a loop-native C2-L only if (c) bounded and (d) local and (e) pass: (c) fails, so NOT continued as designed.
Checks: s_X frame stoquastic (max off-diag -0.25; s_N +0.25 fails); round trip 7e-17 (wrong sign 0.2-0.64); FN(psi0) - E0 -3e-13.
4x6 FN uses LOBPCG (eigsh stalled; matched eigsh to 1e-12 at 4x4). Caveats: Jastrow draws not paired across sizes (5 vs
11 classes); 4x4 is exactly Marshall from t 0.18. Extrapolated sampled-reconstruction cost C = e^{0.067 N}: 72 (8x8), 1.5e4 (12x12), 2.6e9 (18x18).

#### Debate round 1 with Astra (cj-f4d4b2023b16, appended to ansatz/REVIEW_astra_2026-10-26.md): what I concede
- WRONG (mine): "J2's reconstruction kernel is positive, J1's is not": exp(+u H_J1) is entrywise >= 0 in the Ising basis
  (-H_J1 stoquastic). Both routes are a positive kernel on a signed boundary vector; no structural J2 advantage remains.
- R1 overlay = an intensive suppression scale, not proof of local independent defects (a collective sector with an O(1)
  gap gives the same overlay at width 4). Locality test: connected double-swap response vs separation.
- R2 (c): the rule fired, but two points also fit C ~ N^0.96 (C/N 0.211 vs 0.208): exponential vs polynomial is undecided;
  a third size (6x6) separates them (exp predicts C(36) ~ 8.9, linear ~ 7.5 at t = 0.16).
- "Benign" narrowed to: smooth 2-body Jastrow errors amplified ~1.2x at two sizes; learnability and FN tax unmeasured.
- R3's product decoder psi_L ~ f^{1-M} prod_b (a f + b f_b) divides by f: near-zeros of |Phi| (where the dropped residual
  sign lives) blow up. Astra's own 4x4: t 0.16 overlap 2.3e-6, E_var 0.654/site; t 0.20 overlap 0.979, E_var 3.2e-3.
  The R3 worker reproduces t 0.16 independently (overlap 2.2853e-6, E_var 0.65362; r3_4x4.log). J1 control: overlap 0.976.

### R3 (2026-10-26, before any run): an O(N) local evaluator of psi0 from the stored pre-image (exact 4x4 + 4x6, CPU ws1, < 1 h; extends r2_recon.py)
Stored: Phi~ = s_N |Phi_t| (residual sign dropped), t in {0.16, 0.20}. One-bond factor g_b(y) = (exp(-t h_b) Phi~)(y) / Phi~(y)
(exact 2-site action, h_b = J2 S_i.S_j). L1: psi_L(y) = Phi~(y) prod_{b in J2} g_b(y). Cost: 2N ratio evaluations per amplitude.
Control: the same construction for the J1 pre-image Phi_J1 = exp(-u H_J1) psi0 at matched fidelity, factors exp(+u h_b) over J1 bonds.
Measured: sign-error weight sum_{sign psi_L != sign psi0} |psi0|^2; E_var and own-sign E_FN per site; references: exact
reconstruction E_FN (R2 (e)) and Marshall-FN. Check that can fail: with a single bond in the product (all others g = 1) on a
Phi~ built as exp(+t h_b) psi0 for that one bond, L1 must return psi0 to 1e-12 (and not with t of the wrong sign).
- Pass: E_FN(psi_L) <= 0.1 x Marshall-FN at both sizes, and E_FN/site grows < 1.5x from 4x4 to 4x6. Pass makes psi_L =
  (positive network) x (local J2 factors) the loop-native candidate: then 6x6 (GPU <= 0.5 h) and the fit-based loop tax.
- Kill: E_FN(psi_L) >= 0.5 x Marshall-FN at both sizes: the reconstruction is non-local, the stored pre-image closes at
  "mild exponential, C = e^{0.067 N}". Between: inconclusive, read the sign-error weight by decade.

#### R3 result (ws1 CPU, /scratch/a/A.Otaifi/ansatz_runs/r3/r3_4x4.json, r3_4x6.json, code ansatz/code/r3_local.py; raw JSON read by the lead)
overlap | E_var/site | E_FN/site (Marshall-FN 7.27e-3 / 6.95e-3): 4x4 J2 t.16 2.3e-6 | 0.654 | 9.7e-3 (truncated FN); J2 t.20
0.979 | 3.2e-3 | 2.8e-3; J1 t.16 0.976 | 4.7e-3 | 2.7e-3. 4x6 J2 t.16 2.8e-9 | 0.585 | 1.25e-2; J2 t.20 2.1e-7 | 0.460 | 2.5e-3;
J1 t.16 2.8e-6 | 0.418 | 9.4e-3; J1 t.20 1.8e-8 | 0.543 | 3.6e-3. Verdict: J2 t 0.16 KILL (1.34x / 1.80x Marshall-FN); t 0.20
INCONCLUSIVE by the rule (0.385x / 0.36x) but amplitudes destroyed at 4x6 (overlap 2e-7), unusable as a guide. Cause: the
product divides by near-zeros of |Phi| (g_b down to -928; |psi_L| spans 6e-47..0.125). Checks pass and can fail.
Debate round 2 (cj-ab59faf861ee): agrees with the R3 verdict and with closing; no remaining disagreement that changes a decision.
Its corrections, accepted: R3 rejects two decoders (additive, independent-bond product), not every local reconstruction; the
oracle-magnitude result is approximate sign recovery (478x below Marshall-FN), not "fully carried"; learning the magnitudes
is a separate open problem; C ~ e^{0.067N} vs ~N undecided; my round-2 prompt misnamed the C2-T1 bases (they were
|psi0|-Marshall, Marshall-FN, no-J2-hop GS, as in Dead ends; full-H won 4.45x / 5.55x / 1.03x). Under the no-refit refocus it
sees no role for partial-H hops (no positive evidence) and recommends retiring the refit-tax gate.
Recommendation to the PI: close the angle, no further compute; memo update sorted failure / hint / insight.

### C2-L-exp (2026-10-26 18:50, before any run; PI: "go ahead and takeover"): the exp-form pre-image in the FN loop (exact 6x6, theorie A40)
Why reopened: I stopped R1 step 3 on R2's two-point extrapolation (C/N 0.211 vs 0.208: exponential vs linear undecided;
C ~ 70 at 8x8 even if exponential). The PI wants a loop candidate with the refit. Target and readout as C2-L (G-tax phi_0,
K_k, FN; baseline tax_1 5.107e-6, tax_2 1.657e-6, gate 0.82e-6). Code ansatz/code/c2le_gtax.py (copy of c2l_gtax.py), runs
/scratch/a/A.Otaifi/ansatz_runs/c2le_6x6/. Phi_t = exp(+t H_J2) phi_0, store s_N|Phi_t| (exact) or s_N e^{f_fit}; psi = exp(-t H_J2)(stored).
- Step 0: Astra round 3 GO / NO-GO (cj-5b12c48420cc). A derived NO-GO stops it; a GO WITH CHANGES is adopted if it alters what we learn.
- Step 1 exact store, t in {0.12, 0.16, 0.20, 0.24} (<= 1.5 A40-h): KILL if tax_B(2) >= 1.657e-6 at every t.
- Step 2 positive RBM (2442 params, C2-L-1 arm (ii) settings) at the best t of step 1 (lowest tax_B(2) with W_M ratio <= 1e-2)
  (<= 5 A40-h): PASS tax_B(1) < 1.657e-6 and tax_B(2) < 0.82e-6; KILL tax_B(2) >= 1.657e-6 (inconclusive only if held-out
  loss falls > 1% per 10 steps over the last 50); fit rms by decade beside the rebuilt state's log error by decade.
- Step 3 (PASS only): rebuild cost on 8x8 walkers. Ceiling for the whole takeover: 8 A40-h, no H100.
- Amendment after Astra round 3 (GO WITH CHANGES, before any run; adopted): fit-arm t = the SMALLEST t whose exact-store
  tax_B(1) < 1.657e-6 and tax_B(2) < 0.82e-6 (else lowest tax_B(2); W_M condition dropped); same-fit sign(Phi_t) readout as a
  diagnostic only; actual-fit-error control: tax of phi_0 e^{eps}, eps = log|Phi_fit| - log|Phi_t| (no decoder), decade bins
  by |phi_0|^2; taxes converged to 1e-8/site; W_M, t* labelled FN-target diagnostics. Astra: no derivation rules out a pass
  (exp t 0.20 damps mid-spectrum errors 8.2x vs additive cond 30's 15.5x, reversed at the low end 66 vs 30); its central
  estimate tax_B(2) ~ 1e-5 (range 1e-6..1e-4), P(pass) ~ 10%; mine ~25%. Conceded: C ~ 70 at 8x8 is NOT shown affordable
  (plain signed sampling needs ~(C-1)/eps^2 ~ 7e5 paths per configuration at 1%); step 3 stays a separate gate.

Next, in order: (1) [done: PASS] read lu_4x6_fine.json and decide R1 by its rule; (2) on PASS, the exponential form at exact 6x6
(sector Krylov expm with the J2 table ansatz_runs/c2_t0b_6x6/tables/nb_J2.npy and sector_ops.py; GPU <= 0.5 h): t*, LM
overlap; (3) on PASS at 6x6, pre-register the loop-native C2-L: store |Phi_t| with the loop carrying the residual
sign, reconstruct psi = exp(-t H_J2)(s * |Phi_fit|) exactly, K_2 + FN, tax vs G-tax 1.657e-6, positive-RBM fit.
Open risk: applying exp(-t H_J2) to walkers at 8x8+ is not designed (a short-time sign problem or a local expansion).

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
- C6 re-test after review (C6-T0b KILL, 4x4, cip 17470732, ansatz_runs/c6_t0/r4x4b, verified): optimising the
  stoquastic couplings directly for the dressed E_var, 5 starts incl. -H_J2 + H_J1, all end at 1.024e-2/site vs the
  dressed-Marshall-FN baseline 4.17e-3 (2.45x worse); every same-sublattice XY goes to 0; GS fidelity with Phi 0.974.
  Checks: exact H gives 2e-16, J2 XY flipped 8.7e-2. Kill sound: a sign-free few-body parent cannot carry the J2 XY.
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
- C5 continuation after review (cip 17470740-2, exact_vmc_4x6/cont/verdict.json, verified): coupled/standard E_var
  0.65 (R1), 1.02 (R2), 0.96 (R4), 3 seeds each; still INCONCLUSIVE by the rule, no pass in reach (bar 0.5). The
  standard arm never learns a sign (W_M = 0), so the R1 gain is the coupled arm flipping a few signs. Closed.
- C5 amplitude-sign coupled output (C5-T0 INCONCLUSIVE, parked; jobs 17469983/4): coupled/standard E_var 0.73 (R1),
  1.30 (R2, one seed), 0.97 (R4, within seed spread 3e-4); no consistent advantage, pass needed <= 0.5.
- C3 re-test after review (C3-T0-2 KILL, C3-FN KILL; cip 17470728, ansatz_runs/c3_t0_2_4x6, verified VERDICT line):
  base was already converged (overlap 0.8889297 at 60 and 111 evaluations); derivative states move (S) only 5.28e-3 ->
  5.23e-3; at rank 24, V 6.36e-4 vs R' 2.8/4.1/3.3e-4 (V/R' 1.55-2.29); E_FN V 4.0e-4 vs R' 1.9-2.8e-4. Kill sound.
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
- 2026-10-26: REOPENED at the PI's request (look again, judge by the loop's bar, not one-shot sign); R1 started.
- 2026-10-24: ANGLE CLOSED (day 2 of ~5): all candidates dead or closed after an adversarial review and re-tests;
  memo ansatz/ANSATZ_MEMO.md; handoff handoffs/2026-10-24-ansatz-pioneer.md. Compute used ~10 A40-h, 0 H100-h, plus CPU jobs.
- 2026-10-23: plan created; greeting with blind-spot reading and C1-C3 sent to the PI; C4 added.
- 2026-10-23 evening: C1 dead; C2 lead (4x4, 4x6); C3 inconclusive at 4x4, 4x6 running; asked the PI what counts
  as a breakthrough; Paderborn gpu_h100 queue est. 5 days (QoS normal not allowed there; default cont used).

## Cockpit
- STATUS (in progress, medium): Reopened at the PI's request: testing whether the nearly sign-free pre-image, stored by a small positive network, makes one loop round cheaper than today's refit.
- NEXT (in progress, medium): Exact 6x6 test without a fit first, then with the fitted network; pass means the loop gets its first working candidate.
