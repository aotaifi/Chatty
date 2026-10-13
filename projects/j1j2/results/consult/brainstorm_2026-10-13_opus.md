# Fresh-eyes brainstorm on the fixed-cost storage obstacle (2026-10-13, Opus)

Input read: only `OBSTACLE_2026-10-13.md` (facts 1-5). No design files or project notes were read, by instruction.
Energies "per site" below mean E/N - E_0/N.

## 0. A framing point before the ideas

Any representation with fixed capacity and a fixed evaluation cost is a variational family. A loop that stores each step in
that family can at best converge to the family's best member, which is what brute-force SR/minSR training aims at. Facts 2-3
say the near-critical ground state is outside what current fixed-cost families can reach on the low-weight sector
(configurations below the typical-set probability, about 8% of the weight at 6x6, where a Lanczos step does its work).
So "store F(psi_k) in a fixed-cost net, repeat" cannot beat a converged NQS of the same family, however clever the loss is.
A viable design has to drop at least one of these assumptions:

- (A) the end product is a stored deterministic wavefunction (it could instead be a stochastic projector that carries the
  parts nobody can store);
- (B) the stored object has to be a wavefunction at all (a bound may only need a sign field plus bond ratios on a few bonds);
- (C) capacity is fixed (it could grow linearly in k, which is still far below (2.3N)^p);
- (D) a log-additive network in the S^z basis is the representation (determinants or a rotated basis could be used instead);
- (E) the final estimator only uses the stored object (an expensive exact estimator could be corrected cheaply with a
  surrogate).

The ideas below each drop one of these. They are ranked by (chance of working) x (scalability) in section 2.

## 1. Ideas

### Idea 1: bond-field fixed node. Store a sign field and a reciprocal bad-bond ratio field, not a wavefunction (drops A and B)

**Idea.** The lattice fixed-node bound (ten Haaf et al., PRB 51, 13039, 1995) has a property that seems to be unused here. Its
proof needs only (i) a sign field s(x), which defines the "bad" bonds where s(x) H_xy s(y) > 0, and (ii) a positive number
r_xy on each bad bond with r_yx = 1/r_xy, which enters the diagonal wall term V(x) = sum_bad |H_xy| r_xy. The inequality
<phi|H_eff - H|phi> >= 0 is an AM-GM bound, applied bond by bond:
|H_xy|(r phi_x^2 + phi_y^2/r - 2|phi_x phi_y|) >= 0, and this holds for **any** reciprocal r. Nothing requires r_xy to be
a ratio psi(y)/psi(x) of one global function. So E_FN(s, r) >= E_0 for any sign field and any reciprocal bond field. The
importance-sampling (guiding) amplitude psi_G of the GFMC run is then a third, separate object: it affects only the
variance, not the bias. The mixed estimator of H_eff with any positive psi_G returns the eigenvalue of H_eff.

The loop then becomes:
1. Improve s with stored one-hop sign steps. Fact 5 says this works at fixed cost.
2. Train a bond network r_theta(x, b), antisymmetric in log, that is only ever queried on bad bonds. Its objective is E_FN
   itself, not pointwise fidelity.
3. Let GFMC (Green-function Monte Carlo, a stochastic imaginary-time projector) on H_eff(s, r) generate all the amplitude
   structure, tails included. That structure is never stored.

**Two exact statements that make this attractive.**
(a) min over r of E_FN(s, r) = E_s*, the best variational energy of any state with sign s. Proof sketch: take r from the
best amplitude a*, then E_FN <= E_var(s a*) = E_s*. Conversely, phi_FN has sign s, so E_FN >= <phi_FN|H|phi_FN> >= E_s*.
(b) The update r <- |phi_FN(y)/phi_FN(x)| on bad bonds is a guaranteed-descent coordinate step. With phi fixed, that choice of
r makes <phi|H_eff(r)|phi> = <phi|H|phi>, which is <= E_FN(old r). Then E_FN(new r) <= <phi|H_eff(new r)|phi>.
So the loop is monotone and every iterate is an upper bound. Fitting errors cost only upper-bound quality, never validity.
Near the optimum the error in log r enters at **second order**: dE ~ sum_bad |H_xy phi_x phi_y| (cosh(d log r) - 1).

**Why it could work given the facts.**
- Fact 4 shows that the wall ratios are the lever: 9.2e-5 per site with ViT ratios, 1.95e-5 with exact ratios, same sign.
- Fact 5 shows that the sign part stores well.
- Facts 2-3 (tails, non-additivity) hurt pointwise wavefunction storage. Here only bad-bond ratios matter, they enter
  quadratically, and they no longer have to be consistent with each other around closed loops.
- The r network may take the base's own one-hop features at x as inputs (E_L, V, W). Fact 1 says these make the step
  learnable, and they are already computed for the local energy.
- There is no composition over k: the projector does the iterating.

**Main risk.** Bad bonds may point mostly into the very tail configurations where networks have errors of 0.1-0.9 in log
ratio. Then a single fitted r field recovers only part of the gap between ViT ratios and E_s*. A second risk is that E_s* for
the best storable sign is itself not below "ViT + 2 Lanczos + variance extrapolation". Then the sign, not the amplitude, is
the bottleneck. GFMC also brings its own costs: population-control bias and autocorrelation.

**Cheapest exact 6x6 test (kills or keeps it).** All of it uses the 15.8M-orbit table with deterministic Lanczos on the
sparse H_eff, and no GFMC sampling. Take s = the best stored sign net. If s and r come from symmetric nets, H_eff keeps the
lattice symmetry, so the orbit table still applies. Compute:
- (a) E_FN(s, r_ViT);
- (b) E_FN(s, r_exact);
- (c) E_s*, by iterating r <- |phi_FN ratios| exactly, which takes a handful of Lanczos solves;
- (d) E_FN(s, r_net), with a small bond net trained by the exact gradient
  dE_FN/dtheta = <phi_FN| dH_eff/dtheta |phi_FN> (Hellmann-Feynman, because H_eff is diagonal in theta).

Proposed pre-registration:
- **pass** if (d) recovers >= 70% of the (a) -> (c) gap **and** (d) lies below the ViT + 2-Lanczos + variance-extrapolated
  estimate;
- **kill the r part** if (d) recovers < 30%;
- **kill the route** if (c) is not below ViT + 2 Lanczos for the sign after 0, 1 and 2 stored sign steps.

The cost is a few CPU-hours.

**Cost, honestly.** One GFMC walker step costs one hop: about 2.3N guide evaluations plus r on the bad bonds of x, i.e. the
same as one VMC local energy. Hand estimate at 8x8 (to be measured): 2.3N is about 147. Two exact Lanczos steps cost about
147^2 = 2.2e4 evaluations per sample. GFMC needs perhaps 1e2-1e3 correlated steps per independent sample, i.e. 1.5e4-1.5e5
evaluations, so it costs about the same as p = 2 but gives the p -> infinity limit of the FN projection. Compared with
training: about 1e4 SR steps x 1e4 samples x 147 = 1.5e10 evaluations, against about 1e3 walkers x 1e4 steps x 147 = 1.5e9
for the GFMC run. This needs no retraining of other people's networks, which fits the add-on pitch. Scaling: GFMC with
fixed-node and Lanczos-improved guides was run by Sorella, Becca and co-workers on J1-J2 clusters of hundreds of sites.

**Literature.** ten Haaf et al. 1995 (lattice FN bound). Sorella, PRL 80, 4558 (1998), and Capriotti & Sorella, GFMC with
stochastic reconfiguration, applied to J1-J2. Lin, He & Lu, Chin. Phys. B 31, 080203 (2022): RBM-guided GFMC on J1-J2.
arXiv:2406.12207: PEPS-guided GFMC on J1-J2. Continuum DMC nodal gradient optimisation, arXiv:2112.14182, is the analogue of
the E_FN objective in the continuum. I found no lattice work that separates the wall ratios from the guiding function. That
was a short search, so it needs checking before any claim of novelty.

### Idea 2: released-node / Krylov-from-paths on top of Idea 1. Sample p steps instead of summing (2.3N)^p (drops "the hop sums are exact and deterministic")

**Idea.** The (2.3N)^p cost exists only because H^p psi(x) is summed exactly. A random walk of p hops with weight
prod H_xy psi_G(y)/psi_G(x) estimates the same quantity at a cost linear in p per path. Imaginary-time cross-correlations then
give a small generalized eigenproblem, as in correlation-function MC (Ceperley & Bernu 1988) or Krylov-projected QMC
(Blunt, Alavi & Booth, PRL 2015). Released-node GFMC started from the Idea-1 FN population is the simplest version. It
removes the FN bias with a controlled transient instead of exact Lanczos sums.

**Why it could work.** The sign problem in gauge s is governed by Delta_B = E_0 - E_B. Here E_B is the ground energy of the
sign-free Hamiltonian |H| in that gauge. A sign that stores well (fact 5) means few bad bonds, and so a small Delta_B. At
6x6-8x8 the "average sign" may decay slowly enough for 10-50 projection steps.

**Main risk.** The variance grows like exp(2 tau Delta_B N). It is fine at 8x8 and hopeless at about 1e3 sites unless Delta_B
per site is tiny. Fact 5's "global steps not diluted below about 1e3-1e4 sites" suggests the sign is good, but that is not the
same quantity.

**Cheapest 6x6 test.** Run Lanczos for the lowest eigenvalue of |H| in gauge s on the orbit table, for the best stored sign.
That gives Delta_B, and with it the maximum projection time at a fixed sample budget, before any sampling code is written.
Kill if the reachable projection time is shorter than the FN-to-exact relaxation time. That second quantity comes from the
same table.

**Cost.** Linear in p per path, plus an exponential variance prefactor. That beats (2.3N)^p for p >= 3 whenever
exp(2 tau Delta_B N) < (2.3N)^{p-1}.

### Idea 3: a neural Krylov subspace. Store each step's residual direction as its own net and combine them by a small eigenproblem (drops C: capacity grows linearly in k, and fit errors do not compound)

**Idea.** Do not fit the Lanczos state psi_{k+1}. Fit only the new direction phi_k ~ (H - E_k) Psi_k as a separate,
spin-only, fixed-cost net. Then solve the (k+1)x(k+1) sampled generalized eigenproblem (H_ij, S_ij) over
{psi_0, phi_1, ..., phi_k}. The final state sum_j c_j phi_j is an explicit wavefunction, so its energy is a true variational
upper bound up to statistics. Fit errors in phi_k reduce the gain of that step but do not poison later steps: the
eigenproblem picks whatever helps.

**Why it could work.** Fact 2's failure is error compounding: tails are wrong in psi_1, so the hop of psi_1 is wrong. In a
subspace, each phi_k only has to overlap with the true residual, not reproduce it. Fact 1 says a single-step target is
learnable.

**Main risk.** Wang, He & Lu, "NQS Lanczos" (arXiv:2502.01264, PRB 113, 085120 (2026)), report underfitting of Lanczos
states in exactly the frustrated regime. The residual direction may be carried by the same tail configurations, so its
spin-only fit may overlap poorly. The overlap matrix S also becomes ill-conditioned as k grows. Chen et al. (NeurIPS 2022:
RBM sums plus Lanczos on J1-J2) is the closest precedent.

**Cheapest 6x6 test.** psi_0 = ViT.
1. Compute the exact residual (H - E_0) psi_0 on the table.
2. Fit a spin-only ViT-sized net to it, weighted the same way the energy weights it.
3. Solve the 2x2 problem exactly on the table.
4. Repeat for the 3x3 problem.

Pass if step 2 of the subspace keeps >= 60% of the exact second Lanczos gain. Kill if it keeps < 30%, as the stored-state
loop did.

**Cost.** Per sample, k net evaluations plus a one-hop local energy for each: k x 2.3N. Training is k supervised fits, each
cheaper than an SR training. This is linear in k, against (2.3N)^p.

### Idea 4: multilevel / control-variate Lanczos energy. A cheap surrogate plus a few exact expensive samples (drops E: the final estimator need not use the stored object only)

**Idea.** Keep the published baseline, p Lanczos steps plus variance extrapolation, but compute it unbiasedly and cheaply.
Train a surrogate f(x) for the expensive local quantity, e.g. (H^p psi)(x)/psi(x) or the p-step local energy. Then write
E_p = <f> + <exact - f>. The first term is averaged over many cheap samples. The second uses only n_2 samples at full
(2.3N)^p cost (multilevel Monte Carlo). There is no bias, so the energy is exactly the variational p-step value up to
statistics.

**Why it could work.** Fact 2: spin-only nets are precise (0.003-0.006) on the typical set, and the splice (exact tail + net)
keeps 91%. A surrogate correlated at rho ~ 0.9-0.95 with the exact estimator cuts the expensive sample count by
1/(1 - rho^2), about 5-10x. Tail errors only cost variance, not bias.

**Main risk.** It does not change the exponential scaling in p, only its prefactor. If the tail contribution dominates the
variance of (exact - f), rho may be lower.

**Cheapest test.** This needs no new exact computation: use the existing 8x8 two-hop sample data.
1. Train a surrogate on half of the samples.
2. Measure rho and Var(exact - f)/Var(exact) on the other half.

The cost reduction is read off directly. Kill if it is below 3x.

**Cost.** n_1 C_net + n_2 (2.3N)^p, with n_2/n_1 ~ (1 - rho^2). It is a sure constant-factor win at p = 2-3, but not a
fixed-cost loop.

### Idea 5: determinantal representation with backflow. Nodes as smooth zero crossings, built-in non-local non-additivity (drops D: log-additive nets)

**Idea.** Store the improvements as a backflow correction f_ij(x) = f_ij + NN_ij(x) to the pairing matrix of a
Gutzwiller-projected Pfaffian (or hidden-fermion determinant: Robledo Moreno, Carleo, Georges & Stokes, PNAS 2022). A
Pfaffian's log is intrinsically non-additive and non-local under double swaps. Near a node the Pfaffian vanishes through
cancellation of a smooth object, instead of a log-network having to produce a steep, singular log|psi|. Fact 3 says the
residual relative to PP is heavy-tailed exactly near PP's nodes; backflow moves those nodes depending on the configuration.

**Main risk.** Fact 3 shows the PP base's nodes are significantly misplaced. Backflow may need a large network to move them,
which brings back the same capacity problem. Evaluation is O(N^3) and single-swap ratios are O(N^2) for plain PP. With backflow,
every neighbour needs a full O(N^3) re-evaluation, so one hop costs about a factor N more than for plain PP.

**Cheapest 6x6 test.** Supervised fine-tune of the existing RBM+PP with a small backflow head on the exact table, using a
tail-weighted loss. Measure the bond log-ratio error on configurations with |psi|^2 < 1e-10, against the ViT's 0.1-0.9. Also
measure the rms I(b,b') that the state reproduces. Kill if the tail error stays > 0.1 at equal parameter count.

**Cost.** It is a single fixed-cost state, with the same cost class as ViT VMC. It is useful only if it beats the ViT on the
tails.

### Idea 6: change the basis. Translation-invariant plaquette/dimer rotations that shrink bad bonds and tails (drops the S^z basis)

**Idea.** Rotate by a product of 2x2-plaquette (or dimer) unitaries, either fixed (plaquette singlet/triplet eigenbasis) or
optimised to minimise Delta_B or the FN wall term. This follows Levy & Clark, PRL 126, 216401 (2021), and "sign-optimised
QMC", arXiv:2607.24679. In the rotated basis, H has a constant factor more off-diagonal terms. In exchange the ground state
may have fewer near-node configurations, smaller non-additivity and fewer bad bonds. That helps Ideas 1-3 directly.

**Main risk.** Finding the best basis is NP-hard in general. Published gains for 2D frustrated magnets are modest: the
triangular ladder had no sign-free 2-site basis, though the sign improved. The connectivity growth eats into the gain.

**Cheapest test.** 4x6 exact (2^24 amplitudes, full vector, J2 = 0.5). Compare the plaquette eigenbasis with the S^z and
Marshall bases on three numbers: Delta_B, the FN wall term with exact amplitudes, and the tail weight / rms I(b,b'). Kill if
Delta_B and the wall term do not drop by > 2x. 6x6 would need a full 2^36 vector, so it is skipped for the first test.

**Cost.** H connectivity grows by a constant (about 4-16x per plaquette term). It is otherwise neutral, and scalable if it
helps.

### Idea 7: signed linear factors instead of log-additive corrections, with a product chain that is periodically re-distilled (drops "corrections live in log space")

**Idea.** The Lanczos factor g = 1 + alpha(E_L - E) is additive over bonds in linear space and crosses zero at newly flipped
configurations. Store g_k as a signed, linear-output net: psi_{k+1} = psi_k g_k. Keep a short chain of these factors, which
is linear in k, and distil it into the base every m steps.

**Main risk.** It probably inherits fact 2's tail problem, because g_k still has to know the current guide's one-hop ratios
from spins alone. It is the least likely to help. It is listed because the zero-crossing argument is cheap to check.

**Cheapest test.** Repeat the existing iteration-2 storage test with a linear signed head instead of a log head, on the same
6x6 data. Kill if the kept fraction is still < 31%.

**Cost.** k small nets per evaluation, linear in k.

## 2. Ranking: (chance of working) x (scalability)

| rank | idea | chance | scalability | why |
|---|---|---|---|---|
| 1 | Bond-field fixed node (sign + reciprocal bad-bond ratios, GFMC evaluator) | medium-high | high (GFMC done at hundreds of sites for J1-J2) | storing the amplitude becomes unnecessary; exact upper bound; monotone loop; uses facts 4-5 directly |
| 2 | Released-node / Krylov-from-paths on top of 1 | medium | medium (exponential in Delta_B N) | linear-in-p "Lanczos"; Delta_B is measurable exactly at 6x6 first |
| 3 | Neural Krylov subspace of residual nets | medium-low | high (linear in k) | errors do not compound, but the published NQS-Lanczos underfits in this regime |
| 4 | Multilevel / control-variate Lanczos energy | high (for a constant factor) | low (still (2.3N)^p) | a sure cost cut for the baseline; cheapest test (existing 8x8 data) |
| 5 | Backflow Pfaffian / hidden-fermion store | low-medium | medium-high | right structure for non-additivity; fact 3 warns about PP nodes |
| 6 | Basis rotation (plaquette/dimer) | low-medium | high if it works | amplifies 1-3; literature gains are modest |
| 7 | Signed linear factor chain | low | medium | probably the same tail problem |

**Recommended order.** Run Idea 1's 6x6 table test first: (a)-(d) plus E_s* for the sign after 0/1/2 stored sign steps. It
settles whether "store the sign, project the amplitude" is the route. In parallel, run Idea 4's test on the existing 8x8
two-hop data; it is free and gives the baseline a cheaper estimator either way. Run Idea 2's Delta_B computation together
with Idea 1's test, since it uses the same table and code. Ideas 3, 5 and 6 are fallbacks if Idea 1's ceiling E_s* turns out
to be limited by the sign.

## Sources
- [ten Haaf et al., lattice fixed node (summary page)](https://homes.lorentz.leidenuniv.nl/~saarloos/Correlateds/fixednode.html)
- [Sorella, GFMC with stochastic reconfiguration, cond-mat/9803107](https://arxiv.org/pdf/cond-mat/9803107)
- [Lin, He, Lu, RBM-guided GFMC for J1-J2, Chin. Phys. B 31, 080203 (2022)](https://cpb.iphy.ac.cn/article/doi/10.1088/1674-1056/ac615f)
- [PEPS-guided GFMC for J1-J2, arXiv:2406.12207](https://www.arxiv.org/pdf/2406.12207)
- [Tensor network states + GFMC, arXiv:2006.15608](https://arxiv.org/pdf/2006.15608)
- [Gradient descent optimisation of fermion nodes in DMC, arXiv:2112.14182](https://arxiv.org/pdf/2112.14182)
- [Krylov-projected QMC, Blunt, Alavi, Booth, arXiv:1409.2420](https://arxiv.org/pdf/1409.2420)
- [Wang, He, Lu, Generalized Lanczos method for NQS, arXiv:2502.01264](https://arxiv.org/html/2502.01264v2)
- [Chen et al., Systematic improvement of NQS using Lanczos, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract.html)
- [Ledinauskas & Anisimovas, Scalable imaginary-time evolution with NQS, SciPost Phys. 15, 229 (2023)](https://arxiv.org/pdf/2307.15521)
- [Levy & Clark, Mitigating the sign problem through basis rotations, arXiv:1907.02076](https://arxiv.org/pdf/1907.02076)
- [Sign-optimized Quantum Monte Carlo, arXiv:2607.24679](https://arxiv.org/pdf/2607.24679)
- [Robledo Moreno et al., hidden fermion determinant states, arXiv:2111.10420](https://arxiv.org/pdf/2111.10420v1)
