# Does the design-B loop converge to the ground state? (2026-11-17)

Worker report on paper, with toy numerics (dimension <= 11, numpy). Scripts: `/scratch/a/A.Otaifi/theory_checks/2026-11-17_loop_convergence/`
(`loop_lib.py` + `check*.py`); the core of each is inline in section 5.

## Answers first

- **Overall.** Not from *every* start with nonzero ground-state overlap: there are such starts that land exactly on an excited
  eigenstate in one round (checked, section 4.3). But they are non-generic, excited eigenstates are unstable, and the loop with
  optimal alpha and a perfect copy cannot cycle or stop at a non-eigenstate fixed point. 3000/3000 random starts on random
  frustrated toys reached the ground state. A full proof of convergence still needs one lemma (Q2).
- **Q1.** Yes, a low excited eigenstate psi_n is a fixed point, as long as its kept graph is connected. It is unstable in two
  ways. (i) Energy: once E_var < E_n the loop can never return to psi_n, and such states lie arbitrarily close to psi_n.
  (ii) Linearisation: the FN step maps a guide perturbation delta' to delta phi = R W delta'. This map has exactly n eigenvalues
  mu > 1 (n = number of levels below E_n) and all others in [0, 1). The optimal Lanczos step multiplies every component below
  E_n by more than 1, or drops the energy below E_n by an O(1) amount. Checked: n expanding eigenvalues in 214/214 cases; the
  finite-difference Jacobian matches R W to a median of 7e-10.
- **Q2.** Monotone descent gives convergence to *some* eigenstate (Zangwill's theorem) provided the loop map is continuous at
  every limit point that is not an eigenstate. The map is continuous except at two kinds of guide: (a) a guide zero at a
  configuration whose nonzero neighbours all have one sign (the FN output jumps by O(1), checked); (b) a degenerate FN ground
  state (disconnected kept graph). Missing lemma: the orbit stays a finite distance from these sets. Lanczos (alpha != 0) is
  what avoids them. The frozen-sign loop (alpha = 0) does stall there: 59/59 wrong-sign starts stopped at non-eigenstates.
- **Q3.** E_FN[psi'_0] < E_1 is a correct sufficient condition, given the Q2 lemma. Without that lemma it still gives a
  rigorous, non-decreasing lower bound on the ground-state weight (Eckart). A sharper version is a run-time certificate: any
  round k with E_FN[psi'_k] < E_1 suffices. More generally, once E_var < E_n the loop can never converge to psi_n or anything
  above it. The condition is not necessary: most of the 3000 toy starts began above E_1.
- **Q4.** With optimal alpha and a perfect copy, a cycle is impossible (strict descent) and a stall would need the Q2 failure.
  None appeared in 3000 runs. Counterexamples I built and checked:
  (i) a start with ground-state overlap -0.214 whose loop sits exactly on psi_1 from round 1 on (N = 6);
  (ii) a fixed alpha that is too large gives an exact 2-cycle between non-eigenstates (N = 8);
  (iii) alpha = 0 stalls at non-eigenstates.
- **Imperfect copy.** Descent survives a round iff eps_k <= Delta_k (that round's gain). Near the ground state,
  limsup (E_var - E_0) <= eps/(1 - rho), with rho the local energy contraction per round. Toy: the stall matched to 4%.
  Globally, the ground state is guaranteed (Eckart) if E_var[psi_0] + sum_k eps_k < E_1.
- **Fixed alpha.** Every conclusion survives if 0 < -alpha <= 1/(E_max - E_var[psi_k]); |alpha| <= 1/(E_max - E_0) suffices.
  The map is then also linear and continuous in step 1. Beyond that bound monotonicity can fail, and the 2-cycle appears.

## 1. Notation and conventions

H real symmetric on X, H_xy >= 0 off-diagonal. psi = S A. Guide g. A bond is violating if H_xy g(x) g(y) > 0, i.e.
g(x), g(y) have the same sign; it is kept if they have opposite signs. H_FN[g] as in the brief. W[g] := H_FN[g] - H.
E_var[chi] = <chi|H|chi>/<chi|chi>. E_FN[g] = lowest eigenvalue of H_FN[g]. Levels E_0 < E_1 <= ... in the sector.
Same-sign clusters of g: connected components of the graph of violating bonds. The kept graph of g is the graph of kept bonds.
**Guide zeros (wall convention).** If g(x) = 0, then V(x) = +infinity, and x is removed from H_FN together with its bonds; phi(x) = 0.
This is the limit g(x) -> 0 from either side when x has nonzero neighbours of both signs (section 3.3).

## 2. Facts used, each checked

F1 [std, ten Haaf et al. cond-mat/9412037 Eqs. 1-4] H_FN[g] g = H g. W[g] = sum over violating bonds of H_xy [[r, -1], [-1, 1/r]], r = g(y)/g(x) > 0.
   So W >= 0 and W g = 0.
F2 [std, Perron-Frobenius] In g's sign basis, H_FN[g] has off-diagonals -H_xy on kept bonds and 0 elsewhere. If the kept graph is
   connected, its ground state is nondegenerate and has strictly g's sign everywhere.
F3 [own, from the definition] H_FN[g] depends on g only through sgn g and the ratios g(y)/g(x) inside each same-sign cluster.
   V sums only over violating bonds, and violating bonds join only same-sign configurations.
   So rescaling a whole cluster by lambda > 0 does not change H_FN.
F4 [std, 2x2 Ritz problem] Let sigma = ||(H - E)psi|| / ||psi||. In the orthonormal basis {psi, v}, v = (H - E)psi/(sigma ||psi||), the Ritz matrix is
   [[E, sigma], [sigma, E + D]], with D = <v|(H - E)|v>. Its lowest eigenvalue is < E strictly iff sigma > 0, i.e. iff psi is not an eigenstate.

**Chain (a), checked: true.**
- E_var[psi'] <= E_var[psi] (F4).
- E_FN[psi'] <= <psi'|H_FN|psi'>/<psi'|psi'> = E_var[psi'] (F1).
- E_var[phi] = E_FN[psi'] - <phi|W|phi> <= E_FN[psi'] (F1).
- Perfect copy: E_var[psi_{k+1}] = E_var[phi_k].
So E_var[psi_k] is non-increasing. The wall convention keeps every link: for chi with chi(x) = 0 at the walls, the removed rows
contribute nothing to either side.
Numerics: 3000 runs (300 random H, N = 3-10, 10 starts each, 300 rounds). 39 link violations, all in rounds whose guide had
min|psi'|/max|psi'| <= 4e-16: V ~ 1e16 and eigh round-off. Rerun with configurations at |psi'| <= 1e-12 max deleted exactly
(check1d): 0 violations, including the 97 rounds whose guide had zeros. Exact zeros arise naturally, from FN outputs
supported on one kept-graph block.

**Fixed points (b), checked: true, but the stated reason is not the proof.** H_FN g = H g alone does not give it, because the guide
is psi', not psi. Proof [own]: at a fixed point psi_{k+1} = psi_k, so every link of the chain is an equality. In particular
E_var[psi'] = E_var[psi], so sigma = 0 by F4, so psi is an eigenstate.

**Strict descent [own].** E_var[psi_{k+1}] < E_var[psi_k] whenever psi_k is not an eigenstate (F4 is strict). Consequences:
no cycles of any period, and E_var[psi_k] -> E* >= E_0.

## 3. Derivations

### 3.1 Q1: excited eigenstates are fixed points

Let psi_n (normalised) be an eigenstate with no zeros whose kept graph is connected. Then sigma = 0, so psi' = psi_n.
By F1, H_FN[psi_n] psi_n = E_n psi_n. psi_n has the guide's sign everywhere, so by F2 it is the FN ground state: E_FN = E_n and phi = psi_n.
The same holds with zeros under the wall convention: (H_FN psi_n)(y) = (H psi_n)(y) because psi_n vanishes on the walls.
If the kept graph is disconnected, every block has ground energy E_n and the FN ground state is degenerate, so step 2 is not
defined. Toys, N = 8, fraction of eigenstates with a degenerate FN ground state: n = 1: 0.00, n = 2: 0.00, n = 3: 0.04,
n = 6: 0.50, n = 7: 1.00. The top state is the Perron vector of H, whose kept graph is empty. So low levels are clean fixed points.

### 3.2 Q1: instability

**(i) Energy argument [own, rigorous].** Assume E_n is nondegenerate and n >= 1, and that the loop map T is continuous on U \ {psi_n}
for a neighbourhood U of psi_n. (It is: no other eigenstate lies in U, Lanczos is continuous off eigenstates, and the FN step is
continuous near the guide psi_n by F2 and 3.3.)
Take any start in U with E_var < E_n. Such starts lie arbitrarily close, e.g. psi_n + eps psi_0.
- The orbit never converges to psi_n, since E_var[psi_k] <= E_var[start] < E_n.
- If the orbit stayed in a closed U, it would have a limit point p != psi_n with E_var[p] = E*.
- Continuity at p gives E_var[T p] = E*, but strict descent gives E_var[T p] < E*. Contradiction.
So the orbit leaves U: psi_n is Lyapunov-unstable.

**(ii) Linearisation of the FN step [own].** psi_n has no zeros and a connected kept graph, with FN gap Delta_FN > 0.
Set H_FN0 = H_FN[psi_n], W = H_FN0 - H, and R = Q (H_FN0 - E_n)^{-1} Q, where Q = 1 - |psi_n><psi_n|.
Take the guide g = psi_n + delta'. For small delta' the sign does not change, so the violating set is fixed. Then

  dV(x) psi_n(x) = sum_viol H_xy (delta'_y - r_xy delta'_x) = -(W delta')(x),   r_xy = psi_n(y)/psi_n(x).

First-order perturbation theory gives

  delta phi = R W delta'                                                                   (L_FN)

(<psi_n|W delta'> = 0 because W psi_n = 0).
Check: if psi_n has no violating bonds (n = 0 with an exact sign structure), then W = 0 and delta phi = 0. FN with exact nodes
ignores the guide amplitude, as it should.
Spectrum: R W is similar to R^{1/2} W R^{1/2} >= 0, so its eigenvalues mu are real and >= 0. W v = mu (H_FN0 - E_n) v is equivalent to

  (H - E_n) v = (1 - mu)(H_FN0 - E_n) v   on Q.

H_FN0 - E_n > 0 on Q. So by Sylvester's law of inertia [std], the number of mu > 1 equals the number of negative eigenvalues of
(H - E_n) on Q, which is n. The others satisfy 0 <= mu < 1; mu = 0 is the kernel of W, the cluster rescalings of F3.
So the FN step alone has exactly n expanding directions, and **at the ground state (n = 0) the FN step is a contraction**.
Numerics: 214/214 eigenstates (60 random H, N = 4-8) had #(mu > 1) = n. FD Jacobian vs R W: median max-difference
7e-10 at h = 1e-6. The worst case, 1.6e-3 at h = 1e-7, had FN gap 1.5e-4: second-order terms.
Example, N = 8, mu sorted:
- n = 0: 0.72, 0.68, ...
- n = 1: 4.41, 0.89, ...
- n = 2: 3.17, 1.03, 0.96, ...
- n = 3: 3.54, 1.96, 1.30, 0.96, ...

**(iii) The optimal Lanczos step near psi_n [own].** Write delta = sum_{m != n} c_m psi_m and
D(delta) = sum (E_m - E_n)^3 c_m^2 / sum (E_m - E_n)^2 c_m^2. This D is the D of F4 to leading order, is O(1), and depends
only on the direction of delta.
- D > 0: alpha = -1/D + O(delta^2) and c_m -> (1 + (E_n - E_m)/D) c_m. Every component below E_n (m < n) grows. Components with
  E_m - E_n > 2D also grow in norm, but not in energy.
- D < 0: the Ritz value is <= E_n + D + o(1), an O(1) drop below E_n. The orbit leaves at once (case (i)).
- The step is homogeneous of degree 1 in delta but not linear, and it is discontinuous at psi_n (alpha depends on the direction).
  So "the linearisation of the loop" exists only piecewise. The energy argument (i) does not need it.
- Pure ground-state perturbation, delta = eps psi_0: span{psi, H psi} = span{psi_n, psi_0}, and Lanczos returns psi_0 exactly in one round.

**Escape rate, checked:** psi_1 + 1e-10 noise reached E_var < E_1 - 1e-3 after 10 rounds (1e-6 noise: 4 rounds), then reached
the ground state to 1e-15.

### 3.3 Signs near psi_n, and where the FN map is discontinuous

- **psi_n without zeros:** no sign flips for |delta| < min|psi_n|.
- **Zero of psi_n at x:** (H psi_n)(x) = 0 forces both signs among the nonzero neighbours: sum_{psi_n(y)>0} H_xy |psi_n(y)| =
  sum_{psi_n(y)<0} H_xy |psi_n(y)|. To first order the FN equation at x then gives phi(x) = g(x) * (kept sum)/(violating sum)
  = g(x), whichever sign g(x) takes. Neighbour rows see H_yx g(x) whether the bond is kept or violating. So the FN map is
  differentiable through the flip [own].
  The Jacobian is block-triangular, with mu = 1 exactly on each zero site (neutral). The expanding count becomes the number of
  levels of H restricted to the nonzero configurations below E_n; by Cauchy interlacing this is between n - |Z| and n.
  Checked (N = 6, n = 4, one zero): one-sided derivatives along e_x agree to 3e-5 at h = 1e-6. Jacobian moduli are
  12.3, 8.2, 2.06, 1.00, 0, and the restricted H has 3 levels below E_n.
- **Discontinuity (a):** a guide zero at x whose nonzero neighbours all share one sign s.
  - g(x) -> 0 with sign s: every bond of x is violating, x is isolated in the kept graph, and phi(x) = 0.
  - Opposite sign: every bond is kept, V(x) = 0, and phi(x) = O(1).
  Checked: phi(x) = 0 versus -0.584 at g(x) = +-1e-3, +-1e-6, +-1e-9.
- **Discontinuity (b):** a disconnected kept graph means a possibly degenerate FN ground state, and a jump between blocks under a
  small change of the guide. It occurs in practice: the 39 round-off events above came from guides with exact zeros produced by
  FN outputs supported on one block.

### 3.4 Q2: convergence to some eigenstate

[std, Zangwill's global convergence theorem; Luenberger, Linear and Nonlinear Programming, sec. 7.7] Three conditions:
(1) the iterates lie in a compact set (the unit sphere, mod sign);
(2) E_var is continuous and strictly decreasing off the solution set (the eigenstates) — shown in section 2;
(3) T is continuous at every non-solution point.
Then every limit point is an eigenstate. Since E_var[psi_k] -> E*, every limit point is an eigenstate at E*. If E* is a
nondegenerate level, its eigenstate is isolated mod sign, so the orbit converges to it.

Condition (3) holds at psi unless psi' lies in the discontinuity set D = (a) or (b) of 3.3. The proof needs only continuity at the
limit points, so the **missing lemma** is: *the orbit's limit points have psi' outside D.*
Why it can fail: alpha = 0 (FN with its own state as guide). The sign is frozen, E_var still falls strictly, and no fixed point
can exist unless the start already has an eigenstate's sign. So the orbit must run into D: amplitudes on "wrong-sign"
configurations decay to 0.
Checked: 59/59 wrong-sign starts (N = 6) ended at non-eigenstates, for example E_var - E_0 = 0.149 < E_1 - E_0 = 0.667 with
residual |H psi - E psi| = 0.458, three configurations at |psi| -> 0 (down to 4e-22), and no sign flips.
The Lanczos step is the only step that can change a sign: at |psi(x)| -> 0, psi'(x) -> alpha (H psi)(x). That is why I expect
the lemma to hold for alpha != 0. It is not proven.

### 3.5 Q3: sufficient conditions

The chain gives E_var[psi_j] <= E_FN[psi'_k] for all j > k. So if E_FN[psi'_k] < E_1 at any round k, every later limit eigenstate
lies below E_1, hence is the ground state (needs the Q2 lemma).
Without the lemma [std, Eckart]: E_var >= w E_0 + (1 - w) E_1 gives the ground-state weight
w_k >= (E_1 - E_var[psi_k])/(E_1 - E_0), which is non-decreasing in k. So the brief's condition is right, and the run-time version
(any round) is sharper. In general, the loop can converge only to a level at or below E_var[psi_k] for every k.
Practical point: this needs E_1 in the sector the loop preserves (e.g. S_z = 0 includes the S = 1 triplets unless total S is fixed).
I did not check whether the 8x8 FN energies are below E_1.

### 3.6 Q4: counterexample to "any start with nonzero ground-state overlap" [own, checked]

By F3, every guide in F_n = {g: sgn g = sgn psi_n, and psi_n's ratios inside each same-sign cluster} has FN ground state exactly
psi_n. A state with both signs has >= 2 clusters, so F_n contains guides with <psi_0|g> != 0. Every such g has
E_var[g] >= E_FN = E_n, consistent with 3.5.
N = 6 instance (seed 7; E = -1.535, -0.877, -0.173, 0.012, 0.828, 2.995):
- psi_1 has clusters [0 0 1 0 2 2]. Rescaling cluster 0 by 3 gives a guide with <psi_0|g>/|g| = -0.183 and FN overlap
  |<psi_1|phi>| = 1 - 1e-15.
- A stored start whose optimal-Lanczos guide lies in F_1 (solved by least squares, residual 3e-16):
  psi = (0.071, 0.253, -0.887, -0.046, 0.202, 0.318), with <psi_0|psi> = -0.214 and E_var = 0.0237.
  Rounds 1-4: E_var = -0.8772816836105 = E_1, which is exact.
- The set of such starts is a cluster-rescaling family pulled back through Lanczos, so I expect it to have measure zero.
  Not proven: the map is not a local diffeomorphism (R W has a kernel), so the stable-manifold argument of Lee et al.
  arXiv:1602.04915 does not apply directly.

## 4. Imperfect copy and fixed alpha

**Copy error** psi_{k+1} = phi_k + e_k, with E_var[psi_{k+1}] <= E_var[phi_k] + eps_k.
- Then E_var[psi_{k+1}] <= E_var[psi_k] - Delta_k + eps_k, where Delta_k = E_var[psi_k] - E_var[phi_k] >= 0 sums the gains of
  the Lanczos, FN and W links. Descent holds iff eps_k <= Delta_k [own].
- Near the ground state, take e = E_var - E_0 and E_var[phi_k] - E_0 <= rho (E_var[psi_k] - E_0). Then e_{k+1} <= rho e_k + eps,
  so limsup e <= eps/(1 - rho) [std, the same recursion as the error propagation of the learned power method, arXiv:2303.08902
  (abstract only)]. This is the biased fixed point.
- Toy (N = 8, rho = 0.02-0.06 measured):

| relative copy noise | stall <E_var - E_0> | mean cost of one copy error | eps/(1 - rho) |
|---|---|---|---|
| 1e-2 | 2.02e-3 | 1.82e-3 | 1.94e-3 |
| 1e-3 | 1.91e-5 | 1.78e-5 | 1.89e-5 |
| 1e-4 | 1.87e-7 | 1.74e-7 | 1.86e-7 |

  The bound uses the mean cost, so the per-round maxima exceed it (6.9e-3 at 1e-2).
- Globally: E_var[psi_k] <= E_var[psi_0] + sum_j eps_j, so the Q3 certificate becomes E_var[psi_0] + sum eps < E_1.
- Fixed points are no longer eigenstates, and copy noise also kicks the state off excited saddles.
- Caffarel-Pinar-Scemama arXiv:2609.01301 (abstract only) report a rising FN energy and a biased attractive node for SHDMC in
  any finite basis. That is the eps_k > Delta_k regime.

**Fixed alpha = -tau.** Write B = 1 - tau(H - E_k). If B >= 0, i.e. tau <= 1/(E_max - E_k), then E_var[B psi] <= E_var[psi]
[own]. Proof: with moments m_j = <psi|B^j|psi>, Cauchy-Schwarz gives m1^2 <= m0 m2 and m2^2 <= m1 m3, so m1 m2 <= m0 m3,
i.e. R_B(B psi) >= R_B(psi). Equality holds only for eigenstates.
- tau <= 1/(E_max - E_0) is therefore safe for all rounds, and everything above survives: chain, fixed points = eigenstates,
  no cycles, instability.
- Step 1 becomes linear, so continuity at eigenstates is restored, and the linearised loop is the true linear map
  R W (1 - tau(H - E_n)).
- Cost: on a lattice E_max - E_0 is extensive, so the per-round Lanczos gain is ~1/N of the optimal one. The FN step is what
  contracts (rho << 1 here).
- Beyond the bound (N = 8): tau = 3/(E_max - E_0) was still monotone in 50/50 runs (the bound is sufficient, not necessary).
  At tau = 10/(E_max - E_0), 50/50 runs had E_var increases, and the orbit settles on an **exact 2-cycle** between non-eigenstates:
  - |psi_{k+2} - psi_k| = 3.8e-14 against |psi_{k+1} - psi_k| = 0.31;
  - E_var - E_0 = 0.176 (E_1 - E_0 = 0.912), ground-state weight 0.92, residual 0.64.

## 5. Numerical checks (core code)

```python
# loop_lib.py (abridged): random frustrated H, optimal Lanczos step, lattice FN, loop
def rand_H(N, p=0.7, rng=None):          # connected random graph, H_xy ~ U(0.2,1) > 0, diagonal ~ N(0,1)
def lanczos(H, psi, alpha=None):         # psi + alpha (H-E) psi; alpha from the 2x2 Ritz problem in span{psi, (H-E)psi}
def fn_ham(H, g, ztol=0.0):              # violating if H_xy g_x g_y > 0: drop hop, V(x) += H_xy g_y/g_x; |g|<=ztol -> wall
def fn(H, g, ztol=0.0):                  # lowest eigenpair of fn_ham, sign aligned with g; returns phi, E_FN, FN gap
def loop(H, psi0, nround, alpha=None, copy_err=0.0, rng=None)
```

```python
# check2.py core: FN Jacobian at psi_n vs R W, and the count of mu > 1
HF0 = fn_ham(H, psin); W = HF0 - H
G = pinv(Q @ (HF0 - E[n]) @ Q); GW = qb.T @ G @ W @ qb             # qb: orthonormal basis of Q
J[:, i] = qb.T @ (F(psin + h qb_i) - F(psin - h qb_i)) / (2h)        # F = normalised, sign-aligned FN ground state
assert #(eig(GW) > 1) == n
```

```python
# check3.py core: guide in F_1 and its Lanczos preimage
g = p1 * where(cluster == 0, 1, 3.0)                                 # FN(g) == psi_1 exactly
res(z) = [P_k @ normalise(lanczos(H, z))[cluster == k] for k] + [|z|^2 - 1]   # P_k projects off psi_1 inside cluster k
least_squares(res, random z0) -> psi with <psi_0|psi> = -0.214
```

### 5.1 Summary of runs

| check | setting | result |
|---|---|---|
| check1 | 300 H (N = 3-10) x 10 starts, 300 rounds, optimal alpha | 3000/3000 end at the ground state (residual < 1e-6) |
| check1b | same, chain links | 39 violations, all with min|psi'|/max <= 4e-16 (round-off of V ~ 1e16) |
| check1c | same, wall as a 1e12 diagonal | 30 violations, all 2.5-5.5e-5 = eigh round-off of the 1e12 entry |
| check1d | same, zero configurations (|psi'| <= 1e-12 max) deleted exactly | 0 violations in 97 rounds with guide zeros, 0 overall |
| check2/2b | 60 H, N = 4-8, eigenstates with min|psi_n| > 1e-3 | 214 clean fixed points; #(mu>1) = n in 214/214; FD vs R W median 7e-10 |
| check2c | 300 H, N = 8 | FN-degenerate fraction by n: 0, 0, .04, .12, .26, .50, 1.00 |
| check3 | N = 6 | start with GS overlap -0.214 -> psi_1 exactly; 1e-10 noise escapes in 10 rounds |
| check4 | zero of psi_n; same-sign zero | differentiable through the flip (mu = 1 neutral); O(1) jump 0 vs -0.584 |
| check5 | copy noise; fixed alpha | stall = eps/(1 - rho) within 4%; tau <= 3/(E_max - E_0) monotone, tau = 10/(...) not |
| check5b | tau = 10/(E_max - E_0) | exact 2-cycle, E_var - E_0 = 0.176 |
| check6 | alpha = 0 | 59/59 wrong-sign starts stall at non-eigenstates with zeros |

## 6. Not settled

1. **The Q2 lemma.** For alpha != 0 (optimal or 0 < -alpha <= 1/(E_max - E_0)), I could not show that the orbit's limit points have
   psi' outside D (same-sign guide zeros, disconnected kept graph). No toy run came near it. The cheapest way to kill the lemma
   would be a search for an orbit whose min|psi'_k| -> 0 with residual bounded away from 0; I did not run it.
2. **Measure of the excited-state basins.** Measure zero is expected but not proven: R W has a kernel (F3), so the map is not a
   diffeomorphism and the standard stable-manifold argument does not apply as stated.
3. **Degenerate levels.** If E* is degenerate, the orbit converges to the eigenspace, not necessarily to one vector. If the FN
   ground state is degenerate at the guide, step 2 needs a selection rule, which I did not analyse.
4. **The global rate.** It depends on rho = local FN contraction mu_max(R W at psi_0) times the Lanczos factor. In the toys
   rho ~ 0.03-0.06 per round in energy. Nothing here says how rho scales with lattice size or with the FN gap at 8x8.
5. **The project's present target is not this loop.** The c-relaxed Jacobi storage target is not phi (STORAGE_PROBLEM.md
   l.75-76), so everything above applies to design B with a perfect or eps-bounded copy only.

Literature read: abstracts only of arXiv:2609.01301, 2303.08902 and 1602.04915. ten Haaf cond-mat/9412037 and Zangwill/Luenberger
are cited from standard knowledge, not re-read. I did not read 0808.3417 in this task.
