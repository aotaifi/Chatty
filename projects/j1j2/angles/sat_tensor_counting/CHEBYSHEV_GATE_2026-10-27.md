# Constructive test: spectral filtering and SAT-style operator-word counting (2026-10-27)

**Verdict: TESTED / FAIL as a scalable new solution.** A degree-2 Chebyshev-filtered callable Jastrow guide improves small J1–J2 ground-state fidelity substantially, but (i) this is mathematically ordinary Krylov K2, not a novel SAT contraction; (ii) globally filtering independent copies fails size consistency; and (iii) a naive patch-factorized correction causes dramatic rare-tail amplitude distortion on a coupled 4×4 lattice. Do not resurrect the failed generic K2 or region products in the main FN–Krylov loop on this evidence.

## What was independently computed

Square spin-1/2 J1–J2 Heisenberg Hamiltonian with J1=1, J2=0.5, S^z=0. Two geometries are *not equivalent*: 4×4 is **periodic**, while 4×5 is **open**. ED using SciPy `eigsh`, as an external validation oracle only. Guide: `phi(x) ∝ s_M(x) exp[-κ H_xx]`, with κ either optimized by exact small-system VMC or fixed at 1.2. This is a cheap callable Jastrow guide; not the project's neural/FN guide.

Oracle-free **construction** (meaning no E0, E1 or Emax is needed to set the polynomial) chooses
```
cut = <phi|H|phi>                    (variational energy; estimable by VMC)
upper = (1/4) sum_b J_b               (rigorous spectral upper bound)
X = (upper + cut - 2H)/(upper - cut)
phi2 = T_2(X) phi = (2 X^2 - 1) phi.
```
Note that in this ED benchmark κ and cut were evaluated using complete wavefunction vectors; at scale they require a separate VMC optimizer/estimator. The exact spectral-gap *diagnostic* uses ED, but the above filter construction does not.

| geometry | initial squared overlap | degree-2 squared overlap | E(phi)-E0 | E(phi2)-E0 | degree-2 wrong-sign mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4×4 periodic; κ=1.2 | 0.61982 | 0.91785 | 1.02703 | 0.18092 | 0.00769 |
| 4×5 open; κ optimized | 0.70259 | 0.90062 | 0.57453 | 0.12729 | 0.00168 |
| 4×5 open; κ=1.2 | 0.64120 | 0.86396 | 0.62891 | 0.17113 | 0.00181 |

For the periodic 4×4 fixed-κ guide, the **optimal Ritz K2** state (same span of `phi, H phi, H² phi`) attains *better* energy error 0.09895 and fidelity 0.95721. Chebyshev K2 is a restricted, non-optimal member of that existing space. It cannot justify another global-K2 campaign.

An independently programmed **callable endpoint** formula was checked against the full sparse-matrix result on 60 random configurations, maximum absolute discrepancy at ~1e-17:
```
e(x) = (H phi)(x)/phi(x)
h2(x) = H_xx * e(x) + sum_(y != x) H_xy phi(y)/phi(x) * e(y)
phi2(x) / phi(x) = 2a²-1 -4ab e(x) +2b² h2(x),
a=(upper+cut)/(upper-cut), b=2/(upper-cut).
```
This computes the endpoint amplitude by up to two Hamiltonian hops, without materializing an N-spin vector. Its worst-case per-amplitude cost remains O(B²) naive for B local bonds, and it does not solve FN amplitude-learning.

## Size-consistency falsification (exact moments, no exponential replication)

Take k independent identical **periodic 4×4** blocks with `phi_total=phi^(⊗k)`, `H_total=sum_r H_r`, and use one **global** degree-2 Chebyshev filter with the variational cut and rigorous spectral bound both scaled by k. The norm of the filtered vector is computed exactly from the first four single-block moments of X (i.e. no `(C(16,8))^k` wavefunction).

| # blocks k | total spins | unfiltered squared overlap | global degree-2 squared overlap |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 0.61982 | 0.91785 |
| 2 | 32 | 0.38418 | 0.65421 |
| 4 | 64 | 0.14759 | 0.27547 |
| 8 | 128 | 0.02178 | 0.04288 |
| 16 | 256 | 0.000475 | 0.000961 |
| 32 | 512 | 2.25e-7 | 4.63e-7 |

As `k → ∞`, `X_global = (X_1+...+X_k)/k → E[X_block]=1` under the trial product measure, so `||T_2(X_global) phi_total||² →1`. Yet the squared ground-state overlap has factor `f0^k`; a fixed-degree global polynomial cannot avoid this decay. This is **not new**: existing J1–J2 design documents already warned that fixed global Lanczos depth is not size-consistent. The moment calculation gives a quantitative regression test specific to the proposed SAT/Chebyshev angle.

## First size-consistent candidate and failure

Product of separate degree-2 Chebyshev factors on four disjoint 2×2 patches of the **coupled** 4×4 periodic lattice, with each patch's variational energy cut and bond upper bound. On κ=1.2, mixing factors `(1-t)+t f_patch(x)` over t in `{0,.125,.25,.375,.5,.625,.75,1}` gave **no improvement**. t=.125 increased E-E0 from 1.027 to 12.375, and t=1 to 12.429. This particular naive product is unstable; do **not** generalize this to a no-go theorem for controlled/clipped, optimized region methods. Clipping and connected-region sign parity have already been proposed in `DESIGN_scalable_loop_2026-10-13.md`; avoid re-testing them solely to rescue the SAT angle.

## Why the new OpenAI result does not circumvent this failure

The OpenAI paper's Appendix D, Theorem D.1 claims a 2^(O(s/sqrt(log s))) exact contraction of signed bounded-degree Boolean tensor networks with s binary wires, conditional on correctness of the newly released work: https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf .

A degree-d endpoint Hamiltonian-word expansion uses `O(d log B)` nondeterministic operator-selector bits, but **this is not automatically an O(d log B)-wire bounded-degree Boolean tensor network or a 3-CNF with that many variables**. A verifier calculating intermediate spin states and a callable initial guide requires additional circuit gates/ancilla variables or large table tensors. All of them count. It is incorrect to plug selector bits alone into Theorem D.1 or Theorem 1.1 and claim a 2^o(N) quantum propagator solver. The naive exact enumeration still uses O((2B)^d) terms, with only polynomial cost when d is fixed; fixed d does not achieve scalable fidelity by the independent-block falsification.

Chebyshev projection in QMC predates 2026 (e.g. *Rapidly convergent quantum Monte Carlo using a Chebyshev projector*, Faraday Discussions 2024, DOI 10.1039/D4FD00035H). No claim of new Chebyshev filtering is warranted.

## Gate and decision

- **ESTABLISHED numerical checks:** oracle-free Chebyshev D2, endpoint-local formula vs matrix, Ritz K2 control, exact independence/moment replication test, naive patch-product counterexample.
- **REJECT:** claim that the selector-bit count alone yields a SAT speedup; the naive 2×2 patch product; degree-2 global filtering as a scalable solver.
- **PARK:** general tensor-counting implementation until a *fully specified* bounded-degree signed network beats the known O(N)-frontier exact contraction or provides a useful polynomial-space tradeoff with quantified constants.
- **DO NEXT (main project):** tackle the actually measured 8×8 FN walker→callable amplitude-ratio accuracy and its unseen-neighbor bias; use `projects/j1j2/README.md`, `FAILED_ROUTES.md`, and `DESIGN_scalable_loop_2026-10-13.md` rather than restarting K2. This is a separate practical bottleneck, not an immediate consequence of OpenAI's theorem.

Scripts and JSON outputs for this experiment were created and executed in the conversation's reproducibility bundle; see the companion download link in the ChatGPT reply. The figures above are exact small-ED / moment experiments and **not** large-lattice evidence.
