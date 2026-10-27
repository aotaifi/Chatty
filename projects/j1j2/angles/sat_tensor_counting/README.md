# 3-SAT / universal tensor contraction vs J1–J2 QMC sign problem
**Date:** 2026-10-27  
**Decision:** Do not implement a SAT/cluster-cancellation experiment on the square-lattice J1–J2 model. The *generic* translation is mathematically valid conditional on OpenAI's new theorem, but its proven bound loses asymptotically to geometric tensor contraction. A useful follow-up must first exhibit a fundamentally compressed representation or a target not covered by the geometric argument.

## Source and verification status
- OpenAI, *The Exponential Time Hypothesis Is False*, 17 Oct, revised 25 Oct 2026, especially Theorem 1.1, Sections 2.2–2.7 and Appendix D, Theorem D.1: https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf
- Lean formalization scope: https://github.com/openai/math/blob/main/lean/docs/378.md
- OpenAI announced expert shepherding on 2026-10-26; **external scientific verification is ongoing**, so results below relying on Theorem D.1 are explicitly *conditional*.
- Prior J1–J2 work: `projects/j1j2/FAILED_ROUTES.md`, `projects/j1j2/STATUS.md`, `projects/j1j2/krylov_sign_structure/`. Additional older exact block-operator work is documented in the Mac's `Desktop/research_run/PRE_ML_CYCLE_LAB.md` and `MIGRATION_PROMPT.md`.

## What the paper *actually* permits
Appendix D, Theorem D.1, claims that a bipartite Boolean tensor network of `s` binary wires with bounded (or <= s^0.9) vertex degrees, efficiently computable entries over a suitable truncated ring of bounded 2-adic precision, and polynomial description size can be **exactly** contracted in deterministic
```
poly(input size) * 2^(O(s/sqrt(log s)))   time
poly(input size)                       space.
```
This is stronger than a black-box decision-SAT analogy and works with **signed** entries. Tensor entries need not be positive. It avoids statistical sign variance by carrying out exact arithmetic, not by constructing sign-free stochastic weights. The theorem is not a polynomial-time cure of generic QMC.

For fixed-bit integer local weights, select sufficiently many 2-adic bits that the signed integer contraction is uniquely recoverable (for local bounded tensors, O(s) precision suffices). Rational weights with fixed dyadic denominator can be rescaled to integers. Arbitrary irrational gates `exp(-Delta tau h)` require a separate approximation/precision analysis: the exact theorem applies to the discretized finite-precision tensor network, not unconditionally to the continuum imaginary-time propagator.

## Square-lattice comparison: decisive bottleneck

Write `N=L^2` physical spins and `M` Trotter/time slices. Fixed-locality J1–J2 gates yield a bounded-degree **3D space-time** tensor network with `s=Theta(L^2 M)` wires (up to fixed constants). Hence the paper's theorem gives the *conditional, generic guarantee*
```
T_OAI = poly(s) * 2^(O(L^2 M / sqrt(log(L^2 M)))).
```
The same network can be exactly contracted by a spatial sweep (frontier O(LM)) or imaginary-time sweep (frontier O(L^2)), giving an unconditional standard upper bound
```
T_sweep = poly(s) * 2^(O(min(LM,L^2))).
```
Periodic spatial boundaries and next-nearest-neighbor couplings change only constant factors. This is a comparison of known **upper bounds**, not a lower bound against potential improvements.

| Regime | Paper's generic exponent | Geometric sweep exponent |
|---|---|---|
| `M=O(1)` | `O(L^2/sqrt(log L))` | `O(L)` |
| `M=Theta(L)` (e.g. critical beta~L with fixed step) | `O(L^3/sqrt(log L))` | `O(L^2)` |
| `M>=L` | `O(L^2 M/sqrt(log(L^2 M)))` | `O(L^2)` |

For any polynomially bounded M >=1, the ratio of the **nominal exponents** is `L/sqrt(log(L^2 M))` for M<=L, or `M/sqrt(log(L^2 M))` for M>=L. Both become large in the large-L regimes. Therefore applying Theorem D.1 as a *black box to the ordinary Trotter network* offers no certified runtime advantage for square-lattice J1–J2, and its constants are unknown.

### What a compressed formulation would have to beat
In the critical `M=Theta(L)` regime, a hypothetical compressed Boolean/tensor encoding with `s_eff` wires only gains a better exponent *by the generic theorem* if
```
s_eff / sqrt(log s_eff) = o(L^2).
```
A sufficient (not necessary) condition is `s_eff=o(L^2 sqrt(log L))`. The standard space-time construction has `s=Theta(L^3)`: roughly a factor `L/sqrt(log L)` too large at the exponent level. Merely changing SSE to CT-INT, grouping local words, or replacing tensors with CNF does not supply this compression.

At constant M the sweep exponent is O(L), and a theorem-based advantage would demand `s_eff/sqrt(log s_eff)=o(L)`, despite an explicit local description size Omega(L^2). This rules out a straightforward one-variable-per-gate encoding as a helpful route for this geometry.

## Does the SU(2) permutation formulation remove imaginary time?
No, without a new identity. For spin 1/2,
```
S_i · S_j = (1/2)P_ij - 1/4,
```
and the trace of a product of swap permutations is `2^(number of cycles)`. Thus `Tr H^k` is an exact signed weighted sum of operator words. This avoids summing explicit initial spin assignments, but a length-k ordered word chosen from Theta(N) local operators has Theta(k log N) raw choice bits. At low temperature a controlled polynomial approximation to exp(-beta H) typically needs `k=Omega(beta*N)` in a naive norm-based construction; this does **not** produce O(N) counting variables at beta growing with L. This is an observation about *explicit witness/path encodings*, not an unconditional lower bound on every possible global algebraic identity.

This permutation-trace direction was **already explored** in the physical block/CTQMC group-algebra laboratory; do not repeat it.

## Relationship to the sign problem and existing project
- **Established in our earlier CTQMC work:** local grouped operators are not all positive. The triangle has exact negative order-7 words; finite-time profiles can defeat single-edit positive capacity; arbitrary-precision equal-time n=100 tests do not establish global polynomial cancellation. No repeat of those tests is justified by the new theorem.
- **Conditional on paper:** a signed finite Boolean tensor contraction can be computed without Monte Carlo cancellation and in subexponential time in its number of wires.
- **Not established:** subexponential-in-*physical spin number* projection at beta~L, a polynomial algorithm, improved fixed-node amplitude learning, a better 6x6/8x8 FN–Krylov loop, or any gain at accessible lattice sizes.
- **Important caveat:** conventional exact sweep may consume exponential frontier memory. The paper claims polynomial *space*, potentially a distinct tradeoff, but its generic runtime guarantee is weaker for this geometry.

## Research gate and next action
**REJECT:** direct SAT conversion of the existing space-time worldline graph; another local cluster-cancellation trial; a small signed enumeration claiming success.

**PARK:** implementing the very recent unoptimized general tensor-contraction theorem before independent scrutiny and a favorable complexity estimate.

**TEST (theoretical only):** Can the *specific observable* needed by FN–Krylov be represented by a finite signed tensor network/circuit with `s_eff` satisfying `s_eff/sqrt(log s_eff)=o(L^2)` at beta~L, **including the cost of evaluating each tensor entry and its bit precision**? A positive construction must supply:
1. A fully specified mapping for an endpoint-conditioned propagator or a callable amplitude ratio, not only a scalar `Tr H^k`.
2. A proof of correct signs/weights and finite precision, with all internal contraction and oracle costs counted.
3. An asymptotic comparison against `2^O(L^2)` ED/time-sweep, plus a concrete constant-size exact-ED check.
4. A comparison with the current walker-accessible FN–Krylov one-step approach; no global ED oracle hidden in an input tensor.

**Current verdict:** No such representation has been found. The OpenAI result supplies a universal theoretical exact counting algorithm, not a demonstrated new J1–J2 solver. Prefer continuing the established FN–Krylov amplitude bottleneck until the above gate passes.
