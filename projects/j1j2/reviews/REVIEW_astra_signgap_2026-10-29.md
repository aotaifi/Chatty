**[Derived] The \(10^{-4}\) claim is unsupported by these data. Its essential assumption is cancellation of the finite-size terms omitted by the fit—not merely a small leading coefficient.**

1. **[Derived] What controls the expansion.** Write \(e_a=E_a/N\), \(a=0,B\). For a fixed torus shape \(G\), conventional critical scaling gives
   \[
   e_a(L,G)-e_{a,\infty}=L^{-(2+z_a)}F_a(L/\xi_a,G,\text{boundary/sector data})+\cdots .
   \]
   Thus \(L^{-3}\) requires appropriate \(z=1\) low-energy physics; it does not follow from locality or stoquasticity.

   **[Derived, conditional]** In an ordered \(O(n)\) phase on a square torus, with velocity \(v\), stiffness \(\rho_s\), and \(\hbar=1\),
   \[
   e(L)=e_\infty-\frac{0.7188725(n-1)v}{L^3}
   +\frac{(n-1)(n-2)v^2}{8\rho_sL^4}+O(L^{-5}).
   \]
   Here \(n\) counts order-parameter components, not sites. The \(L^{-4}\) term matters especially when \(\rho_s\) is small. [Hasenfratz–Niedermayer, Eq. 2.18](https://arxiv.org/html/hep-lat/9212022v1#S2.SS4)

   **[Exact]** At \(J_2>0\), \(H_B\) lacks \(H\)'s rotated SU(2) symmetry; its continuous spin symmetry is U(1).
   **[Conjecture]** Its expected easy-plane ordered phase has one linear Goldstone mode, versus two for Néel-ordered \(H\).
   **[Derived, conditional]** Consequently, even in the proposed \(J_2=0.3\) control,
   \[
   c_0-c_B=-0.7188725(2v_0-v_B);
   \]
   no symmetry enforces \(v_B=2v_0\). Moreover, the \(O(3)\) \(L^{-4}\) term survives while the corresponding \(O(2)\) term vanishes.

   **[Conjecture]** At \(J_2=0.5\), assuming either Néel order or a particular spin liquid is additional physics input: published conclusions still differ. [Qian–Qin](https://arxiv.org/abs/2309.13630), [Fan et al.](https://arxiv.org/abs/2603.14425)
   **[Derived, conditional]** If \(H\) is fully gapped on compatible tori, its corrections can instead be exponentially small; subtracting gapless \(H_B\) then *introduces* an \(L^{-3}\) correction into \(\delta\). A small nonzero \(c_0-c_B\) reduces the amplitude, not the asymptotic power.

   **[Exact]** Torus vectors \(\mathbf a_i\) impose \(\mathbf k\!\cdot\!\mathbf a_i\in2\pi\mathbb Z\). The stated Marshall bond assignment requires \(a_{ix}+a_{iy}\) even.
   **[Derived]** Aspect ratio, shear, allowed ordering/nodal momenta, and sector crossings affect finite-size energies. For example, \((\pi/2,\pi/2)\) is allowed on \(L=4,8\) squares but not \(6,10\). Pooling tilted clusters by \(N\) alone discards this information. Use identical geometries and charge sectors for each pair, and verify the actual ground-state symmetry sector.

2. **[Derived] What the kill test actually tests.** For the same linear fitting/extrapolation operator \(P\), define \(r_a=P[e_a]-e_a(L_*)\). Then
   \[
   \widehat e_0^{\,\delta}-e_0(L_*)=r_0-r_B,\qquad
   \widehat e_0^{\,\mathrm{direct}}-e_0(L_*)=r_0.
   \]
   The fivefold requirement is \(|r_0-r_B|\le|r_0|/5\). If \(e_B\) lies exactly in the fitted model family, \(r_B=0\): both predictions have **identical errors**, however small \(c_0-c_B\) is.

   **[Derived]** A clean held-out failure rejects the specified small-cluster predictor. Incorrect shapes/sectors or unconverged \(E_B\) instead invalidate the test. \(J_2=0.3\) is a comparison regime, not a guaranteed positive control; only \(J_2=0\) guarantees \(\delta=0\).

   **[Derived: analytic counterexample]** An omitted \(0.5L^{-4}\) term, after fitting \(a+bL^{-3}\) at \(N=16,32\), produces prediction errors at \(N=(36,40,64,100)\):
   \[
   (-2.73,\,-5.21,\,-15.17,\,-21.78)\times10^{-5}.
   \]
   This simple example passes both nearby holdouts and fails both targets. Additional training points constrain it, but do not remove the general crossover problem. Several fit forms sharing an omitted effect are correlated checks. Freeze shapes, forms, weights and exclusions; \(N=36\) has already informed this proposal.

   **[Derived]** Test (d) is weak: the central-value DMRG window is \([-0.4994,-0.4990]\), entirely below the FN bound, making that inequality redundant. The DMRG number is itself a truncation-error extrapolation, not certified truth. [Gong et al., Table I](https://arxiv.org/html/1311.5962v3)

3. **[Exact] Useful identities and bounds.** In the common configuration space, set
   \[
   H_\lambda=H_B+\lambda V,\qquad
   V=2J_2\sum_{\langle\!\langle ij\rangle\!\rangle}(S_i^xS_j^x+S_i^yS_j^y).
   \]
   Hellmann–Feynman, interpreted almost everywhere through crossings, gives
   \[
   \delta_N=\frac1N\int_0^1\langle V\rangle_{\lambda,N}\,d\lambda .
   \]
   Ground energy is concave in \(\lambda\); variational comparison and the modulus inequality give
   \[
   \max\{0,\langle V\rangle_H/N\}\le\delta_N
   \le\langle V\rangle_B/N\le\|V\|/N\le2J_2.
   \]
   The last bound uses \(2N\) diagonal bonds. These imply neither monotonicity in \(N\) nor cancellation of finite-size corrections.
   **[Derived]** The integral identifies the needed cancellation as cancellation of finite-volume transverse bond correlations along the entire path. Its \(J_2\) off-diagonals become positive for \(\lambda>1/2\), so it does not supply a sign-free integration algorithm.

4. **[Derived] Present precision is insufficient.** Taking the supplied rounded \(\delta_6=0.385\) literally gives \(\delta_4=0.3832122351\). Two-point fits yield:

   | Assumed correction | \(\widehat\delta_8\) | \(\widehat\delta_{10}\) |
   |---|---:|---:|
   | \(L^{-2}\) | 0.385626 | 0.385915 |
   | \(L^{-3}\) | 0.385435 | 0.385590 |
   | \(L^{-4}\) | 0.385301 | 0.385383 |

   **[Derived]** These illustrate model sensitivity, not an uncertainty distribution. For the \(L^{-3}\) fit,
   \[
   \widehat\delta_8=-0.243421\delta_4+1.243421\delta_6,\quad
   \widehat\delta_{10}=-0.330105\delta_4+1.330105\delta_6.
   \]
   Nearest rounding of \(\delta_6\) permits uncertainties \(6.22\times10^{-4}\) and \(6.65\times10^{-4}\), respectively—before model error.
   **[Exact]** Per-site errors add directly: \(\widehat e_0-e_0=(\widehat e_B-e_B)+(\widehat\delta-\delta)\); there is no \(1/N\) suppression.
   **[Derived]** More precise small-cluster inputs, controlled QMC projection/statistics, and validated extrapolation residuals are all required.

5. **[Exact] Value relative to a variational energy.** A trial state provides \(e_B\le e_0\le e_{\rm var}\); the extrapolated prediction supplies neither a state nor a bound.
   **[Derived]** Exceeding a variational upper bound beyond the prediction’s declared error refutes it; falling below that bound establishes no accuracy. A verified recent 10×10 comparator is the reported CNN–MPS value \(-0.4976939(2)\), whose sampling error does not bound its variational bias. [Fan et al., Table 1](https://arxiv.org/html/2603.14425v1)
   **[Conjecture]** With independently validated errors, this could be a useful inexpensive energy estimator. Beating a variational number numerically would not constitute a better variational solution or a solution of the sign problem.