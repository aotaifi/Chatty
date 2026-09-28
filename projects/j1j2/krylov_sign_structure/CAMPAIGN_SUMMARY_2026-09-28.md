# Campaign Summary — Projected-Krylov Sign Reconstruction
**Date:** 2026-09-28  
**Project:** square-lattice spin-1/2 J1-J2 sign problem at J2/J1=1/2, with triangular-lattice controls  
**Subproject:** projects/j1j2/krylov_sign_structure

## 1. Original question

A hidden-sign ViT/NQS benchmark suggested that a single label-free Krylov/local-energy coordinate almost reconstructs the frustrated J1-J2 ground-state signs.

Write

\[
\psi_s(x)=a(x)s(x),\qquad s(x)\in\{\pm1\},
\]

and define

\[
r_s(x)=\frac{(H\psi_s)(x)}{\psi_s(x)}.
\]

The observed one-step rule was

\[
s'(x)=s(x)\,\mathrm{sign}[t-r_s(x)].
\]

The campaign asked:

1. Why can one scalar coordinate reconstruct the signs so well?
2. Is this special to J1-J2, to the square lattice, or to starting near the right sign structure?
3. Does iteration converge to the exact ground-state signs?
4. What obstructs convergence?
5. Can the sign update be combined with fixed-node amplitude improvement into a self-improving loop?

---

## 2. Exact structural identity

The update is not an ad hoc classifier. It is exactly the sign of a one-step Krylov vector:

\[
[(t-H)\psi_s](x)=[t-r_s(x)]\psi_s(x),
\]

therefore

\[
\mathrm{sgn}[(t-H)\psi_s](x)
=
s(x)\,\mathrm{sign}[t-r_s(x)].
\]

Our nonlinear map is therefore

\[
a s_k
\;\xrightarrow{\,t_k-H\,}\;
\phi_k
\;\xrightarrow{\text{keep sign only}}\;
a\,\mathrm{sgn}(\phi_k).
\]

The amplitudes are restored after each step.

## 3. Label-free threshold and monotonicity

At every iteration, \(t_k\) is chosen by minimizing the fixed-amplitude variational energy over all realizable thresholds of \(r_k\).

Because choosing \(t>\max_x r_s(x)\) leaves the current sign pattern unchanged, the threshold family always contains \(s\). Hence

\[
E[T(s)]\le E[s].
\]

So the map is an **energy-non-increasing projected-Krylov sign descent**.

With the exact ground-state modulus \(a=|\psi_*|\),

\[
H(a s_*)=E_0(a s_*),
\]

hence

\[
r_*(x)=E_0
\]

for every nonzero component. The exact ground-state sign structure is therefore a fixed point and a global minimum of the fixed-amplitude sign problem.

---

## 4. Square-lattice scan: the one-step effect

Exact diagonalization on periodic \(4\times4\), \(S^z=0\).

Using Marshall signs:

| J2/J1 | O_S before | O_S after one step |
|---:|---:|---:|
| 0.0 | 1.000000 | 1.000000 |
| 0.2 | 1.000000 | 1.000000 |
| 0.4 | 0.999128 | 0.999983 |
| 0.5 | 0.974538 | 0.999319 |
| 0.6 | 0.795572 | 0.957131 |
| 0.8 | 0.123664 | 0.061971 |
| 1.0 | 0.064215 | 0.171346 |

At large J2, a stripe/J2-adapted gauge restores the spectacular one-step behavior:

- J2=0.8: \(0.899845\to0.975927\)
- J2=1.0: \(0.955819\to0.994884\)

Initial conclusion: a good gauge makes one step very powerful.

## 5. Iteration changes the picture

Keeping the exact amplitudes fixed and recomputing \(r_k\) after every sign update:

### J2/J1=0.5, Marshall
\[
0.974538
\to0.9993189
\to0.99999823
\to1.
\]

Exact signs after **3 updates**.

### J2/J1=0.6, Marshall
\[
0.795572
\to0.957131
\to0.991267
\to0.999444
\to1.
\]

Exact after **4 updates**.

### J2/J1=0.8, Marshall
Despite the bad first step:

\[
0.123664
\to0.061971
\to0.098225
\to0.219270
\to0.469895
\to0.507588
\to0.848923
\to0.962882
\to0.991197
\to0.999846
\to1.
\]

Exact after **10 updates**.

### J2/J1=1.0, Marshall
\[
0.064215
\to0.171346
\to0.290203
\to0.339260
\to0.910391
\to0.990778
\to0.998562
\to0.999975
\to0.99999941
\to1.
\]

Exact after **9 updates**.

**Revised conclusion:** a good gauge is not required for eventual convergence on the tested square cluster. It explains why the *first* step looks miraculous and shortens the path.

---

## 6. Basin-size experiment

Starting from the exact sign structure, randomly flip configurations until a target fraction \(q\) of the physical probability weight has the wrong sign.

Across J2=0.5, 0.6, and 1.0:

- all 32/32 sampled starts recovered for \(q\le0.42\);
- \(q=0.42\) corresponds to initial sign overlap only about \(0.16\);
- close to \(q=0.5\), success becomes model- and pattern-dependent.

So the ground-state basin is empirically very large in random physical-weight Hamming distance.

But distance alone is not enough: structured states can have tiny overlap and still sit in very different invariant basins.

---

## 7. Triangular-lattice control

Nearest-neighbor spin-1/2 triangular Heisenberg antiferromagnet on periodic \(6\times3\), \(S^z=0\).

Three simple structured gauges were iterated:

- maxcut_x
- parity_y
- parity_xy

All flow to wrong fixed points with essentially zero ground-state sign overlap.

Yet random corruption around the exact triangular signs also contracts strongly back:

- 8/8 recovery through \(q=0.45\);
- 4/8 recovery at \(q=0.49\).

Therefore triangular frustration does **not** destroy the ground-state attractor. The structured gauges begin in other basins.

---

## 8. Exact symmetry-sector invariant

Let \(P\) be a symmetry with

\[
[P,H]=0,
\qquad
Pa=a,
\]

and suppose

\[
P(a s)=\chi(a s).
\]

Then numerator and denominator of \(r_s=(Has)/(as)\) transform with the same character, so

\[
r_s(Px)=r_s(x).
\]

Therefore every threshold mask is \(P\)-invariant and the projected-Krylov map preserves \(\chi\).

Hence

\[
\boxed{\text{wrong exact symmetry character } \Rightarrow \text{ impossible to reach the target sector}.}
\]

### Triangular audit

The exact ground state has \(T_x=+1\).

- maxcut_x: \(T_x=-1\)
- parity_xy: \(T_x=-1\)

A full audit of all 72 affine automorphisms of the \(6\times3\) triangular bond graph also closes parity_y:

for

\[
P:(x,y)\mapsto(-x,2-y),
\]

- exact ground state: \(P=-1\)
- parity_y start: \(P=+1\)
- parity_y wrong fixed point: \(P=+1\)

Thus **all three tested triangular structured failures have exact symmetry-sector obstructions at their core**.

---

## 9. Symmetry-protected basins have finite thickness

Break the parity_y symmetry by randomly perturbing a physical-weight fraction \(q\) of its signs, then iterate again.

12 trials per q:

| perturbation q | recovery to exact GS |
|---:|---:|
| 0.0001 | 0/12 |
| 0.0005 | 0/12 |
| 0.001 | 0/12 |
| 0.005 | 0/12 |
| 0.01 | 0/12 |
| 0.02 | 2/12 |
| 0.05 | 5/12 |
| 0.10 | 8/12 |

Infinitesimal symmetry breaking is not enough. The exact symmetry sector is the invariant core of a **finite nonlinear attraction basin**.

---

## 10. Approximate amplitudes: where exactness is lost

The mechanism tests above used exact ground-state amplitudes. To remove this crutch in a controlled way, use the exact modulus from a *different* J2 value as the trial modulus for the target Hamiltonian.

### Target J2=0.5

Using the J2=0.4 modulus:

- amplitude fidelity \(F_a=0.9404\)
- terminal sign overlap \(O_S=0.99683\)

Using the exact J2=0.5 modulus:

- \(O_S=1\)

### Target J2=1.0

Using the J2=0.8 modulus:

- amplitude fidelity \(F_a=0.9690\)
- terminal sign overlap \(O_S=0.99981\)

Using the exact J2=1.0 modulus:

- \(O_S=1\)

Even rather poor amplitude moduli can still produce surprisingly good signs, but the fixed point becomes biased.

**Key separation:**

\[
\boxed{\text{exact amplitudes} \Rightarrow \text{exact sign attractor}}
\]

while

\[
\boxed{\text{good approximate amplitudes} \Rightarrow \text{very accurate but biased sign fixed point}.}
\]

This identifies amplitude quality as the main remaining algorithmic bottleneck.

---

## 11. The parent-loop idea

The natural self-improving loop is

\[
s_k
\;\xrightarrow{\mathrm{FN/sign\mbox{-}free\ solve}}\;
a_{k+1}
\;\xrightarrow{\mathrm{projected\ Krylov\ sign\ descent}}\;
s_{k+1}
\;\xrightarrow{\mathrm{FN}}\;
a_{k+2}
\to\cdots
\]

Conceptually:

- FN/sign-free solve improves amplitudes **inside the current sign chamber**;
- projected-Krylov descent supplies an explicit **sign-changing chamber-crossing step**;
- improved signs should reduce FN bias;
- improved amplitudes should improve the next sign reconstruction.

The unresolved decisive question is whether the **combined map is jointly contractive** or can settle into a self-consistent wrong amplitude/sign fixed point.

The next benchmark should therefore be a fully closed \(4\times4\) loop with no exact-ground-state information used in any update, measuring only afterward

\[
E_k-E_0,\qquad
O_S^{(k)},\qquad
F_a^{(k)}.
\]

---

## 12. What is already known in the literature

The broad idea of recursively improving a QMC trial state is **not new**.

### Iterative lattice fixed-node dynamics
Kairon & Clark, *Geometric View of Iterative Fixed-Node Dynamics* (arXiv:2609.16308, 2026) analyze repeatedly replacing a lattice-FN trial state with the FN ground state of its associated effective Hamiltonian. They show the exact ground state is a stable fixed point and characterize sign chambers and support-collapse boundaries.

Crucially, that iteration preserves the sign chamber. It addresses the **amplitude** half of our proposed loop.

Paper: https://arxiv.org/abs/2609.16308

### Self-healing diffusion Monte Carlo
Reboredo, Hood & Kent, PRB 79, 195117 (2009), and follow-ups develop recursive fixed-node DMC schemes that use projected information to improve the trial wavefunction/nodes. Later work demonstrates systematic reduction of nodal/sign errors.

This is closely related in spirit, but it is continuum nodal-surface optimization rather than the specific lattice sign-field map studied here.

Paper: https://doi.org/10.1103/PhysRevB.79.195117

### Self-consistent constrained-path AFQMC
There are also schemes that reconstruct improved trial states/natural orbitals from constrained-path AFQMC output and iterate them.

Example: https://arxiv.org/abs/2303.10301

### Lanczos/Krylov improvement
Few-Lanczos-step and stochastic-reconfiguration improvements of variational states are standard and have been combined with QMC/FN calculations.

---

## 13. Precise literature-gap question

A targeted literature search found many nearby components, but **no exact match yet** for the specific lattice iteration

\[
\boxed{
s_k
\xrightarrow{\rm lattice\ FN}
a_{k+1}
\xrightarrow{\rm explicit\ sign\ of\ }(t-H)(a_{k+1}s_k)
s_{k+1}
}
\]

with the Krylov step used specifically as an explicit **sign-changing chamber-crossing rule**, followed recursively by a new FN amplitude solve.

Therefore the novelty question is now sharply formulated:

> Has a lattice fixed-node amplitude solve ever been alternated recursively with an explicit sign-changing projected-Krylov threshold update, and has convergence of that combined map been studied?

The current targeted search says **not found**, not “proved absent.”

---

## 14. Current campaign verdict

### ACCEPT
- The one-step rule is exactly a Krylov-sign projection.
- Label-free threshold optimization gives a monotone fixed-amplitude energy descent.
- With exact amplitudes, the exact signs are a strong attractor on the tested square and triangular clusters.
- The one-step J2=0.5 miracle is the first exceptionally strong contraction step of that dynamics.
- Sign space contains multiple macroscopic basins.
- Exact symmetry characters provide rigorous invariant basin cores.
- Approximate amplitudes preserve high sign accuracy but bias the terminal fixed point.

### REJECT / CORRECT
- “One step generically solves frustrated signs.”
- “A good gauge is required for eventual square-lattice convergence.”
- “Triangular frustration destroys the Krylov attractor.”
- “Initial sign overlap alone predicts basin membership.”
- “The sign update itself is obviously the expensive part.”

### NEXT DECISIVE TEST
Close the full finite-size loop:

\[
\boxed{\mathrm{FN\ amplitudes}\leftrightarrow\mathrm{Krylov\ signs}}
\]

with no oracle information in the updates.

If that map converges to \(E_0\), \(F_a=1\), and \(O_S=1\) from a nontrivial starting sign pattern, the campaign advances from a sign-mechanism result to a genuine self-correcting QMC algorithmic principle.

---

## 15. Main durable files

- BASIN_DYNAMICS_VERDICT_2026-09-28.md
- SYMMETRY_BASIN_INVARIANT_2026-09-28.md
- TRIANGULAR_PARITY_BASIN_2026-09-28.md
- AMPLITUDE_ROBUSTNESS_2026-09-28.md
- KRYLOV_SIGN_STRUCTURE_NOTE_2026-09-28.pdf
- STATUS.md
- RESULTS_LOG.md

---

## 16. Final closure: the no-target-oracle loop reaches the exact signs

After the campaign summary above was written, the decisive fully recursive benchmark was completed.

The crucial correction relative to the older 4x4 loop is that the Krylov coordinate is recomputed from the **current** sign pattern,

\[
r_k=\frac{H(a_{k+1}s_k)}{a_{k+1}s_k},
\]

rather than from a fixed Marshall reference.

For target J2/J1=0.5, initialize with:
- Marshall signs;
- the exact ground-state modulus of the unfrustrated J2=0 Hamiltonian.

No target-J2 ground-state information enters an update.

The coupled loop

\[
s_k \xrightarrow{\rm FN} a_{k+1}
\xrightarrow{\rm current-sign\ projected\ Krylov} s_{k+1}
\]

reaches the exact target sign structure at **iteration 100**.

Final diagnostics:
\[
O_S=1,\qquad
F_a=0.9999946178,\qquad
E-E_0=2.6044\times10^{-5}.
\]

A best-case control initialized with the exact target modulus reaches the exact signs at iteration 27.

### Final campaign verdict

The mechanism subproject is **CLOSED / ACCEPTED at finite size**.

The demonstrated finite-size principle is:

> A standard lattice fixed-node amplitude solve alternated with a current-sign, label-free projected-Krylov threshold update can form a self-correcting loop and reach the exact ground-state sign chamber without target-ground-state information entering the recursive updates.

The remaining research problem is scaling: whether sampled/compact FN amplitudes preserve this loop with polynomial resources as L increases.

See \`FINAL_VERDICT_2026-09-28.md\`.
