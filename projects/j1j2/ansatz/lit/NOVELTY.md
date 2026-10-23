# Adversarial novelty check: candidate ansätze C1-C4

Worker report for the "ansatz pioneer" lead, 2026-10-23. Literature only; no jobs, no commits.
Reading levels: [full text] = whole paper read; [section read] = the relevant section(s) read
from the arXiv HTML (via the arxiv MCP cache, `~/Documents/.arxiv-mcp-cache/`); [abstract only] =
abstract or search snippet. Wiki: `second_brain/wiki/sources/szabo2020neural.md` (read).
`paper/RELATED_WORK_NOTES.md` (read; covers the Lanczos/FN lineage used for C2).

Tool note: `search_semantic` and `search_google_scholar` returned empty results for every query
this session (service problem, not absence of papers); the arXiv API plus WebSearch (extended)
carried the search. Absence below means "not found by these searches", not proof of absence.

| | Verdict | Most dangerous prior paper |
|---|---|---|
| C1 sign-rule interference | PARTIALLY ANTICIPATED | Schurov, Kravchenko, Katsnelson, Bagrov, Westerhout, arXiv:2508.09870 (2025) |
| C2 positive state through a learned local signed kernel | PARTIALLY ANTICIPATED (global-alpha version ANTICIPATED) | Sorella, PRB 64, 024512 (2001), cond-mat/0009149 |
| C3 gauge-fluctuating partons | PARTIALLY ANTICIPATED | Emonts et al., PRD (2023), arXiv:2211.00023 (structure); Ivanov & Senthil, PRB 66, 115111 (2002) (physics) |
| C4 plaquette-frame positive state | PARTIALLY ANTICIPATED (as an idea), NOVEL as a VMC/FN implementation for J1-J2 | Herz, Schäfer, Gonzalez, Luitz, arXiv:2607.24679 (2026) |

---

## C1. Sign-rule interference: psi = s_Neel A + s_stripeX B + s_stripeY C, A,B,C >= 0

### Closest works
1. **Schurov et al., arXiv:2508.09870 (2025), "Learning complexity of many-body quantum sign
   structures through the lens of Boolean Fourier analysis"** [section read: Intro, Sec. 2 results
   list, Sec. 3.3, Discussion]. Expands exact ED sign structures (24 spins) in parity functions
   chi_I. Marshall is a single parity; for the square J1-J2 model the sign expansion is "dominated by
   three types of regular sublattices: Marshall-Peierls (checkerboard) for small J2, stripes for
   large J2 ..., and 'stretched checkerboard' at intermediate values". Sec. 3.3 feeds selected
   parities as extra NQS inputs (supervised only). HOW C1 DIFFERS: they diagnose; they never write a
   variational state as a sum of parity-signed components with configuration-dependent positive
   weights, and their polynomial ansätze "cannot yet be directly used in the context of variational
   optimization". DANGER: the physical premise of C1 (Néel + stripe parities dominate near 0.5) is
   published, and it flags a third "stretched checkerboard" character that C1 omits.
2. **Szabó & Castelnovo, PRR 2, 033075 (2020), arXiv:2002.04613** [section read + wiki page].
   Phase = arg of a sum of unit-modulus phasors exp(i b_n + i w·sigma), i.e. a sum of sign
   characters that interfere. At J2 = 0.5 the kernels "mix the chequerboard with stripes of the
   J1 = 0 Marshall rule" (App. E) but converge to the Marshall rule (the "Marshall trap").
   HOW C1 DIFFERS: Szabó's components have equal (unit) modulus and are not tied to the amplitude;
   C1 lets the amplitude of each sign sector vary with x, so the sign is decided by which positive
   amplitude dominates locally. Szabó's 2023 Benasque talk states "sum of terms structure seems
   critical to get nontrivial signs" [abstract only, snippet].
3. **Choo, Neupert, Carleo, PRB 100, 125124 (2019), arXiv:1903.06713** [section read: "Sign
   structure of the ground state"]. Writes both limiting rules (-1)^{M_A} with A = sublattice or
   alternate rows/columns, imposes ONE of them as a prior and notes the bias. No sum of priors.
4. **Chen, Hendry, Weinberg, Feiguin, NeurIPS 2022, arXiv:2206.14307** [section read]. Linear
   combinations of RBMs via symmetry projection, then Lanczos. Components share one sign prior; no
   Néel/stripe mixture. Relevant because C4v projection of a single stripe-X component already
   produces the stripe-Y term of C1 (so B and C need not be independent networks).

### Search strings (arXiv API, WebSearch, paper-search)
1. arXiv: `"sign rule" Neel stripe superposition J1-J2 wave function`
2. arXiv: `abs:"Marshall sign" AND abs:superposition AND abs:frustrated` (0 hits)
3. arXiv: `abs:"mixture of experts" AND abs:"neural quantum states"` (0 hits)
4. arXiv: `abs:"partial Marshall" OR abs:"modified Marshall sign" OR abs:"generalized Marshall sign"`
5. paper-search (arxiv, semantic, openalex, crossref, scholar): `neural quantum states sign structure Marshall stripe superposition J1-J2`
6. Web: `neural quantum state superposition Marshall sign and stripe sign rule J1-J2 frustrated square lattice ansatz`
7. Web: `variational wave function linear combination of states with different sign structures frustrated antiferromagnet nodes cancellation`
8. Web: `"mixture of experts" neural quantum states sign structure frustrated magnet`
9. Web: `Boolean Fourier analysis sign structure ground state J1-J2 Walsh parity Marshall rule arXiv 2025`
10. Web: `wave function sum of two Marshall-signed components Neel and collinear sign rules J1-J2 nodes "sign rule" combination variational`
11. Web: `neural network quantum state J1-J2 "sign rule" for each sublattice choice combined Néel checkerboard and columnar stripe sign prior sum of two networks frustrated regime`
12. Web: `wave function ansatz linear superposition of Néel-like and stripe-like (collinear) states J1-J2 square lattice exact diagonalization overlap ground state intermediate phase`

### What is still open
A variational (VMC/SR or FN-guide) ansatz whose sign comes only from interference of a few fixed
analytic parity rules weighted by learned positive amplitudes; not found anywhere. Cite Schurov
2025 for the premise and Szabó 2020 for the phasor-sum precedent. Test: include the "stretched
checkerboard" character of Schurov et al. as a fourth component, or show it is not needed.

---

## C2. Positive state through a learned local signed kernel: psi(x) = sum_y K_theta(x,y) Phi(y), Phi >= 0

### Closest works
1. **Sorella, PRB 64, 024512 (2001), cond-mat/0009149** [section read: Secs. I-V]. The SR state is
   psi'(x) = r_x psi^f(x) with r_x = sum_k alpha_k O^k_x, O^k_x = <psi_G|O^k|x>/<psi_G|x>, for an
   ARBITRARY set of operators O^k; with O^k = H^k it is the Lanczos state; he states that because
   r_x "is not necessarily positive or negative, the nodes can be changed and improved". Choosing
   O^1 = H_J1, O^2 = H_J2 gives exactly (1 + alpha1 H_J1 + alpha2 H_J2) psi_G; choosing bond
   operators h_b gives a bond-dependent constant alpha_b. HOW C2 DIFFERS: Sorella's alpha_k are
   numbers fixed by the SR/Euler conditions; the reference psi_G is a signed Jastrow-Slater state; no
   configuration-dependent alpha_b(x), no positive Phi by construction, no NN.
2. **Becca, Hu, Iqbal, Parola, Poilblanc, Sorella, J. Phys. Conf. Ser. 640, 012039 (2015),
   arXiv:1412.2656** [section read: Sec. 3]. p-step Lanczos (1 + sum_m alpha_m H^m) on projected
   fermion states for J1-J2; global alpha_m only.
3. **Chen et al., arXiv:2206.14307 (NeurIPS 2022)** [section read] and **Wang, He, Lu, PRB (2025),
   arXiv:2502.01264** [section read: Secs. I-II.3]. Lanczos on NQS for J1-J2; Wang et al. fit the
   Lanczos vector H psi - a psi with a sign + amplitude CNN (supervised). Global coefficients; the
   starting state is already signed.
4. **Hamiltonian variational ansatz** (quantum-computing literature; exp(-i theta_k H_k) per operator
   group) [abstract only, snippet]. Separate coefficients per Hamiltonian part, but unitary,
   global, and not on a positive base state.

### Search strings
1. Web: `neural quantum state Lanczos step bond-dependent coefficients configuration-dependent "(1 + alpha H)" learned kernel frustrated J1-J2`
2. Web: `variational wave function apply Hamiltonian to positive Marshall-signed state generates sign structure frustrated J2 term "one Lanczos step" nodes`
3. Web: `Chen Hendry Weinberg Feiguin "Systematic improvement of neural network quantum states using Lanczos" NeurIPS 2022`
4. Web: `neural network wave function with Hamiltonian applied layer "local energy" kernel psi(x)=sum_y K(x,y) phi(y) nodeless bosonic state sign structure frustrated`
5. Web: `operator Jastrow OR "operator-valued" correlator wave function spin "bond-dependent" Lanczos step separate coefficients J1 J2 variational Monte Carlo`
6. Web: `power method neural quantum state imaginary time step "(1-tau H)" applied to neural network ansatz sign structure nodes frustrated magnet`
7. Web: `"neural Lanczos" OR "Hamiltonian-informed" neural quantum states local operator applied sign`
8. Web: `variational ansatz "configuration-dependent" Lanczos coefficient OR "local Lanczos" OR "site-dependent Lanczos" wave function lattice spin Hubbard`
9. arXiv: `abs:"Lanczos" AND abs:"neural" AND abs:"sign structure"` (0 hits)
10. Prior project search in `paper/RELATED_WORK_NOTES.md` (Sorella 2001/2002, Ledinauskas 2023, Gravina 2025: NQS fitted to (1 - tau H) psi).

### What is still open
(1 + alpha1 H_J1 + alpha2 H_J2) Phi with constant alphas is ANTICIPATED (Sorella 2001
formalism; do not claim it). Open: configuration- or bond-dependent learned alpha_b(x) acting on a
positive Phi, and the variant where only the frustrating J2 bonds carry the signed kernel. Not
found. Caveat for the lead: if alpha_b(x) is allowed to depend on x without restriction, C2 contains
every state (alpha absorbs psi/Phi), so the novelty claim rests on the locality/smoothness
restriction, which should be stated precisely.

---

## C3. Gauge-fluctuating partons: psi(x) = sum_phi w_theta(phi) P_G Pf/det(x; u_phi)

### Closest works
1. **Emonts, Kelman, Borla, Moroz, Gazit, Zohar, arXiv:2211.00023 (PRD 2023)** [section read: Sec.
   IV.1-IV.3]. Gauged Gaussian fermionic PEPS: the state is a sum over Z2 gauge-field configurations
   Q of a Gaussian fermionic state in that background, sampled by Metropolis over Q with weight
   |psi(Q)|^2 = sqrt(det(...)); VMC over the parameters. HOW C3 DIFFERS: physical gauge fields of a
   lattice gauge theory (Q is part of the Hilbert space, not summed out), no Gutzwiller projection,
   weights fixed by the PEPS, not a learned w_theta; not a spin model. Structurally the closest
   "sum over Z2 backgrounds of Gaussian fermion states with MC over the gauge field". Follow-up
   with fermionic matter: Kelman, Borla, Emonts, Zohar 2024 [abstract only, snippet].
2. **Ivanov & Senthil, PRB 66, 115111 (2002), cond-mat/0204043** [section read: Secs. I-III].
   Projected BCS with or without an hc/2e vortex (equivalently changed boundary conditions) gives
   the vison sectors; symmetric/antisymmetric combinations correspond to the dimer-model sectors.
   Used as a diagnostic of topological order, not as a learned superposition minimising energy;
   no sum over local vison/flux configurations.
3. **Zhang, Grover, Turner, Oshikawa, Vishwanath, PRB 85, 235151 (2012), arXiv:1111.2342**
   [abstract only]. Linear combinations of projected states with different boundary conditions
   (minimally entangled states), coefficients fixed by entanglement entropy, not energy.
4. **Budaraju, Iqbal, Becca, Poilblanc, arXiv:2307.01149 (2023)** [abstract only]. Projected
   kagome Dirac states with monopoles / flux patterns inserted, compared one at a time by energy;
   no superposition.
5. **Luo, Carleo, Clark, Stokes, PRL 127, 276402 (2021), arXiv:2012.05232** [abstract only].
   Gauge-equivariant NQS for lattice gauge theories (Z2 loop gas); gauge fields physical, no
   partons.

### Search strings
1. Web: `superposition of Gutzwiller-projected wave functions over Z2 flux configurations visons variational Monte Carlo spin liquid weighted sum gauge field`
2. Web: `Ivanov Senthil projected wave functions fractionalized phases visons boundary conditions topological sectors`
3. Web: `neural network weights sum of Pfaffians OR determinants parton gauge configurations variational wave function frustrated Heisenberg "gauge fluctuations" beyond mean field projected`
4. Web: `linear combination of Gutzwiller projected Pfaffian states different mean-field ansatz flux variational Monte Carlo "multi-Pfaffian" OR "sum of Pfaffians" spin model`
5. Web: `parton wave function Monte Carlo sampling of Z2 gauge field configurations fermion determinant variational state spin liquid "gauge field" sampled together with spins`
6. Web: `Luo Carleo Clark Stokes gauge equivariant neural networks Z2 lattice gauge theory fermions 2012.05232`
7. Web: `Emonts Kelman Borla Moroz Gazit Zohar fermionic gauged Gaussian PEPS Z2 Monte Carlo sampling gauge field configurations`
8. Web: `spin liquid variational wave function superposition of projected BCS states different boundary conditions or flux insertion weighted sum minimally entangled states Gutzwiller`
9. Web: `"resonating" Hartree-Fock OR "sum of Slater determinants" OR "multi-configuration" Gutzwiller projected variational Monte Carlo frustrated Heisenberg J1-J2 linear combination symmetry-broken mean-field states`
10. arXiv: `abs:"Gutzwiller" AND abs:"gauge field" AND (abs:"Monte Carlo sampling" OR abs:"fluctuating") AND abs:"spin liquid"`
11. arXiv: `abs:"neural" AND abs:"parton" AND (abs:"gauge" OR abs:"Gutzwiller") AND abs:"wave function"` (0 hits)
12. arXiv: `abs:"projected" AND abs:"superposition" AND (abs:"flux" OR abs:"vison") AND abs:"variational"` (no relevant hits)
13. arXiv: `abs:"linear combination" AND abs:"Pfaffian" AND abs:"variational Monte Carlo"` (0 hits)

### What is still open
For a spin model: a learned-weight superposition over local Z2 flux (vison) configurations of
Gutzwiller-projected BCS states, optimised by energy, with the flux configurations summed out
(sampled jointly with x). Not found. The topological-sector-only version (four BC sectors) is
essentially Ivanov-Senthil / Zhang et al. and should not be sold as new. Not checked: the full
text of mVMC/multi-Pfaffian work from the Imada group (Tahara-Imada 2008 uses one Pfaffian plus
projections, abstract-level only) and resonating Hartree-Fock (Fukutome/Tomita) for spin models.

---

## C4. Plaquette-frame positive state (2x2 crossed-plaquette eigenbasis; VMC/FN in that frame)

### Closest works
1. **Herz, Schäfer, Gonzalez, Luitz, arXiv:2607.24679 (2026), "Sign-optimized Quantum Monte
   Carlo"** [section read: Abstract, Sec. 4.1, model list, references]. Optimises 2- and 3-site
   cluster unitaries (and uses dimer/trimer/rung eigenbases) to minimise non-stoquasticity, then
   SSE. Models: fully frustrated and triangular two-leg ladders, maple-leaf lattice. A 6-site
   hexagon basis was considered but too slow. No square J1-J2, no 2x2 crossed plaquette, no VMC/NQS
   in the rotated frame.
2. **Torlai, Carrasquilla, Fishman, Melko, Fisher, PRR 2, 032060(R) (2020), arXiv:1906.04654**
   [section read: Method and Results]. Positivises given MPS ground states with 1- and 2-qubit gate
   circuits; triangular two-leg ladder (1D J1-J2 limit, ring exchange). Recovers Marshall; deeper
   circuits needed in the spin-Bose-metal phase. Not 2D, no variational use in the frame.
3. **Hangleiter, Roth, Nagaj, Eisert, Sci. Adv. 6, eabb8341 (2020), arXiv:1906.02309** [abstract
   only]; **Levy & Clark, PRL 126, 216401 (2021), arXiv:1907.02076** [abstract only, Hubbard];
   **Klassen & Terhal; Marvian, Lidar, Hen** (complexity results, not re-read). Local basis
   optimisation for QMC; frustrated ladders and Hubbard, not square J1-J2 plaquettes.
4. **Cluster-basis QMC**: Alet, Damle, Pujari PRL (2016) arXiv:1511.01586; Honecker/Wessel et al.
   frustrated ladder arXiv:1704.01973; Wessel et al. Shastry-Sutherland in a cluster basis
   arXiv:2112.03296 [abstract only]. Sign-free or sign-reduced only when cluster spin is conserved;
   square J1-J2 plaquette tiling is not of that type.
5. **Older J1-J2 plaquette attempts**: Wojtkiewicz, cond-mat/0611550 (dimer pseudo-basis, Hamiltonian
   as sum over plaquettes; claimed sign-free; refuted by a 2008 PRB Comment, PRB 77, 146401);
   K. Beach, APS March Meeting 2008 abstract "Ameliorating the sign problem for frustrated magnets
   using plaquette grouping" (valence-bond basis, plaquette-grouped updates, J1-J2 square lattice)
   [abstract only, snippet; no paper found]; Nyfeler et al., PRL 100, 247206 (2008),
   arXiv:0803.3538 (nested cluster algorithm, frustrated square lattice, moderate T) [section read:
   Intro/method]. Also 1D-only single-qubit brute-force positivisation: Bannykh, Sotnikov,
   Mazurenko, PRB 113, 184405 (2026), arXiv:2511.17957 [abstract only].

### Search strings
1. Web: `J1-J2 square lattice plaquette basis sign problem quantum Monte Carlo local basis rotation frustrated`
2. Web: `neural quantum state cluster basis plaquette basis variational Monte Carlo frustrated magnet J1-J2 local unitary transformation sign structure`
3. Web: `Torlai Carrasquilla Fishman Melko Fisher wave-function positivization automatic differentiation J1-J2`
4. Web: `neural quantum states learnable local basis rotation improves frustrated Heisenberg sign structure positive amplitudes`
5. Web: `Levy Clark "Mitigating the sign problem through basis rotations" models J1-J2 two-dimensional`
6. Web: `fixed-node or variational Monte Carlo in dimer basis or plaquette basis J1-J2 square lattice sign problem reduced "plaquette basis"`
7. Web: `Kevin Beach "plaquette" sign problem frustrated magnets J1-J2 quantum Monte Carlo 2008`
8. Web: `"Ameliorating the sign problem for frustrated magnets using plaquette grouping"`
9. Web: `neural quantum states dimer basis OR singlet-triplet basis OR cluster basis Shastry-Sutherland OR frustrated ladder variational Monte Carlo basis choice sign structure`
10. Web: `basis dependence neural network quantum states local unitary rotation optimize stoquasticity then positive ansatz variational ground state frustrated spins`
11. Web: `variational Monte Carlo neural network wave function written in tetrahedron OR plaquette OR "supersite" local basis pyrochlore OR checkerboard OR J1-J2 positive amplitudes sign-free representation`
12. arXiv: `abs:"plaquette basis" AND (abs:"sign problem" OR abs:"Monte Carlo")` (0 hits)
13. arXiv: `abs:"sign problem" AND abs:"basis" AND abs:"J_1-J_2" AND (abs:"square lattice" OR abs:"square-lattice")`

### What is still open
The idea "rotate to a cluster eigenbasis to tame signs" is old (cluster-basis QMC, positivisation,
sign-optimised QMC). Not found: (i) the 2x2 crossed-plaquette eigenbasis for the square J1-J2 model
measured by average sign / non-stoquasticity; (ii) any VMC/NQS or lattice FN carried out in that
frame with a positive (or Marshall-like) amplitude network. Cheap kill test before investing: ED
average sign of the exact 4x4 / 6x6 ground state in the plaquette frame versus the Marshall frame
(Szabó Tab. III gives <s>_MSR = 0.9745 at 4x4, 0.9650 at 4x6, J2 = 0.5). Note the tiling covers
only half of the J1 bonds and a quarter of the J2 bonds intra-plaquette, so inter-plaquette terms
stay non-stoquastic; Herz et al. found the hexagon (6-site) basis already too slow for SSE, a cost
that VMC does not pay.

---

## Not checked
- Becca & Sorella textbook (2017) chapters on generalized Lanczos (possible explicit
  (1 + alpha1 H1 + alpha2 H2) examples).
- Full text of Zhang et al. 1111.2342, Hangleiter 2020, Levy-Clark 2021, Wessel 2112.03296,
  Beach 2008 (abstract only).
- Semantic Scholar / Google Scholar and citation-graph forward searches (services returned empty
  this session); forward citations of Szabó 2020, Schurov 2025 and Ivanov-Senthil 2002 should be
  re-run when the service works.
