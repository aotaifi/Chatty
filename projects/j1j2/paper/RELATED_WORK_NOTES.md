# Related-work notes: label-free Krylov sign update + lattice fixed-node amplitude alternation (J1-J2)

Compiled 2026-10-03 by web search only. Bibliographic data (authors, journal, volume, page/article number, year, DOI) were checked against the Crossref API and/or the arXiv API / publisher page unless marked UNVERIFIED. Statements about what a paper "does" come from the abstract, plus full-text reading where marked [full text]. Anything marked [abstract only] should be re-read before making a strong priority claim.

Method under comparison (M): psi = s*a; (1) s'(x) = s(x) sgn[T - r(x)], r = (H psi)/psi, T chosen to minimise the variational energy, no sign labels; (2) a_new = ground state of the lattice fixed-node Hamiltonian of the current guide; (3) alternate to convergence; at scale a NN learns phi_FN (from walkers, or by minimising the frozen FN energy with SR).

Flags: (a) = label-free local-energy threshold sign rule; (b) = alternation sign <-> FN amplitude; (c) = monotone condition <H>_new <= E_FN[old].

## Bottom line (read this first)

The closest prior work is Sorella (2001, with the 2002 lecture notes) plus Reboredo-Hood-Kent (2009). Together they pre-empt the IDEAS behind (a), (b) and (c). What I could not find anywhere is the specific package: a stand-alone, amplitude-independent, label-free sign update, closed into an alternation with exact lattice FN, with a guaranteed monotone chain, and a NN trained on phi_FN. Position the paper as an adaptation and closing of the loop, not as invention of the sign rule.

| Item | (a) sign rule | (b) alternation | (c) monotone |
|---|---|---|---|
| Sorella PRB 64, 024512 (2001) + lecture notes (2002) | PRE-EMPTED in substance (sgn r_x, r_x = 1 + alpha e_L, alpha from energy/SR condition) | PARTIAL (one sweep: "FN over the Lanczos-step wavefunction"; SR "self-consistent improvement of amplitudes and nodes") | PARTIAL (chain E_FN-expectation <= E_FN-ground <= <H>_guide, Eq. 10 in the notes); iterated monotonicity not stated in what I read |
| Reboredo-Hood-Kent PRB 79, 195117 (2009) | PARTIAL (new nodes = nodes of e^{-tau(H-E_T)} Psi_FN, i.e. to linear order sgn[(1+tau E_T) psi - tau H psi]; no energy-minimising T) | PRE-EMPTED for continuum DMC (iterate FN <-> node update) | PRE-EMPTED as a claim: the new guide has energy <= E(Psi_FN) (stated for the exact integral) |
| Caffarel-Pinar-Scemama arXiv:2609.01301 (2026) | no | critique | shows SHDMC iteration can RAISE the FN energy (1D toy), i.e. (c) is not automatic in SHDMC |
| Westerhout-Katsnelson-Bagrov Commun. Phys. 6, 275 (2023) | PARTIAL (signs by minimising energy at fixed amplitude = Ising problem; the rule below is a parallel local-field update of that Ising model) | PROPOSED, not implemented (Suppl. Note 5: sign optimisation inside a VMC loop) | no |
| Qin-Shi-Zhang PRB 94, 235119 (2016); Qin PRB 107, 235124 (2023) | no | PRE-EMPTED in spirit (QMC <-> independent-particle self-consistency of the constraint) for AFQMC, not for sign/lattice FN | no |
| Pilati et al. PRE 100, 043301 (2019) | no | PARTIAL for amplitudes only (PQMC walkers -> RBM -> new guide, iterated), stoquastic models | no |

## 1. Self-healing DMC and any lattice analogue

**Reboredo, Hood, Kent, PRB 79, 195117 (2009)** [full text, Sec. II; arXiv:0808.3417]. Optimises the DMC trial function using the walker (mixed-estimator) distribution. Core idea: the FN ground state has a kink at the trial nodes; smoothing it, Psi_T~ = integral Psi_FN(R+R') delta~(R') dR' (a short-time unconstrained propagator, ideally e^{-tau(H-E_T)} Psi_FN), moves the nodes toward the exact ones; iterate. They state the new guide has "an energy less than or equal to" that of Psi_FN. HOW IT DIFFERS: continuum first-quantised electrons, node moved via a convolution/determinant-expansion fit rather than a local-energy sign test; no variational choice of T; no guarantee beyond the heuristic argument; no NN. This is the main priority risk for (b) and (c) and conceptually for (a), since sgn[(T-H)psi] is the linear-order lattice version of sgn[e^{-tau(H-E_T)} psi].

**Bajdich, Tiago, Hood, Kent, Reboredo, PRL 104, 193001 (2010)** [abstract only]. Applies SHDMC to atoms/molecules (O, N2, C20). Same distinction as above.

**Reboredo, PRB 80, 125110 (2009)** (excited states), **J. Chem. Phys. 136, 204101 (2012)** (complex/magnetic, "steps beyond fixed phase"; arXiv:1008.0359), **Reboredo and Kim, J. Chem. Phys. 140, 074103 (2014)** (finite temperature) [abstract/title only]. Extensions of SHDMC; all continuum/electronic structure.

**Caffarel, Pinar, Scemama, arXiv:2609.01301 (Sep 2026)** [full text, abstract + Sec. I-II]. Tests SHDMC on a 1D periodic single-particle model with inversion symmetry. Finds that in the standard formulation "the fixed-node energy is found to increase upon iteration, and the node converges to a wrong value", with attractive and repulsive fixed points; a modified nodal update helps partially and "very likely requires the use of a localized basis set". Preprint, not peer reviewed as far as I can see. USEFUL for M: it motivates why an explicit monotonicity guard (c) is non-trivial, and a lattice (localised) basis is exactly the setting they say is favourable.

**Lattice version of SHDMC / iterative node improvement from walkers:** none found in searches (Hubbard/Heisenberg + self-healing, lattice fixed-node + node update). The closest lattice analogue is Sorella's SR/Lanczos construction in item 3. Absence of evidence only; web search is US-only and limited.

## 2. Self-consistent constrained-path AFQMC

**Qin, Shi, Zhang, PRB 94, 235119 (2016)** [abstract only; arXiv:1608.07154]. Replaces a fixed trial state by a density / one-body-density-matrix constraint iterated between an independent-particle calculation and QMC. **Qin, PRB 107, 235124 (2023)** [abstract only; arXiv:2303.10301]. Builds a new Slater-determinant trial state from natural orbitals of the CP-AFQMC mixed-estimator density matrix. HOW THEY DIFFER: fermionic Hubbard AFQMC, trial states restricted to mean-field determinants, no sign rule from H psi, no variational monotone guarantee. Cite as the same philosophy (self-consistent trial-state loop) in a different setting. Does not pre-empt (a), (b) in M's form, or (c).

## 3. Lanczos step and fixed-node / GFMC

**Sorella, PRB 64, 024512 (2001)** [full text, arXiv:cond-mat/0009149]. Generalised Lanczos via stochastic reconfiguration. Contains: SR with reference state psi^f = sgn(psi_G)|psi'|, i.e. signs frozen, amplitudes updated by projection; the one-step Lanczos wavefunction (1 + alpha H)psi_G, whose factor r_x = 1 + alpha e_L(x) "has not a definite sign", so "nodes can be changed and improved"; Table I includes "VMC+LS+FN, the fixed node over the VMC+LS wavefunction" (t-J model), and a comment that for r>0 the scheme "can be thought as a self-consistent improvement of the amplitudes and the nodes". HOW IT DIFFERS: alpha is fixed by the SR/Euler condition for the energy of the full Lanczos state (amplitudes included), applied inside a stochastic GFMC, t-J model, no stand-alone sign-only update, no exact FN diagonalisation loop, no NN, no explicit alternation to convergence.

**Sorella, "Effective hamiltonian approach for strongly correlated lattice models", arXiv:cond-mat/0201388 (2002)** [full text; lecture notes, Euro Winter School Kerkrade; venue otherwise UNVERIFIED]. Most explicit source for (a) and (c)-type statements: with r_x = 1 + alpha e_L, "the phases of the ground state wavefunction are much better represented by the signs of r_x psi_G(x) rather than by the ones corresponding to psi_G(x)"; for the p-step Lanczos state, choose the root z_k so as to "minimize the sign changes" of psi_G, "providing the best possible phases" obtainable with p-1 powers of H; the lattice FN Hamiltonian H^gamma_FN with the Perron-Frobenius argument (psi_FN has the same signs as psi_G); and the theorem E_FN(gamma) <= E(gamma) <= <psi_G|H|psi_G> (expectation of H on the FN state <= FN-Hamiltonian ground energy <= guide energy). Setting T = -1/alpha gives exactly M's rule s' = s sgn[T - r] (for alpha < 0). THIS PRE-EMPTS (a) as a rule. Caveats for the claim of difference: M applies the rule to signs only and picks T on a sign-only energy functional (to be confirmed against the paper's exact definition), whereas Sorella's alpha belongs to the full Lanczos state.

**Sorella, PRL 80, 4558 (1998); Sorella and Capriotti, PRB 61, 2599 (2000)** [abstract/summary]. GFMC with stochastic reconfiguration; FN as the sign-stable base; iterative reconfiguration "enhances the fixed-node approximation". Background for the lattice FN + corrections lineage.

**Hu, Becca, Parola, Sorella, PRB 88, 060402(R) (2013)** [abstract + text skimmed: no sign-rule statement]; **Becca, Hu, Iqbal, Parola, Poilblanc, Sorella, J. Phys. Conf. Ser. 640, 012039 (2015)** [abstract only]. Apply p = 1, 2 Lanczos steps on top of VMC (Gutzwiller-projected fermionic) states for J1-J2 and extrapolate with the energy variance. They do not feed the Lanczos signs into FN; the Lanczos state is evaluated variationally. Related for the J1-J2 benchmark and the p-step Krylov story.

**Becca and Sorella, Quantum Monte Carlo Approaches for Correlated Systems (CUP 2017)**. Textbook for VMC, lattice FN, GFMC-SR and Lanczos improvement. I did not read the relevant chapters; check them before claiming priority.

**van Bemmel, ten Haaf, van Saarloos, van Leeuwen, An, PRL 72, 2442 (1994); ten Haaf, van Bemmel, van Leeuwen, van Saarloos, Ceperley, PRB 51, 13039 (1995)** [abstract]. The lattice FN prescription and its upper-bound proof (effective Hamiltonian ground state is an upper bound of E_0). **ten Haaf and van Leeuwen, arXiv:cond-mat/9510042 (1995)** [abstract + preview]: "extension" via nodal relaxation, i.e. release of the FN constraint starting from the FN solution; it does not iterate the guide. Needed citations for step (2).

**Boninsegni, PRB 52, 15304 (1995)** [abstract]. FN GFMC for the triangular-lattice Heisenberg model (an example of a lattice FN application to a frustrated lattice spin system).

"Fixed-node with Lanczos-improved guide" in the NN era: **Chen, Hendry, Weinberg, Feiguin, NeurIPS 35, 7490-7503 (2022)** [abstract only] (Krylov/Lanczos on RBM states for J1-J2) and **Wang, He, Lu, PRB 113, 085120 (2026)** [abstract only] (supervised NN fit of the Lanczos state, then VMC). Both use H-powers for amplitude AND sign, no FN. Related for how a NN absorbs a Krylov step.

## 4. Sign structure from amplitudes / Hamiltonian-guided signs

**Westerhout, Katsnelson, Bagrov, Commun. Phys. 6, 275 (2023)** [full text, arXiv:2207.10675]. For real Hamiltonians, with amplitudes |psi_i| known, signs minimise E = sum_ij H_ij |psi_i||psi_j| S_i S_j, an auxiliary Ising model on the Hilbert-space basis; it is only mildly frustrated even for frustrated J1-J2 / kagome, solved by a deterministic O(K log K) greedy algorithm. Suppl. Note 5 describes (not implemented) hybridising explicit sign optimisation with a VMC amplitude loop. HOW IT DIFFERS: sign step is a combinatorial energy minimisation at fixed amplitudes (exact amplitudes needed on the sampled set); the amplitude step is plain VMC, not FN. My observation (not in the paper): M's rule is a synchronous local-field (Jacobi) sweep for this Ising model with a global shift T, so (a) is not new as "signs from energy at fixed amplitude", although the T-threshold form is different from their greedy edge ordering. Cite this as the closest "signs bootstrapped from amplitudes" work; it also pre-empts the abstract idea of alternating amplitude and sign steps (b), but not with FN.

**Westerhout, Astrakhantsev, Tikhonov, Katsnelson, Bagrov, Nat. Commun. 11, 1593 (2020)** [abstract; arXiv:1907.08186]. Supervised NN learning from exact ground-state data; generalisation collapses with frustration; the sign structure is much harder to learn than amplitudes. Motivation for a label-free sign route. Uses labels, so it does not pre-empt (a).

**Szabo and Castelnovo, Phys. Rev. Research 2, 033075 (2020)** [abstract; arXiv:2002.04613]. Two-network (amplitude/phase) ansatz; finds Marshall-sign low-energy states obstruct VMC for frustrated magnets. Motivation only.

**Others (optional):** Bukov, Schmitt, Dupont, SciPost Phys. 10, 147 (2021) (non-stoquastic ground state in a rugged NQS landscape); Roth, Szabo, MacDonald, PRB 108, 054410 (2023) (deep group-CNN VMC for J1-J2); **Ou, Huang, Ozolins, PRB 112, 165122 (2025)** [abstract only; arXiv:2510.02051]: modified SR with a LARGER imaginary-time step for the phase network than for the amplitude, no sign pretraining (J1-J2). Conceptually the large-tau sign step is related to (a) (large tau = lower T = more aggressive flips), but via parametric NQS updates, not a label-free local-energy threshold and not FN. A search for "signs from local energy / H psi thresholds" in NQS found nothing else.

## 5. NN guides for lattice FN / GFMC and learning from projector walkers

- **Lin, He, Lu, Chin. Phys. B 31, 080203 (2022)** [abstract only]: GFMC with an RBM guide for J1-J2. **Qin, PRB 102, 125143 (2020)** [abstract; arXiv:2006.15608] and **Lin, Guo, He, Xie, Lu, PRB 109, 235133 (2024)** [read via summary of arXiv:2406.12207]: PEPS guide for lattice FN GFMC, which fixes the nodes of the guide throughout; plain imaginary-time Green's function; the guide is never updated; no Lanczos/SR. **Lin, He, Guo, Lu, Chin. Phys. B 33, 117504 (2024)** [abstract only; arXiv:2503.08450]. These fix a guide whose signs are inherited from a variational state; M changes the guide's signs through the loop. No pre-emption of (a)-(c).
- **Inack, Santoro, Dell'Anna, Pilati, PRB 98, 235145 (2018); Pilati, Inack, Pieri, PRE 100, 043301 (2019); Brodoloni and Pilati, PRE 110, 065305 (2024)** [abstracts]. NN (RBM) guides for projective QMC; the 2019 paper trains the RBM on the PQMC walker distribution in repeated stints (KL/likelihood). This is the prior art for "learn the amplitude from walkers and re-run" and for iterating learn-and-project. Models are sign-problem free (quantum Ising, spin glass), so no node issue.
- **Ren, Fu, Wu, Chen, Nat. Commun. 14, 1860 (2023)** [abstract]: NN trial functions for continuum FN-DMC of molecules. Single FN step, no node iteration.
- **Ledinauskas and Anisimovas, SciPost Phys. 15, 229 (2023)** and **Gravina, Savona, Vicentini, Quantum 9, 1803 (2025)** [abstracts]: fit an NQS to a projected target (1 - tau H)psi (or time-evolved state) by supervised/infidelity minimisation with a frozen target. Closest ML-side analogues for "NN learns the FN/projected target"; they target the full signed state with no FN and no sign step. Gravina et al. treat quench dynamics (Ising); Ledinauskas treats J1-J2 ground states.
- Not found: any lattice work that (i) trains a NN on phi_FN by minimising the frozen FN energy with SR; (ii) alternates NN FN-learning with a sign update. Treat as unconfirmed absence.

## 6. PEPS-guided GFMC for J1-J2 and related lattice FN + tensor-network works

Lin et al. PRB 109, 235133 (2024) is the known work: PEPS guide, FN with fixed nodes, E_g = -0.5029(1) per site at J2=0.5 on 6x6 (per summary; verify against the paper before quoting). Related: Qin PRB 102, 125143 (2020) (the idea), Lin et al. Chin. Phys. B 33, 117504 (2024). A 2014-era "projector QMC with MPS" (arXiv:1403.3125) surfaced in search but I did not verify it; omitted. Nothing here pre-empts (a)-(c): the guide is static.

## Verdict on novelty

Not novel by itself (cite and acknowledge): (a) the Lanczos-step phase correction sgn[(1 + alpha H)psi] (Sorella 2001/2002; energy-optimised alpha, "FN over the Lanczos-step state"); (b) iterating FN projection and node improvement from walker information (SHDMC 2009; Sorella's SR self-consistency; Westerhout's proposed amplitude/sign loop); (c) the existence of the inequality chain <H>_{FN state} <= E_FN <= <H>_guide (ten Haaf et al.; Sorella Eq. 10) and the SHDMC statement that the smoothed guide has energy <= that of Psi_FN.

Plausibly novel (each needs your own statement and proof): (i) a stand-alone, amplitude-independent sign update with T chosen by the sign-only variational energy and no labels, as a deterministic exact-lattice step (distinct from Sorella's stochastic SR and Westerhout's greedy Ising solver); (ii) the explicit monotone guarantee <H>_{new} <= E_FN[old], with a nonempty fallback: because T = infinity returns the old signs, a T-scan that includes it gives <H>_new <= <H>_{phi_FN, old signs} <= E_FN-Hamiltonian ground energy [old], by Sorella Eq. 10 / ten Haaf; this is my derivation, check it; Caffarel et al. 2026 show the corresponding guarantee fails for SHDMC in practice; (iii) the full alternation to a self-consistent fixed point on a frustrated 2D lattice (J1-J2) with exact FN; (iv) the NN-at-scale layer (learning phi_FN from walkers or via frozen FN-energy SR). Suggested framing: "we turn the Lanczos-step phase correction of Sorella into a label-free sign update and close the loop with lattice FN, in the spirit of self-healing DMC, with a monotone guarantee".

## Recommended citations

Must: Sorella 2001; Sorella 2002 (notes); Reboredo-Hood-Kent 2009; van Bemmel 1994; ten Haaf 1995; Westerhout 2023; Caffarel 2026 (preprint); Lin 2024. Should: Sorella-Capriotti 2000; Sorella 1998; Hu 2013; Becca-Sorella 2017; Westerhout 2020; Szabo-Castelnovo 2020; Qin-Shi-Zhang 2016; Qin 2020; Pilati 2019; Inack 2018; Ledinauskas 2023; Gravina 2025; Ou 2025; Chen 2022; Wang 2026; Bajdich 2010. Optional: remaining entries below.

## BibTeX

All fields below were checked against Crossref/arXiv, except where noted in a comment.

```bibtex
@article{Sorella2001,
  author  = {Sorella, Sandro},
  title   = {Generalized Lanczos algorithm for variational quantum Monte Carlo},
  journal = {Phys. Rev. B},
  volume  = {64},
  pages   = {024512},
  year    = {2001},
  doi     = {10.1103/PhysRevB.64.024512},
  eprint  = {cond-mat/0009149},
  archivePrefix = {arXiv}
}

@misc{Sorella2002notes,
  author = {Sorella, Sandro},
  title  = {Effective hamiltonian approach for strongly correlated lattice models},
  year   = {2002},
  eprint = {cond-mat/0201388},
  archivePrefix = {arXiv},
  note   = {Lecture notes, Euro Winter School, Kerkrade. Final publication venue UNVERIFIED (Crossref lists a different-titled AIP Conf. Proc. 690, 318 (2003), doi 10.1063/1.1632143, which I did not confirm is the same text)}
}

@article{Reboredo2009,
  author  = {Reboredo, F. A. and Hood, R. Q. and Kent, P. R. C.},
  title   = {Self-healing diffusion quantum Monte Carlo algorithms: Direct reduction of the fermion sign error in electronic structure calculations},
  journal = {Phys. Rev. B},
  volume  = {79},
  pages   = {195117},
  year    = {2009},
  doi     = {10.1103/PhysRevB.79.195117},
  eprint  = {0808.3417},
  archivePrefix = {arXiv}
}

@article{Bajdich2010,
  author  = {Bajdich, Michal and Tiago, Murilo L. and Hood, Randolph Q. and Kent, Paul R. C. and Reboredo, Fernando A.},
  title   = {Systematic Reduction of Sign Errors in Many-Body Calculations of Atoms and Molecules},
  journal = {Phys. Rev. Lett.},
  volume  = {104},
  pages   = {193001},
  year    = {2010},
  doi     = {10.1103/PhysRevLett.104.193001},
  eprint  = {0912.3826},
  archivePrefix = {arXiv}
}

@article{Reboredo2012,
  author  = {Reboredo, Fernando Agust{\'i}n},
  title   = {Many-body calculations of low-energy eigenstates in magnetic and periodic systems with self-healing diffusion Monte Carlo: Steps beyond the fixed phase},
  journal = {J. Chem. Phys.},
  volume  = {136},
  pages   = {204101},
  year    = {2012},
  doi     = {10.1063/1.4711023},
  eprint  = {1008.0359},
  archivePrefix = {arXiv}
}

@misc{Caffarel2026,
  author = {Caffarel, Michel and Pinar, Manon and Scemama, Anthony},
  title  = {Self-Healing Diffusion Monte Carlo applied to a simple fermionic model: A critical assessment of the method},
  year   = {2026},
  eprint = {2609.01301},
  archivePrefix = {arXiv},
  note   = {Preprint; not peer reviewed as far as I could determine}
}

@article{QinShiZhang2016,
  author  = {Qin, Mingpu and Shi, Hao and Zhang, Shiwei},
  title   = {Coupling quantum Monte Carlo and independent-particle calculations: Self-consistent constraint for the sign problem based on the density or the density matrix},
  journal = {Phys. Rev. B},
  volume  = {94},
  pages   = {235119},
  year    = {2016},
  doi     = {10.1103/PhysRevB.94.235119},
  eprint  = {1608.07154},
  archivePrefix = {arXiv}
}

@article{Qin2023,
  author  = {Qin, Mingpu},
  title   = {Self-consistent optimization of the trial wave function within the constrained path auxiliary field quantum Monte Carlo method using mixed estimators},
  journal = {Phys. Rev. B},
  volume  = {107},
  pages   = {235124},
  year    = {2023},
  doi     = {10.1103/PhysRevB.107.235124},
  eprint  = {2303.10301},
  archivePrefix = {arXiv}
}

@article{SorellaCapriotti2000,
  author  = {Sorella, Sandro and Capriotti, Luca},
  title   = {Green function Monte Carlo with stochastic reconfiguration: An effective remedy for the sign problem},
  journal = {Phys. Rev. B},
  volume  = {61},
  pages   = {2599--2612},
  year    = {2000},
  doi     = {10.1103/PhysRevB.61.2599},
  eprint  = {cond-mat/9902211},
  archivePrefix = {arXiv}
}

@article{Sorella1998,
  author  = {Sorella, Sandro},
  title   = {Green Function Monte Carlo with Stochastic Reconfiguration},
  journal = {Phys. Rev. Lett.},
  volume  = {80},
  pages   = {4558--4561},
  year    = {1998},
  doi     = {10.1103/PhysRevLett.80.4558},
  eprint  = {cond-mat/9803107},
  archivePrefix = {arXiv}
}

@article{Hu2013,
  author  = {Hu, Wen-Jun and Becca, Federico and Parola, Alberto and Sorella, Sandro},
  title   = {Direct evidence for a gapless {$Z_2$} spin liquid by frustrating {N\'eel} antiferromagnetism},
  journal = {Phys. Rev. B},
  volume  = {88},
  pages   = {060402(R)},
  year    = {2013},
  doi     = {10.1103/PhysRevB.88.060402},
  eprint  = {1304.2630},
  archivePrefix = {arXiv}
}

@article{Becca2015,
  author  = {Becca, Federico and Hu, Wen-Jun and Iqbal, Yasir and Parola, Alberto and Poilblanc, Didier and Sorella, Sandro},
  title   = {Lanczos steps to improve variational wave functions},
  journal = {J. Phys.: Conf. Ser.},
  volume  = {640},
  pages   = {012039},
  year    = {2015},
  doi     = {10.1088/1742-6596/640/1/012039},
  eprint  = {1412.2656},
  archivePrefix = {arXiv}
}

@book{BeccaSorella2017,
  author    = {Becca, Federico and Sorella, Sandro},
  title     = {Quantum Monte Carlo Approaches for Correlated Systems},
  publisher = {Cambridge University Press},
  year      = {2017},
  doi       = {10.1017/9781316417041},
  note      = {Chapter-level content not checked}
}

@article{vanBemmel1994,
  author  = {van Bemmel, H. J. M. and ten Haaf, D. F. B. and van Saarloos, W. and van Leeuwen, J. M. J. and An, G.},
  title   = {Fixed-Node Quantum Monte Carlo Method for Lattice Fermions},
  journal = {Phys. Rev. Lett.},
  volume  = {72},
  pages   = {2442--2445},
  year    = {1994},
  doi     = {10.1103/PhysRevLett.72.2442}
}

@article{tenHaaf1995,
  author  = {ten Haaf, D. F. B. and van Bemmel, H. J. M. and van Leeuwen, J. M. J. and van Saarloos, W. and Ceperley, D. M.},
  title   = {Proof for an upper bound in fixed-node Monte Carlo for lattice fermions},
  journal = {Phys. Rev. B},
  volume  = {51},
  pages   = {13039--13045},
  year    = {1995},
  doi     = {10.1103/PhysRevB.51.13039},
  eprint  = {cond-mat/9412037},
  archivePrefix = {arXiv}
}

@article{Boninsegni1995,
  author  = {Boninsegni, Massimo},
  title   = {Ground state of a triangular quantum antiferromagnet: Fixed-node Green-function Monte Carlo study},
  journal = {Phys. Rev. B},
  volume  = {52},
  pages   = {15304--15311},
  year    = {1995},
  doi     = {10.1103/PhysRevB.52.15304}
}

@article{Westerhout2023,
  author  = {Westerhout, Tom and Katsnelson, Mikhail I. and Bagrov, Andrey A.},
  title   = {Many-body quantum sign structures as non-glassy {Ising} models},
  journal = {Commun. Phys.},
  volume  = {6},
  pages   = {275},
  year    = {2023},
  doi     = {10.1038/s42005-023-01388-6},
  eprint  = {2207.10675},
  archivePrefix = {arXiv}
}

@article{Westerhout2020,
  author  = {Westerhout, Tom and Astrakhantsev, Nikita and Tikhonov, Konstantin S. and Katsnelson, Mikhail I. and Bagrov, Andrey A.},
  title   = {Generalization properties of neural network approximations to frustrated magnet ground states},
  journal = {Nat. Commun.},
  volume  = {11},
  pages   = {1593},
  year    = {2020},
  doi     = {10.1038/s41467-020-15402-w},
  eprint  = {1907.08186},
  archivePrefix = {arXiv}
}

@article{SzaboCastelnovo2020,
  author  = {Szab{\'o}, Attila and Castelnovo, Claudio},
  title   = {Neural network wave functions and the sign problem},
  journal = {Phys. Rev. Research},
  volume  = {2},
  pages   = {033075},
  year    = {2020},
  doi     = {10.1103/PhysRevResearch.2.033075},
  eprint  = {2002.04613},
  archivePrefix = {arXiv}
}

@article{Ou2025,
  author  = {Ou, Xiaowei and Huang, Tianshu and Ozoli{\c n}{\v s}, Vidvuds},
  title   = {Improving neural network performance for solving quantum sign structure},
  journal = {Phys. Rev. B},
  volume  = {112},
  pages   = {165122},
  year    = {2025},
  doi     = {10.1103/fqxr-r8vw},
  eprint  = {2510.02051},
  archivePrefix = {arXiv}
}

@article{Bukov2021,
  author  = {Bukov, Marin and Schmitt, Markus and Dupont, Maxime},
  title   = {Learning the ground state of a non-stoquastic quantum Hamiltonian in a rugged neural network landscape},
  journal = {SciPost Phys.},
  volume  = {10},
  pages   = {147},
  year    = {2021},
  doi     = {10.21468/SciPostPhys.10.6.147},
  eprint  = {2011.11214},
  archivePrefix = {arXiv}
}

@article{Roth2023,
  author  = {Roth, Christopher and Szab{\'o}, Attila and MacDonald, Allan H.},
  title   = {High-accuracy variational Monte Carlo for frustrated magnets with deep neural networks},
  journal = {Phys. Rev. B},
  volume  = {108},
  pages   = {054410},
  year    = {2023},
  doi     = {10.1103/PhysRevB.108.054410},
  eprint  = {2211.07749},
  archivePrefix = {arXiv}
}

@inproceedings{Chen2022,
  author    = {Chen, Hongwei and Hendry, Douglas and Weinberg, Phillip and Feiguin, Adrian},
  title     = {Systematic Improvement of Neural Network Quantum States Using Lanczos},
  booktitle = {Advances in Neural Information Processing Systems 35},
  pages     = {7490--7503},
  year      = {2022},
  doi       = {10.52202/068431-0544},
  eprint    = {2206.14307},
  archivePrefix = {arXiv},
  note      = {Page range and DOI from Crossref; editors/publisher not checked}
}

@article{Wang2026,
  author  = {Wang, Jia-Qi and He, Rong-Qiang and Lu, Zhong-Yi},
  title   = {Generalized Lanczos method for systematic optimization of neural-network quantum states},
  journal = {Phys. Rev. B},
  volume  = {113},
  pages   = {085120},
  year    = {2026},
  doi     = {10.1103/m4c7-qz8l},
  eprint  = {2502.01264},
  archivePrefix = {arXiv}
}

@article{Lin2024PRB,
  author  = {Lin, He-Yu and Guo, Yibin and He, Rong-Qiang and Xie, Z. Y. and Lu, Zhong-Yi},
  title   = {Green's function {Monte Carlo} combined with projected entangled pair state approach to the frustrated {$J_1$-$J_2$} Heisenberg model},
  journal = {Phys. Rev. B},
  volume  = {109},
  pages   = {235133},
  year    = {2024},
  doi     = {10.1103/PhysRevB.109.235133},
  eprint  = {2406.12207},
  archivePrefix = {arXiv}
}

@article{Lin2024CPB,
  author  = {Lin, He-Yu and He, Rong-Qiang and Guo, Yibin and Lu, Zhong-Yi},
  title   = {A hybrid method integrating {Green's} function {Monte Carlo} and projected entangled pair states},
  journal = {Chin. Phys. B},
  volume  = {33},
  pages   = {117504},
  year    = {2024},
  doi     = {10.1088/1674-1056/ad84c9},
  eprint  = {2503.08450},
  archivePrefix = {arXiv}
}

@article{Lin2022,
  author  = {Lin, He-Yu and He, Rong-Qiang and Lu, Zhong-Yi},
  title   = {Green's function {Monte Carlo} method combined with restricted {Boltzmann} machine approach to the frustrated {$J_1$-$J_2$} Heisenberg model},
  journal = {Chin. Phys. B},
  volume  = {31},
  pages   = {080203},
  year    = {2022},
  doi     = {10.1088/1674-1056/ac615f}
}

@article{Qin2020,
  author  = {Qin, Mingpu},
  title   = {Combination of tensor network states and {Green's} function {Monte Carlo}},
  journal = {Phys. Rev. B},
  volume  = {102},
  pages   = {125143},
  year    = {2020},
  doi     = {10.1103/PhysRevB.102.125143},
  eprint  = {2006.15608},
  archivePrefix = {arXiv}
}

@article{Inack2018,
  author  = {Inack, E. M. and Santoro, G. E. and Dell'Anna, L. and Pilati, S.},
  title   = {Projective quantum {Monte Carlo} simulations guided by unrestricted neural network states},
  journal = {Phys. Rev. B},
  volume  = {98},
  pages   = {235145},
  year    = {2018},
  doi     = {10.1103/PhysRevB.98.235145},
  eprint  = {1809.03562},
  archivePrefix = {arXiv}
}

@article{Pilati2019,
  author  = {Pilati, S. and Inack, E. M. and Pieri, P.},
  title   = {Self-learning projective quantum {Monte Carlo} simulations guided by restricted {Boltzmann} machines},
  journal = {Phys. Rev. E},
  volume  = {100},
  pages   = {043301},
  year    = {2019},
  doi     = {10.1103/PhysRevE.100.043301},
  eprint  = {1907.00907},
  archivePrefix = {arXiv}
}

@article{Brodoloni2024,
  author  = {Brodoloni, L. and Pilati, S.},
  title   = {Zero-temperature {Monte Carlo} simulations of two-dimensional quantum spin glasses guided by neural network states},
  journal = {Phys. Rev. E},
  volume  = {110},
  pages   = {065305},
  year    = {2024},
  doi     = {10.1103/PhysRevE.110.065305},
  eprint  = {2407.05978},
  archivePrefix = {arXiv}
}

@article{Ren2023,
  author  = {Ren, Weiluo and Fu, Weizhong and Wu, Xiaojie and Chen, Ji},
  title   = {Towards the ground state of molecules via diffusion {Monte Carlo} on neural networks},
  journal = {Nat. Commun.},
  volume  = {14},
  pages   = {1860},
  year    = {2023},
  doi     = {10.1038/s41467-023-37609-3},
  eprint  = {2204.13903},
  archivePrefix = {arXiv}
}

@article{Ledinauskas2023,
  author  = {Ledinauskas, Eimantas and Anisimovas, Egidijus},
  title   = {Scalable imaginary time evolution with neural network quantum states},
  journal = {SciPost Phys.},
  volume  = {15},
  pages   = {229},
  year    = {2023},
  doi     = {10.21468/SciPostPhys.15.6.229},
  eprint  = {2307.15521},
  archivePrefix = {arXiv}
}

@article{Gravina2025,
  author  = {Gravina, Luca and Savona, Vincenzo and Vicentini, Filippo},
  title   = {Neural Projected Quantum Dynamics: a systematic study},
  journal = {Quantum},
  volume  = {9},
  pages   = {1803},
  year    = {2025},
  doi     = {10.22331/q-2025-07-22-1803},
  eprint  = {2410.10720},
  archivePrefix = {arXiv}
}
```

## Items not verified or only partly read

- Qin-Shi-Zhang 2016, Qin 2023, Bajdich 2010, Sorella-Capriotti 2000, Hu 2013, Becca 2015, Lin 2022/2024, Ou 2025, Chen 2022, Wang 2026, Pilati 2019, Brodoloni 2024, Ren 2023, Ledinauskas 2023, Gravina 2025: content judged from abstracts/summaries; bibliographic data verified.
- Lin 2024 PRB: statements about the guide being static come from a summariser reading of arXiv:2406.12207; confirm in the text.
- Becca-Sorella book chapters on lattice FN + Lanczos: not read.
- Sorella 2002 notes: venue unverified (lecture notes, arXiv only).
- Caffarel et al. 2026: preprint, read partially (abstract through Sec. II).
- Web search is US-only/limited; "none found" statements mean not found, not proven absent. Two more checks worth doing before submission: Google Scholar forward citations of Sorella 2001 and Reboredo 2009 for "lattice", "spin", "Heisenberg", "neural".
