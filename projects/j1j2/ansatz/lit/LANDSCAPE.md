# Variational ansatz landscape: spin-1/2 J1-J2 square lattice near J2/J1 = 0.5

Literature map for the ansatz-pioneer angle. Compiled 2026-10-23, read-only (no jobs).
Energies E/N in units of J1, spin-1/2 S.S convention, L x L torus (PBC) unless stated.
References: 6x6 ED -0.503810 (Schulz et al. 1996, wiki `schulz1996magnetic`); 10x10 no exact
value, reference = Chen-Heyl zero-variance extrapolation -0.497715(9) (arXiv:2302.01941;
Hu 2013 variance extrapolation -0.49781(2) disagrees by 1e-4, see
`results/literature/J1J2_BENCHMARKS_COST.md`). "rel" = (E - E_ref)/|E_ref|, my arithmetic.

Provenance tags on every claim:
- [full text] = paper read in full (here: only the three wiki-ingested papers, via
  `second_brain/wiki/sources/*.md`, which were checked against the published PDF).
- [section read] = the table/section named was read by me today (arXiv HTML/LaTeX), or by the
  2026-10-07/23 benchmark survey (marked "via BENCH" = `results/literature/J1J2_BENCHMARKS_COST.md`).
- [abstract only] = abstract or a figure caption quoted elsewhere; re-read before relying on it.

Benchmark numbers already verified in BENCH are reused, not re-derived.

## 0. Summary table

| # | Family | Builds in | Structurally cannot / measured failure | Best J2=0.5 number (rel) | Key ids |
|---|---|---|---|---|---|
| 1 | Jastrow + Marshall sign (MSR), stripe/collinear rules | exact sign of the unfrustrated limit; positive amplitudes; Neel/stripe correlations via Jastrow | any sign change relative to MSR. ED: MSR-weighted sign 0.9745 (4x4), 0.9650 (4x6), ~0.88 extrapolated 10x10; MSR breaks at J2 ~ 0.26 (CCM) | no standalone J1-J2 energy found; MSR is the prior inside most rows below; 6x6 MSR sign accuracy 0.9804 (2502.01264 Tab. 3) | 2002.04613; cond-mat/9804079; EPL 25, 545 (1994); 2502.01264 |
| 2 | Gutzwiller-projected partons / RVB (Z2 PSG), + Lanczos | fractionalised spinons, Z2 PSG, gapless Dirac nodes, power-law correlations, singlet; few parameters | p=0 projected state 5.2e-3 off at 6x6; needs p Lanczos steps + variance extrapolation; fixed nodal structure set by one mean-field determinant | 6x6 p=2 -0.503571(3) (4.7e-4); 10x10 p=2 -0.497549(2) (3.3e-4) | 1304.2630; 2005.12941; 2411.14114 |
| 3 | Pair-product / Pfaffian (mVMC) + RBM + quantum-number projection | singlet pairing, total-spin and lattice-momentum projection; RBM correlator on top | sublattice-restricted PP worse at 8x8 (-0.498460 vs -0.498886); cost grows with N^3-N^4; sign = Pfaffian x RBM phase | 6x6 -0.503765(1) (8.9e-5); 10x10 -0.497629(1) (1.7e-4) | 1709.06475; 2005.14142 |
| 4 | Backflow / neural backflow / hidden fermions | configuration-dependent orbitals: sign structure changes with configuration | no J1-J2 square energy found for any backflow or hidden-fermion state | none found (only Hubbard / t-J) | 1807.10770; 2111.10420; 2311.09450; 2507.10705 |
| 5 | NQS: RBM, CNN, GCNN, ViT, ResNet, CTWF, RNN, foundation | lattice symmetries (GCNN/projection), long-range attention, depth; usually MSR prior | without prior: Marshall trap (phase std 2.8e-3 rad from MSR at 10x10); autoregressive: rel 4.7e-3 at 10x10; supervised sign fit falls 100 -> 95 -> 89 % (L = 4, 6, 10) | 10x10 ResNet -0.4976921(4) (4.6e-5); ViT -0.497634(1) (1.6e-4); 6x6 none tabulated for deep NQS except CTWF rel only | 2302.01941; 2310.05715; 2211.07749; 2503.10462; 2502.09488 |
| 6 | Sign-structure learning / positivisation | sign as separate object: Ising-on-basis, greedy, phase nets, basis rotations | given exact amplitudes signs are easy at 16 sites; jointly with VMC they lock to MSR; easing the sign by local basis change is NP-complete in general | no energy (method studies); greedy recovers 16-site J2=0.55 signs exactly | 2207.10675; 1907.08186; 2011.11214; 1906.04654; 1906.02309; 2508.09870 |
| 7 | Tensor networks (DMRG, PEPS, PEPS-GFMC, SBS/EPS/CPS) and TN+NN | area law; DMRG exact-ish on small tori; PEPS 2D area law, OBC finite PEPS to 24x24; CNN-MPS hybrid | torus DMRG unconverged at 10x10 (M=8192: 4.4e-3); PEPS D<=6 guide 1.8e-2 at 6x6; SBS 5.5e-2 at 10x10 | 6x6 CNN-MPS -0.50380737(5) (5.2e-6); 10x10 CNN-MPS -0.4976939(2) (4.2e-5) | 1311.5962; 2009.01821; 2406.12207; 2603.14425 |
| 8 | Lanczos/Krylov-boosted states, Hamiltonian-informed | (1 + a H + b H^2) psi: H-generated nodes, systematic | cost exponential in p for exact steps; NN fit of Lanczos state underfits at L >= 6 | 6x6 RBM p=2 -0.50378(4) (6.0e-5); 10x10 GWF p=2 above | 2206.14307; 2502.01264; 1304.2630 |
| 9 | Interference of components (sums of NQS/MPS, projections, multi-Pfaffian, sector sums) | nodes from cancellation between components | no J1-J2 multi-Pfaffian or flux/topological-sector sum found; existing sums are 2-3 terms | 6x6 3-NQS superposition sign accuracy 0.9983 vs MSR 0.9804; energy -0.503178(4) (1.25e-3) | 2502.01264; 2603.14425 (n=2 MPS sum); 2005.14142 |

## 1. Jastrow + Marshall sign rule; MSR violation studies

**Physics built in.** Sign = Marshall rule (-1)^{N_up on sublattice A}, exact at J2 = 0 by the
Marshall-Lieb-Mattis theorem; the stripe (collinear) rule is the J1 = 0 analogue. Amplitudes
from a Jastrow factor carry Neel (or stripe) correlations. Nothing about fractionalisation,
VBS or the near-node region: the state has no nodes beyond MSR sign flips.

**Cannot represent.** Any configuration whose true sign differs from MSR.
- ED average signs at J2 = 0.5, PBC: Marshall-adjusted <s>_MSR = 0.9745 (4x4), 0.9650 (4x6),
  ~0.88 at 10x10 by exponential extrapolation, not a computation (Szabo-Castelnovo
  arXiv:2002.04613, Tab. III) [full text]. Read as ~6% (4x6) and ~12% (10x10 extrap.) of the
  |psi|-weighted mass carrying the "wrong" Marshall sign.
- 6x6 J2 = 0.5: MSR sign accuracy 0.9804, 0.9352 at J2 = 0.55, against ED (Wang-He-Lu
  arXiv:2502.01264, Tab. 3) [section read].
- Onset: CCM gives MSR breakdown at J2/J1 ~ 0.26 on the square lattice (Bishop-Farnell-Parkinson,
  PRB 58, 6394 (1998), cond-mat/9804079) [abstract only]. Richter-Ivanov-Retzlaff, EPL 25, 545
  (1994) argue from ED weights that MSR "survives" approximately up to large J2 [abstract only];
  Voigt-Richter-Ivanov, Physica A 245, 269 (1997) treat excited states [title only, cited in
  1807.09422]. Zeng-Parkinson 1995 not located by search.
**Best number.** No J2 = 0.5 energy of a plain Jastrow x MSR state found. MSR is the sign prior of
Choo 2019 (CNN), Hu 2013 (fermionic, implicit), CNN-MPS 2603.14425 (10x10 best uses the Marshall
transform, text below Tab. 1) [section read].

## 2. Gutzwiller-projected partons / RVB (Z2 PSG), Lanczos steps, variance extrapolation

**Physics built in.** Schwinger-fermion mean field with a Z2 PSG (Z2Azz13 / d-wave-type pairing),
Gutzwiller projection to one fermion per site: exact SU(2) singlet, fractionalised spinons,
gapless Dirac points -> power-law spin and dimer correlations, PSG fixes how symmetries act on
spinons. Sign structure comes from one projected determinant/Pfaffian; on top, a long-range
Jastrow. Lanczos steps (1 + a1 H + a2 H^2) add H-generated sign changes.

**Cannot / evidence.**
- Bare projected state is not accurate: 6x6 p=0 -0.50117(1), rel 5.2e-3; p=1 -0.50323, p=2
  -0.503571(3) (Hu-Becca-Parola-Sorella arXiv:1304.2630, Tabs. I-III) [section read, via BENCH].
  The extrapolated energy -0.50382(1) (Tab. IV) is *below* ED by 1e-5, i.e. the variance
  extrapolation is not a bound.
- Nodes are those of a single mean-field determinant: nothing outside this one-determinant
  manifold except via Lanczos powers or added NN factors (Ferrari-Becca-Carrasquilla
  arXiv:1906.00463, GWF+RBM 10x10 -0.49575(3), rel 3.9e-3, quoted in 2002.04613 Tab. I)
  [full text of the quoting paper].
- Competing reading: Gutzwiller-guided DMRG supports a Z2 Dirac SL characterised by its PSG
  (Jin-Tu-Zhang arXiv:2411.14114) [abstract only]; DMRG/FAMPS find no SL, direct Neel-VBS DQCP at
  0.535(3) (Qian-Qin arXiv:2309.13630) [abstract only]. The PSG-built-in ansatz is thus not
  confirmed to be the right state class at 0.5.
**Best numbers.** 6x6 p=2 -0.503571(3) (4.7e-4); 8x8 p=2 -0.49886(1); 10x10 p=2 -0.497549(2)
(3.3e-4) [section read, via BENCH]. Ferrari-Becca arXiv:2005.12941 (singlet/triplet spectra,
level crossings) [abstract only].

## 3. Pair-product / Pfaffian (mVMC) + RBM + quantum-number projection

**Physics built in.** Pair-product (geminal) f_ij in spin language = general RVB singlet pairing,
Pfaffian evaluation; total-spin S = 0 projection and lattice-momentum/point-group projection
(sum over symmetry images, i.e. a built-in interference of 16 x |T| components); RBM factor adds
non-local correlations and a phase. Tahara-Imada arXiv:0807.1303 (projection framework)
[title only]; Nomura-Darmawan-Yamaji-Imada arXiv:1709.06475 (RBM+PP) [abstract only].

**Cannot / evidence.**
- Restricting f_ij to a 4x4 sublattice structure (needed for large L) costs accuracy: 8x8
  -0.498886(1) (full) vs -0.498460(6) (4x4 sublattice) (Nomura-Imada arXiv:2005.14142,
  Tab. I, App. B) [section read, via BENCH]. Large-L results all use restricted PP.
- 16 hidden units only at 6x6; error saturates at ~1e-4 (6x6 8.9e-5, 10x10 1.7e-4), above deep
  NQS at 10x10 (4.6e-5). No published 6x6 trend with more hidden units found.
**Best numbers.** 6x6 -0.503765(1) (8.9e-5); 8x8 -0.498886(1) (best published UB at 8x8); 10x10
-0.497629(1) (1.7e-4) [section read, via BENCH]. Used as the guide for level-spectroscopy claim of
a Dirac-type nodal SL (PRX 11, 031034).

## 4. Backflow, neural backflow, hidden fermions

**Physics built in.** Orbitals phi_k(r_i; x) depend on the whole configuration x, so the
determinant's nodes move with the configuration; hidden-fermion determinants (augmented space,
NN-constrained) are universal (Robledo Moreno et al. arXiv:2111.10420) [abstract only]; neural
backflow "directly alters the sign structure" (Luo-Clark arXiv:1807.10770) [abstract only];
unified view NNBF = HFDS (Liu-Clark arXiv:2311.09450) [title only]; NN-augmented Pfaffians
(Chen-Wan-Sengupta-Georges-Roth arXiv:2507.10705) [title only].

**Cannot / evidence.** Not a structural failure, a missing test: searches found no backflow or
hidden-fermion energy for the spin J1-J2 square model. All hits are Hubbard / t-J / Emery
(e.g. 2311.05749, 2411.10430, 2511.07566). Cost: per-sample determinant update O(N^3) with
configuration-dependent orbitals, no fast-update [general knowledge, not from a paper read].
Backflow in Gutzwiller VMC is established for the Hubbard large-U limit (Tocchio et al.
arXiv:0805.1476, 1102.3017) [title only]. Searches: `abs:"backflow" AND abs:Heisenberg AND
abs:frustrated` (0 hits); `abs:"hidden fermion" AND abs:"neural"` (12 hits, none J1-J2 spins).
**Best number.** None found at J2 = 0.5. The nearest relative is CNN-MPS's "backflow MPS"
(family 7), whose MPS tensors depend on the configuration through a CNN.

## 5. Neural quantum states (RBM, CNN, GCNN, ViT, ResNet+MinSR, CTWF, RNN, foundation models)

**Physics built in.** Translation by convolution or symmetrisation; full space group + spin
flip by projection (GCNN: Roth-MacDonald arXiv:2104.05085, Roth-Szabo-MacDonald arXiv:2211.07749);
long-range correlations by attention (ViT, Viteritti/Rende arXiv:2310.05715; ViT-SA to 20x20
arXiv:2602.02665); depth (ResNet + MinSR, Chen-Heyl arXiv:2302.01941). Most use the Marshall
sign as a prior (Choo arXiv:1903.06713; cRNN arXiv:2207.14314 "applied on top" [section read]);
Rende 2310.05715 trains without a prior (BENCH). No explicit fractionalisation, PSG or node model.

**Cannot / evidence.**
- Marshall trap: phasor-sum CNN at 10x10 converges to a phase with std 2.8e-3 rad from MSR,
  although the exact state is not Marshall-signed; energy -0.494757(12), rel 5.9e-3
  (2002.04613, Fig. 2, Tab. I) [full text]. Symmetry violation 10x larger at 0.5 than at 0 (Tab. II).
- Rugged landscape, not expressivity, blocks sign learning for decoupled amplitude/phase nets
  (Bukov-Schmitt-Dupont arXiv:2011.11214, SciPost Phys. 10, 147) [abstract only].
- Generalisation of the sign "drops abruptly" with frustration in supervised learning
  (Westerhout et al. arXiv:1907.08186) [abstract only]. Supervised fit of a Lanczos state: sign
  accuracy 100 % (L=4), 95.8 % (L=6), 91.2 % (L=10, J2=0.5); amplitude loss 1e-8 -> 0.21 -> 0.46
  (2502.01264, Tab. 1) [section read].
- Autoregressive nets: RNN at 10x10 rel 4.7e-3, sigma^2/N 1.0e-3 (quoted in Chen-Naik-Heyl
  arXiv:2503.10462, Fig. 2 text, its ref. [49]) [section read]; their sampling is incompatible
  with the optimal symmetrisation (Reh-Schmitt-Gaertner arXiv:2301.06788) [abstract only].
- Complex RBM: faithful only when the phase is connected to a stoquastic point; "hindered not only
  by the sign structure but also by their amplitudes" (Park-Kastoryano arXiv:2012.08889, 1D only)
  [abstract only].
- Cost: best 10x10 ViT ~1,900 A100-h; ResNet record cost not stated (BENCH).
**Best numbers (10x10).** ResNet2+MinSR -0.4976921(4) (4.6e-5); ViT HF checkpoint -0.4976764(7)
per 2503.10462 (7.9e-5); deep ViT+SRt -0.497634(1) (1.6e-4); GCNN -0.497437(7) (5.6e-4); CNN Choo
-0.49516(1) (5.1e-3) [section read, via BENCH]. Note: 2603.14425 Tab. 1 labels -0.4976764(7) as
"CTWF 2025 [5]"; 2503.10462 itself attributes that number to the factored-attention ViT
(N_p = 434,760) and gives CTWF only in a figure -> mislabel in 2603.14425 [section read].
**6x6.** No deep-NQS energy tabulated; CTWF Tab. 1 gives rel. errors for RBM/CNN/transformer
variants at 6x6, magnitude 2e-3 to 1.5e-2 in the printed column, units not stated in the text I
read [section read; re-read before quoting]. Choo CNN -0.50185(1) (3.9e-3) [via BENCH].
**Foundation / large models.** FNQS (Rende et al. arXiv:2502.09488, J2 in [0.4, 0.6], 10x10,
V-scores only); inverse design 2609.40209; KZ dynamics 2609.05556; scaling laws 2606.02794
(V-score ~ f^-0.86(3)) [abstract / via BENCH]. HQT arXiv:2607.00398 claims 8x8 -0.5001(1),
below every extrapolated reference: treat as unreliable (BENCH).

## 6. Sign-structure learning and positivisation

**Physics built in.** The sign as its own object: (a) Ising model on the Hilbert-space basis with
couplings |psi_i||psi_j|H_ij, solved greedily (Westerhout-Katsnelson-Bagrov arXiv:2207.10675)
[full text]; (b) separate phase networks with tailored optimisers (evolution strategy, Chen et al.
arXiv:2111.06411; larger imaginary-time step for phase, Ou-Huang-Ozolins arXiv:2510.02051)
[abstract only]; (c) local basis rotations that remove or reduce signs (Torlai et al.
arXiv:1906.04654; Hangleiter et al. arXiv:1906.02309; Levy-Clark arXiv:1907.02076; sign-optimised
QMC, Herz et al. arXiv:2607.24679; brute-force single-qubit positivisation of 1D J1-J2,
Bannykh et al. arXiv:2511.17957) [abstract only]; (d) Boolean-Fourier / Walsh analysis of sign
complexity (Schurov et al. arXiv:2508.09870; "Walshness", Paul et al. arXiv:2610.00505, recovers
MSR as optimal basis) [abstract only].

**Cannot / evidence.**
- Greedy sign recovery is exact on the 16-site J1-J2 cluster at J2 = 0.55 *given exact
  amplitudes* (2207.10675 Tab. 1) [full text]; 16 sites is the cluster Schulz 1996 calls
  anomalous for 0.3 < J2 < 0.8 (wiki). No J1-J2 test beyond 16 sites; cluster version reaches
  sign overlap ~0.9 on 32-36 site kagome/pyrochlore/random, "with a chance of about 70%".
- Basis-rotation easing is NP-complete in general for nearest-neighbour Hamiltonians and simple
  bases (1906.02309) [abstract only]. Single-qubit rotations: tested on 1D J1-J2 and ladders and
  the maple-leaf lattice, not the 2D square J1-J2 at 0.5 (2511.17957, 2607.24679) [abstract only].
- Boolean-Fourier polynomials "cannot yet be directly used in variational optimization"
  (2508.09870) [abstract only].
**Best number.** None: no family-6 paper reports a J2 = 0.5 square energy beyond 16 sites, except
Ou et al. 2510.02051 (J1-J2, size/number not read) [abstract only].

## 7. Tensor networks and TN + NN hybrids

**Physics built in.** Area-law entanglement. DMRG on tori (Gong et al. arXiv:1311.5962; Wang-Sandvik
level crossings arXiv:1702.08197); finite OBC PEPS to 24x24 with power-law spin and dimer
correlations (Liu et al. arXiv:1806.07031, arXiv:2009.01821) [abstract only]; PEPS-guided
fixed-node GFMC (Lin et al. arXiv:2406.12207) [full text]; string-bond / entangled-plaquette /
correlator-product states (Sfondrini et al. PRB 81, 214426; Mezzacapo arXiv:1203.6381; Changlani et
al. arXiv:0907.4646) [title only]; CNN+PEPS (Liang-Dong-He arXiv:2009.14370) [title only];
neural disentangler + configuration-dependent ("backflow") MPS, sum of n = 2 MPS (nuTNS, Fan et al.
arXiv:2603.14425) [section read].

**Cannot / evidence.**
- 2D DMRG: 10x10 torus M = 8192 SU(2) gives -0.495530, rel 4.4e-3; SI truncation error 8e-5
  with 32000 states (BENCH, 1311.5962 Tab. I) [section read, via BENCH].
- PEPS: thermodynamic-limit 2x2 PEPS tiled on a 6x6 torus, D = 6 guide energy -0.4947(4), rel
  1.8e-2; after fixed-node GFMC -0.5033(4) (1.0e-3); fixed-node gain collapses at J2 = 0.55
  (delta ~ 0.001) (2406.12207, Tabs. I-II) [full text]. Critical (power-law) correlations need
  D -> infinity; Zhao-Takahashi-Sandvik dispute the 2018 PEPS SL claim (arXiv:1909.12788) [title only].
- SBS: 10x10 J2 = 0.5 -0.4705, rel 5.5e-2 (quoted in Liang et al. arXiv:1807.09422, Fig. 3
  caption) [section read].
**Best numbers.** 6x6 CNN-MPS (h, D, l) = (32, 20, 20): -0.50380737(5), rel 5.2e-6; pure CNN
(D = 1) -0.5037957(4), 2.8e-5; DMRG -0.503805 (1e-5). 10x10 CNN-MPS -0.4976939(2) (4.2e-5, below
Chen-Heyl -0.4976921(4)); T-MPS -0.4976923(2). 16x16 CNN-MPS -0.4969140(5); 20x20 -0.4967987(6) (C4v),
below ViT-SA -0.496732(1) (2603.14425, Tab. 1 and text) [section read]. ~2e5 parameters, MARCH
optimiser, 4096 samples, 15000 steps (Suppl. Tab. S1). Cost in GPU-h not found. Preprint.

## 8. Lanczos / Krylov-boosted and Hamiltonian-informed states

**Physics built in.** psi_p = sum_{k<=p} a_k H^k psi_0: nodes generated by H acting on the
guide, monotone energy improvement, variance-based extrapolation. Implementations: GWF + p steps
(1304.2630); RBM + Lanczos (Chen-Hendry-Weinberg-Feiguin arXiv:2206.14307); supervised NN fit of
each Lanczos vector then VMC on the amplitude nets (Wang-He-Lu arXiv:2502.01264, PRB 113, 085120)
[section read]; NN fit to (1 - tau H) psi (Ledinauskas-Anisimovas arXiv:2307.15521) [abstract
only]. Sorella's lattice FN over a Lanczos-step state is the projector analogue
(`paper/RELATED_WORK_NOTES.md` sec. 3).

**Cannot / evidence.** Exact p-step evaluation costs ~ (zN)^p local-energy terms per sample; NN
compression of Lanczos states underfits at L >= 6 (amplitude overlap 0.846 at 6x6, J2 = 0.5,
2502.01264 Tab. 2) [section read]. The p = 1 gain at 10x10 is small: -0.495627(6) -> -0.496102(7)
(rel 4.2e-3 -> 3.2e-3; Tab. 4) [section read].
**Best numbers.** 6x6 RBM p=2 -0.50378(4) (6.0e-5) [via BENCH, 2206.14307 Tab. 2]; 6x6 NQS-Lanczos
VMCL p=2 -0.503178(4) (1.25e-3); 10x10 GWF p=2 -0.497549(2) (3.3e-4).

## 9. Interference of several components (nodes from cancellation)

**Built in / where it exists.**
- Symmetry projection = coherent sum over |G| images (RBM+PP, GCNN, ViT symm. stages): nodes can
  appear where images cancel; this is how all best 10x10 numbers are reached (BENCH).
- Superposition of p+1 separately parametrised NQS with Lanczos coefficients: 6x6 J2 = 0.5 sign
  accuracy rises from 0.9804 (MSR) to 0.9983(1) (SLL) / 0.9981(1) (VMCL), J2 = 0.55: 0.9352 ->
  0.9960(1) (2502.01264, Tab. 3) [section read]. The energy (1.25e-3) remains far from the
  single-network records, so the sign gain does not reach the energy.
- nuTNS sums n = 2 backflow MPS (2603.14425 Suppl. S2.4) [section read]; hidden-fermion and
  determinant states are sums over permutations (family 4).
**Not found.** Multi-Pfaffian / multi-determinant expansions for the J1-J2 spin model; sums of
Gutzwiller states over flux or topological sectors on the torus; mixture-of-experts NQS for
ground states. Searches: `abs:"multi-Pfaffian" OR abs:"sum of Pfaffians" OR abs:"multiple
Pfaffians"` (7 hits, none spin models); `abs:"topological sectors" AND abs:"Gutzwiller" AND
abs:"spin liquid" AND abs:"superposition"` (0); `abs:"mixture of experts" AND abs:"quantum
states"` (1 hit, quantum circuits); `abs:"neural quantum state" AND (abs:"linear combination" OR
abs:"weighted sum")` (3 hits, dynamics / AFQMC trials).

## 10. Gaps: what no family represents natively

Search tool: `mcp__arxiv__search_papers` (title/abstract fields) unless stated; OpenAlex for
pre-arXiv items. "0 hits" = absence in abstracts, not proof of absence.

**G1. An explicit model of the nodal region: where the sign flips and how small |psi| is there.**
Every family carries the sign either as MSR x smooth correction (1, 5), as the nodes of one
mean-field determinant (2, 3), or as an unconstrained phase net (5, 6). None parametrises the
nodal set or ties low amplitude to the sign boundary (as a fixed-node guide needs).
Evidence it is missing and matters: MSR misses ~3-12% of |psi|-mass (family 1 numbers); sign
accuracy of a supervised NN falls to 91% at 10x10 (2502.01264 Tab. 1); the Marshall trap
(2002.04613). Searches: `abs:"neural" AND abs:"quantum states" AND (abs:"nodal" OR abs:"nodes")
AND abs:"lattice"` (0 hits); `abs:"neural" AND abs:"wave function" AND abs:"small amplitude" AND
abs:"sign"` (0); `abs:"fixed-node" AND abs:"neural" AND (abs:"spin" OR abs:"lattice")` (1 hit,
continuum fermions 2411.02257).

**G2. A closed-form (non-Marshall) sign rule at J2 ~ 0.5, or a sign-domain description.**
MSR is known to break near J2 ~ 0.26 (cond-mat/9804079) and the stripe rule holds only deep in the
collinear phase; no replacement rule, domain-wall picture or "defect in MSR" ansatz exists.
Westerhout 2023 states nothing about sign domains (wiki coverage note); Boolean-Fourier sign
polynomials are not yet variational (2508.09870). Searches: `abs:"sign rule" AND (abs:"beyond
Marshall" OR abs:"non-Marshall" OR abs:"generalized Marshall")` (1 hit, triangular bosons
0809.1667); `abs:"sign structure" AND (abs:"domain wall" OR abs:"domains") AND abs:"wave function"`
(0); `abs:"Marshall sign rule" AND (abs:violation OR abs:frustrated)` (6 hits, none a new rule);
OpenAlex "violation of Marshall-Peierls sign rule frustrated J1-J2" (Richter 1994, Bishop 1998,
Voigt-Richter 2000; all diagnose, none propose a rule).

**G3. PSG / symmetry fractionalisation inside a neural or tensor-network ansatz.** Only
partons (family 2) and Gutzwiller-guided DMRG (2411.14114) carry a PSG; NQS impose only the
physical symmetry group (projection), so they cannot distinguish Z2 from U(1) PSG classes with the
same symmetric amplitudes. Searches: `abs:"neural" AND abs:"projective symmetry group"` (1 hit,
2501.00096, parton VMC + FRG, no NQS); `abs:"Gutzwiller" AND abs:"restricted Boltzmann" AND
abs:frustrated` (1 hit, 1906.00463, PSG fixed by the parton part, not learned).

**G4. Interference across topological / flux sectors.** A gapped or gapless Z2 SL on the torus
has sector-dependent ground states; no ansatz for J1-J2 sums projected states (or NQS) over
sectors, and NQS have no handle on sectors at all. Searches: `abs:"topological sectors" AND
abs:"Gutzwiller" AND abs:"spin liquid" AND abs:"superposition"` (0);
`abs:"neural" AND abs:"spin liquid" AND (abs:"topological sector" OR abs:"topological degeneracy"
OR abs:"flux sector")` (0).

**G5. Configuration-dependent orbitals (backflow / hidden fermions) for the spin J1-J2 model.**
The one family that moves nodes with the configuration natively has never been benchmarked on it.
Searches: `abs:"backflow" AND abs:Heisenberg AND abs:frustrated` (0); `abs:"hidden fermion" AND
abs:"neural"` (12, none J1-J2); `abs:"Pfaffian" AND abs:"Heisenberg" AND abs:"neural" AND
abs:"frustrated"` from 2023 (0). Closest: CNN-MPS "backflow MPS" (2603.14425).

**G6. Sign learned from H without labels inside the ansatz** (the project's own route). No ansatz
updates its signs by a local-energy / H psi rule and feeds them to lattice FN; see
`paper/RELATED_WORK_NOTES.md` (Sorella 2001/2002, SHDMC 2009 pre-empt the ideas, not the package).
Searches recorded there; repeated here: `abs:"fixed-node" AND abs:"neural"` (above, 1 hit,
continuum). BENCH: no FN/GFMC with a deep-NQS guide published for J1-J2 at 0.5.

## 11. Papers to read in full before decisions

- arXiv:2603.14425 (Fan, ..., Qin, Xiang 2026, nuTNS / CNN-MPS): new best variational numbers at
  6x6, 10x10, 16x16, 20x20; uses MSR at 10x10; preprint, cost not stated; mislabels the
  -0.4976764(7) row. Read sections only. Candidate wiki ingest.
- arXiv:2502.01264 (Wang-He-Lu, NQS Lanczos): the only measured sign accuracies vs ED at 6x6 for
  MSR and for a superposition ansatz; underfitting numbers.
- arXiv:2302.01941 (Chen-Heyl): 10x10 record and ZVE reference; sign prior not verified (LaTeX
  and HTML not retrievable by the tool today).
- arXiv:2011.11214 (Bukov-Schmitt-Dupont): only abstract read; claims landscape, not
  expressivity, blocks sign learning.

## Not checked

Hu 2013, Gong 2014, Nomura-Imada 2021, Rende 2023, Chen-Feiguin 2022 numbers are reused from BENCH,
not re-read. Zeng-Parkinson 1995 not found. Mezzacapo EPS energies not extracted. CTWF 6x6 table
units unclear. Liu et al. 2018 PEPS (1806.07031) numbers not read. Qian-Qin 2309.13630 energies
not read.
