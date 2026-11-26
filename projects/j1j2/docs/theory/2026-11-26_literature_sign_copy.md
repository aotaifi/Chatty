# Literature scout: why a spins-only CNN underfits the one-step sign, and what fixes it (2026-11-26)

Conclusion first: the literature on sign learning (Westerhout, Schurov, Bukov) reports a *generalisation* or
*optimisation-landscape* failure; ours is an *underfit of the training set*, insensitive to width, depth and loss,
with identical refits. That signature matches Shalev-Shwartz et al.'s "uninformative gradient" failure for
sign-of-product / parity targets, which they prove is independent of architecture size and fix by
*decomposition* (per-part supervision) or by handing the hard feature in as input. Our label is exactly such a
composite: y(x) = sgn sum_b t_b(x), t_b = J_b rho_b s(z_b), with every t_b known exactly per row.

Reading labels: "read in full", "read section X", "abstract only", "wiki note" (project second-brain ingest, not
re-read today). TRIED = /project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/TRIED_IDEAS.md.

## Items

| # | arXiv | Read | Result that matters (number) | Assumption; does our problem break it? | TRIED covers? |
|---|---|---|---|---|---|
| 1 | 1907.08186 Westerhout et al., NQS of frustrated magnets: generalisation and sign | abstract only | Supervised sign learning from a subset generalises poorly as frustration rises; signs much harder than amplitudes; "generalisation rather than expressibility" is the issue | Training set small vs Hilbert space. BROKEN: we fail on our own 1.17M rows, so their diagnosis does not transfer | A8-A12, K11-K12 (spins-only sign nets) |
| 2 | 2508.09870 Schurov, ..., Westerhout, Boolean Fourier analysis of sign structures | read abstract, Sec. 2 findings list, Sec. 3.2 end, Sec. 3.3 (Table 1), Sec. 4 | Frustrated signs are dominated by *high-degree* parities (amplitudes: degree 2-4); square lattice at intermediate J2 shows Marshall, stripe and "stretched checkerboard" sublattice parities. Adding <= 8 parity functions sampled by Fourier weight as extra inputs: kagome-24 test overlap 0.260(17) -> 0.719(6); degree-2 parities *hurt* (0.13-0.23) | 24 sites, exact labels, 1-hidden-layer MLP, generalisation metric. Partly broken: our failure is fit, not transfer, and our target depends on continuous rho | NOT covered as classifier inputs; ANSATZ_MEMO C1 used the patterns as wavefunction components (dead) |
| 3 | 2610.00505 Paul et al., Walshness | read abstract, main-text definitions (Eqs. 1-4), frustrated-AFM section | Proves: bounded Walsh-spectrum weight at high Hamming degree <=> compact NQS (several architectures). Single-site basis optimisation (a classical spin model) recovers Marshall/stripe and at g=0.5 gives a canted basis beyond Marshall | Basis change limited to single-site unitaries; tested on XY/XYZ, not J1-J2 Heisenberg. A sign gauge g(x) that already carries the high-degree part lowers the Walsh weight of the residual target | NOT covered (gauge used so far = old sign s) |
| 4 | 2207.10675 Westerhout, Katsnelson, Bagrov, signs as non-glassy Ising models | wiki note | With exact |psi| fixed, signs are the ground state of an Ising model on the basis; greedy recovers exact 16-site J2=0.55 signs; cluster greedy overlap ~0.9 at 32-36 sites. Our label y is that model's local-field sign (one synchronous field update) | Exact amplitudes. Holds for our labels (guide amplitudes given) | A27 (greedy orbit flips), A1/A5 |
| 5 | 2002.04613 Szabo-Castelnovo; 2011.11214 Bukov-Schmitt-Dupont | wiki note (Szabo); abstract only (Bukov) | Separate phase net (phasor sum, phase-only pretraining at uniform amplitude) learns Marshall but locks to it at J2=0.5 on 10x10 (Szabo). Bukov: rugged landscape, "expressivity and sampling not primary limits" | Variational (energy) training, not supervised. Our supervised underfit is the same lock-in seen from another side: P38 found flip errors point to the old sign | A35, P28 (phase classes), K12 |
| 6 | 1703.07950 Shalev-Shwartz, Shamir, Shammah, Failures of gradient-based deep learning | read Sec. 2 (parities, Thm 1) and Sec. 3.1-3.2 | Thm 1: for a family H of uncorrelated targets the gradient's dependence on the target falls as 1/|H|, *for any architecture*: wider/deeper does not help. Target = product of k sign(.) parts: end-to-end fails for k >= 3 after 20000 iterations; adding a loss on each part solves it in 2500 | Uncorrelated target family, Gaussian inputs (theory). Our symptoms (width/depth/loss-invariant, refits identical, 55% in every decade) match; the per-part labels exist for us (t_b) | NOT covered: P38 regressed the *total* field v; B23 BondNet used per-bond *inputs* for amplitude |
| 7 | 2003.03828 Chrysos et al., Pi-nets (polynomial nets) | abstract only | Multiplicative (Hadamard-product) layers give high-degree polynomial expansions directly | Natural-image setting. A product layer represents s(z_b)*rho_b-type terms exactly in principle; no evidence on sign margins | NOT covered |
| 8 | 2311.09450 Liu-Clark, NNBF <-> hidden-fermion determinants | abstract only | Row selection in determinant states gives "significant sign and amplitude modulation between nearby configurations": the node moves through orbitals, not through a classifier threshold | Fermionic Hubbard. Our PP sign classes (A35, P28: all 12 worse than sgn(M1)) already tested the non-backflow version | A35, P28, A42 (PP sign); backflow-Pfaffian *sign* not tried; P10: backflow breaks cheap ratios (36.5x) |
| 9 | 2510.26906 Loehr-Clark, Enhancing NNBF; 2110.06390 Kochkov et al., graph networks | abstract only | One Lanczos step, DMC and symmetry projection give similar gains on NNBF (Hubbard 4x16); GNN message passing on the Hamiltonian graph reaches SOTA on Heisenberg models | Energy, not stored sign. Our loop already beats SR by 1.06e-3/site from a bad start (A50) | D1, C18, A26 (GNN rankers) |
| 10 | 2305.14294 Sinibaldi et al., projected quantum evolution; SHDMC 0808.3417 | abstract only | Fitting each step by infidelity minimisation is unbiased where tVMC is biased near zeros; per-step fit error is the controlled quantity | Unitary dynamics. Same lesson as our "tax per round" bookkeeping | C17 (SHDMC, Caffarel 2609.01301, Kairon-Clark 2609.16308), B10-B12 |

Not found: any paper that stores a Lanczos/Krylov *sign step* in a network, or that learns sgn of a
Hamiltonian-weighted neighbour sum. Searches: arXiv `abs:"sign structure" AND abs:"neural network" AND abs:frustrated`,
`abs:"Lanczos step" AND ...`, alphaxiv discover (Lanczos/local-energy-input NQS; sign-learning with separate nets).

## Ranked remedies for the underfit of y = sgn sum_b J_b rho_b s(z_b)

1. **Per-bond decomposition head** (prior ~40%). Let the CNN emit one bond-centred channel per bond direction
   (4 per site: x, y, two diagonals), sum them with a fixed layer, h_hat = sum_b t_hat_b, and train on
   sum_b (t_hat_b - t_b)^2 / Lambda^2 plus a margin loss on sgn h_hat, with t_b exact per row. Why: each row then
   carries ~4N real targets instead of one bit, the exact remedy of item 6 for a sign-of-composite target; the
   sum-then-sign structure is architectural, so the net no longer has to discover cancellation. Cost: one forward
   pass, same as now. Not in TRIED (P38 fitted the total, B23 used bond inputs). Main risk: t_b contains rho_b, and
   the amplitude side showed spins-only nets keep only 6% of a ratio-borne correction; the sum of per-bond errors
   must stay below |h| at the margin. Cheapest kill test (CPU/1 GPU-h): linear probe of the existing CNN's last
   features onto each t_b; if per-bond R^2 is high and the flip recall still 55%, the head fixes it; if R^2 is low,
   it will not.
2. **Gauge by a cheap proxy field, learn the residual** (prior ~30%). Compute h_tilde = sum_b J_b rho_tilde_b
   s_tilde(z_b) from a proxy with O(1) ratios (plain PP by the determinant lemma, or Jastrow; sign Marshall or
   sgn PP), feed its bond terms as input channels, and train on y * sgn(h_tilde). Why: by item 3 the gauge removes
   the high-degree Walsh weight the proxy already carries, and by item 6 the hard feature is handed in rather than
   found. Cost: plain-PP all-bond ratios O(N^2) after one inverse; at neighbours rank-2 updates (P10: cheap for plain
   PP only). Partly covered: C16 (psi_P features 0.02 for amplitude), B21 (67%), A36 (psi_P walls on the fly,
   fails). Main risk: the proxy is wrong exactly near the nodes where the flips live (C16 mechanism), so the residual
   stays hard there.
3. **Parity augmentation** (prior ~15%). Estimate |psi|^2-weighted Walsh coefficients of the flip indicator
   f = y*s on the 1.17M rows over a dictionary of sublattice parities (checkerboard, stripes, stretched checkerboard,
   their translates), add the top <= 8 as input channels (O(N) each). Why: items 2-3; item 2 measured 0.26 -> 0.72
   with 8 parities. Not in TRIED. Main risk: a flip indicator of a near-cancelling continuous field probably has no
   sparse Walsh spectrum, and item 2's gain is in generalisation, not fit. Deciding quantity (rung 2, existing rows, no
   training): the largest dictionary Walsh coefficient of f; near zero (say < 0.05) drops it.
All three are [conj]; none is shown to beat the confidence gate (~2.3x cost). Per project CLAUDE.md point 9,
each enters the account as an equation to derive, not a run.
