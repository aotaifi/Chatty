# Literature check for the gated sign step (2026-11-27, subagent report condensed by FN-lead; NOT re-read by me)
Tags: [T] read in text by the subagent, [A] abstract only, [M] memory.

Verdict: nobody found doing "cheap gate + exact sign near the node" in QMC or NQS. Our failure of the learned sign step is published; a finite-sample guarantee for a gate exists but needs independent calibration rows and a bounded loss.

A. Sign learning in NQS
- Westerhout et al. arXiv:1907.08186 [T]: signs of frustrated ground states fit on the full basis (overlap > 0.94) but do not generalise from 1% of it near J2/J1 = 0.5; sharp threshold in training-set size; their metric is psi^2-weighted.
- Wang, He, Lu arXiv:2502.01264 [T]: closest to our learned route. Sign net + amplitude net trained on each Lanczos state, J1-J2. Sign accuracy 95.8% (6x6 p=1), 86.0% (6x6 p=2), 91.2% (10x10 p=1, 1.2M rows). Their diagnosis: near an eigenstate the step's target is small differences plus noise, and the net abandons those signs.
- Ou, Huang, Ozolins arXiv:2510.02051 [T]: 6x6 at 0.5: 49.76% of symmetrised configurations non-Marshall, psi^2-weighted Marshall overlap 0.9608: flips sit at small |psi|.
- Szabo-Castelnovo arXiv:2002.04613 [A], Bukov-Schmitt-Dupont arXiv:2011.11214 [A]: optimisation failures, not generalisation.
- Westerhout-Katsnelson-Bagrov arXiv:2207.10675 [A]: signs computed from amplitudes via an auxiliary Ising model; nearest "compute, do not learn".
- Kairon & Clark arXiv:2609.16308 [A]: lattice fixed-node iteration; ground-state sign chamber is a stable fixed point, amplitude dependence disappears under self-consistent iteration. TO READ.
B. Published energies per site at J2/J1 = 0.5 [T unless noted]
- 10x10: Hu et al. arXiv:1304.2630 p=0 -0.49521(1), p=1 -0.49718(1), p=2 -0.497549(2); Nomura-Imada -0.497629(1) (via Rende's table); Rende ViT arXiv:2310.05715 -0.497634(1); Chen-Heyl arXiv:2302.01941 -0.4976921(4).
- 8x8: Hu et al. p=0 -0.49656(1), p=1 -0.49855(1), p=2 -0.49886(1). Hu et al. contains NO fixed-node energies. No NQS + fixed-node J1-J2 paper found; no equal-cost comparison found.
C. Guarantees for a gate
- Learn then Test arXiv:2110.01052 [T]: P(risk <= alpha) >= 1-delta; needs i.i.d. calibration rows from the deployment law, loss in [0,1].
- Conformal risk control arXiv:2208.02814 [T]: expectation bound only; loss monotone in the threshold and bounded.
- For us [subagent's derivation, unvalidated]: miss cost is monotone in the gate threshold; rescale by a hard bound; rows from independent chains spaced >> tau, drawn from the walker law; certifying a miss fraction 1e-4 at delta 0.05 needs ~3e4 independent rows.
D. LLM long tail: not achieved and not needed for their metric (LeBrun et al. arXiv:2203.12788 [A]: rare sequences systematically underestimated).

Compare with FN-lead's blind guess (written before the search): agrees on all three points (tail is forgiven by their metric; the oracle is our advantage; guarantees exist). New: Wang et al.'s warning bears on CHAINING (later sign steps have noisier targets), and Kairon-Clark may bear on why the adapted amplitude matters.
