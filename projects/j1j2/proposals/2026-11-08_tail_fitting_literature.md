# Tail-fitting literature, j1j2 (2026-11-08)
Reading only. Wiki first (2 source pages: reboredo2009self, caffarel2026self; no j1j2 folder, no SWO page), then arXiv MCP.
All fit-to-our-setting remarks are my reading, not published claims.

## Category verdicts
(1) Nodal/fixed-node optimisation: only 2 relevant hits (below); both treat node LOCATION, not amplitude ratios.
(2) Supervised/projective NQS fitting: 2 relevant; neither addresses small amplitudes or importance weights.
(3) Generic ML: relevant on importance weighting under misspecification; NOTHING found on bounds for
    log-ratio fits, tail oversampling or relative-error objectives that fits our setting.
(4) Hybrid/gated wavefunctions on |psi|: NOTHING relevant found (MoE hits are generic ML, no gating on amplitude).
## Ranked list
1. 2609.01301 Caffarel-Pinar-Scemama, SHDMC critical assessment. Claim: L2 refit of FN solution weights the
   bulk, drifts to a wrong node; a loss restricted to a window near the node (eps<~0.3) cures it (Eq. 25-30).
   Assumes 1D ring, noise-free, Fourier basis; our setting: same failure mode (bulk-weighted fit, node-sensitive
   energy) but 2D lattice, ratio-type error, no window notion. Read: wiki source page (full text ingested 2026-10-26).
2. 0808.3417 Reboredo-Hood-Kent, self-healing DMC. Claim: refit coefficients to the FN mixed density
   Psi_T*Psi_FN; iterate; noise acts as annealing. Assumes fixed-node density is the weight (so nodal region
   is weighted by construction) and statistics grow per block; ours uses |B|^(2beta) samples, no iteration
   closing the loop. Read: wiki source page.
3. 1811.12423 Kochkov-Clark, SWO. Claim: fit psi_W to target psi_T = (1-beta H)psi_W by L = sum (psi_T-psi_W)^2/p(c),
   p ~ |psi_W|^2 (Eq. 2), equivalent to infidelity to leading order. Assumes errors that matter are
   weighted by |psi|^2; ours: energy depends on ratios at |psi|^2 ~ 1e-9..1e-15, which this loss gives ~0
   weight. This is the closest published template and has no tail handling. Read: Sec. III (Algorithm 1, Eq. 2).
4. 2410.10720 Gravina-Savona-Vicentini, neural p-tVMC. Claim: Sec 3.2: with sampling from the ansatz, the
   estimator with best gradient SNR (Rao-Blackwellised, "cmc") beats the control-variate one; infidelity
   gradient SNR is the diagnostic; NGD with damping. Assumes sampling from pi_psi, unitary targets; ours:
   hybrid/tempered sampling not covered, but the gradient-SNR diagnostic transfers. Read: Sec 3.2-3.3.
5. 2303.04020 Gogolashvili et al., when is importance weighting needed. Claim: if the model is misspecified
   (nonparametric or parametric), IW correction is needed for the best approximation on the TEST
   distribution; if well specified it is not. Assumes kernel ridge regression, bounded weights; ours: B is
   misspecified in the tail (a determinant cannot represent T there), so reweighting toward the
   blocked-neighbour region is the right direction; weights spanning 1e-15 break the boundedness. Read: abstract only.
6. 1910.06324 Lam-Li-Prusty, robust importance weighting for covariate shift. Claim: weights estimated from
   data are noisy; robust (distributionally) weighting beats plug-in. Assumes estimated density ratio; ours:
   the ratio is known analytically (|B|^2 vs target band), so the estimation issue is absent, only variance
   remains. Read: abstract only.
7. 2311.14820 Szoldra, error bounds on NQS overlap MC estimator. Claim: overlap estimator contains amplitude
   ratios psi_a/psi_b that can be unbounded; gives bounds in fidelity and sample count. Assumes sampling from
   one state, global fidelity; ours is a log-ratio over specific bonds, not an overlap, so the bound
   does not apply, but it states exactly why ratio estimators need samples where the denominator is small.
   Read: abstract only.
8. 2006.15608 Qin (PEPS+GFMC) and 2406.12207 Lin et al. (GFMC+PEPS, J1-J2). Claim: a good guide with the
   correct nodal structure in the bulk gives fixed-node energy; no refit of the guide near nodes. Assumes the guide
   is fixed (not learned); ours fits the guide, so the tail of the guide matters. Read: abstracts only
   (2406.12207 not opened beyond the search snippet).
9. 2411.02257 Freitas-Abreu-Vitiello, NN nodal-structure optimisation. Claim: optimise node by minimising
   the DMC/VMC energy of an NN guide for fermions. Assumes continuum fermions, energy loss; ours is a
   supervised fit to a given T (no energy loss). Read: abstract snippet only; applicability to lattice unverified.
10. 2201.09128 Havlicek, amplitude ratios and NQS. Claim: NQS give access to amplitude ratios, tied to
    distribution testing; sample complexity of ratio access depends on tails. Assumes learning-theory
    setting; no practical remedy. Read: abstract snippet only (low confidence on relevance).
11. 2010.05316 Schatzle-Hermann-Noe, convergence to FN limit in deep VMC. Claim: expressive NN guides reach the
    fixed-node limit when the nodal structure is learnable. Assumes energy-trained continuum fermions;
    ours is a supervised fit where energy-trained pressure on the nodes is missing. Read: abstract only.
12. 2309.01120 Lichtenberg et al., double clipping. Claim: weight truncation cuts variance at the cost of bias;
    one-sided clipping bias can be reduced by a second clip. Assumes bounded rewards in off-policy evaluation;
    ours would clip band weights (1e-15 decade), bias is then a controlled loss of tail accuracy. Read: abstract only.

Not ranked: 0906.4359, 1309.4665 (SHDMC extensions, not read); MoE papers (2305.03288, 2405.13997): none on |psi|-gating.
## Not covered by the literature
- No paper found fitting a wavefunction under an objective dominated by ratios across a sign-violating bond
  at |psi|^2 ~ 1e-9..1e-15. SWO/p-tVMC losses all sit on |psi|^2-weighted L2 or infidelity.
- No gated/hybrid (bulk model + tail model) NQS found. The idea stays our own design; not literature-backed.
- No bound on log-ratio (e_x - e_y) error from per-sample or L2 fit error found.
## Takeaways
1. 2609.01301: replace bulk-weighted L2 by a loss localised on the region that sets the node (our blocked
   bonds); their small-window loss is the only published cure of this failure mode (toy model only).
2. 1811.12423 + 2303.04020: SWO's p ~ |psi_W|^2 weighting gives tail points ~0 weight; B is misspecified in the
   tail, so reweighting is justified; test whether our band-uniform weights are what the theory asks for.
3. 2410.10720: use gradient SNR on the tail samples as the diagnostic for whether the 1% fails by noise or by capacity.

## Lead reading (2026-11-08)
- Finding 1 (Caffarel, node-window loss) is already in our design: pool_fit's band-uniform decades + blocked-bond terms (PLAN.md l.39-40: Test 1 used a node-weighted variant). Not a new lever.
- Finding 3 (Gravina 2410.10720 gradient SNR on the deep/blocked samples) is a cheap addition to the R1 (a)/(b) separator: low SNR -> sampling (oversample), high SNR but no progress -> class (architecture angle).
- No published method targets ratio fits at |psi|^2 1e-9..1e-15; no gated bulk/tail wavefunction found.
