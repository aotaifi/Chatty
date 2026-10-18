# J1–J2, 20 sites: audited candidate prioritization and CNN learning (2026-10-18)

## Scope and caveats
- Exact tilted 20-site J1–J2 torus at J2/J1 = 0.5, Sz=0, 184,756 basis states, 4,654 symmetry orbits.
- Fixed guide signs after an energy-optimized Krylov update and three subsequent FN amplitude relaxations. All corrections shown here use this fixed guide amplitude.
- ED ground-state signs are scoring labels only, not training targets or selection information.
- **Oracle-cost limitation:** measured FN rates, guide amplitudes and orbit interactions were available for the *entire* 20-site basis. No claim of scalable discovery or sampling-cost advantage follows.
- CNN work teaches full *updated* Krylov signs relative to Marshall, not exact ED ground-state signs.

## Controlled sign-candidate rankings
- Eight amplitude-tempered candidate seeds per trial, including all one-hop neighbors. Compare on the identical pool.
- Whole symmetry orbits corrected together. Fixed-amplitude energy delta recomputed independently and agrees to 1e-8.

| 40 orbits (~1,600 states) | Energy reduction |
|---|---:|
| FN_suppression | 0.000647335 |
| Krylov_field | 0.000126219 |
| individual_flip_gain | 0.000878006 |
| greedy_energy | 0.000910056 |
- Greedy orbital sign corrections achieve 94.4% of the global energy-optimized Krylov gain (0.000963543) at this fixed amplitude.
- Initial ED wrong-sign weight 0.00109428; after 40 greedy orbits 0.00043716; full optimized Krylov update 0.00034757.
- Greedy energy is exact marginal fixed-amplitude energy minimization on the orbit graph. This is mathematically an Ising-coordinate-descent sign updater and provides a monotonicity guard **only at fixed amplitudes**.

## Corrected CNN pilot
- Same training candidate pool (2,882 symmetry orbits) from 600 amplitude-tempered seed configurations plus all one-hop neighbors; 20% symmetry-orbit holdout; no ED in labels or sampling.
- Periodic CNN, 16 channels, 3 blocks, binary cross entropy, 3,000 optimizer steps, five initializations for each sampling rule.
- Each guided sampler has 70% uniform baseline plus 30% score-weighted sampling, equal training steps.

| Training sampler | Mean fixed-amplitude E (5 seeds) | Mean held-out sign error | Mean ED wrong-sign probability |
|---|---:|---:|---:|
| baseline | -10.414802484 | 7.9240% | 0.0004664 |
| suppression | -10.414689265 | 8.6404% | 0.0004828 |
| energy_gain | -10.413678531 | 9.5848% | 0.0004994 |
- Keep-guide energy -10.414447972; full Krylov target energy -10.415411515; lower is better.
- **No stable NN advantage** from prioritization in this short study. Ordinary sampling achieves the best mean energy and classification accuracy, while the score-guided classifiers sometimes flip energetically costly signs.
- A physically useful score for ranking individual corrections does not automatically imply a useful distribution for BCE training.

## Next experiment
1. Generate candidates only using sampler-discoverable guides and 1-hop neighbors; never scan unseen global FN amplitudes.
2. Measure local energy-gain scores at those candidates using queried neighboring guide amplitudes.
3. Explore energy-gated, symmetry-orbit sign corrections rather than globally retraining a fresh classifier each round.
4. Train composite updated signs with an explicit *retain-old-sign* constraint / energy-sensitive loss. Assess performance against the keep-guide baseline, energy and wrong edge weight.
5. Move to 6x6 only after the sampler-accessible experiment works at 20 sites.

## Reproducibility files
- `j1j2_candidate_ranking20.py`, `j1j2_orbit_priority20.py`, `j1j2_orbit_greedy20.py`
- `j1j2_orbit_greedy20_results.json`, `j1j2_orbit_priority20_results.json`
- `train_sign20_corrected.py`, `sign20_corrected_cnn_results.json`, `sign20_corrected_cnn_3000steps_extra.log`
- Inputs: `sign20_training_data.npz`, `sign20_suppression.npz`, `audit_sign20_targets.npz`