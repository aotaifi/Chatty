# Callable neural-amplitude pilot — 2026-10-18

Status: **completed diagnostic; not a scalable algorithm**. This follow-up to `README.md` uses a genuinely trained, callable convolutional amplitude guide to select local sign corrections. The 20-site exact-FN solution supplies only the teacher amplitudes for training and post-hoc physical energy evaluation, not correction decisions.

## Reproducible implementation
Script: `j1j2_neural_guide20.py` (conversation artifact, pending source upload into this GitHub directory).
Input: `sign20_training_data.npz`, `audit_sign20_targets.npz` from the archived 20-site exact run. CPU PyTorch 2.10, NumPy/SciPy. Run `STEPS=900 python j1j2_neural_guide20.py`; output `j1j2_neural_guide20_results.json`.

## Protocol
1. Sample 1,700 configurations from exact 20-site FN a(x) (temper beta=.5), expand by one Hamiltonian hop, and fit a 24-channel, 3-convolution, translation-invariant, spin-flip-symmetrized CNN to log a(x); 900 AdamW steps. This training distribution and the 49,747 exact FN amplitude labels are ORACLES, not a scalable procedure.
2. Propose local spin exchanges with a Metropolis chain targeting the learned amplitude a_net(x). Candidate orbits consist of walker states + neighbors.
3. Rank up to 300 candidate symmetry orbits with observed signed bond products using the network's amplitudes and fixed guide signs; query neighbors of selected candidates through the callable network.
4. Greedily accept up to 40 symmetry-orbit sign flips with positive **predicted** marginal energy gain. Every decision uses only network amplitudes, not exact FN amplitudes or ED signs.
5. Score the same sign flips against true **fixed-amplitude** FN energy only at evaluation time.

## Measured results
Train: 44,600 labelled states, 49,747 distinct teacher amplitude queries (incl neighbors), 3,843 training symmetry orbits, 900 optimization steps (~47 sec CPU).
Held-out physical bond ratio diagnostic over 5,000 a^2-weighted Hamiltonian hops: RMS(log[a_net(x)/a_net(y)] - log[a_FN(x)/a_FN(y)]) = **0.14098**, MAE = **0.10455**, 90th-percentile abs error = **0.20712**.
Four independent walker seeds, # neural query sites 13,539–13,805, 300 candidate orbits per trial:
- seed 301: selected 17 orbits; predicted gain 0.000931185; actual gain **+0.000700233**
- seed 302: 7 orbits; predicted 0.000336880; actual **+0.000319254**
- seed 303: 7 orbits; predicted 0.000398443; actual **+0.000372115**
- seed 304: 6 orbits; predicted 0.000170579; actual **+0.000159784**
Mean actual fixed-a energy improvement **+0.000387846**, mean predicted **+0.000459272**, 0/4 energy-worsening trials, mean 13,696 distinct neural-amplitude evaluations.

## Interpretation, limitations, next experiment
- A genuine imperfect CNN amplitude evaluator can still produce energy-beneficial locally discovered corrections in this one 20-site pilot. This is a *feasibility signal*, not evidence of asymptotic scaling or superior neural sign training.
- 49,747 exact full-FN training labels (27% of this Hilbert space) are the dominant non-scalable ingredient. The original sign guide is likewise looked up from an exact full-space array, not a callable sign net.
- Walker proposal is a toy local Metropolis chain, not GFMC; no self-consistent FN update after the sign flips. Fixed-amplitude energy only. Selection uses CNN amplitudes and globally known fixed signs; source baseline comparison is across different seeds, not equal pooled costs.
- The next decisive work should replace exact FN teacher labels with walker-accessible samples, train both callable sign+amplitude guides, and compare against a sampling-matched baseline with acquisition cost including teacher labels, rejected moves, neural evaluations, and sign training.
- A four-seed pilot does not justify extrapolation to 6x6 or 8x8.

## Source artifact in this conversation
Full reproducible script and result JSON remain available as `/mnt/data/j1j2_neural_guide20.py` and `/mnt/data/j1j2_neural_guide20_results.json` until copied to GitHub. Do not claim these code artifacts are committed without verifying them.
