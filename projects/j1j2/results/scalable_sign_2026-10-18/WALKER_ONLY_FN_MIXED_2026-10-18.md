# Walker-only FN amplitude learning and local sign repair — 2026-10-18

## Objective and scope
Replace the previous **~50,000 exact FN-amplitude training labels** with accessible fixed-node Monte Carlo samples. Keep the ground-state ED/FN solution for **independent final evaluation only**. This tests whether a learned callable amplitude guide gives reliable ratios on Hamiltonian bonds encountered by walkers, as a step toward 6x6 and 8x8. It does **not** yet prove scaling.

## Algorithm (no oracle FN amplitudes in training)
J1-J2 on the tilted 20-site torus, J2/J1=0.5, S^z=0, dimension 184,756. Signs are **frozen from the earlier optimized Krylov chamber** (stored lookup table, not learned by this run). Start with a purely analytic callable amplitude guide g0(x)=exp(-0.30 Hxx), independent of the exact FN solution. It defines F_{g0}[s], whose off-diagonals retain negative signed Hamiltonian bonds and whose diagonal compensates positive bonds with g0(y)/g0(x).
- Sample the positive importance-transformed projector B_xy=(g0(y)/g0(x)) [delta_xy - dt (F_g0 - E_ref)_xy], dt=0.01 and E_ref=-10, using branching/resampling walkers. Its asymptotic distribution is f(x) proportional to g0(x) phi_FN(x).
- Independently Metropolis sample the known **g0(x)^2** reference q(x).
- Train the periodic, translation-invariant, spin-inversion-symmetric CNN as a logistic noise-contrastive density-ratio classifier: log f/q = log(phi_FN/g0)+const. The callable amplitude is a_theta(x)=g0(x) exp(h_theta(x)). No exact FN labels or ED labels are supplied to the classifier.
- Evaluate at 20 sites with a separate sparse exact diagonalization of **the same F_g0**. Exact FN eigenvectors must never enter walker generation, NCE training, candidate selection or sign gating. Treat this 20-site solve strictly as post-hoc audit.

## Verified results
3,500 walkers, 450 burn + 450 projector steps, samples every fourth step: 395,500 raw positive samples (65,532 unique) and 395,500 reference samples; 103,084 unique projector rows evaluated, 8,246,720 local bond probes, ~55.8% of the Hilbert space. 1,200 CNN optimizer steps.
- Exact FN eigenvalue -9.524577481807206; residual 8.68e-9 for the **analytic g0-defined FN operator** (NOT the earlier oracle experiment).
- On 4,507 physical probability-weighted valid bonds, RMS local log-ratio error: initial Jastrow **0.9950**; walker-trained CNN **0.1826** (5.45x reduction). FN amplitude overlap improved **0.6085 -> 0.9875**.
- IMPORTANT: on 2,344 bonds around the actual **selected correction orbits**, CNN RMS log-ratio error is **0.3309**, not 0.1826. This is the relevant off-support bottleneck.
- Four walker local-screening trials with 300 evaluated orbit candidates each, at most 40 greedy accepted flips. Scoring the changes against the **exactly evaluated FN amplitude**, CNN-picked corrections improved energy in **2/4** trials and worsened in **2/4**; individual true energy gains: -0.001358878, -0.000012800, +0.005145458, +0.004208244. Mean +0.001996 is dominated by positive outliers and **is not a safe acceptance rule**. Baseline Jastrow-picked flips also improved in 2/4, mean +0.000138. These are fixed-amplitude energy differences, not reoptimized FN energies.
- Average ED wrong-sign probability improvement was negative for **both** schemes, so optimized energy and ground-truth sign accuracy disagree.
- 700-walker ablation (otherwise same settings): 79,100 raw positive samples, 42,672 unique projector rows, 3,413,760 local bond probes; physical RMS log ratio **0.2672**, FN amplitude overlap **0.9682**. Reduced cost still helps vs g0, but accuracy degrades.
- Loss-focused NCE with 4x positive local frustration weighting applied to **both** positive/negative classes (same target density ratio) **failed to help**: ordinary versus focused physical RMS 0.18256 vs 0.18472; selected-correction-bond RMS 0.33092 vs 0.33259. Do not promote this heuristic.
- An additional 128-walker ablation is underway; append only after results are verified.

## Important limitations
1. The 20-site basis-to-index and frozen sign table are still globally stored. They are **not exact FN amplitude labels**, but the sign table must ultimately become a callable learned sign model.
2. The projector is a local FN mixed-walker population-control prototype; population bias, thermalization and correlation times require checks. Full-space row memory accelerates this small benchmark but is NOT scalable.
3. The 3,500-walker run visited more than half of this tiny Hilbert space. Complexity per attempted move is polynomial in N (bond locality), but **network generalization and query support at N=36/64 are unproven**.
4. Learned amplitude quality measured on physical high-weight bonds is much better than on rare candidate bonds. Fixed-amplitude gains predicted by the approximate guide need not hold at the true FN amplitude or after reoptimization.
5. Prior exact-label pilot and this analytic-guide FN problem use **different FN potentials**; their energies are not directly comparable.

## Files and re-run
- `j1j2_walker_only_amplitude20.py`: positive FN projector, independent g0^2 Metropolis samples, walker-only density ratio CNN, local candidate screening, and query counts.
- `j1j2_walker_only_eval20.py`: exact sparse-FN benchmark *only after training*, ratio accuracy and actual sign-update energy at frozen FN amplitude.
- `j1j2_walker_focus20.py` and `j1j2_walker_focus20_eval.py`: failed sign-frustration-weighted NCE ablation.
- `j1j2_walker_only_amplitude20_results.json`, `j1j2_walker_only_amplitude20_eval.json`, `j1j2_walker_only_amplitude20_lowbudget_results.json`, `j1j2_walker_focus20_eval.json`: numerical output.
- Dependencies: `j1j2_neural_guide20.py` (CNN), `j1j2_local_query_diagnostic.py` (torus geometry), `j1j2_budgeted_orbit20.py` (orbit gate), and `sign20_training_data.npz` + `audit_sign20_targets.npz` generated by `prepare_sign20.py` and `audit_sign20_targets.py`. The latter benchmark files contain ED/old-FN arrays but are **not consulted for training**.
- Original scripts assume /mnt/data. `reproduce.py` stages patched copies in a scratch directory; run data stage first, then walker stage.

## Next research discriminator
Run genuinely small fixed walker populations (e.g. 128), vary N, measure **candidate-edge** ratio errors, FN walker/sample and unique-neighbor query cost, actual fixed-phi and self-consistent FN energy changes. Replace the fixed sign-table lookup by callable signs, then test uncertainty-aware sign gating or robust variational backtracking. Avoid larger CNNs or global FN oracle labels just to improve this 20-site score.
