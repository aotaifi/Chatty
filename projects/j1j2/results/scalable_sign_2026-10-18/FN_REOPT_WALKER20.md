# Independent FN reoptimization after walker-learned sign updates (2026-10-18)

**Scope:** same 20-site, J2/J1=0.5, frozen Krylov sign chamber, analytic Jastrow guide g0=exp(-0.30 H_xx), single NCE CNN trained on FN mixed walkers and *no exact FN-amplitude labels*. The exact 20-site FN solver appears **only in post-hoc validation**.

The consistency-matched FN operator F_{s,g0} has energy -9.524577481807 and eigensolver residual 7.16e-9. The walker-trained NN amplitude (g0*exp(logit)) has FN Rayleigh energy -9.4013837145, overlap 0.98753, RMS log-ratio error 0.17986 on 5606 physical hopped bonds. Bare Jastrow has Rayleigh -6.214669636 and RMS log-ratio 0.99380. About 8,246,720 Hamiltonian-neighbor probes were used to generate 395,500 mixed-walker training samples; no scalable cost advantage established.

### Fixed-amplitude sign audit (post-hoc same-guide FN amplitudes)
Maximum selected orbits k and true *fixed-amplitude* improvement -deltaE:
|k|improving seed trials/4|mean gain|
|---:|---:|---:|
|1|4/4|0.0023253|
|2|4/4|0.0024699|
|5|3/4|0.0023731|
|all positive estimated-gain flips|2/4|0.0019955|

**The crucial additional test:** After selecting those orbit flips **without the FN ground-state oracle**, recompute the **FN eigenvalue** for the new sign chamber with the *same analytic Jastrow g0 guide*:
| Flips | New FN solve improves / four candidate-pool trials | Mean FN energy gain |
|---|---:|---:|
| first 1 orbit | 3/4 | 0.0046681 |
| first 2 orbits | **4/4** | 0.0051590 |
| all selected orbits | **4/4** | **0.0211174** |

Baseline FN eig = -9.52457748. Exact post-flip FN solves were *not* available to the selection algorithm; they evaluate the proposal post hoc. All-selected runs improved FN more than a 2-orbit cap in this small test, **despite fixed-amplitude energy increasing in two cases**. Thus small k is a conservative fixed-a heuristic, not evidence of an optimal update size or safe global monotonicity.

Per-seed delta E_FN (negative means improvement):
- Seed 301: k1 +0.0000967, k2 -0.0005596, all -0.0253946.
- Seed 302: k1 -0.0001831, k2 -0.0002903, all -0.0087047.
- Seed 303: k1 -0.0092931, k2 -0.0098930, all -0.0104585.
- Seed 304: k1 -0.0092931, k2 -0.0098930, all -0.0399121.

**Caveats**: one NN seed, four *correlated* walker seeds; some first corrections repeat. Jastrow g0 is a weak starting guide compared with the pre-existing FN-amplitude guide (different operator). Global state-to-index lookup and full sign table used for 20-site acceleration. Exact FN after each flip used only as evaluation, not for scalable acceptance decisions. We do not know the result under genuinely approximate sign prediction or after a self-consistent FN/sign iteration.

### Reproduction code
- `j1j2_walker_only_amplitude20.py` (FN projector + density-ratio CNN + local candidate selection)
- `j1j2_walker_only_consistent_fn20.py` (same-guide FN amplitude eigenvalue + local-ratio/frozen-a scoring)
- `j1j2_walker_only_gate_depth20.py` (archive greedy ordered sign proposals and gate-depth curves)
- `j1j2_walker_only_fn_reopt20.py` (new-sign FN eigensolves; independent validation)
- Corresponding `*_results.json`, and existing code `j1j2_local_query_diagnostic.py`, `j1j2_budgeted_orbit20.py`, `j1j2_neural_guide20.py`.
Store archive together with prepped 20-site `sign20_training_data.npz` and `audit_sign20_targets.npz`. Set project paths if replaying outside /mnt/data.

### Next test for scaling
Repeat the NCE+FN loop with a real *callable* sign model, direct bitstring neighbor/diagonal calculations (no 2^N lookup), and several independent walker populations; compare guide g0 and learned g1 in **the same sign chamber**, then measure post-FN energy, local ratio uncertainty, and *total* projector-neighbor-model evaluations. Only advance to 6x6/8x8 if this is demonstrably tractable.
