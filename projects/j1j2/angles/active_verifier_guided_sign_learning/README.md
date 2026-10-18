# AVSL — Active Verifier-Guided Sign Learning
**Research recalibration, 2026-10-18.** Status: hypothesis and first controlled discovery probe; **not** a demonstrated scalable FN–Krylov algorithm.

## Why this angle exists
We want a scalable algorithm for the frustrated spin-1/2 square-lattice J1–J2 model near J2/J1=0.5 that *discovers* rare valuable sign defects, *decides* trustworthy corrections, and *generalizes* them to unseen configurations. The objective is not to maximize sign classification accuracy on exact 20-site data.

Four different blockers were conflated in our recent experiments:
1. **Support/discovery:** walkers predominantly visit well-described parts of configuration space; no reweighting of an already-zero-support distribution discovers unseen states.
2. **Local information:** a Krylov decision needs sign and amplitude information at Hamiltonian neighbors. Walker-trained amplitude networks have largest ratio errors near proposed rare corrections.
3. **Verification:** a flip that looks favorable under an approximate amplitude can be harmful under the actual FN amplitude; independently beneficial flips may interact unfavorably if accepted simultaneously.
4. **Generalization:** even if we verify discrete candidates, learning the composite next sign function with a neural network may corrupt large regions not in the verified set.

**Change of question:** Can a bounded, adaptive explorer *find* more useful sign-correction opportunities per expensive guide evaluation, and can independently verifiable local feedback make iterative sign learning reliable? Do not begin by training a larger amplitude CNN.

## Relevant established observations (see original dated archive)
- FN suppression enriches wrong-sign candidates but is not itself a flip label: correctly signed states also shrink. Strong-collapse suppression and the local Krylov imbalance become algebraically close.
- The corrected 20-site CNN sampling comparison (5 seeds, 3k steps) showed no gain from simple suppression-weighted training versus ordinary neighbor-augmented sampling.
- At a fixed screening cost, prioritization by observed signed bonds recovered approximately 3.2x the frozen-amplitude energy gain of random screening (12 toy sampling seeds), **with full-space exact FN amplitudes still callable on demand**.
- Exact-FN guide greedy selection of only 40 symmetry orbits captured 94.4% of the full optimized-Krylov fixed-a energy improvement; this relied on an amplitude oracle and did not demonstrate a scalable neural update.
- At 128 FN walkers, the walker-only NCE guide had RMS log-amplitude-ratio error 0.557 on bulk bonds versus 0.638 on selected correction bonds; only 1/4 sign-update trials improved exact fixed-FN energy. Larger walker populations helped the guide but touched a considerable fraction of the small 20-site state space.
- Independent log-amplitude noise can reverse apparently favorable energy decisions. Nearby FN iterations, with structured correlated errors, transferred decisions much better than independent perturbations.
- An older 6x6 raw K2 update sometimes *raised* variational energy by roughly 0.65–0.96 (with uncertainties); false-positive flips matter. Learn iterative **composite sign targets**, not independent stacked factor nets.
- **Size-consistency warning:** our 2026-10-13 design showed that a *single global* Krylov threshold can cease finding O(1)-scale local defects as N grows; use local features/defects for exploration and test intensive observables, rather than silently treating global threshold flips as the final scalable policy. See `projects/j1j2/DESIGN_scalable_loop_2026-10-13.md`.

## Mathematical separation
For a *fixed* positive amplitude a and sign field s, let psi(x)=s(x)a(x), and define the frozen-guide energy E[s,a] = (psi,H psi)/(psi,psi). For a simultaneous sign flip of a finite set F, the numerator decrease is
```
gain_a(F) = 4 sum_{x in F, y outside F} H_xy s(x)s(y) a(x)a(y).
```
Terms internal to F cancel. A positive *exactly evaluated frozen-a gain* certifies only frozen-a energy improvement under this **particular callable guide**. It does **not** certify a lower converged fixed-node energy, an improved ground-state sign, or neural-network generalization. A proposed sign flip and an accepted variational sign flip are different actions.

Local Krylov sign proposal from callable a and current signs s:
```
c_K(x) = sign( (I - t H) [s a](x) / [s a](x) )
       = sign(1 - t r(x)), where r(x)=(H [s a])(x)/[s a](x).
```
Use a threshold t selected on *held-out* validation energies. This label differs from the sign of the one-coordinate energy gradient, so do not silently substitute the latter and call it an optimized Krylov target.

## AVSL loop (proposal, not yet established)
A. **Explore on policy, not only from FN weight.** From a bounded mixture of walker samples, tempered/restarted seeds and local Hamiltonian-neighbor moves, build a frontier of *new* candidate states/orbits; use cheap encountered-bond disagreement/frustration as a prioritizer. Preserve a fixed fraction of unbiased/tempered exploration to maintain coverage.
B. **Acquire an expensive label selectively.** For candidates, query a callable sign/amplitude guide on the representative and neighbors and compute the Krylov proposal, edge evidence and estimated marginal energy gain. *All* guide queries, symmetry calls, proposal steps and sample generation count toward cost.
C. **Verify conservatively.** Compare candidates at frozen guide amplitude using the correct simultaneous/orbit crossing-edge energy formula. Require a large enough improvement under plausible local-ratio uncertainty; if no calibrated bound exists, record the proposal as **unverified**, not accepted. Independently evaluate sampled physical-H energy and/or FN energy when feasible.
D. **Replay without label fossilization.** Store (configuration/orbit, generation policy, current s, tentative Krylov label, amplitudes/ratio confidence, predicted and observed gain, version/iteration); re-label old examples after updating the guide. Keep false positives and representative old-guide examples.
E. **Learn composite signs, not ED signs.** Learn the *iteration-k proposed sign field* relative to Marshall or the current guide; regularize to the prior sign network away from verified support. Test held-out orbit energy/flip weight and self-consistent FN energy, rather than just BCE.
F. **Repeat.** Re-sample from updated signs; maintain independent replay and coverage diagnostics.

## Design adapted from LLM/RL work, with non-analogies
- **DAgger / on-policy aggregation:** policy action changes future state distribution. Here sign updates change FN walker support and hence future errors; train on induced distribution and old replay, not only the original FN samples.
- **OpenAI process supervision / active feedback:** locally verified steps can be better labels than judging a whole trajectory by a single final energy. But a Krylov label is *not* verified by its own approximate local field: a separate verifier is needed.
- **Anthropic Constitutional AI / RLAIF:** iterative critique and revision from imperfect self-generated feedback. Caution: accepting model-generated feedback without calibration reinforces mistakes; this is an analogy, **not** a claim Anthropic uses our method.
- **OpenAI weak-to-strong:** a weak local teacher might still be useful to a larger global learner; this is not guaranteed and the learner can reproduce teacher errors.
- **Prioritized replay / DAPO dynamic sampling:** spend updates on informative examples while retaining diversity. Do not naively drop *all* currently easy or zero-reward states; those define the sign function outside discovered defects.
- **Verifier gaming hazard:** a proxy energy or ratio estimate optimized with the same guide can spuriously confirm errors. Use genuinely independent held-out physical energy, FN energy, or ED on small systems as post-hoc audits.

## First experiment: exploration-only, before network training
See [EXPERIMENT_01.md](EXPERIMENT_01.md) and the self-contained `experiments/active_verifier_guided_sign_learning/discovery_probe.py`. Compare a passive frontier versus an active signed-bond frontier, under the same unique on-demand amplitude query limit, seed distribution and capped verification actions. First dry run: a 4x4 J1-J2 model with a **callable analytic** guide and symmetry orbits (no FN/ED oracle in explorer). Treat it as a software/algorithm diagnostic, **not** the first positive physics result. Next benchmark must use the actual 20-site walker sign guide and 128-walker FN population.

## Decision gates
1. **Discovery gain:** active finds more *independently evaluated* profitable correction orbits at equal total cost than random frontier and tempered/hop baselines, consistently across seeds, **without** querying globally known amplitudes.
2. **Verification fidelity:** measured precision of accepted flips under an independent FN/physical-H energy audit, including false-positive cost, and robust uncertainty thresholds.
3. **Sample efficiency:** improvement per unique queried x, bond evaluations, walker update and optimizer step as N grows; record fraction of accessible state space for small N only.
4. **Learning:** composite sign NN trained on acquired corrections outperforms no-update and ordinary one-hop training on both actual energy and generalization to untouched orbits.
5. **Size consistency:** study intensive energy gain per site and local edge disagreement, not only global sign overlap or raw global threshold flips.

## Existing source/archive
- `projects/j1j2/results/scalable_sign_2026-10-18/` (20-site code, mixed walkers, actual FN diagnostics and current limitations).
- `projects/j1j2/DESIGN_scalable_loop_2026-10-13.md` (important region-local, size-consistent proposals).
- `projects/j1j2/results/sign_learnability/README.md` (composite sign supervision and neighbor-augmentation benchmark).

The new direction was formulated with historical context retrieval; that retrieval is **not** an independent mathematical or numerical verification.
