# EXPERIMENT 04 — Controlled 20-site sign-chamber transfer (2026-10-19)

**Status: completed first run, independently validated; limited transfer, not scalable success.** This report supersedes any implication from the 4x4 pilot that the bond-feature policy generalizes.

## Precise question

Train a small contextual-bandit reward classifier to predict whether a symmetry-orbit sign flip lowers **fixed-amplitude physical Hamiltonian energy** in the Marshall sign chamber; test **without retraining** in the improved post-Krylov chamber on exactly the same frozen 20-site FN amplitude function. Compare on disjoint withheld symmetry orbits with random, a cheap observed-signed-bond score, and a geometry+central-amplitude classifier. This isolates sign-chamber shift, **not** the full sign→FN→sign self-consistent iteration or lattice-size scaling.

Cluster: tilted 20-site periodic J1-J2 torus, J2/J1=0.5, S^z=0 sector D=184756; Hamiltonian-neighbor exchange hops (40 J1 + 40 J2 bonds). `s_M` reconstructed from the stored sublattice mask; `s_K` is the 60880-flip improved stored Krylov sign in `sign20_training_data.npz`, and both chambers use the **same** normalized stored positive FN amplitude vector. All guide/ground-state arrays remain full-space, **on-demand oracles**. ED ground-state labels are not used for policy training or candidate selection.

## Acquiring candidate pools and rewards

- Starting configurations drawn by Metropolis exchange moves targeting the frozen guide amplitude a(x), burn=2200, thin=13, target 140 roots; include *one-hop* Hamiltonian neighbors of roots; transform into complete translation + global spin-flip orbit representatives. About 2300–2500 candidate orbits per episode.
- Compute frontier features from **observed root-to-neighbor signed coupling** h_xy s_x s_y a_x a_y, plus spin geometry and representative a(x). Control network receives only geometry, representative amplitude and local sign (NO observed coupling features).
- Source labels are sign of *independently queried complete symmetry-orbit frozen-amplitude gain*. For one candidate orbit g,
  `G_g=4*|g|*sum_{y neighbors of rep(g), orbit(y)!=g} H_gy s_g s_y a_g a_y`.
  On multiple accepted orbits account for internal crossing-edge interactions; take at most 12 greedy energy-decreasing flips.
- Crucial distinction: the policy learns *which proposed single-orbit energy correction to prioritize*, **not** an energy-optimized global Krylov threshold or an exact ground-state sign function. This is a supervised contextual-bandit reward model, **not** GRPO policy-gradient training.
- Orbit partition: stable 20% group-level withheld test; 18 source training episodes with 110 random labeled orbits each, total 1980 unique training labeled orbits; 20 independent held-out walker seeds. Networks use matching 32x32 tanh MLP, AdamW, 1000 minibatch iterations.
- Per test: B=22 candidate orbit rewards checked per method on a *common* candidate pool. Each method starts from its own copy of the same post-frontier amplitude/sign cache; expensive extra amplitude queries are counted **separately**, not shared among competing methods.

## Observations (first network seed 19)

**Mean frozen-amplitude physical energy decrease** (higher better, 20 paired walker seeds, identical amplitude and candidate pools):

| Ranking | Marshall source | Post-Krylov target |
|---|---:|---:|
| learned observed-bond features | +0.000249136 | +0.00000103630 |
| learned geometry + central amplitude + local sign | +0.0000286178 | +0.00000177403 |
| heuristic observed signed bonds | **+0.000729884** | 0 |
| random | +0.000170509 | +0.000000319905 |

With B=22, beneficial *single-orbit* candidates found in Marshall source: learned 14.15, local-control 15.65, signed heuristic 2.10, random 3.45. **Many of the neural model's positives have vanishingly small energy value.** The heuristic finds far fewer but higher-value corrections. In the post-Krylov target, the learned model finds 2.15 profitable candidates vs 2.2 for the local-control network: the extra observed-bond features do not yield an advantage on the actual improvement criterion. Gains ~1e-6 in the post-Krylov target are microscopically small and are NOT grounds for claiming useful transfer.

Training rewards: 408/1980 positive source orbit labels (20.6%). Distinct guide-amplitude queries from source training summed across 18 separately cached episodes: about 220798, counting episode-overlapping states repeatedly. Held-out frontier+features use ~9000 distinct amplitude values per test, and B=22 verification actions require an additional ~580–690 distinct amplitude values per method. The walker exploration is a *toy* amplitude-Metropolis sampler, not FN/GFMC; cost counting is not a proven asymptotic estimate. All methods spent identical 22 expensive orbit-evaluations, but **not identical exact unique amplitude queries**; the heuristic actually used slightly fewer.

**Independent audit:** 32 stored selected sign sets over both chambers compared against complete Hamiltonian expectation computed via a separate full-basis bond sum. Max |predicted physical energy decrease - actual decrease| = 2.38e-14. 100 random complete orbits checked for exact sign and amplitude symmetry invariance. The score and its sign are numerically consistent.

## Interpretation

1. **Not a success for neural policy transfer.** At best, small positive target gains occur on the post-Krylov chamber, but the local geometry/amplitude control is at least as good. No demonstration of rare sign errors captured at useful energy value.
2. **The acquisition objective is wrong** if we only learn `I[G_g>0]`. It encourages many ultra-small gains. The original source chamber already shows strong physical-energy advantage from signed-bond heuristic.
3. The **notion of independent verifier still needs improvement**: all these sign rewards use the same full-space fixed FN amplitude reference. If the amplitude guide is learned from realistic 128 walkers, our previously observed local-ratio uncertainty can reverse a proposed gain.
4. Most concerning cost: source reward-label acquisition is nontrivial, candidate pool covers large part of the small 20-site orbit space, and no learned path-based exploration policy has yet been trained; here the policy **reorders a preconstructed frontier**.
5. A global Krylov sign threshold and a static whole-orbit reward classifier need not be size-consistent at N≥36. Retain the region-local defect design from `DESIGN_scalable_loop_2026-10-13.md`.

## Next decision (not automatically authorized)

Before training another policy, explicitly decide whether to target **expected verified energy gained per expensive query**, not binary probability of a positive correction. Proper evaluation requires new sign chambers with **reoptimized FN amplitudes**, a real local candidate proposer (not a 2300-orbit prescreen), and comparison against the signed-bond heuristic **including training cost**. If target rewards are essentially zero, do not claim machine-learning progress.

## Reproducibility assets
- Local run source: `avsl_transfer20.py` (unmodified Python 3, NumPy, SciPy and PyTorch) and `avsl_transfer20_validate.py`.
- Run data: `avsl_transfer20_results.json` (all seeds, per-arm gains and extra queries), `avsl_transfer20_validation.json` (independent audit).
- Input `sign20_training_data.npz` prepared with `results/scalable_sign_2026-10-18/prepare_sign20.py`; run uses `/mnt/data` explicitly, edit BASE path or stage copies in a scratch directory.
- Scripts/results have also been exported as conversation artifacts in the active research chat. Unless separately uploaded into this repo, this note alone is a **report**, not a runnable replacement for the source files.
