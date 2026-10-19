> **Superseded:** The complete 4-root-law, 4,096-patch, independently validated result (plus executable scripts and all 16 full raw runs) is now archived in `EXPERIMENT_05_COOPERATIVE20_RESULT_2026-10-19.md`. This earlier checkpoint contains obsolete missing-archive warnings and only three sampling laws.

# AVSL 20-site cooperative patch diagnostic — provisional checkpoint (2026-10-19)

**Status: exploratory numerical data reproduced in isolated scratch on MacBook-Air-2; no scalable FN/Krylov conclusion. Full raw JSON and exact executed script remain in the isolated machine's /tmp scratch directory, not yet mirrored to GitHub due an interrupted Remote Desktop connection. Do not treat this note as complete reproducibility.** This note records supported aggregate values to avoid losing the progress.

## Goal
Decide whether isolated profitable single-orbit labels are missing materially valuable **jointly profitable but individually unfavorable** sign updates in *bounded, sampled configuration-graph neighborhoods*.

## Input guide and physics
- Regenerated the 20-site tilted 4x5 `S^z=0` sector reference using copies of `results/scalable_sign_2026-10-18/fn_collapse_exact20_krylov_v2.py` and `prepare_sign20.py`, replacing only `/mnt/data` with an isolated scratch path.
- Hamiltonian `J2/J1=0.5`, sector dimension 184,756. Six Marshall FN updates, a Krylov sign update, three post-Krylov FN updates, exactly as in the existing benchmark generator. Regeneration reported final post-FN `E_FN=-10.411321676977218` and `SAVED 8320`; this is the existing guide preparation, not a new physical model.
- Data file on Mac: `/tmp/avsl05-20site-structural-data/sign20_training_data.npz`. Regeneration log: `/tmp/avsl05-20site-structural-data/regen.log`.
- Existing exact amplitudes/signs were **reference backends for evaluation**. The proposal sampler itself used amplitudes to set its root distribution. These are not deployable walker amplitudes.

## Gain-blind proposal protocol
For each seed 30260–30263, draw 256 roots under one of three root laws, independently of gain: (i) local-neighbor Metropolis with `pi ~ a` (`tempered_a`), (ii) Metropolis with `pi ~ sqrt(a)` (`a_half`), or (iii) uniform state roots (`uniform_states`, a post-hoc ceiling control, NOT scalable). At each root, grow an 8-orbit connected patch in the Hamiltonian-neighbor orbit-quotient graph using uniformly random frontier candidates; no energy labels enter patch generation. For each patch score all 2^8 subsets with the **exact frozen-amplitude** boundary-energy formula:
`G(S)=4/Z sum_{g in S,h notin S} W[g,h]`.
Compare unrestricted within-patch subset maximum, the best single, conservative greedy improvement restricted to initially profitable singles, and a random subset. An exact **8-orbit subset optimization is an oracle audit**, not a viable scalable policy. Prefixes of random patch-growth order at sizes 1,2,4,8 constitute gain-blind sign-update controls. Each seed measures 256 patches; 4 seeds share the same guide and correlated proposal space.

## Measured exploratory results
All gains below are **normalized total-system energies**, not energy/site. Each root-law row aggregates 4 seeds × 256 patches = **1024 correlated patches**.

| Root law | Patches with any positive subset | Positive pair with both singles nonpositive | Positive oracle cluster with all selected singles nonpositive | Mean best subset gain | Mean greedy singles gain | Mean oracle minus greedy |
|---|---:|---:|---:|---:|---:|---:|
| pi~a | 148 | 0 | 0 | 2.77780724e-6 | 2.76567469e-6 | +1.21325519e-8 |
| pi~sqrt(a) | 183 | 0 | 0 | 1.95949742e-6 | 1.95792055e-6 | +1.57686600e-9 |
| uniform roots | 263 | 0 | 0 | 1.59433848e-6 | 1.59415302e-6 | +1.85458111e-10 |

For `pi~a`, individual seed mean gains (oracle/greedy): 30260 `2.9874e-6 / 2.9568e-6`, 30261 `2.9064e-6 / 2.8885e-6`, 30262 `3.1037e-6 / 3.1037e-6`, 30263 `2.1138e-6 / 2.1137e-6`. Four of the 1024 patches showed an oracle improvement above 1e-12 relative to greedy singles.

### Instrumented *per-seed* query accounting
On `pi~a`: average post-scoring number of **unique exact amplitude oracle entries** was 27,843.75 per seed, and mean total runtime was 0.883 s on the Mac CPU (reference array already generated). First 16-anchor smoke test passed with 1,284 unique amplitude entries after the sampler, 5,778 after evaluation and 0.257 s elapsed; 8-orbit prefix and full-set checks were exercised. On the two additional root laws, mean exact-amplitude entries were 30,412.75 (`a_half`) and 28,858.75 (uniform), with mean wall times 0.888 and 1.324 seconds per seed respectively. Calls used cached NumPy exact-vector lookups, **not NQS inference or real FN walker costs**. Full counters (amp requests, sign requests, neighbor probes, orbit expansion and cache) and patch-wise records are in local `cooperative20_<law>_<seed>.json` scratch files.

## Interpretation — narrow
- **Evidence against an immediate *small-connected-orbit cooperative-label bottleneck***: none of the 3,072 tested gain-blind eight-orbit patches contained a positive two-orbit action with both singles nonpositive or a positive subset consisting exclusively of nonpositive singles. The unrestricted subset oracle barely beat initially-positive greedy singles on average.
- **Not evidence against all collective sign structures**: Hamiltonian-connected orbit patches of size 8 can miss dispersed or larger correlated changes; three samplers probe different regions, but none is the actual 128-walker FN mixed distribution. Raw patches overlap, so treating 3,072 as independent statistical replicates is invalid.
- **Not a size-consistent rule or a compression test**: no full callable `s'(x)=s(x)eta(x)` learned or cross-validated, no whole-lattice size consistency, no complete per-arm fully loaded compute cost and no post-FN reprojection. A powerful oracle may still exist outside proposal support, but may be irrelevant for deployable AVSL.
- **Software caveats:** full H independent validation of randomly chosen joint subsets, gauge/global-phase checks, reference-regeneration SHA and independent support/compression evaluation are not completed here. Do not merge as final numerical evidence until raw script/results and verification are archived.

## Location, source and next gate
- Executed source on Mac isolated GitHub scratch branch clone: `/tmp/avsl05-gpt6-isolated/projects/j1j2/experiments/active_verifier_guided_sign_learning/avsl_cooperative20.py`. File was generated and fixed/tested in scratch, **not yet committed**.
- Reference regenerated in `/tmp/avsl05-20site-structural-data`; summaries include `cooperative20_30260.json` etc. Existing tracked paper/code/guide was not modified.
- Next action: archive raw source/results if connectivity returns; independently cross-check several joint updates against direct full H and verify orbit symmetry; compute missed global opportunity under audit-only full-orbit enumeration to separate **proposal support** from ranking; only if material cooperative gains survive should local-rule compression receive training budget. Otherwise resume **Experiment 05 Stage-0 callable backend/latency preflight**, not another large 20-site supervised classifier campaign.
