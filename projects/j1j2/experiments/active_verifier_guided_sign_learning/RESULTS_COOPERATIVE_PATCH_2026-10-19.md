# AVSL cooperative-patch diagnostic — 2026-10-19

Status: **completed bounded 20-site numerical scout**, pending additional independent full-H and compression-rule audits. No NN training, GFMC walkers, FN reoptimisation or 6×6/8×8 run.

## Research question
Is an individual-orbit positive fixed-amplitude gain label missing many **profitable joint flips with all individually adverse constituents** in *gain-blind, walker-neighbour-accessible* small configuration-graph patches?

## Reference data / reproduction
The previously archived exact 20-site *post-Krylov/FN* reference was regenerated afresh in an isolated Mac scratch directory from:
- `projects/j1j2/results/scalable_sign_2026-10-18/fn_collapse_exact20_krylov_v2.py`
- `projects/j1j2/results/scalable_sign_2026-10-18/prepare_sign20.py`

Only their literal /mnt/data paths were redirected to `/tmp/avsl05-20site-structural-data/` on **scratch copies**. Preparation took about 6.4s; the log reported post2 FN energy `-10.411321676977218`, and wrote `sign20_training_data.npz` (20 sites, Sz=0 dimension 184,756). **The non-committed NPZ is regenerated from the tracked scripts.**

Exploration (on 4 independent random seeds per root law; same frozen physics):
- `n_anchors = 256` gain-blind sampled spin configurations per seed;
- grow **one connected eight-orbit candidate patch** per root, by uniformly choosing among Hamiltonian-neighbour orbit opportunities. The candidate generator does **not** read fixed-amplitude sign-flip gains or ED ground-state signs;
- score all eight individual orbit boundaries with the exact *stored* post-Krylov FN amplitude/sign vector **as an evaluation oracle**;
- test all 255 nonempty subsets of that **fixed discovered patch**, and compare best hindsight subset with greedy selection of individually profitable orbits, singletons, keep, and a gain-blind random subset. Count pairs where both individual gains are <=0 yet the joint gain >0, and oracle-best subsets composed entirely of individually adverse orbits.
- `root_law=tempered_a`: exactly the legacy toy Metropolis acceptance min[1,a(y)/a(x)], 2,200 burn steps, 13 thin, started from uniformly random legal basis configuration.
- `root_law=a_half`: same proposal, acceptance min[1,(a(y)/a(x))^0.5] (flatter diagnostic).
- `root_law=uniform_states`: random 20-site basis states drawn from the *complete basis*, **evaluation-only support ceiling, not deployable walker sampler**.
- seeds `30260,30261,30262,30263`; 256 patches per seed per law, i.e. **1,024 patches per law, 3,072 total**.

## Measured results
Values below are normalized **frozen-amplitude total variational energy gains**, positive meaning improvement, using the SAME amplitudes for all candidate sign changes. Source: execution on Mac `/tmp/avsl05-gpt6-isolated/projects/j1j2/experiments/active_verifier_guided_sign_learning/avsl_cooperative20.py`; raw JSONs named `/tmp/avsl05-20site-structural-data/cooperative20_<law>_<seed>.json` (tempered_a filenames omit the law). **Scratch paths are local artifacts, not GitHub links.**

| Root distribution | Profitable best-subset patches / 1024 | Any profitable pair with both individual gains nonpositive | Oracle-positive subset whose every single is adverse | Mean best subset gain | Mean greedy-singles gain | Extra oracle-over-greedy |
|---|---:|---:|---:|---:|---:|---:|
| a^1 Metropolis | 148 | 0 | 0 | 2.777807238e-6 | 2.765674686e-6 | 1.213255191e-8 |
| a^0.5 Metropolis | 183 | 0 | 0 | 1.959497420e-6 | 1.957920554e-6 | 1.576865999e-9 |
| Uniform full-basis oracle | 263 | 0 | 0 | 1.594338475e-6 | 1.594153017e-6 | 1.854581109e-10 |

- Only **4/1,024**, **4/1,024**, **3/1,024** patches, respectively, showed a best subset exceeding the greedy-singles result by >1e-12.
- Unique exact amplitude oracle values cached in a representative run averaged ~27,844 (a^1), ~30,413 (a^0.5), ~28,859 (uniform). Mean wall per experiment was ~0.88s, ~0.89s and ~1.32s respectively on the Mac, **not counting guide construction, FN solves or model training**. Cache misses/query counts are machine- and backend-specific; they do **not** establish 6×6 scaling.
- The fourth seeds in each law use the same physical Hamiltonian and FN sign chamber. This is **not twelve independent physical instances**.
- Score verification relies on the crossing-edge formula, previously audited against full H at ~1e-14 in the transfer benchmark. **This new implementation still needs its own independent full-H regression test and local-rule compression evaluation** before making a stronger statement.

## Interpretation / decision
**TEST gate yielded a bounded negative result:** no observed *all-individually-adverse but jointly profitable* pair/subset in 3,072 connected patches of <=8 symmetry orbits across three root laws. Even a deliberately broadened full-basis **oracle** root control did not reveal this hypothesized common local structure.

**REJECT as current priority:** a new collective-orbit-QUBO training pipeline on the premise that short, entirely adverse orbit clusters dominate post-Krylov missed gains.

**ACCEPT as cheap baselines:** keep, single-orbit gain ranking, and the small-group joint score (correctly accounts for interaction). Keep a size-consistent local-rule C_w arm in Experiment 05, which does evaluate the **full joint energy of a learned rule**.

**OPEN:** larger or nonconnected cooperative moves; rare walker-inaccessible tails; whether local *symmetry-covariant sign rules* can represent distributed multi-orbit corrections; real callable guide ratio accuracy and true post-FN changes. None is resolved by this bounded 20-site test.

## Algebra
For disjoint orbit sets A,B and `Z=sum a²`, `G(A union B)=G(A)+G(B)-8W_AB/Z`, where W_AB counts H_xy a_xa_y s_xs_y **once** from A to B. Negative W_AB reflects satisfied internal bonds, so a large 'synergy' term alone is **not** proof of a useful correction. Measure **actual joint gain against keep** and, for all-bad pairs, whether positive gain survives under the full external boundary objective. `G(all)=0`: global sign reversal is irrelevant.

## Immediate reproducibility blocker
The local scratch run completed, but the Mac Remote Desktop Commander connection became unresponsive while preparing the archive. The core uncommitted scout script and full per-patch raw JSON currently exist at the scratch paths above; until fetched/uploaded, **this commit is the exact protocol and consolidated numerical readout, not a complete independently rerunnable archive**. Never claim that the raw files were committed until GitHub retrieval verifies them.

Next: recover script and raw results, unit-test orbit-coupling/independent H identity, then decide whether Experiment 05's short walker preflight should proceed.
