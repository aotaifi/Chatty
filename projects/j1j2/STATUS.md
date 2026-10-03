# J1-J2 research status


### 2026-10-03 exact-size K1 benchmark for paper
A second exactly tractable ground-state benchmark was completed on the 20-site bipartite skew torus T=(4,0),(1,5), Sz=0, D=184,756 at J2/J1=0.5. Using the exact positive ground-state amplitude, Marshall signs, and the same fixed-amplitude energy-optimized one-step Krylov threshold used in the 4x4 paper benchmark, wrong-sign probability falls 0.0136471 -> 0.000275626 (49.5x) and relative energy error epsilon_rel=(E-E0)/|E0| falls 0.0202556 -> 0.000840386 (24.1x). This is the preferred second exact-size K1 result; 6x6 ED is infeasible (Sz=0 dimension ~9.1e9), and 2x2 is too trivial. Source: `krylov_sign_structure/results/groundstate_k1_20site_exact_J2p5.json`.

## Goal
Find an exact or genuinely scalable way around the sign problem for the square-lattice spin-1/2 SU(2) J1-J2 Heisenberg model at J2/J1 = 1/2.

## Current constructive route
Iterate:
1. choose a compact guide/sign model s_theta(x);
2. solve amplitudes with a sign-free fixed-node projector for those fixed signs;
3. use sampled amplitudes/energies to improve the sign model;
4. rerun fixed node with the updated signs.

## Current bottleneck
The one-step Krylov sign family is highly accurate but not exact. The dominant unresolved scalability problem is the fixed-node -> amplitude handoff: raw FN walkers encode a mixed distribution, while the Krylov rule needs callable neighbor amplitude ratios. Naive walker histograms already scale badly on 4x4. The current leading handoff is a compact density-ratio correction: if FN walkers sample f_k(x) proportional to a_k(x) a_FN,k(x) and the reference sampler gives q_k(x) proportional to a_k(x)^2, then f_k/q_k is proportional to a_FN,k/a_k. Fit only g_k(x)=log[f_k/q_k], so a_{k+1}(y)/a_{k+1}(x)=[a_k(y)/a_k(x)] exp[g_k(y)-g_k(x)]. First falsify this handoff on controlled 4x4, then close the real loop at 6x6.

## Fixed-node scaling test
A best-case exact 4x4 experiment already shows severe raw-population cost: ~1e6 ideal independent walkers give reconstructed-sign overlap ~0.99747 and 3e6 give ~0.99876, versus the exact-amplitude one-step ceiling ~0.99926. Raw walker histograms are therefore not an acceptable scalable amplitude representation; a compact amplitude model is required before further large-size FN looping.

## Immediate next step
A simpler boundary-free global construction has superseded the high-capacity correction-model attempt. Define the Marshall local energy r_M(x)=(H psi_M)(x)/psi_M(x) with psi_M=a_ViT s_M, and reconstruct signs by the one-step Krylov rule s_t(x)=s_M(x) sign[t-r_M(x)]. The threshold t is inferred without hidden signs by two-means or Otsu clustering of the sampled r_M distribution. This gives validation weighted hidden-sign overlap ~0.9986 on 6x6 and ~0.9990-0.99923 on 8x8; hidden signs are diagnostics only. A clean 6x6 fixed-amplitude energy comparison with 64 independent chains gives E_K-E_ViT=+0.00420+/-0.01035 total energy (0.41 sigma, statistically indistinguishable; no evidence that Krylov beats ViT) and E_K-E_M=-0.17247+/-0.04489 (3.84 sigma improvement over Marshall). Thus the label-free rule reproduces essentially the ViT fixed-amplitude energy as well as its high-weight sign structure. The 8x8 physical |a|^2 validation was extended to 4096 samples with a threshold frozen from alpha=1.2: one mismatch was found, giving raw O=0.9995117; MCMC autocorrelation implies only ~1100-1300 effective samples, so the rare-error scaling is still unresolved. Exact 4x4 standard lattice-FN iteration shows the loop improves sharply but does not lock to the exact node; even energy-optimizing t leaves a tiny recurrent sign sector. The decisive current obstruction is the FN-walker -> callable-amplitude handoff: raw populations require millions of ideal walkers already on 4x4. Next route, if pursued, must insert and falsify a compact amplitude model between FN walkers and the Krylov update. The local free-boundary cluster and independent one-shell cavity routes are not the leading route.

## Stop criteria
SUCCESS: evidence and mechanism for polynomially scaling exact/scalable reconstruction.
FAILURE: a decisive exponential bottleneck in sampling, projection, or sign reconstruction that survives guide/model improvements.


## Latest decisive step — 2026-09-28
The 4x4 density-ratio amplitude handoff has passed: a four-bin scalar g(r) correction with eta=0.5 keeps the closed FN -> amplitude -> Krylov loop stable using O(10^4) samples per reference/mixed distribution, versus O(10^6) raw-histogram walkers. The remaining test moved to 6x6.

A direct-ViT 6x6 Krylov-FN pilot has now completed using the label-free frozen threshold t=-14.9857997791, M=32 walkers, beta_target=0.9, burn_beta=0.3, tau_max=0.025. It gives Emean=-18.12798165, tail=-18.12338784, naive snapshot SE=0.00595. This is encouraging relative to the earlier unmatched Marshall pilot near -18.00, but not yet a controlled comparison. Exact on-the-fly Krylov signs required about 1.17e6 ViT amplitude evaluations because the FN move needs second-shell r_M values; the method remains polynomial but with a large direct-ViT prefactor.

A matched Marshall-FN control with the same M, beta window, tau_max, initial pool and seed convention is currently running under Chatty job 20260928-084549-47862 (target PID 47864 at launch), job dir /Users/aliotaifi/Chatty/jobs/20260928-084549-47862. Expected result: /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_marshall_matched32.npz. Per AGENTS.md, do not poll it from ChatGPT; completion is emailed. When the user returns, read the job files and compare directly against gfmc_6x6_krylov_vit_pilot.npz. If Krylov-FN is materially lower, proceed to the 6x6 four-bin g(r) amplitude handoff; otherwise reassess.


### Matched Marshall control returned; replicated control launched
The first matched Marshall-FN control (same M=32, beta_target=0.9, burn_beta=0.3, tau_max=0.025, same initial-pool construction) completed successfully:
- Emean = -18.10006382
- naive within-run SE = 0.07685
- late tail4 mean = -18.03133 +/- 0.05764
- 8 stored projected snapshots.

Compared with the completed direct-ViT Krylov-FN pilot:
- Krylov Emean = -18.12798165 +/- 0.00595 naive
- difference in central values = -0.02792 in favor of Krylov.
However the Marshall M=32 control is much too noisy for a decisive statement; one projected snapshot was -18.5893, dominating the variance.

Therefore do NOT yet proceed to the 6x6 g(r) handoff. An 8-replica matched Marshall control at the same M=32 and projection parameters has been launched under Chatty job 20260928-084938-48256 (target PID 48258), job dir /Users/aliotaifi/Chatty/jobs/20260928-084938-48256. It will write per-seed gfmc_6x6_marshall_rep_seed*.npz and aggregate /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_marshall_replicas8.npz. Per AGENTS.md, do not poll this job from ChatGPT. After the completion email/user return, compare the between-replica mean/SE with the Krylov pilot. Proceed to the 6x6 four-bin g(r) handoff only if the energy separation is statistically meaningful.


### 6x6 matched energy verdict and g(r) collection — 2026-09-28
The replicated matched Marshall-FN control completed (8 independent M=32 runs, beta_target=0.9, burn_beta=0.3, tau_max=0.025):
- mean of run means = -17.97267437
- between-replica SE = 0.02954654
- mean tail4 = -18.00366842
- between-replica SE on tail4 = 0.02520821.

The completed direct-ViT Krylov-FN pilot under the same M/beta/tau scale gives Emean=-18.12798165 (within-run naive SE 0.00595). The central separation is about -0.15531 total energy in favor of Krylov, consistent with the earlier fixed-amplitude gain. This is sufficient to pass the physics gate: Krylov signs materially improve the FN projector over Marshall at 6x6.

A first four-bin density-ratio fit using only the 416 mixed walkers from that Krylov pilot gave:
edges = [-inf, -18.07206, -17.97521, -17.88591, inf]
g = [-0.00949, -0.19851, +0.16378, +0.04238].
However 8 chronological chunks show per-bin SD about 0.21-0.27, comparable to the signal; 416 correlated samples are therefore not enough to freeze the 6x6 amplitude correction.

To reduce this uncertainty without rebuilding the expensive ViT/r_M cache twice, a new durable job is running two independent direct-ViT Krylov-FN replicas in one process, M=32, beta_target=2.0, burn_beta=0.4, tau_max=0.025, saving every post-burn population and fitting per-replica plus combined four-bin g(r) while the cache stays alive.

Job: 20260928-090211-49031
Target PID at launch: 49033
Job dir: /Users/aliotaifi/Chatty/jobs/20260928-090211-49031
Expected outputs:
- /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_krylov_collect_seed9001.npz
- /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_krylov_collect_seed9002.npz
- /Users/aliotaifi/j1j2_vit_bench/gr_6x6_krylov_fn_collect2.npz

Per AGENTS.md, do not poll this job from ChatGPT. After completion, compare the two independently fitted g vectors. If they agree within useful tolerance, freeze the combined g with eta=0.5 and launch the next 6x6 FN iteration; if not, the handoff remains sample-limited.


### 6x6 g(r) collection retry after wrapper bug — 2026-09-28
Job 20260928-090211-49031 failed immediately for an implementation-only reason: gfmc_6x6_krylov_collect_gr.py called run(..., outfile=...) but the generated run() signature did not include outfile, and the function also lacked the intended return mixed,Es. No physics calculation ran beyond model loading; this failure does not alter any prior numerical result.

Both bugs were fixed, the script recompiles, and the corrected durable copy is under projects/j1j2/experiments/gfmc_6x6_krylov_collect_gr.py.

Retry job:
- job id: 20260928-090512-49166
- target PID: 49168
- job dir: /Users/aliotaifi/Chatty/jobs/20260928-090512-49166
- cwd: /Users/aliotaifi/j1j2_vit_bench
- expected outputs: gfmc_6x6_krylov_collect_seed9001.npz, gfmc_6x6_krylov_collect_seed9002.npz, gr_6x6_krylov_fn_collect2.npz.

Per AGENTS.md, do not poll the retry from ChatGPT. On completion email/user return, inspect the durable job and g-vector agreement.


### 6x6 four-bin g(r) collection completed — stability criterion failed
Two independent M=32 direct-ViT Krylov-FN replicas to beta=2.0 completed successfully. Their inferred four-bin corrections are:
- seed 9001: g=[-0.07476,-0.12416,+0.00238,+0.19504], Emean=-18.15082
- seed 9002: g=[-0.07255,+0.06148,-0.06179,+0.07305], Emean=-18.05315
- combined: g=[-0.07539,-0.03176,-0.02981,+0.13629].

The lowest-r bin is reproducible, but the middle/high bins differ by up to ~0.19, and the two FN energies differ by ~0.098. Thus M=32 direct-ViT FN sampling is still too noisy to freeze g(r); do not launch the second 6x6 FN iteration with the combined g. This is a sampling/population-noise bottleneck, not yet a falsification of the four-bin representation. Next work must either increase effective FN population/independent sampling or provide a cheaper callable amplitude guide that permits larger populations while remaining validated against direct-ViT.


### g(r) population-scaling test launched — 2026-09-28
Current bottleneck is replica-to-replica noise in the 6x6 four-bin FN amplitude correction. To distinguish ordinary Monte Carlo noise from a deeper population/correlation problem, a matched population-scaling sweep is running with the Krylov sign guide and direct ViT amplitudes fixed.

Design:
- M = 32, 64, 128
- two independent replicas at each M
- beta_target = 1.2
- burn_beta = 0.4
- tau_max = 0.025
- same frozen Krylov threshold and same four-bin q-mass partition at every M
- fit g(r) separately for each replica and combined at each M
- estimate per-bin sigma_g = |g1-g2|/sqrt(2)
- primary diagnostic: RMS sigma_g over bins 1-3 and RMS*sqr(M)
- also track replica energy spread.

Each population size runs in a separate Python process so direct-ViT/r_M caches are reused within the two replicas at that M but released before moving to the next size, preventing unbounded cache growth.

Durable job:
- job id: 20260928-102846-50383
- target PID: 50385
- job dir: /Users/aliotaifi/Chatty/jobs/20260928-102846-50383
- runner: /Users/aliotaifi/Chatty/projects/j1j2/experiments/run_gr_population_scaling.sh

Expected outputs:
- /Users/aliotaifi/j1j2_vit_bench/gr_6x6_population_scaling_M32.npz
- /Users/aliotaifi/j1j2_vit_bench/gr_6x6_population_scaling_M64.npz
- /Users/aliotaifi/j1j2_vit_bench/gr_6x6_population_scaling_M128.npz
- /Users/aliotaifi/j1j2_vit_bench/gr_6x6_population_scaling_summary.npz

Interpretation criterion:
- if sigma_g approximately falls as 1/sqrt(M), the handoff is sample-limited but scalable in the ordinary Monte Carlo sense;
- if sigma_g saturates or decreases much more slowly while energy/population noise persists, the FN mixed-distribution handoff has a deeper correlation/population bottleneck.

Per AGENTS.md, do not poll this job from ChatGPT; completion will be emailed.

## Mechanism subproject closed: exact 4x4 self-correcting FN/Krylov loop — 2026-09-28

A corrected 4x4 benchmark now iterates the **current sign state**, rather than repeatedly rebuilding a Marshall-referenced correction:

r_k=(H a_{k+1}s_k)/(a_{k+1}s_k),
s_{k+1}=s_k sign(t_k-r_k),

with t_k selected label-free by exact fixed-amplitude energy minimization over grouped r_k thresholds.

At target J2/J1=0.5:
- exact-target-modulus + Marshall control reaches exact signs at iteration 27;
- a no-target-oracle bootstrap initialized with the J2=0 modulus + Marshall signs reaches exact target signs at iteration 100;
- final no-oracle diagnostics: O_S=1, F_a=0.99999462, E_guide-E0=2.60e-5 total.

Target ED information was used only for offline diagnostics, not in the recursive FN or sign updates.

This resolves the finite-size mechanism question positively and explains why the older 4x4 loop stalled: it used a fixed Marshall-referenced sign coordinate, a different nonlinear map.

This does NOT supersede the current 6x6 population-scaling job or solve the thermodynamic sign problem. It sharpens the parent question: the remaining bottleneck is whether FN walkers can be compressed into a sufficiently accurate callable amplitude representation so that the self-correcting current-sign loop survives at 6x6/8x8 with polynomial sampling/population cost.


### Population scaling completed; closed-loop iteration 2 launched — 2026-09-28
The durable population-scaling job 20260928-102846-50383 finished successfully (DONE marker present; job.json remained stale as "running"). Matched two-replica results:

M=32:
- RMS sigma_g over bins 1-3 = 0.27216
- RMS*sqrt(M) = 1.53957
- replica E means = -18.12634, -18.11056
- energy difference = -0.01578

M=64:
- RMS sigma_g over bins 1-3 = 0.12209
- RMS*sqrt(M) = 0.97670
- replica E means = -18.08768, -18.11007
- energy difference = +0.02240

M=128:
- RMS sigma_g over bins 1-3 = 0.03501
- RMS*sqrt(M) = 0.39613
- replica E means = -18.10808, -18.10679
- energy difference = -0.00129
- stable combined correction g0 = [-0.0409143,+0.0032384,+0.0254417,+0.0120599].

Conclusion: no evidence for a population-noise floor. Replica disagreement collapses strongly with increasing M, faster than 1/sqrt(M) across these three tested points (too few points/replicas to claim a scaling exponent, but clearly inconsistent with saturation). The 6x6 handoff is ordinary-sampling limited in this regime rather than blocked by an obvious correlation/population catastrophe.

Because the M=128 correction is small and stable, the actual second closed-loop iteration has been launched with:
a1(x)=a0(x) exp[0.5*g0_bin(r0(x))],
using r0 as a static coordinate to keep callable-amplitude depth constant across iterations. The next Krylov threshold is inferred label-free from the corrected amplitude using robust weighted two-means. Two independent M=128 FN replicas will then be run and a second correction g2 fitted in the same static coordinate.

Durable closed-loop job:
- job id: 20260928-124604-53026
- target PID: 53028
- job dir: /Users/aliotaifi/Chatty/jobs/20260928-124604-53026
- script: /Users/aliotaifi/Chatty/projects/j1j2/experiments/gfmc_6x6_closedloop_iter2.py

Expected outputs:
- /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_closedloop_iter2_seed9701.npz
- /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_closedloop_iter2_seed9702.npz
- /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_closedloop_iter2_summary.npz

Per AGENTS.md, do not poll this job from ChatGPT; use DONE/stdout as authoritative on return because job.json has shown stale status in prior completed jobs.


### Production migrated to Paderborn H100 — 2026-09-28
The Mac closed-loop iteration-2 process (PID 53028, Chatty job 20260928-124604-53026) was stopped intentionally because direct-ViT evaluation had become the wall-time bottleneck and Paderborn GPU resources are substantially better.

A clean production workspace now exists at:
- /pc2/users/h/hpcalot/chatty
- /pc2/users/h/hpcalot/chatty/j1j2/{src,inputs,results,slurm}

README files on Paderborn explain the project and source-of-truth policy. The Mac remains the development/durable source of truth; Paderborn is a production backend.

Staged inputs are minimal:
- 6x6 ViT checkpoint
- energy_krylov_vs_vit_6x6_indep.npz
- krylov_scaling_6x6_a1.20_tr4096_va2048.npz
- gr_6x6_population_scaling_M128.npz
- current closed-loop and population-scaling scripts
- nqsmagic source without historical params/.git.

A pinned Python 3.11 GPU environment was installed at:
- /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311
Versions match the Mac for JAX/Flax/NetKet/Numpy/Numba/Scipy. JAX GPU validation succeeded.

Steady-state ViT benchmark:
- Mac M2 CPU: ~2.61k states/s at batch 4096
- Paderborn A40: ~32.64k states/s
- Paderborn H100: ~782.36k states/s
The checksum matched across backends. H100 is therefore the production backend.

Current production Slurm job:
- job id: 3466194
- partition: gpu_h100
- resources: 1x H100, 16 CPUs, 64 GB RAM
- walltime: 4 h
- state when submitted/last checked: RUNNING on gpu1001
- batch script: /pc2/users/h/hpcalot/chatty/j1j2/slurm/closedloop_iter2_h100.sbatch
- stdout: /pc2/users/h/hpcalot/chatty/j1j2/results/closedloop-iter2-3466194.out
- stderr: /pc2/users/h/hpcalot/chatty/j1j2/results/closedloop-iter2-3466194.err
- result directory: /pc2/users/h/hpcalot/chatty/j1j2/results/closedloop_iter2_3466194
- native Slurm END/FAIL email is enabled and routed to the Chatty notification address.

Per AGENTS.md, do not poll/wait from ChatGPT. After the completion email/user return, read the Paderborn result files, sync decisive outputs back to the Mac, and continue the loop.


### Paderborn H100 iter2 first attempt failed; corrected retry queued — 2026-09-28
Production job 3466194 reached corrected-amplitude diagnostics successfully on H100 but failed before FN projection due to an implementation-only namespace bug in gfmc_6x6_closedloop_iter2.py: line 109 called bare ensure_r(pool) instead of b.ensure_r(pool). Environment, checkpoint loading, CUDA/JAX, and pre-FN diagnostics were healthy. The failed run reported T1=-14.651543935454582, physical-weighted sign change 0.0 on the diagnostic pool, and reweighted guide energy -18.11655157 before hitting the NameError.

The Mac source of truth was fixed, py_compile passed, and a static scan confirmed there are no remaining bare ensure_r(...) calls. Corrected source was synced to Paderborn and resubmitted as Slurm job 3466255 on gpu_h100 with 1 H100, 16 CPUs, 64 GB RAM, 4 h walltime. Native Slurm END/FAIL email was explicitly updated to the Chatty notification address. At last check the replacement job was PENDING for Priority; it had not started and had no stdout/stderr yet. Do not treat it as failed; wait for allocation/email.


### 6x6 four-bin g(r) reproducibility test — 2026-09-28
The corrected two-replica M=32 Krylov-FN collection completed. Replica results:
- seed 9001: Emean=-18.15082, tail8=-18.14134, 2336 mixed walkers / 791 unique
- seed 9002: Emean=-18.05315, tail8=-18.07980, 2176 mixed walkers / 754 unique.
Four-bin fits:
- g_9001 = [-0.07476, -0.12416, +0.00238, +0.19504]
- g_9002 = [-0.07255, +0.06148, -0.06179, +0.07305]
- combined = [-0.07539, -0.03176, -0.02981, +0.13629].
The replicas agree only in the first bin and disagree materially in bins 2 and 4; their projected energies also differ by ~0.098. Pooling them is therefore not a controlled amplitude handoff.

Early/late splits show strong within-replica drift, especially seed 9001 (bin4 g: 0.287 -> 0.100), indicating finite-population/autocorrelation noise is the leading explanation rather than proven failure of the four-bin representation.

Next falsification changes population size rather than extending M=32 time. Two independent M=128, beta_target=0.9, burn_beta=0.3 direct-ViT Krylov-FN runs are launched under Chatty durable workflow. If their g vectors still disagree materially, treat the 6x6 four-bin handoff as failing.


### Decisive handoff verdict — 2026-09-28
The 6x6 1D four-bin g(r_M) density-ratio handoff is now falsified as a reproducible amplitude representation. Two M=128 independent direct-ViT Krylov-FN replicas gave g vectors [-0.09836,+0.12777,+0.00997,-0.03872] and [+0.04305,-0.02240,-0.03037,+0.00973], differing by ~0.14-0.15 in the first two bins despite 4x larger population. Do not scale this same representation further. The Krylov sign/FN physics result remains positive; the failed component is the scalar amplitude handoff. Next work should test a compact multivariate/direct correction with constant-cost evaluation.


### Revised bottleneck after FN convergence audit — 2026-09-28
The immediate bottleneck is now the 6x6 FN projector/mixed-distribution sampler, not proven failure of the four-bin amplitude representation. Independent M=128 Krylov-FN replicas have distinct late physical observables (C2 0.1569 vs 0.1236; staggered m^2 0.1428 vs 0.1073) and late energies -18.1280 vs -18.0807, well beyond within-run fluctuations. A multivariate cheap-feature density-ratio classifier fits each population but transfers below random (~0.42-0.44 AUC), showing population-specific bias. Do not train richer g models until independent FN runs converge to the same mixed distribution. Next task: stabilize/validate projector sampling (larger population, improved branching/reconfiguration, or exact small-system calibration of population bias) and establish cross-seed observable agreement.


### Controlled 4x4 diagnosis revises 6x6 handoff verdict — 2026-09-28
A same-estimator 4x4 test with exact known f/q shows raw fixed-population GFMC snapshots produce non-reproducible four-bin g estimates even when the representation is exactly adequate. At M=128, two 4x4 replicas still differ by ~0.07-0.12 per bin from each other and by norm ~0.09-0.11 from exact g_true. Therefore the 6x6 M=32/M=128 disagreement primarily diagnoses an estimator/effective-sample-size problem, not a proven failure of scalar g(r_M). Pivot: repair mixed-distribution sampling/estimation before increasing model complexity.


### 6x6 closed-loop fixed-point verdict — 2026-09-29
See `results/fn_krylov_closedloop_6x6_verdict_2026-09-29.md`. Paderborn H100 job 3466255 completed successfully in 29m41s. After the stable g0 handoff, the reconstructed guide had physical-weighted sign_change=0.0. The second four-bin correction was g2=[0.00378861,-0.01087963,0.00257648,0.00450005], while pairwise replica-noise estimates were [0.01067656,0.07082232,0.04207335,0.10212746]; every |g2_i|<0.36 sigma and RMS(g2)=0.00632 vs RMS(noise)=0.06582. Mean FN E=-18.11297225 with between-replica SE=0.00362079. Treat 6x6 loop as a practical fixed point in the tested regime. Do not inject noisy g2 into iteration 3. Next decisive step is 8x8 scaling of the converged protocol.


### 8x8 scaling campaign launched — 2026-09-29
The 6x6 closed loop is treated as converged in the tested regime (see results/fn_krylov_closedloop_6x6_verdict_2026-09-29.md), so iteration 3 was intentionally not launched because g2 was noise-dominated and the Krylov guide had zero physical-weighted sign change.

Next decisive scaling test: actual 8x8 fixed-node projector.

Inputs:
- threshold T8 = -28.37107876288694 from the label-free alpha=1.2 robust kmeans 10-90 construction;
- checkpoint vit_J2=0.50_N=8x8_k=0.mpack;
- physical |a|^2 pool krylov_phys8_a2_fixedT.npz;
- alpha=1.2 train sample krylov_scaling_8x8_a1.20_tr4096_va2048.npz.
A one-state smoke test reproduced the stored 8x8 local quantity exactly: r=-35.930856324924726.

Two Paderborn H100 jobs were submitted in parallel:
1. Krylov-FN population/handoff run: Slurm 3471544
   - 1 H100, 16 CPUs, 128 GB, 6 h
   - M=128, two independent replicas, beta_target=1.2
   - after projection, fit the first 8x8 four-bin density-ratio correction g0^(8).
2. Matched Marshall-FN control: Slurm 3471545
   - same resources and projector parameters, two independent replicas.

At last check both were PENDING for Priority and had not started; no stderr/stdout yet.

Robust Mac-side remote Slurm watchers are now in place via tools/local_loop/watch_paderborn_slurm.sh. Unlike the earlier watcher, this version treats SSH/VPN failures as transient and only exits after sacct reports a terminal Slurm state. Native Paderborn END/FAIL mail remains enabled as a second notification path.


### True second-Krylov node test launched — 2026-09-29
Goal: determine whether a genuine second Krylov application improves the nodal/sign structure beyond the production one-step guide, independently of the FN amplitude loop.

Definition:
psi0(x)=s_M(x)a0(x)
psi1=(T1-H)psi0 with T1=-14.985799779143964
r1(x)=(H psi1)(x)/psi1(x)
psi2=(T2-H)psi1
so s_K2=s_K1*sign(T2-r1).

T2 is learned label-free from the existing alpha=1.2 train sample, reweighted from a0^alpha to |psi1|^alpha by |T1-r0|^alpha. Reweighting ESS remains healthy: ~1926/4096 train and ~938/2048 validation.

The test will measure:
- physical-weighted K1->K2 sign-change mass on train/validation;
- change on the independent physical |a0|^2 sample;
- hidden-ViT sign overlap only as a diagnostic, not for threshold selection;
- threshold stability across 1-99, 5-95, and 10-90 robust weighted two-means fits.

Paderborn H100 Slurm job: 3471990.
At last check: PENDING for Priority.
Robust terminal-state Chatty watcher attached via watch_paderborn_slurm.sh.


### True K2 node test verdict — 2026-09-29
See `results/true_k2_vs_k1_6x6_verdict_2026-09-29.md`. A genuine second Krylov step changes only ~0.8-2.0% of physical 6x6 configurations but raises the fixed-amplitude energy relative to K1 for all three label-free T2 choices: +0.646±0.322, +0.903±0.377, +0.960±0.381. Thus naive K2 is worse than K1 on 6x6 and should not replace the production one-step guide. This does not prove the K1 node exact; it closes the straightforward K2 continuation route.


### Migration checkpoint — 2026-09-29
Main campaign status:
- 6x6 FN -> amplitude -> Krylov loop reached a practical fixed point in the tested M=128, beta=1.2 regime.
- After first handoff, physical-weighted Krylov sign change was 0.0 and second amplitude correction g2 was statistically consistent with zero.
- True K2 side test completed: K2 changes only ~0.8-2.0% of physical configs but raises fixed-amplitude energy relative to K1 for all tested label-free T2 choices:
  +0.64584 +/- 0.32244,
  +0.90285 +/- 0.37750,
  +0.96024 +/- 0.38078.
  Conclusion: naive second-Krylov continuation is worse than K1 on 6x6. Do not use K2 as production node update.
- Parked side angles are listed in OPEN_ANGLES.md and should not distract from the main campaign.

Active heavy scaling jobs:
- 3471544: 8x8 Krylov-FN, M=128, two replicas, H100, currently pending for Priority. Last estimated start: 2026-10-01 11:55 local cluster time.
- 3471545: 8x8 Marshall-FN control, M=128, two replicas, H100, currently pending for Priority. Last estimated start: 2026-10-01 17:55 local cluster time.
- These are intentionally left on H100 because A40 ViT inference benchmark was ~24x slower.
- Main next scientific question: does the 6x6 fixed-point / modest-resource behavior survive at 8x8?

Operational notes:
- Durable source of truth: /Users/aliotaifi/Chatty/projects/j1j2
- Read STATUS.md first, then OPEN_ANGLES.md only if needed.
- Paderborn production workspace: /pc2/users/h/hpcalot/chatty/j1j2
- VPN helper on ws3: bash /project/theorie/a/A.Otaifi/paderborn/pc2-vpn-start
- Robust remote Slurm watcher script: /Users/aliotaifi/Chatty/tools/local_loop/watch_paderborn_slurm.sh

### Finite-temperature / CTQMC side angle — first positive transient test (2026-09-30)
The parked finite-T/CTQMC analogue was activated as a separate side branch. On exact 4x4 at J2/J1=0.5, propagator columns psi_y(x,tau)=<x|exp(-tau H)|y> were evolved by interleaving short positive fixed-node propagation with an on-the-fly sign update s_{n+1}=sign[(I-dt H)a_n s_n], with exact propagation used only for diagnostics. At dt=0.05, tau=0.5: stripe Marshall weighted sign mismatch is 0.106638 versus 1.259e-6 adaptive; random is 0.076801 versus 4.825e-5 adaptive. Full-state fidelities remain ~0.986 and ~0.985 respectively. This is not standard FN because the sign constraint moves during imaginary-time evolution. The unresolved scalability issue is replacing one full transient vector per boundary y by a shared/sampleable representation in joint (x,y) or history space. A uniform-positive-guide falsifier is running as Chatty job 20260930-062608-96686; see results/finite_tau_ctqmc_side_angle_2026-09-30.md.

Finite-T side-angle update: the beta=0.5 uniform-positive first-guide test passed essentially identically to the J2=0-guide run, removing that initialization crutch. Cross-column exact diagnostics then showed that a y-independent sign correction does not transfer (16-column LOO error ~0.348 vs Marshall ~0.103), but a joint correction indexed by endpoint difference d=x xor y transfers strikingly well: LOO mismatch ~0.00715 for 16 columns and ~0.00543 for 64 columns at tau=0.5, versus Marshall ~0.1067. Hamming weight of d alone gives no gain, so geometry of the difference pattern matters. Raw XOR lookup remains exponential; this is structural evidence for a joint (x,y) representation, not yet a scalable QMC algorithm. See results/finite_tau_ctqmc_side_angle_2026-09-30.md.

Finite-T side-angle structural update: shortest operator-path parity predicts exact 4x4 propagator signs extremely well at short/intermediate imaginary time. At tau=0.5 across 64 columns, weighted sign error is 4.17e-4 vs Marshall 0.1148. On 12,800 sampled endpoint pairs, configuration-graph shortest distance exactly matched a polynomial min-cost matching distance on the physical lattice (100% distance/parity agreement; not yet a proof). The sign rule degrades with tau: errors 3.8e-12, 8.55e-7, 4.14e-4, 2.70e-2, 0.203, 0.405 for tau=0.1,0.25,0.5,1,2,4. This supports repeated short-time sign reconstruction with amplitude refresh rather than a single global K2/K3-style polynomial. See results/finite_tau_ctqmc_side_angle_2026-09-30.md.

Finite-T side-angle 20-site scaling result: on a bipartite 20-site skew torus (Sz=0 D=184,756), exact propagator columns confirm the endpoint minimum-matching parity sign rule. Sampling 5,000 endpoints from each of four exact columns gives zero matching-sign errors at tau=0.25 (20k samples total; Marshall 3.905% mean error) and one error in 20k at tau=0.5 (5e-5 mean; Marshall 8.545%). The short-time rule therefore survives beyond 4x4. Next step is an actual sampled short-time QMC kernel with repeated positive propagation + sign refresh, not another finite-size mechanism test.

### FN-refresh -> projected sign-step scaling angle opened — 2026-09-30
Angle #1 (why K1 works) is closed on 6x6: the useful signal is the amplitude-weighted off-diagonal J1/J2 competition field. Raw true-K2 is not the production route.

The separate scalable iteration is now tracked as OPEN_ANGLES.md angle #9:
`K1 signs -> FN amplitude refresh -> recompute local-energy field -> label-free projected sign step`.

Important correction: 6x6 job 3466255 had already performed exactly this test. After the FN amplitude handoff, the threshold shifted from -14.9857997791 to -14.6515439355 but the physical-weighted sign change was exactly 0; the next amplitude correction was consistent with noise. Thus 6x6 reached a practical fixed point after one projected sign step plus FN refresh.

8x8 first-FN job 3471544 completed successfully. Measured four-bin correction:
`g0^(8) = [0.14059278, 0.01262784, -0.08930914, -0.06367385]`.
Replica mean energies: -31.88701572 and -31.88383250.

Dedicated 8x8 closed-loop iteration-2 Slurm job 3491791 is now submitted/running on H100. It consumes the measured g0^(8), recomputes the projected sign field, measures K1->s2 sign change, then runs two FN replicas and estimates the next amplitude correction. Durable watcher: Chatty job 20260930-065655-98000. See `results/fn_refresh_projected_sign_angle_2026-09-30.md`.

Finite-T side-angle algorithmic correction: despite near-perfect endpoint matching signs, the 20-site absolute-history measure still has strong internal cancellations. Factoring out the endpoint matching sign raises the average history sign from ~1.95e-4 to ~0.379 at tau=0.25 and from ~4.98e-8 to ~0.0609 at tau=0.5, but does not make it sign-free. This is the first direct sampled-QMC obstruction: endpoint sign prediction and history-level sign severity are distinct. Shorter-tau decay sweep is running as Chatty job 20260930-070228-98300; use it to decide whether repeated short adaptive blocks can remain polynomial or still accumulate exponential sign loss.

### 8x8 first-FN scaling verdict + amplitude-handoff separation — 2026-09-30
See `results/8x8_krylov_fn_scaling_verdict_2026-09-30.md`.

Completed matched H100 runs:
- Krylov-FN Slurm 3471544, M=128, two replicas: E=-31.88701572 and -31.88383250; mean -31.88542411; between-replica SE 0.00159161.
- Marshall-FN Slurm 3471545, M=128, two replicas: E=-31.70382648 and -31.65112726; mean -31.67747687; between-replica SE 0.02634961.
- Central difference E_K-E_M=-0.20794724 +/- 0.02639763 from replica spread (~7.9 sigma diagnostic with only two replicas).

Conclusion: the one-step K1 sign guide plus FN projection survives cleanly to 8x8 at modest M=128. The failure point is not the first FN solve.

The 8x8 four-bin density-ratio handoff is much less controlled:
- g0^(8)=[0.14059278,0.01262784,-0.08930914,-0.06367385]
- pair-noise estimates=[0.09486770,0.01755663,0.07170566,0.04069966].
No bin is above 2 pair-noise units.

Independent 6x6 audit Slurm 3479622 found the previous four-bin a1 correction did not reduce the fixed-sign local-energy-dispersion proxy (SD ratios 1.0366 train, 1.0232 validation). A translation-invariant fixed-amplitude sign-correction network produced zero held-out flip mass in all three seeds; this is no positive evidence for a residual nodal correction, but not a proof of optimality because STE training may be stuck.

A subsequent 8x8 g0 refresh changed only 0.00279486 of reweighted training sign mass (3 sampled states), yet the nested callable r0->a1->r1->sign evaluation caused the second FN implementation to grow to ~14.7M cached r0 states and ~625.7M ViT evaluations by beta~0.44, then OOM at 128 GB. Thus the present static-g closed-loop has a severe polynomial-prefactor/cache wall even though the physical node update is tiny.

Operational conclusion: preserve K1 as the successful scalable sign extractor. Do not invest further in the static four-bin g wrapper. The amplitude branch should move to a directly callable learned FN-amplitude representation (frozen deep-feature residual as a final compact test, or full-network FN-informed retraining). Coordinate with the separate amplitude-learning project rather than duplicate it.


### Direct callable learned-FN amplitude integration scaffold — 2026-09-30
The static four-bin \(g(r)\) handoff remains closed. A new flat integration path is now in
\`experiments/callable_fn_loop.py\`: one backend supplies batched \(\log a_{\rm FN}(x)\)
directly, after which the code recomputes the local-energy field, label-free K1 threshold,
K1 signs, and the FN projector without any recursive \(r_0\to g(r_0)\to a_1\) dependency.
The backend contract is documented in \`AMPLITUDE_HANDOFF_API.md\`.

Regression through the original 8x8 ViT backend passed: on 64 stored K1 states,
max \(|\Delta r|=7.18\times10^{-13}\) and mean \(|\Delta r|=7.21\times10^{-14}\).
A 4x4 constant-amplitude engine smoke test also passed.

The actual learned-FN amplitude artifact from the separate amplitude-learning branch is not
yet present on this Mac or in the searched Library, so no learned-amplitude physics result is
claimed yet. Next step is purely the handoff: provide that backend/checkpoint and its
threshold/pool sample bundle, then run the flat learned-amplitude K1/FN loop and compare
node change plus FN energy to the established K1-FN baseline. See
\`results/callable_fn_amplitude_integration_2026-09-30.md\`.

### Learned callable FN amplitude branch promoted to production — 2026-09-30
The missing amplitude-learning dependency has now been built inside the main project rather than
left external. Using the two independent 8x8 M=128 K1-FN mixed populations as separate domains,
a translation-invariant periodic CNN density-ratio residual transfers at AUC 0.6412
(rep1->rep2) and 0.6361 (rep2->rep1), versus only ~0.53 for a linear correlation model.
Cross-replica calibration shows the raw residual is overconfident; the production callable
amplitude is log a_learn = log a_ViT + eta*g_theta with eta=0.35967 (mean of 0.3207 and
0.3986 held-out calibrations). The backend is deterministic and finite.

The original alpha=1.2 threshold samples are reweighted by exp(2 eta g_theta). Train+validation
are combined with sample-count weighting, giving threshold ESS=358.0; the learned physical
initialization pool has ESS=2338.3. Artifacts:
- results/fn_residual_cnn_8x8.mpack
- results/learned_fn_handoff_8x8.npz
- experiments/learned_fn8_callable_amplitude_adapter.py
- experiments/callable_fn_loop.py
- results/learned_fn_amplitude_8x8_2026-09-30.md

Production H100 job 3494473 is submitted with two M=128 replicas (seeds 12001,12002),
beta_target=1.2. It recomputes the label-free K1 field from the learned amplitude and runs the
FN projector with the same flat callable amplitude. The earlier 3494470 submission was
intentionally cancelled before use to replace train-only threshold inference with the combined
train+validation threshold sample. Durable watcher will email on terminal Slurm state.

### Learned-FN 8x8 production backend hedge — 2026-09-30
H100 learned-amplitude job 3494473 remains pending for Priority (scheduler estimate now
2026-10-01 06:25). To avoid leaving the main campaign idle, an otherwise identical A40
duplicate was submitted as Slurm job 3494725 with a 12 h walltime. It started immediately
on node fpga1701 at 2026-09-30 14:31 local time and is now the active production path.
A durable Chatty watcher is attached. If the A40 run returns a clean complete result first,
use it for the physics verdict and cancel the queued H100 duplicate; otherwise retain the
H100 job as fallback.

Finite-T side-angle exact adaptive 20-site test completed: dt=0.05 to tau=0.5 on D=184,756 gives full fidelities 0.9999922 and 0.9998929 for two columns, with weighted sign mismatch masses 1.20e-6 and 2.62e-5. The mechanism therefore survives well beyond 4x4. Do not repeat the brute-force exact sparse-exponential implementation; next step is the actual finite-population walker realization of positive propagation + sign refresh.

Finite-T side-angle stage verdict: exact 20-site adaptive positive/FN propagation passes nearly exactly to tau=0.5; endpoint sign features d,n2 are polynomial via physical-lattice matching (4000/4000 lexicographic matches). Rich polynomial amplitude features (d,n2,Ediag,J1/J2 flippable counts) have oracle fidelity 0.9926/0.9901 at tau=0.5. A finite-pop table-GFMC prototype reaches ~0.9975 guide fidelity near tau=0.2 but only ~0.933 by tau=0.5 at M=8192; smaller refresh intervals and aggressive/symmetric-prior updates worsen support collapse. Therefore the active bottleneck is learned amplitude estimation from walkers, not sign structure or positive propagation. Next: pair-conditioned callable amplitude model trained from independent walker populations; stop manual table tuning.

Finite-T learned-guide update: direct amplitude regression from reweighted FN walkers (no ED training) transfers across replicas with grouped log-amplitude RMSE 0.243 and exact diagnostic fidelity 0.951/0.938 at tau=0.5. Correct guide changes require walker handoff weights g_new/g_old; omitting this was a bug and produced artificial collapse. With the corrected positive GFMC and delayed learned-guide switch at tau=0.15, the hard 20-site column gives final walker amplitude fidelity 0.93687 (M=8192), 0.97039 (M=32768), 0.97926 (M=65536). The residual 1-F scales approximately M^-0.537 over these points, consistent with ordinary Monte Carlo population error rather than visible exponential degradation at this size. Next: system-size scaling of required M / handoff ESS, then larger systems.

### 8x8 learned classifier-amplitude production test fails — 2026-09-30
A40 job 3494725 completed both M=128 learned-amplitude FN replicas; Slurm shows FAILED
only because the final summary script had a 4096-vs-6144 postprocessing broadcast error.
The physics outputs are valid. The learned threshold moved from T0=-28.37108 to
T1=-30.85727. On the original 4096 threshold-train states, 649 states (15.84%) change
K1 sign; this is 17.75% of the old physical weight and 44.78% under learned reweighting.
The two FN replica means are -27.00354 and -27.84396, pair mean -27.42375 with
between-replica SE proxy 0.42021, versus the established original K1-FN
-31.885424 +/-0.001592. Thus eta=0.35967 classifier residual is decisively not a
production-safe amplitude update.

The callable FN propagation kernel was checked line-by-line against the successful original
8x8 GFMC implementation and has the same fixed-node construction/branching/resampling core.
Do not interpret this as a projector rewrite failure. Treat the classifier update strength/
representation as the failed object.

Next gate is an eta trust-region scan (A40 Slurm 3495590) over
eta={0,.01,.02,.05,.10,.20,.35967}, recomputing the K1 threshold and node-change mass.
If strong damping does not produce a controlled useful direction, close the classifier
residual branch and move to the full-network walker-MLE amplitude update.
See results/learned_fn8_classifier_verdict_2026-09-30.md.

Finite-T preferred algorithm update: decouple FN amplitude guide from MC importance sampling. Use learned a_theta only in the FN diagonal sign-flip potential, but keep q(x)=1 for walkers; this removes guide-handoff population collapse. On 20 sites at tau=.5, hard-column amplitude fidelity is 0.98075 (M=8192) and 0.98642 (M=32768), versus 0.94345/0.96155 with uniform FN amplitude. At M=32768 the corrected learned-FN run remains 0.9543 at tau=.75 and 0.89845 at tau=1. Endpoint-sign mismatch is still only 6.22e-4 at tau=1, so long-time error is amplitude-model extrapolation (current model trained only to .5). Next stage: train a second amplitude model from corrected uniform-importance FN walkers over tau=.5..1 and close the loop.

### Exact 4x4 gate for full-network walker-MLE amplitude update — 2026-09-30
The proposed MLE amplitude identity was tested exactly. For mixed walkers
f proportional to a*phi_FN, ideal MLE of a_new^2 to f gives
a_new proportional to sqrt(a*phi_FN). Because the lattice FN sign-flip potential depends
on current guide amplitude ratios, the actual iteration is the nonlinear map
a_k -> sqrt(a_k*phi_FN[a_k,s_k]) with interleaved K1 signs.

On exact 4x4 (dim 12,870, E0=-8.45792335), the ideal half-step map passes strongly.
From exact amplitudes + Marshall signs, the first half-step reaches E=-8.44578657 and
sign overlap 0.99941675; the weighted log-distance to the current FN target is halved
essentially exactly. From a deliberately bad uniform-amplitude Marshall guide the first
half-step is non-monotone (E -5.0667 -> -4.4023), but the loop recovers to -8.4249 by
round 11 and -8.44962 by round 29 with sign overlap 0.997918.

A finite-sample diagnostic trained a fresh small full-amplitude MLP: guide pretraining fidelity
was only 0.96776, and after MLE on 10k mixed walkers (2,085 unique) fidelity to the exact
ideal half-step was only 0.94984; learned K1 E=-7.98753 versus ideal -8.44579.
Interpret this as capacity/initialization failure, not MLE failure: production must fine-tune
the existing expressive guide network or an equivalently capable warm-started model, not
train a small replacement from scratch. See results/fn_mle_halfstep_4x4_2026-09-30.md.

Meanwhile the classifier-residual branch has exact eta scan 3495590 and matched eta=0.20
M=128 pilot jobs 3495604 (A40) / 3495603 (H100 fallback). Linearized full-sample estimate
suggests eta=0.20 changes ~1.76% of old physical node weight versus 17.75% at eta=0.3597.

Exact 20-site endpoint-sign stress test over 8 columns falsifies strict tau-independence of s=(-1)^d: mean weighted mismatch is 2.41e-4 at tau=.5, 4.29e-3 at .75, 1.90e-2 at 1.0, 8.29e-2 at 1.5, 0.169 at 2.0. Thus actual zero crossings do occur. The rule is excellent in the short/intermediate regime used so far, but long-beta algorithms must compose short blocks / refresh signs rather than freeze the initial endpoint parity forever.

Finite-T learned-refresh loop now works on 20 sites: fixed uniform guide full fidelity at tau=.5 is 0.99557/0.97175; one walker-trained guide-matched density-ratio refresh gives 0.99625/0.98712; two refreshes give 0.99722/0.99778. Independent-replica residual AUCs are ~0.885 for refresh 1 and ~0.684 for refresh 2. This is the first direct evidence that the proposed positive-FN/walker/amplitude-refresh loop self-corrects beyond 4x4 without ED amplitudes in the update.

Finite-T learned-refresh convergence: fresh 20-site run gives full fidelities iter0 0.99557/0.97175, iter1 0.99570/0.98518, iter2 0.99741/0.99705, iter3 0.99716/0.99856. Cross-replica residual AUC falls 0.887 -> 0.692 -> 0.595, consistent with approaching a fixed point. Iter3 slightly worsens the already-easy column, so further updates must be validation-gated rather than unconditional.

Finite-T learned-amplitude attempt: independent-replica classifier transfers strongly (AUC 0.9897) but fails quantitative amplitude reconstruction; ED diagnostic fidelities only ~0.73-0.75 at tau=0.5 raw and worse after monotone calibration. Therefore density-ratio classification against uniform negatives is rejected for this finite-T handoff. The next estimator must learn local log-amplitude ratios / projector consistency directly from walker dynamics; high classification AUC is not an acceptance criterion.

### Threshold-measure correction and MLE promotion — 2026-09-30
Audit of the failed learned-classifier 8x8 branch found a threshold-measure bug in the new
handoff. The historical K1 threshold T0=-28.3710787629 was obtained by unweighted robust
two-means on states already sampled from a0^alpha with alpha=1.2. The learned handoff instead
fed physical a^2 importance weights into the threshold fitter. This is wrong: for
a_eta=a0 exp(eta g), threshold-sample reweighting must be proportional to exp(alpha eta g),
whereas the physical initialization pool correctly uses exp(2 eta g).

Consequently the prior eta trust scan 3495590 and eta=0.20 pilots 3495604/3495603 are
invalid as threshold/node diagnostics: their eta=0 regression already gives T=-31.71 rather
than T0=-28.371. They were cancelled/closed and must not be used for physics conclusions.
The eta=0.3597 learned-amplitude FN run still demonstrates that the tested classifier handoff
was not production-safe, but its exact node-change attribution is confounded by this threshold
bug.

The amplitude campaign is now promoted to the theoretically cleaner full-network walker-MLE
half-step. Exact 4x4 already validated the ideal map a_new proportional to sqrt(a_g phi_FN).
For 8x8, the existing expressive ViT is warm-started and optimized with the stochastic
maximum-likelihood score gradient: FN mixed walkers are positive samples and fresh current-ViT
samples provide the model expectation. Replica 1 walkers train; replica 2 is strictly held out.
Paderborn H100 job 3497176 runs 160 steps at batch 512, LR=2e-5. Acceptance requires positive
held-out rep2 log-likelihood gain with healthy guide importance-sampling ESS. Only an accepted
checkpoint proceeds to corrected alpha=1.2 K1 threshold/node diagnostics and then matched FN.

Critical finite-T sign correction: exact 20-site sweep over 8 full columns and tau=0.005..8 falsifies tau-independent endpoint sign (-1)^d. First resolved zero crossing found at tau~0.01405245 for one microscopic-weight endpoint; weighted mismatch stays tiny through short tau but grows to mean 2.41e-4 at tau=.5, 1.90e-2 at tau=1, 0.169 at tau=2, and 0.388 at tau=8. Exact ground-state sign products disagree strongly with (-1)^d for generic y, while tau=8 propagated signs converge to the ground-state pattern. Therefore use (-1)^d only as short-time/initial guide; production finite-T loop must refresh signs with tau, likely via the existing Krylov/K1 sign step after each positive amplitude block.

### 8x8 warm-start full-network MLE optimizer gate fails — 2026-10-01
H100 3497176, A40 3497177, and quick A40 3497195 all completed and independently reject
the present Adam update. At step 1, held-out rep2 log-likelihood gains are -0.2524, -0.2878,
and -0.2594 respectively; every run selects step 0 as the best checkpoint. One Adam step at
LR=2e-5 already changes guide log-amplitudes with RMS ~0.33-0.36 and collapses guide-pool
importance ESS from 4096 to ~805-993. The sampler state convention was checked explicitly:
NetKet Spin(1/2) states are +/-1, matching FN walker encoding. Thus this is an optimizer/
estimator issue, not an input-convention bug.

The exact 4x4 MLE half-step identity remains valid. The next and only useful diagnostic is a
full-batch empirical KL gradient at theta0 (all rep1 walkers vs stored a0^2 guide samples)
with a line search measured on independent rep2. H100 job 3500546 is submitted. If an
infinitesimal step yields positive held-out gain, switch to trust-region/small-step training;
if not, close the present finite-population MLE estimator at M=128 and increase walker/sample
quality or change the estimator.

### Infinitesimal 8x8 MLE line search rejects Euclidean gradient — 2026-10-01
H100 3500546 completed the full-batch empirical KL-gradient line search using all rep1 FN
walkers against the stored guide a0^2 sample, with rep2 strictly held out. The Euclidean
MLE direction does not transfer even infinitesimally: for the Adam-first-step direction at
scale 1e-9, train log-likelihood gain is +5.58e-6 while held-out rep2 gain is -1.92e-5;
the same sign persists throughout the small-step regime. Raw normalized-gradient directions
also give negative held-out gain. Therefore the failure of the earlier Adam runs is not just
an oversized learning rate: the rep1 empirical Euclidean MLE gradient itself is misaligned
with the independent-replica objective at M=128.

This still leaves SR/natural-gradient preconditioning open, because S^{-1} can rotate the
direction in function space. Direct cross-replica SR derivative test submitted as H100
3502036, scanning diagonal shifts 1 down to 1e-3. If g_rep2 dot S^{-1} g_rep1 remains
negative, close optimizer tuning and attribute the bottleneck to finite-population amplitude
gradient estimation/sample quality.

### Standard SR test inconclusive due QGT memory; switched to on-the-fly SR — 2026-10-01
H100 3502036 completed the cross-replica setup and measured a strongly anti-aligned Euclidean
MLE gradient between independent FN replicas: cosine(g1,g2) = -0.8528923. However the
standard NetKet SR/QGT implementation OOMed for every tested diagonal shift, requesting
18.5-78.7 GiB temporary allocations, so no SR-preconditioned derivative was obtained.
This is a representation/memory failure, not an SR physics verdict.

NetKet 3.22.4 provides the large-model route via on-the-fly QGT / MinSR-style sample-space
machinery. A new H100 job 3505954 forces QGTOnTheFly with iterative CG and chunk_size=128,
scanning diagonal shifts {1,0.1,0.01,0.001}. The decisive quantity remains
g_rep2 dot S^{-1} g_rep1: positive means SR rescues a transferable infinitesimal MLE update;
negative means optimizer geometry does not cure the M=128 estimator noise.

### Matrix-free SR partially succeeds: SR rotates MLE gradient into transferable direction — 2026-10-01
H100 3506027 built a matrix-free empirical Fisher on 512 guide samples. Euclidean rep1/rep2
MLE gradients remain strongly anti-aligned, cosine=-0.8528923. Crucially, for diagonal shift
lambda=1, the converged SR solve gives train directional gain g1^T S^-1 g1=11.8118 and
held-out directional gain g2^T S^-1 g1=+3.84998, with relative linear-solve residual
9.58e-6. Thus SR/natural-gradient geometry does rescue the infinitesimal direction across
independent FN replicas at this regularization. The job OOM-killed only when proceeding to
additional shifts; this does not invalidate the lambda=1 result.

Next gate: finite SR line search at lambda=1 only, avoiding repeated QGT solves. H100 job
3506041 tests held-out likelihood, guide ESS and delta-logamp RMS versus step size.

### FIRST VIABLE 8x8 CLOSED LOOP: FN-walker MLE + SR improves K1-FN energy — 2026-10-01
The project has now passed a full physical 8x8 feedback-loop gate. Baseline ViT/K1 fixed-node
energy was -31.8854241096 +/- 0.00159161. A matrix-free SR-preconditioned MLE update
(lambda=1, eta=0.01) trained on baseline FN walkers gives held-out replica likelihood gain
+0.015575, guide-pool ESS 3235.7/4096, and corrected alpha=1.2 threshold ESS 2140.5/4096.

The K1 threshold shifts from T0=-28.3710787629 to T1=-28.8241669031. On the historical
a0^1.2 threshold sample, 184/4096 states change K1 branch: 4.492% raw or 9.273% under the
correct new a1^1.2 reweighting. This threshold inference does NOT use the previously wrong
physical a^2/iw weighting.

Matched M=128 FN replicas with the updated amplitude/signs give
seed13001 -31.8967878286 and seed13002 -31.9056697050, hence
E_iter1=-31.9012287668 +/- 0.00444094 (two-replica spread proxy), Delta E=-0.0158046572
relative to baseline. Combined with baseline spread this is ~3.35-sigma diagnostically,
though two replicas are not enough for a precision statistical claim.

This is the first viable production demonstration of:
FN walkers -> MLE force -> SR amplitude update -> K1 sign refresh -> lower FN energy.
Result: results/fnmle_sr_closedloop_8x8_iter1_2026-10-01.md
Iteration 2 is now only a convergence/fixed-point diagnostic.

### 8x8 SR round-1 closed-loop energy is non-monotone / worse — 2026-10-01
Production rep1-trained SR checkpoint (lambda=1, eta=.01) passed the held-out MLE gate and
changed only 0.988% of physical a1^2 K1 sign mass. Optimized two-replica FN closure job
3506074 completed cleanly: E=-31.8455282 and -31.8323841, mean -31.8389561 with
between-replica SE 0.0065720. Relative to the established K1-FN baseline -31.8854241,
this is +0.046468 (worse). Tails disagree (-31.91894,-31.78744), so no evidence of a
physical improvement after one interleaved half-step. This does not by itself falsify the
ideal MLE loop because exact 4x4 showed non-monotone early iterations.

Before round 2, retrain the validated SR direction on both independent first-FN replicas.
The previous production checkpoint used rep1 only because rep2 was held out for validation;
after validation, a pooled natural-gradient refit is the statistically appropriate production
estimator and should reduce walker noise.

### Iteration 2 physical gate FAILS despite MLE/SR validation — 2026-10-01
Iteration-2 SR training itself looked excellent: independent FN-replica MLE gradients became
aligned, cosine +0.856823 (vs -0.852892 at iteration 0), SR held-out derivative stayed
positive (+3.0611), and the validation-optimal step shrank to eta=0.003 with held-out
log-likelihood gain +0.0072479 and pool ESS 3275.9. This is a convergence-like signal in
the amplitude-learning objective.

However the physical K1/FN gate rejects the resulting second update. The updated threshold
is T2=-28.8817117534. Matched M=128 FN replicas give
seed14001 E=-31.8059041913 and seed14002 E=-31.8248371930, hence
E_iter2=-31.8153706921 +/- 0.00946650.
This is substantially worse than both iteration 1 (-31.9012287668 +/- 0.00444094) and the
original baseline (-31.8854241096 +/- 0.00159161).

Conclusion: FN-walker MLE + SR is a viable one-step amplitude/sign improvement, but the
MLE validation objective is not sufficient to choose repeated physical updates. The outer
loop must be physically acceptance-gated (or use a smaller trust region tied to K1/FN
change); do not blindly iterate the validation-optimal MLE step.

### Iteration-2 frozen-sign attribution + trust-region scan — 2026-10-01
To isolate the second-round failure, freeze the successful iteration-1 sign rule s1 (a1,T1) and change only the guide amplitude from a1 to the iter2 SR amplitude a2. H100 seed13001 completed: E_FN(a2,s1)=-31.8464636184, versus the matched iter1 (a1,s1) seed13001 value -31.8967878286, degradation +0.0503242102 before any K1 refresh. Seed13002 job 3507698 is still running; do not call the attribution final until it completes.

Theory clarification: ten Haaf lattice FN guarantees E_FN[psi_T] is an upper bound and no higher than the variational energy of that same psi_T, but the lattice H_FN changes with trial amplitude ratios. It does not imply monotonic E_FN under repeated replacement/update of the guide. This is consistent with our exact 4x4 ideal MLE half-step, which can be nonmonotonic from a poor guide, and with recent iterative lattice-FN work (Kairon/Clark arXiv:2609.16308) showing obstruction/support-collapse behavior in non-ground-state sign chambers.

Next falsifier: keep s1 frozen and scan smaller iter2 SR steps along the exact same validated direction. Job 3508334 reconstructs the direction and saves eta={0.0003,0.001,0.003}; dependent H100 jobs 3508341 and 3508342 run matched seed13001 FN for eta=0.0003 and 0.001. eta=0.003 is already represented by frozen-s1 job 3507697. If a smaller eta restores FN descent, use a physical FN trust-region gate; if all positive steps degrade, treat the current FN-amplitude direction/map as locally non-descent in this approximate sign chamber and redesign the amplitude objective/neighbor-ratio information rather than blaming K1.

### Frozen-sign attribution closed; neighbor-support bottleneck quantified — 2026-10-01
The iteration-2 failure is now localized to the amplitude update, not the K1 refresh. With the successful iteration-1 sign rule s1 held fixed, the iter2 amplitude a2 gives matched H100 FN energies -31.8464636184 and -31.8094748149, mean -31.8279692166. The corresponding iteration-1 (a1,s1) mean is -31.9012287668, so the amplitude replacement degrades the projected energy by +0.0732595502 before any sign refresh. K1 should not be blamed for the round-2 failure.

A support audit using 96 representative a1^2 states finds 13,940 one-hop Hamiltonian edges and 13,648 unique neighbor targets. Only 0.7819% of edge targets occur in the reference pool; direct overlap with the two FN mixed-walker populations is 7.17e-5, and the union remains 0.7819%. Thus more than 99% of the local amplitude ratios used by FN/K1 are supplied by network generalization rather than directly constrained sample points. This gives a concrete mechanism by which global walker-MLE likelihood can improve while the physically decisive local ratios drift.

Next tests:
- H100 3509013: frozen s1, same iter2 SR direction at eta=0.0003.
- H100 3509014: frozen s1, same iter2 SR direction at eta=0.001.
- H100 3509077: neighbor-shell audit comparing successful a0->a1 and failed a1->a2 updates, including seen/unseen local-ratio changes, K1 margins, and flip rates.
If a smaller eta restores FN descent, adopt a physical FN trust-region acceptance gate. If not, move the amplitude objective toward neighbor/local-ratio or energy-gradient information rather than further tuning K1.

### Neighbor-shell audit result — 2026-10-01
H100 job 3509077 completed. Using 96 representative a1^2 base states gives 13,940 one-hop edges / 13,648 unique neighbor targets; only 0.7819% of edge targets are in reference support and walker-only overlap is ~7.17e-5.

Successful update a0->a1:
- seen-edge delta log[a(y)/a(x)] RMS = 0.05568
- unseen-edge RMS = 0.41753 (~7.50x seen)
- K1 flip rate = 1.406% overall, 0 on seen targets, 1.417% unseen.

Failed update a1->a2:
- seen-edge delta log-ratio RMS = 0.00747
- unseen-edge RMS = 0.14646 (~19.60x seen)
- K1 flip rate = 0.179%, 0 on seen targets, 0.181% unseen
- flips are concentrated near the K1 boundary: median pre-update margin 0.2869 for flips vs 2.867 for nonflips.

Interpretation: round-2 failure is not explained by a globally larger amplitude step; in fact its absolute unseen-edge motion is smaller than round 1. The sharper signal is a growing seen-to-unseen generalization gap as the walker-MLE objective converges. All observed node changes occur off sampled support. This motivates a neighbor/local-ratio or FN-energy-based training objective if the small-step physical trust test also fails.

Trust-region H100 jobs 3509013 (eta=0.0003) and 3509014 (eta=0.001) are still running as of this update; durable watchers remain attached.

### Iteration-2 SR trust-region test closes negative — 2026-10-01
With successful s1 frozen, smaller parameter-space steps along the exact same round-2 SR/MLE direction were tested on seed13001. Baseline (a1,s1) is E=-31.8967878286. Results: eta=0.0003 -> -31.7969390681; eta=0.001 -> -31.8105365478; eta=0.003 -> -31.8464636184 (and independent eta=0.003 seed13002 -> -31.8094748149). All tested positive steps are worse than the baseline; there is no evidence for a small physical descent window. Do not continue learning-rate scans.

Combined with the neighbor-shell audit, the working diagnosis is objective mismatch/generalization rather than step-size tuning: walker-MLE/SR improves its own held-out objective but leaves >99% of FN/K1 neighbor ratios unconstrained by direct samples, and the seen-to-unseen edge-motion ratio worsens markedly in round 2. Next route should change the amplitude objective to one that directly constrains FN/local-energy neighbor ratios (e.g. fixed-sign FN-energy / local-residual tangent-space SR), while preserving K1 as the sign extractor.

### Fixed-sign physical-energy SR test launched — 2026-10-01
After the MLE/SR trust-region direction failed at eta={3e-4,1e-3,3e-3}, stop learning-rate scans. New hypothesis: the amplitude update should be driven by the fixed-sign physical variational-energy force, which directly contains Hamiltonian-neighbor ratios and therefore supplies information on unseen neighbor states.

For psi_theta=s1*a_theta with s1 frozen:
g_k = 2 Cov_{a1^2}(O_k,E_L^H), and delta theta = -(S+lambda I)^(-1) g.
One H100 job 3509464 tests this with ntrain=nval=256, Fisher n=256, lambda=1, and a geometric trust radius chosen so RMS(delta log a)=0.05 on the training sample. There is no learning-rate sweep. The candidate is saved only if held-out fixed-sign variational energy decreases and held-out ESS remains >80%.

If this gate passes, the next test is frozen-s1 FN with the accepted amplitude. If it fails, do not tune the step size immediately; reassess whether the force must include the guide-dependence of H_FN itself or a more explicit FN/local-residual tangent projection.

### Important implementation correction: 8x8 full-network sign refresh was not current-sign recursive — 2026-10-01
Inspection of `experiments/callable_fn_loop.py` and `cluster/paderborn/fnmle8_sr_iter2_fn_h100.sbatch` shows that the full-network iteration-2 K1 refresh always reconstructed
r_M=(H a s_M)/(a s_M),  s_new=s_M sign(T-r_M).
Thus the 8x8 iteration-2 sign step was another Marshall-referenced K1 reconstruction, not the genuine recursive map established on exact 4x4:
r_k=(H a s_k)/(a s_k),  s_{k+1}=s_k sign(T-r_k).

This does NOT undo the amplitude attribution: the separate frozen-s1 experiment proved that a2 already worsens FN before any sign refresh. But the production 8x8 loop has not yet tested the true current-sign recursion and must not be described as having done so.

Exact 4x4 controls:
- one fixed-sign energy/natural-gradient amplitude step followed by exact energy-optimal current-sign K1 reaches exact signs by iteration 28 and E-E0=2.12e-7 by iteration 48 when the amplitude trust step is physically acceptance-gated;
- the same amplitude evolution with production-style weighted two-means K1 stays in the Marshall sign chamber (O_sign=0.974538) while the amplitude optimizes inside that wrong chamber.
Therefore two-means is not adequate for the recursive sign step in this control.

Next sign implementation: sampled fixed-amplitude energy minimization over the one-dimensional current-sign K1 threshold family. This is label-free and scalable: each sampled Hamiltonian edge contributes a sign-dependent term that flips only while the threshold lies between r_k(x) and r_k(y), so the full threshold-energy curve can be accumulated in O(N_edges + N_nodes log N_nodes).

### Current-sign K1 implementation correction + scalable threshold regression — 2026-10-01
Important correction: the 8x8 production K1 engine in experiments/callable_fn_loop.py reconstructs signs as s_M*sign(T-r_M), so the earlier full-network 8x8 refresh was Marshall-referenced rather than the genuine recursive map r_k=H(a s_k)/(a s_k), s_{k+1}=s_k sign(T-r_k). This does not alter the frozen-s1 amplitude attribution, but it means true recursive K1 has not yet been tested at 8x8.

A scalable sampled energy-optimal current-sign threshold implementation now exists in experiments/current_sign_energy_threshold8.py. Its O(N_edges + N_nodes log N_nodes) threshold-energy accumulator was regression-tested against the exact 4x4 projected-Krylov optimizer after a nontrivial amplitude step: both return T=2.223832698769076, exactly the same sign pattern, and energies agree to ~3e-15.

Exact 4x4 control also shows the threshold selector is essential: energy-gated amplitude steps + exact energy-optimal current-sign K1 reach exact target signs by iteration 28 and E-E0=2.12e-7 by iteration 48; replacing the sign threshold by weighted two-means leaves the sign pattern stuck in the Marshall chamber even while amplitudes improve.

H100 job 3510370 now tests sampled energy-optimal current-sign K1 on the existing 8x8 (a1,s1) state. H100 job 3509464 independently tests one fixed-sign physical-energy SR amplitude step with a geometric trust radius and held-out physical-energy gate.

### Fixed-sign physical-energy SR gate passes; FN validation launched — 2026-10-01
The new amplitude direction is driven by the fixed-sign physical variational-energy force rather than walker-MLE:
g = 2 Cov_{a1^2}(O,E_L^H), with an SR/QGT solve (shift=1) and geometric trust radius RMS(delta log a)=0.05.
H100 job 3509464 passed the held-out gate cleanly: val E -31.7420809202 -> -31.7815216591 (Delta=-0.0394407388), held-out variance 4.72187 -> 4.12518, ESS=255.57/256. The accepted checkpoint is results/fixedsign_energy_sr8_3509464/fixedsign_energy_sr.mpack.

Decisive next test is now running with the successful sign structure s1 frozen:
- H100 3511069 seed13001
- H100 3511070 seed13002
Both use the accepted physical-energy-SR amplitude and compare directly against the matched a1,s1 FN baseline (mean -31.9012287668). If FN energy improves, promote physical-energy SR as the amplitude update; if it worsens despite the held-out variational-energy improvement, the remaining mismatch is specifically between variational-energy descent and the guide-dependent lattice-FN projector.

Separately, sampled current-sign energy-optimal K1 job 3510370 OOMed before a physics result. This is an implementation/cache-expansion failure, not evidence against recursive K1; fix bounded-memory neighborhood evaluation before rerunning.

### Recursive current-sign K1 OOM fix staged — 2026-10-01
Job 3510370 OOMed because the recursive sign evaluation materialized the full nested three-hop shell at once. The current-sign equations are unchanged; experiments/current_sign_energy_threshold8.py now evaluates parent signs/current r in bounded chunks. H100 smoke 3511095 uses ntrain=nval=32 to verify memory/runtime and obtain a preliminary sampled energy-optimal threshold before scaling the sample size. Do not interpret the 32+32 physics result as decisive unless the validation signal is unexpectedly large; its primary purpose is to validate the corrected evaluator.

### Fixed-sign physical-energy SR fails at FN-projector level — 2026-10-01
The held-out variational-energy SR gate from job 3509464 did not transfer to the guide-dependent lattice-FN projector. With s1 frozen, H100 jobs 3511069/3511070 give E_FN=-31.8351413999 and -31.8649168824, mean -31.8500291411 with two-replica SE 0.0148877412. The matched a1,s1 baseline is -31.9012287668 +/- 0.00444094, so the accepted variational-energy step worsens FN by +0.0511996257 before any sign refresh.

Conclusion: neighbour-aware physical variational-energy descent is still the wrong amplitude objective. The next amplitude force must include the guide dependence of H_FN itself; this points to the true lattice-FN Hellmann-Feynman gradient rather than more MLE or ordinary H-energy tuning.

### Exact 4x4 E_FN Hellmann-Feynman + SR control — 2026-10-01
Implemented experiments/efn_hf_sr_exact4x4.py. For violating edges K_xy=s_x H_xy s_y>0 and FN ground modulus phi, the exact guide log-amplitude derivative is
dE_FN/dtheta_k = sum_x phi_x^2 sum_y K_xy a_y/a_x [O_k(y)-O_k(x)].
A random-direction finite-difference check agrees to 7.59e-10 absolute.

At the same trust radius RMS(delta log a)=0.05 from the Marshall-sign/J2=0-amplitude control, the plain full log-amplitude gradient lowers E_FN by -0.0348120, while the full Fisher/natural-gradient direction lowers it by -0.0607300, about 1.74x larger descent, with comparable max |delta log a| (~0.30). This independently supports QGT/SR preconditioning of the genuine E_FN force. Result: results/efn_hf_sr_exact4x4.json. The result/formula were posted to Tim in the Research Workspace.

### Pure-measure estimator + recursive-K1 sampler status — 2026-10-01
Forward-walking descendant weighting was tested as a way to estimate the phi_FN^2 measure required by the exact E_FN gradient from mixed GFMC walkers. On exact 4x4 with M=20000, mixed walkers give gradient cosine 0.4056 to the exact force; forward walking improves to 0.5734 at 200 steps and peaks at 0.6203 at 500 steps, then falls to 0.5913 at 1000 as descendant noise grows. Relative gradient error remains >1 throughout. This is not credible for the production M=128 regime; do not promote naive forward walking as the 8x8 gradient estimator. Result: results/forward_walking_efn_gradient_exact4x4.json.

Recursive current-sign K1 smoke 3511095 confirms the bounded-memory three-hop evaluator works (no OOM), but its old alpha=1.2 proposal/reweighting produced validation ESS=4.06/32 and is statistically unusable. experiments/current_sign_energy_threshold8.py now samples directly from the handoff physical a1^2 pool (pool_states/pool_weights), so train/val energy samples are equal-weight. The exact 4x4 threshold regression remains exact to ~3e-15. H100 job 3515830 (64 train + 64 val, 4h) is running under durable watcher 20261001-181145-428.

### Mixed-walker E_FN force estimator — 2026-10-01
For fixed signs, the exact guide force is <F_theta>_{phi_FN^2}, with F_theta(x)=sum_{viol y} K_xy a(y)/a(x)[O(y)-O(x)]. Using the extrapolated pure identity phi^2 = 2 a phi - a^2 + O((phi-a)^2), and the exact pairwise cancellation <F_theta>_{a^2}=0, gives g_EFN = 2 <F_theta>_{mixed a phi} + O((phi-a)^2). Exact 4x4 full-space test: mixed-force cosine to exact HF force 0.97231 initially and 0.99551 after two ideal MLE half-refreshes; matched full-coordinate natural-gradient trust RMS(dlog a)=0.05 gives Delta E_FN=-0.058395 vs exact-force -0.060730. However direct finite sampling in the full log-amplitude coordinate is extremely noisy: at M=128 mean cosine ~0.032 (100 replicas), rising only to ~0.286 at M=8192. Therefore do not launch 8x8 from the raw estimator alone; the decisive unresolved test is whether the NN tangent-space + QGT projection suppresses this sampling noise. Tim was informed in Research Workspace seq 41 with exact definitions and the caveat that the 0.972/0.995 values are deterministic full-space, not finite-walker results.

### Recursive current-sign K1 physical-a1^2 threshold: 64+64 validation fails — 2026-10-02
H100 job 3515830 completed successfully (01:27:17). With 64 a1^2-resampled train and 64 validation states, the sampled energy-optimal recursive current-sign threshold selected T=-2.8900222642. Training energy changed -31.97365308 -> -31.98073272 (apparent gain -0.00707964), but independent validation changed -31.52634090 -> -31.42198936, i.e. worsened by +0.10435154 with delta SE 0.10193363 (about 1.02 sigma in the wrong direction), ESS=64.

The diagnostic change_mass_global_aware=0 means the candidate multiplier is globally constant on the sampled validation base states; neighbor multipliers are not necessarily constant, which explains a nonzero edge-energy change. Thus the 64+64 sampled threshold does not validate a useful recursive-K1 update and appears dominated by threshold overfit/noise. This does not falsify recursive K1 itself; it falsifies this small-sample threshold estimate as a production update. No larger rerun launched automatically.

### 8x8 mixed-E_FN tangent-space/QGT gate queued — 2026-10-02
The active amplitude test is H100 Slurm job 3523616 using `experiments/mixed_efn_force_sr8.py` with the successful a1 guide and fixed s1. It forms the mixed-walker sign-flip-potential force on two independent FN replicas, solves the NN tangent-space QGT/SR system from replica 1, and requires transfer to replica 2 plus guide-pool ESS >80% before saving a candidate. Raw full-coordinate finite-walker force remains too noisy; this tangent-space transfer is the decisive gate.

As of the latest inspection, job 3523616 is PENDING for scheduler Priority with no stdout/result yet. The redundant untagged local watcher was removed; the remaining watcher uses project tag `J1J2_FN_LOOP` and description `mixed E_FN force + QGT 8x8 gate`.

Do not extend the amplitude branch solely from the internal training gate. Before any frozen-s1 FN validation is submitted, check Tim's Research Workspace reply to the seq45 question asking whether he already tested this same finite-walker mixed-V_sf/E_FN + QGT estimator. The workspace service endpoint was unavailable during this resume, and the durable conversation state read through seq46 contains no recorded post-seq45 Tim answer, so no duplication claim is currently justified.

### Clarification: 3523616 did not fail — 2026-10-02
A failure email seen after watcher cleanup referred to local Chatty watcher job `20261002-073621-40922`, which was intentionally terminated while removing a redundant untagged watcher; it exited -15/SIGTERM. The actual Paderborn Slurm physics job 3523616 remains PENDING (Priority), ExitCode 0:0, runtime 0. The correctly tagged `J1J2_FN_LOOP` watcher remains active. Do not interpret that watcher-cleanup email as a physics/job failure.

### Tim/Claude finite-walker mixed-force result recovered — 2026-10-02
Research Workspace seq48 directly answers the duplication question. On Tim's 4x4 guide, the exact mixed-force identities are confirmed, but the plain finite-walker mixed estimator is noise-dominated: estimated walkers for cosine 0.7 grow from ~8e4 at a Marshall-like start to ~2.5e8 near the fixed point; at 644 walkers the cosine is ~0.03. A covariance estimator using exact w=phi/a helps strongly, but fitted-w bias reappears near the fixed point. One-hop neighbour coverage remains crucial. Tim has NOT run QGT/SR preconditioning of the finite-walker mixed force in network tangent space, nor its off-support behavior.

Therefore Slurm 3523616 is not a duplicate. It is an intentionally harsh production-scale test on refreshed a1,s1 with nforce=128 per independent replica: raw cross-replica gradient agreement may be poor, but the decisive quantity is whether the QGT/SR direction trained on replica 1 has the correct directional derivative on replica 2 and passes trust/ESS gates. If it passes, validate with frozen-s1 FN energy; if it fails, this strongly supports finite-walker signal-to-noise as the amplitude bottleneck and argues against merely increasing walker count near the fixed point.

### Matched pre-refresh mixed-force/QGT control submitted — 2026-10-02
Tim/Claude seq48 shows that raw finite-walker mixed-force SNR collapses as the guide approaches the FN fixed point, while the NN-tangent/QGT version remains untested. To distinguish a generic failure of tangent/QGT projection from a near-fixed-point signal-collapse failure, keep the production refreshed-guide gate 3523616 (a1,s1, nforce=128/replica) and run a matched earlier-guide control using the original 8x8 ViT amplitude a0, K1 threshold T0=-28.37107876288694, and the two existing independent K1-FN mixed populations seed10501/10502.

Control Slurm job: 3523738 (`mixed_efn_force_sr8_a0_h100.sbatch`), nforce=128, Fisher n=256, shift=1, same trust/ESS acceptance logic. The estimator script is now backward-compatible with explicit threshold and pool selection; 3523616's existing invocation is unchanged. Both jobs have `J1J2_FN_LOOP` watchers. Paderborn watcher status queries were corrected to use ws1, where the rootless PC2 VPN actually runs.

Decision rule: if a0 passes cross-replica SR transfer while a1 fails, finite-walker signal collapse near the fixed point is the bottleneck; do not cure it by brute-force M scaling. If both fail, tangent/QGT projection itself does not rescue M=128 and the mixed-force route needs variance reduction / a different estimator. If a1 passes, immediately perform frozen-s1 FN validation before any sign refresh. If both pass, prioritize the a1 frozen-s1 physical gate.

### Mixed-force 8x8 implementation OOM and corrected reruns — 2026-10-02
Original refreshed-guide gate 3523616 started and reached edge construction successfully (128 bases/replica; 5625 and 5596 violating edges; mean ~43.9 bad edges/base; ~2.52M amplitude evaluations for K1 signs) but FAILED after 25:53 from H100 GPU OOM during the monolithic reverse-mode force gradient: JAX attempted a ~23.67 GiB allocation with the graph already at ~69 GiB. This is an implementation/resource failure, not a physics rejection of the estimator. Pending control 3523738 was proactively cancelled because it used the same unsafe gradient path.

`mixed_efn_force_sr8.py` now accumulates the mathematically identical mean V_sf force/gradient in blocks of 8 base states and can checkpoint/load the expensive violating-edge data. Line-search force values use the same block decomposition. Reruns have 2h walltime and copy the edge cache on exit: 3537066 = refreshed a1,s1 gate; 3537067 = matched pre-refresh a0,s0 control. Both use nforce=128, Fisher n=256, shift=1 and J1J2_FN_LOOP watchers via ws1. Do not count 3523616 or cancelled 3523738 as physics outcomes.


### 8x8 mixed E_FN force/QGT gate result + physical validation — 2026-10-02
Corrected H100 runs completed successfully:
- 3537066 refreshed a1,s1: PASS. Raw gradient cosine 0.7951. QGT/SR derivatives g1·S^-1g1=125.739 and g2·S^-1g1=13.708, so descent along -S^-1g1 transfers to independent replica. Trust 0.01 accepted: replica1 Vsf 5.34026 -> 5.20821 (Delta=-0.13205), replica2 5.36237 -> 5.34752 (Delta=-0.01485), pool ESS 3366.3, dlog RMS 0.07635.
- 3537067 matched pre-refresh a0,s0 control: FAIL. Raw gradient cosine is misleadingly high (0.9713), but held-out QGT/SR derivative g2·S^-1g1=-0.1765, and trust 0.01 improves train Vsf 5.66328 -> 5.55127 while worsening validation 5.29033 -> 5.33900. No trust point passes.
Thus NN/QGT projection is not summarized by raw force cosine; the refreshed a1 direction passes the predeclared cross-replica gate while a0 does not. This is not yet a physical FN success.

Per predeclared rule, immediately launched frozen-s1 physical FN validation of the accepted 3537066 checkpoint:
- 3549023 seed13001
- 3549024 seed13002
Both compare the accepted amplitude against the same frozen s1 sign rule/T1=-28.824166903057147 using the existing frozen_s1_a2_fn8 harness, with J1J2_FN_LOOP watchers. Only if the two-replica physical E_FN improves over baseline a1,s1 (~-31.9012287668) should the sampled-HF amplitude branch remain active.

Separately, Tim/Claude seq72-74 clarifies the next fallback if physical validation fails: naive frozen-H_FN VMC with a generic positive NQS still inherits a near-node local-energy tail, but factorized node-preserving a_new=a_k*exp(r_theta) gives finite-variance node-preserving gradients in his analysis. Exact 4x4 size refresh is robust; GCNN residual-ratio fitting retains 93-97% of refresh gain early but only 27-38% near fixed point, suggesting representation becomes the late-stage bottleneck. Priority fallback test is exact 4x4 early/late frozen-H_FN VMC with the factorized residual ansatz, not further brute-force walker-count scans.


### Mixed E_FN/QGT physical gate fails — 2026-10-02
Frozen-s1 physical validation of the accepted refreshed-guide mixed-force/QGT step (3537066) completed:
- 3549023 seed13001: E_FN=-31.8450719385
- 3549024 seed13002: E_FN=-31.8651315232
Mean = -31.8551017309 with two-replica SE proxy 0.0100297924.
Matched a1,s1 baseline is -31.9012287668 +/- 0.00444094, so the accepted amplitude worsens physical E_FN by +0.0461270359 despite passing the independent-replica Vsf/QGT surrogate gate. Therefore the sampled mixed-HF/QGT branch is not promoted; do not spend more compute on trust/step-size scans or brute-force walker counts.

Per the predeclared fallback, next amplitude test is the factorized node-preserving frozen-operator route: a_new(x)=a_k(x) exp(r_theta(x)), trained on frozen H_FN[a_k,s_k]. First do the cheap exact 4x4 early-vs-late guide test suggested by Tim/Claude (seq72-74), measuring local-energy tail behavior and fraction of exact size-refresh gain retained. This distinguishes estimator pathology from residual network expressivity before any 8x8 production attempt.

### Factorized frozen-H_FN exact 4x4 falsifier launched — 2026-10-02
After the mixed-HF/QGT physical gate failed, moved to the predeclared fallback. New script `experiments/frozen_hfn_factorized_vit_exact4x4.py` tests the node-preserving residual ansatz a_new=a_k*exp(r_theta) using a 4x4 ViT from the same nqsmagic ViT family as production (4 layers, d_model=60, heads=10, b=2, translationally invariant). The residual is defined as f_theta(x)-f_theta0(x), so r=0 exactly at initialization and the starting guide is preserved.

Two exact-loop guide points are used: iteration 1 (phi-weighted target log-ratio RMS ~0.108) and iteration 10 (~0.0083). For each guide the same residual architecture is trained for 1000 full-Hilbert steps by (i) exact frozen-H_FN Rayleigh energy and (ii) oracle exact log(phi_FN/a_k) regression. Metrics at 250 and 1000 steps include fidelity, phi-weighted log-ratio RMS, fraction of exact frozen-operator gain, rebuilt guide-dependent E_FN gain, and local-energy tail/variance diagnostics. This directly separates objective quality from representation capacity.

Paderborn H100 jobs:
- 3550647: early guide, exact-loop iteration 1
- 3550648: late guide, exact-loop iteration 10
Both have J1J2_FN_LOOP watchers. At submission both were pending Priority. Do not scale this route to 8x8 until this exact early/late comparison is read.

### Factorized frozen-H_FN exact 4x4 falsifier result — 2026-10-03
Both exact H100 jobs completed cleanly.
- Early guide (iter1, target log(phi/a) RMS 0.107627): frozen-H_FN residual ViT at 1000 steps reaches fidelity 0.999819 to phi_FN, recovers 97.39% of the exact frozen-Rayleigh gain, and after rebuilding H_FN recovers 98.95% of the exact size-refresh E_FN gain.
- Late guide (iter10, target RMS 0.008313): frozen-H_FN residual ViT at 1000 steps reaches fidelity 0.999693 and, after rebuilding H_FN, recovers 81.41% of the exact size-refresh E_FN gain. The direct oracle log(phi/a) regression with the same residual architecture recovers only 36.60% of the rebuilt E_FN gain.
Interpretation: the factorized frozen-operator objective is materially better than direct ratio fitting near convergence and is strongly viable early. The exact refresh map is not the problem. Late-stage residual representation/optimization is still the bottleneck; the frozen Rayleigh objective itself is not yet monotone at the late guide with current optimizer/hyperparameters, so do not scale to 8x8 blindly. Next useful work is a small exact late-guide optimizer/trust-region study, not another estimator branch.

### Project-lead decision: amplitude optimizer/sign representation — 2026-10-03
After reviewing prior angles, SR/natural gradient is now the default optimizer for the frozen-H_FN residual-amplitude inner solve. Adam is retained only as a diagnostic baseline, not a production choice. Evidence: exact 4x4 deterministic E_FN natural gradient gave 1.7445x larger improvement than plain log-amplitude gradient at matched trust; 8x8 MLE Euclidean gradients failed while matrix-free SR produced the first transferable successful step; sampled mixed-HF MinSR failure near the node was estimator-noise/heavy-tail limited, not a verdict against SR.
Euclidean trust late-guide controls 3551872/3551873 both descended monotonically but recovered only ~14% of the rebuilt exact FN refresh after 40 steps, confirming poor Euclidean parameter geometry.
Full-state exact network-QGT jobs 3551934/3551935 OOMed due JAX capturing ~13.9 GB of constants; implementation failure only.
Replacement job 3551952 uses the exact full-Hilbert frozen-H_FN energy gradient with a deterministic 1024-state MinSR/QGT metric, scans shifts {1,0.1,0.01} at each step, and accepts the direction/line-search step only by exact frozen-H_FN energy. This is the current decisive late-guide optimizer test.
Sign-representation decision after Tim seq79: do not distill K1/Jacobi signs into a phase network. Historical repeated K1 was closer to on-the-fly analytic evaluation; the old static-g amplitude handoff caused a second-shell recursion and cache explosion (~14.7M r0 states / ~625.7M ViT evaluations). Current architecture instead keeps a directly callable amplitude NQS and analytic local sign extractor so each sign query returns to one Hamiltonian-neighbour shell. Sign distillation stays parked unless analytic evaluation itself becomes the bottleneck.

### Exact full-space frozen-H_FN SR passes; 8x8 promoted — 2026-10-03
Memory-safe full-space SR controls on the exact 4x4 late guide completed:
- job 3551953, shift 0.1: 30 accepted steps; frozen-Rayleigh gain fraction 0.9988307; rebuilt E_FN refresh gain fraction 0.9970658; phi fidelity 0.9999999650; phi-weighted log-ratio RMS 1.85e-4.
- job 3551954, shift 0.01: 30 accepted steps; frozen-Rayleigh gain fraction 0.9985525; rebuilt E_FN refresh gain fraction 0.9968016; phi fidelity 0.9999999488; phi-weighted log-ratio RMS 2.25e-4.
Conclusion: the late-stage failure of Adam/direct-ratio fitting was optimizer/QGT implementation, not residual-network representability. Promote the factorized directly-callable amplitude route with frozen-H_FN SR.
8x8 pilot job 3554536 launched from successful a1,s1 checkpoint. It uses the exact starting frozen-H_FN VMC force, matrix-free SR with shifts {1,0.1}, trust RMS {0.001,0.002,0.005,0.01}, and independent held-out line-search gating on the truly frozen H_FN[a1,s1] operator. No physical FN validation is allowed unless this frozen-operator gate passes.
Sign policy remains analytic/on-the-fly; no sign-network distillation.

### Exact late-guide frozen-H_FN SR succeeds; 8x8 promotion gate launched — 2026-10-03
Proper chunked full-space SR resolves the late-guide exact 4x4 problem. Jobs 3551953 (shift 0.1) and 3551954 (shift 0.01) both completed 30/30 accepted trust-gated steps at RMS(dlog a)=0.002.
- shift 0.1: final frozen energy -8.4491093888 vs exact frozen ground -8.4491105285; 99.883% of exact frozen-Rayleigh gain recovered. Rebuilt guide-dependent E_FN -8.4506447182 vs exact-refresh -8.4506492331; 99.707% of exact rebuild gain recovered. Fidelity(phi_FN)=0.9999999650; phi-weighted log-ratio RMS=1.8547e-4.
- shift 0.01: 99.855% frozen gain and 99.680% rebuilt gain; essentially same conclusion.
Conclusion: late-stage failure of Adam/direct-ratio regression was optimizer geometry, not residual-network capacity. Promote directly callable residual amplitude + frozen-H_FN SR, default shift ~0.1. Do not use Adam as production optimizer.
8x8 one-step promotion gate submitted as H100 job 3554670. Starting point is established a1,s1. It freezes H_FN[a1,s1] (including old sign-flip diagonal), computes the exact VMC energy force on 256 a1^2-sampled train states, SR/QGT on 256 states with shift 0.1, and line-searches function-space trusts {0.002,0.005,0.01}. Acceptance requires both train and held-out 256-state frozen-operator energy descent with ESS >80%. Only an accepted checkpoint proceeds to matched frozen-s1 physical FN projector validation.
Tim was updated at workspace seq84. Sign-network distillation remains parked; analytic local sign extraction + directly callable amplitude is the chosen architecture.

### Frozen-H_FN residual-NQS + full-space SR exact 4x4 gate PASSES — 2026-10-03
Proper chunked full-space SR jobs 3551953 (shift 0.1) and 3551954 (shift 0.01) completed successfully and supersede the 1024-state MinSR proxy.
Late exact-loop guide (iteration 10, target log(phi/a) RMS 0.008313):
- shift 0.1: 30/30 accepted exact-energy-lowering SR steps; frozen-Rayleigh gain fraction 0.998831; rebuilt guide-dependent E_FN refresh gain fraction 0.997066; fidelity(phi)=0.9999999650; phi-weighted log-ratio RMS=1.85e-4.
- shift 0.01: 30/30 accepted; frozen gain fraction 0.998553; rebuilt E_FN gain fraction 0.996802; fidelity(phi)=0.9999999488; log-ratio RMS=2.25e-4.
Verdict: the late-stage failure was optimizer/QGT implementation geometry, not lack of representational capacity. Factorized directly-callable amplitude a_new=a_k*exp(r_theta), trained on frozen H_FN[a_k,s_k] with SR, reproduces essentially the full exact size refresh even near the fixed point.
Production choice: use SR diagonal shift 0.1 as default (slightly better final gain and stronger regularization than 0.01), with function-space trust RMS(delta log a)=0.002 and exact/held-out energy acceptance where available. Adam and Euclidean-gradient trust are diagnostic baselines only.
Next: scale this specific architecture to 8x8 at frozen s1. Train residual from r=0 on H_FN[a1,s1], then validate the learned amplitude by independent frozen-s1 M=128 FN replicas before any sign refresh. Only after amplitude passes physical E_FN do K1/Jacobi sign comparison/update.

### Exact late-guide frozen-H_FN + SR verdict — 2026-10-03
Chunked full-space SR jobs 3551953/3551954 completed cleanly and decisively pass the late-guide exact 4x4 gate.
- shift=0.1 (3551953): 30/30 accepted; 99.8831% of exact frozen-Rayleigh gain; 99.7066% of rebuilt guide-dependent E_FN refresh; fidelity(phi_FN)=0.9999999650; phi-weighted log-ratio RMS=1.8547e-4.
- shift=0.01 (3551954): 30/30 accepted; 99.8553% frozen gain; 99.6802% rebuilt E_FN refresh; fidelity=0.9999999488; log-ratio RMS=2.2486e-4.
Interpretation: the late-stage residual is not a hard representation ceiling for this ViT family. Adam/direct-ratio failures were primarily objective/optimizer geometry. Promote factorized warm-start a_new=a_k*exp(r_theta) trained on frozen H_FN with SR/natural gradient. Use shift~0.1 as default and function-space RMS dlog trust. Keep signs analytic/on-the-fly; no sign distillation.
Next: 8x8 sampled frozen-H_FN SR pilot on the established a1,s1 checkpoint, with independent train/validation samples and physical frozen-s1 FN validation before any Krylov sign refresh.

### 8x8 frozen-H_FN SR promotion state — 2026-10-03
Exact 4x4 chunked full-space SR is now decisive: shift 0.1 recovers 99.883% of exact frozen-Rayleigh gain and 99.707% of rebuilt E_FN refresh; shift 0.01 is essentially identical. Production choice remains shift 0.1.
Authoritative 8x8 promotion gate is H100 job 3554536 using the stricter frozen_hfn_sr8_pilot.py: frozen H_FN[a1,s1], shifts {1,0.1}, trusts {0.001,0.002,0.005,0.01}, independent train/validation frozen-operator energy gate, ESS requirement, warm-started ViT residual.
Queued duplicate 3554670 was cancelled before start to avoid wasting an H100 slot.
A matched frozen-s1 physical validation sbatch is staged as cluster/paderborn/frozen_hfn_sr8_physical_validate_h100.sbatch but must NOT be submitted unless 3554536 passes and produces frozen_hfn_sr8.mpack. If it passes, run two independent M=128 seeds against the established a1,s1 baseline before any sign refresh.
At last check 3554536 is RUNNING on gpu1004, elapsed 1:06:28.

### 8x8 frozen-H_FN residual-SR pilot launched — 2026-10-03
Exact 4x4 chunked full-space SR gate passed essentially exactly; production default is SR shift 0.1 with factorized callable amplitude a_new=a_k*exp(r_theta).
First 8x8 pilot job 3554836 uses the successful a1,s1 state and constructs the genuinely FROZEN operator H_FN[a1,s1]: sign-flip diagonal is frozen from a1, while candidate local energy varies only through kept stoquastic offdiagonal ratios.
Training uses one sampled SR direction from 256 a1^2-distributed train states, 256 held-out states, Fisher/QGT n=256, shift=0.1. It scans function-space trust RMS(delta log a)={0.001,0.002,0.004}. A candidate is saved only if train and held-out frozen-H_FN energies both decrease and ESS remains >90%.
If accepted, immediately run two independent frozen-s1 M=128 FN projector replicas against the matched a1,s1 baseline before any K1/Jacobi sign refresh. No iterative 8x8 training is allowed before that physical gate.
At submission job 3554836 is PENDING (Priority); J1J2_FN_LOOP watcher attached.

### 8x8 factorized frozen-H_FN SR pilot launched — 2026-10-03
Exact 4x4 late-guide full-space chunked SR passed essentially exactly, so the branch is promoted to sampled 8x8.
Production design deliberately differs from the old one-step physical-energy SR test: warm-start from validated a1,s1, freeze H_FN[a1,s1] once, then take up to five small SR steps with shift=0.1 and RMS delta-log-amplitude trust=0.002. The frozen diagonal and allowed edges remain fixed across inner steps; trial local energies use current amplitude ratios against that fixed operator.
Fresh a1^2 samples are drawn with three independent seeds: 512 train, 512 val1, 512 val2. Each step is accepted only if frozen-operator energy decreases on train AND both held-out sets and ESS stays >80% on all three. No sign refresh is allowed during this pilot.
Paderborn H100 job 3554843, 128 GB, 4 h. If it passes, next action is two-replica frozen-s1 GFMC physical validation of the saved amplitude checkpoint before any K1/Jacobi sign refresh. If it fails, do not tune Adam or MLE; diagnose sampled SR force/QGT variance at fixed frozen operator.

### 8x8 frozen-H_FN + SR pilot launched — 2026-10-03
Exact 4x4 late-guide full-space chunked SR decisively passed: shift 0.1 recovered 99.883% of frozen Rayleigh gain and 99.707% of rebuilt E_FN refresh; shift 0.01 was essentially identical. This promotes the factorized residual amplitude route to 8x8.
Pilot job 3554893 uses the established successful a1,s1 checkpoint. H_FN[a1,s1] is frozen once. A directly callable ViT residual is trained for up to 5 SR steps with shift 0.1 and function-space RMS dlog trust 0.002. Three independent fresh a1^2 samples of 512 states are used (train, val1, val2). Every accepted step must lower the frozen-operator energy on all three sets and keep each ESS above 80%. No sign refresh is allowed during this gate.
If this sample-level gate passes, the saved candidate must next pass matched two-replica frozen-s1 GFMC against the established a1,s1 baseline E_FN=-31.9012287668 before any Krylov/Jacobi sign update.

### 8x8 frozen-H_FN factorized SR pilot launched — 2026-10-03
Exact 4x4 late-guide chunked full-space SR passed decisively: shift 0.1 recovered 99.883% of frozen-Rayleigh gain and 99.707% of rebuilt E_FN refresh; shift 0.01 was nearly identical. Promote frozen-H_FN + factorized warm-start + SR.
8x8 pilot job 3554908 starts from the validated a1,s1 checkpoint (vit8_fnmle_sr_eta001.mpack, T1=-28.824166903057147). H_FN[a1,s1] is frozen once. Trial amplitudes are a_theta=a1*exp(delta_theta); signs are not refreshed during the pilot.
Sampling/optimizer: fresh independent a1^2 sets, 256 configs each for train/val1/val2; 3 SR steps max; shift=0.1; function-space trust RMS dlog=0.002. Every accepted step must lower the frozen-operator energy on train, val1, and val2 and maintain ESS >80% on all three.
Acceptance rule: only if the sampled frozen-operator gate passes do we promote the saved amplitude to an independent two-replica frozen-s1 GFMC physical validation. No K1/Jacobi sign refresh before that physical amplitude-only gate.
Watcher tag J1J2_FN_LOOP attached.

### Ledinauskas-Anisimovas fixed-target ITE audit — 2026-10-03
Tested their 2023 fixed-target imaginary-time idea as a direct adaptation to our current inner problem: replace physical H by frozen H_FN[a_k,s_k], keep sign fixed, form the adaptive Euler target t=(I-dtau H_FN)b, and train the same residual ViT with exact full-Hilbert overlap loss while holding the target fixed.
At the late exact 4x4 guide (iter10), the target construction itself is excellent: adaptive dt=0.04987565 gives target energy -8.4488820523 from Ebase=-8.4481358359, capturing 76.56% of the exact frozen-HFN gain to Efn=-8.4491105285 in one Euler step. The current guide and target are already almost identical: normalized squared overlap exp(-L0)=0.999973816.
However the Euclidean/Adam fit is badly conditioned for this warm-start residual ViT. First-epoch trace job 3559202 shows that even lr=1e-5 raises energy on the first step to -8.36867 and after 20 fixed-target Adam steps remains above the starting energy (-8.44337). Larger lr values 5e-5,2e-4,1e-3 are dramatically worse. Thus the paper's successful Adam advantage does not transfer to our near-converged parametrization; Adam's per-parameter normalization overshoots a target that is already extremely close.
Long 1000-step controls 3558935/3558936 were cancelled to save H100 time after this decisive trace. Project conclusion: retain the physical insight that a fixed Euler target is a good local target, but do not replace frozen-HFN+SR with their Adam fitting scheme. SR remains the production optimizer; target-ITE is parked unless a future need specifically motivates a metric-free optimizer.

### Ledinauskas–Anisimovas frozen-H_FN control — 2026-10-03
Purpose: test only the part relevant to our amplitude inner solve: adaptive Euler ITE target training as a possible cheaper alternative to SR. Sign remains frozen; no sign learning is introduced.
Important correction: initial jobs 3559490/3559498 incorrectly held each target for 50 Adam steps unconditionally and were cancelled. The paper's actual rule is to keep the target fixed only until the optimized NQS energy falls below E_old-sigma_E, then refresh it.
Faithful exact-4x4 adaptation now uses exact frozen H_FN, overlap loss, adaptive optimal Euler dt, sigma_E=sqrt(Var(E_loc)/1e5) to match the paper's 1e5 energy-statistics sample scale, and energy-triggered target refresh. Jobs 3560026 (lr=1e-3) and 3560054 (lr=2e-4) are running.
The exact first Euler target itself is promising: target E=-8.44888205 from Ebase=-8.44813584, i.e. ~76.5% of the exact frozen refresh gain before any network-fitting error. The test therefore isolates whether cheap Euclidean/Adam fitting can realize the target reliably enough to compete with SR.

### 4x4 ED benchmark protocol for scalable FN-amplitude learning — 2026-10-03
Goal of ED is not merely to show a network can fit phi_FN. It is to identify a training objective and scalable validation metrics that still make sense at 8x8+.
Candidate currently running: direct residual target delta(x)=log[a_T(x)/a(x)]=log|R(x)| for one frozen-H_FN Euler step, trained as weighted pointwise MSE under a^2.
Evaluation hierarchy:
1) TRAINING OBJECTIVE (must scale): pointwise residual loss on sampled x. Do not optimize against exact phi_FN.
2) DOWNSTREAM-RELEVANT DIAGNOSTIC (must scale): Hamiltonian-edge difference error in delta(y)-delta(x), since FN/Krylov only needs updated log-amplitude ratios. This is more important than global overlap.
3) INNER PHYSICS GATE (must scale): frozen-H_FN Rayleigh/local-energy estimate on independent held-out samples, with variance/ESS.
4) OUTER PHYSICS GATE (must scale): rebuild H_FN with candidate guide and measure independent FN/GFMC energy. This remains the acceptance criterion.
5) ED-ONLY ORACLES (diagnostic only): exact phi_FN fidelity, exact log-ratio RMS, exact fractions of frozen and rebuilt refresh gain. Use these only to learn which scalable metrics correlate with truth.
Decision rule: promote a learning objective only if low held-out residual/edge error predicts lower frozen energy and lower rebuilt E_FN. If pointwise delta MSE fits well but misses edge differences or physical gates, next control is an explicit Hamiltonian-edge loss on delta(y)-delta(x), not broader hyperparameter tuning.

### Finite-T CTQMC amplitude-relearning gate — 2026-10-03
Finite-T implementation is intentionally parked before finer CTQMC integration. The active gate is callable amplitude relearning after a short K2 block. Reuse the shared factorized NQS/SR machinery from the ground-state FN branch; do not run a separate architecture search. Paderborn A40 job 3565142 is the current discriminator: 20-site reconstructed M=100k guide, dt=.0125, factorized PairGNN residual, 20 nonlinear overlap-SR steps. PASS requires improved one-block K2 target fidelity and improved blind Hamiltonian-neighbor log-ratio RMSE. If it passes, next is repeated exact-block -> learned-refresh recursion to beta>3, then sampled/walker SR. If it fails, park finite-T compression and return the failure to the shared amplitude-learning branch.

### Finite-T / CTQMC amplitude-relearning subangle PARKED — 2026-10-03
Final discriminator 3565142 completed. Factorized nonlinear overlap-SR on the exact 20-site one-block K2 target did not pass promotion: best early target-fidelity gain was only +1.81e-5, blind edge RMSE improved only ~0.27% at that point; after 20 steps edge RMSE improved only ~0.61% while physical fidelity fell from the sign-only/unchanged-amplitude baseline 0.855531 to 0.851822. Exact K2 target physical fidelity is 0.859770. Verdict: remaining bottleneck is callable positive-amplitude relearning with accurate Hamiltonian-neighbour ratios, not the adaptive K1/K2 sign mechanism. Park finer CTQMC implementation and all independent finite-T compressor searches. Reopen only after the shared FN amplitude-learning branch produces a materially better scalable callable-amplitude handoff. Closure note: results/finite_tau_amplitude_relearning_subangle_closure_2026-10-03.md. No active finite-T jobs.

### Edge-residual benchmark + 8x8 SR OOM repair — 2026-10-03
4x4 direct pointwise delta=log|R| regression result: exact one-step ITE target gives 78.85% of exact rebuilt-FN refresh, while best pointwise residual regression (lr=1e-4) gives 45.90%. Global target fidelity 0.999888 is therefore not a sufficient proxy for FN physics.
Next benchmark locked before seeing results: train Hamiltonian-edge differences [r(y)-r(x)]-[delta_T(y)-delta_T(x)] with scalable weights proportional to |H_xy| a(x)a(y). Two exact-4x4 controls submitted: job 3565720 physical-H edges; job 3565721 frozen-H_FN allowed edges; both lr=1e-4, same residual ViT. Compare held-out/scalable edge errors against exact frozen/rebuilt FN gain.
8x8 sampled frozen-H_FN SR pilot 3554914 OOM after 2 accepted transferable steps. Both accepted steps lowered train, val1, val2 frozen-energy estimates with high ESS. OOM was implementation-side: full-batch VJP closure was JIT-captured as ~4.32GB constants during QGT lowering. Edge cache survived at results/frozen_hfn_sr8_3554914/frozen_hfn_sr8_edges.npz.
Repair: frozen_hfn_sr8_pilot_chunked.py loads exact saved train/val samples+frozen edges and replaces only the Fisher-vector product with 64-state JVP/VJP chunks plus explicit CG. Gradient, frozen operator, shift=.1, trust=.002, line search, ESS and dual-validation gates unchanged. Retry job 3565752 submitted, 5 steps, cache reused.

### 4x4 Hamiltonian-edge residual result — 2026-10-03
Edge-difference training is materially better aligned with rebuilt FN physics than pointwise delta regression, but Adam still overshoots if run blindly.
Physical-H edge loss (job 3565720, lr=1e-4): best logged checkpoint step 250 had physical-edge MSE 2.0148e-4, frozen-H_FN energy -8.448403601 (27.47% of exact frozen refresh), rebuilt E_FN -8.450124995 (65.93% of exact rebuilt refresh). At step 1000 Adam drifted to 51.80% rebuilt refresh. Exact one-step target itself is 78.85%.
Frozen-H_FN-edge loss (job 3565721): best logged checkpoint step 500 had frozen-edge MSE 4.3228e-4 and rebuilt refresh 54.70%; final step 1000 drifted to 18.27%. Thus physical-H edge differences are the better target in this test.
Comparison: pointwise delta=log|R| regression best final control recovered 45.90% rebuilt refresh; physical-H edge loss improves this to 65.93% at best checkpoint. Global fidelity remains insufficient.
Interpretation: the scalable object to train is the local Hamiltonian-edge log-ratio correction. Need scalable early stopping/validation, not fixed training steps. ED oracle must not choose checkpoint at large size.
8x8 chunked SR retry 3565752 has loaded saved edge cache and completed first chunked CG solve without prior multi-GB XLA constant capture; still running.

### Physical-H amplitude -> current-sign K1 A/B check — 2026-10-03
Re-audit found this route already has strong exact-4x4 evidence: experiments/fixedsign_energy_trust_k1_exact4x4.py performs physically acceptance-gated fixed-sign H-energy amplitude descent followed by exact energy-optimal current-sign K1 each iteration. It reaches exact signs at iteration 28 (O_sign=1, exact sign hash cf9515a52383) and E-E0=2.12e-7 by iteration 48. Thus physical-H amplitude -> Krylov is a genuine viable finite-size loop, not a strawman.
Existing 8x8 physical-H-trained ViT checkpoint: results/fixedsign_energy_sr8_3509464/fixedsign_energy_sr.mpack. Its held-out fixed-sign physical energy improved, although its frozen-s1 FN energy worsened; that latter fact does not answer whether it improves the *next sign update*.
Fair 8x8 test requires splitting parent signs from current amplitude: s1 must stay defined by original a1/T1, while r=(H a s1)/(a s1) uses candidate amplitude. New script current_sign_energy_threshold8_splitamp.py implements exactly this and reweights the existing a1 pool to candidate a^2.
Submitted matched jobs with ntrain=nval=128: 3566702 baseline a1 with parent s1; 3566703 physical-H-trained a_H with same parent s1. Compare independent-validation current-sign K1 energy delta/change mass. Both have J1J2_FN_LOOP watchers.
