# J1-J2 research status

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
