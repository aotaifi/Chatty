# J1-J2 finite-temperature adaptive CTQMC — 2026-10-01

## Goal
Build a genuine adaptive sign construction inside continuous-time QMC/operator histories.
No purification; no reduction to ground-state/projector FN.

## Construction tested
For endpoint sign gauge s, use the positive fixed-node generator F[g,s].
The bad-edge residual C[g]=S H S-F[g,s] obeys C[g]g=0.
Hence the scalable amplitude target is the local bad-edge ratio a_j/a_i, not signed-history cancellation.

The sign channel is updated separately by the finite-time K1 rule
s_{k+1}=sign[(I-dtau H)(s_k a_k)].
Exact vectors below are diagnostics only; guide updates use independent CT walker replicas.

## 20-site falsifiers passed
Residual target: at tau=.5 on hard column y=59279,
uniform guide ||C a||=1.12868, one-block-lag guide=.52859, exact guide ~0.
Corresponding Euler next-step fidelities: .997642, .999390, .999758.
Thus the local FN residual tracks actual propagation quality.

K1 with exact transient amplitudes tracks the moving finite-time sign extremely well:
at tau=1.0 fixed distance-sign mismatch mass=6.2248e-4,
repeated K1 mismatch=4.55e-9.

Raw endpoint-count ratios are noisy. At tau=.25, M=100k full-length CT paths give
bad-edge log-ratio RMSE ~.20-.25. Replica-only cross-entropy damping selects eta=.90;
one walker refresh improves full FN fidelity .996969 -> .999622.

## Hard tau=.5 result
Full-length weighted CT paths have low ESS (~5-6k of 100k), but replica-CV damping still works:
one refresh improves full fidelity .971753 -> .984274.
Three weighted-path refreshes reach .988880.

Replacing full-length weighting by positive CT sequential Monte Carlo with resampling
every dtau=.05 fixes the inner weight degeneracy. For M=100k under the uniform FN guide,
minimum pre-resampling block ESS is ~93k; replica fidelities to the exact FN amplitude
are .97193/.97252 and replica cross-fidelity is .94674.

Three CT-SMC guide refreshes, with damping chosen only by independent-replica validation:
iter0 .971753
iter1 .987186
iter2 .991467
final .993437 full fidelity.
Selected eta values: .55, .30, .20.
Replica cross-fidelity stays .943-.954 and minimum block ESS stays ~88.8k-93.1k.

Final K1 next-block sign mismatch from the refreshed FN amplitude is 2.09e-5,
versus 4.65e-5 from simply carrying the original distance sign to tau=.55.

## Verdict
ACCEPT: the finite-T adaptive FN <-> K1 idea survives its first genuine CTQMC implementation.
The successful inner process is positive continuous-time FN propagation + short-block SMC;
there is no released signed-history accumulation.

NEXT decisive task: actually apply the K1-updated sign to the next short CT block and
verify composed propagation across a sign-changing interval. After that, replace the
dense 20-site endpoint histogram guide by a scalable callable/local-ratio model and
measure walker-number/system-size scaling.

No cluster jobs were submitted.

## Code/results
- experiments/finite_tau_ct_local_ratio_20site.py
- experiments/finite_tau_ct_iterative_fn_krylov20.py
- experiments/finite_tau_ct_smc_fn20.py
- experiments/finite_tau_ct_smc_iterative_fn_krylov20.py
- results/finite_tau_ct_smc_iterative_fn_krylov20.json

## Composed K1 -> next CT/FN block test
The saved M=100k, tau=.5, three-refresh CT-SMC state was reconstructed exactly from its recorded seeds:
eta=.55,.30,.20 and the same block ESS values as the original run.
Final tau=.5 exact-FN fidelity was reproduced: F_full=.993436624.

A fresh pair of CT-SMC replicas under the final guide gave a sampled tau=.5 amplitude with F_amp=.981532.
A fourth replica-CV guide update chose eta=.15.
For the exact tau=.55 target, sign mismatch was:
- carry old distance sign: 4.6465e-5
- K1 from sampled/CV guide: 2.7302e-5
- K1 from exact FN amplitude: 2.0919e-5.

The same sampled tau=.5 population was propagated through one additional dtau=.05 positive FN block.
Direct sampled full fidelities at tau=.55 were:
- carry: .963316
- sampled-K1 sign: .964360
- oracle-FN-K1 sign: .963305
with ESS ~89.6k-89.7k. Different branch RNG seeds make the small full-fidelity differences noisy.

Noise-free sparse propagation from the same tau=.5 amplitude, changing only the sign constraint, resolves the comparison:
starting from the exact FN amplitude:
- carry: F_full=.991209937
- sampled-K1: .991218601
- oracle-FN-K1: .991224293.
Starting from the sampled q amplitude:
- carry: .985736337
- sampled-K1: .985744937
- oracle-FN-K1: .985750870.

Verdict: PASS. Feeding the K1-updated sign into the next positive CT/FN block is stable and gives the expected ordered improvement. This closes the immediate composition falsifier at tau=.5 -> .55. Next test is recursive multi-block composition (.55,.60,...) with sign refresh after each block.

## Recursive sampled composition tau=.5 -> 1.0
The two M=100k CT-SMC replicas were reconstructed at tau=.5 from the saved three-refresh state and then propagated recursively in dtau=.05 blocks to tau=1.0.
After every block, the amplitude guide was updated by independent-replica CV and the adaptive branch applied K1 before the next block.
A matched control branch carried the original distance sign throughout.

No instability occurred. Minimum block ESS remains about 8.87e4/1e5 through tau=1, with ~3.85e4 unique endpoints per replica at tau=1.

At tau=1:
- adaptive K1: F_full=.9050097, F_amp=.9056860, sign mismatch=3.47086e-4
- fixed sign: F_full=.8997424, F_amp=.9004369, sign mismatch=6.22481e-4.
Thus recursive K1 cuts the sign mismatch by ~44% and improves full fidelity by ~5.27e-3 absolute at tau=1.

The adaptive branch is slightly worse at tau=.60 (F_full .969778 vs .970385), but becomes better by tau=.65 and remains better thereafter; the gain is especially clear at .85-.95.
Replica cross-fidelity at tau=1 is .8121 adaptive vs .8232 fixed, so the exact-fidelity gain is not an artifact of better replica agreement.

Verdict: PASS. Repeated sampled CT-SMC/FN -> guide refresh -> K1 -> next CT-SMC/FN composition remains stable across ten short blocks through tau=1 and improves the long-time sign/full-state accuracy relative to carrying the fixed endpoint sign.
Result: krylov_sign_structure/results/finite_tau_ct_smc_recursive_k1_20site.json

## Long-beta scaling campaign: beta up to 2
The recursive sampled CT-SMC/FN -> replica-CV guide -> K1 loop was extended to beta=2 on hard y=59279.
CV was corrected to allow eta=0 (no guide update), avoiding forced accumulation of noise.
Three populations were run: M=25k,50k,100k, with matched fixed-distance-sign controls.

Adaptive full fidelities:
beta=1.0: .77976, .85365, .89667 for M=25k,50k,100k.
beta=1.25: .69293, .78643, .84638.
beta=1.5: .61840, .72219, .79560.
beta=1.75: .54517, .66832, .76267.
beta=2.0: .48368, .64337, .73199.

Fits of 1-F ~ M^-alpha give adaptive alpha:
beta=.75: .641; 1.0: .546; 1.25: .500; 1.5: .450; 1.75: .469; 2.0: .473.
Thus over beta~1-2 the population dependence is close to ordinary M^-1/2, with no visible exponential-in-M catastrophe.
Writing 1-F ~= C(beta)/sqrt(M), C rises roughly 33.4,48.3,62.4,73.7,82.0 for beta=1,1.25,1.5,1.75,2.
Over this narrow range a linear fit C ~=49.1 beta-13.6 has R2=.988; this is evidence only, not an asymptotic scaling claim.

The matched fixed-sign branch has almost identical M exponents. Adaptive K1 lowers sign mismatch but does not uniformly improve full fidelity at beta=2:
M=25k: adaptive .48368 vs fixed .47479;
M=50k: .64337 vs .62921;
M=100k: .73199 vs fixed .73889.
At beta=2 sign mismatch is adaptive .01210 vs fixed .01316 for M=100k.
Hence amplitude estimation dominates the long-beta full-state error.

Direct original-history sign baseline on the same column:
raw average sign =1.08e-4 at beta=.25; 2.24e-8 at .5; 5.46e-12 at .75; 1.38e-15 at 1.0; 3.55e-19 at 1.25; 9.58e-23 at 1.5.
Corresponding raw variance penalty 1/<sign>^2 is ~5.2e29 already at beta=1.
Endpoint-distance guided residual average sign also decays strongly: .378 at .25, .0677 at .5, .00967 at .75, .00126 at 1.0, 1.57e-4 at 1.25, 1.95e-5 at 1.5.
Thus the positive constrained loop avoids the explicit exponential signed-history variance collapse on this N=20 test, replacing it in the observed regime by ordinary population noise plus approximation bias.

Exact-transient-amplitude recursive K1 remains essentially exact through beta=2:
sign mismatch 2.11e-9 at .5, 4.55e-9 at 1.0, 6.51e-8 at 1.5, 2.78e-9 at 2.0.
Therefore the sampled beta=2 sign error is not an intrinsic K1 limitation; the bottleneck is long-beta amplitude/guide learning.

Current verdict: first concrete evidence of sign-problem mitigation, not yet a cure.
The decisive next task is to improve scalable amplitude/local-ratio learning so that its error does not grow with beta, then repeat this M-scaling across system size N. Do not spend more effort on K1 itself unless a new falsifier appears.

## Long-beta population scaling: dense-guide loop
Completed N=20 hard-column runs to beta=2 for M=25k,50k,100k.

At beta=2, adaptive full fidelities:
M=25k: .483682
M=50k: .643374
M=100k: .731990.
Fixed-sign controls:
.474789, .629205, .738886.

Adaptive sign mismatch at beta=2 improves with M only mildly:
.0128278, .0122447, .0120956,
while fixed distance-sign mismatch is .0131608.
Thus K1 keeps helping the sign channel, but at M=100k the adaptive branch has slightly worse total fidelity than fixed because amplitude/guide error dominates.

A log-log fit of beta=2 infidelity versus population gives
1-F_adapt ~ M^-0.473 and 1-F_fixed ~ M^-0.504,
i.e. ordinary ~1/sqrt(M) convergence at fixed beta.

Inverting per-beta fits to estimate M needed for F~.9 gives, adaptive:
beta=.75 ~4.0e4
1.0 ~1.0e5
1.25 ~2.3e5
1.5 ~4.9e5
1.75 ~6.4e5
2.0 ~7.8e5.
Fixed-sign estimates are of the same order and growth.
A crude exponential fit gives M_0.9 ~ exp(2.37 beta) adaptive vs exp(2.31 beta) fixed; these exponents are only indicative because beta>1 points extrapolate beyond measured M=100k.

Verdict: current dense per-configuration guide implementation does NOT yet show improved beta-scaling. The adaptive sign channel works, but the bad scaling has moved into amplitude/guide estimation.

Next falsifier launched: replace histogram guide updates after tau=.5 by a callable density-ratio guide trained from one CT-SMC replica against samples q~g, validated on the second replica, then continue K1 recursively.
LMU Slurm job 16779962, M=25k, beta=2. Dense-guide baseline at same M is F=.483682.

## Callable-guide long-beta pilot
LMU job 16779962 completed successfully for M=25k, beta=2.

Replacing dense histogram guide updates after tau=.5 by a callable replica-trained density-ratio guide gives:
F_full(beta=2): .577602 versus dense-guide .483682 at the same M=25k.
Absolute gain +.09392.
The gain grows with beta: +.0077 at .75, +.0364 at 1.0, +.0544 at 1.25, +.0697 at 1.5, +.0859 at 1.75, +.0939 at 2.0.
Sign mismatch at beta=2 also improves modestly: .0124345 callable vs .0128278 dense.

Using the dense beta=2 population fit 1-F ~ M^-0.473, callable M=25k corresponds to about M~37k dense walkers (~1.5x effective saving).
Late-beta validation AUC is only ~.52-.53, so the residual correction is weak but still materially improves the amplitude channel.

This is promising but one population is insufficient to claim improved beta scaling.
Submitted callable scaling array job 16780012:
task 0 M=50k, task 1 M=100k, both beta=2.
Mail notifications use a.otaifi@lmu.de.

## Expanded long-beta scaling matrix
User approved using available LMU Slurm resources for a stronger answer.

Existing callable scaling job 16780012:
- M=50k,100k at beta=2.

New matrix array job 16780017 (19 tasks, max 12 concurrent):
- callable adaptive: M=25k,50k,100k,200k at beta=2.5
- callable adaptive: M=25k,50k,100k,200k at beta=3.0
- independent-seed callable replicas: M=25k,50k,100k at beta=2
- dense-guide controls: M=50k,100k,200k at beta=2.5; M=100k,200k at beta=3.0
- callable-guide / fixed-sign controls (K1 disabled): M=50k,100k at beta=2.5 and M=100k at beta=3.0.

This matrix is designed to distinguish population scaling, deep-beta behavior, seed variance, and the separate contributions of amplitude compression versus K1 sign adaptation.
LMU email notifications use a.otaifi@lmu.de.

## Ratio-native FN/K1 attack
After the long-beta matrix showed that global callable amplitude compression develops model bias and that K1 can improve sign mismatch while worsening full fidelity, the next formulation targets only the local ratios rho_xy=g_y/g_x that enter both FN and K1.

FN is evaluated ratio-native:
Delta F_x = sum_bad H_xy rho_xy.
K1 is also ratio-native:
L_x/g_x = s_x(1-dtau H_xx) - dtau sum_y H_xy s_y rho_xy.

New experiment:
krylov_sign_structure/experiments/finite_tau_ct_smc_edgeratio_longbeta20.py
Learns a locally integrable residual potential f(x) from sampled Hamiltonian edges so f(y)-f(x) estimates the residual log-ratio
log[a(y)/a(x)] - log[g(y)/g(x)].
Replica B chooses eta by edge-ratio validation; low-information smoke test at M=1000 correctly selected eta=0.

LMU Slurm array 16780298 submitted and running:
- M=25k beta=2.5
- M=50k beta=2.5
- M=100k beta=2.5
- M=100k beta=3.0
Mail notifications: a.otaifi@lmu.de.

## Edge-ratio histogram estimator falsified; oracle ratio gap
All edge-ratio histogram runs completed.
At beta=2.5:
M=25k F=.274708
M=50k F=.339390
M=100k F=.434253
At beta=3.0, M=100k F=.348447.
These are substantially worse than both dense-guide and global-callable baselines.

The diagnostic is decisive: replica edge RMSE barely improves over the zero-correction baseline (e.g. M=100k, beta=2.5 final .693074 -> .692144), yet nonzero eta updates accumulate and degrade the guide. Raw count quotients on neighboring configurations are therefore too noisy and should be rejected as the local-ratio estimator.

Crucially, exact-ratio oracle runs show the ratio-native FN/K1 formulation itself is strong:
beta=3.0: F=.6911 (25k), .8186 (50k), .8982 (100k), .9432 (200k).
beta=5.0: F=.7972 (50k), .8865 (100k).
Thus the remaining bottleneck is estimating local ratios from samples, not the FN/K1 ratio-native architecture.

Next estimator: pairwise local pseudolikelihood / ratio matching using only walkers x~a and Hamiltonian neighbors y. No histogram quotient is formed. Model uses ell=log g+f_theta and minimizes E_x sum_y log(1+exp(ell_y-ell_x)); replica B selects eta. This objective has the desired local-ratio optimum on the undirected configuration graph.

Submitted LMU array 16780379:
beta=2.5 at M=25k,50k,100k
beta=3.0 at M=50k,100k,200k
All six tasks running. Custom concise per-task STARTED/FINISHED/FAILED mail enabled; Slurm verbose mail disabled.

## Pairwise ratio estimator: first long-beta result and data-scaling follow-up
Pairwise local pseudolikelihood removes histogram quotients and learns local guide corrections from walkers x~a and Hamiltonian neighbors y.

Completed array 16780379:
beta=2.5: F=.65848 (25k), .72534 (50k), .77029 (100k).
beta=3.0: F=.70926 (50k), .73792 (100k), .74070 (200k).
This substantially outperforms the global-callable branch and beats dense at matched 50k/100k around beta=2.5. At beta=3 it is competitive at 100k but saturates by 200k, still far below exact-ratio oracle (.8186,.8982,.9432 for 50k,100k,200k).

Important implementation diagnosis: all these runs capped pairwise training at nfit=20k walkers with K=4, i.e. exactly 80k training pairs per block independent of total M. Thus extra walkers above 20k did not feed the ratio learner, providing a concrete explanation for the observed M-saturation.

Added exact edge-ratio diagnostics (evaluation only): weighted RMSE/correlation for current-guide ratios and updated ratios against exact N=20 amplitudes on held-out sampled edges.

Submitted data-scaling array 16780485, all six tasks running:
- M=100k,beta=3,nfit=50k,K4,e3
- M=100k,beta=3,nfit=100k,K4,e2
- M=200k,beta=3,nfit=50k,K4,e3
- M=200k,beta=3,nfit=100k,K4,e2
- M=200k,beta=3,nfit=200k,K2,e2
- M=100k,beta=2.5,nfit=100k,K4,e2
Optimizer example-passes are kept roughly comparable while unique walker coverage grows.

## Pairwise data-scaling result: 20k cap was not the whole plateau
Array 16780485 completed. Increasing pair-training data from nfit=20k up to 50k/100k/200k did not restore systematic M-scaling:
- M=100k,beta=2.5,nfit=100k: F=.76790 (essentially same as prior .77029).
- M=100k,beta=3,nfit=50k: F=.75832; nfit=100k: F=.76183 (prior nfit=20k: .73792).
- M=200k,beta=3,nfit=50k: F=.75369; nfit=100k: .74802; nfit=200k,K2: .75225 (prior nfit=20k: .74070).
Thus more learner data gives only modest improvement and the M=200k plateau remains. The hypothesis that the 20k cap alone caused the plateau is falsified.

Exact-physical edge diagnostics reveal a deeper mismatch: at late tau, validation-selected pairwise updates often lower held-out pair loss while worsening physical edge-ratio RMSE, even though correlation may improve. This alone does not identify learner failure because the learner targets the current FN walker amplitude, not the exact physical amplitude.

Next decisive decomposition:
1. exact one-block FN target using precisely the same clipped-ratio F[g,s] as the CT sampler;
2. compare walker histogram vs exact FN target;
3. compare learned ratio update vs exact FN ratios and physical ratios;
4. independent oracle-capacity control for same pairwise MLP at tau=1,2,3.

Corrected FN diagnostic job: 16781161 (replaces canceled 16781130, whose diagnostic used the older floor-based ct_fn_params and was not bit-identical to the sampler).
Pairwise oracle-capacity job: 16781155. Early tau=1 control: RMSE 1.60 -> .73, ratio correlation 0 -> ~.85, eta=1, showing the objective/model can learn substantial ratio structure under perfect sampling.

## Decisive diagnosis: FN self-target bias; second-order physical propagation repairs noisy guides
The corrected exact one-block FN diagnostic shows the pairwise learner tracks the FN target reasonably well, while the FN target itself drifts away from the physical amplitude. For M=100k:
- tau=.55: walker->FN fidelity .9801; FN->physical fidelity .9812.
- tau=1.0: walker->FN .9487; FN->physical .9396.
- tau=1.5: walker->FN .9430; FN->physical .8773.
- tau=2.0: walker->FN .9387; FN->physical .8202.
Accepted pairwise updates generally improve FN-edge ratio diagnostics but can worsen exact physical edge-ratio RMSE. Hence the late-beta plateau is primarily self-consistent FN bias, not simply poor walker statistics.

Independent oracle-capacity control for the same pairwise MLP/objective with perfect physical samples:
tau=1: edge ratio RMSE 1.60 -> .73, corr ~.85.
tau=2: 1.92 -> .93, corr ~.81.
tau=3: 2.27 -> 1.40, corr ~.70.
Thus model/objective capacity degrades with tau but is not the main failure.

Repeated physical local propagation controls from exact tau=.5:
first-order (I-dt H): F(beta=3)=.9365.
second-order (I-dt H + dt^2 H^2/2): F(beta=3)=.99811.

More importantly, second-order propagation from the actual reconstructed noisy guides strongly repairs initial error:
M=25k: F(.5)=.70693 -> F(1)=.97224 -> F(1.5)=.98984 -> F(2)=.99291 -> F(2.5)=.99516 -> F(3)=.99701.
M=100k: F(.5)=.84931 -> .97844 -> .98967 -> .99229 -> .99471 -> .99674.
Therefore the reconstructed guide is not the fundamental bottleneck; physical imaginary-time propagation is an attractor that repairs substantial guide error. The bad branch arose from learning the FN walker distribution as the amplitude truth.

New architecture under test: keep positive FN walkers only as a support sampler, but train the guide on the deterministic second-order local physical log-amplitude increment
dlog g_x = log |[(I-dt H + dt^2 H^2/2)(s g)]_x| - log g_x.
This requires only local/two-hop guide ratios and does not use exact-state oracle information. Sign update uses the same second-order local field. Array 16781445 (M=25k,50k,100k to beta=3) is running.

## Active-support K2 cache: no neural compression may be needed
One-block exact support coverage at M=25k:
- walker endpoints only: 19,501 states, covering 98.699% of exact physical probability mass and 97.306% of the K2-target mass; updating K2 exactly only there gives F=.78411 vs full K2 F=.78605 and amplitude overlap with full K2=.993924.
- +1 Hamiltonian hop: 158,880 states, physical mass .9999846, K2 overlap .9999899.
- +2 hops: 184,739/184,756 states, effectively exact.
Thus the positive walkers identify almost all physically important support even when the global Hilbert space is much larger than M.

Recurrent K2 cache array 16781496 launched for M=25k,100k and radius 0/1. Early M=25k radius-0:
tau=.55 guide F=.78425 vs full-K2 .78605;
tau=1.0 guide F=.94109 vs full-K2 .95771.
At tau=1 the active endpoint set still carries ~94.58% physical mass, explaining the controlled gap.
M=25k radius-1 at tau=.55 is essentially identical to full K2 (cache-to-K2=.9999897).

The second-order field is exactly radius-2 local. Define
h_x=(H psi)_x/g_x = H_xx s_x + sum_y H_xy s_y g_y/g_x,
b_x=(H^2 psi)_x/g_x = H_xx h_x + sum_y H_xy (g_y/g_x) h_y,
then z_x=s_x-dt h_x+(dt^2/2)b_x.
A 200-state numerical check against explicit sparse H^2 gave max absolute error 4.26e-14, RMS 3.17e-15, max relative error 1.06e-15.
Therefore the K2 update can be evaluated on demand from a radius-2 neighborhood; no global H^2 construction is intrinsically required.

## Importance-guided K2 cache: sampler reaches the ordinary i.i.d. finite-M ceiling
Importance transform checked algebraically against Q^{-1} F Q for q=g: max matrix discrepancy 3.41e-13, RMS 9.14e-15.

Radius-1 K2 cache is a strong mechanism control but at N=20 touches most of D=184,756:
- M=25k: guide F(beta=3)=.996955, active=161,939.
- M=100k: guide F(beta=3)=.996736, active=179,989.
Thus near-exact guide quality here is not by itself evidence of large-L scalability; active-set growth must be tracked as O(M z), not as a fraction of this small Hilbert space.

With q=g importance sampling, evaluate walkers in probability space because f=q a~a^2 when g~a. At beta=3:
M=5k: guide F=.97142, walker probability BC^2=.67139, TV=.33898.
M=10k: guide F=.98564, walker BC^2=.76882, TV=.26513.
M=25k: guide F=.99371, walker BC^2=.86029, TV=.18173.

Exact i.i.d. samples drawn directly from the true |psi|^2 distribution give, over 20 seeds:
M=5k: BC^2=.67228+-0.00377, TV=.32954+-0.00390.
M=10k: BC^2=.76997+-0.00166, TV=.25531+-0.00237.
M=25k: BC^2=.86281+-0.00073, TV=.17570+-0.00130.
Therefore the importance walkers are essentially at the finite-sample histogram ceiling; the remaining BC<1 is ordinary sample discreteness, not the sign problem. Small residual TV excess indicates mild population bias/correlation but is much smaller than the earlier amplitude-sampling failure.

This separates the architecture cleanly:
(1) K2 physical local update removes the self-consistent FN amplitude bias;
(2) q=g importance transform removes the branching/population collapse;
(3) the remaining N=20 histogram error is close to exact i.i.d. sampling noise.
Next decisive test: fixed-M beta=5 and beta=10 to test whether error/cost remains stable with imaginary time.

## Decisive FN-target diagnosis and K2-residual pivot
The pairwise FN-histogram learner is not primarily sample-limited. Exact one-block diagnostics using the same clipped-ratio F[g,s] as the CT sampler show:
- M=100k: walker-vs-exact-FN fidelity remains high: .980 at tau=.55, .949 at 1.0, .943 at 1.5, .939 at 2.0.
- But exact-FN-target vs physical-amplitude fidelity falls: .981 at .55, .940 at 1.0, .877 at 1.5, .820 at 2.0.
- Accepted pairwise updates typically improve FN-ratio metrics while often worsening exact-physical ratio RMSE.
Therefore the late-beta plateau is mainly self-consistent FN bias: the learner is successfully learning a biased FN target.

Independent oracle-capacity control for the same pairwise MLP/objective on perfect physical samples:
tau=1: edge log-ratio RMSE 1.60 -> .73, corr ~.85.
tau=2: 1.92 -> .93, corr ~.81.
tau=3: 2.27 -> 1.40, corr ~.70.
Thus model/objective capacity degrades with tau but is not the primary failure.

Krylov/Taylor integrator control starting from exact tau=.5 state:
1st order repeated dt=.05: F=.984(beta1), .949(1.5), .922(2), .921(2.5), .937(3).
2nd order psi'=(I-dt H + dt^2 H^2/2)psi:
F=.99928(beta1), .99761(1.5), .99670(2), .99720(2.5), .99811(3);
sign mismatch at beta3 ~1.62e-5.
So second-order local propagation nearly removes the time-discretization ceiling.

New architecture under test:
- keep positive FN walkers only as support/sampling mechanism;
- do NOT learn the FN histogram as the amplitude truth;
- compute a local second-order physical residual target from the current (s,g):
  psi2=(I-dt H + dt^2 H^2/2)(s g),
  Delta log g_x = log|psi2_x| - log g_x;
- train the guide correction on FN-visited states;
- set the sign chart from sign(psi2).
This breaks the self-reinforcing FN-amplitude bias while remaining local (two-hop for H^2).

New matched beta=3 campaign: Slurm array 16781533
- task 0: M=25k, nfit=25k
- task 1: M=100k, nfit=100k
Both running at submission.
Long FN diagnostic 16781161 was externally cancelled after ~32 min, but the decisive trend through tau=2.25 had already been extracted. Short diagnostic 16781199 completed.

## Beta-scaling verdict at N=20: solved by K2 physical guide + importance sampling
Long-beta array 16781537 completed.

M=10k:
- beta=3: guide F=.98564, walker probability BC^2=.76882, TV=.26513.
- beta=5: guide F=.99188, walker BC^2=.75049, TV=.26879.
- beta=10: guide F=.99305, walker BC^2=.74846, TV=.27504.

M=25k:
- beta=3: guide F=.99371, walker BC^2=.86029, TV=.18173.
- beta=5: guide F=.99705, walker BC^2=.84968, TV=.18733.
- beta=10: guide F=.99802, walker BC^2=.84404, TV=.19134.

Exact i.i.d. |psi|^2 histogram ceilings:
beta=3: M10k BC^2=.76997+-0.00166, TV=.25531+-0.00237; M25k BC^2=.86281+-0.00073, TV=.17570+-0.00130.
beta=5: M10k BC^2=.74718+-0.00229, TV=.27316+-0.00218; M25k BC^2=.84744+-0.00104, TV=.18827+-0.00152.
beta=10: M10k BC^2=.74069+-0.00155, TV=.27773+-0.00163; M25k BC^2=.84294+-0.00071, TV=.19207+-0.00119.

Thus the importance walkers remain essentially at the ordinary finite-sample ceiling through beta=10. There is no evidence that required M grows exponentially with beta on this N=20 test. The old exponential-looking beta cost was caused by FN self-target bias plus uniform-importance amplitude estimation, not by an unavoidable time-direction sign catastrophe.

Important limitation / new bottleneck:
6x6 ViT physical support is exponentially diffuse relative to explicit active neighborhoods. Using physical-weight resamples from the saved 4096/2048 train/validation pools:
- M=128: ~116 unique seeds; radius-1 active ~9.6k states, zero measured validation weight; radius-2 active ~358.5k states, only .0295+- .0103 validation weight.
- M=512: ~389 unique seeds; radius-1 active ~31.7k, validation weight ~8.98e-5.
- M=1024: ~658 unique seeds; radius-1 active ~53.3k, validation weight ~8.98e-5.
- M=2048: ~1040 unique seeds; radius-1 active ~83.8k, validation weight ~1.35e-4.
- M=4096: ~1505 unique seeds; radius-1 active ~120.7k, validation weight ~2.69e-4.
Therefore an explicit active dictionary cannot be the scalable 6x6 guide representation. Large-N success requires a callable/compressed function that generalizes K2 physical updates across configuration space. The beta problem is closed; guide compression/system-size scaling is now the decisive open problem.

## Exact K2 from reconstructed guide: decisive control
Slurm 16781893 completed cleanly. Applying the second-order local propagator exactly from the same reconstructed guides repairs large initial amplitude error:

M=25k:
- tau=.5: full F=.70693
- beta=1: .97224
- 1.5: .98984
- 2: .99291
- 2.5: .99516
- 3: .99701

M=100k:
- tau=.5: full F=.84931
- beta=1: .97844
- 1.5: .98967
- 2: .99229
- 2.5: .99471
- 3: .99674

Therefore K2 itself is not the problem and is strongly self-correcting even from a noisy reconstructed guide. The failed learned-K2 branches are representation/distillation failures:
- absolute K2 residual learner, beta3 M100k: guide F=.29865, walker F=.68062;
- edge-ratio K2 residual learner, beta3 M100k: guide F=.15687, walker F=.42759.
Conclusion: do not abandon K2. Redesign how the callable guide is updated so that it approximates the exact local K2 action without uncontrolled global extrapolation. The next question is a scalable distillation/parameter-update scheme, not another FN-histogram or edge-potential regression variant.

## SR/tangent-projection pivot after K2 distillation failure
Literature check: stochastic reconfiguration (SR)/TDVP is explicitly the variational projection of imaginary-time evolution in wave-function geometry; projected-tVMC combines infidelity projection with higher-order propagators. This directly matches our observed failure of pointwise/edge regression.

One-block SR-like tangent projection control:
- guide correction is projected in a neural tangent space, weighted by p(x)=g(x)^2;
- ridge/diagonal shift selected on an independent validation sample;
- width h=8 => 593 tangent parameters.
Results at tau=.5 -> .55:
M25k, ns=2k: exact K2 physical F=.786052; projected physical F=.771202; projected-vs-K2 amplitude fidelity=.992002.
M25k, ns=8k: projected F=.771635; target fidelity=.992210.
M100k, ns=2k: exact K2 F=.885496; projected F=.880122; target fidelity=.997120.
M100k, ns=8k: projected F=.880737; target fidelity=.997241.
Thus 2k samples already saturate the one-block projection; this is qualitatively better than Adam distillation and is not sample-starved.

Recursive test now uses exactly one setting (ns=2000,h=8), no tuning sweep:
Slurm 16782924, M=25k and 100k, beta=3.

Research Workspace collaboration resumed. Tim reports the same support pathology in the independent 8x8 FN-amplitude project: >99% of local neighbor ratios are extrapolated in their audit, and sign flips concentrate on unseen near-boundary neighbors. Posted our finite-tau K2/SR result as researcher.chatty seq 26 and asked whether their FN-energy-gradient update uses SR/natural-gradient geometry or plain parameter gradient.

## State-only SR closed; move to K2-local support closure
State-only SR/tangent projection is not sufficient even when one-step fidelity is extremely high.
M=100k recursive beta=3:
- dt=.05: final physical F=.38955; local target fidelity ~.9958 at beta3.
- dt=.025: final physical F=.39727; local target fidelity ~.99875 at beta3.
- dt=.0125: final physical F=.38927; local target fidelity ~.99966 at beta3.
Thus shrinking dt improves local projection error but does not rescue recursion. The failure is local-neighbor structure, not global fidelity.

Direct one-block neighbor audit at M=100k, dt=.0125 showed state-only SR global target fidelity .999763, but unseen validation-neighbor edge log-ratio RMSE did not improve (~.25046 -> .25095). Hybrid state+one-hop-edge SR likewise failed unseen edges: .25263 -> .25287, despite target fidelity .999765. Seen edges improved strongly.

Interpretation: K2 uses two-hop information. A tangent solve constrained only on sampled x and x->y edges leaves the next-step amplitudes on y and y->z underconstrained. New test expands sampled support through the Hamiltonian graph and constrains state residuals on the one-hop closure plus edge residuals on the second hop. On the 20-site problem, 2000 sampled centers generate ~2% one-hop and ~21% two-hop Hilbert-space coverage, so this is not full enumeration. Real M=100k, dt=.0125 closure job: 16786359.

## Neighbor-state SR also insufficient; representation pivot
Real neighbor-state metric job 16786453 (M=100k, dt=.0125, ns=2000, K=4) completed. It used only ~2.06% unique Hilbert states and moved unseen-neighbor edge RMSE in the right direction but only weakly: .33983 -> .33030 (~2.8%), while seen-neighbor RMSE worsened .03639 -> .04640 and global K2 target fidelity fell to .99840. This is not strong enough to promote recursively.

Persistent physical-H TDVP one-block control 16786405 likewise had excellent global K2 fidelity .999760 but unchanged/worse unseen-neighbor RMSE .24049 -> .24055, while seen neighbors improved .01156 -> .00813. Therefore putting neighbor amplitudes inside E_loc(x) is not sufficient when the tangent/metric samples only center states.

Two-hop closure SR 16786359 expanded to ~55% of the 20-site Hilbert space and ~790k second-hop edges, yet only improved unseen edge RMSE .55722 -> .54234, worsened state RMSE .02515 -> .04185, and reduced target fidelity to .99839. Closed as too expensive and too weak.

Conclusion: the remaining bottleneck is callable-guide representation/generalization to Hamiltonian-neighbor states, not optimizer geometry. Existing small fully connected feature MLP variants are closed. A pure-JAX graph message-passing representation test is now running (16787010): same exact K2 one-block correction, same M=100k/ns=2000 center sample, but graph-equivariant message passing. Decisive metric is unseen-neighbor edge log-ratio RMSE.

## K2 distillation / representation campaign: current closure
Further one-block M=100k, dt=.0125 controls sharpen the failure mode:
- persistent physical-H TDVP on g^2 states: global K2 target fidelity .999760, but unseen neighbor edge RMSE .24049 -> .24055 (no improvement);
- sampled-neighbor-state tiny-MLP SR (K=4): unseen edge RMSE .33983 -> .33030 (~2.8% improvement), seen edge worsens;
- state-supervised PairGNN: unseen edge RMSE .24432 -> .24322 (no meaningful improvement), target fidelity .999708;
- PairGNN + sampled neighbor states K=4: unseen edge .32821 -> .30999 (~5.6% improvement), target fidelity .998215;
- edge-supervised PairGNN on 30k local K2 correction edges: unseen edge .45504 -> .40608 (~10.8% improvement), target fidelity .996885. Better, but still far from the accuracy needed for stable recursion.
Independent oracle GNN local-ratio capacity at tau=.5 is strong when trained directly on exact amplitude edge ratios: amplitude fidelity .989984 and K1-next sign error 9.23e-9. Thus graph representation has real capacity, but ordinary physical/state support does not constrain the rare boundary ratios sufficiently.

Naive exact local-ratio K2 recursion without distillation was also falsified computationally. For one sampled state on the 20-site graph:
- k=0 R2 evaluation touched 51 states;
- k=1 already touched 9,473 states, with ~428k cached local ratios;
- k=2 became prohibitively expensive and was terminated.
So exact recursive ratio propagation expands rapidly in configuration space and is not itself the scalable cure.

Combined conclusion: changing optimizer, dt, simple width, MLP->GNN, state-vs-edge loss, or shallow neighbor augmentation does not solve the key issue. The remaining bottleneck is support acquisition near low-margin/nodal-crossing configurations, already identified in the earlier checkpoint. Crucially, a K2 teacher can label such targeted states locally from the current guide, so future support does not have to come from rare physical FN walkers.

Next test: auxiliary tempered/boundary sampling q_alpha proportional to g^(2 alpha), alpha<1, measure acquisition of low-K2-margin and sign-crossing states and the cost/ESS tradeoff. If successful, mix physical g^2 support with targeted boundary support for local K2-ratio training.

## Nodal singularity and sparse online boundary-cache architecture
Further diagnostics explain why smooth amplitude distillation fails even after targeted support acquisition. For K2,
  g_{k+1}(x)=g_k(x)|R_k(x)|,
so Delta log g(x)=log|R_k(x)|. Near a K2 node/crossing R_k -> 0 and the log-amplitude correction is singular.
At M=100k, dt=.0125:
- low-margin |R|<.2 states: median Delta log g=-2.38, minimum -11.14;
- bulk |R|>=.2: minimum about -1.61;
- total g^2-weighted correction RMS is only .0161.
Thus almost all of the wavefunction sees a small smooth update, while a tiny nodal set sees huge log corrections. A single smooth network is structurally ill-suited to represent both.

Support diagnostics:
- physical g^2 crossing mass = 1.44e-7;
- one Hamiltonian-neighbor draw from g^2 has crossing-hit probability 2.65e-4 (~1840x enrichment);
- 2000 physical centers generate ~101k one-hop edge occurrences / ~18k unique neighbor states and include ~15-22 crossing states depending seed.
Full-shell GNN training still only modestly improves unseen-edge error (~.516 -> .462), and boundary-stratified loss fails (boundary edge RMSE ~2.87 -> 2.85), confirming that the singular target rather than mere absence of examples is the issue.

Sparse online cache test:
Use a smooth bulk GNN correction everywhere, but for the current walker/guide one-hop shell detect B={R<0 or |R|<.2} from the guide-local K2 factor and store exact K2 correction on B plus their immediate neighbors.
On an independent validation shell:
- 30 boundary states detected;
- B+neighbors cache = 1263 states = 0.68% of D;
- boundary-edge RMSE 2.13 -> 5.65e-8 (essentially exact);
- overall edge RMSE .1813 -> .1766;
- global K2 target fidelity remains .999714.
A cache built from another replica does not transfer, but transfer is unnecessary: the intended algorithm rebuilds the sparse cache online each block from its own current walker shell.

Decisive recursive test:
bulk = state-SR tangent projection; boundary = online exact sparse cache rebuilt each dt block; dt=.0125, ns=2000, h=8, beta=3.
First cluster attempt 16788303 failed only due a hard-coded local ROOT path. Fixed script uses ROOT relative to __file__. Resubmitted as Slurm job 16788310; initially pending by Priority.
Baseline to beat: state-only SR at same dt ends beta3 with physical F~.3893.

## Tempered-support GNN moved to Paderborn
LMU submission 16793973 failed at scheduling/infrastructure stage. User explicitly requested Paderborn.
PC2/Noctua A40 job 3523617 submitted on 2026-10-02:
- M=100000, dt=.0125
- edge-GNN K2 residual training
- q_train = 0.5 g^2 + 0.5 g^(2 alpha), alpha=.6
- 30k train edges, 10k validation edges, 15 epochs
- audits physical-shell, tempered-shell, and low-margin/crossing boundary errors separately.
At submission: PENDING, no stdout/stderr yet.

## Tempered-neighbor GNN representation pivot: first recursive success
After closing state-only SR, hybrid one-hop SR, two-hop closure SR, persistent tiny-MLP TDVP, neighbor-state SR, and direct edge-net variants as insufficient on unseen Hamiltonian-neighbor ratios, we tested a graph representation plus deliberate support broadening.

Coverage result for q_alpha(x) ∝ g(x)^(2 alpha), expanding sampled centers to all Hamiltonian neighbors:
- alpha=0.8, 2000 centers: ~10.4% of Hilbert space touched, ~97.75% of physical g^2-weighted neighbor-target edge weight covered.
- alpha=0.8, 5000 centers: ~17.3% touched, ~98.95% physical edge weight covered.
Thus local support closure can cover nearly all physically relevant edges without full enumeration.

A clean PairGNN K2-residual audit at dt=.0125 showed that architecture alone does NOT solve unseen extrapolation: on a fixed blind edge split unseen residual RMSE only changed .04929 -> .04862 despite global K2 target fidelity .999992. Direct antisymmetric edge-ratio modeling likewise barely improved unseen edges. Therefore representation alone is not enough; support broadening is essential.

Tempered closure GNN test:
- alpha=.8, 2000 centers, all Hamiltonian neighbors added as state targets;
- train set ~18-22k unique states (~10-12% of D) per block;
- full 48-channel, 4-layer PairGNN residual model, 4 epochs/block;
- exact 20-site physical K2 target used only for this structural control.

High-quality recursive exact-start benchmark from tau=.5, dt=.0125:
step/tau/full fidelity/one-step target fidelity
1/.5125: .99998237 / .99998282
2/.5250: .99991455 / .99997432
3/.5375: .99980312 / .99997531
4/.5500: .99967758 / .99997989
5/.5625: .99955945 / .99998579
6/.5750: .99941818 / .99998936
7/.5875: .99919941 / .99998141
8/.6000: .99905423 / .99999208
9/.6125: .99875230 / .99997453
10/.6250: .99840903 / .99997611
Sign mismatch remains tiny (<1e-6 weight at step 10).

This is the first learned recursive branch that does not collapse. Error accumulates smoothly rather than self-reinforcing catastrophically. However this benchmark starts from the exact tau=.5 state. Next decisive test: same unchanged architecture/support strategy starting from the realistic reconstructed M=100k guide. If it repairs that guide while retaining recursive stability, promote to the scalable finite-T architecture.

## Candidate scalable finite-tau update after tempered-GNN success
The first non-collapsing learned recursion suggests the following scalable formulation.

For psi(x)=s(x)g(x), define local first- and second-order physical propagation ratios
E1(x)=(H psi)_x/psi_x,
E2(x)=(H^2 psi)_x/psi_x = sum_y H_xy [psi(y)/psi(x)] E1(y),
R2(x)=1-dtau E1(x)+(dtau^2/2)E2(x).
Then
Delta log g(x)=log|R2(x)| (up to a normalization gauge),
s_new(x)=s(x) sign R2(x).
These quantities require only one- and two-hop Hamiltonian-neighbor evaluations of the callable guide/sign chart.

Training/support strategy:
1. sample centers from tempered q_alpha(x) proportional to g(x)^(2 alpha), alpha≈0.8, with local Metropolis;
2. add all Hamiltonian neighbors of every sampled center as explicit state targets;
3. evaluate the local K2 residual on that closure;
4. train/update a physical-lattice graph guide on Delta log g;
5. lazily cache/update signs on states touched by walkers/neighbors from sign R2.

Cost is polynomial: O(ns*z) closure states and O(ns*z^2) local K2 work (z = Hamiltonian graph degree), plus graph-network evaluation. The endpoint distance/J2-count global features used in the 20-site prototype should be computed by physical-lattice matching/assignment, not by full configuration-space shortest paths, in the scalable implementation.

Important unresolved scaling risk: the 20-site result alpha=.8, ns=2000 gives ~97.8% g^2-weighted neighbor-target coverage while touching ~10% of D, but this is not a size-scaling result. Must measure independent-sample physical edge coverage versus ns, alpha, and N without exact enumeration before claiming polynomial accuracy.

Exact-start high-quality recursive benchmark (PairGNN, alpha=.8, 2000 centers, all neighbors, 4 epochs/block, dt=.0125) survived 10 blocks to tau=.625 with F=.998409 and no collapse. Decisive realistic reconstructed-guide M=100k run is Slurm 16794033.

## Realistic reconstructed-guide tempered PairGNN test: promotion gate failed
LMU Slurm job 16794033 completed successfully on 2026-10-02 (exit 0).
Configuration: M=100000, dt=.0125, alpha=.8 tempered centers, 2000 centers/block, all Hamiltonian neighbors added, 48-channel 4-layer PairGNN, 4 epochs/block.

Starting from the actual reconstructed M=100k guide at tau=.5 (previous physical fidelity ~.84931), the recursion gives:
- tau=.5125: F_full=.862421, one-step K2-target fidelity=.996301
- tau=.5250: F_full=.820593, target fidelity=.992559
- tau=.5375: F_full=.759224, target fidelity=.988764
- tau=.5500: F_full=.712072, target fidelity=.992681
- tau=.5625: F_full=.682228, target fidelity=.994438

Verdict: FAIL promotion gate. The exact-start tempered-neighbor PairGNN stability does not survive a realistic off-manifold reconstructed guide. High one-step target fidelity remains misleading: after a slight first-step repair, recursive physical fidelity collapses rapidly.

Interpretation: support broadening is necessary but not sufficient. The next diagnostic should measure the physical/unseen Hamiltonian-neighbor log-ratio error during this realistic-guide recursion and isolate which off-manifold ratio modes remain uncorrected. Do not promote the current PairGNN into the scalable loop yet.

## Sparse online nodal-boundary cache recursion: partial repair, long-time failure
LMU Slurm job 16788310 completed successfully (exit 0). This branch uses state-SR for the smooth bulk plus an online exact cache for low-margin/crossing K2 states and their neighbors, rebuilt every dt=.0125 block from the current guide shell.

Starting from the reconstructed M=100k tau=.5 guide (F~.84931), the cache strongly repairs the early trajectory:
- tau=.75: F_full=.943506
- tau=.85: peak F_full=.947970
- tau=1.0: F_full=.933895
but then drifts:
- tau=1.5: .792668
- tau=2.0: .615366
- tau=2.5: .480322
- tau=3.0: .396644.
State-only SR at the same dt ended beta=3 at ~.38927, so the cache gives a major transient improvement but only a small asymptotic gain.

Per-step projection-target fidelity remains ~.9997 throughout, and the online cache stays sparse (typically ~0.1%-0.4% of D). Thus the nodal singularity is a real early failure mode but is not the whole recursive collapse. A second, bulk/off-manifold Hamiltonian-neighbor error remains and accumulates despite excellent global one-step fidelity.

Next discriminator: on the reconstructed guide, measure how much exact physical state/edge weight is actually covered by q_alpha(g)+neighbor closure. If physical coverage collapses relative to the exact-start ~97.8% edge coverage, the issue is self-reinforcing wrong-guide support; if coverage remains high, the remaining failure is representation/training of the off-manifold bulk correction itself.

## 2026-10-02 continuation: reconstructed-guide edge discriminator
Short support diagnostic job 16794266 was attempted on LMU but failed operationally before physics execution because k2_support_coverage20.py was not executable (exit 13). No scientific inference is drawn from it.

Rather than bypass the execution restriction, resumed the already-existing LMU tempered-edge GNN audit, which directly measures physical/tempered/boundary edge-ratio error on the reconstructed M=100k guide:
- script: experiments/k2_gnn_tempered_edges_oneblock20.py
- wrapper: experiments/k2_gnn_tempered_edges_lmu.sbatch
- M=100000, dt=.0125, alpha=.6, 50/50 physical+tempered edge training
- 30k train edges, 10k validation edges, 15 epochs
- custom chatty_notify.sh notifier; Slurm mail disabled
Submitted as LMU Slurm job 16794277.

Decision: if this audit substantially lowers physical-shell unseen/local edge RMSE while boundary error remains the main residual, combine tempered PairGNN bulk with the sparse online boundary cache. If physical-shell edge RMSE remains poor despite tempered support, stop tuning the current PairGNN and move to an explicitly Hamiltonian-edge/local-ratio representation.

## 2026-10-02 quick check: matched exact-vs-reconstructed PairGNN edge audit
LMU job 16794277 failed after 4 s before any physics because the selected module environment lacks flax (ModuleNotFoundError). No package was installed and no scientific inference is drawn.

Replaced it with a PyTorch diagnostic using the exact 48-channel/4-layer PairGNN and alpha=.8 + full one-hop neighbor-closure state training used in the realistic recursive run. The audit reports physical-state support mass, physical Hamiltonian-edge coverage, and seen/unseen edge log-ratio RMSE. It runs matched exact-start and reconstructed-start controls so the earlier ~97.8% exact-start coverage is internally calibrated.

Submitted LMU array 16794532:
- task 0: exact-start control
- task 1: reconstructed M=100k guide
At last check both tasks are PENDING because required nodes are down/drained/reserved. Slurm mail is disabled; custom chatty_notify.sh is enabled.

## 2026-10-02 corrected matched PairGNN coverage audit: support closed as bottleneck
Corrected LMU array 16794959 completed successfully. The previous edge audit had double-weighted source probabilities during evaluation; rerun fixes this.

Exact-start control (alpha=.8, 2000 centers + all one-hop neighbors):
- train support 10.21% of D
- exact physical state mass in support 99.4505%
- physical Hamiltonian-edge weight with both endpoints in support 97.5951%
- target fidelity 0.99999159
- physical edge baseline RMSE 0.02205
- projected RMSE 0.01816; seen 0.01387; unseen 0.07689
This reproduces the earlier ~97.8% coverage claim and validates the metric.

Reconstructed M=100k guide:
- train support 15.99% of D
- exact physical state mass in support 99.6352%
- physical Hamiltonian-edge weight with both endpoints in support 98.5307%
- guide-weighted edge coverage 92.0768%
- target fidelity only 0.9887198
- physical edge baseline RMSE 0.09645
- after state-supervised PairGNN: 0.11667 (worse)
- seen edge RMSE 0.09800; unseen 0.53134

Conclusion: missing tempered/neighbor support is NOT the bottleneck. The realistic guide has even better physical support coverage than exact-start, but the state-supervised PairGNN corrupts the required local Hamiltonian ratios. This closes support broadening as the primary explanation.

Next discriminator submitted as LMU array 16796207:
- same 48-channel/4-layer PairGNN
- direct Hamiltonian-edge log-ratio objective (f_j-f_i = target_j-target_i)
- 30k train edges, 10k validation, alpha=.8, 50/50 guide p and tempered q_alpha centers
- exact-start and reconstructed-start matched tasks
Decision: success only on direct-edge loss => objective was bottleneck; failure on reconstructed guide => current PairGNN representation is insufficient off-manifold and should be replaced by explicit Hamiltonian-edge/local-ratio message passing.

## 2026-10-02 direct-edge PairGNN discriminator: representation bottleneck confirmed
LMU array 16796207 completed successfully.

Matched direct Hamiltonian-edge supervision, same 48-channel/4-layer PairGNN, alpha=.8, 30k train edges, 10k validation, 15 epochs.

Exact-start:
- physical edge RMSE 0.02212 -> 0.01799
- guide-weighted edge RMSE 0.02998 -> 0.02711
- target fidelity 0.9999977

Reconstructed M=100k guide:
- physical edge RMSE 0.08991 -> 0.08335 (~7% improvement)
- guide-weighted edge RMSE 0.19088 -> 0.17008 (~11% improvement)
- target fidelity 0.996941
- validation edge RMSE 0.22536
- train RMSE plateaus ~0.217 and validation ~0.226 by epochs 10-15, so this is not an undertraining/early-stopping issue.

Conclusion: direct edge supervision is better than state-supervised loss, but only modestly. The current PairGNN representation cannot express the off-manifold K2 local-ratio correction accurately enough on a realistic reconstructed guide. Close further optimizer/epoch/support tuning on this representation.

Next branch: explicit Hamiltonian-edge/local-ratio message-passing model that consumes (x,y) as an edge object and predicts Delta log g(y)-Delta log g(x) directly, rather than constructing it as f(y)-f(x) from two state scores. Preserve tempered center sampling + one-hop closure and sparse nodal boundary cache.

## 2026-10-02 edge-native integrable local-potential branch
After direct-edge PairGNN saturated far above the required off-manifold local-ratio accuracy, started a representation-level test rather than further optimizer/support tuning.

Architecture:
- translation-equivariant state potential F_theta(x;y)=sum_i phi_theta(local message-passing environment of site i, conditioned locally on endpoint y)
- 4 message-passing layers on J1/J2 physical-lattice adjacency, hidden width 48
- trained only through Hamiltonian-edge differences F_theta(x')-F_theta(x)
- therefore edge ratios are integrable/cycle-consistent by construction, unlike an unconstrained edge predictor
- preserves extensivity/locality: a local spin exchange changes only nearby local contributions within the receptive field

Matched audit:
- same hard endpoint column, M=100000 reconstructed guide
- dt=.0125, alpha=.8 tempered support, 50/50 p and q_alpha source mixture
- 30k train edges, 10k validation edges, 15 epochs
- exact-start and reconstructed-start controls
Submitted as LMU array 16799429. Both tasks entered RUNNING.
Promotion criterion: reconstructed physical-edge RMSE must improve materially beyond the direct-edge PairGNN result 0.08335 (baseline ~0.0899), ideally approaching the exact-start scale. If not, local integrable patch potential is also insufficient and the next test is an explicit antisymmetric edge model with a cycle-consistency penalty / local K2 analytic features.

## 2026-10-02 integrable local-potential GNN: no material off-manifold gain
LMU array 16799429 completed successfully.

Architecture: F_theta(x;y)=sum_i phi_theta(local message-passing environment), 4 physical-lattice message-passing layers, width 48, trained only through Hamiltonian-edge differences. Edge predictions are integrable/cycle-consistent by construction.

Exact-start control:
- physical edge RMSE 0.02082 -> 0.01640
- target fidelity 0.99999762
- validation edge RMSE 0.03242

Reconstructed M=100k:
- physical edge RMSE 0.10016 -> 0.09347 (~6.7% improvement)
- guide-weighted edge RMSE 0.18823 -> 0.16809 (~10.7% improvement)
- target fidelity 0.995979
- train RMSE ~0.2211, validation ~0.2174 after 15 epochs

Conclusion: integrability/local sum pooling does not materially outperform direct-edge PairGNN on the realistic guide. Close this architecture as a solution to the off-manifold bottleneck.

Physics diagnosis strengthened: K2 correction depends explicitly on current-guide local-energy fields E1=(H psi)/psi and E2=(H^2 psi)/psi, but the tested networks only see spin/endpoint geometry. Submitted LMU job 16799457 to quantify how much physical edge target is captured by first-order E1 alone versus the full E1+E2 analytic K2 factor. If E1 explains most edge structure, next scalable model should ingest current-guide local-energy / neighbor-ratio-star features rather than merely increase neural capacity.

## 2026-10-02 K2 correction decomposes almost entirely into nonlinear K1 local-energy factor
LMU diagnostic 16799457 completed successfully on reconstructed M=100k guide at tau=.5, dt=.0125.

Physical-edge audit (530,534 Hamiltonian edges sampled from exact physical distribution):
- full K2 correction edge RMS: 0.0894253
- naive linear -dt E1 edge prediction error: 0.140760 (bad; nonlinearity matters)
- nonlinear log|R1| = log|1-dt E1| edge prediction error versus full K2: 0.00507437
- exact R2 self-check: 0
- physical-state RMS of log|R2|-log|R1|: 0.0059817
Typical scales: median |E1|=8.63, median |E2|=73.66.

Interpretation: ~94% of the K2 edge-field RMS is already captured by the exact radius-1 current-guide local-energy factor log|1-dt E1|. The tested neural networks were asked to infer this guide-dependent quantity from spin/endpoint geometry without seeing the current-guide local-energy environment. This is now the leading explanation for off-manifold failure. E2 is a small but potentially cumulatively important remainder.

Next discriminators:
1. fresh PairGNN trained from scratch on absolute K2 target amplitude edge ratios (not residual correction), LMU job 16799530;
2. exact repeated K1 vs K2 physical propagation from saved reconstructed M100k guide at dt=.0125 through beta=3, LMU job 16799537, to determine whether the small E2 remainder is cumulatively essential.

## 2026-10-02 K1 vs K2 at dt=.0125: analytic K1 backbone is already strong, E2 needed for precision
LMU job 16799537 completed successfully using saved reconstructed M=100k tau=.5 guide and exact full-space physical propagation through beta=3.

Fidelity:
- tau=.5: both .849313
- beta=1: K1 .977835, K2 .978794
- 1.5: K1 .986536, K2 .992738
- 2.0: K1 .987645, K2 .996852
- 2.5: K1 .990597, K2 .998627
- 3.0: K1 .993860, K2 .999407

Sign mismatch at beta3:
- K1: 7.20e-5
- K2: 5.69e-6

Thus at dt=.0125 the nonlinear radius-1 K1 factor is already a very strong physical propagator, consistent with the edge diagnostic where log|1-dt E1| predicts the full K2 edge field to RMSE .00507. However E2 remains cumulatively important for precision, especially sign accuracy. Recommended architecture is an analytic K1 backbone plus a small K2/E2 correction, provided the updated guide can be compressed into a callable representation without reintroducing off-manifold error.

Fresh absolute-target PairGNN compression test remains running as job 16799530.

## 2026-10-02 absolute-target PairGNN compression also fails
LMU job 16799530 completed successfully. Instead of learning Delta log g relative to the reconstructed guide, a fresh PairGNN was trained directly on absolute K2 target amplitude edge ratios.

Results:
- K2 target physical fidelity: .859956 from starting guide .849503
- PairGNN target fidelity: .971936
- but PairGNN physical fidelity: .849360, essentially no improvement over start
- physical target-edge RMSE: current guide .09729 -> fresh PairGNN .22331 (much worse)
- validation edge RMSE .57689 after 15 epochs

Conclusion: close fresh absolute-target PairGNN compression too. The oracle graph capacity result does not translate to this realistic support/training problem.

## E2 remainder is small but broad, not cache-sparse
LMU job 16799552 completed.
At tau=.5 reconstructed M100k, dt=.0125:
- physical RMS epsilon2 = log|R2|-log|R1| = .005982
- |epsilon2|>.005 covers 89.34% physical mass and 94.26% of epsilon2 squared error
- |epsilon2|>.01 covers only .392% physical mass but still 20.47% squared error
- near-node proxy |R1|<.2 captures only ~10.28% of epsilon2 squared error
Thus E2 cannot be handled solely by a sparse nodal cache. It is a small, broadly distributed precision correction.

Current architecture conclusion:
- dominant bulk update: exact analytic log|1-dt E1|
- E2: broad small analytic/reweighting correction
- neural compression families tested so far are closed
- next scalability test is recursive callable K1 cost growth; LMU job 16799572 submitted for exact memoized shell-growth counts.

## 2026-10-02 exact recursive K1 callable cost is manageable for short blocks
LMU job 16799572 completed in 35 s. Exact memoized K1 callable recursion from representative reconstructed-guide states at dt=.0125:

Hard state x=59279:
- k=1: 58 total cached states
- k=2: 1,270
- k=3: 12,790
- k=4: 64,402

Independent x=10660:
- k=1: 56
- k=2: 1,201
- k=3: 12,044
- k=4: 61,293

This is much milder than exact recursive K2 and supports short analytic-K1 blocks (2-3 microsteps) before guide refresh. At fixed block depth the cost grows polynomially with Hamiltonian degree; unlimited recursion still cannot scale.

## New compression branch: normalized autoregressive MLE
The earlier K2 importance-guided walker experiment already showed walker probabilities essentially at the exact i.i.d. finite-M ceiling. Therefore sampler quality is not the missing piece; compression of the positive distribution is.

No autoregressive/MLE guide exists in the repo. Submitted a minimal proof-of-concept LMU job 16799606:
- fixed-Sz sector autoregressive GRU, hidden 96
- exact sector-normalized likelihood via sequential up-spin-count masking
- conditioned on endpoint column
- trained by MLE on only 25k i.i.d. samples from the one-step K2 target distribution, matching the sample scale where importance K2 walkers were already near-i.i.d.
- 30 epochs
- full D=184,756 evaluation of target fidelity, physical fidelity, and Hamiltonian-edge log-amplitude RMSE

This deliberately tests compression capacity/objective first, with perfect K2-target samples. If it fails, autoregressive MLE is closed at this capacity. If it succeeds, next test is identical training on actual importance-K2 walkers, then recursive refresh blocks.

## 2026-10-02 normalized AR MLE pilot: likelihood works, Hamiltonian-neighbor compression fails
LMU job 16799606 completed successfully. Fixed-Sz sector autoregressive GRU (hidden 96), exact normalization, 25k ideal i.i.d. K2-target samples, 30 epochs.

Training was stable with no large train/validation gap:
- val NLL 7.0717 -> 6.1920
- sector probability normalization check 0.99999991
- 4,173 unique configurations in the 25k training sample

But physics metrics fail:
- target fidelity only .955543
- physical fidelity .835700, worse than starting guide .849503
- current-guide target edge RMSE .09068
- AR target edge RMSE .26085 (much worse)

Conclusion: plain normalized MLE learns the high-probability distribution but does not constrain Hamiltonian-neighbor ratios, because the finite target sample contains few unique configurations. This is the same off-neighbor information problem in a different objective.

Next test submitted as LMU job 16799727:
- same normalized fixed-Sz AR model
- 25k K2-target MLE samples
- plus 30k tempered Hamiltonian-neighbor teacher edges from q_alpha, alpha=.8
- hybrid loss = NLL + dimensionless edge MSE normalized by target-edge RMS^2
- exact normalization and full-D physics audit retained
Decision: success => normalized AR refresh route survives; failure => simple GRU AR capacity is closed and a stronger 2D AR NQS is required.

## 2026-10-02 hybrid autoregressive MLE + Hamiltonian-edge supervision also fails
LMU job 16799727 completed successfully.

Model:
- fixed-Sz autoregressive GRU, hidden 96
- 25k ideal i.i.d. K2-target MLE samples
- 30k tempered Hamiltonian-neighbor teacher edges, alpha=.8
- hybrid NLL + normalized edge-MSE loss
- 30 epochs

Training remained stable:
- val NLL ~6.20
- scaled edge RMSE decreased 0.731 -> 0.409
- 4,268 unique train configurations
- normalization check 1.00000002

But physics metrics fail:
- starting physical fidelity: 0.849503
- one-step exact K2 target physical fidelity: 0.859956
- learned AR target fidelity: 0.963810
- learned AR physical fidelity: 0.849169 (no physical improvement)
- current-guide target edge RMSE: 0.10120
- learned AR target edge RMSE: 0.25423 (much worse)

Conclusion: explicit edge supervision does not rescue the small sequential GRU AR model. Close simple GRU autoregressive compression at this capacity/objective. The remaining AR option is a stronger genuinely 2D architecture (e.g. masked transformer / 2D graph-autoregressive model) that can preserve normalization while representing local Hamiltonian-neighbor ratios.

Message-board check after seq75: no reply yet from @fabi-do.

## 2026-10-03 branch consolidation: reuse shared factorized-SR amplitude engine
Ground-state FN and finite-T branches share the generic callable-amplitude problem, so finite-T stops independent compressor architecture searches.

Important distinction from old finite-T state-only SR:
- old method = one linear tangent projection around a randomly initialized tiny MLP;
- new shared idea = node/sign-preserving factorization g_new=g_old*exp(r_theta), nonlinear multi-step SR with a warm-started callable model and small function-space trust steps.

Finite-T-specific objective is NOT frozen-H_FN energy. New discriminator uses natural-gradient/SR ascent of overlap with the one-block K2 target. Its force is E_q[O]-E_p[O], with p proportional to g_theta^2 and q proportional to g_theta*g_K2, so it has a future QMC estimator through the local K2 reweighting factor.

Paderborn A40 job 3555005:
- 20-site reconstructed M=100k guide at tau=.5
- dt=.0125, exact K2 one-block target
- factorized 48-channel/4-layer PairGNN residual
- 20 nonlinear SR steps, ns=1024, trust=.002, shift=.1
- exact full-space distributions used only for this structural benchmark
- decisive metrics: K2 target fidelity AND blind Hamiltonian-edge log-ratio RMSE.
Success => port same overlap-SR force to walker estimates and test repeated short blocks to higher beta.
Failure => factorized SR does not by itself fix the finite-T off-neighbour problem; do not duplicate FN architecture work.

## 2026-10-03 subangle checkpoint: amplitude relearning is the gate before finer CTQMC

Decision:
Do NOT advance the broader finite-T CTQMC implementation until the callable-amplitude refresh is closed. The sign-update/short-time propagation mechanism is already strong enough to isolate this subproblem cleanly.

Why:
- exact K1/K2 propagation from the reconstructed 20-site guide improves with beta through beta=3;
- the dominant K2 update is the explicit nonlinear K1 local-energy factor log|1-dt E1|;
- recursive exact callable evaluation grows rapidly in Hamiltonian shells, so long-beta work requires periodic compression/refresh;
- prior state-only SR/TDVP/GNN/AR methods failed mainly because they did not preserve blind Hamiltonian-neighbor amplitude ratios, even with excellent global fidelity.

Shared-work policy:
The neighboring ground-state FN project owns the generic factorized callable-amplitude/SR machinery. Finite-T does NOT launch a parallel architecture campaign. This branch only supplies the finite-dt projection objective and recursion tests.

Current finite-T discriminator:
Original job 3555005 failed before physics due only to NumPy attempting in-place mutation of a read-only JAX array. One-line copy fix applied; experiment unchanged.
Replacement Paderborn A40 job: 3565142 (RUNNING at checkpoint).

Experiment:
- exact 20-site benchmark, reconstructed M=100k guide at tau=.5
- dt=.0125
- factorized residual PairGNN, 48 channels / 4 layers
- 20 nonlinear overlap-SR trust steps, ns=1024, trust=.002, shift=.1
- objective: maximize overlap with exact one-block K2 amplitude target
- pass requires BOTH improved K2 target fidelity and reduced blind Hamiltonian-edge log-ratio RMSE.

Decision tree after 3565142:
PASS -> next test is repeated exact-K2-block -> learned-refresh recursion on the 20-site benchmark to beta well beyond 3, before any CTQMC integration. Then replace exact SR expectations by walker estimates and measure sample cost.
FAIL -> do not invent another finite-T compressor. Feed the failure into the shared amplitude-learning project and keep finer CTQMC implementation parked.

Subangle verdict for now:
The unresolved finite-T bottleneck is no longer sign repair; it is repeated relearning/compression of a callable positive amplitude with accurate Hamiltonian-neighbor ratios. High-beta feasibility reduces to whether this refresh error can remain bounded under repeated short blocks with polynomial sampling/computational cost.
