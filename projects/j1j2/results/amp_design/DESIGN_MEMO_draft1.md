# Amplitude update for the FN/Krylov loop: design memo (2026-10-06, draft 1)

Scope: step 2 of the loop, i.e. learning a callable a_{k+1} ~ phi_FN[a_k, s_k] from samples.
Tests: exact 4x4, CPU, about 1.5 CPU-h in total. Code is in `experiments/amp_design/` and raw output in this folder.
Every test below uses linear log-models: `tab` has one parameter per translation orbit (P = 822); `rf` has P = 1000 translation-symmetric random features and must generalise.
The ViT has **not** been tested with the proposed method. Notation: F = H_FN[a_k, s_k], E_F[b] = <b|F|b>/<b|b>, u* = log phi_FN - log a_k.
The default test state is "k0" = net H with its own signs. Its frozen gain is G = E_F[a_k] - E_FN = 3.3e-4 (2.1e-5 per site), and RMS(u*) = 0.0036.

## 0. Bottom line
1. **The update is stiff and heavy-tailed. It is not capacity-limited.**
   - F has a fixed-node "wall": the diagonal term sum_viol |H_xy| a(y)/a(x) is huge at low-amplitude configurations that sit on sign-violating (frustrated) edges.
   - At k0, lambda_max(F) = 2.4e3, while H has a bandwidth of about 20. These wall configurations carry only 3.4e-5 of the a^2 weight.
   - Consequences:
     - First-order methods (ITE, SR = natural gradient) need 10-50 exact steps.
     - Estimators built on samples from a^2 have unbounded weights a(y)/a(x) exactly where the information is.
     - The current validation gate can be exploited.
2. **Two changes together give >= 90% of the ideal per-iteration gain at N = 1e4 per step on 4x4.** The two changes:
   - (a) A Gauss-Newton step whose metric is the Hamiltonian-edge ("Dirichlet") form of the stoquastic F.
   - (b) Samples drawn from a tempered |b|^{2 beta} distribution (beta = 1/4 to 1/2), reweighted.
   - Either change alone fails. This mirrors the sign problem: the training distribution matters more than capacity.
3. **The amplitude refit is VMC in disguise unless it is solved to completion.**
   - The frozen Rayleigh quotient majorises the fixed-sign VMC energy and touches it at the guide.
   - So the first SR steps of the refit *are* VMC steps. This explains the equal-cost tie.
4. **Outer loop: over-relaxation omega = 2 on log a, or Anderson mixing, halves the number of iterations** in the exact loop. omega >= 2.5 diverges.
5. **Referee:** use the majorisation chain as a cheap per-iteration certificate, with tempered paired VMC. Run the FN (GFMC) referee only every few iterations.

## 1. Three identities (exact, checked numerically)
- **(I1) Majorisation (the ten Haaf bound viewed as an MM algorithm).**
  - For a violating edge, |H_xy| (b_x^2 rho + b_y^2/rho) >= 2|H_xy| b_x b_y, where rho = a_y/a_x.
  - Hence E_F[b] >= <bs|H|bs>/<b|b> for every b, with equality at b ∝ a_k.
  - So grad E_F = grad <H>_{fixed s} at the guide. Checked: rel. diff 7e-13; gap >= 0 for random b.
  - The ideal amplitude step is therefore an exact majorise-minimise (MM) step on the fixed-sign VMC energy [Hunter & Lange, Am. Stat. 58, 30 (2004)].
  - A truncated inner solve degrades to VMC. The loop's amplitude advantage exists only if the inner solve is nearly complete.
- **(I2) Ground-state transform.**
  - For stoquastic F with F phi = E_FN phi:
    E_F[phi e^v] - E_FN = 1/2 E_{x~phi^2}[ sum_y |F_xy| (phi_y/phi_x)(v_x - v_y)^2 ] + O(v^3).
  - The Hessian in log-amplitude space is therefore the edge-weighted graph Laplacian of F.
  - For b_theta = a_k exp(O.d), this gives the Gauss-Newton matrix A = E_{b^2}[ sum_y |F_xy| (b_y/b_x) (O_x - O_y)(O_x - O_y)^T ].
  - The standard VMC gradient is g = 2 E_{b^2}[(E_loc - E)(O - <O>)], which is approximately -O^T A u* (first order).
  - So the step d = -A^{-1} g is the **edge-weighted least-squares fit of the network to the exact phi_FN**, and it needs no knowledge of phi.
  - This settles the "implicit target" problem in the quadratic regime, which we are in (fidelity 0.99999).
- **(I3) What the FN energy sees.**
  - E_FN[a, s] depends on a only through the violating-edge diagonal.
  - Envelope argument: E_FN[a, s] - E_FN[phi, s] ≈ sum_viol |H_xy| phi_x phi_y (delta log rho_xy)^2.
  - This is the 6x6 oracle finding (ratios on frustrated edges set E_FN) in closed form. The wall in bullet 0.1 is the same object.

## 2. Diagnosis: what is proven vs. hypothesis
| claim | status | evidence (4x4, k0 unless noted) |
|---|---|---|
| Frozen problem is stiff (FN wall) | proven | lambda_max(F) = 2427. Max wall 2429. 672 configurations have wall > 100, with a^2 weight 3.4e-5. |
| First-order methods need many steps | proven | Exact steepest descent / ITE with optimal tau ~ 9e-4: 38% (1 step), 65% (10), 92% (50), 98% (100) of G. Exact SR in log space with oracle step: 36% (1), 53% (3), 81% (10), 99.4% (30). One Euler target (Ledinauskas-Anisimovas) is only 38% here, against 77% at the old late guide. |
| Krylov / power targets need many hops | proven | Rayleigh-Ritz in K_n(F, a): 38% / 47% / 71% / 85% / 93% for n = 1..5 hops. |
| Newton with edge metric solves it | proven (exact) | Edge-GN: 94.6% in 1 step (best step factor 1.00, so the quadratic model is calibrated) and 99.8% in 2 steps. rf: 94.4 / 99.7%. k1: 95.5 / 99.9%. |
| Current protocol (trust 0.02, paired reweighted val gate, stop at first reject) loses most of the gain | proven in tab replica; consistent with the ViT runs | Protocol replica: 0 steps accepted at N = 1e4. Gain -2.3 at 1e5 (the gate accepted energy-raising steps). 0.24 at 1e6. k1 at 1e6: one seed exploded. Same SR direction with an oracle step: 0.52 / 0.89 at 1e5 / 1e6. ViT loop: 0.3-0.5 at 1e7. |
| a^2 sampling is the noise source | proven (tab, rf) | GN with a^2 samples: -0.7 (1e4), 0.23 (1e5, 2 steps). GN with \|b\|^{2 beta}, beta = 1/4: 0.84 / 0.97 / 0.99 (tab, N = 1e4, steps 1-3). rf: 0.66 / 0.85 / 0.90 / 0.92 (λ = 1e-6). With beta = 1/2 the edge weight becomes \|F_xy\| b_y, which is bounded, instead of b_y/b_x, which is not. |
| Energy estimators are heavy-tailed | proven | Paired estimate of E_F[phi] - E_F[a] (true -3.3e-4): std 4.4e-4 at beta = 1, N = 1e5; 2.4e-5 at beta = 1/4, N = 1e5 (z = 14); 8e-5 at beta = 1/4, N = 1e4. [Trail, PRE 77, 016703/016704 (2008)] |
| Loop-relevant gain follows | proven (rf) | Tempered edge-GN, 4 steps x 1e4: next-iteration E_FN after the sign step reaches 84-85% (k1) and ~100% (k0) of the ideal amplitude gain. The violating-edge ratio error drops 10x along with the kept-edge error. |
| Amplitude carries the loop after iteration 1 | proven | Next E_FN with no update vs ideal update: k0 2.27e-4 vs 2.13e-4 (the sign step does most); k1 2.13e-4 vs 1.79e-4 (all from amplitude). Matches the 6x6 oracle. |
| Same gains with the 155k ViT | **hypothesis** | Not tested. Risks: GN linearisation of a deep net, damping choice for P >> N, cost of Jacobians on neighbours. |
| beta < 1 stays usable at 6x6/8x8 | **hypothesis** | The ESS of weights b^{2-2 beta} falls with L. beta = 0 (uniform) is excellent on 4x4 but useless at large L. |

## 3. Literature: what each method would do for us
- **Fitting to an explicit projected target** (supervised updates and power-method + supervised learning):
  - Papers:
    - SWO [Kochkov & Clark, arXiv:1811.12423]
    - infidelity fits [Jónsson, Bauer, Carleo, arXiv:1808.05232; Medvidović & Carleo, arXiv:2009.01760]
    - p-tVMC [Sinibaldi et al., arXiv:2305.14294; Gravina, Savona, Vicentini, arXiv:2410.10720]
    - power method + kernel ridge regression [Giuliani et al., arXiv:2303.08902]
    - fixed Euler-ITE target [Ledinauskas & Anisimovas, arXiv:2307.15521]
  - For us, the target must be p_n(F) a_k. One hop gives only 38% of G at k0, and 93% needs 5 hops; the 6x6 shells hold 1.3e3 states at one hop and 4.9e4 at two.
  - The fidelity estimator's SNR scales as sqrt(I) (Sinibaldi). With I ~ 1e-5 we would need control variates and still very large N.
  - Gravina et al. find that a natural-gradient inner loop is essential. That fits with the stiffness above.
  - Verdict: useful only as a fallback (candidate 3).
- **SR / minSR / SRt / SPRING:**
  - Papers: [Chen & Heyl, arXiv:2302.01941; Rende et al., arXiv:2310.05715; Goldshlager, Abrahamsen, Lin, arXiv:2401.10190]
  - These make Fisher-metric steps affordable for P >> N.
  - Armegioiu et al. [arXiv:2507.10835] show that SR is power iteration projected onto the tangent space. On a stiff F it is first-order and slow; this is our current method.
  - SPRING's Kaczmarz momentum is a principled way to accumulate information over steps.
- **Second-order methods:**
  - Papers:
    - linear method [Nightingale & Melik-Alaverdian, PRL 87, 043401, arXiv:physics/0010066; Umrigar et al., PRL 98, 110201, arXiv:cond-mat/0611094; Toulouse & Umrigar, arXiv:physics/0701039]
    - Newton [Umrigar & Filippi, arXiv:cond-mat/0412634]
    - SR + Hessian [Sorella, PRB 71, 241103, arXiv:cond-mat/0502553]
    - **RGN** [Webber & Lindsey, arXiv:2106.10558]: solves (H̄ + eps^{-1}(S + eta)) d = -g; vanishing-variance estimators; parallel tempering over |psi|^{2i/m}, used only for mixing
    - LM for NQS [Frank & Kastoryano, arXiv:2104.11011]: about 10x fewer iterations than SR, but noisier
    - projected Newton / inverse iteration [Armegioiu et al.]
    - energy natural gradient [Müller & Zeinhofer, arXiv:2302.13163]
  - Our candidate 1 is RGN/LM specialised to a stoquastic operator. Because F is stoquastic, H̄ - E S becomes the PSD edge form (I2). It can be estimated from edges with bounded weights, and needs no eigenproblem.
  - The prior work does not give us a sampling distribution for the stiff region.
- **Heavy tails and sampling:**
  - Papers: [Trail, PRE 77, 016703 and 016704 (2008), arXiv:0909.5505 / 0909.5504; Trail & Maezono, arXiv:1011.4344; Pathak & Wagner, arXiv:2002.01434]
  - These fix infinite variance near nodes by changing the sampling or regularising the estimator. Our tempered |b|^{2 beta} is the same idea, applied to the FN wall.
- **QMC-data-driven and FN-guided NQS:**
  - Papers:
    - [Inack et al., PRB 98, 235145, arXiv:1809.03562]
    - [Pilati, Inack, Pieri, arXiv:1907.00907]: the guide is retrained by KL divergence to walkers; sign-free Ising only
    - [Gu et al., arXiv:2507.02644]: transformer as an FN-GFMC trial for Hubbard, one-shot
    - [Lin et al., arXiv:2406.12207]: PEPS-guided GFMC on J1-J2
    - data-enhanced VMC [Czischek et al., arXiv:2203.04988; Moss et al., arXiv:2308.02647]
  - We found no iterated NQS -> FN -> retrain loop on J1-J2.
  - Our walker MLE and density-ratio attempts failed for a structural reason: walker densities give phi as a *density*, with no zero-variance property. Resolving a 0.4% correction needs ~(1/0.004)^2 ≈ 6e4 effective samples per resolved degree of freedom. E_loc gives A u* with noise proportional to the correction.
- **Fixed-point acceleration:**
  - Papers: Anderson / DIIS [Walker & Ni, SIAM J. Numer. Anal. 49, 1715 (2011); Pulay, CPL 73, 393 (1980)]
  - Exact 4x4 FN error after 12 iterations: plain 3.6e-5, omega = 2 1.1e-5, Anderson(2-5) 1.1-1.2e-5. To reach 5e-5 takes ~9.5 iterations plain and ~5.5 with omega = 2.
- **FN background:**
  - Papers: [van Bemmel et al., PRL 72, 2442 (1994); ten Haaf et al., PRB 51, 13039, arXiv:cond-mat/9412037; Sorella & Capriotti, arXiv:cond-mat/9902211]

## 4. Candidates (ranked)
**C1. Tempered edge Gauss-Newton (a "stoquastic RGN"). Recommended.**
- Model: b_theta = a_k exp(r_theta), residual network, warm-started.
- Per GN step:
  1. Draw N samples x ~ |b|^{2 beta} by MCMC (a change of acceptance exponent in the existing sampler), with self-normalised weights w ∝ b^{2-2 beta}.
  2. For each x, take all H_FN neighbours y. These are needed for E_loc anyway, and the sign step needs them too.
  3. Estimate g = 2 sum_x w_x (E_loc(x) - Ē)(O_x - Ō).
  4. Estimate A = sum_x w_x sum_y |F_xy| (b_y/b_x)(O_x - O_y)(O_x - O_y)^T. With beta = 1/2 the weights are bounded by |F_xy| b_y.
  5. Solve (A + λ D) d = -g with CG, using matrix-free products ΔJ v = (Jv)[x] - (Jv)[y]. Alternatively use the sample-space dual over N·k edge rows, minSR-style.
- 2-4 GN steps per iteration. Fresh samples each step. **No validation gate.**
- Damping λ: Levenberg-Marquardt on the ratio of predicted gain (-g.d/2) to gain measured on an independent tempered paired sample.
- Then the sign step. Optionally relax the outer step to log a_{k+1} = log a_k + omega (log b - log a_k), omega ≈ 1.5-2.
- Cost per GN step: about one SR step plus Jacobians on the neighbours. On 4x4 the unique set {x, y} is at most D. On 6x6/8x8, subsample k neighbours per x (an unbiased edge estimate), giving roughly (1+k) times the SR Jacobian cost.
- Why it should reach >= 90%: exact 1-2 steps give 95-99.8%. With sampling (rf, N = 1e4 per step, 4 steps) it gives 92% frozen gain and 85-100% of the loop-relevant gain, against 0-24% for the current protocol at N <= 1e6.
- Failure modes and what to watch:
  - GN linearisation of the ViT: watch the predicted/measured ratio.
  - λ too large: rf drops from 0.92 at λ = 1e-6 to 0.44 at λ = 1e-4.
  - ESS collapse of tempered weights on large lattices.
  - Neighbour-Jacobian cost.
  - MCMC autocorrelation at beta < 1.
- Scaling: neighbours are part of the training data, which addresses the 8x8 finding that more than 99% of one-hop ratios are never sampled. If ESS(beta) collapses, a local alternative is a defensive mixture: x ~ b^2 plus one-hop moves across violating edges, with density known from one hop. **Untested.**

**C2. Minimal repair of the current SR loop (cheapest code change).**
- Changes:
  - tempered sampling
  - no first-reject stop and no paired-reweight gate
  - step length from the quadratic model, t* = -g.d / d^T A d, on independent samples
  - 10-30 steps
- tab with oracle step and beta = 1/2: 0.52 / 0.72 / 0.92 at N = 1e4 (2/4/10 steps), and 0.79 / 0.90 / 0.98 at 1e5.
- rf: only 0.45-0.63 after 10 steps. The stiffness makes SR need 5-10x more steps than C1.
- Use it as the control for C1.

**C3. Explicit Krylov target with an edge-difference GN fit** (Kochkov-Clark / Giuliani-type supervised fit, using the C1 metric).
- Use only if C1's implicit target fails, for example through GN nonlinearity.
- One hop gives only 38% of G; going beyond two hops is unaffordable on 6x6.

Not recommended:
- walker MLE / density-ratio fitting, or fitting phi_FN from long walker histories (no zero-variance property, replica dependence; ledger)
- Adam on pointwise or ITE targets (45-66%, drift)
- any gate built on reweighted validation estimates, which can be exploited (rung 2 and here)

## 5. Smallest decisive test, and the stop rule
**T1 (single frozen problem, ViT, about 1 CPU-h).**
- Setup: net-H k0 frozen problem; existing exact i.i.d. sampler, with q = a^{2 beta} for beta in {1/4, 1/2}; 155k residual ViT.
- Run: 4 C1 steps at N = 1e4 per step, 2 seeds. Same-N controls: C2, and the current protocol.
- Report: exact frozen gain per step; predicted/measured gain ratio; next-iteration E_FN.
- **Drop C1 if** the frozen gain after 4 steps is < 0.6 at N = 1e4 and < 0.8 at N = 1e5 for every beta/λ tried (rf gives 0.92 / 0.94).
- If the predicted/measured ratio is far from 1, the problem is ViT nonlinearity: try last-layers-only GN, or C3.

**T2, if T1 passes (10-iteration loop).**
- Setup: net H, N = 1e4 per GN step, 3 steps per iteration; omega in {1, 2}.
- Pass: eps_FN(it 10) <= 1e-4 at <= 5 CPU-h. For reference, ideal is 4.6e-5; the current best is 1.6e-4 at 25 CPU-h; equal-cost VMC is ~1.8e-4.
- Fail: >= 1.5e-4.
- Fairness: equip the VMC control with the same tempered edge-GN, because the gradients coincide (I1).

**6x6 pre-check, which needs no learning.** With the existing sampler at beta = 1/2, measure the variance of the paired E_F difference on the it2 frozen problem. If tempering does not cut the std by >= 5x, as it does on 4x4 (18x), C1's premise fails at 6x6.

## 6. On the coordinator's points (6x6 it2)
1. **Near-ideal contraction.** This needs the inner solve to be essentially complete (I1). C1 is the only option tested here that does that from N ~ 1e4 per step.
2. **Cost for 5+ iterations.** C1 replaces about 20 noisy SR steps with 2-4 GN steps. omega = 2 roughly halves the iterations.
3. **Referee for ~1e-5 per site.**
   - Per iteration, certify with the bound chain E_FN[b, s] <= <bs|H|bs> <= E_F[b]. That means a tempered, paired VMC estimate of E_F[b_new] - E_F[a_k].
   - On 4x4 this resolves the 2.1e-5/site k0 gain at z = 4 with 1e4 samples and z = 14 with 1e5 (beta = 1/4); a^2 sampling gives z = 0.3 / 0.8.
   - Run the GFMC referee (M >= 512, medians, same seeds) only every 3-5 iterations, on accumulated gains.
4. **Accumulating walker information.**
   - Not as the primary route; see §3.
   - Information is better accumulated across GN steps and iterations: a running average of A over iterations (it changes slowly while g must stay fresh), SPRING-type momentum, and outer relaxation or Anderson mixing. Only the latter has been tested.

## 7. Open questions for the lead
1. Should the inner objective stay "frozen gain fraction"? omega = 2 shows that overshooting the MM step helps the outer loop. A better scoring metric may be the outer contraction per CPU-h.
2. Is the cost of the neighbour Jacobians acceptable on 6x6? If not, which neighbour subsampling k is acceptable? Or should GN be restricted to the last ViT blocks?
3. ESS(beta) at 6x6/8x8, and whether the existing MCMC can sample |b|^{2 beta} cheaply.
4. Given (I1), should the paper's VMC baseline also get tempered edge-GN? An honest comparison needs it, and it may close the loop's advantage.
5. Should we fine-tune the ViT itself, or keep growing a residual chain (evaluation cost per hop)?

Files:
- `experiments/amp_design/amp_design_4x4.py` (main), `amp_design_variants_4x4.py` (beta/λ scans), `outer_accel_4x4.py`, `paired_referee_4x4.py`, `loop_metric_4x4.py`
- Outputs: `results/amp_design/*.json|log`
- The main run was stopped after k0/k1 (k1stale exact rows only; k5 not run). One seed set; std over 2-3 seeds is in the logs.
