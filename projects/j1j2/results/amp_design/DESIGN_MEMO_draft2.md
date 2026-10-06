# Amplitude update for the FN/Krylov loop: design memo (draft 2, 2026-10-06)

Scope: step 2 of the loop, the amplitude refresh. Draft 2 adds Test A (FN surrogate vs a good VMC optimiser) and Test B (6x6 sampling pre-check), and revises the recommendation. Draft 1 is kept in `DESIGN_MEMO_draft1.md`.

Testbed:
- 4x4 tests are exact, CPU only, about 3 CPU-h in total.
- They use linear log-models: `tab` has one parameter per translation orbit (P = 822); `rf` has 1000 symmetric random features and must generalise.
- **None of them uses the ViT.**
- The 6x6 pre-check used 0.37 A40-h.

Notation:
- F = H_FN[a_k, s_k]
- E_F[b] = <b|F|b>/<b|b> is the frozen surrogate
- E_H[b] = <bs|H|bs>/<b|b> is the fixed-sign VMC energy
- States: k0 = net H with its own signs; k1 = net-H amplitude refreshed (ideal FN), with the first Krylov signs (w_s 2e-6)
- Score: eps of the next-iteration E_FN after the exact sign step

## 0. Bottom line (revised)
1. **The FN surrogate is a worse amplitude objective than the fixed-sign VMC energy.** E_F majorises E_H and touches it at the guide (I1). The ideal FN refresh is therefore one conservative majorise-minimise (MM) step on E_H. The "stiff wall" diagnosed in draft 1 is the curvature of that majoriser, not of the physics.
   - With exact data, 4 Rayleigh-Gauss-Newton (RGN) steps on E_H at k1 reach eps_FN_next = 3.8e-5 (rf) and 1.7e-5 (tab).
   - The *ideal* FN refresh reaches only 1.79e-4. At k1, 30 RGN steps reach 6e-6.
   - Draft 1's C1 (Gauss-Newton on E_F) can at best reproduce the ideal FN step. **C1 is withdrawn as the lead.**
2. **The PSD claim survives only in a weaker form.** Both parts below use tempered samples:
   - The part of the stoquastic structure that helps is the edge Laplacian over the *sign-respecting* edges, used as a **preconditioner for the VMC gradient** ("hybrid").
   - It is robust at N = 1e4 per step, where plain RGN on E_H blows up or does nothing. Hybrid beats C1 in every state and model (table, §2).
   - At N = 1e5, plain RGN on E_H beats both by 4-10x when the signs are good.
   - The FN wall is not needed for this.
3. **What the loop adds is the sign step, not the amplitude objective.** RGN on E_H with the network's own signs (k0) reaches 1.76e-4 at N = 1e5. After one Krylov sign step (k1), the same optimiser reaches 4.0e-5. Fixed-sign second-order VMC pays off only once the signs are right.
   - Proposed loop: Krylov sign step (one hop), then fixed-sign second-order VMC amplitude update, with FN used only as the referee and bound.
4. **The 6x6 sampling premise fails as stated.**
   - At 6x6, tempering (beta = 1/2) cuts the paired-difference std only 1.3-1.45x, against the 5x I asked for, and beta = 1/4 is worse. ESS/N is 5% and 1.6%.
   - The unpaired std of E_F[a] does drop 2.5x at beta = 1/2.
   - The candidate b (SR it2 "a2small") has RMS δlog b of 0.11 in the tempered region, against 0.006 on a^2. Uncontrolled extrapolation to low-|a| configurations may be what dominates the paired variance; this is the rung-2 exploit seen from the sampling side.
   - At 4x4 with beta = 1 and N = 1e4, nothing works: every method is within 0-10% of no update.
   - **Resolving and certifying ~1.5e-5/site per iteration at 6x6 is the binding constraint.** It is not settled.
5. Unchanged from draft 1:
   - Outer over-relaxation with omega = 2, or Anderson mixing, roughly halves the number of exact-loop iterations.
   - Validation gates built on reweighted estimates can be exploited.

## 1. Identities
- **(I1) Majorisation.**
  - For violating edges, |H_xy| (b_x^2 rho + b_y^2/rho) >= 2|H_xy| b_x b_y, where rho = a_y/a_x.
  - Hence E_F[b] >= E_H[b], with equality and equal gradient at b ∝ a_k. Checked: gradient difference 2e-16.
  - Consequences:
    - The ideal FN refresh is an MM step [Hunter & Lange, Am. Stat. 58, 30 (2004)].
    - Iterating FN refreshes at *fixed* sign converges to a stationary point of E_H.
    - min_b E_H[b] <= E_FN[a_k, s_k] in full capacity.
- **(I2) Ground-state transform.** For stoquastic F: E_F[phi e^v] - E_FN = 1/2 E_{phi^2}[sum_y |F_xy| (phi_y/phi_x)(v_x - v_y)^2] + O(v^3).
- **(I3) Hessian of E_H in log-amplitude space.** Checked against finite differences to 2e-4 relative.
  - M = 4 E[(E_loc - E) δO δO^T] + E[sum_y (-H^s_xy)(b_y/b_x) ΔO ΔO^T], with H^s = diag(s) H diag(s).
  - Violating edges (H^s_xy > 0) enter with *negative* weight.
  - RGN [Webber & Lindsey, arXiv:2106.10558] keeps 2E[...] in the first term.
  - At k0, M is PSD but nearly singular in directions that cross violating edges. Undamped Newton steps there blow up; in `tab`, max|d| = 41.
  - The kept-edge part A_+ = E[sum_{kept y} |H_xy| (b_y/b_x) ΔO ΔO^T] is PSD and well conditioned. This A_+ is exactly draft 1's A_F: the kept off-diagonals of F are those of H^s.

## 2. Test A: FN surrogate vs a good VMC optimiser (4x4, k0 and k1)
Protocol:
- Same model; samples x ~ |b|^{2 beta}, with weights b^{2-2 beta} and fresh samples each step.
- Same N and 4 steps for every method, then the same exact Krylov sign step.
- Methods:
  - **C1:** (A_F + λ)^{-1} g_F.
  - **hybrid:** (A_F + λ)^{-1} g_H. g_H is the VMC gradient; g_F = g_H at step 1.
  - **RGN:** (M_RGN + μS + λI)^{-1} g_H, grid over μ, λ and an optional S-norm trust cap.
  - **RGNA:** (M_RGN + μ A_F)^{-1} g_H.
- The linear method (non-symmetric estimator) was too ill-conditioned with near-singular S. It needs the usual S-normalisation, so it is omitted.
- Two seeds. Each cell is the best member of the method's grid, with no oracle line search.

**eps of next-iteration E_FN** (×1e-4; beta = 1/4 unless stated; "no upd." = sign step only):

| state, model | no upd. | ideal FN step | C1 N=1e4 / 1e5 | hybrid N=1e4 / 1e5 | RGN N=1e4 / 1e5 | RGN exact, 4 steps / 30 steps |
|---|---|---|---|---|---|---|
| k0, rf | 2.27 | 2.13 | 2.14 / 2.13 | 2.04 / 2.00 | 2.08* / 1.76 | 1.60 / 1.55 |
| k0, tab | 2.27 | 2.13 | 2.14 / 2.13 | 1.99 / 1.99 | 2.08 / 1.45 | 1.46 / – |
| k1, rf | 2.13 | 1.79 | 1.85 / 1.84 | 1.51 / 1.47 | 1.91 / **0.40** | 0.38 / **0.062** |
| k1, tab | 2.13 | 1.79 | 1.79 / 1.79 | 1.25 / 1.18 | unstable / **0.17** | 0.17 / – |

\*k0 rf: <H> got worse (eps 3.1e-4 vs 2.6e-4); the gain is in E_FN only.

beta and RGNA at N = 1e4:
- beta = 1/2: C1 1.86-1.91 at k1, hybrid 1.60-1.69, RGNA(μ=0.3) 1.26 (rf) but 2.00 (tab), i.e. erratic.
- beta = 1: no method gains at k1 (2.11-2.13); RGN and RGNA(0.3) diverge.

Reading:
- FN-surrogate objective < VMC objective in every column with N >= 1e4 tempered.
- The advantage of the stoquastic structure is robustness: the A_F preconditioner. At small N, a well-tuned RGN cannot use the larger gain because its curvature estimate across violating edges is noisy and nearly singular.
- **Untested:** whether RGN with an adaptive trust region accepted on fresh validation samples closes the 1e4 gap. That is the natural next optimiser, between hybrid and RGN.

## 3. Test B: 6x6 pre-check
Setup: it2 frozen problem, G1 = (a1, stored net + hop). ViT at full fp32 with 154,980 parameters. N = 16384 per beta. 1024 Metropolis chains, autocorrelation negligible. Errors are jackknife over 16 chain groups.

| beta | acceptance | ESS/N | std E_F[a] /site | paired D̂ = E_F[b] - E_F[a] /site | std(D̂) ratio vs beta=1 | wall median / 99% / max |
|---|---|---|---|---|---|---|
| 1 | 0.14 | 1 | 4.0e-5 | -5.1(2.4)e-5 | 1 | 2.5 / 11 / 93 |
| 1/2 | 0.27 | 0.053 | 1.6e-5 | +2.0(1.8)e-5 | 0.78 (0.69 chain-jk) | 5.5 / 51 / 395 |
| 1/4 | 0.37 | 0.016 | 5.1e-5 | +3.9(3.2)e-5 | 1.34 | 9.8 / 173 / 1947 |

- b = SR it2 candidate "a2small". The D̂ means disagree across beta at the 2.4σ level, which suggests heavy tails. No D̂ is resolved.
- Neighbours per x: 81.6 in total, 23.6 of them violating, mostly J2 edges.
- A40 timings per sample:
  - forward pass on all neighbours: 0.29 ms
  - gradient at x only: 0.028 ms
  - gradient at x plus violating neighbours: 0.64 ms
  - gradient at x plus all neighbours: 2.14 ms
- Jacobians must be contracted (VJP or chunked), not stored: 620 KB per state.
- Cost is dominated by the local data of the hop-sign guide, about 300 s per 16k samples, not by Jacobians. With kept-edge subsampling at k = 8, the A_+ products cost about 0.25 ms per sample.
- Code: `experiments/amp_design/precheck_6x6.py`; output in `precheck_6x6.json`; ws1 `~/chatty_amp_design/run_16848750/`.

Reading:
- Tempering helps the plain estimator (2.5x less std at beta = 1/2) but not this paired difference.
- The ideal per-iteration gain is about 1.5e-5/site. At beta = 1, N = 16k, the paired std is 2.4e-5. A 3σ certificate needs std 5e-6, i.e. about 3.7e5 samples, if the error falls like 1/sqrt(N) (heavy tails make that optimistic). That is about 2 A40-h of local data per certificate.
- Per-iteration certification at 6x6 is therefore not affordable. Gains must be certified cumulatively.

## 4. What the loop adds beyond a good VMC optimiser
1. **Signs, at one-hop cost.**
   - A single Krylov step takes w_s from 2.4e-4 to 2e-6 on 4x4 (every VMC run kept w_s >= 6e-5) and from 1.3e-4 (ViT) to 4e-5 on 6x6.
   - Test A shows this is what makes second-order amplitude optimisation pay: RGN gets 1.76e-4 at k0 versus 0.40e-4 at k1 (rf, N = 1e5).
   - From poor starts (rung 2), the sign step is the whole game.
2. **A variational referee that is lower than <H>.** E_FN <= <H> for any guide, about 10-25% lower in eps here. The majorisation chain E_FN <= <bs|H|bs> <= E_F gives a cheap one-sided check.
3. **A PSD preconditioner from the stoquastic part.** It needs H only, not F. Test A supports it as a robustness device at small N, not as a source of extra gain.
4. **Not supported:** the FN surrogate as the amplitude objective; this was draft 1's C1.

Implications for the paper:
- The honest baseline is "same network, same second-order fixed-sign VMC optimiser, network signs" against "same, Krylov signs".
- The coordinator's 6x6 oracle says that at ViT quality the signs no longer limit FN. A 6x6 win must therefore come from the amplitude optimiser, so it would be a VMC-optimiser result, unless the Krylov signs make fixed-sign RGN converge where ViT-SR stalled. That is what k0 vs k1 shows on 4x4.

## 5. Literature (verified; what it does for us)
- **RGN / Newton / linear method:**
  - Papers:
    - Webber & Lindsey, arXiv:2106.10558
    - Umrigar et al., arXiv:cond-mat/0611094
    - Toulouse & Umrigar, arXiv:physics/0701039
    - Nightingale & Melik-Alaverdian, arXiv:physics/0010066
    - Umrigar & Filippi, arXiv:cond-mat/0412634
    - Sorella SRH, arXiv:cond-mat/0502553
    - Frank & Kastoryano (LM for NQS), arXiv:2104.11011
    - Armegioiu et al. (SR = projected power iteration; RGN = projected Rayleigh-quotient iteration), arXiv:2507.10835
  - Test A confirms their picture on our problem: second-order methods on E_H converge in a few steps once noise allows.
  - Our addition: the kept-edge stoquastic Laplacian as a PSD damping or preconditioner, plus tempered sampling.
- **Infidelity / projected fits:**
  - Papers: Kochkov & Clark, arXiv:1811.12423; Jónsson, Bauer, Carleo, arXiv:1808.05232; Medvidović & Carleo, arXiv:2009.01760; Sinibaldi et al., arXiv:2305.14294; Gravina, Savona, Vicentini, arXiv:2410.10720; Giuliani et al., arXiv:2303.08902; Ledinauskas & Anisimovas, arXiv:2307.15521
  - A one-hop Euler target gives only 38% of the frozen gain. The fidelity SNR scales as sqrt(I).
  - Not recommended.
- **SR at scale:** minSR, arXiv:2302.01941; SRt, arXiv:2310.05715; SPRING, arXiv:2401.10190. These are the baseline optimisers. SPRING-style momentum is the natural way to accumulate A over steps.
- **Heavy tails:** Trail, arXiv:0909.5504 / 0909.5505; Pathak & Wagner, arXiv:2002.01434. These are relevant to §3; a regularised estimator may beat tempering at 6x6.
- **NQS and projective QMC:** Inack et al., arXiv:1809.03562; Pilati et al., arXiv:1907.00907; Gu et al., arXiv:2507.02644; Lin et al. (PEPS-guided GFMC on J1-J2), arXiv:2406.12207. No iterated NQS → FN → retrain loop on J1-J2 found.
- **Fixed node:** van Bemmel et al., PRL 72, 2442; ten Haaf et al., arXiv:cond-mat/9412037; Sorella & Capriotti, arXiv:cond-mat/9902211.
- **Acceleration:** Walker & Ni 2011 (Anderson); Pulay 1980 (DIIS).

## 6. Concrete plan
**T1 (4x4, ViT, CPU, about 3 CPU-h): amplitude optimiser at fixed Krylov signs.**
- Start: net H (23.7k-parameter ViT), fine-tuned directly with a warm start. Guide: k1 signs (first Krylov step on phi_FN[net H]), plus k0 signs as the control.
- Exact sampler at beta in {1/4, 1/2}; N in {1e4, 1e5}; 6 steps; 2 seeds.
- Optimisers:
  - SR (current, tempered, no gate)
  - hybrid (A_+ preconditioner)
  - RGN with A_+ damping and a trust region accepted on fresh validation samples
  - C1 (reference)
- Score: eps_FN_next and eps<H>.
- **Pass:** some optimiser reaches <= 1.2e-4 at N = 1e4 and <= 6e-5 at N = 1e5 at k1. The ideal FN step gives 1.79e-4; rf gives 1.5e-4 and 4e-5.
- **Fail:** nothing beats 1.79e-4. Then ViT nonlinearity or noise dominates: fall back to hybrid with more steps, or last-blocks-only.

**T2 (4x4, ViT loop, about 10 CPU-h).**
- 10 iterations of Krylov sign step + best-T1 optimiser (omega in {1, 2} on the amplitude change).
- Control: the same optimiser with net signs (plain second-order VMC) at equal samples and CPU-h.
- Score: eps_FN per CPU-h.
- **Pass:** loop <= 0.5x the control at equal cost. **Fail:** within 0.8x; then the sign step is not worth its cost on 4x4.

**T3 (6x6 at the ViT, GPU), only if T2 passes.**
- (a) Noise budget, about 1.5 A40-h: train b with one hybrid step from G2 at beta = 1 and 1/2. Measure the paired std of E_H[b] - E_H[a] at N = 64k (it should fall with N). Also try a Pathak-Wagner-type regularised estimator.
  - Go only if the extrapolated N for 3σ on 1.5e-5/site is <= 2e5.
- (b) Three iterations: stored net + hop signs, then 4 hybrid/RGN steps at N = 3.2e4 (about 10 min/step on an A40, dominated by local data), with omega = 2.
  - About 3 GPU-h per iteration, including the sign step at about 1.1 GPU-h. Certify cumulatively only (one paired check at the end, about 2 GPU-h).
- (c) FN referee: M = 512, 64 populations, same seeds as the ViT guide, medians. About 5 A40-h.
- **Pass:** E_FN below the ViT guide's FN by >= 2σ (about 5e-5/site).
- **Fail:** < 1σ after 3 iterations, about 20 GPU-h in total. Then the paper claim becomes "Krylov sign correction matches the best network at one-hop cost".

## 7. Open questions for the lead
1. Do we accept dropping the FN surrogate as the amplitude objective, so that the FN role becomes signs plus referee? This changes the paper's framing of "the FN/Krylov loop".
2. Is the T2 control (same optimiser, net signs) the right baseline, or should it also learn signs with a complex ViT, as net H did?
3. 6x6: is a cumulative certificate (one paired check per 3 iterations, plus a final FN) acceptable to you as a referee argument?
4. Should RGN-with-trust-region on fresh validation samples be prototyped first on the linear models (about 0.3 CPU-h), before T1?

Files:
- 4x4: `vmc_vs_fn_4x4.py` (Test A), `amp_design_4x4.py`, `amp_design_variants_4x4.py`, `loop_metric_4x4.py`, `outer_accel_4x4.py`, `paired_referee_4x4.py`
- 6x6: `precheck_6x6.py`
- Outputs: `results/amp_design/*.json|log`; Test A logs are `vmc_vs_fn_*`.
- Draft-1 evidence (stiffness, tempered C1 scans, outer acceleration, 4x4 paired referee) is in `DESIGN_MEMO_draft1.md` and the logs.
