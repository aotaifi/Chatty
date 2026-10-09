# J1-J2 FN/Krylov project: research map (living document, updated 2026-10-07)

One page that holds the thread of our discussions. Details live in CLAUDE_TAKE.md (log) and results/*/README.md.

## 1. What would make this worth publishing (PI's criteria)
1. **Scale:** new results where other methods have none. Survey (results/literature/J1J2_BENCHMARKS_COST.md):
   - 8x8: an energy below -0.49889 (RBM+PP -0.498886) would be new.
   - 12x12, 14x14, 18x18: no deep-NQS results at all (the clearest gaps).
   - 10x10: saturated (record -0.4976921); only a cost win or a controlled extrapolation counts.
2. **Cost:** cheaper than brute-force NQS at equal accuracy. Bar: the 10x10 ViT took ~1.9e3-3.8e3 A100-h.
3. **Separate pitch:** FN/Krylov repairs the signs of bad or imperfect NQS guides ("enhance your NQS").

## 2. Established (with the file that holds the evidence)
| Result | Where |
|---|---|
| Krylov sign step (label-free, energy-optimal T): exact signs in 3-6 steps given the exact amplitude; depth grows slowly with N (16 -> 36 sites) | paper, sign_design_tests |
| Exact FN/Krylov loop reaches the GS; iteration count flat 16 -> 36 sites; Anderson ~3x faster | paper |
| Bad starts: any in-sector GS content -> GS; exact eigenstates are fixed points; other symmetry sectors trap; Anderson unsafe from bad starts | bad_start/ |
| Stored signs: composite sign net + ONE exact hop = recursion at fixed cost; train on tempered samples + one-hop neighbours (self-sealing); Krylov step factors are unlearnable, composite signs are learnable | stored_signs*/, sign_learnability/ |
| 6x6 signs: composite net + hop w_s 2e-5 vs ViT 1.3e-4 | amp_design/T3 |
| FN-surrogate amplitude objective = one majorise-minimise step on fixed-sign VMC; second-order fixed-sign VMC beats it 4-10x | amp_design/DESIGN_MEMO.md |
| 4x4 from a trained net: Krylov sign + second-order fixed-sign VMC beats the best tuned VMC (signs+amplitude) 3.3-7.2x at equal CPU-h | amp_design/vit_t1t2* |
| 6x6 from the ViT: energy tie; near the ViT the remaining FN error is the amplitude, not the signs | learned_loop_6x6_it2, amp_design/T3 |
| Corrections: our ViT is NOT state of the art at 6x6/8x8 (RBM+PP is better) | CLAUDE_TAKE 2026-10-07 |

## 3. Open questions, in the order we attack them
1. ~~Why does the loop stall at the ViT level on 6x6?~~ **Answered 2026-10-07 (stall_6x6/README.md):** not information (ideal exact loop from the symmetrised ViT passes RBM+PP in 3 iterations, 8.8e-6 at 12) and not signs; the network cannot absorb the FN update accurately enough: energy cost 0.33 sigma^2/site, so each update must be written back accurately (0.004 rms is the white-noise case; the real criterion is the edge-difference energy form Q(e) <~ 5e-6, review finding 4). Symmetrisation is worth 2.4e-5 for free. **New open question 1: an amplitude representation/optimiser that reaches 0.004 rms.**
2. ~~Do large-sample amplitude steps beat their own control?~~ **Answered 2026-10-07: no.** Large N fixes generalisation (steps verify at z 2-8), but at the ViT amplitude the fixed-sign optimum is within ~1e-5/site; loop arm ties, control D gains -0.35(0.31)e-5 (amp_design/AMP6_table.md).
3. **Cost race from scratch:** Marshall + amplitude-only net + Krylov signs vs training a full NQS, total GPU-h to a target energy (6x6 validation -> 8x8 target < -0.49889). Waits for 1 (architecture choice).
4. **Add-on pitch at scale:** one Krylov sign step on the public 10x10 ViT checkpoint (nqs-models); cheap check.
5. **Baseline to beat for any energy claim:** Lanczos step + variance extrapolation (Hu et al. 2013).

## 3b. Independent review 2026-10-07 (results/review/REVIEW_2026-10-07.md) — accepted points
- Critical: at a VMC-converged network the FN amplitude step has zero first-order gain (majorisation); every amplitude test needs a same-capacity VMC control.
- Critical: 4x4 "3.3-7.2x at equal CPU-h" is not fair/transferable (exact sign table on 4x4; CPU-h implementation-dependent; 1.2-1.8x at equal step count). Do not claim it as is.
- Critical: one Lanczos step is the untested baseline at equal hop cost. Exact 6x6 comparison running (results/lanczos_baseline_6x6/).
- Major: FN/DMC referee biased by ~1e-5 at M=128; calibrate vs exact 6x6 FN before any 1e-5 claim.
- Major: symmetrisation gain 2.4e-5 is ~1 sigma vs the ViT's VMC error; costs 16x per evaluation.
- Reviewer's viability verdict: not viable as a ground-state solver at 8x8-10x10; narrow add-on survives only if it beats a Lanczos step at equal hop cost.

## 3c. Pre-registered verdict thresholds (reviewer, 2026-10-07, set BEFORE the results)
- Lanczos vs Krylov sign (exact 6x6, equal hop cost): toward viable if sign-only recovers >= 50% of the Lanczos gain, or Krylov+Lanczos beats Lanczos alone by >= 2e-5; kill the sign-only add-on if Lanczos wins >= 5x.
- Write-back with same-capacity VMC control: upgrade only if the FN-target arm captures >= 0.5 of the gain AND beats plain VMC >= 1.5x at equal samples; tie -> drop the FN target (sign step and FN bound survive); < 0.1 -> cost case worsens.
- Referee: DMC - exact FN bias < 3e-6 at fixed protocol (M >= 512) and shrinking ~1/M -> FN bound usable at 1e-5; guide-dependent or non-shrinking -> withdraw DMC-based 6x6/8x8 differences.
- Surviving regimes and the evidence required: sign-poor guides (8x8+, >= 3x cheaper than complex VMC to the same energy, all costs, >= 3 seeds, calibrated referee); Krylov+Lanczos must beat two Lanczos steps; FN as a calibrated bound on a strong hop-free guide (8x8 below -0.49889 or first error-barred 12x12/14x14).
- Killed now: 4x4 equal-CPU claim; more write-back into the converged ViT; from-scratch cost race at 8x8/10x10 until the write-back + VMC-control test passes; histogram/MLE routes; "beat the ViT" and w_s as figures of merit.

## 3d. Lanczos baseline verdict (exact 6x6, results/lanczos_baseline_6x6/, 2026-10-07)
- From the symmetrised ViT psi_P (1.32e-4): one Lanczos step 2.55e-5, two steps 8.6e-6; one Krylov sign step 1.27e-4. Lanczos wins 20x in gain -> per the pre-registered threshold the sign-only add-on is KILLED. Krylov+Lanczos (3.6e-5) is worse than Lanczos alone.
- FN with the Lanczos-p1 guide: 1.89e-5 (below RBM+PP 4.5e-5). Candidate route: strong NQS guide + Lanczos step(s) + calibrated FN at 8x8+ (check novelty: FN with Lanczos-step guides exists for Gutzwiller states, Sorella 2001 / Becca et al.).

- PI caveat (2026-10-07): the 20x is one size. A fixed number of Lanczos steps is a global operation whose per-site gain is expected to shrink with N; the Krylov sign step acts per configuration and may stay size-intensive. Exact scaling test N = 16..36 running (results/lanczos_vs_krylov_scaling/). Result (results/lanczos_vs_krylov_scaling/, f3d10a9): up to N=36 one Lanczos step removes a fixed fraction (0.73-0.92) of the start error at every N, no shrinkage; the Krylov step removes only the sign part of the error (0 for sign-exact guides, 0.84-1.0 for Marshall-type starts, where it beats Lanczos by ~1.2x). No evidence of Krylov overtaking; Lanczos shrinkage expected only for N*dE > gap (N ~ 1e2-1e3), untested. Kill stands for good guides; the sign step matters only for sign-dominated guides.

## 3e. Write-back verdict (results/writeback/DESIGN.md, 2026-10-07) — INCONCLUSIVE (Adam optimizer, not converged; re-test with SR/minSR/Gauss-Newton running)
- The FN amplitude update is white-noise-rough in the energy metric (0.97 of white noise) and tail-concentrated (81% of the gain on 7.6% of the weight). Exact oracle data: energy-metric residual net keeps 17% of one FN iteration, infidelity 8%; with realistic VMC samples < 1%, tying plain fixed-sign VMC.
- Decision: stop writing FN amplitude updates back (no loop, no 8x8 write-back). Amplitude from ordinary VMC; FN/Krylov for signs and as referee. Reopen only if a non-local pair-product residual or stronger tail tempering passes 50% on the same oracle harness.
- Best remaining route: strong (symmetrised) NQS + Lanczos step(s) + calibrated FN at 8x8+ (3d), pending referee calibration.
- 2026-10-08 REOPENED (PI + Research Workspace seq100): the loop's novel part is the alternating construction (Krylov sign -> flatten composite sign -> FN amplitude refinement, calibrated referee); NQS+Lanczos and NQS+FN alone are known (Chen et al. NeurIPS 2022; Wang/He/Lu PRB 113, 085120 (2026); NQS->FN-GFMC on Hubbard/t-J). Write-back failed with |psi|^2-type sampling; new test samples the ENERGY METRIC (tail proposal ~ violating/kept-edge weight) and fits edge differences with FN Laplacian weights. Reopen rule: >= 50% of one FN iteration on oracle data (results/writeback_tail/).

## 3f. Tail concentration vs N (results/tail_vs_N/, 2026-10-08)
- For loop guides the weight fraction carrying half the FN gain is flat in N (12-20%, 16..36 sites); it is NOT a size trend. But the number of configurations to be corrected explodes (82 orbits at 16 -> 1.6e7 orbits at 36), so the 4x4 sampled success was memorisation of the whole support; at 6x6+ the tail must be learned by generalisation.
- The better the guide, the more tail-heavy the gain (k=1,3,8 at 36: 19.9 -> 11.6 -> 7.1%; the symmetrised ViT 1.0% carries 50%). Write-back is hardest near convergence, easiest from mediocre guides.
- Decisive next: the learnability test (held-out tail orbits vs shuffled-target control), running in writeback_tail.

## 3g. Imaginary-time fitting with H_FN (results/itfit_6x6/, 2026-10-08) — INCONCLUSIVE, cause identified
- Stiffness SOLVED: lambda_max(F) = 1.6e6; explicit steps hopeless, but a semi-implicit step (wall implicit, kept hops explicit; local, positive) captures 87% of an FN iteration in one step, 97% in two; the unprojected loop with it tracks the ideal loop within 5%.
- Fitting a zero-hop residual CNN: <= 3.7% (oracle 8.3%), not converged. Cause: the FN target is an explicit function of ONE-HOP quantities of the guide (kept weight W, wall V, H_xx); a zero-hop net cannot represent it.
- Net + one exact semi-implicit hop recovers 82% of iteration 1 (<H> 1.32e-4 -> 8.9e-5, E_FN 1.02e-4 -> 7.10e-5), then stalls (the stored net cannot carry later changes).
- Next: give the network one-hop inputs from the frozen base (f = g(x, log a_P, W_P, V_P)): same 'net + one hop' cost as the signs, no recursion. Exact pre-test (>= 50% per iteration) approved and running.

## 3h. Learnability of the tail correction (results/writeback_tail/, wtKA1a vs wtKB, 2026-10-08) — POSITIVE
- 20% of orbits held out with all their edges removed: held-out tail capture 20.7% = training tail 20.9%; shuffled-target control 0.4%. The tail correction is STRUCTURED and the net GENERALISES; no memorisation gap.
- Limit = representation precision (~a quarter of each weight decade, falling with depth), not generalisation, not noise. Next: one-hop guide inputs (wtKA2, running) and the tail expert (wtEXP, running); itfit predicts the inputs W, V, H_xx are what is missing.

## 3i. WRITE-BACK SOLVED on oracle data (results/writeback_tail/, wtKA2, 2026-10-08)
- A 37k net with three guide-local one-hop inputs (log a, V = FN wall term, W = kept-edge weight) captures 88.7% of one exact FN iteration (quadratic 89.2%), 93% of the <H> gain (1.319e-4 -> 8.86e-5; exact FN step 8.41e-5), ~90% in every weight decade down to 1e-14, held-out tail 89.7% = training 89.8%. Copy error 0.0027 rms (< 0.004). Spin-only nets: 20-31%; tail expert and edge sampling did not help.
- Cause of all earlier failures: the FN correction is a function of one-hop guide quantities; spin-only nets cannot compute them. Same 'net + one hop' structure as the stored signs.
- itfit route (results/itfit_6x6/, e8e72c2): realistic tempered samples, 5k net on frozen-base one-hop features, semi-implicit FN target: 95% / 80% / 69% per iteration; E_FN 1.02e-4 -> 4.74e-5 in 3 iterations (ideal 3.81e-5); with one exact hop <H> 4.53e-5 vs RBM+PP 4.47e-5. Same-model VMC control NOT yet run (open condition).
- Next: pre-registered Step 2 (realistic samples, no phi_FN; same-input fixed-sign VMC control; pass >= 0.5 and >= 1.5x control; guide-evaluation cost per sample -> 8x8), then 3-iteration jump and the 6x6 loop vs RBM+PP. itfit route merged into this.

## 3j. Step 2 PASSED (realistic data, results/writeback_tail/ section 6b, 284e623, 2026-10-09)
- No phi_FN anywhere: semi-implicit FN target T = log(1+W) - log(1+(H_xx+V-E_a)) from the guide only, edge least squares, Adam: 73.4% of one FN iteration (<H> 1.319e-4 -> 1.032e-4), held-out tail = training tail (75.4%).
- Same-input VMC controls (frozen-FN VMC and fixed-sign <H> VMC, Adam and minSR) LOSE energy (-15%) or diverge; itfit's same-model VMC control also fails. Cause: a realistic VMC energy estimate scatters ~200x the per-iteration gain, while the FN step gives a zero-variance per-configuration target. This is the loop's genuine edge over VMC.
- Open caveats (monitored): smaller-step VMC controls (wtS2c) running; minSR destroys the bulk on the regression (Adam works); itfit loop vs RBM+PP: 4.53e-5 vs 4.47e-5. Next: multi-iteration loop with the FEAT net (queued), then 8x8 cost/feasibility.
- 6x6 LOOP with realistic write-back (6c, 6f3e6d4): after 5 iterations exact E_FN 3.96e-5 < RBM+PP 4.47e-5; <H> 4.64e-5 (RBM+PP 4.47e-5). Needs ~5 iterations where the ideal loop needs ~3. CAVEAT for scaling: here each new guide is stored as an exact 6x6 table la_{k+1} = la_k + f; with current-guide features a real callable chain recurses (k hops after k iterations) -> amortising the guide stack is the next engineering problem (frozen-base features avoid recursion at 0.95/0.80/0.69). 8-iteration run (wtLOOP8, f37f810): <H> passes RBM+PP at iteration 6 (4.03e-5); after 8 iterations <H> 3.16e-5, E_FN 2.75e-5 (exact, 6x6). Loop runs at ~half the ideal speed. 8x8 plan: results/writeback_tail/SCALING_PLAN.md (~40-60 H100-h; options B frozen-base features / C re-basing by distillation / D caching; 6x6 pre-tests first).

## 3k. Review section E (dec53f5) — accepted points
- Critical: one exact Lanczos step on psi_P (callable, one hop, no training) gives <H_ph> 2.55e-5 / E_FN 1.89e-5, better than the table loop after 8 iterations (3.16e-5 / 2.75e-5). Quote all loop numbers against Lanczos at equal hop count; the fair test is 'loop guide + Lanczos' vs 'ViT + Lanczos'.
- 'Passes RBM+PP' is a TABLE result; the only callable variant (option B) stops at 5.6-6.2e-5.
- Needed: >= 3 seeds; amplitude-only trajectory (no exact Krylov sign); supervised-Lanczos control (same net regressed on log(1+alpha(E_L-E))).
- Scaling: distillation not ruled out in principle (target is a function of x alone; the obstacle is log-ratio precision ~3e-3 on ~135 bonds). Avoid message passing on the spin-swap graph and the warm ViT; prefer a bond-output ratio net or input-gradient features. Pre-tests: V/W sensitivity, ratio fidelity per decade, end-to-end callable 3/6 iterations with 3 seeds (pass: >= 70% of table gain, <= 3 base passes per configuration, callable + Lanczos >= 2x below 2.55e-5).
- Running now: Lanczos-on-top, supervised-Lanczos control, seeds, w_s along the loop.

## 4. Decisions taken (and why)
- RULE (PI, 2026-10-08): a route is declared failed only when the failure is understood and pinned to a conceptual limit that cannot be fixed. Before that: diagnose (optimizer, convergence, sampling, literature) and try the better method. Section 3e write-back verdict is downgraded to INCONCLUSIVE (cause: Adam, not converged).
- FN = signs + referee/bound; amplitude objective = fixed-sign VMC energy with a second-order optimizer (test A, 2026-10-06).
- Signs are stored as composite nets, always used with one exact hop (angle 2).
- FN referee protocol (calibrated 2026-10-07, results/fn_calibration_6x6/): tau_max 0.025, M=512, beta window 0.8-2.4, BETA-TIME-AVERAGED mean over >= 500 populations (bias <= 3e-6, SE ~3e-6). The old step-averaged estimator is biased by -11..+25e-6 (guide-dependent) and medians are biased: do NOT use either. The it2_fn DMC numbers (oracles, G2, K3vit, K3a1, ViT guide) are WITHDRAWN until re-run; ll6_fn runs can be re-evaluated from saved per-step curves.
- "Beat the ViT" is not a result by itself; 6x6 is a lab with exact references.

## 5. Parked ideas (not dropped)
- Tail-steered FN walkers (importance function flatter than the guide, e.g. a^beta or a mixture, reweighted): needed at 8x8+ for tail data; queued AFTER the 6x6 oracle tests show the tail is learnable (>= 50%). Risk: weight fluctuations grow with N (ESS 5% at beta=1/2 on 6x6) -> use a mixture importance function.
  Recipe (Chatty, seq102): freeze F = H_FN[a,s] INCLUDING the violating-edge diagonal sum_bad K_xy a(y)/a(x); introduce a separate positive importance g only for kept-edge rates -F_xy g(y)/g(x) and E_loc^g = F_xx + sum_kept F_xy g(y)/g(x) (branching, adaptive tau). Never put g into the bad-edge diagonal (it changes the FN Hamiltonian). Walkers then sample g*phi_FN; old a*phi_FN samples cannot be reweighted into missing tail support. First test on an exact 4x4 frozen operator vs ED (energies, tail histograms, ESS/diversity), then 8x8. Current K1FNEngine does not support it.
- Walker-histogram / MLE amplitude fitting (no zero-variance property; ledger).
- 8x8 sign machinery alone (cheap; useful for the add-on pitch).
- Separate paper repo for Overleaf; paper rewrite after the angles settle.
- Possible NQS collaborators (ViT authors, CQSL).

## 5b. Compute note
- When Paderborn is back from maintenance (expected ~Oct 9-12), move GPU-heavy work there (H100s; ws1 A40s are slow and the queue is congested). Access via ws1 VPN; if login2 is down use `ssh -o HostName=login1 -o HostKeyAlias=login2 paderborn` (same host key). Check `sinfo` there first.

## 6. Running now (2026-10-08)
| Work | Owner | Budget |
|---|---|---|
| Write-back re-test: tail sampling + energy loss; optimizer arms SR/minSR and Gauss-Newton in Q (Adam = baseline); minSR fine-tuning the ViT itself; 'jump' fit of the ViT to the 3-iteration FN state + VMC polish (results/writeback_tail/) | Opus | <= 15 GPU-h |
| Imaginary-time fitting with H_FN (SR/Gauss-Newton fits, explicit vs implicit steps, stiffness pre-registered; sampling from |a|^2 vs FN mixed a*phi_FN (walker idea) vs tail; VMC control) (results/itfit_6x6/) | Opus | <= 15 GPU-h |
| Research Workspace: review request seq99; Chatty replied seq100; awaiting Tim | - | - |

## 7. Pending at 2026-10-08 evening (usage limit hit)
- Queued on ws1 (writeback_tail, results land in /project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/<job>_<id>/wt.json; expected starts 20:45 -> 05:40): wtC48 16948146 (63k), wtC96 16998622 (250k), wtEXP 16961949 (tail expert), wtKA 16976428 (learnability, real target, neighbourhood inputs, edge sampling), wtKB 17009145 (shuffled control), wtKGN 16976430 (global Gauss-Newton). Nobody folds them in: resume the writeback_tail worker or read the wt.json files directly. Low fair share because other campaigns hold ~730 running jobs; move to Paderborn when back.
- itfit_6x6 (imaginary-time fitting with H_FN) worker was running; check results/itfit_6x6/.
