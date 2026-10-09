# Claude's take on the FN–Krylov loop (2026-10-03)

Working notes from the paper-restructuring session. Opinion, not established results.

## Where the evidence is encouraging
- **Sign half is solved given a good amplitude.** Exact 4x4 amplitude + Marshall: 3–4 label-free Krylov steps reach the exact signs for every J2/J1 in 0.4–1.0 (paper Fig. 1). On 6x6 one step matches the ViT signs energetically (Fig. 3).
- **Amplitude half has a guarantee.** Lattice FN obeys E0 <= E_FN <= <H>_guide (ten Haaf et al.); the exact FN/Krylov loop reaches the exact ground state on 4x4 and the 20-site torus, from a start (J2=0 amplitude) where Krylov alone is stuck.
- **One half-iteration already scales.** 8x8, ViT amplitude + Krylov signs + FN: E/N = -0.49821, among the best published same-geometry values, and replica-stable.

## Worries
1. **The exact loop is slow (~100 iterations, power-law-like decay).** Cause: lattice FN moves forbidden elements to the diagonal weighted by guide amplitude ratios a(y)/a(x) (see `build_fixed_node` in `krylov_sign_structure/experiments/closed_fn_krylov_exact4x4.py`). So E_FN is exact only if the guide amplitude is already exact, and a_{k+1} = phi_FN[a_k, s_k] is a slowly contracting fixed-point map. Evidence: on 4x4 the sign error is flat at 1e-6 from iteration 20 to 99 while the energy keeps decreasing — only the amplitude is moving.
   (Correction to an earlier remark in the session: FN does NOT depend on the signs only.)
2. **Amplitude noise feeds into the signs.** With an approximate amplitude, Krylov returns the energy-optimal signs *for that amplitude*, not the true ones; with a poor amplitude the sign error can even increase (Fig. 2, amplitude from J2=0.6).
3. **8x8 iterations so far fail.** The iteration-1 FN gain was not reproduced by an independent rebuild; iteration 2 was worse than the start.
4. **Sampling.** An older sampled 4x4 run stalled near 1e-3 energy error; a clean rerun (histogram amplitude, energy-optimal threshold) is in progress.
5. **FN systematics not checked at 8x8.** Only M=128, beta=1.2, two replicas; population-control bias and projection-length dependence are unknown.

## Ideas
- **Accelerate the amplitude fixed point** (Anderson/DIIS mixing on log a over the last few iterates). Quick exact 4x4 test: iterations to reach the exact signs / a given energy error vs plain iteration.
- **Use the loop as a correction to a good network**, not as a solver from scratch: one or a few iterations from a strong NQS. The critical question becomes whether a learned amplitude can beat the ViT amplitude by a small margin so that one more iteration helps.
- **Track cost explicitly**: total = iterations x (FN walkers x projection length + learning samples). Measure each on 4x4 -> 6x6 -> 8x8 to get scaling, not single points.
- **Fair III.D comparison**: same ViT amplitude, guides Marshall / Krylov / ViT signs, energies before and after FN at matched M, beta, replicas.

## Update — end of 2026-10-03 session

New results (all in the paper branch `j1j2-organize`):
- **Repeated Krylov, exact 4x4 amplitude**: exact signs in 3-4 steps for all J2/J1 in 0.4-1.0 (Fig. 2).
- **Approximate amplitude (4x4, amplitude of another J2)**: energy still drops but plateaus; Krylov signs are the best signs for that amplitude, sometimes further from the true ones (Fig. 3).
- **Repeated Krylov, 6x6 ViT amplitude** (job 3576208, Paderborn CPU): step 2 does not lower the energy (+1.7(1.8)e-4/site), though it halves the sign disagreement with the ViT. One step already sits at an energy-stationary point of the threshold family. Step 3 not run (needs ~12-15 CPU node-h).
- **Anderson acceleration** of the exact loop: 3-4x fewer iterations (4x4: 100 -> 31; 20-site: 224 -> 59).
- **Geometric-mean update** a <- sqrt(a phi_FN) (what walker MLE gives): converges exactly 2x slower than a <- phi_FN.
- **Sampled 4x4 loop** (phi_FN from counts): floor in energy error ~1/N (1.3e-2, 1.7e-3, 2.2e-4 for 1e6, 1e7, 1e8 samples/iteration); real walkers worth ~3x fewer samples than i.i.d.
- **III.D matched comparison (M=128, beta=1.2)**: FN with Krylov signs is ~6e-4/site ABOVE the ViT itself on 6x6 and 8x8; FN with the full ViT guide reproduces the ViT energy. The ViT alone (-0.498823 on 8x8) beats every 8x8 literature value we compare to.
- **Possible FN bound violation on 6x6**: E_FN[Krylov guide] = -0.502984 lies above the guide's own variational energy (~ -0.5033..-0.5035). Either the two use different thresholds T or the FN runs are biased (beta / population). Check running (fn_bound_check_6x6).

Key insight for the learning step: guaranteed improvement needs <H>_{new guide} <= E_FN[old], not just <H>_new <= <H>_old. Train on the frozen <H_FN> (Rayleigh quotient of H_FN[a_k, s_k]); use the FN run as referee (gap <H_FN>_theta - E_FN) and as the reported energy. Ledger of all past attempts: AMPLITUDE_LEARNING_LEDGER.md.

Running at end of session: 6x6 symmetric ED (Opus), 4x4 learning-ladder rung 2 with frozen <H_FN> + SR from samples (Opus), 6x6 FN bound check (Sonnet), related-work literature check (Sonnet).

## Update — 2026-10-05
- **Loop iterations vs N (exact, 16..36 sites)**: iterations to eps<1e-6 ~82-104 plain, 32-57 Anderson; to w_s<1e-8 ~89-125 plain, 28-70 Anderson. No visible growth with N. Only sign errors on configurations of weight <1e-12 take longer at larger N. (krylov_sign_structure/results/closed_fn_krylov_6x6/)
- **6x6 baseline corrected again**: Krylov-guide FN = -0.503100(40) from 76 populations (8-population -0.503288 was low). FN lowers the guide by 2.6(6)e-4; 6e-4 above the ViT.
- **6x6 learned loop, iteration 1**: E_FN = -0.503667(63), equal to ViT within errors; true sign error 8e-5 < ViT 1.3e-4. Control: second Krylov step alone -0.503596(52) -> most gain from the sign step.
- **Real scaling problem found**: recursive sign definition — each Krylov step adds a neighbour hop (1.3e3, 5e4, 1.2e6 configs per state), iteration 2 would cost >50 GPU-h. Fix to try: store/distill signs after each iteration (sign network or per-iteration sign table on samples) so evaluation cost stays at one hop.
- FN at M=128 is noisy (single populations scatter ~3e-4/site, one fell below E0): need many populations or larger M for any claim.

## Update — stored signs (2026-10-05, results/stored_signs/)
- Recursive Krylov signs can be replaced by a stored sign (cost ~1e-5 s per evaluation, independent of k), but only if it is trained on the right distribution.
- **Self-sealing:** the configurations whose sign must flip have tiny amplitude, because FN suppresses a(x) on wrong-sign states. |a|^2 samples therefore miss the flips (a table covering 99.5% of |a|^2 weight missed 100% of the flip weight), and stored-vs-recursive disagreement measured under the current a looks deceptively small. Any learned sign (or amplitude) must be trained on a broadened distribution, e.g. tempered |a|^{2 beta} with beta=0.5, or with one-hop neighbours.
- 4x4: translation-invariant net on tempered samples (beta=0.5), N=1e4 per step, comes close to the exact loop (it30: w_s 2.5e-5, eps 7.4e-5 vs exact 8.9e-7 / 5.2e-5); table+neighbours matches exactly but only because 4x4 is fully covered. Untempered |a|^2 variants stall at w_s ~1e-2.
- Next: test the tempered sign net on 6x6 (stored K2 vs exact K2, then K3 at one-hop cost).

## Update — 2026-10-06 (6x6 iterations, 4x4 net start)
- **6x6 iteration 1, 64 FN populations:** -0.503708(29) mean, median -0.503654; 17/64 populations below E0 at M=128 (heavy low tail). Tie with the ViT. (results/learned_loop_6x6/it1_fn_extended/)
- **6x6 stored signs:** K1 nets match/beat the recursion; stored K1 net + one exact hop (= K2) gives w_s 6.0e-5 and +5.8(1.8)e-5/site vs ViT — signs 2-3x better than the ViT's. Stored nets alone are lossy for K>=2 (miss ~43% of rare flips); the final exact hop is essential. (results/stored_signs_6x6/)
- **6x6 iteration 2:** G2 = (a2, D2 + hop) E_FN -0.503687(23); ViT's own guide under FN -0.503666(8) (M=512); G2 - ViT-guide -1.7(2.5)e-5. Tie. Cost ~4-5 GPU-h/iteration (vs >50 recursive). (results/learned_loop_6x6_it2/IT2_VERDICT_2026-10-06.md)
- **Oracle (decisive):** ViT amplitude + exact signs -> FN -0.503682(21) = ViT guide; exact amplitude + ViT signs -> -0.503731(15). Near the ViT the FN error is set by the amplitude on frustrated edges, not by the signs. Ideal loop contracts ~0.85-0.89/iteration => ~1.5e-5/site per ideal iteration at the ViT level, below referee resolution; a 2-sigma win needs ~5 ideal iterations.
- **4x4 from a VMC-trained net:** sign step w_s 2.4e-4 -> 2e-6 in one step; learned amplitude update realises 0.3-0.5 of the frozen gain per iteration; loop ~tied with continued VMC at equal CPU. (results/net_start_4x4/)
- **Conclusion:** the amplitude update is THE bottleneck; design discussion in results/amp_design/ before any new runs.

## Takeover log (from 2026-10-06 18:30, PI away; check-in Wed 2026-10-07 09:00)
Agreed defaults: T3 go if T2 (Krylov signs + best optimizer) <= 0.5x the standard-VMC control error at equal CPU-h on both seeds (one seed -> rerun with a third seed); ceiling 30 GPU-h ws1 + 300 CPU-h, no Paderborn; T2 fail -> no T3, write fallback framing; angle follow-ups <= 20 CPU-h each; no paper edits, emails, workspace posts, chatty-slurm-watch, cancelling others' jobs. Deleting run data only if quota is critical (then our own stale runs; Ruby campaign -> ask ruby_runs).
- 18:30 quota /project/theorie: 302/475 GB, 411k/475k files.
- 20:17 all three workers (design T1/T2, angle 1, angle 2) were cut off by an API rate limit at ~18:30 (reset 20:10); resumed with their context. Their ws1 jobs kept running (badstart20 x3, sl-eval, sl-pool-*).
- 21:27 angle 2 DONE (361f138; numbers re-checked against tables_8e4.md): sign step shape is not the bottleneck; exact GS sign learnable (N36 186k net rel 6.3e-5); Krylov FACTORS c_k (k>=2) unlearnable (c2 rel 0.86), COMPOSITE sign easy (+hop = exact chain) -> store composite signs, not stacked factors (explains lossy 6x6 K2/D2 nets); one-hop neighbours essential, data not the limit, params needed grow N32->36; one exact hop removes 96-99% of net error at all N. Cost ~21 GPU-h (launched before takeover).
- 21:31 T1/T2 DONE (7235899). Re-derived from vit/*.json: adaptive loop (Krylov sign + 3 RGN-TR steps/iter) beats best complex-ViT RGN control >=2x at every CPU-h on both seeds (1 CPU-h: 3.1e-5/4.0e-5 vs 1.24e-4; 3 CPU-h: 6.3e-6/1.3e-5 vs 5.7e-5); -97% <H> error vs net H at 3 CPU-h, E_FN 4e-6. Caveat: 4x4 sign step exact. T2 PASS -> T3 GO (6x6, <=30 GPU-h, stop rule after 2 iterations). Control tuning <=20 CPU-h in parallel.
- 22:04 angle 1 DONE (10cda71; rows checked in tables.md): any generic GS content in the sector -> exact loop reaches GS (random signs, uniform amp, mixtures down to GS weight 1e-8; escape ~ln(1/w)/0.35 its from phi_1); exact eigenstates are fixed points (unstable in-sector); other symmetry sectors trapped; exact zeros of a are an absorbing support trap; Anderson gets stuck at excited fixed points (use plain/safeguarded steps from bad starts). Mac disk 97% full (6.6 GB free) — not from our runs; worker ran pkill -f multiprocessing.spawn during cleanup.
- 06:55 T3 DONE (f8b155f), stop rule after 2 iterations. Re-checked T3_table.md: final guide <H> -0.503662(22), paired vs ViT -8.2(4.3)e-6/site (1.9 sigma; the worker's summary quoted -0.6(0.3)e-5 — table value used here), below the 2-sigma rule; FN referee not run. Signs: composite net + hop(a1) w_s 2(1)e-5 (ViT 1.3e-4) = best 6x6 sign. Amplitude: 1 of 6 steps verified (-8.3(3.9)e-6); with P=155k >> N=8-16k, steps do not generalise to fresh samples. 9.7 of 30 GPU-h used (2080 Ti). 4x4 tuned control: loop still 3.3-7.2x lower error at equal CPU-h (best of 7 complex-ViT runs).
- 06:55 quota /project: 305 GB, 418k files (OK). Mac data volume 95% (10 GB free).

### Check-in summary (Wed 2026-10-07 09:00)
- 4x4: Krylov sign + second-order fixed-sign VMC beats the best tuned VMC control 3.3-7.2x at equal CPU-h; -97% error vs trained net at 3 CPU-h. Caveat: exact sign step on 4x4.
- 6x6 from the ViT: signs 6x better than the ViT's at one-hop cost; energy tie (1.9 sigma). Bottleneck: amplitude generalisation with P >> N at affordable N; ViT amplitude leaves < ~1e-5/site per step.
- Angles: (1) bad starts converge if any in-sector GS content; eigenstates/other sectors trap; Anderson unsafe from bad starts. (2) sign step shape not a bottleneck; store composite signs; one hop repairs 96-99%.
- Decisions for PI: paper framing; whether to pursue the P >> N amplitude problem (last-layer updates / larger N) or 8x8 sign machinery.
- 2026-10-07 08:05 PI: goal = unblock 6x6 amplitude learning. Plan sent to design partner: (A) zero-hop composite sign net for amplitude phases (hop only at sign steps) -> local energy ~82 evals/sample instead of ~7e3; (B) RGN-TR at N=1e5-5e5 (N >= P); (C) restricted subspace / small correction factor if still overfitting; (D) same-optimizer VMC control. Gate: one large-N iteration with verified gain > 2 sigma. Budget <= 25 GPU-h.
- 2026-10-07 CORRECTION (literature survey, results/literature/J1J2_BENCHMARKS_COST.md, 1640822): our ViT is NOT state of the art at 6x6/8x8. 6x6: RBM+PP -0.503765(1), DMRG -0.503805, exact -0.503810 vs ViT -0.503654. 8x8: RBM+PP -0.498886(1), Hu VMC+Lanczos -0.49886(1) vs ViT -0.498823(35) and our FN -0.49821. The earlier notes "ViT beats every 8x8 literature value" and "-0.49821 among the best published" are wrong. "Beating the ViT" is not a new result. 10x10 record -0.4976921(4) (Chen-Heyl; -0.4971633 was the 16x16 value). Cost bar: 10x10 ViT ~1.9e3-3.8e3 A100-h. Gaps: 12x12/14x14/18x18 without deep-NQS results; 8x8 below -0.49889 would be new. Public 10x10 ViT checkpoint (nqs-models) usable as a guide.
- 2026-10-07 11:14 PI + lead agreed: use 6x6 (ED, exact phi_FN, RBM+PP) as the lab for WHY the loop stalls before larger sizes. Stall diagnosis worker started (results/stall_6x6/): (1) ViT capacity floor by supervised fit to exact |psi0|, (2) projected ideal loop (exact phi_FN projected onto the ViT manifold), (3) architecture headroom check. Budget 15 GPU-h + 100 CPU-h. From-scratch cost race deferred until this answers.
- 2026-10-07 13:07 AMP6 gate (f0198bd; table re-checked): large N fixes generalisation and steps verify, but the fixed-sign optimum of the ViT amplitude is within ~1e-5/site of the ViT; zM arm ends +0.67(0.90)e-5 vs ViT, control D -0.35(0.31)e-5. Loop does not beat its control. Decision (lead): no further AMP6 variants (options 1-2 chase <1e-5); wait for the stall diagnosis (compression vs information) before choosing the architecture for the from-scratch cost race. 7.8 of 25 GPU-h used.
- 2026-10-07 16:52 STALL DIAGNOSIS (3132dfe; numbers checked in README): exact, full symmetric sector. Symmetrising the ViT gains 2.4e-5 (projected ViT -0.5036777). Ideal exact loop from it: E_FN 1.02e-4 -> 6.5e-5 -> 4.9e-5 (~RBM+PP after 3 its) -> 8.8e-6 at it 12: FN information is plentiful. Every projected (network) loop stalls at the ViT level: the loop stalls where the projection error equals the per-iteration gain; dE ~ 0.33 sigma^2/site, keeping 90% of an iteration needs <= 0.004 rms log-amplitude accuracy (update rms 0.011). Pointwise supervised fits raise E; exact-verified SR on the ViT ~1e-6/step. Bottleneck = writing the FN update into a network at 0.004 rms. Incident: worker's 7.2 GB file put /home on ws1 over quota ~6 min at 11:27 (moved to /project).
- 2026-10-07 22:17 Write-back route FAILED (results/writeback/): FN update is white-noise-rough in the energy metric and tail-concentrated; oracle best 17%, realistic samples <1% = plain VMC. Amplitude via VMC; FN/Krylov = signs + referee. Next: Lanczos + calibrated FN route (calibration running).
- 2026-10-08 07:43 Research Workspace: posted seq99 (status since Oct 4 + request for independent review, 4 questions) to researcher.tim in thread 0cc53438. Last read seq 99. (.research_workspace_state.json lives only in ~/Chatty, not edited.)
- 2026-10-08 16:18 Research Workspace seq101 to researcher.chatty: tail write-back status (31% best) + 2 questions (tail share of its 4x4 full-space SR success; decoupled importance function in its GFMC). Last read 101.
- 2026-10-09 10:10 Workspace seq105 (Chatty): no no-go for compression; distil the whole a_k into one flat callable net; first arm direct symmetry-respecting ViT with Dirichlet-weighted log-ratio loss; re-base every 2-3 steps. Reviewer E prefers bond-ratio net, warns against warm ViT. Decision: compression pre-tests with both arms (+warm ViT, +one-layer edge aggregation), pre-registered (results/compress_6x6/). Last read seq105.
