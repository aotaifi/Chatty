# Ingest report: arXiv 2606.02794 (Rende et al. 2026, Scaling Laws for NQS)

Date 2026-11-13. Worker for the j1j2 lead. Citekey `rende2026scaling`.

## Commits (wiki repo `/project/theorie/a/A.Otaifi/ai-workflow/second_brain`)
- `034db4c7d8761ad8d8713d1c4fda53a210d14fbe`: the source page, new concept pages `v_score` and
  `scaling_laws_for_nqs` (same commit, as AGENTS.md requires to avoid dangling links), assets, parameters (+5),
  sources_to_explore (+4, Nomura & Imada incremented), index, log ("focus approved by j1j2 lead").
- Write lock taken 10:15 and released after the commit.

## Premise mismatch (focus item 2 and done-test b)
The paper has **no energies or variances at all**, and **no 6x6, 8x8 or 10x10 data**. Its sizes are L = 12, 16, 20
(square J1-J2) and 12, 16, 18, 20 overall, all with PBC. It reports only V-scores (in figures) and the fitted exponents
(Tab. I). J2/J1 = 0.5 is the only square J2 > 0.

## Route per item
| item | route |
|---|---|
| Eq. 1 (V-score), Eq. 2 (H) | transcribed, `.tex` (Route A), checked on page 2 image |
| Eqs. 3, 4 (scaling form, size consistency) | transcribed, `.tex`; page images p3, p4 stored |
| Eqs. 5, 6, 7 (spatial attention, forward cost, f = C M N_opt) | transcribed, `.tex`; page images p5-p7 stored |
| Tab. I (alpha, beta, log A, four models) | transcribed in full, `.tex`, checked on the page 4 image |
| Fig. 2 (square J1-J2, L = 12, 16, 20; 18 points) | data read from the vector figure (`images/collapse.pdf`, `mutool clean -d`, marker centres and error-bar ends through tick positions) -> `assets/rende2026scaling/fig2-square-j1j2-points.csv`; approximate |
| Fig. 3 (L = 12, b = 4 / b = 2 / b = 2 + translations), Fig. 5 (L = 12, M = 16384 / 20480) | vector read -> `fig3-fig5-L12-points.csv`; approximate; cross-check: their b = 4 / first six points equal Fig. 2's L = 12 points |
| Fig. 1 (four models collapsed) | image only; square J1-J2 x positions spot-checked against Fig. 2(b) |
| Fig. 4 (forward cost vs n) | image only |

Equation numbers follow the PDF. No DOI exists (preprint), so nothing was checked against a journal version.

## Not processed
Introduction literature on ML scaling laws; triangular-lattice physics; Fig. 1 points for the other three models; Fig. 4 points.
Not done because outside the approved concept list (the auto-mode guard also refused the first attempt):
- `concepts/j1j2_square_heisenberg.md`: add a Rende2026 column to the conventions table (energy unit and bond counting not
  stated; energy reported: none, V-score only; PBC L = 12, 16, 20; sector not imposed, translation projection in Fig. 3 only)
  and a key-source bullet. **Proposed.**
- `concepts/j2_over_j1.md`: a "What it sets" row: transformer V-score compute exponent alpha = 1.29(2) at J2 = 0, 0.86(3) at 0.5
  (Tab. I). **Proposed.**
- `projects/j1j2/`: AGENTS.md gives a lead its own project folder but says nothing that lets a worker create one, so it was
  not created. **Proposed hot_cache.md** (about 170 words), to be written by the lead:
  > Scope: j1j2, square J1-J2 at J2 = 0.5, PBC. Wiki is the record of published benchmarks; our results stay in PLAN.md.
  > Predecessor folder: projects/sign_prob (Lin2024, Schulz1996, Szabo2020, Westerhout2023, guide-refit loops).
  > Active thread: compute and accuracy of NQS at J2 = 0.5.
  > - rende2026scaling: transformer V-score ~ f^-0.86(3) with size collapse, L = 12-20 only, no energies; best L = 12
  >   V-score ~ 1.9e-3 with translation projection (Fig. 3); ~1e5 A100 h total; no Lanczos.
  > - Bars (ketworks table, check the paper before quoting): 8x8 -0.498963(2) (2302.04919); 10x10 -0.497694 (2603.14425),
  >   -0.497810(2) Lanczos zero-variance estimate (1304.2630).
  > Proposed concept pages: Marshall sign rule, stochastic reconfiguration (carried over from sign_prob).

## Done-test
**(a) Exponent refit (pass).** Script `assets/rende2026scaling/refit_fig2_powerlaw.py`, output `refit_fig2_powerlaw.out`.
Fit of ln V = ln A - alpha ln f + beta ln N to the 18 Fig. 2 points, unweighted:
alpha = 0.861(26), beta = 0.924(37), ln A = 12.17(54) (Gaussian resampling of the half error bars, 4000 draws).
Paper: alpha = 0.86(3). Difference +0.001 = **0.03 x the quoted error**. The fitted beta/alpha = 1.073 equals the rescaling
ratio measured independently between Fig. 2 panels (a) and (b) (1.0733 at all three L), so this is the fit they made.
Caveat: weighted by the plotted error bars the fit gives alpha = 0.887(7), beta = 1.18(2), chi2 = 556 for 15 dof. The
seed-spread bars are far smaller than the scatter about the power law, so the quoted 0.03 is a resampling error of an
unweighted fit, not a goodness-of-fit error. Per-size slopes (unweighted): 0.89 (L = 12), 0.84 (16), 0.86 (20).
`quotable.py` not run: the inputs are published figure points, not our chains.

**(b) Energies vs the ketworks table (not applicable).** `ketboard.py --L 8/--L 10 --bc pbc --j2 0.5 --update` ran
("Already up to date"). The paper gives no 8x8 or 10x10 energy, so there is no lower / equal / higher verdict; the source
wins and has nothing to compare. Table best rows for reference: 8x8 -0.498963(2) per site (2302.04919, RBM+PP, Varbench bar);
10x10 best NN -0.497694(2e-7) (2603.14425, CNN-MPS), best overall -0.497810(2e-5) (1304.2630, zero-variance extrapolation,
ref-estimate). The paper is not in the ketworks table.

## For j1j2
- 8x8 / 10x10 J2 = 0.5 energies and V-scores: none in the paper (sizes L = 12, 16, 20 only; Fig. 2, Sec. I).
- V-score definition: N (<H^2> - <H>^2) / <H>^2, total H, no energy offset (Eq. 1).
- Exponent at J2 = 0.5, square, PBC: alpha = 0.86(3), beta = 0.93(4), ln A = 12.2(5) (Tab. I).
- V-score at J2 = 0.5, n_l = 8, b = 4, unsymmetrised: ~8.1e-3 (L = 12), 8.8e-3 (16), 9.1e-3 (20) (Fig. 2, figure-read).
- Best: ~1.86e-3 at L = 12 with b = 2 + translation projection (Fig. 3, figure-read); saturation ~6-7e-3 above 12 layers without symmetries (Fig. 5).
- Compute: f = C M N_opt, forward passes only (Eq. 7); 2e10-2.2e11 "FLOPs" at J2 = 0.5 (Fig. 2); ~1e5 A100 GPU h for the whole study (Acknowledgments).
- Optimiser MARCH with minSR-type SR, shift 1e-4, M = 16384 samples, 1e4 steps (Sec. IV A).
- Symmetries: none in the main data; translation projection only in Fig. 3. No Lanczos step, variance extrapolation or fixed node (Secs. IV A-D).
- Open: the plotted FLOP scale is ~1e3 below n_l n d^2 for the stated net (Fig. 4, Eq. 6); it moves only A, not alpha.
