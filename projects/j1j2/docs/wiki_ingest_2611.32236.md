# Wiki ingest: arXiv:2611.32236 (Kliczkowski et al. 2026), 2026-11-26

**Result.** Ingested as `sources/kliczkowski2026augmented.md`, wiki commit `9921eff`
(repo `/project/theorie/a/A.Otaifi/ai-workflow/second_brain`). Lock taken and released. Filed under
`projects/sign_prob/` (hot_cache), because AGENTS.md gives no right to create `projects/j1j2/`
(same reading as the 2026-11-13 Rende ingest). Done-test passed; the paper has **no tables**.

## Done-test 1: implicit SR reduces to explicit SR to first order in tau
From Eq. 11, $(S+\lambda I+\delta\tau K)\mathbf d=-\mathbf F$, with $M=(S+\lambda I)^{-1}$:

$$\mathbf d=-M\mathbf F+\delta\tau\,MKM\mathbf F+O(\delta\tau^2)=\mathbf d_{\rm SR}+\delta\tau\,MKM\mathbf F+O(\delta\tau^2),$$

so the parameter step $\delta\boldsymbol\theta=\delta\tau\,\mathbf d=\delta\tau\,\mathbf d_{\rm SR}+O(\delta\tau^2)$ is explicit SR (Eq. 6)
at first order. At the state level, $(1+\delta\tau\hat G_n)^{-1}=1-\delta\tau\hat G_n+O(\delta\tau^2)$ is Eq. 8.
Numerical check on a random 40-dim problem with a log-linear ansatz, objects built from the
transcribed definitions (A4, A6, C2, Eq. 11): $\vert\mathbf d-\mathbf d_{\rm SR}\vert/\vert\mathbf d_{\rm SR}\vert$ scales with slope 0.999 in
$\delta\tau$; after subtracting $\delta\tau MKM\mathbf F$ the remainder scales as $\delta\tau^2$ (9.4e-4, 9.8e-6, 9.9e-8).
Broken control $(S+\lambda I+K)$: slope 0.000, FAIL as required. Galerkin residual
$\max\vert\langle T_j\vert i\rangle\vert=6.9\times10^{-15}$. Sample estimators S1-S3 (SM S7) equal $\langle T\vert T\rangle$,
$\langle T\vert G\vert\Phi\rangle$, $\langle T\vert G\vert T\rangle$ to 4e-16; dropping the $\tilde O_kG_{\rm loc}$ term in $B_k$ fails (0.79).
Two-level iAITE recurrence: ratio to $4\epsilon^2/(\delta\tau^2\Delta_*^3)$ goes 0.68, 0.93, 0.998 ($\delta\tau=0.5$);
iITE $\epsilon'/\epsilon=0.4447$ vs $1/(1+\mathcal T)^2=0.4444$. No inconsistency in paper or transcription.
Script and output: `wiki/assets/kliczkowski2026augmented/check_isr.py`, `check_isr.out`.

## Done-test 2: benchmark numbers vs page image
No `tabular` in `main.tex` or `supplemental.tex`. Substitute: the 20 energies printed inside
Fig. 4 (vector text layer of `Figures/fig_4.pdf`) checked on the 250-dpi render of page 5
(`page5-fig4-energies-vit-4x4-to-8x8.png`): 20/20 agree, no mismatch. ViT $N_p$ values (page 15)
agree. Extra: wiki ED (`ed4x4.py`, control $J_2=0.5$ gives Schulz's -0.528620) reproduces the
4x4 values -0.619874 ($J_2=0.2$) and -2.511887 (TFI $h=2$). Against Schulz N=36 $A_1$ -0.599046,
the 6x6 $J_2=0.2$ ViT runs sit at $\epsilon_{\rm rel}$ 4.7e-5 (iITE), 9.2e-5 (iAITE), 1.1e-4 (explicit ITE).

## Routes
- Eqs. 1-11, A1-A9, B1-B6, C1-C5, SM S1-S8: arXiv v1 `.tex` (route A); no DOI exists.
  Checked on rendered pages 3, 4, 7, 8, 18, 19, 20 (route C): no mismatch. SM numbers restart per
  SM section, cited as "SM S7 Eq. S10".
- Fig. 4 energy table: figure text layer + page image. Figs. 1-3, S1-S5: captions and text only.

## Not processed / not done
- No data points read off Figs. 1-3, S1-S5 (only Fig. 4 printed labels); reference list skimmed.
- Not edited (outside the approved list), proposed instead: a `stochastic_reconfiguration`
  concept page (explicit SR, implicit SR, Sorella2001 link); a convention note on `v_score`
  (this paper uses $N\mu_2/(E-E_\infty)^2$, Rende2026 uses $\langle H\rangle^2$); Kliczkowski2026 rows on
  `j1j2_square_heisenberg`; creating `projects/j1j2/`.
- Paper-side remarks (in the source page): "agree within ~1e-5 per site" vs printed spread
  3.8e-5 at 6x6 $J_2=0.2$, and at 8x8 TFI explicit ITE is lower than iITE by 1.4e-5; no error bars
  on any energy; quadratic convergence derived for the noiseless two-level frozen-generator case only.
