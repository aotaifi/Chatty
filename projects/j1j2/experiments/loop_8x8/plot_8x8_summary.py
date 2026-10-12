"""8x8 J2/J1=0.5: where our numbers stand vs the literature (energy per site, variational upper bounds + GS estimates).
Sources: results/loop_8x8/README.md (FU), fu_l8vmc_17138725_vmc.json, fu_l8dira_17138726_direct.json,
results/literature/J1J2_BENCHMARKS_COST.md (Hu 2013, Nomura & Imada 2021, Gong 2014)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results/loop_8x8"
pts = [  # label, E/N, err, colour, marker
    ("Hu 2013\nVMC + 2 Lanczos", -0.49886, 1e-5, "0.45", "s"),
    ("Nomura 2021\nRBM+PP", -0.498886, 1e-6, "#009E73", "s"),
    ("ViT", -0.498823, 3.5e-5, "#CC79A7", "o"),
    ("ViT\nsymmetrised", -0.4988474, 1.06e-5, "#CC79A7", "o"),
    ("+ one\nwrite-back", -0.498884, 2.4e-5, "#D55E00", "D"),
    ("+ 1 Lanczos\n(direct VMC)", -0.4989476, 2.31e-5, "#0072B2", "o"),
    ("+ 1 Lanczos\n(combined)", -0.4989530, 1.64e-5, "#0072B2", "o"),
]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.axhspan(-0.49907, -0.49905, color="0.85", lw=0)
ax.text(6.45, -0.49906, "GS estimate, Hu (extrap.)", ha="right", va="bottom", fontsize=9, color="0.35")
ax.axhspan(-0.4993, -0.4991, color="0.93", lw=0)
ax.text(6.45, -0.49921, "GS estimate, DMRG (extrap.)", ha="right", va="center", fontsize=9, color="0.45")
ax.axhline(-0.498886, color="#009E73", lw=0.8, ls=":")
for i, (lab, e, er, c, m) in enumerate(pts):
    ax.errorbar(i, e, yerr=er, fmt=m, color=c, ms=7, capsize=3, lw=1.2, mfc=c if i < 2 else c)
ax.axvline(1.5, color="0.7", lw=0.8)
ax.text(0.5, -0.49875, "literature", ha="center", fontsize=10, color="0.4")
ax.text(4.0, -0.49875, "ours (from the ViT)", ha="center", fontsize=10, color="0.4")
ax.set_xticks(range(len(pts))); ax.set_xticklabels([p[0] for p in pts], fontsize=9)
ax.set_ylim(-0.49930, -0.49872); ax.set_xlim(-0.6, 6.6)
ax.set_ylabel("E / N   (8x8, $J_2/J_1 = 0.5$)")
ax.ticklabel_format(axis="y", useOffset=False)
ax.text(0.01, 0.02, "lower = better; all points are variational upper bounds", transform=ax.transAxes, fontsize=9, color="0.4")
fig.tight_layout(); fig.savefig(R / "fig_8x8_summary.png", dpi=200); fig.savefig(R / "fig_8x8_summary.pdf")
print("ok")
