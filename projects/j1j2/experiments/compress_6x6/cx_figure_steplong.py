"""6x6: how far the loop iteration goes on exact tables (no networks): error vs iteration, 16 iterations from psi_P."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "results/compress_6x6"
E0 = 0.50380965
d = json.load(open(R / "runs/cx_step_long16.json"))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.6, 4.4))
st = d["start"] / E0
for name, lab, c, m in (("SI_1", "old target (fixed-node)", "#D55E00", "D"), ("Lc_0.3", "new target (clipped Lanczos)", "#0072B2", "o"),
                        ("L", "exact Lanczos, one step at a time", "0.5", "")):
    k = [0] + [r["k"] for r in d["runs"][name]]; e = [st] + [r["err"] / E0 for r in d["runs"][name]]
    ax.plot(k, e, (m + "-") if m else "--", color=c, lw=1, ms=5, label=lab)
for y, lab, c in ((9e-5, "RBM+PP (Nomura 2021)", "#009E73"), (1e-5, "DMRG, 8192 states", "0.35")):
    ax.axhline(y, color=c, lw=1, ls=":"); ax.text(8, y * 1.1, lab, ha="center", fontsize=9, color=c)
ax.set_yscale("log"); ax.set_ylim(4e-7, 4e-4); ax.set_xlim(-0.4, 16.4)
ax.set_xlabel("loop iteration"); ax.set_ylabel(r"$\epsilon = (E - E_{\rm exact})/|E_{\rm exact}|$")
ax.text(0.03, 0.04, "exact 6x6 tables, no networks; start: symmetrised ViT", transform=ax.transAxes, fontsize=9, color="0.4")
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.93))
fig.tight_layout(); fig.savefig(R / "fig_steplong_6x6.png", dpi=200); fig.savefig(R / "fig_steplong_6x6.pdf")
print("ok")
