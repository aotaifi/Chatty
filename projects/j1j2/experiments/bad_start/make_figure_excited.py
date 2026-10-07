#!/usr/bin/env python3
"""Two-panel figure: exact FN/Krylov loop from excited-state guides (4x4, J2/J1=0.5, plain loop)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[2] / "results" / "bad_start"
R = json.loads((D / "data" / "bad_start_4x4_plain.json").read_text())["runs"]
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
SEL = [("b_ex1", "#D55E00", "v", r"pure $\phi_1$ (first excited state)"),
       ("c_mix1_w1e-08", "#56B4E9", "D", r"$\phi_1$ + $10^{-8}$ ground-state weight"),
       ("c_mix1_w0.01", "#0072B2", "s", r"$\phi_1$ + $10^{-2}$ ground-state weight"),
       ("d_S1", "#CC79A7", "P", r"lowest $S\!=\!1$ state (other sector)")]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
for nm, col, mk, lab in SEL:
    h = R[nm]["hist"]; x = np.arange(len(h["eps_FN"]))
    for ax, key, floor in ((a, "eps_FN", 1e-10), (b, "ov_gs", 1e-34)):
        y = np.maximum(np.array(h[key], float), floor)
        ax.plot(x, y, "-", color=col, lw=0.9)
        m = (x % 60) == 0
        ax.plot(x[m], y[m], mk, color=col, ms=4.5, mfc="white" if mk in "os" else col, mew=1.0, ls="none", label=lab)
for ax in (a, b):
    ax.set_yscale("log"); ax.set_xlim(-10, 780); ax.set_xlabel("loop iteration")
a.set_ylim(1e-10, 1); b.set_ylim(1e-34, 3)
a.set_yticks([10.0 ** k for k in range(-10, 1, 2)]); b.set_yticks([10.0 ** k for k in range(-32, 1, 8)])
a.set_ylabel(r"FN energy error $(E_{\rm FN}-E_0)/|E_0|$")
b.set_ylabel(r"ground-state weight $|\langle\phi_0|\psi\rangle|^2$")
b.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.45))
for ax, L in ((a, "(a)"), (b, "(b)")):
    ax.text(-0.17, 1.02, L, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
fig.savefig(D / "fig_excited_start.png", dpi=200); fig.savefig(D / "fig_excited_start.pdf")
print("saved")
