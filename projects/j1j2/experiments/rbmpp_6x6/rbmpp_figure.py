"""Figure: the loop iteration (clipped-Lanczos target, exact 6x6 tables) started from different guides.
(a) relative error vs iteration for the ViT, RBM+PP (converged, A), RBM+PP checkpoints (B2, B1) and the PP alone;
(b) equal hop count: k loop iterations vs k joint Lanczos steps on the same guide (A and B2).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / "results"
R = ROOT / "rbmpp_6x6"
E0 = 0.50380965
vit = json.load(open(ROOT / "compress_6x6/runs/cx_step_long16.json"))
G = [("ViT", vit, "ViT (symmetrised)", "#2a78d6", "o"),
     ("B2", json.load(open(R / "runs/step_B2.json")), "RBM+PP, early (B2)", "#eb6834", "s"),
     ("A", json.load(open(R / "runs/step_A.json")), "RBM+PP, converged (A)", "#1baf7a", "D"),
     ("PP", json.load(open(R / "runs/step_PP.json")), "PP alone", "#eda100", "^"),
     ("B1", json.load(open(R / "runs/step_B1.json")), "RBM+PP, early (B1)", "#e87ba4", "v")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2))
exp_fmt = FuncFormatter(lambda v, _: f"$10^{{{int(round(__import__('math').log10(v)))}}}$")
for tag, d, lab, c, m in G:
    rows = d["runs"]["Lc_0.3"]
    k = [0] + [r["k"] for r in rows]; e = [d["start"] / E0] + [r["err"] / E0 for r in rows]
    a.plot(k, e, m + "-", color=c, lw=1, ms=4, label=lab)
a.axhline(9e-5, color="0.45", lw=0.8, ls=":", label="RBM+PP published (Nomura & Imada 2021)")
a.set_yscale("log"); a.set_xlim(-0.4, 16.4); a.set_ylim(5e-7, 4e-3)
a.yaxis.set_major_locator(LogLocator(10)); a.yaxis.set_major_formatter(exp_fmt)
a.set_xlabel("loop iteration k (clipped-Lanczos target)"); a.set_ylabel(r"relative error $(E-E_0)/|E_0|$")
a.legend(frameon=False, fontsize=8, loc="upper right")
a.text(-0.12, 1.02, "a", transform=a.transAxes, fontsize=13, fontweight="bold")
for tag, d, lab, c, m in G[1:3]:
    rows = d["runs"]["Lc_0.3"][:4]
    k = [0] + [r["k"] for r in rows]; e = [d["start"] / E0] + [r["err"] / E0 for r in rows]
    b.plot(k, e, m + "-", color=c, lw=1, ms=4, label=f"{lab}: k loop iterations")
    lz = [d["start"]] + d["lanczos_ref"][:4]
    b.plot(range(len(lz)), [x / E0 for x in lz], m + "--", color=c, lw=1, ms=4, mfc="white",
           label=f"{lab}: k joint Lanczos steps")
b.set_yscale("log"); b.set_xlim(-0.2, 4.2); b.set_ylim(4e-7, 1e-3); b.set_xticks(range(5))
b.yaxis.set_major_locator(LogLocator(10)); b.yaxis.set_major_formatter(exp_fmt)
b.set_xlabel("hops k (applications of H)"); b.set_ylabel(r"relative error $(E-E_0)/|E_0|$")
b.legend(frameon=False, fontsize=8, loc="upper right")
b.text(-0.12, 1.02, "b", transform=b.transAxes, fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, "exact 6x6 J1-J2 tables (J2/J1 = 0.5), no networks; full space-group x spin-flip projection",
         ha="center", fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(R / "fig_rbmpp_loop_6x6.png", dpi=200); fig.savefig(R / "fig_rbmpp_loop_6x6.pdf")
print("ok")
