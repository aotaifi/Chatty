#!/usr/bin/env python3
"""Figure: exact FN/Krylov loop from bad starting guides (J2/J1=0.5)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

D = Path(__file__).resolve().parents[2] / "results" / "bad_start"
def load(lat, meth):
    p = D / "data" / f"bad_start_{lat}_{meth}.json"
    runs = json.loads(p.read_text())["runs"] if p.exists() else {}
    for r in runs.values(): r["hist"]["_it"] = r["it"]
    return runs
P4, A4, P20, A20 = load("4x4", "plain"), load("4x4", "anderson"), load("20", "plain"), load("20", "anderson")

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
# colour-blind-safe (Okabe-Ito)
C = dict(base="#000000", ex1="#D55E00", mix2="#0072B2", mix8="#56B4E9", rand="#009E73", other="#CC79A7", sym="#E69F00")
fmt = FuncFormatter(lambda v, p: r"$10^{%d}$" % int(round(np.log10(v))) if v > 0 else "")

def logaxis(ax, lo, hi):
    ax.set_yscale("log"); ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(int(np.log10(lo)), int(np.log10(hi)) + 1, 2)]))
    ax.yaxis.set_major_formatter(fmt); ax.minorticks_off()

def curve(ax, h, key, color, marker, label, floor=1e-15, every=40, ls="-"):
    y = np.maximum(np.array(h[key], float), floor); x = np.array(h["_it"])
    ax.plot(x, y, ls, color=color, lw=0.9, zorder=2)
    m_ = (x % every) == 0
    ax.plot(x[m_], y[m_], marker, color=color, ms=4.2, mfc="white" if marker in "os" else color,
            mew=1.0, ls="none", zorder=3, label=label)

SEL = [("a_base_J2zero_Marshall", "base", "o", "J$_2$=0 amplitude + Marshall"),
       ("e_randsign_uniform_s0", "rand", "^", "uniform amplitude, random signs"),
       ("c_mix1_w0.01", "mix2", "s", r"$\phi_0$ weight $10^{-2}$ in $\phi_1$"),
       ("c_mix1_w1e-08", "mix8", "D", r"$\phi_0$ weight $10^{-8}$ in $\phi_1$"),
       ("b_ex1", "ex1", "v", r"pure $\phi_1$ (same sector)"),
       ("d_S1", "other", "P", r"lowest state, other sector ($S\!=\!1$)")]
fig, axs = plt.subplots(2, 2, figsize=(7.6, 6.6), constrained_layout=True)
(a, b), (c, d) = axs
for nm, ck, mk, lab in SEL:
    if nm in P4:
        curve(a, P4[nm]["hist"], "eps_FN", C[ck], mk, lab)
        curve(b, P4[nm]["hist"], "w_s", C[ck], mk, lab)
logaxis(a, 1e-10, 1.0); logaxis(b, 1e-14, 1.0)
for ax, yl in ((a, r"FN energy error $\epsilon_{\rm FN}=(E_{\rm FN}-E_0)/|E_0|$"),
               (b, r"wrong-sign probability $w_s$")):
    ax.set_xlabel("iteration"); ax.set_ylabel(yl); ax.set_xlim(-10, 790)
a.set_title("plain loop, 4x4", fontsize=9, loc="right"); b.set_title("plain loop, 4x4", fontsize=9, loc="right")
h_, l_ = a.get_legend_handles_labels()
fig.legend(h_, l_, ncol=3, fontsize=7.5, loc="outside upper center", handletextpad=0.3, columnspacing=1.2)
# (c) Anderson
SELA = [("a_base_J2zero_Marshall", "base", "o", "baseline"),
        ("e_randsign_exactamp_s0", "rand", "^", r"exact $|\phi_0|$, random signs"),
        ("c_mix1_w0.1", "mix2", "s", r"$\phi_0$ weight $10^{-1}$ in $\phi_1$"),
        ("c_mix1_w0.01", "mix8", "D", r"$\phi_0$ weight $10^{-2}$ in $\phi_1$"),
        ("b_ex1", "ex1", "v", r"pure $\phi_1$")]
for nm, ck, mk, lab in SELA:
    if nm in A4: curve(c, A4[nm]["hist"], "eps_FN", C[ck], mk, lab, every=20)
logaxis(c, 1e-10, 1.0); c.set_xlim(-5, 400); c.set_xlabel("iteration")
c.set_ylabel(r"$\epsilon_{\rm FN}$"); c.set_title("Anderson-accelerated loop, 4x4", fontsize=9, loc="right")
c.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.38))
# (d) escape iterations vs initial ground-state weight
def esc(h, thr=1e-2):
    e = np.array(h["eps_FN"]); i = np.where(e < thr)[0]; return int(i[0]) if len(i) else np.nan
pts = {"4x4": [], "20": []}
for lat, P in (("4x4", P4), ("20", P20)):
    for nm, r in P.items():
        if nm.startswith(("c_mix1_w", "b_ex1", "c_mix1m")) and "m_w" not in nm:
            w = r["hist"]["ov_gs"][0]
            pts[lat].append((np.log10(max(w, 1e-34)), esc(r["hist"], 1e-2)))
for lat, mk, col, lab in (("4x4", "o", C["mix2"], "4x4 (D=12870)"), ("20", "s", C["sym"], "20 sites (D=184756)")):
    if pts[lat]:
        xy = sorted(pts[lat]); d.plot(*zip(*xy), "-", color=col, lw=0.9); d.plot(*zip(*xy), mk, color=col, ms=5, mfc="white", mew=1.0, label=lab)
d.set_xlabel(r"$\log_{10}$ of $\phi_0$ weight in the start")
d.set_ylabel(r"iterations to $\epsilon_{\rm FN}<10^{-2}$"); d.set_xlim(-34, 0.5)
d.legend(fontsize=7.5, loc="upper right"); d.set_title("plain loop: escape time", fontsize=9, loc="right")
for ax, L in zip((a, b, c, d), "abcd"):
    ax.text(-0.19, 1.02, L, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
fig.savefig(D / "fig_bad_start.png", dpi=200); fig.savefig(D / "fig_bad_start.pdf")
print("saved")
