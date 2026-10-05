#!/usr/bin/env python3
"""Figures for results/sign_design_tests (one per experiment: a, b, c). Usage: python make_figures.py [a|b|c ...]"""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "sign_design_tests"
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "axes.labelsize": 10, "axes.titlesize": 10,
                     "legend.fontsize": 8.3, "xtick.labelsize": 9, "ytick.labelsize": 9})
FLOOR = 1e-14


def panel(ax, letter):
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def decades(ax):
    """Log y axis labelled by the exponent only (tick k means 10^k)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=40))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{np.log10(v):.0f}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=40))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", alpha=.25, lw=.6)


def save(fig, stem):
    fig.savefig(RES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(RES / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_b():
    cols = plt.cm.viridis(np.linspace(0.0, 0.9, 6))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(wspace=.38))
    for n, c in zip((16, 20, 24, 28, 32, 36), cols):
        j = json.load(open(RES / f"b_depth_N{n}.json")); rows = [r for r in j["rows"] if r["d"] <= 6]
        d = np.array([r["d"] for r in rows]); ws = np.array([r["w_s"] for r in rows]); ep = np.array([r["eps"] for r in rows])
        for a_, v in zip(ax, (ws, np.abs(ep))):
            ex = v < FLOOR
            a_.plot(d, np.maximum(v, FLOOR), "-", color=c, lw=.9, alpha=.8)
            a_.plot(d[~ex], v[~ex], "o", color=c, ms=4.5, label=f"N={n}" if a_ is ax[0] else None)
            a_.plot(d[ex], np.full(ex.sum(), FLOOR), "v", mfc="none", color=c, ms=5)
    for a_, lab, t in zip(ax, (r"wrong-sign probability $w_s$", r"relative energy error $\epsilon$"), ("sign error", "energy error")):
        decades(a_); a_.axhline(FLOOR, color="0.6", lw=.6, ls=":"); a_.set_ylim(5e-15, 0.1); a_.set_xlim(-0.2, 6.2)
        a_.set_xlabel("Krylov depth $d$ (exact amplitude)"); a_.set_ylabel(r"$\log_{10}$ " + lab); a_.set_title(t)
    ax[0].axhline(1e-6, color="C3", lw=.7, ls="--")
    ax[0].text(6.1, 1.6e-6, r"$10^{-6}$", color="C3", fontsize=7.5, ha="right")
    from matplotlib.lines import Line2D
    h, l = ax[0].get_legend_handles_labels()
    h.append(Line2D([], [], marker="v", mfc="none", color="0.3", ls="", ms=5)); l.append(r"exact ($<10^{-14}$)")
    ax[0].legend(h, l, frameon=False, ncol=1, fontsize=7.5, loc="lower left", handletextpad=.3, labelspacing=.25)
    panel(ax[0], "a"); panel(ax[1], "b")
    fig.suptitle("Krylov depth from Marshall with the exact amplitude, symmetric sector", fontsize=9, y=1.02)
    save(fig, "fig_b_depth_vs_N")


def fig_a():
    ex = json.load(open(ROOT / "results/stored_signs/runs_4x4/main2_exact.json"))["history"]
    runs = {w: [json.load(open(f)) for f in sorted(glob.glob(str(RES / f"a4x4_{w}_s?.json")))] for w in ("net", "hop")}
    its = np.arange(0, 41)
    def ser(runs_, key, q):
        return np.array([[(r["history"][0][q] if i == 0 else r["history"][i][key][q]) for i in its] for r in runs_])
    exw = np.array([[h["w_s"] for h in ex[:41]]]); exe = np.array([[h["eps"] for h in ex[:41]]])
    spec = [("exact recursive loop", "k", "o", exw, exe)]
    if runs["net"]:
        spec.append(("stored net alone", "C3", "s", ser(runs["net"], "net", "w_s"), ser(runs["net"], "net", "eps")))
        spec.append(("net + 1 hop (same $a$)", "C0", "^", ser(runs["net"], "hop_same", "w_s"), ser(runs["net"], "hop_same", "eps")))
    if runs["hop"]:
        spec.append(("loop fed by net + 1 hop", "C2", "D", ser(runs["hop"], "work_srec", "w_s"), ser(runs["hop"], "work_srec", "eps")))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=.36))
    for lab, c, m, W, E in spec:
        for a_, A in zip(ax, (W, E)):
            A = np.maximum(A, 1e-12); mean = np.exp(np.mean(np.log(A), axis=0))
            a_.plot(its, mean, "-", color=c, lw=.9, alpha=.85)
            a_.plot(its[::3], mean[::3], m, color=c, ms=4, label=lab if a_ is ax[0] else None)
            if len(A) > 1: a_.fill_between(its, A.min(0), A.max(0), color=c, alpha=.15, lw=0)
    for a_, lab, t in zip(ax, (r"wrong-sign probability $w_s$", r"relative energy error $\epsilon$"), ("sign error", "energy error")):
        decades(a_); a_.set_xlabel("loop iteration"); a_.set_ylabel(r"$\log_{10}$ " + lab); a_.set_title(t); a_.set_xlim(0, 40)
    ax[0].set_ylim(3e-7, 0.05); ax[1].set_ylim(1e-5, 0.1)
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper right", handletextpad=.3, labelspacing=.3)
    panel(ax[0], "a"); panel(ax[1], "b")
    fig.suptitle("4x4, stored tempered sign net (N=1e4 samples), exact FN amplitude; mean and range of 2 seeds", fontsize=8.5, y=1.02)
    save(fig, "fig_a_net_plus_hop_4x4")


def fig_c():
    E0s = 0.5038096538908783
    d4 = {n: json.load(open(RES / f"c_delta_4x4_{n}.json"))["rows"] for n in ("iid", "smooth")}
    sig = sorted(set(r["sigma"] for r in d4["iid"]))
    def mean(rows, var, q, s):
        return np.mean([r[var][q] for r in rows if r["sigma"] == s])
    c6 = json.load(open(RES / "c6.json")) if (RES / "c6.json").exists() else None
    spec = [("plain $r$, iid noise", "C0", "o", "-", "iid", "r"), ("plain $r$, smooth noise", "C0", "s", "--", "smooth", "r"),
            (r"$\delta_V=r-r_{\rm kept}$ (smooth)", "C3", "^", "-", "smooth", "dV"),
            (r"$\delta=r-r_{\rm FN}\equiv 0$ (Marshall)", "0.4", "v", ":", "smooth", "delta")]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(wspace=.38))
    for lab, c, m, ls, nz, var in spec:
        for a_, q in zip(ax, ("w_s", "eps_sign")):
            y = np.array([mean(d4[nz], var, q, s) for s in sig])
            a_.plot(sig, np.maximum(y, 1e-9), ls, color=c, lw=.9, alpha=.85)
            a_.plot(sig, np.maximum(y, 1e-9), m, color=c, ms=4.5, label=lab if a_ is ax[0] else None)
    if c6:
        v = c6["variables"]
        for var, c, xo in (("r", "C0", 0.05), ("dV", "C3", 0.07), ("delta", "0.4", 0.03)):
            t = v[var]["Topt"]
            ax[0].plot([xo], [max(t["w_s"], 1e-9)], "*", color=c, ms=11, mec="k", mew=.5, zorder=5)
            ax[1].plot([xo], [max(t["dE_site"] / E0s, 1e-9)], "*", color=c, ms=11, mec="k", mew=.5, zorder=5,
                       label=r"stars: 6x6 ViT ($\sigma\approx0.05$)" if var == "r" else None)
    for a_, lab, t in zip(ax, (r"wrong-sign probability $w_s$", r"sign-attributable energy error"), ("sign error", "energy error")):
        decades(a_); a_.set_xlabel(r"log-amplitude noise $\sigma$"); a_.set_ylabel(r"$\log_{10}$ " + lab); a_.set_title(t)
    ax[0].set_ylim(2e-5, 0.05); ax[1].set_ylim(3e-4, 0.1)
    ax[0].legend(frameon=False, fontsize=7, loc="lower right", handletextpad=.3, labelspacing=.3)
    if c6: ax[1].legend(frameon=False, fontsize=7, loc="lower right")
    panel(ax[0], "a"); panel(ax[1], "b")
    save(fig, "fig_c_delta_vs_r")


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c"]
    for w in which:
        {"c": fig_c, "a": fig_a, "b": fig_b}.get(w, lambda: print("no figure", w))()
