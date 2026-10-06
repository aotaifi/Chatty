#!/usr/bin/env python3
"""Figures + tables for results/sign_learnability from summary.json (aggregate.py).
Usage: python make_figures.py [SUMMARY_JSON]"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "sign_learnability"
SUM = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "summary.json"
plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "axes.labelsize": 10, "axes.titlesize": 10,
                     "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9})
NS = [16, 20, 24, 28, 32, 36]
STY = {  # target: (label, colour, marker)
    "gs": (r"exact sign $\sigma_0$", "k", "o"),
    "c1": (r"step label $c_1$", "C0", "s"),
    "c2": (r"step label $c_2$", "C1", "^"),
    "c3": (r"step label $c_3$", "C3", "v"),
    "m2": (r"full sign $s_2 M$", "C2", "D"),
    "m3": (r"full sign $s_3 M$", "C4", "P"),
}
S = json.load(open(SUM))
RUNS = S["runs"]
META = {int(k): v for k, v in S["meta"].items()}


def panel(ax, letter, x=-0.16):
    ax.text(x, 1.03, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def logax(ax, which="y"):
    a = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    a.set_major_locator(LogLocator(base=10, numticks=40))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{np.log10(v):.0f}"))
    a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=40))
    a.set_minor_formatter(NullFormatter())
    ax.grid(axis=which, which="major", alpha=.25, lw=.6)


def save(fig, stem):
    fig.savefig(RES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(RES / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sel(**kw):
    base = dict(beta=0.5, nbr=1, size="M", lam=0.0, epochs=10, max_steps=40000, kern=3, init_seed=0)
    base.update(kw)
    return [r for r in RUNS if all(r.get(k) == v for k, v in base.items())]


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else np.nan


def val(target, N, key, **kw):
    rs = sel(target=target, N=N, **kw)
    if "seed" not in kw:
        rs = [r for r in rs if r["seed"] in (0, 1)]
    return gmean([key(r) if callable(key) else r[key] for r in rs]), rs


def ref_next(r):
    """exact-chain sign error at the same total depth as net + one hop"""
    t = r["target"]
    if t == "gs": return None
    k = int(t[1:])
    st = META[r["N"]]["steps"]
    return st[k + 1]["w_s"] if k + 1 < len(st) else None


FLOOR = 1e-12
NCOL = {24: "#9ecae1", 28: "#4292c6", 32: "#08519c", 36: "#08306b"}
NSC = [24, 28, 32, 36]
CONV = dict(min_steps=80000, max_steps=80000)
SIZE_ORDER = ["XS", "S", "M", "L", "XL"]


def one(**kw):
    rs = sel(**kw)
    return rs[0] if rs else None


def rel(r):
    return max(r["w_lab"], FLOOR) / r["w_triv"] if r else np.nan


def fig_scaling():
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 6.3), gridspec_kw=dict(wspace=.38, hspace=.5))
    ax = ax.ravel()
    # (a) net alone vs N (8e4 steps, n = 3e4, M)
    for t in ("gs", "m2", "c1", "c2", "c3"):
        lab, c, m = STY[t]
        y = [rel(one(target=t, N=N, n=30000, seed=0, **CONV)) for N in NSC]
        ax[0].plot(NSC, y, "-", color=c, lw=.8, alpha=.8); ax[0].plot(NSC, y, m, color=c, ms=5, label=lab)
    ax[0].axhline(1, color="0.5", lw=.7, ls=":")
    logax(ax[0]); ax[0].set_ylabel(r"$\log_{10}\, w_\mathrm{net} / w_\mathrm{triv}$")
    ax[0].set_title(r"net alone ($M$ net, $n=3\cdot10^4$)")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right", handletextpad=.3, labelspacing=.2)
    # (b) contraction of one exact hop on top of the net vs the exact chain
    for t in ("gs", "m2", "c1", "c2"):
        lab, c, m = STY[t]
        y = []
        for N in NSC:
            r = one(target=t, N=N, n=30000, seed=0, **CONV)
            y.append(r["w_hop"] / r["w_net"] if r and r["w_net"] > 0 else np.nan)
        ax[1].plot(NSC, y, "-", color=c, lw=.8, alpha=.8); ax[1].plot(NSC, y, m, color=c, ms=5, label=lab)
    for k, ls in ((1, "--"), (2, ":")):
        y = [META[N]["steps"][k + 1]["w_s"] / META[N]["steps"][k]["w_s"] for N in NSC]
        ax[1].plot(NSC, y, ls, color="0.45", lw=1.0, label=rf"exact chain $s_{k}\to s_{k+1}$")
    logax(ax[1]); ax[1].set_ylabel(r"$\log_{10}\, w_\mathrm{hop} / w_\mathrm{net}$")
    ax[1].set_title("one exact hop on top of the net")
    h_, l_ = ax[1].get_legend_handles_labels()
    ax[1].legend(h_[-2:], l_[-2:], frameon=False, fontsize=7, loc="lower right", handletextpad=.3, labelspacing=.2)
    # (c) samples, (d) parameters; gs filled, c1 open
    for N in NSC:
        for t, mfc, ls in (("gs", NCOL[N], "-"), ("c1", "none", "--")):
            pts = [(r["n"], rel(r)) for r in sel(target=t, N=N, seed=0, size="M", **CONV)]
            if pts:
                x, y = zip(*sorted(pts)); ax[2].plot(x, y, ls, color=NCOL[N], lw=.8)
                ax[2].plot(x, y, "o", color=NCOL[N], mfc=mfc, ms=4.5)
            pts = [(r["nparams"], rel(r)) for r in RUNS if r["target"] == t and r["N"] == N and r["seed"] == 0 and r["n"] == 30000
                   and r["beta"] == 0.5 and r["nbr"] == 1 and r["lam"] == 0 and r["min_steps"] == 80000 and r["max_steps"] == 80000]
            if pts:
                x, y = zip(*sorted(pts)); ax[3].plot(x, y, ls, color=NCOL[N], lw=.8)
                ax[3].plot(x, y, "o", color=NCOL[N], mfc=mfc, ms=4.5)
    for a in ax[2:]:
        logax(a); logax(a, "x"); a.set_ylabel(r"$\log_{10}\, w_\mathrm{net} / w_\mathrm{triv}$")
    ax[2].set_xlabel(r"$\log_{10}$ tempered samples $n$ ($+$ neighbours)"); ax[2].set_title(r"data ($M$ net, 28k parameters)")
    ax[3].set_xlabel(r"$\log_{10}$ parameters"); ax[3].set_title(r"capacity ($n=3\cdot10^4$)")
    hl = [Line2D([], [], color=NCOL[N], marker="o", ls="-", ms=4, lw=.8) for N in NSC] + \
         [Line2D([], [], color="0.3", marker="o", ls="-", ms=4, lw=.8), Line2D([], [], color="0.3", marker="o", mfc="none", ls="--", ms=4, lw=.8)]
    ax[3].legend(hl, [f"N={N}" for N in NSC] + [r"$\sigma_0$", r"$c_1$"], frameon=False, fontsize=7, loc="lower left",
                 handletextpad=.3, labelspacing=.2, ncol=2)
    for a in ax[:2]:
        a.set_xlabel("sites $N$"); a.set_xticks(NSC)
    for a, l in zip(ax, "abcd"): panel(a, l)
    fig.suptitle(r"Learning the sign with the exact amplitude ($J_2/J_1=0.5$, symmetric sector, $8\cdot10^4$ Adam steps)", fontsize=9, y=.995)
    save(fig, "fig_sign_learnability_scaling")


def fig_mechanism():
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.2), gridspec_kw=dict(wspace=.36))
    H = S["hists"]
    key = lambda t, N: f"{t}_b0.5_s0_n30000_M_nbr1_lam0.0_ep10_mx80000@N{N}"
    # (a) wrong fraction per decade of per-configuration |psi0|^2
    for t, ls, mfc in (("gs", "-", None), ("c1", "--", "none")):
        for N in (28, 32, 36):
            h = H.get(key(t, N))
            if not h: continue
            D = h["decades"]; tot = np.array(D["tot"]); lw = np.array(D["lab_wrong"]); dec = np.array(D["dec"]) + .5
            m = (tot > 1e-12) & (lw > 0)
            ax[0].plot(dec[m], lw[m] / tot[m], ls, color=NCOL[N], lw=.8)
            ax[0].plot(dec[m], lw[m] / tot[m], "o", color=NCOL[N], ms=3.5, mfc=mfc or NCOL[N])
            if t == "gs":
                med = dec[np.argmin(np.abs(np.cumsum(tot) - 0.5))]
                ax[0].axvline(med, color=NCOL[N], lw=.7, ls=":")
    logax(ax[0]); ax[0].set_xlabel(r"$\log_{10} |\psi_0(x)|^2$ per configuration")
    ax[0].set_ylabel(r"$\log_{10}$ wrong fraction per decade")
    ax[0].set_title(r"errors sit in the low-$|\psi_0|$ tail")
    hl = [Line2D([], [], color=NCOL[N], ls="-", lw=.8, marker="o", ms=3.5) for N in (28, 32, 36)] + \
         [Line2D([], [], color="0.3", marker="o", ls="-", ms=3.5, lw=.8), Line2D([], [], color="0.3", marker="o", mfc="none", ls="--", ms=3.5, lw=.8),
          Line2D([], [], color="0.3", ls=":", lw=.7)]
    ax[0].legend(hl, [f"N={N}" for N in (28, 32, 36)] + [r"$\sigma_0$", r"$c_1$", "median weight"], frameon=False, fontsize=6.8,
                 loc="lower left", ncol=2, handletextpad=.3, labelspacing=.2)
    # (b) margin: c1 net errors vs all states, cumulative in |r0 - T0|
    for N in (28, 32, 36):
        h = H.get(key("c1", N))
        if not h: continue
        Mg = h["margin"]; e = np.array(Mg["edges_log10"]); xe = 10 ** e[1:]
        for k_, ls, lw_ in (("wrong", "-", 1.1), ("all", "--", .8)):
            v = np.array(Mg[k_]["pos"]) + np.array(Mg[k_]["neg"])
            ax[1].plot(xe, np.cumsum(v) / v.sum(), ls, color=NCOL[N], lw=lw_)
    ax[1].set_xscale("log"); ax[1].set_xlim(1e-3, 1e2); ax[1].set_ylim(0, 1.02)
    ax[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{np.log10(v):.0f}"))
    ax[1].set_xlabel(r"$\log_{10} |r_0(x) - T_0|$"); ax[1].set_ylabel(r"cumulative $|\psi_0|^2$ weight fraction")
    ax[1].set_title(r"$c_1$ errors sit at the threshold")
    hl = [Line2D([], [], color="0.3", ls="-", lw=1.1), Line2D([], [], color="0.3", ls="--", lw=.8)] + \
         [Line2D([], [], color=NCOL[N], ls="-", lw=1.1) for N in (28, 32, 36)]
    ax[1].legend(hl, [r"$c_1$ net errors", "all states"] + [f"N={N}" for N in (28, 32, 36)], frameon=False, fontsize=7.2,
                 loc="upper left", handletextpad=.3, labelspacing=.2)
    # (c) optimisation: error vs Adam steps (n = 1e4, M)
    for N in (28, 32, 36):
        for t, mfc, ls in (("gs", NCOL[N], "-"), ("c1", "none", "--")):
            rs = [r for r in RUNS if r["target"] == t and r["N"] == N and r["n"] == 10000 and r["size"] == "M" and r["seed"] == 0
                  and r["beta"] == 0.5 and r["nbr"] == 1 and r["lam"] == 0 and (r["epochs"] == 10 and r["max_steps"] in (40000, 20000, 80000))
                  and (r["max_steps"] == 40000 or r["min_steps"] == r["max_steps"])]
            pts = sorted((r["nsteps"], rel(r)) for r in rs)
            if pts:
                x, y = zip(*pts); ax[2].plot(x, y, ls, color=NCOL[N], lw=.8); ax[2].plot(x, y, "o", color=NCOL[N], mfc=mfc, ms=4.5)
    logax(ax[2]); logax(ax[2], "x"); ax[2].set_xlabel(r"$\log_{10}$ Adam steps (batch 1024)")
    ax[2].set_ylabel(r"$\log_{10}\, w_\mathrm{net} / w_\mathrm{triv}$"); ax[2].set_title(r"optimisation ($M$ net, $n=10^4$)")
    hl = [Line2D([], [], color="0.3", marker="o", ls="-", ms=4, lw=.8), Line2D([], [], color="0.3", marker="o", mfc="none", ls="--", ms=4, lw=.8)]
    ax[2].legend(hl, [r"$\sigma_0$", r"$c_1$"], frameon=False, fontsize=7.5, loc="lower left")
    for a, l in zip(ax, "abc"): panel(a, l, x=-0.2)
    save(fig, "fig_sign_learnability_mechanism")




def tables():
    """markdown tables for README (8e4-step protocol)."""
    out = []
    f = lambda x: "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1e}"
    out.append("| N | target | w_triv | w_net (label err.) | rel. | w_net vs ED | w_hop | exact chain at same depth |")
    out.append("|---|---|---|---|---|---|---|---|")
    for N in NSC:
        for t in ("gs", "m2", "c1", "c2", "c3"):
            r = one(target=t, N=N, n=30000, seed=0, **CONV)
            if not r: continue
            k = None if t == "gs" else int(t[1:])
            ref = "0" if t == "gs" else f"s{k}: {f(META[N]['steps'][k]['w_s'])}, s{k+1}: {f(META[N]['steps'][k+1]['w_s'])}"
            out.append(f"| {N} | {t} | {f(r['w_triv'])} | {f(r['w_lab'])} | {f(rel(r))} | {f(r['w_net'])} | {f(r['w_hop'])} | {ref} |")
    out.append("")
    out.append("| N | target | " + " | ".join(SIZE_ORDER[1:]) + " |")
    out.append("|---|---|" + "---|" * 4)
    for N in NSC:
        for t in ("gs", "c1", "c2"):
            row = []
            for sz in SIZE_ORDER[1:]:
                r = one(target=t, N=N, n=30000, seed=0, size=sz, **CONV)
                row.append(f(rel(r)) if r else "-")
            out.append(f"| {N} | {t} | " + " | ".join(row) + " |")
    out.append("")
    out.append("| N | target | n=1e3 | 1e4 | 3e4 | 1e5 |")
    out.append("|---|---|---|---|---|---|")
    for N in NSC:
        for t in ("gs", "c1"):
            row = []
            for n in (1000, 10000, 30000, 100000):
                r = one(target=t, N=N, n=n, seed=0, **CONV)
                row.append(f(rel(r)) if r else "-")
            out.append(f"| {N} | {t} | " + " | ".join(row) + " |")
    (RES / "tables_8e4.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    fig_scaling()
    fig_mechanism()
    tables()
