#!/usr/bin/env python3
"""Sketch of the two exact clusters (square-lattice J1-J2, periodic boundaries).

(a) 4x4 torus, periodic vectors T1=(4,0), T2=(0,4).
(b) tilted 20-site torus, T1=(4,0), T2=(1,5).  This matches the builders in
    krylov_sign_structure/experiments/{groundstate_k1_20site_exact,
    closed_fn_krylov_exact20,finite_tau_matching_20site_exact}.py, where the
    sites are (x, y) with x=0..3, y=0..4 and red(x, y) wraps y by 5 while
    shifting x by -1, i.e. (x, y) ~ (x+4, y) ~ (x+1, y+5).  Here each site is
    drawn exactly at the code's (x, y); the parallelogram cell is placed so it
    encloses precisely these 20 points.
Sublattice A = (x+y) even (the Marshall sign counts down spins on A); both
periodic vectors have even x+y, so the checkerboard is consistent.

Writes figures/fig1_lattices.{pdf,png}.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "figures"

plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "legend.fontsize": 8,
})

COL_A = "#1f4e9c"   # sublattice A (blue)
COL_B = "#e08a1e"   # sublattice B (orange)
COL_GHOST = "#cfcfcf"
COL_CELL = "#555555"
COL_T = "#b0243a"
COL_BOND = "#222222"

CLUSTERS = [
    dict(label="(a)  $4\\times4$ ($N=16$)", T1=(4, 0), T2=(0, 4), ex=(1, 1),
         corner=(-0.5, -0.5)),
    dict(label="(b)  tilted 20-site ($N=20$)", T1=(4, 0), T2=(1, 5), ex=(2, 2),
         corner=(-1.0, -0.5)),
]
# Cell corners are chosen so that the integer points inside the parallelogram are
# exactly the code's site set {(x, y): 0 <= x < 4, 0 <= y < LY} (checked below).


def cell_sites(T1, T2, CORNER):
    """Integer points r with r - CORNER = a T1 + b T2, 0 <= a, b < 1."""
    M = np.array([T1, T2], float).T
    Minv = np.linalg.inv(M)
    pts = []
    for x in range(-2, 12):
        for y in range(-2, 12):
            a, b = Minv @ (np.array([x, y]) - CORNER)
            if 0 <= a < 1 and 0 <= b < 1:
                pts.append((x, y))
    return pts


def draw(ax, c):
    T1, T2 = np.array(c["T1"]), np.array(c["T2"])
    CORNER = np.array(c["corner"], float)
    sites = cell_sites(T1, T2, CORNER)
    n = abs(int(T1[0] * T2[1] - T1[1] * T2[0]))
    assert len(sites) == n, (len(sites), n)
    sset = set(sites)
    LY = n // 4
    assert sset == {(x, y) for x in range(4) for y in range(LY)}, sorted(sset)

    corners = np.array([CORNER, CORNER + T1, CORNER + T1 + T2, CORNER + T2])
    xmin, ymin = corners.min(0)
    xmax, ymax = corners.max(0)
    m = 0.72

    # faint periodic images around the cell
    ghosts = set()
    for (x, y) in sites:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                gx, gy = np.array([x, y]) + i * T1 + j * T2
                if (gx, gy) in sset:
                    continue
                if xmin - m < gx < xmax + m and ymin - m < gy < ymax + m:
                    ghosts.add((int(gx), int(gy)))
    if ghosts:
        g = np.array(sorted(ghosts))
        ax.scatter(g[:, 0], g[:, 1], s=9, c=COL_GHOST, lw=0, zorder=1)

    # simulation cell
    ax.add_patch(Polygon(corners, closed=True, fill=True, fc="#f4f4f4",
                         ec=COL_CELL, lw=0.8, zorder=0))

    # couplings on one example site
    ex = np.array(c["ex"])
    for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        p = ex + d
        ax.plot(*zip(ex, p), color=COL_BOND, lw=1.3, zorder=2,
                solid_capstyle="round")
    for d in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        p = ex + d
        ax.plot(*zip(ex, p), color=COL_BOND, lw=1.0, ls=(0, (2.4, 1.6)),
                zorder=2)
    # inline labels on one J1 bond (west) and one J2 bond (south-east)
    box = dict(boxstyle="round,pad=0.14", fc="#f4f4f4", ec="none")
    ax.text(ex[0] - 0.58, ex[1], "$J_1$", ha="center", va="center",
            fontsize=8, zorder=4, bbox=box)
    ax.text(ex[0] + 0.6, ex[1] - 0.6, "$J_2$", ha="center", va="center",
            fontsize=8, zorder=4, bbox=box)

    # sites, coloured by checkerboard sublattice
    s = np.array(sites)
    isA = (s.sum(1) % 2) == 0
    ax.scatter(s[isA, 0], s[isA, 1], s=30, c=COL_A, ec="white", lw=0.5,
               zorder=3)
    ax.scatter(s[~isA, 0], s[~isA, 1], s=30, c=COL_B, ec="white", lw=0.5,
               zorder=3)
    ax.scatter([ex[0]], [ex[1]], s=70, facecolors="none", edgecolors=COL_BOND,
               lw=0.9, zorder=3)

    # periodic vectors from the lower-left corner
    for T, name, off in [(T1, "$\\mathbf{T}_1$", (0.0, -0.36)),
                         (T2, "$\\mathbf{T}_2$", (-0.40, 0.0))]:
        ax.add_patch(FancyArrowPatch(CORNER, CORNER + T, arrowstyle="-|>",
                                     mutation_scale=9, lw=1.3, color=COL_T,
                                     shrinkA=0, shrinkB=0, zorder=5))
        mid = CORNER + 0.5 * T + np.array(off)
        ax.text(*mid, name, color=COL_T, ha="center", va="center",
                fontsize=9)
    ax.text(*(CORNER + T1 + np.array([0.14, 0.0])),
            f"$({T1[0]},{T1[1]})$", color=COL_T, ha="left", va="center",
            fontsize=7.5)
    ax.text(*(CORNER + T2 + np.array([-0.15, 0.0])),
            f"$({T2[0]},{T2[1]})$", color=COL_T, ha="right", va="center",
            fontsize=7.5)

    ax.set_title(c["label"], fontsize=9, pad=2)
    ax.set_aspect("equal")
    ax.axis("off")
    return (xmin - m - 0.15, xmax + m), (ymin - m, ymax + m)


def main():
    # same length scale in both panels: common y-span, width ratios = x-spans
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    lims = [draw(ax, c) for ax, c in zip(axes, CLUSTERS)]
    H = max(yl[1] - yl[0] for _, yl in lims)
    for ax, (xl, yl) in zip(axes, lims):
        yc = 0.5 * (yl[0] + yl[1])
        ax.set_xlim(*xl)
        ax.set_ylim(yc - H / 2, yc + H / 2)
    axes[0].get_gridspec().set_width_ratios([xl[1] - xl[0] for xl, _ in lims])
    handles = [
        Line2D([], [], ls="", marker="o", ms=5.5, mfc=COL_A, mec="white",
               label="sublattice A"),
        Line2D([], [], ls="", marker="o", ms=5.5, mfc=COL_B, mec="white",
               label="sublattice B"),
        Line2D([], [], color=COL_BOND, lw=1.3, label="$J_1$"),
        Line2D([], [], color=COL_BOND, lw=1.0, ls=(0, (2.4, 1.6)),
               label="$J_2$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               handlelength=1.6, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.10,
                        wspace=0.04)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_lattices.pdf")
    fig.savefig(OUT / "fig1_lattices.png", dpi=240)
    print("wrote", OUT / "fig1_lattices.pdf", OUT / "fig1_lattices.png")


if __name__ == "__main__":
    main()
