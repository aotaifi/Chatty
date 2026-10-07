#!/usr/bin/env python3
"""Aggregate results/lanczos_vs_krylov_scaling/lvk_N*.json -> table (markdown), fits, figure."""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'lanczos_vs_krylov_scaling')
files = sorted(glob.glob(os.path.join(R, 'lvk_N*.json')), key=lambda f: int(os.path.basename(f)[5:-5]))
D = [json.load(open(f)) for f in files]
for d in D:   # merge the extra guides (A2, C2) computed in lvk2_N*.json
    for pf in ('lvk2', 'lvk3'):
        f2 = os.path.join(R, f"{pf}_N{d['N']}.json")
        if os.path.exists(f2):
            d['guides'].update(json.load(open(f2))['guides'])
Ns = np.array([d['N'] for d in D])
OPS = ['start', 'K', 'L1', 'KL1', 'L2', 'L1L1']
GUIDES = ['A', 'B', 'Cx', 'Cm', 'A2', 'C2', 'J']
NAMES = {'A': 'A: exact |psi0| + Marshall', 'B': 'B: |GS(J2=0)| + Marshall',
         'Cx': 'C-x: noisy |psi0| + exact signs', 'Cm': 'C-m: noisy |psi0| + Marshall',
         'A2': 'A2: exact |psi0| + signs after 2 Krylov steps', 'C2': 'C2: noisy |psi0| + the same A2 signs',
         'J': 'J: |psi0| e^(kappa n_NN) + exact signs'}


def stat(d, g, op, key):
    v = np.array([r[op][key] for r in d['guides'][g]])
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


tab = {}
for g in GUIDES:
    for op in OPS:
        for key in ('dE', 'w_s', 'var'):
            tab[g, op, key] = np.array([stat(d, g, op, key) for d in D])  # (nN, 2)
gain = lambda g, op: np.array([[np.mean([r['start']['dE'] - r[op]['dE'] for r in d['guides'][g]]),
                                (np.std([r['start']['dE'] - r[op]['dE'] for r in d['guides'][g]], ddof=1) / np.sqrt(len(d['guides'][g])))
                                if len(d['guides'][g]) > 1 else 0.0] for d in D])

lines = []
for g in GUIDES:
    lines.append(f"\n**Guide {NAMES[g]}** (per-site energy error dE = (E-E0)/N, and wrong-sign weight w_s; "
                 f"{'mean over seeds' if g[0]=='C' else 'single deterministic guide'})\n")
    lines.append("| N | start dE | K dE | L1 dE | K+L1 dE | L2 dE | L1,L1 dE | gain K | gain L1 | gain L1/K | w_s start | w_s K | w_s L1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    gK, gL = gain(g, 'K')[:, 0], gain(g, 'L1')[:, 0]
    for i, N in enumerate(Ns):
        t = lambda op, k='dE': tab[g, op, k][i, 0]
        ratio = (gL[i] / gK[i]) if gK[i] > 1e-12 else float('nan')
        lines.append(f"| {N} | {t('start'):.3e} | {t('K'):.3e} | {t('L1'):.3e} | {t('KL1'):.3e} | {t('L2'):.3e} | {t('L1L1'):.3e} | "
                     f"{gK[i]:.3e} | {gL[i]:.3e} | {ratio:.2f} | {t('start','w_s'):.2e} | {t('K','w_s'):.2e} | {t('L1','w_s'):.2e} |")

lines.append("\n**Fractional gain (gain / start error)**: K, L1 (and L2)\n")
lines.append("| guide | quantity | " + " | ".join(f"N={n}" for n in Ns) + " |")
lines.append("|---|---|" + "---|" * len(Ns))
for g in GUIDES:
    st = tab[g, 'start', 'dE'][:, 0]
    for op in ('K', 'L1', 'L2'):
        lines.append(f"| {g} | {op} | " + " | ".join(f"{x:.3f}" for x in gain(g, op)[:, 0] / st) + " |")

# power-law fits  gain ~ c N^-p (N>=20 and all N)
fits = {}
for g in GUIDES:
    for op in ('K', 'L1', 'L2', 'KL1'):
        y = gain(g, op)[:, 0]
        ok = y > 1e-2 * tab[g, 'start', 'dE'][:, 0]
        if ok.sum() >= 3:
            p = np.polyfit(np.log(Ns[ok]), np.log(y[ok]), 1)
            fits[g, op] = (-p[0], np.exp(p[1]))
lines.append("\n**Power-law fits gain(N) ~ c N^(-p)** (all six sizes; p = 0: size-intensive, p = 1: gain ~ 1/N)\n")
lines.append("| guide | p(K) | p(L1) | p(L2) | p(K+L1) |")
lines.append("|---|---|---|---|---|")
for g in GUIDES:
    f = lambda op: f"{fits[g,op][0]:+.2f}" if (g, op) in fits else "n/a (gain 0)"
    lines.append(f"| {g} | {f('K')} | {f('L1')} | {f('L2')} | {f('KL1')} |")
open(os.path.join(R, 'tables.md'), 'w').write("\n".join(lines) + "\n")
print("\n".join(lines))

# cost table
cl = ["\n| N | antiparallel NN bonds | antiparallel NNN bonds | neighbours per config (1 hop) | 2-hop estimate (neighbours^2) |", "|---|---|---|---|---|"]
for d in D:
    c = d['cost']
    cl.append(f"| {d['N']} | {c['nn_hops']:.2f} | {c['nnn_hops']:.2f} | {c['hops_per_config']:.2f} | {c['hops_per_config']**2:.0f} |")
open(os.path.join(R, 'cost.md'), 'w').write("\n".join(cl) + "\n")
print("\n".join(cl))

# figure
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
col = {'K': '#d95f02', 'L1': '#1b9e77', 'L2': '#7570b3', 'KL1': '#666666'}
lab = {'K': 'Krylov sign step', 'L1': 'Lanczos, 1 step', 'L2': 'Lanczos, 2 steps', 'KL1': 'Krylov, then Lanczos 1'}
mk = {'K': 'o', 'L1': 's', 'L2': '^', 'KL1': 'D'}
fig, axs = plt.subplots(2, 4, figsize=(16.0, 7.4), sharex=False)
for ax, g in zip(axs.flat[:7], GUIDES):
    st = tab[g, 'start', 'dE'][:, 0]
    ax.plot(Ns, st, ls=':', color='k', lw=1, marker='x', ms=5, label='start error (reference)')
    for op in ('K', 'L1', 'L2', 'KL1'):
        gg = gain(g, op)
        y, e = gg[:, 0], gg[:, 1]
        ok = y > 1e-12
        if op == 'K' and np.max(y) < 1e-2 * np.max(st):
            ax.text(0.04, 0.05, 'Krylov sign step: gain ~ 0 (no flip lowers E)', transform=ax.transAxes, fontsize=7.5, color=col['K'])
            continue
        if ok.sum() == 0:
            continue
        ax.errorbar(Ns[ok], y[ok], yerr=None, color=col[op], lw=0.9 if op != 'KL1' else 0.7, ls='-' if op != 'KL1' else '--', marker=mk[op], ms=5 if op != 'KL1' else 3.5, label=lab[op], zorder=3 if op != 'KL1' else 1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.set_title(NAMES[g], fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=(2, 3, 5))); ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(10, 10)))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.tick_params(axis='y', which='both', labelsize=7)
ax = axs.flat[7]
for g, c in zip(GUIDES, ['#1f77b4', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#ff7f0e', '#2ca02c']):
    gk, gl = gain(g, 'K')[:, 0], gain(g, 'L1')[:, 0]
    ok = gk > 1e-2 * tab[g, 'start', 'dE'][:, 0]
    if ok.sum():
        ax.plot(Ns[ok], gk[ok] / gl[ok], marker='o', ms=4, lw=0.9, color=c, label=g)
ax.axhline(1, color='k', lw=0.6, ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.minorticks_off()
ax.set_title('ratio  gain(Krylov) / gain(Lanczos 1 step)', fontsize=10); ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.grid(alpha=0.25, lw=0.5)
for ax in axs[1]:
    ax.set_xlabel('N (sites)')
for ax in axs[:, 0]:
    ax.set_ylabel('energy gain per site  ($J_1$)')
axs[0, 0].legend(fontsize=7.5, loc='best', frameon=False)
fig.suptitle('One Lanczos step vs one Krylov sign step, exact J1-J2 ($J_2/J_1=0.5$), symmetric sector', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(R, 'fig_gain_vs_N.png'), dpi=170)
fig.savefig(os.path.join(R, 'fig_gain_vs_N.pdf'))
