#!/usr/bin/env python3
"""Markdown tables for results/tail_vs_N/README.md from metrics.json."""
import json, os
from analyze import RES, load, metrics
data = load()
rows = []
def get(cl, key):
    for s in data[cl]['states']:
        if (s['k'] if s['k'] is not None else 'vit') == key: return s
def fmt_n(x): return f'{x:.2e}' if x >= 1e4 else f'{x:.0f}'
out = []
for key, title in ((1, 'loop after k = 1'), (3, 'loop after k = 3'), (8, 'loop after k = 8'), (6, 'loop after k = 6'), ('vit', 'symmetrised ViT psi_P')):
    lines = []
    for cl in data:
        s = get(cl, key)
        if s is None: continue
        m = metrics(s); m['D'] = data[cl]['D']; r = m['c_exact']
        Nc = s['N_cfg']
        sp = lambda k: (f"{m[k]*100:.2f}% ({m[k+'_n']:.1e})" if k in m else 'n/a')
        lines.append(f"| {s['N']} | {Nc:.2e} | {m['G']:.2e} | {m['Q_over_G']:.2f} | {r['t50']:.2f} | {r['xt50']:+.2f} | "
                     f"{r['mass_below_50']*100:.1f}% | {r['norb_below_50']:.2e} / {m['D']:.2e} | {r['ncfg_below_50']:.2e} ({r['frac_cfg_below_50']*100:.1f}%) | "
                     f"{r['t80tail']:.2f} | {r['mass_below_80tail']*100:.1f}% | {r['norb_below_80tail']:.2e} | {sp('support50')} | {sp('support80')} |")
    if lines:
        out.append(f'\n**{title}**\n')
        out.append('| N | N_cfg | G /site | Q/G | t50 | x50 | phi^2 mass below t50 | orbits below t50 / orbits | configs below t50 (share) | t80 | mass below t80 | orbits below t80 | min #configs (share) for 50% of G | for 80% of G |')
        out.append('|' + '---|' * 14)
        out += lines
open(os.path.join(RES, 'tables.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
