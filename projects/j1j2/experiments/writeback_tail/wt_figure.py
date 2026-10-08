"""Figure for results/writeback_tail:
(a) training: share of one FN iteration's frozen-FN gain captured (common validation estimator), 28k arms;
(b) exact final shares for every trained arm and the exact (training-free) projection bounds;
(c) where the gain sits: per-decade share of the quadratic gain of delta vs what the best arms capture.
python wt_figure.py WT_EXACT_JSON RUN_JSON [RUN_JSON ...] -> results/writeback_tail/writeback_tail_6x6.{png,pdf}
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'results', 'writeback_tail')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
CURVE = {'O-R28': (C1, '-', 'Adam, 28k, old'), 'TA-R28': (C2, '-', 'Adam, 28k, adaptive tail'),
         'O-R111': (C1, '--', 'Adam, 111k, old'), 'TA-R111': (C2, '--', 'Adam, 111k, adaptive tail'),
         'SR-O-R28': (C1, ':', 'minSR, 28k, old'), 'SR-TA-R28': (C2, ':', 'minSR, 28k, adaptive tail')}

ex = json.load(open(sys.argv[1]))
arms = {}
for f in sys.argv[2:]:
    r = json.load(open(f))
    if r.get('spec', {}).get('n_iter', 1) != 1: continue          # multi-iteration targets: README only
    G0 = r['G0']
    for a in r.get('arms', []): arms[a['tag']] = a

plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(14.5, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.95], wspace=0.66)
ax, bx, cx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# ---------------------------------------------------------------- (a) curves
for tag, (col, ls, lab) in CURVE.items():
    if tag not in arms: continue
    v = arms[tag]['val']; v0 = v[0]['val_LE']
    st = [q['step'] for q in v]; fr = [100 * (v0 - q['val_LE']) / G0 for q in v]
    ax.plot(st, fr, color=col, lw=2, ls=ls, label=lab)
ax.axhline(50, color=INK2, lw=1, ls='--'); ax.text(300, 51.5, 'reopen bar: 50%', color=INK2, fontsize=7.5)
ax.axhline(17.3, color=MUTED, lw=1, ls=(0, (1, 2)), label='writeback E-M (2026-10-07)')
ax.set_xlabel('training steps (1024 configurations x 8 neighbours each)')
ax.set_ylabel('frozen-FN gain captured (%, validation estimate)')
ax.set_title('(a) training: old vs adaptive-tail sampling', loc='left', fontsize=9, color=INK)
ax.set_ylim(-3, 60); ax.set_xlim(0, 21000); ax.grid(axis='y', color=GRID, lw=0.8)
ax.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.83), ncol=2)

# ---------------------------------------------------------------- (b) final shares + exact bounds
rows = []   # (label, value %, kind)
order = [('R28k', ['O-R28', 'T-R28', 'TA-R28', 'TX-R28', 'TAX-R28']), ('R111k', ['O-R111', 'TA-R111']),
         ('R28k + log a input', ['O-R28a', 'T-R28a', 'TA-R28a']), ('R28k, bulk samples only', ['TB-R28']),
         ('minSR', ['SR-O-R28', 'SR-TA-R28', 'SR-O-ViT']), ('Gauss-Newton / LM', ['GN-O-R28', 'GN-TA-R28'])]
LABEL = {'O-R28': 'Adam R28k, old', 'T-R28': 'Adam R28k, tail', 'TA-R28': 'Adam R28k, adaptive tail',
         'TX-R28': 'Adam R28k, tail, identity loss', 'TAX-R28': 'Adam R28k, adapt. tail, identity loss',
         'O-R111': 'Adam R111k, old', 'TA-R111': 'Adam R111k, adaptive tail',
         'O-R28a': 'Adam R28k + log a, old', 'T-R28a': 'Adam R28k + log a, tail', 'TA-R28a': 'Adam R28k + log a, adapt. tail',
         'TB-R28': 'Adam R28k, bulk samples only', 'SR-O-R28': 'minSR R28k, old', 'SR-TA-R28': 'minSR R28k, adaptive tail',
         'SR-O-ViT': 'minSR, ViT itself (155k), old', 'GN-O-R28': 'Gauss-Newton R28k, old',
         'GN-TA-R28': 'Gauss-Newton R28k, adapt. tail'}
for grp, tags in order:
    for t in tags:
        if t in arms:
            kind = 'old' if ('O-' in t and not t.startswith('TO')) else 'tail'
            rows.append((LABEL.get(t, t), 100 * arms[t]['final_best']['frac_gain'], kind))
bas = {b['basis']: b for b in ex['bases']}
res = {r['set']: r for r in ex['restricted']}
rows.append(('any f, bulk >= 1e-9 (92% mass)', 100 * res['bulk>=1e-9']['exact_frac'], 'bound'))
rows.append(('any f, bulk >= 1e-10 (98% mass)', 100 * res['bulk>=1e-10']['exact_frac'], 'bound'))
rows.append(('any f, bulk >= 1e-11 (99.6% mass)', 100 * res['bulk>=1e-11']['exact_frac'], 'bound'))
rows.append(('any f, tail < 1e-10 (1.8% mass)', 100 * res['tail<1e-10']['exact_frac'], 'bound'))
rows.append(('bins of (log a, V, W), 1650', 100 * bas['(log a, log V, log W), 16^3']['exact_frac'], 'bound'))
rows.append(('bins of log a, 256 groups', 100 * bas['log a, 256 bins']['exact_frac'], 'bound'))
col = {'old': C1, 'tail': C2, 'bound': '#b5b4ae'}
y = np.arange(len(rows))[::-1]
for yy, (lab, val, kind) in zip(y, rows):
    bx.barh(yy, val, height=0.68, color=col[kind])
    bx.text(max(val, 0) + 1.2, yy, f'{val:.1f}%', va='center', fontsize=7, color=INK)
bx.set_yticks(y); bx.set_yticklabels([r[0] for r in rows], fontsize=7)
bx.axvline(50, color=INK2, lw=1, ls='--'); bx.axvline(0, color=INK2, lw=0.8)
bx.set_xlim(-25, 100); bx.grid(axis='x', color=GRID, lw=0.8)
bx.set_xlabel('exact frozen-FN gain captured (%)')
bx.set_title('(b) trained arms (blue: old, orange: tail) and exact bounds (grey)', loc='left', fontsize=9, color=INK)

# ---------------------------------------------------------------- (c) per-decade
dec = [d for d in ex['delta_gain_decades'] if -16 <= d['decade'] <= -6]
xs = np.array([d['decade'] for d in dec]); share = np.array([100 * d['share'] for d in dec])
cx.bar(xs, share, width=0.72, color='#d9d8d2', label='gain of one FN step (per decade)')
best_tail = max([t for t in arms if 'O-' not in t and t != 'TB-R28' and 'captured_decades' in arms[t]['final_best']],
                key=lambda t: arms[t]['final_best']['frac_gain'], default=None)
for t, c_, mk in (('O-R28', C1, 'o'), ('O-R111', '#7fb0eb', 'o'), (best_tail, C2, 's')):
    if t is None or t not in arms or 'captured_decades' not in arms[t]['final_best']: continue
    cd = {d['decade']: 100 * d['share'] for d in arms[t]['final_best']['captured_decades']}
    cx.plot(xs, [cd.get(k, 0) for k in xs], color=c_, lw=2, marker=mk, ms=5, label=f'captured by {LABEL.get(t, t)}')
cx.axvspan(-16.5, -9.5, color='#f3f2ee', zorder=-5)
cx.text(-16.3, -1.3, 'tail: per-configuration phi^2 < 1e-10', fontsize=7, color=INK2)
cx.set_xlabel('decade of per-configuration phi_FN^2 (log10)')
cx.set_ylabel('share of the quadratic gain Q(delta) (%)')
cx.set_title('(c) where the gain sits and what is captured', loc='left', fontsize=9, color=INK)
cx.set_xticks(xs); cx.grid(axis='y', color=GRID, lw=0.8); cx.set_ylim(-2.2, 28)
cx.legend(frameon=False, fontsize=6.8, loc='upper right')
fig.suptitle('Write-back of one FN iteration with energy-metric (tail) sampling, 6x6 J1-J2, exact sector, oracle phi_FN',
             x=0.01, ha='left', fontsize=10, color=INK)
fig.subplots_adjust(left=0.045, right=0.99, top=0.86, bottom=0.11)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'writeback_tail_6x6.{ext}'), dpi=200)
print('saved')
