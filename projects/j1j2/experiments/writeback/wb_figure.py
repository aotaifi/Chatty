"""Figure for results/writeback: (a) share of the frozen-FN gain captured during training, (b) exact final numbers.
python wb_figure.py RUN_JSON [RUN_JSON ...]   ->  results/writeback/writeback_6x6.{png,pdf}
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'results', 'writeback')
COL = {'E-M': '#2a78d6', 'I-M': '#eb6834', 'Vfn-M': '#1baf7a', 'Vh-M': '#eda100', 'E-L': '#e87ba4',
       'E-M-lr1e-2': '#008300'}
LAB = {'E-M': 'energy metric, oracle phi (28k)', 'I-M': 'infidelity, oracle phi (28k)',
       'Vfn-M': 'VMC on frozen H_FN (28k)', 'Vh-M': 'VMC on fixed-sign H (28k)',
       'E-L': 'energy metric, oracle phi (111k)', 'E-M-lr1e-2': 'energy metric, oracle, lr 1e-2 (28k)'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

arms, G0, Hstart, Hstep = [], None, None, None
for f in sys.argv[1:]:
    r = json.load(open(f))
    G0 = r['diag']['G0_frozen_gain_site']; Hstart = r['ref']['H_psiP']; Hstep = r['ref']['H_phi_ownsign']
    arms += [a for a in r.get('arms', []) if a['tag'] in COL]
order = [t for t in COL if any(a['tag'] == t for a in arms)]
arms = sorted(arms, key=lambda a: order.index(a['tag']))

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
ends = []
for a in arms:
    v = a['val']; v0 = v[0]['val_LE']
    st = [q['step'] for q in v]; fr = [100 * (v0 - q['val_LE']) / G0 for q in v]
    ax.plot(st, fr, color=COL[a['tag']], lw=2, label=LAB[a['tag']])
    ends.append([fr[-1], st[-1], a['tag']])
ends = [e for e in ends if e[2] in ('E-M', 'E-L', 'I-M', 'Vh-M')]   # selective direct labels; the rest via legend
ends = [[y, x, 'Vfn-M, Vh-M' if t == 'Vh-M' else t] for y, x, t in ends]
for y, x, t in ends:
    ax.annotate(t, (x, y), xytext=(4, 0), textcoords='offset points', va='center', color=INK2, fontsize=8)
ax.axhline(50, color=INK2, lw=1, ls='--'); ax.text(200, 51.5, 'pass: 50% of the gain', color=INK2, fontsize=8)
ax.axhline(7.7, color=INK2, lw=1, ls=':'); ax.text(200, 9.0, '7 one-hop features', color=INK2, fontsize=8)
ax.set_xlabel('Adam steps'); ax.set_ylabel('frozen-FN gain captured (%)')
ax.set_title('(a) training: share of one exact FN iteration written back', loc='left', fontsize=9, color=INK)
ax.grid(axis='y', color=GRID, lw=0.8); ax.set_ylim(-5, 60); ax.set_xlim(0, max(a['val'][-1]['step'] for a in arms) * 1.12)
ax.legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, 0.80))

ys = list(range(len(arms)))[::-1]
for y, a in zip(ys, arms):
    fb = a['final_last']
    fg = 100 * fb['frac_gain']; hg = 100 * (Hstart - fb['H_ownsign']) / (Hstart - Hstep)
    bx.barh(y + 0.18, fg, height=0.34, color=COL[a['tag']])
    bx.barh(y - 0.18, hg, height=0.34, color=COL[a['tag']], alpha=0.45, hatch='////', edgecolor='white', lw=0)
    bx.text(max(fg, 0) + 1, y + 0.18, f'{fg:.1f}%', va='center', fontsize=7.5, color=INK)
    bx.text(max(hg, 0) + 1, y - 0.18, f'{hg:.1f}%', va='center', fontsize=7.5, color=INK2)
bx.set_yticks(ys); bx.set_yticklabels([a['tag'] for a in arms])
bx.axvline(50, color=INK2, lw=1, ls='--'); bx.axvline(0, color=INK2, lw=0.8)
bx.set_xlim(-3, 60); bx.grid(axis='x', color=GRID, lw=0.8)
bx.set_xlabel('% of the exact FN step (exact, final parameters)')
bx.set_title('(b) solid: frozen-FN gain; hatched: <H>(b, s_P) gain', loc='left', fontsize=9, color=INK)
fig.suptitle('Write-back of one FN iteration into a residual factor on the symmetrised ViT (6x6, exact sector)',
             x=0.01, ha='left', fontsize=10, color=INK)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'writeback_6x6.{ext}'), dpi=200)
print('saved')
