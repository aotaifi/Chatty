#!/usr/bin/env python3
"""Collect raw run JSONs (fetched from the cluster results dir) -> thinned JSONs, markdown tables, figure.
usage: summarize.py RAWDIR OUTDIR"""
import json, sys, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

raw = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
L10 = np.log10


def loop(name):
    p = raw / name
    if not p.exists(): return None
    d = json.load(open(p)); return d['history'], d['terminal']


def vmc(name):
    p = raw / name
    if not p.exists(): return None
    d = json.load(open(p)); h = d['history']
    step = np.array([r['step'] for r in h]); cpu = np.array([r['cpu_h'] for r in h])
    eps = np.array([r['eps_guide'] for r in h]); epsn = np.array([r['eps'] for r in h]); ws = np.array([r['w_s'] for r in h])
    return dict(step=step, cpu=cpu, eps_guide=eps, eps_net=epsn, w_s=ws)


def smooth(x, n=100):
    c = np.cumsum(np.r_[0, x]); y = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - n); y[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return y


def thin_loop(name):
    r = loop(name)
    if r is None: return None
    hist, term = r
    keys = ['it', 'cpu_h', 'E', 'eps', 'w_s', 'F_amp', 'E_FN', 'eps_FN', 'frozen_gain_frac', 'phi_logratio_rms_before', 'phi_logratio_rms',
            'cf_exactphi_eps', 'cf_exactphi_w_s', 'sr_accepted', 'sr_stop', 'n_sign_flips', 'samples_used', 'sec']
    return dict(terminal=term, history=[{k: h[k] for k in keys if k in h} for h in hist])


# ---- datasets ----
NETS = {'H': dict(label='net H (no sign prior)', vmc_cont={'N4000_lr05': 'Hcont_N4000_lr05_ckpt.json', 'N4000_lr02': 'Hcont_N4000_lr02_ckpt.json',
                                                           'N1e4_lr02': 'Hcont_N1e4_lr02_ckpt.json'}),
        'G': dict(label='net G (Marshall sign rule)', vmc_cont={'N4000_lr02': 'Gcont_N4000_lr02_ckpt.json'}),
        'J': dict(label='net J (short training)', vmc_cont={'N4000_lr05': 'Jcont_N4000_lr05_ckpt.json'})}
LOOPS = {n: sorted(os.path.basename(p) for p in glob.glob(str(raw / f'loop{n}_*.json'))) for n in NETS}
EXACT = {n: (raw / f'exactloop_{n}.json') for n in NETS}

thin = {}
for n in NETS:
    for f in LOOPS[n]:
        thin[f] = thin_loop(f)
    if EXACT[n].exists(): thin[f'exactloop_{n}.json'] = dict(history=json.load(open(EXACT[n])))
    for k, f in NETS[n]['vmc_cont'].items():
        v = vmc(f)
        if v is not None:
            sel = np.arange(0, len(v['step']), 20)
            thin[f'vmccont_{n}_{k}.json'] = {kk: [float(x) for x in vv[sel]] for kk, vv in v.items()}
for f, d in thin.items():
    (out / f).write_text(json.dumps(d))

# ---- start (training) summaries ----
trains = {}
for n, f in (('H', 'H_nom_fast_ckpt.json'), ('G', 'G_ms_fast_s3e-3_ckpt.json'), ('J', 'J_nom_short_ckpt.json')):
    d = json.load(open(raw / f)); h = d['history']
    trains[n] = dict(final=h[-1], cpu_h=d['cpu_hours'], steps=d['steps_done'], args=d['args'],
                     last100_eps=float(np.mean([r['eps'] for r in h[-100:]])), last100_eps_guide=float(np.mean([r['eps_guide'] for r in h[-100:]])),
                     min_eps=float(min(r['eps'] for r in h)))
(out / 'training_summary.json').write_text(json.dumps(trains, indent=1))

# ---- tables ----
lines = []
for n in NETS:
    lines.append(f'\n### {NETS[n]["label"]}: loop runs\n')
    lines.append('| run | iters | cpu_h | eps(it0) | eps best / final | E_FN eps final | w_s final | SR steps accepted |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for f in LOOPS[n] + [f'exactloop_{n}.json']:
        if f not in thin or thin[f] is None: continue
        h = thin[f]['history']
        last = h[-1]
        eps = [r['eps'] for r in h[1:]]; efn = [r.get('eps_FN', r.get('E_FN')) for r in h]
        cpu = last.get('cpu_h', float('nan'))
        acc = sum(r.get('sr_accepted', 0) for r in h[1:]) if 'sr_accepted' in last else '-'
        lines.append(f"| {f.replace('.json','')} | {len(h)-1} | {cpu:.1f} | {h[0]['eps']:.2e} | {min(eps):.2e} / {last['eps']:.2e} | {last['eps_FN']:.2e} | {last['w_s']:.1e} | {acc} |")
(out / 'tables.md').write_text('\n'.join(lines))
print('\n'.join(lines))

# ---- equal-cost comparison ----
cmp_lines = ['\n| net | loop run | loop cpu_h | loop eps / E_FN eps | VMC cont (same extra cpu_h) eps_guide, 100-step mean [best mean so far] |', '|---|---|---|---|---|']
for n in NETS:
    for f in LOOPS[n]:
        if thin.get(f) is None: continue
        h = thin[f]['history']; c = h[-1]['cpu_h']
        row = []
        for k, vf in NETS[n]['vmc_cont'].items():
            v = vmc(vf)
            if v is None: continue
            t = v['cpu'] - v['cpu'][0]; sm = smooth(v['eps_guide'])
            i = int(np.searchsorted(t, c)); i = min(i, len(t) - 1)
            row.append(f"{k}: {sm[i]:.2e} [{sm[:i+1][99:].min() if i>=99 else sm[:i+1].min():.2e}] (t={t[i]:.1f})")
        cmp_lines.append(f"| {n} | {f.replace('.json','')} | {c:.1f} | {h[-1]['eps']:.2e} / {h[-1]['eps_FN']:.2e} | " + '; '.join(row) + ' |')
(out / 'equal_cost.md').write_text('\n'.join(cmp_lines))
print('\n'.join(cmp_lines))

# ---- VMC continuation final numbers ----
for n in NETS:
    for k, vf in NETS[n]['vmc_cont'].items():
        v = vmc(vf)
        if v is None: continue
        t = v['cpu'] - v['cpu'][0]; sm = smooth(v['eps_guide'])
        print(n, k, 'extra cpu_h %.1f' % t[-1], 'steps', int(v['step'][-1]), 'eps_guide 100-step mean now %.2e' % sm[-1],
              'w_s mean last100 %.1e' % v['w_s'][-100:].mean())

# ---- figure ----
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'1e4': '#999999', '1e5': '#0072B2', '1e6': '#009E73', '1e7': '#D55E00', 'exact': '#000000', 'vmc': '#CC79A7'}
fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.2))
(a, b), (c, d) = ax

def draw_loop(axx, f, key, color, label, ls='-', ms=4):
    t = thin.get(f)
    if t is None: return
    h = t['history']
    x = [r['it'] for r in h]; y = [r[key] for r in h]
    axx.plot(x, L10(np.maximum(y, 1e-12)), ls, color=color, lw=0.8, marker='o', ms=ms, label=label)

SEL = [('loopH_N1e4_s0.json', '1e4', 'loop N=1e4'), ('loopH_N1e5_s0.json', '1e5', 'loop N=1e5'),
       ('loopH_N1e6_s0.json', '1e6', 'loop N=1e6'), ('loopH_N1e7_s0.json', '1e7', 'loop N=1e7')]
t0 = trains['H']['final']
# best VMC continuation (lowest final smoothed eps_guide over the H runs)
vm = {k: vmc(f) for k, f in NETS['H']['vmc_cont'].items() if vmc(f) is not None}
vbest = min(vm, key=lambda k: smooth(vm[k]['eps_guide'])[-1])
vfin = smooth(vm[vbest]['eps_guide'])[-1]
for axx, key in ((a, 'eps'), (b, 'eps_FN'), (c, 'w_s')):
    for f, ck, lab in SEL:
        draw_loop(axx, f, key, C[ck], lab)
    draw_loop(axx, 'exactloop_H.json', key, C['exact'], 'ideal loop (exact phi_FN)', ls='--', ms=2.5)
a.axhline(L10(t0['eps_guide']), color='0.5', lw=0.8, ls=':'); a.text(29.5, L10(t0['eps_guide']) + 0.05, 'trained net', color='0.35', ha='right', fontsize=8)
a.axhline(L10(vfin), color=C['vmc'], lw=0.8, ls=':'); a.text(29.5, L10(vfin) - 0.17, f'VMC continued ({vm[vbest]["cpu"][-1]-vm[vbest]["cpu"][0]:.0f} CPU-h)', color=C['vmc'], ha='right', fontsize=8)
b.axhline(L10(t0['eps_guide']), color='0.5', lw=0.8, ls=':')
c.axhline(L10(t0['w_s']), color='0.5', lw=0.8, ls=':')
c.axhline(L10(np.mean(vm[vbest]['w_s'][-100:])), color=C['vmc'], lw=0.8, ls=':')
for axx, yl in ((a, r'$\log_{10}\,\varepsilon$ of $\langle H\rangle_{\rm guide}$'), (b, r'$\log_{10}\,\varepsilon$ of $E_{\rm FN}$'), (c, r'$\log_{10}\,w_s$')):
    axx.set_xlabel('loop iteration'); axx.set_ylabel(yl)
    axx.set_xlim(-0.5, 30)
a.set_ylim(-5.6, -3.2); b.set_ylim(-5.6, -3.2); c.set_ylim(-7.0, -3.4)
a.legend(frameon=False, fontsize=7.5, loc='lower left')
# (d) eps vs cumulative extra CPU-h
for f, ck, lab in SEL[1:]:
    t = thin.get(f)
    if t is None: continue
    h = t['history']
    d.plot(L10(np.maximum([r['cpu_h'] for r in h[1:]], 1e-2)), L10([r['eps'] for r in h[1:]]), '-', color=C[ck], lw=0.8, marker='o', ms=4, label=lab)
vl = {'N4000_lr05': ('VMC cont., lr 0.05', C['vmc'], '-'), 'N4000_lr02': ('VMC cont., lr 0.02', '#E69F00', '-'), 'N1e4_lr02': ('VMC cont., N=1e4, lr 0.02', '#56B4E9', '-')}
for k, v in vm.items():
    t = v['cpu'] - v['cpu'][0]; sm = smooth(v['eps_guide'])
    sel = np.arange(100, len(t), 100)
    d.plot(L10(t[sel]), L10(sm[sel]), '-', color=vl[k][1], lw=0.8, marker='s', ms=3, label=vl[k][0])
d.axhline(L10(t0['eps_guide']), color='0.5', lw=0.8, ls=':')
d.set_xlabel(r'$\log_{10}$ extra CPU-h (16 cores)'); d.set_ylabel(r'$\log_{10}\,\varepsilon$ of $\langle H\rangle_{\rm guide}$')
d.set_ylim(-4.05, -3.35)
d.legend(frameon=False, fontsize=7.5, loc='lower left', ncol=2)
for axx, l in zip((a, b, c, d), 'abcd'):
    axx.text(-0.16, 1.04, f'({l})', transform=axx.transAxes, fontsize=11, fontweight='bold')
fig.suptitle('4x4 J1-J2 (J2/J1=0.5): FN/Krylov loop started from a VMC-trained network (net H, 23.7k params)', fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out / 'net_start_4x4.png', dpi=200); fig.savefig(out / 'net_start_4x4.pdf')
print('figure ok')
