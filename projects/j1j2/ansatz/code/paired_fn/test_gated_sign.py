#!/usr/bin/env python3
"""test_gated_sign.py OUT.json [--n2 2000] [--n3 20000]: controls 2 and 3 of the gated one-step sign (CPU or any GPU; wchunk small for the 4 GB T1000).
Control 2: s_B at d_gate = +inf vs the stage-0 label sign K[s306] = sg * y on stored stage-0 states (table.npz); broken: alpha sign flipped.
Control 3: flip rate s_B != s306 per decade -3..-7 at d_gate = -5 on stage-0 tempered states, stratified by decade, vs the stage-0 table rate per decade
  (population rate of the table, and state-by-state agreement); broken: alpha flipped, and gate -7 (must show 0 flips at -5/-6)."""
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = '/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0'
os.environ.setdefault('GFMC_M1_CODE', BASE + '/code/m1'); os.environ.setdefault('G8_PROD8', '/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s306/train/params_final.npz')
os.environ.setdefault('G8_PROD8_SHA', '45fa54e601bcf9c1fc79d24825066e9f58bbabca855b6a0de88ce6e120c36c5d'); os.environ.setdefault('G8_CONSTS', BASE + '/inputs/consts_s306.json')
os.environ.setdefault('S306_CODE', BASE + '/code')
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--n2', type=int, default=2000); ap.add_argument('--n3', type=int, default=20000)
ap.add_argument('--wchunk', type=int, default=32); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--table', default='/project/theorie/a/A.Otaifi/runs/j1j2/sign_strong/s0/r/table.npz'); a = ap.parse_args()
import jax; print('device', jax.devices(), flush=True)
import gated_sign as GS
T = np.load(a.table); keys = T['keys']; lB2 = T['lB2']; sg = T['sg']; y = T['y']; K = (sg * y).astype(np.int8)
dec_tab = np.floor((lB2 - GS.MREF) / GS.LN10).astype(int); rg = np.random.default_rng(a.seed)
def spins(k): return (((np.asarray(k, np.uint64)[:, None] >> np.arange(64, dtype=np.uint64)[None]) & np.uint64(1)).astype(np.int8) * 2 - 1)
cache = GS.Cache(cap=10 ** 9)
def mk(gate, broken=None):
    g, info = GS.build(gate, cache, broken=broken, wchunk=a.wchunk); return g
def evalz(g, Z, bs=4096):
    out = [g.signs_of_states(Z[i:i + bs]) for i in range(0, len(Z), bs)]; return [np.concatenate([o[j] for o in out]) for j in range(4)]
res = dict(alpha_E=GS.read_sign_constants(), consts_sha=GS.CONSTS_SHA, table=a.table, n_table=int(len(keys)), table_flip_rate=float((y < 0).mean()))
t0 = time.time()
# ------------------------------------------------ control 2
idx = np.unique(np.concatenate([rg.choice(len(keys), a.n2, replace=False)] + [rg.choice(np.nonzero(dec_tab == d)[0], 100, replace=False) for d in range(-9, -2)]))
Z = spins(keys[idx]); res['c2_n'] = int(len(idx)); res['c2_true_flips'] = int((y[idx] < 0).sum())
g = mk(math.inf); s, sb, dec, EL = evalz(g, Z)
c2 = dict(mismatch_sB_vs_K=int((sb != K[idx]).sum()), mismatch_s306_vs_sg=int((s != sg[idx]).sum()), max_abs_dEL=float(np.max(np.abs(EL - T['ELM'][idx]))),
          decade_mismatch=int((dec != dec_tab[idx]).sum()), min_abs_f=float(np.min(np.abs(1 + g.alpha * (EL - g.E)))), n_flipped_sB=int((sb != s).sum()))
gb = mk(math.inf, 'alpha_flip'); s_, sbb, _, _ = evalz(gb, Z); c2['broken_alpha_flip_mismatch'] = int((sbb != K[idx]).sum())
gb = mk(math.inf, 'hashmask'); gb.hash_p = 1e-2; s_, sbb, _, _ = evalz(gb, Z); c2['broken_hashmask1e-2_mismatch'] = int((sbb != K[idx]).sum())
c2['PASS'] = bool(c2['mismatch_sB_vs_K'] == 0 and c2['mismatch_s306_vs_sg'] == 0 and c2['broken_alpha_flip_mismatch'] > 0)
res['control2'] = c2; print('CONTROL2', json.dumps(c2), f'{time.time() - t0:.0f}s', flush=True); json.dump(res, open(a.out, 'w'), indent=1)
# ------------------------------------------------ control 3
rows = {}
for d in range(-3, -8, -1):
    pool = np.nonzero(dec_tab == d)[0]; sel = rg.choice(pool, min(a.n3, len(pool)), replace=False); Z = spins(keys[sel])
    r = dict(n=int(len(sel)), n_decade_in_table=int(len(pool)), table_rate=float((y[pool] < 0).mean()), sample_true_rate=float((y[sel] < 0).mean()))
    for nm, gate, brk in (('g-5', -5, None), ('g-5_alpha_flip', -5, 'alpha_flip'), ('g-7', -7, None)):
        gg = mk(gate, brk); s, sb, dec, EL = evalz(gg, Z); fl = (sb != s); r[nm] = dict(flip_rate=float(fl.mean()), flips=int(fl.sum()),
            err=float(math.sqrt(max(fl.mean() * (1 - fl.mean()), 1e-12) / len(sel))), decade_mismatch=int((dec != d).sum()))
        if nm == 'g-5': r['g-5']['state_by_state_vs_table'] = int((fl != (y[sel] < 0)).sum() if d <= -5 else fl.sum())
    rows[str(d)] = r; print('DECADE', d, json.dumps(r), f'{time.time() - t0:.0f}s', flush=True); res['control3'] = rows; json.dump(res, open(a.out, 'w'), indent=1)
ok = all(rows[str(d)]['g-5']['flips'] == 0 for d in (-3, -4)) and all(rows[str(d)]['g-5']['state_by_state_vs_table'] == 0 for d in (-5, -6, -7))
res['control3_PASS'] = bool(ok); res['sec'] = time.time() - t0; json.dump(res, open(a.out, 'w'), indent=1); print('CONTROL3', 'PASS' if ok else 'FAIL', flush=True)
