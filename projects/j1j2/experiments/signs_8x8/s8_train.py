"""Train the composite stored sign net at 8x8 on the TR data (tempered beta = 1/2 samples + ALL valid neighbours).
  arm V: y = M s_V sgn(T - r_V)   (composite of the ViT sign and one Krylov step, relative to Marshall)
  arm M: y = sgn(T - r_M)         (one Krylov step from Marshall, relative to Marshall)
Held out: chains with id % 5 == 0.  The held-out label error IS the disagreement of s_net with the exact one-hop
Krylov sign K1 (s_net = M sgn(logit), K1 = M y).
  python s8_train.py OUT --arm V --T -20.059 --data train.npz [--steps 60000 --ch 64 --depth 6 --seed 0]
"""
import argparse, json, os, time, sys
import numpy as np
import s8_net as SN

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--arm', required=True); ap.add_argument('--T', type=float, required=True)
ap.add_argument('--data', required=True); ap.add_argument('--steps', type=int, default=60000)
ap.add_argument('--ch', type=int, default=64); ap.add_argument('--depth', type=int, default=6)
ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--batch', type=int, default=1024)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--nheld-log', type=int, default=20000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
log = lambda *s: print(time.strftime('[%H:%M:%S]'), *s, flush=True)
P = dict(np.load(a.data))
r = P['r_V'] if a.arm == 'V' else P['r_M']
c = np.where(r <= a.T, 1, -1)
y = (P['M'].astype(int) * P['s_V'].astype(int) * c) if a.arm == 'V' else c
held = P['chain'] % 5 == 0
res = dict(args=vars(a), n_states=int(len(y)), n_train=int((~held).sum()), n_held=int(held.sum()),
           label_flip_frac_train=float(np.mean(c[~held] < 0)), label_flip_frac_held_x=float(np.mean(c[held & P['is_x']] < 0)),
           label_flip_frac_held_nbr=float(np.mean(c[held & ~P['is_x']] < 0)), label_plus_frac=float(np.mean(y > 0)))
log('DATA', json.dumps(res))
rg = np.random.default_rng(a.seed)
hv = np.nonzero(held)[0]; hv = rg.choice(hv, min(a.nheld_log, len(hv)), replace=False)
net = SN.SignNet(a.ch, a.depth, seed=a.seed); res['npar'] = net.npar
t0 = time.time()
hist = SN.train(net, P['S'][~held], y[~held], P['S'][hv], y[hv], a.steps, lr=a.lr, batch=a.batch, seed=a.seed, log=log)
res['train_sec'] = time.time() - t0; res['hist'] = hist
net.save(os.path.join(a.out, f'net_{a.arm}.pkl'))
# full evaluation
t1 = time.time(); lo = net.logit(P['S']); dt = time.time() - t1
res['cnn_states_per_sec_sym16'] = len(lo) / dt
pred = np.where(lo >= 0, 1, -1); wrong = pred != y
for nm, m in (('train', ~held), ('held_x', held & P['is_x']), ('held_nbr', held & ~P['is_x'])):
    res[f'err_{nm}'] = float(wrong[m].mean())
    res[f'err_{nm}_on_flips'] = float(wrong[m & (c < 0)].mean()) if (m & (c < 0)).any() else None
    res[f'err_{nm}_on_keeps'] = float(wrong[m & (c > 0)].mean())
    res[f'triv_{nm}'] = float(np.mean(c[m] < 0))          # error of the no-flip predictor (base sign)
# error vs margin |r - T| on held-out
mg = np.abs(r - a.T); bins = [0, 0.3, 1, 3, 10, 30, 1e9]
res['held_err_by_margin'] = [dict(lo=bins[i], hi=bins[i + 1], n=int((held & (mg >= bins[i]) & (mg < bins[i + 1])).sum()),
                                  err=float(wrong[held & (mg >= bins[i]) & (mg < bins[i + 1])].mean()) if (held & (mg >= bins[i]) & (mg < bins[i + 1])).any() else None)
                             for i in range(len(bins) - 1)]
# error vs log a decade on held-out
la = P['la']; edges = np.percentile(la[held], [0, 10, 25, 50, 75, 90, 100])
res['held_err_by_la_quantile'] = [dict(lo=float(edges[i]), hi=float(edges[i + 1]),
                                       err=float(wrong[held & (la >= edges[i]) & (la <= edges[i + 1])].mean())) for i in range(6)]
json.dump(res, open(os.path.join(a.out, f'train_{a.arm}.json'), 'w'), indent=1, default=float)
log('DONE', json.dumps({k: v for k, v in res.items() if k != 'hist'}, default=float))
