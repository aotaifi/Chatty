"""Transfer test of one frozen-H_FN SR direction (tuning of shift / sample size for ll6_iter.py).

Draws train sets (sizes --ntrs) and ONE large independent validation set from |a_k|^2, computes the SR
direction for each shift (and the plain gradient), and reports the paired held-out change of the frozen
energy dL(trust) = L_val(theta - eta d) - L_val(theta) (reweighted, chain-jackknife SE) for several trust radii.
Usage: python ll6_tune.py --state S.json [--ntrs 2048,8192] [--nval 8192] [--shifts 0.01,0.1,1,10,inf]
"""
import argparse, json, time
import numpy as np
import ll6_core as C
import jax, jax.numpy as jnp

ap = argparse.ArgumentParser()
ap.add_argument('--state', required=True)
ap.add_argument('--ntrs', default='2048,8192')
ap.add_argument('--nval', type=int, default=8192)
ap.add_argument('--valchains', type=int, default=2048)
ap.add_argument('--shifts', default='0.01,0.1,1,10,inf')
ap.add_argument('--trusts', default='0.005,0.01,0.02,0.04')
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--out', default='tune')
args = ap.parse_args()
st = json.load(open(args.state))
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype='float32')
amp_objs = {}
def amp_of(p):
    if p not in amp_objs: amp_objs[p] = C.Amp(net, net.flat0 if p == 'base' else jnp.asarray(np.load(p), net.DT))
    return amp_objs[p]
guide = C.Guide([amp_of(p) for p in st['chain_params']], st['Ts'], amp_of(st['amp_g']))
theta = net.flat0 if st['amp_g'] == 'base' else jnp.asarray(np.load(st['amp_g']), net.DT)


def draw(n, nch, seed):
    ch = C.Chains(net, nch, seed); ch.advance(theta, args.burn)
    return np.concatenate([ch.advance(theta, args.between) for _ in range(n // nch)])


def resid(th, d, X):
    return net.logabs(th, X) - d['lax'], net.logabs(th, d['child']) - d['lac']


t0 = time.time()
Xv = draw(args.nval, args.valchains, 1000 * args.seed + 5)
dv = C.local_chunked(guide, Xv, 256)
rxv, rcv = resid(theta, dv, Xv); ELv, _ = C.trial_elocs(dv, rxv, rcv)
C.log('VAL', len(Xv), 'L/site', ELv.mean() / 36, 'sec', time.time() - t0)
nch = args.valchains


def paired(th):
    rx1, rc1 = resid(th, dv, Xv); EL1, _ = C.trial_elocs(dv, rx1, rc1)
    w = np.exp(2 * (rx1 - rxv)); w = w / w.mean()
    v1 = (w * EL1).reshape(-1, nch); w2 = w.reshape(-1, nch); v0 = ELv.reshape(-1, nch)
    full = v1.sum() / w2.sum() - v0.mean(); reps = []
    for c in range(nch):
        m = np.ones(nch, bool); m[c] = False
        reps.append(v1[:, m].sum() / w2[:, m].sum() - v0[:, m].mean())
    reps = np.asarray(reps)
    return float(full), float(np.sqrt((nch - 1) / nch * np.sum((reps - reps.mean()) ** 2)))


res = dict(args=vars(args), L_val_site=float(ELv.mean() / 36), rows=[])
for ntr in [int(x) for x in args.ntrs.split(',')]:
    X = draw(ntr, min(ntr, 2048), 1000 * args.seed + ntr)
    d = C.local_chunked(guide, X, 256)
    rx, rc = resid(theta, d, X); EL, _ = C.trial_elocs(d, rx, rc)
    n = len(X); L = EL.mean()
    O = net.jac(theta, X); Ob = (O - O.mean(0, keepdims=True)) / float(np.sqrt(n)); del O   # python float keeps float32
    eps = jnp.asarray((EL - L) / np.sqrt(n), Ob.dtype)
    Km = np.asarray(Ob @ Ob.T, np.float64)
    ev = np.linalg.eigvalsh(Km)
    C.log('NTR', n, 'L/site', L / 36, 'K eig top/median', ev[-1], np.median(ev))
    for sh in args.shifts.split(','):
        if sh == 'inf':
            dvec = Ob.T @ (2 * eps)
        else:
            alpha = np.linalg.solve(Km + float(sh) * np.eye(n), 2.0 * np.asarray(eps, np.float64))
            dvec = Ob.T @ jnp.asarray(alpha, Ob.dtype)
        rms = float(jnp.linalg.norm(Ob @ dvec))
        pred = float(2 * jnp.dot(eps, Ob @ dvec) * np.sqrt(n) / np.sqrt(n))   # train first-order gain per unit eta
        for tr in [float(x) for x in args.trusts.split(',')]:
            eta = tr / rms
            dl, se = paired(theta - eta * dvec)
            row = dict(ntr=n, shift=sh, trust=tr, dL_val=dl, dL_val_se=se, dL_val_site=dl / 36,
                       train_first_order=-eta * pred, sec=time.time() - t0)
            res['rows'].append(row); C.log('ROW', json.dumps(row))
    del Ob
json.dump(res, open(f'{args.out}.json', 'w'), indent=1)
