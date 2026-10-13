"""Review final evaluation (REVIEW_lanczos8x8.md): T5 + T6 pooled direct VMC of psi_1, relaxation test (amendment 2).
  python rv_final.py OUT.json DATA_DIR      (DATA_DIR: dira/direct.npz, rvD{1,2,3}/direct2.npz, rvL{2,3}/late.npz)
Late-time chains rvL2/rvL3 continue chains of rvD2/rvD3 (same chain index), so they share chain ids in the chain-level SE.
"""
import sys, os, json
import numpy as np

N = 64; AL = -0.160951; ER = -31.92677568; RBMPP = -0.498886; HU2 = -0.49886
OUT, D_ = sys.argv[1], sys.argv[2]


def el1(z):
    return ((1 - AL * ER) * z['EL'] + AL * z['EL2']) / (1 + AL * (z['EL'] - ER)) / N


def cse(E, ch):
    ids, inv = np.unique(ch, return_inverse=True); m = np.bincount(inv, weights=E) / np.bincount(inv)
    return float(m.std(ddof=1) / np.sqrt(len(m))), len(m)


def hill(x, frac=0.02):
    a = np.sort(np.abs(x))[::-1]; k = max(int(frac * len(a)), 10)
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


parts = []                                     # (E, chain id, label, time-class)
z = np.load(os.path.join(D_, 'dira', 'direct.npz')); parts.append((el1(z), z['chain'], 'run_a', 'early'))
for k in (1, 2, 3):
    z = np.load(os.path.join(D_, f'rvD{k}', 'direct2.npz')); E = el1(z); sw = z['sweep']
    parts.append((E[sw == 10], z['chain'][sw == 10] + 10000 * k, f'D{k}_sw10', 'early'))
    parts.append((E[sw == 12], z['chain'][sw == 12] + 10000 * k, f'D{k}_sw12', 'mid'))
for k in (2, 3):
    p = os.path.join(D_, f'rvL{k}', 'late.npz')
    if os.path.exists(p):
        z = np.load(p); ok = np.isfinite(z['EL2']); E = el1(z)[ok]
        parts.append((E, z['chain'][ok] + 10000 * k, f'L{k}', 'late'))
res = dict(parts=[dict(label=l, cls=c, R=len(E), E1_site=float(E.mean()), se=cse(E, ch)[0]) for E, ch, l, c in parts])


def pool(sel):
    E = np.concatenate([p[0] for p in parts if sel(p)]); ch = np.concatenate([p[1] for p in parts if sel(p)])
    se, nc = cse(E, ch); srt = np.sort(E)
    return dict(R=len(E), nchains=nc, E1_site=float(E.mean()), se=se, z_below_RBMPP=(RBMPP - E.mean()) / se,
                z_below_Hu2=(HU2 - E.mean()) / se, hill2=hill(E - E.mean()), median=float(np.median(E)),
                trim10=float(srt[10:-10].mean()), trim30=float(srt[30:-30].mean())), E


res['all'], Eall = pool(lambda p: True)
res['early'], _ = pool(lambda p: p[3] == 'early')
res['mid'], _ = pool(lambda p: p[3] == 'mid')
res['late'], _ = pool(lambda p: p[3] == 'late')
res['mid_plus_late'], _ = pool(lambda p: p[3] in ('mid', 'late'))
d = res['late']['E1_site'] - res['mid']['E1_site']; sd = float(np.hypot(res['late']['se'], res['mid']['se']))
res['relaxation_test'] = dict(late_minus_sweep12=d, se=sd, z=d / sd, relaxation_present=bool(d > 2 * sd))
d2 = res['mid_plus_late']['E1_site'] - res['early']['E1_site']; sd2 = float(np.hypot(res['mid_plus_late']['se'], res['early']['se']))
res['early_vs_rest'] = dict(rest_minus_early=d2, se=sd2, z=d2 / sd2)
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
