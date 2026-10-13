"""8x8 pair interaction (locality) from the signs_8x8 two-hop referee chunks (README). CPU only.
  python s8_pairint.py OUT.json RUN_DIR
"""
import sys, os, glob, json, time
import numpy as np
L = 8; N = 64; ONE = np.uint64(1)
q = lambda x, y: (x % L) + L * (y % L)
NN, NNN = [], []
for y in range(L):
    for x in range(L):
        i = q(x, y); NN += [(i, q(x + 1, y)), (i, q(x, y + 1))]; NNN += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
B = NN + NNN; NB = len(B)
BI = np.array([b[0] for b in B], np.uint64); BJ = np.array([b[1] for b in B], np.uint64)
MASKS = (ONE << BI) | (ONE << BJ)
JB = np.array([1.0] * len(NN) + [0.5] * len(NNN))


def sdist(p, r):
    dx = abs(p % L - r % L); dy = abs(p // L - r // L); dx = min(dx, L - dx); dy = min(dy, L - dy)
    return np.sqrt(dx * dx + dy * dy)
DB = np.array([[min(sdist(p, r) for p in B[a] for r in B[c]) for c in range(NB)] for a in range(NB)])
DBr = np.round(DB, 3); keys = sorted(set(DBr[DBr > 0].tolist()))
OUT, RUN = sys.argv[1], sys.argv[2]
acc = {k: np.zeros(8) for k in keys}; missing = 0; nsamp = 0; T0 = time.time()
iu = np.triu_indices(NB, 1)
for f in sorted(glob.glob(os.path.join(RUN, 'chunk_*.npz'))):
    d = np.load(f); U2 = d['U2']; la = d['la']; s = d['s'].astype(np.int64)
    def look(c):
        i = np.clip(np.searchsorted(U2, c), 0, len(U2) - 1); ok = U2[i] == c
        return i, ok
    for x in d['X']:
        nsamp += 1
        v = (((x >> BI) ^ (x >> BJ)) & ONE).astype(bool)
        ix, okx = look(np.array([x], np.uint64)); lx, sx = la[ix[0]], s[ix[0]]
        xb = x ^ MASKS; ib, okb = look(xb); lb = la[ib]; sb = s[ib]
        a_, c_ = iu
        m = v[a_] & v[c_] & (DB[a_, c_] > 0)
        a_, c_ = a_[m], c_[m]
        ibb, okbb = look(x ^ MASKS[a_] ^ MASKS[c_])
        good = okbb & okb[a_] & okb[c_]; missing += int((~good).sum())
        a_, c_, ibb = a_[good], c_[good], ibb[good]
        I = la[ibb] + lx - lb[a_] - lb[c_]
        sig = s[ibb] * sx * sb[a_] * sb[c_]
        w = np.exp(np.clip(lb[a_] + lb[c_] - 2 * lx, -60, 60))
        rc = np.exp(np.clip(lb[c_] - lx, -60, 60)); ra = np.exp(np.clip(lb[a_] - lx, -60, 60))
        ce = 0.5 * (rc + ra) * np.abs(sig * np.exp(np.clip(I, -60, 60)) - 1.0)
        dk = DBr[a_, c_]
        for k in keys:
            mm = dk == k
            if not mm.any(): continue
            Ii = I[mm]; wi = w[mm]
            acc[k] += np.array([mm.sum(), np.sum(Ii * Ii), np.sum(np.abs(Ii)), np.sum(sig[mm] < 0), np.sum(wi),
                                np.sum(wi * Ii * Ii), np.sum(ce[mm]), 0])
    print(f'{os.path.basename(f)} samples {nsamp} missing {missing} {time.time() - T0:.0f}s', flush=True)
res = dict(run=RUN, samples=nsamp, missing_pairs=missing, by_distance=[
    dict(dist=k, n=int(a[0]), rms_I=float(np.sqrt(a[1] / a[0])), mean_abs_I=float(a[2] / a[0]),
         sign_defect_frac=float(a[3] / a[0]), wrms_I=float(np.sqrt(a[5] / a[4])), mean_contrib_err=float(a[6] / a[0]))
    for k, a in acc.items() if a[0] > 0])
json.dump(res, open(OUT, 'w'), indent=1)
for r in res['by_distance']: print(r)
print('DONE', time.time() - T0)
