"""C1-T0 (PLAN_ansatz.md): is the exact 6x6 sign structure that of Neel/stripe sign-rule interference?
Exact psi0 (J2/J1 = 0.5), no training, CPU.   python3 c1_t0.py OUT_JSON [TABLE]
Classes by (s_N, s_X, s_Y): K1 (+,+,+), K2 (+,-,-), K3 (-,+,-), K4 (-,-,+); a symmetric C1 ansatz forces one of K1, K2
to the Marshall sign; R = U / W_M against the no-information null R_null = 0.5 W_M(s_N=+1) / W_M.
"""
import sys, json, time
import numpy as np
OUT = sys.argv[1]
TABLE = sys.argv[2] if len(sys.argv) > 2 else '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'
L = 6; N = 36
xs = np.arange(N) % L; ys = np.arange(N) // L
def mask(sel): return np.uint64(sum(1 << int(i) for i in np.nonzero(sel)[0]))
M_N = mask((xs + ys) % 2 == 0); M_X = mask(xs % 2 == 1); M_Y = mask(ys % 2 == 1); M_BAD = mask(xs % 3 == 0)

def popcount(a):                                   # uint64 array -> int popcount (SWAR)
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((a * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)

def rule(x, m): return 1 - 2 * (popcount(x & m) & 1)          # (-1)^{n_up on m}

t0 = time.time()
z = np.load(TABLE); reps = z['reps']; amp = z['amp'].astype(np.float64); orb = z['orbit'].astype(np.float64)
w = orb * amp ** 2; w /= w.sum()
res = dict(table=TABLE, D=int(reps.size), E0_site=float(z['E0']) / N)
assert np.all(popcount(reps) == 18), 'not the Sz = 0 sector'
sN, sX, sY = rule(reps, M_N), rule(reps, M_X), rule(reps, M_Y)
# ---- check 1: identity s_N s_X s_Y = +1, and a broken rule must violate it
res['identity_violations'] = int(np.sum(sN * sX * sY != 1))
res['identity_violations_broken_control'] = int(np.sum(sN * rule(reps, M_BAD) * sY != 1))
# ---- check 2: class labels constant over orbits (s_N fixed, {s_X, s_Y} as a set), random 1e4 reps x 576 images
def space_group():                                               # copy of st6_sector.space_group (no jax)
    def site(x, y): return (x % L) + L * (y % L)
    def pm(f):
        p = np.empty(N, np.int64)
        for y in range(L):
            for x in range(L):
                xx, yy = f(x, y); p[site(x, y)] = site(xx, yy)
        return p
    c4 = pm(lambda x, y: (-y, x)); sx = pm(lambda x, y: (-x, y)); r = np.arange(N); pts = []
    for k in range(4):
        for refl in (False, True):
            q = r.copy()
            for _ in range(k): q = c4[q]
            if refl: q = sx[q]
            pts.append(q)
    return np.array([pm(lambda x, y: (x + tx, y + ty))[q] for ty in range(L) for tx in range(L) for q in pts])
G = space_group(); assert G.shape == (288, N)
rng = np.random.default_rng(0); pick = rng.choice(reps.size, 10000, replace=False)
b = ((reps[pick][:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.uint64)
bad = 0
for p in G:
    for flip in (0, 1):
        bb = np.zeros_like(b); bb[:, p] = b                      # bit i goes to site p[i]
        if flip: bb = 1 - bb
        y = (bb << np.arange(N, dtype=np.uint64)[None, :]).sum(1).astype(np.uint64)
        n2, x2, y2 = rule(y, M_N), rule(y, M_X), rule(y, M_Y)
        bad += int(np.sum((n2 != sN[pick]) | (np.minimum(x2, y2) != np.minimum(sX[pick], sY[pick]))
                          | (np.maximum(x2, y2) != np.maximum(sX[pick], sY[pick]))))
res['orbit_class_inconsistencies'] = bad
# ---- global sign of psi0 by Marshall majority
sg = np.sign(amp); g = 1.0 if np.sum(w * sg * sN) >= 0 else -1.0; sg = sg * g
viol = (sg != sN) & (amp != 0)
W_M = float(np.sum(w * viol)); res['W_M'] = W_M; res['marshall_avg_sign'] = float(np.sum(w * sg * sN))
cls = {'K1': (sN == 1) & (sX == 1) & (sY == 1), 'K2': (sN == 1) & (sX == -1) & (sY == -1),
       'K3': (sN == -1) & (sX == 1) & (sY == -1), 'K4': (sN == -1) & (sX == -1) & (sY == 1)}
assert int(sum(c.sum() for c in cls.values())) == reps.size
res['classes'] = {k: dict(w=float(np.sum(w * c)), count_frac=float(np.sum(orb * c) / orb.sum()),
                          viol_w=float(np.sum(w * c * viol)),
                          viol_frac_in_class=float(np.sum(w * c * viol) / max(np.sum(w * c), 1e-300)))
                  for k, c in cls.items()}
U = min(res['classes']['K1']['viol_w'], res['classes']['K2']['viol_w'])
W_Mp = res['classes']['K1']['viol_w'] + res['classes']['K2']['viol_w']
R = U / W_M; R_null = 0.5 * W_Mp / W_M
res.update(U=U, R=R, R_null=R_null, R_over_null=R / R_null,
           verdict='PASS' if R <= 0.2 * R_null else ('KILL' if R >= 0.6 * R_null else 'INCONCLUSIVE'))
# ---- by psi0 probability decade (per configuration, normalised)
p_conf = amp ** 2 / np.sum(orb * amp ** 2)
dec = np.floor(np.log10(np.maximum(p_conf, 1e-300))).astype(int)
res['by_decade'] = {}
for d in range(int(dec[amp != 0].max()), max(int(dec[amp != 0].min()), -16) - 1, -1):
    m = dec == d
    if not m.any(): continue
    res['by_decade'][str(d)] = dict(w=float(np.sum(w * m)), W_M=float(np.sum(w * m * viol)),
                                    **{k + '_viol': float(np.sum(w * m * c * viol)) for k, c in cls.items()},
                                    **{k + '_w': float(np.sum(w * m * c)) for k, c in cls.items()})
res['sec'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'by_decade'}, indent=1))
