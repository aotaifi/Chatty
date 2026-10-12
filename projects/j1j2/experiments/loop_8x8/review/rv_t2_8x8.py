"""Review T2/T3 (REVIEW_lanczos8x8.md) on the real 8x8 lattice: independent recomputation of psi_P, E_L, (H^2 psi_P)/psi_P
and E_L1 for referee samples (including the most extreme local energies), with
  - an own bond list built from coordinates (J1: 4 neighbours, J2: 4 diagonals, PBC),
  - an own D4 x spin-flip symmetrisation: the UNPROJECTED network evaluated on 16 images built here from coordinates,
  - an own H application (diagonal J/4 s_i s_j, off-diagonal +J/2),
compared with the stored referee values (which come from l8_core).  Also: invariance of psi_P under random
translations x D4 x flip; batch-composition dependence of log|psi_P| and its sign.
  python rv_t2_8x8.py OUT REF.npz
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import l8_core as C

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); z = dict(np.load(sys.argv[2]))
L = 8; N = 64; AL = -0.160951; ER = -31.92677568; PHI = float(z['phi']); res = dict(phi=PHI)
CK = 'vit_J2=0.50_N=8x8_k=0.mpack'
Gs = C.Guide(CK, sym=True, batch=1024, phi=PHI)
Gu = C.Guide(CK, sym=False, batch=8192, phi=PHI)

# ------------------------------------------------ own lattice
site = lambda x, y: (x % L) + L * (y % L)
B = []
for y in range(L):
    for x in range(L):
        B += [(site(x, y), site(x + 1, y), 1.0), (site(x, y), site(x, y + 1), 1.0),
              (site(x, y), site(x + 1, y + 1), 0.5), (site(x, y), site(x - 1, y + 1), 0.5)]
MBI = np.array([b[0] for b in B], np.uint64); MBJ = np.array([b[1] for b in B], np.uint64); MJ = np.array([b[2] for b in B])
MM = (np.uint64(1) << MBI) | (np.uint64(1) << MBJ)
assert len(set(int(m) for m in MM)) == 256
ONE = np.uint64(1); FULL = np.uint64(2 ** 64 - 1)


def my_nbrs(S):
    S = np.asarray(S, np.uint64)[:, None]
    anti = (((S >> MBI) ^ (S >> MBJ)) & ONE).astype(bool)
    return S ^ MM, anti


def my_diag(anti):
    return np.sum(np.where(anti, -0.25, 0.25) * MJ, axis=1)


# own D4 images: rotations about the origin and reflections, written as coordinate maps
OPS = [lambda x, y: (x, y), lambda x, y: (y, -x), lambda x, y: (-x, -y), lambda x, y: (-y, x),
       lambda x, y: (y, x), lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (-y, -x)]
DST = np.array([[site(*op(i % L, i // L)) for i in range(N)] for op in OPS])     # spin at i moves to DST[g, i]


def image(S, g, flip, tx=0, ty=0):
    S = np.asarray(S, np.uint64); out = np.zeros_like(S)
    for i in range(N):
        j = DST[g, i]; j = site(j % L + tx, j // L + ty)
        out |= ((S >> np.uint64(i)) & ONE) << np.uint64(j)
    return (~out) & FULL if flip else out


def psiP_mine(S):
    """own symmetrisation: sum over 16 images of exp(z_unprojected)."""
    zs = []
    for g in range(8):
        for fl in (0, 1):
            la, ph = Gu.eval(image(S, g, fl)); zs.append(la + 1j * ph)
    zs = np.array(zs); m = zs.real.max(0); s = np.exp(zs - m).sum(0)
    return m + np.log(np.abs(s)), np.angle(s)


sgn = lambda ph: np.where(np.cos(ph - PHI) >= 0, 1.0, -1.0)

# ------------------------------------------------ sample selection: 8 typical + 8 most extreme E_L
EL = z['x_ELa']; R = len(EL); own = z['own']
o = np.argsort(np.abs(EL - np.median(EL)))
pick = list(np.linspace(0, R - 1, 8).astype(int)) + list(o[-8:])
res['picked_EL_site'] = [float(EL[i] / N) for i in pick]
# stored EL2 (l8_eval formula)
hxy = 0.5 * C.JB[z['bond'].astype(np.int64)] * z['x_s'][own] * z['y_s'] * np.exp(np.clip(z['y_la'] - z['x_la'][own], -60, 60))
EL2s = z['x_D'] * EL + np.bincount(own, weights=hxy * z['y_ELa'], minlength=R)
rows = []; t0 = time.time()
for i in pick:
    x = z['X'][i:i + 1]
    n1, a1 = my_nbrs(x); N1 = n1[0][a1[0]]
    n2, a2 = my_nbrs(N1); U = np.unique(np.concatenate([x, N1, n2[a2]]))
    la, ph = psiP_mine(U); psi = sgn(ph) * np.exp(la - la[np.searchsorted(U, x)[0]])     # real psi_P, relative scale
    lu_s, pu_s = Gs.eval(U)                                                           # l8_core projected guide
    look = lambda S: psi[np.searchsorted(U, S)]

    def Hpsi(S):
        nb, an = my_nbrs(S)
        return my_diag(an) * look(S) + np.sum(np.where(an, 0.5 * MJ * look(np.where(an, nb, S[:, None]).ravel()).reshape(an.shape), 0.0), 1)
    hx = Hpsi(x)[0]; hy = Hpsi(N1)
    _, ax = my_nbrs(x); jy = MJ[ax[0]]
    h2x = my_diag(ax)[0] * hx + np.sum(0.5 * jy * hy)
    px = look(x)[0]
    el = hx / px; el2 = h2x / px
    el1 = ((1 - AL * ER) * hx + AL * h2x) / ((1 - AL * ER) * px + AL * hx)
    el1_stored = ((1 - AL * ER) * EL[i] + AL * EL2s[i]) / (1 + AL * (EL[i] - ER))
    rows.append(dict(i=int(i), EL_mine=el, EL_stored=float(EL[i]), dEL=el - EL[i], dEL2=el2 - EL2s[i], dEL1=el1 - el1_stored,
                     EL1_site=el1 / N, n_twohop=len(U), max_dloga_sym=float(np.max(np.abs(la - lu_s))),
                     sign_mismatch=int(np.sum(sgn(ph) != sgn(pu_s))), min_abs_cos=float(np.min(np.abs(np.cos(pu_s - PHI))))))
    print(rows[-1], f'{time.time() - t0:.0f}s', flush=True)
res['twohop'] = rows
res['max_abs_dEL'] = max(abs(r['dEL']) for r in rows); res['max_abs_dEL2'] = max(abs(r['dEL2']) for r in rows)
res['max_abs_dEL1'] = max(abs(r['dEL1']) for r in rows)
res['max_dloga_sym'] = max(r['max_dloga_sym'] for r in rows); res['sign_mismatch_total'] = sum(r['sign_mismatch'] for r in rows)
res['n_states_compared'] = sum(r['n_twohop'] for r in rows)

# ------------------------------------------------ invariance under translations x D4 x flip
rg = np.random.default_rng(1); S = z['X'][rg.choice(R, 512, replace=False)]
l0, p0 = Gs.eval(S); dl = []; sm = 0
for k in range(6):
    g = int(rg.integers(8)); fl = int(rg.integers(2)); tx, ty = (int(v) for v in rg.integers(L, size=2))
    l1, p1 = Gs.eval(image(S, g, fl, tx, ty)); dl.append(float(np.max(np.abs(l1 - l0)))); sm += int(np.sum(sgn(p1) != sgn(p0)))
res['invariance'] = dict(max_dloga=max(dl), sign_mismatch=sm, n=512 * 6)
# stored referee values vs fresh evaluation, and batch-composition dependence
l2, p2 = Gs.eval(z['X'][:2048]); res['stored_vs_fresh'] = dict(max_dloga=float(np.max(np.abs(l2 - z['x_la'][:2048]))),
                                                              sign_mismatch=int(np.sum(sgn(p2) != z['x_s'][:2048])))
perm = rg.permutation(2048); l3, p3 = Gs.eval(z['X'][:2048][perm])
res['batch_order'] = dict(max_dloga=float(np.max(np.abs(l3 - l2[perm]))), sign_mismatch=int(np.sum(sgn(p3) != sgn(p2[perm]))))
C.dump(os.path.join(OUT, 't2.json'), res); print(json.dumps({k: v for k, v in res.items() if k != 'twohop'}, default=float))
