"""Amendment T0-phys (README): (1) the T0 part-(c) one-hop construction for base = ViT and RBM+PP (same-construction
control); (2) physics of the near-node sets: Marshall-violating J1 bonds, Neel vs stripe moments and plaquettes,
ferromagnetic-bond density on X_0.1, W_2, W_1 and by psi0 decade. Exact 6x6, no training.
  python cx_t0phys.py OUT SPEC.json   SPEC: {"refresh": run dir, "la_star_s1": path, "A": RBM+PP sym table}
"""
import os, sys, time
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_t0phys.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
def efn(la, s):
    _, u, info = sec.fn_solve(la, s); return info['dE_FN_site']
res = dict(spec=SPEC)
# ---------------------------------------------------------------- signs and sets (as in cx_veto, tau = 0.1)
z = np.load(TABLE); amp = np.asarray(z['amp'], np.float64); del z
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32)
la0 = jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)); del amp
la_P, s_P = lab.lP0, lab.sP0
_, V, W = lab.VW(la_P, s_P); v = sec.vec(la_P, s_P); E = float(v @ sec.H(v)); del v
EL = lab.D0 + V - W; del V, W
def sg(a): return s_P * jnp.where(1.0 + a * (EL - E) >= 0, 1.0, -1.0).astype(s_P.dtype)
def lclip(a): return la_P + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (EL - E)), 0.3))
sd = float(jnp.sqrt(jnp.sum(sec.vec(la_P, s_P) ** 2 * (EL - E) ** 2)))
r_ = minimize_scalar(lambda a: lab.H_site(lclip(a), sg(a)), bounds=(-3.0 / sd, 0.0), method='bounded', options=dict(xatol=1e-4 / sd))
s1 = sg(float(r_.x)); del EL
RF = SPEC['refresh']
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
la_star = norm_la(jnp.asarray(np.load(SPEC['la_star_s1'])))
X = jnp.exp(la_star - norm_la(la_P)) < 0.1; del la_star
res['check_s2_equals_s1_flipped_on_X'] = int(jnp.sum(jnp.where(X, -s1, s1) != s2))
nW = sec.n / jnp.sum(sec.n)
P0 = sec.vec(la0, s0) ** 2
g = 1.0 if float(jnp.sum(P0 * s0 * s2)) >= 0 else -1.0
s0 = s0 * g; nz0 = P0 > 0
W2 = (s2 != s0) & nz0; W1 = (s1 != s0) & nz0
dec0, _ = lab.decade_of(la0)
log('sets: X', float(jnp.sum(nW * X)), 'W2 psi0', float(jnp.sum(P0 * W2)), 'check', res['check_s2_equals_s1_flipped_on_X'])
# ---------------------------------------------------------------- per-configuration physics features
xs = np.arange(N) % 6; ys = np.arange(N) // 6
cN = jnp.asarray(((-1.0) ** (xs + ys)).astype(np.float32)); cX = jnp.asarray(((-1.0) ** xs).astype(np.float32))
cY = jnp.asarray(((-1.0) ** ys).astype(np.float32))
PA = jnp.asarray(np.arange(N)); PB = jnp.asarray((xs + 1) % 6 + 6 * ys); PC = jnp.asarray(xs + 6 * ((ys + 1) % 6))
PD = jnp.asarray((xs + 1) % 6 + 6 * ((ys + 1) % 6))
MASK_A = jnp.uint64(int(sum(1 << i for i in range(N) if (xs[i] + ys[i]) % 2 == 0)))
NNN_ = len(SS.NN); CH = 4096
@jax.jit
def feat_chunk(i0):
    ii = jnp.minimum(i0 + jnp.arange(CH), sec.D - 1)
    xx = sec.reps[ii]; sig = bits(xx)                                   # (CH, 36) in {-1, +1}
    mN = sig @ cN / N; mX = sig @ cX / N; mY = sig @ cY / N
    a, b, c, d = sig[:, PA], sig[:, PB], sig[:, PC], sig[:, PD]
    neel = jnp.mean((a == d) & (b == c) & (a != b), 1)
    stripe = jnp.mean(((a == c) & (b == d) & (a != b)) | ((a == b) & (c == d) & (a != c)), 1)
    vv = valid_bonds(xx)[:, :NNN_]                                      # antiparallel J1 pairs
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :NNN_], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(CH, NNN_)
    nv = jnp.maximum(jnp.sum(vv, 1), 1)
    v0 = jnp.sum(vv & (s0[ii][:, None] * s0[jj] > 0), 1) / nv
    v2 = jnp.sum(vv & (s2[ii][:, None] * s2[jj] > 0), 1) / nv
    ferro = 1.0 - jnp.sum(vv, 1) / NNN_
    mar = jnp.where(jax.lax.population_count(xx & MASK_A) % 2 == 0, 1.0, -1.0)
    return jnp.stack([mN ** 2, mX ** 2 + mY ** 2, neel, stripe, ferro, v0, v2, mar], 1)
FEAT = np.concatenate([np.asarray(feat_chunk(i0)) for i0 in range(0, sec.D, CH)], 0)[:sec.D]
names = ['M_neel2', 'M_stripe2', 'plaq_neel', 'plaq_stripe', 'ferro_J1', 'J1_viol_s0', 'J1_viol_s2']
P0n = np.asarray(P0); nWn = np.asarray(nW); s0n = np.asarray(s0); mar = FEAT[:, 7]
gm = 1.0 if float(np.sum(P0n * s0n * mar)) >= 0 else -1.0
mwrong = (s0n != gm * mar) & np.asarray(nz0)
d0 = np.asarray(dec0); nzn = np.asarray(nz0)
sets = {'all': nzn, 'X_0.1': np.asarray(X) & nzn, 'W2': np.asarray(W2), 'W1': np.asarray(W1),
        'bulk_dec>=-8': (d0 >= -8) & nzn, 'tail_dec-10..-12': (d0 <= -10) & (d0 >= -12) & nzn}
for q in range(-4, -16, -1): sets[f'dec{q}'] = (d0 == q) & nzn
res['physics'] = {}
for nm, m in sets.items():
    wP = P0n * m; wC = nWn * m; sP = wP.sum(); sC = wC.sum()
    if sC <= 0: continue
    rec = dict(w_psi0=float(sP), w_count=float(sC),
               psi0={k: float((wP * FEAT[:, j]).sum() / sP) if sP > 0 else None for j, k in enumerate(names)},
               count={k: float((wC * FEAT[:, j]).sum() / sC) for j, k in enumerate(names)})
    rec['psi0']['marshall_wrong'] = float((wP * mwrong).sum() / sP) if sP > 0 else None
    rec['count']['marshall_wrong'] = float((wC * mwrong).sum() / sC)
    res['physics'][nm] = rec
    p = rec['psi0']
    if sP > 0:
        log(f'[{nm}] w {sP:.2e}  J1viol(s0) {p["J1_viol_s0"]:.4f} (s2 {p["J1_viol_s2"]:.4f})  Mwrong {p["marshall_wrong"]:.4f}  '
            f'MN2 {p["M_neel2"]:.4f} MS2 {p["M_stripe2"]:.4f}  plaqN {p["plaq_neel"]:.3f} plaqS {p["plaq_stripe"]:.3f}  ferro {p["ferro_J1"]:.3f}')
del FEAT; dump(F, res)
# ---------------------------------------------------------------- (1) the T0 (c) construction for ViT and RBM+PP
ls2 = norm_la(jnp.asarray(np.load(os.path.join(RF, 'la_star_s2.npy'))))
z = np.load(SPEC['A']); bases = {'vit': norm_la(lab.lP0), 'rbmpp': norm_la(jnp.asarray(z['lP'], jnp.float64))}; del z
res['c'] = {}
for nm, lb in bases.items():
    _, V, W = lab.VW(lb, s2); v = sec.vec(lb, s2); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s2) ** 2 * (ELb - Eb) ** 2)))
    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    hist = []
    def obj(a):
        e_ = efn(la_c(a), s2); hist.append((float(a), e_)); log(f'  (c) {nm} alpha {a:.5f}: E_FN {e_:.4e}'); return e_
    r_ = minimize_scalar(obj, bounds=(-3.0 / sdE, 3.0 / sdE), method='bounded', options=dict(maxiter=12, xatol=1e-3 / sdE))
    ac = float(r_.x); lc = norm_la(la_c(ac)); Xc = jnp.exp(ls2 - lc) < 0.1
    res['c'][nm] = dict(alpha=ac, EFN=min(h[1] for h in hist), EFN_veto=efn(jnp.where(Xc, lc - 40.0, lc), s2), scan=hist)
    log(f'(c) {nm}', res['c'][nm]['alpha'], res['c'][nm]['EFN'], res['c'][nm]['EFN_veto']); dump(F, res)
    del ELb
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
