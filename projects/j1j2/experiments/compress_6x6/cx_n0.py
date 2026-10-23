"""Test N0 (README compress_6x6 "Test N0"): is the tail the near-node cancellation region?  No training.
  python cx_n0.py OUT SPEC.json     SPEC: refresh (s2.npy), target_la (phi_1), a2_table (O2b round_1/la_table.npy), nbonds (2^20), seed (7777), dry
Bonds: w-sampled violating bonds (x, y) of the training orbits, w_xy = n_x phi_x phi_y |H_xy| (s_2(x) s_2(y) > 0): VSampler / vbond_of / wx_chunk are
  copied from cx_f2fit.py (commit ecbb0f6; the F2-1 lv_diag drew its 2^20 bonds with seed 7777 as here).  d = (a2_y - a2_x) - (log phi_y - log phi_x),
  a2 = the saved O2b round-1 table la_table.npy (no network evaluation), d^2 per bond.
Cancellation ratio per orbit x: c(x) = |sum_y H_xy phi(y)| / sum_y |H_xy phi(y)| over the antiparallel J1+J2 neighbours (config space, y -> orbit rep, one term per
  bond, H_xy = JB/2 > 0), phi = s_2 exp(log phi_1); computed as phi(y)/phi(x) (clip 60).  c depends on the orbit only: evaluated once per unique endpoint orbit.
Sets: top 0.1% and top 1% of the sample by d^2; the reference = the whole w-sampled sample (2^20 bonds, the top sets are subsets of it: effect 0.1-1% of the sample).
Controls (before any number is used): (A) brute-force c on 40 orbits by plain python (288 space-group images + spin flip, independent canon), rel 1e-10;
  (B) UNSIGNED phi (s = +1): c must be 1 on every one of them (so the check can fail: the signed c differs from 1 on a large share);
  (C) 1500 sampled bonds re-checked by brute force: bond is antiparallel, canon(y) = iy, s2 product > 0, J1 flag equals the bond type.
Marshall: a J1 bond (x, y') is a Marshall-sign violation iff s_2(x) s_2(y') > 0 (a J1 flip changes the A-sublattice up-count by 1, so Marshall alternates).
"""
import os, sys, time
import numpy as np
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_n0.json'); T00 = time.time(); DRY = bool(SPEC.get('dry', False))
NBONDS = int(SPEC.get('nbonds', 2 ** 20)); SEED = int(SPEC.get('seed', 7777))
res = dict(spec=SPEC, done=False)


def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')


def stop(msg, code):
    res['stop'] = msg; res['done'] = True; res['sec'] = time.time() - T00; dump(F, res); progress(msg); sys.exit(code)


lab = Lab(); sec = lab.sec
NNN1 = len(SS.NN)                                                   # bonds 0..NNN1-1 are J1


def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))


ls2 = norm_la(jnp.asarray(np.load(SPEC['target_la']), jnp.float64))
s2 = jnp.asarray(np.load(os.path.join(SPEC['refresh'], 's2.npy')).astype(np.float32))
la_a2 = jnp.asarray(np.load(SPEC['a2_table']), jnp.float64)
assert la_a2.shape == ls2.shape == (sec.D,), (la_a2.shape, ls2.shape, sec.D)
ho_h = np.asarray(lab.HO); ls2_h = np.asarray(ls2); s2_h = np.asarray(s2); a2_h = np.asarray(la_a2)
progress(f'ready {time.time() - T00:.0f}s  D={sec.D}  a2 finite {bool(np.isfinite(a2_h).all())}  sha target/a2/s2 given in run_metadata')

# ============================================================================================ copied from cx_f2fit.py (vbond_of returns also the bond index)
VCH = 2048


@jax.jit
def wx_chunk(i0):
    ii = jnp.minimum(i0 + jnp.arange(VCH), sec.D - 1); real = (i0 + jnp.arange(VCH)) < sec.D
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(VCH, NB)
    bd = vv & (s2[ii][:, None] * s2[jj] > 0) & real[:, None]
    w = jnp.where(bd, 0.5 * JB[None, :] * jnp.exp(ls2[ii][:, None] + ls2[jj]) * sec.n[ii][:, None], 0.0)
    return jnp.sum(w, 1)


@jax.jit
def vbond_of(idx, key):
    x = sec.reps[idx]; v = valid_bonds(x)
    y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    bad = v & (s2[idx][:, None] * s2[iy] > 0)
    lw = jnp.where(bad, jnp.log(0.5 * JB[None, :]) + jnp.clip(ls2[iy] - ls2[idx][:, None], -60, 60), -jnp.inf)
    has = jnp.any(bad, 1)
    sel = jax.random.categorical(key, jnp.where(has[:, None], lw, 0.0), axis=1, shape=(1, x.shape[0])).T
    return x, jnp.take_along_axis(y, sel, 1)[:, 0], jnp.take_along_axis(iy, sel, 1)[:, 0], has, sel[:, 0]


class VSampler:
    def __init__(self, Wx, eligible, seed):
        self.rng = np.random.default_rng(seed); self.key = jax.random.PRNGKey(seed)
        self.idx = np.nonzero(eligible & (Wx > 0))[0]
        cdf = np.cumsum(Wx[self.idx]); self.tot = float(cdf[-1]); self.cdf = cdf / cdf[-1]

    def draw(self, M):
        x_idx = self.idx[np.minimum(np.searchsorted(self.cdf, self.rng.random(M)), self.idx.size - 1)]
        parts = []
        for i in range(0, M, 2052):
            self.key, k_ = jax.random.split(self.key)
            parts.append(vbond_of(jnp.asarray(x_idx[i:i + 2052]), k_))
        x, y, iy, has, sel = (jnp.concatenate([q[j] for q in parts]) for j in range(5))
        assert bool(jnp.all(has)), 'sampled an orbit without a violating bond'
        return dict(ix=x_idx, x=np.asarray(x), y=np.asarray(y), iy=np.asarray(iy), sel=np.asarray(sel), M=M)


# ============================================================================================ cancellation ratio
@jax.jit
def c_chunk(ii, signed):
    """ii (VCH,) orbit indices.  returns c, number of violating J1 bonds, number of violating bonds, sum|H phi| (relative to phi(x)).
    signed False: phi -> |phi| (the deliberately unsigned control; c must be 1)."""
    xx = sec.reps[ii]; vv = valid_bonds(xx)
    yy = jnp.where(vv, xx[:, None] ^ MASKS[None, :], xx[:, None])
    jj = SS.canon(sec.T, sec.reps, yy.reshape(-1)).reshape(ii.shape[0], NB)
    sg = jnp.where(signed, s2[jj].astype(f64), 1.0)
    amp = sg * jnp.exp(jnp.clip(ls2[jj] - ls2[ii][:, None], -60, 60))
    t = jnp.where(vv, 0.5 * JB[None, :] * amp, 0.0)
    den = jnp.sum(jnp.abs(t), 1); c = jnp.abs(jnp.sum(t, 1)) / den
    viol = vv & (s2[ii][:, None] * s2[jj] > 0)
    return c, jnp.sum(viol[:, :NNN1], 1), jnp.sum(viol, 1), den


def c_of(orb, signed=True):
    orb = np.asarray(orb, np.int64); out = [[], [], [], []]
    for i in range(0, orb.size, VCH):
        blk = orb[i:i + VCH]; m = blk.size
        pad = np.concatenate([blk, np.full(VCH - m, blk[0], np.int64)]) if m < VCH else blk
        r = c_chunk(jnp.asarray(pad), signed)
        for k in range(4): out[k].append(np.asarray(r[k])[:m])
    return [np.concatenate(o) for o in out]


# ============================================================================================ brute force (plain python, as cx_f2_lvcheck.py)
perms = SS.space_group(); bl = list(SS.NN) + list(SS.NNN); Jb = [1.0] * len(SS.NN) + [SS.J2] * len(SS.NNN)
reps_h = np.asarray(sec.reps)
POW = (np.int64(1) << np.arange(36, dtype=np.int64)); FULL_ = (1 << 36) - 1
AR288 = np.arange(perms.shape[0])[:, None]


def canon_bf(c):
    b = np.array([(c >> i) & 1 for i in range(36)], np.int64)
    img = np.zeros((perms.shape[0], 36), np.int64); img[AR288, perms] = b[None, :]
    v = img @ POW; m = int(min(v.min(), (FULL_ - v).min()))
    i = int(np.searchsorted(reps_h, np.uint64(m)))
    assert reps_h[i] == np.uint64(m), 'brute-force orbit representative is not in the table'
    return i


def c_bf(ix, signed=True):
    x = int(reps_h[ix]); num = 0.0; den = 0.0; nv1 = 0; nv = 0
    for (i, j), J in zip(bl, Jb):
        if ((x >> i) & 1) == ((x >> j) & 1): continue
        iy = canon_bf(x ^ (1 << i) ^ (1 << j))
        t = 0.5 * J * (float(s2_h[iy]) if signed else 1.0) * float(np.exp(np.clip(ls2_h[iy] - ls2_h[ix], -60, 60)))
        num += t; den += abs(t)
        if float(s2_h[ix]) * float(s2_h[iy]) > 0:
            nv += 1; nv1 += (J == 1.0)
    return abs(num) / den, nv1, nv


# ============================================================================================ W_x and the sample
t_w = time.time(); NW = int(SPEC.get('wx_limit', sec.D)) if DRY else sec.D
Wx = np.zeros(sec.D)
Wx[:NW] = np.concatenate([np.asarray(wx_chunk(i0)) for i0 in range(0, NW, VCH)])[:NW]
progress(f'W_x over {NW} orbits {time.time() - t_w:.0f}s total weight {Wx.sum():.6e}; orbits with a violating bond {int((Wx > 0).sum())}')
vb = VSampler(Wx, ~ho_h, SEED).draw(NBONDS)
ix, iy = vb['ix'], vb['iy']
d = (a2_h[iy] - a2_h[ix]) - (ls2_h[iy] - ls2_h[ix]); d2 = d * d
bond_is_J1 = vb['sel'] < NNN1
srt = np.sort(d2)[::-1]; cs = np.cumsum(srt); tot = cs[-1]
res['sample'] = dict(n_bonds=int(d2.size), mean_d2=float(tot / d2.size), share_top_0p1pct=float(cs[max(int(.001 * d2.size), 1) - 1] / tot),
                     share_top_1pct=float(cs[max(int(.01 * d2.size), 1) - 1] / tot), n_eff_frac=float(tot ** 2 / np.sum(d2 ** 2) / d2.size),
                     frac_bonds_J1=float(bond_is_J1.mean()), unique_x=int(np.unique(ix).size), unique_orbits=int(np.unique(np.concatenate([ix, iy])).size))
progress('sample ' + json.dumps(res['sample']))
dump(F, res)

# ============================================================================================ c per unique endpoint orbit
t_c = time.time(); U, inv = np.unique(np.concatenate([ix, iy]), return_inverse=True)
cU, v1U, vU, denU = c_of(U); progress(f'c on {U.size} unique orbits {time.time() - t_c:.0f}s; finite {bool(np.isfinite(cU).all())}')
if not np.isfinite(cU).all(): stop('c not finite', 2)
cx = cU[inv[:ix.size]]; cy = cU[inv[ix.size:]]; cmin = np.minimum(cx, cy)
v1x = v1U[inv[:ix.size]] > 0; v1y = v1U[inv[ix.size:]] > 0

# ============================================================================================ controls
rr = np.random.default_rng(20261023)
top_end = np.concatenate([ix[np.argsort(-d2)[:200]], iy[np.argsort(-d2)[:200]]])
chk = np.concatenate([rr.choice(top_end, 20, replace=False), rr.choice(np.concatenate([ix, iy]), 20, replace=False)]).astype(np.int64)
c_code = c_of(chk)
bf = [c_bf(int(i)) for i in chk]
cb = np.array([b[0] for b in bf]); rel = np.abs(c_code[0] - cb) / np.maximum(cb, 1e-300)
cA = dict(n=int(chk.size), max_rel_c=float(rel.max()), c_min_checked=float(cb.min()), c_median_checked=float(np.median(cb)),
          viol_counts_match=bool(all(int(c_code[1][k]) == bf[k][1] and int(c_code[2][k]) == bf[k][2] for k in range(chk.size))), ok=bool(rel.max() < 1e-10))
rnd = rr.choice(sec.D, 4096, replace=False).astype(np.int64)
cu_code = c_of(np.concatenate([chk, rnd]), signed=False)[0]
cu_bf = np.array([c_bf(int(i), signed=False)[0] for i in chk])
c_signed_rnd = c_of(rnd)[0]
cB = dict(n_orbits=int(cu_code.size), unsigned_c_max_abs_dev_from_1=float(np.max(np.abs(cu_code - 1))), unsigned_bruteforce_max_abs_dev=float(np.max(np.abs(cu_bf - 1))),
          signed_c_frac_below_0p999_random_orbits=float(np.mean(c_signed_rnd < 0.999)), signed_c_median_random_orbits=float(np.median(c_signed_rnd)),
          ok=bool(np.max(np.abs(cu_code - 1)) < 1e-12 and np.max(np.abs(cu_bf - 1)) < 1e-12 and np.mean(c_signed_rnd < 0.999) > 0.5))
sub = rr.choice(ix.size, 1500, replace=False); bad = dict(not_bond=0, canon=0, not_viol=0, j1flag=0)
SELJ = np.array([J == 1.0 for J in Jb])
for k in sub:
    xk = int(vb['x'][k]); yk = int(vb['y'][k]); dif = xk ^ yk; sites = sorted(i for i in range(36) if (dif >> i) & 1)
    if len(sites) != 2 or tuple(sites) not in set(tuple(sorted(b)) for b in bl) or ((xk >> sites[0]) & 1) == ((xk >> sites[1]) & 1): bad['not_bond'] += 1
    if canon_bf(yk) != int(iy[k]): bad['canon'] += 1
    if not (float(s2_h[ix[k]]) * float(s2_h[iy[k]]) > 0): bad['not_viol'] += 1
    isj1 = any(tuple(sorted(b)) == tuple(sites) and J == 1.0 for b, J in zip(bl, Jb))
    if isj1 != bool(bond_is_J1[k]): bad['j1flag'] += 1
cC = dict(n=int(sub.size), **bad, ok=bool(sum(bad.values()) == 0))
res['controls'] = dict(A_bruteforce_c=cA, B_unsigned=cB, C_sampled_bonds=cC); dump(F, res)
progress(f'CONTROLS A {cA}  B {cB}  C {cC}')
if not (cA['ok'] and cA['viol_counts_match'] and cB['ok'] and cC['ok']): stop('CONTROL FAILED', 2)

# ============================================================================================ sets, quantiles, shares
order = np.argsort(-d2); n1 = max(int(.001 * d2.size), 1); n2 = max(int(.01 * d2.size), 1)
sets = {'top_0.1pct': order[:n1], 'top_1pct': order[:n2], 'reference_all_w_sampled': np.arange(d2.size)}
Q = (0.1, 0.5, 0.9)
qz = lambda v: [float(np.quantile(v, q)) for q in Q]
ref_med = float(np.median(cmin)); out = {}
boot = np.random.default_rng(5)
for nm, s in sets.items():
    below = cmin[s] < ref_med
    out[nm] = dict(n=int(s.size), c_x_q10_50_90=qz(cx[s]), c_y_q10_50_90=qz(cy[s]), min_c_q10_50_90=qz(cmin[s]), share_d2_of_set=float(d2[s].sum() / tot),
                   frac_bonds_min_c_below_ref_median=float(below.mean()), share_d2_with_min_c_below_ref_median=float(d2[s][below].sum() / d2[s].sum()),
                   marshall=dict(frac_x_has_violating_J1_bond=float(v1x[s].mean()), frac_y_has_violating_J1_bond=float(v1y[s].mean()),
                                 frac_either_endpoint=float((v1x[s] | v1y[s]).mean()), frac_both_endpoints=float((v1x[s] & v1y[s]).mean()),
                                 frac_sampled_bond_itself_J1=float(bond_is_J1[s].mean())))
res['reference_median_min_c'] = ref_med; res['sets'] = out
mt = out['top_0.1pct']['min_c_q10_50_90'][1]; ratio = mt / ref_med
bs = np.array([np.median(cmin[order[:n1]][boot.integers(0, n1, n1)]) for _ in range(2000)]) / ref_med
res['decision'] = dict(median_min_c_top0p1=mt, median_min_c_ref=ref_med, ratio=float(ratio), ratio_boot95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                       branch=('NEAR-NODE (ratio <= 1/3)' if ratio <= 1 / 3 else 'DEAD (ratio >= 2/3)' if ratio >= 2 / 3 else 'BETWEEN (1/3 < ratio < 2/3)'),
                       note='reference = the whole w-sampled sample (top sets are subsets of it)')
np.savez_compressed(os.path.join(OUT, 'n0_top1pct.npz'), ix=ix[order[:n2]], iy=iy[order[:n2]], d2=d2[order[:n2]], cx=cx[order[:n2]], cy=cy[order[:n2]])
np.savez_compressed(os.path.join(OUT, 'n0_all_bonds.npz'), ix=ix, iy=iy, d2=d2, cx=cx, cy=cy, bond_is_J1=bond_is_J1)
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
progress('VERDICT DONE ' + json.dumps(res['decision']))
