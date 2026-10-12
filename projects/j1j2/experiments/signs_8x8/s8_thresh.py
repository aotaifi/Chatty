"""Step 1 (CPU, label-free): energy-optimal Krylov threshold T on the ViT's own sign at 8x8, from the existing two-hop
referee data of experiments/loop_8x8 (l8_ref.py; x ~ |psi_P|^2, all valid neighbours y with the guide's one-hop data).

One Krylov sign step:  s1(x) = s(x) sgn[T - r(x)],  r(x) = (H a s)(x) / (a(x) s(x)) = E_L of the guide (a, s) = x_ELa.
Local energy of (a, s1) at x:  E1(x; T) = D_x + sum_y c_xy q_x q_y,  c_xy = J_b/2 s_x s_y a_y/a_x,  q = sgn(T - r).
E(T) = mean_x E1(x; T) is piecewise constant in T; T = +inf is the ViT sign.
Reports per-batch and pooled optimal T, half-sample (by chain) T stability, and a cross-fitted (T chosen on the other
batches) paired gain E(T) - E(+inf) with SE from per-chain means.
  python s8_thresh.py OUT.json ref_a.npz ref_b.npz ...
"""
import sys, json
import numpy as np

L = 8; N = 64


def bond_J():
    nn_, nnn = [], []
    q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn_ += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return np.array([1.0] * len(nn_) + [0.5] * len(nnn))


JB = bond_J()


def load_edges(path, tag):
    d = np.load(path)
    own = d['own']; b = d['bond'].astype(int)
    c = 0.5 * JB[b] * d['x_s'][own] * d['y_s'] * np.exp(d['y_la'] - d['x_la'][own])
    return dict(n=len(d['X']), own=own, c=c, rx=d['x_ELa'], ry=d['y_ELa'], D=d['x_D'], ELa=d['x_ELa'],
                chain=d['chain'] + 100000 * tag, X=d['X'])


def merge(Es):
    out = dict(n=0, own=[], c=[], rx=[], ry=[], D=[], ELa=[], chain=[], X=[])
    for E in Es:
        out['own'].append(E['own'] + out['n']); out['n'] += E['n']
        for k in ('c', 'rx', 'ry', 'D', 'ELa', 'chain', 'X'): out[k].append(E[k])
    for k in ('own', 'c', 'rx', 'ry', 'D', 'ELa', 'chain', 'X'): out[k] = np.concatenate(out[k])
    out['rxe'] = out['rx'][out['own']]
    return out


def curve(E, mask=None):
    """E(T) per site on the candidate grid (midpoints of all r values), restricted to samples in mask."""
    m = np.ones(E['n'], bool) if mask is None else mask
    ke = m[E['own']]; n = m.sum()
    c = E['c'][ke] / n; rx = E['rxe'][ke]; ry = E['ry'][ke]
    base = E['ELa'][m].sum() / n
    rv = np.unique(np.concatenate([rx, ry]))
    cand = np.r_[rv[0] - 1.0, 0.5 * (rv[:-1] + rv[1:]), rv[-1] + 1.0]
    lo = np.minimum(rx, ry); hi = np.maximum(rx, ry)
    a = np.searchsorted(cand, lo, 'left'); b = np.searchsorted(cand, hi, 'left')
    diff = np.zeros(len(cand) + 1)
    np.add.at(diff, a, -2 * c); np.add.at(diff, b, +2 * c)
    cu = base + np.cumsum(diff)[:len(cand)]
    return base / N, cand, cu / N


def eloc(E, T):
    qx = np.where(E['rx'] <= T, 1., -1.); qy = np.where(E['ry'] <= T, 1., -1.)
    e = E['D'].copy(); np.add.at(e, E['own'], E['c'] * qx[E['own']] * qy)
    return e


def chain_mean_se(v, ch):
    u, inv = np.unique(ch, return_inverse=True)
    m = np.bincount(inv, weights=v) / np.bincount(inv)
    return float(v.mean()), float(m.std(ddof=1) / np.sqrt(len(u)))


def opt_T(E, mask=None):
    b, cand, cu = curve(E, mask); j = int(np.argmin(cu))
    return float(cand[j]), float(cu[j] - b), b, cand, cu


if __name__ == '__main__':
    out = sys.argv[1]; files = sys.argv[2:]
    Es = [load_edges(f, i) for i, f in enumerate(files)]
    E = merge(Es); res = dict(files=files, n=E['n'])
    T, g, b, cand, cu = opt_T(E)
    res['pooled'] = dict(T=T, gain_in_sample=g, E_vit=b)
    # flip fraction at T on a^2 samples
    res['pooled']['flip_frac_x'] = float(np.mean(E['rx'] > T))
    grid = np.linspace(-40, -20, 401)
    res['curve_grid'] = grid.tolist(); res['curve'] = (np.interp(grid, cand, cu) - b).tolist()
    # per batch
    res['per_batch'] = []
    for i, Ei in enumerate(Es):
        Ei = merge([Ei]); Ti, gi, bi, _, _ = opt_T(Ei)
        res['per_batch'].append(dict(T=Ti, gain_in_sample=gi, E_vit=bi))
    # half-sample (by chain) splits, 8 random splits
    rg = np.random.default_rng(0); uc = np.unique(E['chain']); halves = []
    for k in range(8):
        sel = set(rg.choice(uc, len(uc) // 2, replace=False).tolist())
        m = np.array([c in sel for c in E['chain']])
        Ta = opt_T(E, m)[0]; Tb = opt_T(E, ~m)[0]
        # cross-fitted gain: T from one half, evaluated on the other
        ea = eloc(E, Tb); eb = eloc(E, Ta)
        dx = np.where(m, ea, eb) - E['ELa']
        halves.append(dict(T_a=Ta, T_b=Tb, xfit_gain=chain_mean_se(dx / N, E['chain'])))
    res['half_splits'] = halves
    # leave-one-batch-out cross-fit
    lobo = []; dx = np.zeros(E['n']); off = 0
    for i, Ei in enumerate(Es):
        rest = merge([Ej for j, Ej in enumerate(Es) if j != i]); Ti = opt_T(rest)[0]
        Ei_ = merge([Ei]); d = (eloc(Ei_, Ti) - Ei_['ELa']) / N
        dx[off:off + Ei['n']] = d; off += Ei['n']
        lobo.append(dict(T=Ti, gain=chain_mean_se(d, Ei_['chain'])))
    res['lobo'] = lobo; res['lobo_pooled_gain'] = chain_mean_se(dx, E['chain'])
    # gain at fixed T on a grid around T (stability of the curve)
    res['gain_vs_T'] = {f'{t:.2f}': chain_mean_se((eloc(E, t) - E['ELa']) / N, E['chain']) for t in
                        np.r_[T - 3, T - 2, T - 1, T - 0.5, T, T + 0.5, T + 1, T + 2, T + 3]}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('curve', 'curve_grid')}, indent=1))
