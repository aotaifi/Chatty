#!/usr/bin/env python3
"""(c) 6x6 J1-J2 (J2/J1=0.5), amplitude a = |psi_ViT| (fp32, matmul precision 'highest', NOT TF32), one Krylov step from Marshall
with decision variable z:   s'(x) = M(x) sgn[T - z(x)]
   r      plain local energy (H a M)/(a M)
   delta  r - r_FN, r_FN = (H_FN[a,M] a)/a, H_FN built as specified (K<0 kept, K>0 -> diagonal with weight a_y/a_x)
          (algebraically identical to r; recorded max|delta| ~ 1e-15 -> z := 0, i.e. 'no flip' = Marshall)
   dV     r - r_kept = V_sf(x) = sum_{y: K_xy>0} K_xy a_y/a_x   (the fixed-node sign-flip potential)
T is label-free and energy-optimal: argmin_T of the defect-attributed Delta E(T)=E[a s'(T)] - E_ViT estimated on an
independent training set A ~ |a|^2 (bond-exchange chains).  Evaluation: (i) w_s on the exact |psi0|^2 samples (ED labels only
for scoring), (ii) Delta E per site on an independent set B ~ |a|^2 with the defect-attributed estimator
(experiments/fn_bound_check_6x6/guide_vmc_sym.py: edges with d_x != d_y attributed to the larger-amplitude endpoint,
d = s' h, h = binarised ViT sign), SE by chain blocks.  Also the reference threshold T_FN = -14.9857997791 for plain r.
Machinery: ll6_core.py (chatty_ll6) levels / Net / Chains.
"""
import argparse, json, time, sys
import numpy as np
import ll6_core as C

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--nchA', type=int, default=512); ap.add_argument('--roundsA', type=int, default=16)
ap.add_argument('--nchB', type=int, default=1024); ap.add_argument('--roundsB', type=int, default=32)
ap.add_argument('--between', type=int, default=4); ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--ned', type=int, default=400000)
ap.add_argument('--chunk', type=int, default=32)
ap.add_argument('--dtype', default='float32')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()
T0 = time.time()
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype=args.dtype); flat = net.flat0
C.log('Net', args.dtype, 'precision', C.jax.config.jax_default_matmul_precision)
VARS = ('r', 'delta', 'dV')
MRB, JB = C.MRB, C.JB                                  # MRB=-1 NN (K<0), +1 NNN (K>0) for Marshall
T_FN = C.T_FN


def node_vars(la1, nb1, Vm1):
    """variables on states of level l given log a on level l+1: la1 = log a on those states, idx etc."""
    pass


def level_vars(la_up, la_self, idx, V):
    w = 0.5 * JB[None, :] * np.exp(la_up[idx] - la_self[:, None])      # J/2 a_y/a_x
    dg = C.diag_vec(V)
    r = dg + np.sum(np.where(V, MRB[None, :] * w, 0.), axis=1)         # (H a M)/(a M): M_x M_y = MRB
    viol = V & (MRB[None, :] > 0); allowed = V & (MRB[None, :] < 0)
    dfn = dg + np.sum(np.where(viol, w, 0.), axis=1)                   # H_FN diagonal
    rfn = dfn - np.sum(np.where(allowed, w, 0.), axis=1)               # (H_FN a)/a, kept hops -J/2 a_y/a_x
    dV = np.sum(np.where(viol, w, 0.), axis=1)
    return dict(r=r, delta=r - rfn, dV=dV)


def zfix(v):
    v = dict(v); d = v['delta']; v['delta'] = np.where(np.abs(d) < 1e-9, 0.0, d); return v


# ---------------------------------------------------------------- sample sets with 2-shell edge data
def edge_data(X):
    """per sample x: variables at x and at each valid neighbour y, rat=a_y/a_x, J, mr, imag(log psi) at x,y."""
    lv, nb = C.build_levels(X, 2)
    la2 = net.logabs(flat, lv[2])
    idx1, V1, self1 = nb[1]
    la1 = la2[self1]
    vv = level_vars(la2, la1, idx1, V1)                                 # variables on every level-1 state
    z1 = net.logpsi_c(flat, lv[1]).imag
    idx0, V0, self0 = nb[0]
    p0 = np.searchsorted(lv[0], X); idx0 = idx0[p0]; V0 = V0[p0]; sx = self0[p0]
    own, col = np.nonzero(V0); y = idx0[own, col]
    out = dict(xi=own, J=JB[col], mr=MRB[col], rat=np.exp(la1[y] - la1[sx[own]]), imx=z1[sx][own], imy=z1[y], imnode=z1[sx])
    for k, v in vv.items(): out[k + '_x'] = v[sx][own]; out[k + '_y'] = v[y]; out[k + '_node'] = v[sx]
    out['n'] = len(X)
    return out


def merge(parts):
    off = 0; out = {}
    for p in parts:
        for k, v in p.items():
            if k == 'n': continue
            out.setdefault(k, []).append(v + off if k == 'xi' else v)
        off += p['n']
    o = {k: np.concatenate(v) for k, v in out.items()}; o['n'] = off
    return o


def node_data(X, chunk=4096):
    """variables at x only (level-1 data): used for the ED samples."""
    outs = {k: [] for k in VARS}
    for i in range(0, len(X), chunk):
        Xc = np.asarray(X[i:i + chunk], np.uint64)
        lv, nb = C.build_levels(Xc, 1)
        la1 = net.logabs(flat, lv[1])
        idx0, V0, self0 = nb[0]; p0 = np.searchsorted(lv[0], Xc)
        vv = zfix(level_vars(la1, la1[self0][p0], idx0[p0], V0[p0]))
        for k in VARS: outs[k].append(vv[k])
    return {k: np.concatenate(v) for k, v in outs.items()}


def sample_set(nch, rounds, seed):
    ch = C.Chains(net, nch, seed); ch.advance(flat, args.burn)
    return np.concatenate([ch.advance(flat, args.between) for _ in range(rounds)])


def collect_edges(X):
    parts = []
    for i in range(0, len(X), args.chunk):
        parts.append(edge_data(X[i:i + args.chunk]))
        if (i // args.chunk) % 50 == 0: C.log('edges', i, len(X), 'evals', net.neval)
    E = merge(parts)
    for k in VARS:
        if k == 'delta':
            mx = float(np.max(np.abs(E['delta_x']))); E['max_abs_delta'] = mx
            for s in ('_x', '_y', '_node'): E['delta' + s] = np.where(np.abs(E['delta' + s]) < 1e-9, 0.0, E['delta' + s])
    return E


def phase_phi(E):
    return 0.5 * float(np.angle(np.mean(np.exp(2j * E['imnode']))))


def dE_edges(E, phi, var, T):
    """per-sample defect-attributed Delta E = E[a s'] - E[psi_ViT] (total energy units)."""
    q = lambda z: np.where(z <= T, 1.0, -1.0)
    hx = np.where(np.cos(E['imx'] - phi) >= 0, 1., -1.); hy = np.where(np.cos(E['imy'] - phi) >= 0, 1., -1.)
    defect = (q(E[var + '_x']) * q(E[var + '_y']) * E['mr'] * hx * hy) < 0          # s'_x s'_y h_x h_y = -1
    sel = defect & (E['rat'] < 1.0)
    return np.bincount(E['xi'][sel], weights=(-2.0 * E['J'] * E['rat'] * hx * hy)[sel], minlength=E['n'])


def chain_mean_se(v, nch):
    cm = np.asarray(v, float).reshape(-1, nch).mean(0)
    return float(cm.mean()), float(cm.std(ddof=1) / np.sqrt(nch))


# ---------------------------------------------------------------- run
XA = sample_set(args.nchA, args.roundsA, 1000 + args.seed); C.log('set A', len(XA))
EA = collect_edges(XA); C.log('A edges', len(EA['xi']), 'max|delta|', EA['max_abs_delta'], 'sec', time.time() - T0)
XB = sample_set(args.nchB, args.roundsB, 2000 + args.seed); C.log('set B', len(XB))
EB = collect_edges(XB); C.log('B edges', len(EB['xi']), 'sec', time.time() - T0)
phiA = phase_phi(EA); phiB = phase_phi(EB)
leak = float(np.mean(np.sin(EB['imnode'] - phiB) ** 2)); C.log('phi', phiA, phiB, 'leak', leak)

Sed = np.load('samples_psi0sq_6x6.npz'); Sx = np.load('samples_extra_psi0sq_6x6.npz')
Xed = np.concatenate([Sed['x'], Sx['x']])[:args.ned].astype(np.uint64)
psi_ed = np.concatenate([Sed['psi'], Sx['psi']])[:args.ned]
M_ed = C.marshall_vec(Xed)
assert np.all(M_ed == np.concatenate([Sed['marshall'], Sx['marshall']])[:args.ned])
Zed = node_data(Xed); C.log('ED node data', len(Xed), 'sec', time.time() - T0)
sgn_ed = np.where(psi_ed >= 0, 1., -1.)


def ws_ed(var, T):
    sp = M_ed * np.where(Zed[var] <= T, 1., -1.)
    nw = int(np.sum(sp != sgn_ed)); n = len(sp); k = min(nw, n - nw)   # global sign: majority orientation
    return k / n, np.sqrt(max(k, 1)) / n


res = dict(args=vars(args), phi=dict(A=phiA, B=phiB), phase_leak=leak, max_abs_delta_A=EA['max_abs_delta'], max_abs_delta_B=EB['max_abs_delta'],
           n_A=len(XA), n_B=len(XB), n_ed=len(Xed), neval=net.neval, variables={})
for var in VARS:
    zA = EA[var + '_node']
    grid = np.unique(np.r_[np.quantile(zA, np.linspace(0.5, 1.0, 241)), np.quantile(zA, np.linspace(0.0, 0.5, 21))])
    grid = np.r_[grid, grid[-1] + 1.0] if len(grid) > 1 else np.r_[grid, grid + 1.0]
    dEA = np.array([dE_edges(EA, phiA, var, T).mean() for T in grid])
    ib = int(np.argmin(dEA)); T_opt = float(grid[ib]); noflip = ib == len(grid) - 1
    rec = dict(T_opt=T_opt, noflip=bool(noflip), dE_train_site_at_Topt=float(dEA[ib] / 36), flip_frac_train=float(np.mean(zA > T_opt)))
    for lab, T in (('Topt', T_opt),) + ((('TFN', T_FN),) if var == 'r' else ()):
        dB = dE_edges(EB, phiB, var, T); m, se = chain_mean_se(dB, args.nchB)
        w, werr = ws_ed(var, T)
        rec[lab] = dict(T=T, dE_site=m / 36, dE_site_se=se / 36, w_s=w, w_s_err=werr, flip_frac_B=float(np.mean(EB[var + '_node'] > T)),
                        flip_frac_ED=float(np.mean(Zed[var] > T)))
    # ranking quality: best w_s over all T on ED samples (labels, diagnostic)
    z = Zed[var]; order = np.argsort(z, kind='mergesort'); agree = (M_ed * sgn_ed)[order] > 0     # M agrees with truth
    kept_wrong = np.r_[0, np.cumsum(~agree)]; flipped_wrong = np.r_[np.cumsum(agree[::-1])[::-1], 0]
    W = (kept_wrong + flipped_wrong) / len(z); res_best = float(np.min(np.minimum(W, 1 - W)))
    rec['w_s_bestT_ED'] = res_best
    res['variables'][var] = rec
    np.savez_compressed(args.out.replace('.json', f'_curve_{var}.npz'), grid=grid, dE_train=dEA)
    C.log('VAR', var, json.dumps(rec))
res['sec'] = time.time() - T0
json.dump(res, open(args.out, 'w'), indent=1)
C.log('DONE', args.out, 'evals', net.neval, 'sec', time.time() - T0)
