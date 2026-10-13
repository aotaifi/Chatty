"""Check (a8), data passes.  For every signs_8x8 referee chunk (x ~ a^2, guide on the two-hop set), bases V (ViT sign)
and M (Marshall), rules G (threshold T on E_L) and C_w (w = 2, 3, 4; parity of connected components of marked w x w
windows): per-sample change E1(x) - E_L(x) of <H> for every threshold on a fixed grid (PREREG [A4]).
  python a8_run.py OUTDIR RUN_DIR [RUN_DIR ...]
Pass 1: statistics for the grids (E_L and max window sum over all one-hop states), reference bond energies e_bar.
Pass 2: Delta matrices and per-sample diagnostics -> OUTDIR/a8_delta.npz.
"""
import sys, os, glob, json, time
from multiprocessing import Pool
import numpy as np
import a8_lib as A

Q = A.Q
WS = (2, 3, 4); BASES = ('V', 'M')


def pass1(path):
    C = A.chunk_states(path); out = {}
    signs = C['signs']
    _, r1, _, _, _, _ = Q.onehop_from_chunk(C['S'], C['U2'], C['la'], signs)   # independent r for the check
    for b in BASES:
        e = A.bond_energies(C, b); r = e.sum(1)
        out[f'chk_{b}'] = float(np.max(np.abs(r - r1[b])))
        out[f'r_{b}'] = r.astype(np.float32)
        for w in WS:
            out[f'mx_{b}{w}'] = (e @ A.WB[w].T).max(1).astype(np.float32)
        ex = e[C['px']]
        out[f'esum_{b}'] = np.array([ex[:, A.IS_NN].sum(), ex[:, ~A.IS_NN].sum(), len(ex) * A.IS_NN.sum()])
    return out


def pass2(args):
    path, grids, ebar = args
    C = A.chunk_states(path); out = dict(chain=C['chain'], nnb=np.bincount(C['own'], minlength=len(C['S'])))
    for b in BASES:
        e = A.bond_energies(C, b); r = e.sum(1); c = A.edge_coeffs(C, b)
        out[f'rx_{b}'] = r[C['px']]
        # G
        g = grids[f'G_{b}']
        q = np.where(r[:, None] <= g[None, :], 1, -1).astype(np.int8)
        out[f'd_{b}_G'] = A.delta_matrix(C, c, q)
        out[f'nflipnb_{b}_G'] = np.add.reduceat((q[C['py']] < 0).astype(np.int16), np.searchsorted(C['own'], np.arange(len(C['S']))), axis=0)
        for w in WS:
            Sw = e @ A.WB[w].T
            vals, nc = A.comp_curves(Sw, A.ADJ[w])
            g = grids[f'C{w}_{b}']
            q, k = A.q_from_curves(vals, nc, g)
            out[f'd_{b}_C{w}'] = A.delta_matrix(C, c, q)
            out[f'ncomp_{b}_C{w}'] = k[C['px']]
            out[f'nflipnb_{b}_C{w}'] = np.add.reduceat((q[C['py']] < 0).astype(np.int16), np.searchsorted(C['own'], np.arange(len(C['S']))), axis=0)
            if w == 4:   # locality premise: share of the total excess in the best 4x4 window, at the samples
                nn, nnn = A.NBOND_W[4]
                dEW = Sw[C['px']] - (nn * ebar[b][0] + nnn * ebar[b][1])
                tot = r[C['px']] - (128 * ebar[b][0] + 128 * ebar[b][1])
                out[f'share4_{b}'] = dEW.max(1) / tot
                out[f'tot_{b}'] = tot
    return out


def make_grid(stat):
    lv = np.r_[np.linspace(0.01, 0.99, 60), 1 - 10 ** -np.linspace(2, 6.5, 180)]
    return np.r_[np.quantile(stat.astype(float), lv), np.inf]


if __name__ == '__main__':
    OUT = sys.argv[1]; runs = sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
    files, runid = [], []
    for k, r in enumerate(runs):
        f = sorted(glob.glob(os.path.join(r, 'chunk_*.npz'))); files += f; runid += [k] * len(f)
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', 6))) as P:
        p1 = P.map(pass1, files)
    meta = dict(files=[os.path.relpath(f, os.path.dirname(runs[0])) for f in files], pass1_sec=time.time() - t0)
    grids, ebar = {}, {}
    for b in BASES:
        meta[f'check_sum_e_minus_r_{b}'] = max(p[f'chk_{b}'] for p in p1)
        grids[f'G_{b}'] = make_grid(np.concatenate([p[f'r_{b}'] for p in p1]))
        for w in WS: grids[f'C{w}_{b}'] = make_grid(np.concatenate([p[f'mx_{b}{w}'] for p in p1]))
        s = sum(p[f'esum_{b}'] for p in p1)
        ebar[b] = (s[0] / s[2], s[1] / s[2])   # per-bond mean over samples (128 NN, 128 NNN bonds each)
        meta[f'ebar_{b}'] = ebar[b]
        meta[f'n_L1_states'] = int(sum(len(p[f'r_{b}']) for p in p1))
    print(json.dumps(meta | {'files': len(files)}, default=float), flush=True)
    t1 = time.time()
    with Pool(int(os.environ.get('NPROC', 6))) as P:
        p2 = P.map(pass2, [(f, grids, ebar) for f in files])
    meta['pass2_sec'] = time.time() - t1
    D = {}
    for k in p2[0]:
        D[k] = np.concatenate([p[k] for p in p2])
    D['run'] = np.concatenate([np.full(len(p['chain']), runid[i]) for i, p in enumerate(p2)])
    D['gchain'] = D['chain'] + 100000 * D['run']
    for k, v in grids.items(): D['grid_' + k] = v
    np.savez_compressed(os.path.join(OUT, 'a8_delta.npz'), **D)
    json.dump(meta, open(os.path.join(OUT, 'a8_run_meta.json'), 'w'), indent=1, default=float)
    print('done', meta, flush=True)
