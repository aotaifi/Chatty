#!/usr/bin/env python3
"""Stage A (CPU, lattice_symmetries env): exact targets and training sets for the sign-learnability study.

For one cluster N (symmetric sector, exact ED amplitude a = |psi0|):
  * ED ground state v0 (orbit basis; p0_r = v0_r^2 = |psi0|^2 weight of the whole orbit r).
  * Exact Krylov chain from Marshall with the exact amplitude (energy-optimal grouped threshold, as in
    sign_design_tests/b_depth_sym.py): s_k = s_{k-1} c_k, c_k = sgn[T_{k-1} - r_{k-1}], k = 1..kmax.
    Stored per k: c_k (oriented so that the p0-majority is +1 = "keep"), s_k, r_{k-1}, T_{k-1}.
  * GS sign relative to Marshall: sigma0 = sgn(v0) * Marshall (p0-majority +1).
  * Training sets: i.i.d. orbit samples from the tempered distribution |psi0(x)|^(2 beta) over configurations,
    i.e. orbit weight q_r ~ n_r^(1-beta) p0_r^beta, for beta in --betas and seeds; nested budgets n:
    smp_n = unique(samples[:n]),  nbr_n = unique(samples[:n] + all one-hop (H-connected) orbits).
Outputs in <out>/N<N>/: *.npy arrays, train_b<beta>_s<seed>.npz, meta.json.
"""
import argparse, json, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sl_common as SC
import closed_fn_krylov_sym6x6 as C
import ed_common as ec
import torus_cluster as tc

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, required=True)
ap.add_argument('--kmax', type=int, default=4)
ap.add_argument('--betas', type=float, nargs='+', default=[1.0, 0.5, 0.3])
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
ap.add_argument('--budgets', type=int, nargs='+', default=[100, 300, 1000, 3000, 10000, 30000, 100000])
ap.add_argument('--cache-dir', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)
t00 = time.time()
cfg = SC.CLUSTERS[args.N]
cluster = None
if cfg['torus'] is not None:
    t = cfg['torus']; cluster = tc.Cluster((t[0], t[1]), (t[2], t[3])); ctag = 'T' + '_'.join(map(str, t)).replace('-', 'm')
    L = 0
else:
    L = cfg['L']; ctag = f'L{L}'
    cl = tc.Cluster((L, 0), (0, L))     # geometry used by the learner must match ed_common's conventions
    assert sorted(map(tuple, cl.nn)) == sorted(map(tuple, ec.bonds(L)[0])) and cl.mask == ec.sublattice_mask(L)
    assert sorted(map(tuple, cl.nnn)) == sorted(map(tuple, ec.bonds(L)[1]))
od = SC.data_dir(args.out, args.N); os.makedirs(od, exist_ok=True)
S = C.Sym(L, 0.5, log=log, cache=os.path.join(args.cache_dir, f'orb_{ctag}.npz'), cluster=cluster)
gsa = types.SimpleNamespace(gs_vec=None, gs_states=None, ed_tol=1e-12)
if args.N == 36:
    gsa.gs_vec = f'{SC.ED36}/vectors_k00_A1_p.npy'; gsa.gs_states = f'{SC.ED36}/states_k00_A1_p.npy'
E0, v0 = C.target_gs(S, gsa, log)
a = np.abs(v0); p0 = v0 * v0
sg0 = np.where(v0 >= 0, 1, -1).astype(np.int8)
M = S.marshall.astype(np.int8)
sig0 = (sg0 * M).astype(np.int8)
if np.sum(p0 * sig0) < 0: sig0 = -sig0


def energy(s):
    psi = a * s; return float(psi @ S.H(psi))


def w_s(s):
    return max(0.0, (1 - abs(float(np.sum(p0 * s * sg0)))) / 2)


meta = dict(N=args.N, cluster=ctag, D=int(S.D), E0=E0, E0_per_site=E0 / args.N, nnz=int(S.nnz),
            n_conf=float(S.n_orb.sum()), steps=[])
np.save(f'{od}/states.npy', S.states.astype(np.uint64))
np.save(f'{od}/n_orb.npy', S.n_orb.astype(np.float32))
np.save(f'{od}/p0.npy', p0)
np.save(f'{od}/sg0.npy', sg0); np.save(f'{od}/marshall.npy', M); np.save(f'{od}/sigma0.npy', sig0)
log('sigma0 flip weight (Marshall error)', float(np.sum(p0[sig0 < 0])), 'w_s(M)', w_s(M))
s = C.canonical(M.copy())
np.save(f'{od}/s_0.npy', s)
meta['steps'].append(dict(k=0, E=energy(s), eps=(energy(s) - E0) / abs(E0), w_s=w_s(s)))
for k in range(1, args.kmax + 1):
    t0 = time.time()
    sn, emin, G, r = S.krylov(a, s)
    c = (sn * s).astype(np.int8)
    if np.sum(p0 * c) < 0: c = -c
    keep, flip = r[c > 0], r[c < 0]
    if len(flip) == 0 or len(keep) == 0:
        T = float('inf') if len(flip) == 0 else float('-inf')
    else:
        assert keep.max() < flip.min(), (keep.max(), flip.min())
        T = 0.5 * (keep.max() + flip.min())
    E = energy(sn)
    rec = dict(k=k, T_prev=T, E=E, eps=(E - E0) / abs(E0), w_s=w_s(sn), groups=int(G),
               flip_weight_p0=float(np.sum(p0[c < 0])), flip_frac_orbits=float(np.mean(c < 0)),
               gap_keep_flip=float(flip.min() - keep.max()) if len(flip) and len(keep) else None,
               sec=time.time() - t0)
    meta['steps'].append(rec); log('STEP', json.dumps(rec))
    np.save(f'{od}/r_{k-1}.npy', r); np.save(f'{od}/c_{k}.npy', c); np.save(f'{od}/s_{k}.npy', sn)
    s = sn
# tempered training sets
ind, ptr = S.indices, S.indptr
nmax = max(args.budgets)
meta['train'] = {}
for beta in args.betas:
    lq = (1 - beta) * np.log(S.n_orb) + beta * np.log(np.maximum(p0, 1e-300))
    q = np.exp(lq - lq.max()); q /= q.sum()
    for seed in args.seeds:
        rg = np.random.default_rng(1000 * seed + int(round(100 * beta)))
        smp = rg.choice(S.D, size=nmax, p=q).astype(np.int64)
        out = dict(samples=smp.astype(np.int32))
        info = {}
        for n in args.budgets:
            u = np.unique(smp[:n])
            nb = np.concatenate([ind[ptr[i]:ptr[i + 1]] for i in u])
            un = np.unique(np.concatenate([u, nb.astype(np.int64)]))
            out[f'smp_{n}'] = u.astype(np.int32); out[f'nbr_{n}'] = un.astype(np.int32)
            info[str(n)] = dict(n_smp=int(len(u)), n_nbr=int(len(un)), p0_smp=float(p0[u].sum()), p0_nbr=float(p0[un].sum()),
                                frac_orbits_nbr=float(len(un) / S.D))
        np.savez(f'{od}/train_b{beta}_s{seed}.npz', **out)
        meta['train'][f'b{beta}_s{seed}'] = info
        log('TRAIN', beta, seed, json.dumps(info))
meta['sec'] = time.time() - t00
json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
log('DONE', time.time() - t00)
