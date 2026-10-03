"""Repeated current-sign Krylov (K1) steps at FIXED ViT amplitude on 6x6, J2/J1=0.5.

a(x) = |psi_ViT(x)| is never changed.  s_0 = Marshall.
    r_k(x)   = (H a s_k)(x) / (a(x) s_k(x))
    s_{k+1}  = s_k * sgn[T_k - r_k]          (convention: +1 if r_k <= T_k)
T_k is label-free and energy-optimal: it minimises the sampled variational energy of
a*s_{k+1} on an independent |a|^2 training sample (full Hamiltonian edges), using the
O(N_edges + N log N) threshold-curve accumulator of current_sign_energy_threshold8.py.
No two-means/Otsu for k>=1.

Outputs per step k:
  (i)  |psi_ViT|^2-weighted wrong-sign mass of s_k vs ViT signs on the alpha=1.2
       train/val sets of krylov_scaling_6x6_a1.20 (samples A, B; weights iw), and
  (ii) e_k - e_ViT per site at fixed amplitude on the 64-chain independent |a|^2
       sample of energy_krylov_vs_vit_6x6_indep.npz, paired, 64-chain SE.
       Both the exact full-edge local energy and the original 16-sampled-edge estimator
       (same RNG seed as energy_krylov_vs_vit_6x6_indep.py) are reported.

Exact recursion: s_k(z) needs amplitudes on the k-hop shell of z.  States are uint64 bit
strings; shells are built with vectorised numpy and deduplicated per chunk.
"""
import os, sys, math, time, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--kmax', type=int, default=3)          # report s_0..s_kmax
ap.add_argument('--ntrain-chains', type=int, default=64)
ap.add_argument('--ntrain-rounds', type=int, default=4)
ap.add_argument('--train-seed', type=int, default=202610031)
ap.add_argument('--neval', type=int, default=256)       # <=256, subset of indep sample (pilot)
ap.add_argument('--nab', type=int, default=-1)          # A/B central states for k<=2 (-1 all)
ap.add_argument('--nab-k3', type=int, default=-1)       # A/B central states for k=3 (-1 all)
ap.add_argument('--batch', type=int, default=16384)
ap.add_argument('--fp32', action='store_true')
ap.add_argument('--out', default='k1_repeat_fixed_amp_6x6')
ap.add_argument('--maxlevel-states', type=int, default=3_000_000)
ap.add_argument('--parts', default='train,eval,ab')    # subset of train,eval,ab
ap.add_argument('--thresholds', default='')            # comma list T_0,..; skips training of those
ap.add_argument('--ab-sets', default='A_train,B_val')
ap.add_argument('--eval-curve', action='store_true')
ap.add_argument('--curve-extra', default='')            # extra comma-separated T values for the eval curve grid    # out-of-sample E(T_{K-1}) curve on the eval sample (diagnostic)
ap.add_argument('--train-range', default='')           # 'start:stop' slice of the training sample (split stage-K training)
ap.add_argument('--eval-range', default='')            # 'start:stop' (multiples of 64) of the 256 eval states
ap.add_argument('--ab-range', default='')              # 'start:stop' slice of each A/B set (for splitting)
args = ap.parse_args()
T0 = time.time()
def log(*a):
    print(f'[{time.time()-T0:8.1f}s]', *a, flush=True)

import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

L = 6; N = 36; J2 = .5
hi = nk.hilbert.Spin(s=.5, N=N)
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
model = ViT(num_layers=4, d_model=60, heads=10, L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
apply = nkjax.HashablePartial(_logpsi_transl_2d, model.apply, 2)
sam0 = nk.sampler.MetropolisExchange(hi, graph=g, d_max=2, n_chains=16, sweep_size=N)
v0 = nk.vqs.MCState(sampler=sam0, apply_fun=apply, n_samples=16,
                    variables=model.init(jax.random.PRNGKey(1234), jnp.zeros((1, N))), n_discard_per_chain=10)
with open('vit_J2=0.50_N=6x6_k=0.mpack', 'rb') as f:
    v0 = flax.serialization.from_bytes(v0, f.read())
log('LOADED nparams', v0.n_parameters, 'backend', jax.default_backend(), jax.devices())

_params = v0.variables
if args.fp32:
    _params = jax.tree_util.tree_map(lambda p: p.astype(jnp.complex64) if jnp.iscomplexobj(p) else
                                     (p.astype(jnp.float32) if jnp.issubdtype(p.dtype, jnp.floating) else p), _params)
_logpsi = jax.jit(lambda pars, x: apply(pars, x))
XDT = np.float32 if args.fp32 else np.float64
NEVAL = [0]

def bits2x(ss):
    a = np.asarray(ss, dtype=np.uint64).reshape(-1, 1)
    return (2 * ((a >> np.arange(N, dtype=np.uint64)) & 1).astype(XDT) - 1)

def evalz_states(states, batch=None):
    """complex log psi for uint64 states, fixed padded batch to avoid recompiles."""
    batch = batch or args.batch
    states = np.asarray(states, np.uint64); n = len(states)
    out = np.empty(n, np.complex128)
    for i in range(0, n, batch):
        s = states[i:i + batch]; m = len(s)
        if m < batch:
            s = np.concatenate([s, np.repeat(s[:1], batch - m)])
        z = np.asarray(_logpsi(_params, jnp.asarray(bits2x(s))))
        out[i:i + m] = z[:m]
    NEVAL[0] += n
    return out

# small-batch evaluator for the MCMC sampler (16..128 states per call)
def evalz_small(X):
    return np.asarray(v0.log_value(jnp.asarray(np.asarray(X, float))))

def bonds():
    nn = []; nnn = []; q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return nn, nnn
NN, NNN = bonds(); ALL = [(i, j, 1., -1.) for i, j in NN] + [(i, j, J2, 1.) for i, j in NNN]
BI = np.array([b[0] for b in ALL], np.uint64); BJ = np.array([b[1] for b in ALL], np.uint64)
JB = np.array([b[2] for b in ALL], float); MRB = np.array([b[3] for b in ALL], float)
MASKS = (np.uint64(1) << BI) | (np.uint64(1) << BJ)
A_MASK = np.uint64(sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0))
NB_ = len(ALL)

def neighbors(S):
    S = np.asarray(S, np.uint64)[:, None]
    valid = (((S >> BI) ^ (S >> BJ)) & np.uint64(1)).astype(bool)
    return S ^ MASKS, valid

def diag_vec(valid):
    # same spins: +J/4, opposite: -J/4
    return 0.25 * JB.sum() - 0.5 * (valid.astype(float) @ JB)

def marshall_vec(S):
    return np.where(np.bitwise_count(np.asarray(S, np.uint64) & A_MASK) % 2 == 0, 1., -1.)

def sample(seed, nchains, nround, burn=300, between=64, alpha=2.0):
    """identical protocol to krylov_scaling_6x6.sample (direct |a|^2 for alpha=2)."""
    rg = np.random.default_rng(seed); ss = []
    for _ in range(nchains):
        pos = rg.choice(N, N // 2, replace=False); q = 0
        for i in pos: q |= 1 << int(i)
        ss.append(q)
    la = evalz_small(bits2x(ss)).real
    out = []; acc = 0; props = 0
    for it in range(burn + between * nround):
        cand = []; valid = []
        for q in ss:
            i, j, J, mr = ALL[int(rg.integers(len(ALL)))]
            ok = ((q >> i) ^ (q >> j)) & 1
            cand.append(q ^ (1 << i) ^ (1 << j) if ok else q); valid.append(bool(ok))
        lb = evalz_small(bits2x(cand)).real
        ac = (np.log(rg.random(nchains)) < np.minimum(0, alpha * (lb - la))) & np.asarray(valid)
        acc += int(ac.sum()); props += nchains
        for k in np.where(ac)[0]: ss[k] = cand[k]
        la = np.where(ac, lb, la)
        if it >= burn and (it - burn + 1) % between == 0: out.extend(ss)
    log('SAMPLE', seed, 'n', len(out), 'accept', acc / props)
    return np.asarray(out, np.uint64)

def run_levels(X0, D, T, need_imag=False):
    """Exact recursion on the D-hop closure of the set X0 (D>=1).

    s_0 = Marshall on level D; r_k on level D-1-k; s_{k+1} = s_k sgn(T_k - r_k) on level D-1-k.
    Levels are nested (level l subset of level l+1), so any array can be restricted downward.
    Stops after r_k when k == len(T) (threshold not yet known).
    """
    levels = [np.unique(np.asarray(X0, np.uint64))]
    nbinfo = []
    for l in range(D):
        NBl, Vl = neighbors(levels[l])
        nxt = np.unique(np.concatenate([levels[l], NBl[Vl]]))
        if len(nxt) > args.maxlevel_states:
            raise MemoryError(f'level {l+1} has {len(nxt)} states')
        idx = np.searchsorted(nxt, NBl).astype(np.int64)
        idx[~Vl] = 0
        selfidx = np.searchsorted(nxt, levels[l])
        nbinfo.append((idx, Vl, selfidx))
        levels.append(nxt)
        del NBl
    top = levels[D]
    z = evalz_states(top)
    las = [z.real[np.searchsorted(top, levels[l])] for l in range(D + 1)]
    z1 = z[np.searchsorted(top, levels[1])] if need_imag else None
    del z
    s_created = {0: (D, marshall_vec(top))}
    r_created = {}
    s_cur = s_created[0][1]
    for k in range(D):
        l = D - 1 - k
        idx, Vl, selfidx = nbinfo[l]
        s_self = s_cur[selfidx]                      # s_k on level l
        w = np.exp(las[l + 1][idx] - las[l][:, None])
        r = diag_vec(Vl) + np.sum(np.where(Vl, 0.5 * JB[None, :] * s_cur[idx] * s_self[:, None] * w, 0.), axis=1)
        r_created[k] = (l, r)
        if k >= len(T): break
        s_cur = s_self * np.where(r <= T[k], 1., -1.)  # s_{k+1} on level l
        s_created[k + 1] = (l, s_cur)
    return dict(levels=levels, las=las, s=s_created, r=r_created, z1=z1, D=D)

def get_level(res, kind, k, level):
    l, arr = res[kind][k]
    if l < level: raise RuntimeError(f'{kind}_{k} only on level {l} < {level}')
    lv = res['levels']
    return arr if l == level else arr[np.searchsorted(lv[l], lv[level])]

def chunks_for(D, n):
    per = {1: 4096, 2: 256, 3: 16, 4: 1}[D]
    return [slice(i, min(n, i + per)) for i in range(0, n, per)]

def collect_edges(X, D, T, ks, rk=None, with_imag=False):
    """Full-edge data for central states X (kept in order, duplicates allowed).
    ks: steps whose signs s_k are needed on level 1; rk: step whose r_k is needed on level 1."""
    X = np.asarray(X, np.uint64)
    rec = dict(xi=[], c0=[], mr=[], bond=[], cosd=[], rx=[], ry=[], **{f'ss{k}': [] for k in ks})
    diag = np.zeros(len(X)); rx_node = np.full(len(X), np.nan); h0 = np.zeros(len(X))
    sx = {k: np.zeros(len(X)) for k in ks}
    for sl in chunks_for(D, len(X)):
        Xc = X[sl]
        res = run_levels(Xc, D, T, need_imag=with_imag)
        lv = res['levels']
        NBx, Vx = neighbors(Xc)
        idx1 = np.searchsorted(lv[1], NBx); idx1[~Vx] = 0
        self1 = np.searchsorted(lv[1], Xc)
        la1 = res['las'][1]
        w = np.exp(la1[idx1] - la1[self1][:, None])
        diag[sl] = diag_vec(Vx)
        sk1 = {k: get_level(res, 's', k, 1) for k in ks}
        for k in ks: sx[k][sl] = sk1[k][self1]
        if rk is not None:
            r1 = get_level(res, 'r', rk, 1); rx_node[sl] = r1[self1]
        if with_imag:
            im = res['z1'].imag
            cosd = np.cos(im[idx1] - im[self1][:, None])
            h0[sl] = np.where(np.cos(im[self1] - PHI) >= 0, 1., -1.)
        V = Vx; xi = np.broadcast_to(np.arange(sl.start, sl.stop)[:, None], V.shape)
        rec['xi'].append(xi[V]); rec['c0'].append((0.5 * JB[None, :] * w)[V])
        rec['mr'].append(np.broadcast_to(MRB[None, :], V.shape)[V])
        rec['bond'].append(np.broadcast_to(np.arange(NB_)[None, :], V.shape)[V])
        for k in ks: rec[f'ss{k}'].append((sk1[k][idx1] * sk1[k][self1][:, None])[V])
        if rk is not None:
            rec['rx'].append(np.broadcast_to(r1[self1][:, None], V.shape)[V]); rec['ry'].append(r1[idx1][V])
        if with_imag: rec['cosd'].append(cosd[V])
        if sl.start % 64 == 0 or sl.stop == len(X):
            log(f'  edges D={D} ks={ks} rk={rk} {sl.stop}/{len(X)} top_states={len(lv[-1])} neval={NEVAL[0]}')
    out = {key: (np.concatenate(v) if v else None) for key, v in rec.items()}
    out.update(diag=diag, rx_node=rx_node, h0=h0, sx=sx)
    return out

def energy_curve(E, k, mask_x=None):
    """E(T) = mean_x [D + sum_y c s_k(y)s_k(x) q(x)q(y)], q=+1 iff r_k<=T. Full edges, equal weights."""
    nx = len(E['diag'])
    keep = np.ones(len(E['xi']), bool) if mask_x is None else mask_x[E['xi']]
    nxe = nx if mask_x is None else int(mask_x.sum())
    c = (E['c0'] * E[f'ss{k}'])[keep] / nxe
    rx = E['rx'][keep]; ry = E['ry'][keep]
    base = (E['diag'].sum() if mask_x is None else E['diag'][mask_x].sum()) / nxe + c.sum()
    rvals = np.unique(np.concatenate([rx, ry]))
    cand = np.r_[rvals[0] - 1.0, 0.5 * (rvals[:-1] + rvals[1:]), rvals[-1] + 1.0]
    lo = np.minimum(rx, ry); hi = np.maximum(rx, ry)
    a = np.searchsorted(cand, lo, 'left'); b = np.searchsorted(cand, hi, 'left')   # flipped iff lo<=T<hi
    diff = np.zeros(len(cand) + 1)
    np.add.at(diff, a, -2 * c); np.add.at(diff, b, +2 * c)
    curve = base + np.cumsum(diff)[:len(cand)]
    return base, cand, curve

def energy_optimal_threshold(E, k):
    base, cand, curve = energy_curve(E, k)
    ib = int(np.argmin(curve)); Tk = float(cand[ib])
    info = dict(T=Tk, Ebase=float(base), Ebest=float(curve[ib]), gain=float(curve[ib] - base),
                gain_per_site=float((curve[ib] - base) / N),
                flip_frac_train=float(np.mean(E['rx_node'] > Tk)), ncand=int(len(cand)))
    # half-sample stability (split by chain parity; sample order is rounds x chains)
    ch = np.arange(len(E['diag'])) % args.ntrain_chains
    for name, m in (('half_even', ch % 2 == 0), ('half_odd', ch % 2 == 1)):
        b2, c2, cu2 = energy_curve(E, k, m)
        i2 = int(np.argmin(cu2)); info[f'T_{name}'] = float(c2[i2]); info[f'gain_{name}'] = float(cu2[i2] - b2)
    # gain at a fixed set of thresholds for reference
    for name, t in (('otsu_ref', T_OTSU_REF), ('otsu_audit', T_OTSU_AUDIT)):
        info[f'gain_at_{name}'] = float(curve[min(len(curve) - 1, np.searchsorted(cand, t))] - base)
    return info, cand, curve

def chain_stats(v, nch=64):
    v = np.asarray(v, float)
    if len(v) % nch: return float(v.mean()), float('nan')
    cm = v.reshape(-1, nch).mean(axis=0)
    return float(v.mean()), float(cm.std(ddof=1) / np.sqrt(len(cm)))

# ---------------------------------------------------------------- inputs
K = args.kmax
ind = np.load('energy_krylov_vs_vit_6x6_indep.npz')
XE_FULL = ind['states'].astype(np.uint64)
assert len(XE_FULL) == 256
E0_, E1_ = (int(t) for t in args.eval_range.split(':')) if args.eval_range else (0, min(256, args.neval))
XE = XE_FULL[E0_:E1_]
ab = np.load('krylov_scaling_6x6_a1.20_tr4096_va2048.npz')
with open('krylov_scaling_6x6_a1.20_tr4096_va2048.json') as f:
    PHI = float(json.load(f)['phi'])
T_OTSU_REF = -14.75594475197843          # threshold of the published K1 energy test (Fig. 3b)
T_OTSU_AUDIT = -14.985799779143964       # alpha=1.2 Otsu 5-95 (mechanism audit / gfmc)

if args.fp32:   # precision audit: fp32 vs fp64 log-amplitudes on 4096 states
    _st = ab['train_states'].astype(np.uint64)[:4096]
    _z32 = evalz_states(_st); _z64 = np.concatenate([evalz_small(bits2x(_st[i:i + 1024]).astype(float)) for i in range(0, 4096, 1024)])
    log('FP32_AUDIT max|dla|', float(np.max(np.abs(_z32.real - _z64.real))),
        'max|dphase|', float(np.max(np.abs(np.angle(np.exp(1j * (_z32.imag - _z64.imag)))))))
    NEVAL[0] = 0
_ = evalz_states(ab['train_states'].astype(np.uint64)[:16])
_tb = time.time(); _ = evalz_states(np.tile(ab['train_states'].astype(np.uint64), 16)[:65536])
log('THROUGHPUT states/s', 65536 / (time.time() - _tb)); NEVAL[0] = 0

PARTS = set(args.parts.split(','))
XT = sample(args.train_seed, args.ntrain_chains, args.ntrain_rounds) if 'train' in PARTS else np.zeros(0, np.uint64)
if args.train_range:
    _a, _b = (int(t) for t in args.train_range.split(':')); XT = XT[_a:_b]
log('TRAIN sample', len(XT), 'unique', len(np.unique(XT)), 'overlap_with_eval', len(np.intersect1d(XT, XE)))

# ---------------------------------------------------------------- thresholds T_0..T_{K-1}
T = [float(t) for t in args.thresholds.split(',') if t.strip()]; thr_info = []
if len(T) < K and 'train' not in PARTS:
    raise SystemExit(f'need {K} thresholds or train part')
for k in range(len(T), K):
    t1 = time.time(); n0 = NEVAL[0]
    E = collect_edges(XT, k + 2, T, [k], rk=k)
    info, cand, curve = energy_optimal_threshold(E, k)
    info.update(k=k, D=k + 2, sec=time.time() - t1, amp_evals=NEVAL[0] - n0,
                r_quantiles=np.quantile(E['rx_node'], [0, .01, .05, .1, .5, .9, .99, 1]).tolist())
    T.append(info['T']); thr_info.append(info)
    log('THRESHOLD', json.dumps(info))
    np.savez_compressed(f'{args.out}_curve_k{k}.npz', cand=cand, curve=curve)
    np.savez_compressed(f'{args.out}_trainedges_k{k}.npz', XT=XT, xi=E['xi'], c=E['c0'] * E[f'ss{k}'], rx=E['rx'], ry=E['ry'],
                        diag=E['diag'], rx_node=E['rx_node'], T_prev=np.asarray(T[:-1]))
    del E

# ---------------------------------------------------------------- evaluation-sample energies
rows_E = []; check = {}; check_sampled = {}; mv = sv = float('nan')
if 'eval' in PARTS:
    t1 = time.time(); n0 = NEVAL[0]
    E = collect_edges(XE, K + 1, T, list(range(K + 1)), rk=(K - 1 if (args.eval_curve and K >= 1) else None), with_imag=True)
    log('EVAL edges sec', time.time() - t1, 'amp_evals', NEVAL[0] - n0)
    Eo = collect_edges(XE, 2, [T_OTSU_REF], [1])
    assert np.array_equal(Eo['xi'], E['xi']) and np.array_equal(Eo['bond'], E['bond'])
    xi = E['xi']; c0 = E['c0']; cosd = E['cosd']
    def eloc(ss):
        e = E['diag'].copy(); np.add.at(e, xi, c0 * ss); return e
    ss_by = {str(k): E[f'ss{k}'] for k in range(K + 1)}; ss_by['k1_otsu_ref'] = Eo['ss1']
    sx_by = {str(k): E['sx'][k] for k in range(K + 1)}; sx_by['k1_otsu_ref'] = Eo['sx'][1]
    e_vit = eloc(cosd); e_mar = eloc(E['mr'])
    assert np.allclose(eloc(ss_by['0']), e_mar)
    check = {}
    check = dict(eM_maxdiff=float(np.max(np.abs(e_mar - ind['eM'][E0_:E1_]))),
                 eVA_maxdiff=float(np.max(np.abs(e_vit - ind['eVA'][E0_:E1_]))))
    log('CHECK_vs_published', check)

    # sampled-edge estimator: identical RNG and selection order to energy_krylov_vs_vit_6x6_indep.py
    RG = np.random.default_rng(202609292); KEDGE = 16
    deg_full = neighbors(XE_FULL)[1].sum(axis=1)
    starts = np.r_[0, np.cumsum(np.bincount(xi, minlength=len(XE)))]
    assert np.array_equal(np.diff(starts), deg_full[E0_:E1_])
    sel_edges = []; sel_x = []; sel_deg = []
    for bi in range(len(XE_FULL)):
        deg = int(deg_full[bi]); kk = min(KEDGE, deg)
        sel = RG.choice(np.arange(deg), size=kk, replace=False)   # same draw as choice(inds) of equal length
        if E0_ <= bi < E1_:
            sel_edges.extend((starts[bi - E0_] + sel).tolist()); sel_x.extend([bi - E0_] * kk); sel_deg.extend([deg] * kk)
    sel_edges = np.asarray(sel_edges); sel_x = np.asarray(sel_x); sel_deg = np.asarray(sel_deg, float)
    cnt = np.bincount(sel_x, minlength=len(XE))
    def sampled_delta(ss):
        d = np.zeros(len(XE)); np.add.at(d, sel_x, c0[sel_edges] * sel_deg * (ss[sel_edges] - cosd[sel_edges]))
        return d / cnt
    check_sampled = {}
    check_sampled = dict(dKA_k1_otsu_maxdiff=float(np.max(np.abs(sampled_delta(ss_by['k1_otsu_ref']) - ind['dKA'][E0_:E1_]))))
    log('CHECK_sampled_vs_published', check_sampled)

    eval_curve = {}
    if args.eval_curve and K >= 1:
        kk = K - 1
        b_, cand_, curve_ = energy_curve(E, kk)
        ib_ = int(np.argmin(curve_))
        def ex_T(t):   # per-x local energy of s_K(t) = s_{K-1} sgn(t - r_{K-1})
            qx = np.where(E['rx'] <= t, 1., -1.); qy = np.where(E['ry'] <= t, 1., -1.)
            e = E['diag'].copy(); np.add.at(e, xi, c0 * E[f'ss{kk}'] * qx * qy); return e
        e_base = eloc(E[f'ss{kk}'])
        grid = sorted(set([T[kk], float(cand_[ib_]), T_OTSU_REF, T_OTSU_AUDIT] +
                          [t for info in thr_info if info['k'] == kk for t in (info.get('T_half_even'), info.get('T_half_odd')) if t is not None] +
                          list(np.quantile(E['rx_node'], [.5, .9, .95, .99, .995, .999])) + [float(t) for t in args.curve_extra.split(',') if t.strip()] + [-17.5, -17.0, -16.5, -16.0, -15.5, -15.0, -14.5, -14.0, -12.0, -10.0, -5.0, 0.0, 5.0]))
        rows_c = []
        for t in grid:
            d = ex_T(t) - e_base; mu, se = chain_stats(d)
            rows_c.append(dict(T=float(t), dE_vs_prev=mu, dE_vs_prev_se=se, dE_per_site=mu / N, dE_per_site_se=se / N,
                               central_flip=float(np.mean(E['rx_node'] > t))))
        eval_curve = dict(k=kk, T_used=T[kk], T_eval_optimal_biased=float(cand_[ib_]),
                          gain_eval_optimal_biased=float(curve_[ib_] - b_), rows=rows_c)
        log('EVAL_CURVE', json.dumps(eval_curve))
        np.savez_compressed(f'{args.out}_evalcurve_k{kk}.npz', cand=cand_, curve=curve_)
    h0 = E['h0']
    rows_E = []
    for key in ss_by:
        ek = eloc(ss_by[key]); df = ek - e_vit; ds = sampled_delta(ss_by[key])
        em, ee = chain_stats(ek); mu, se = chain_stats(df); mus, ses = chain_stats(ds)
        mm = float(np.mean(sx_by[key] != h0))
        row = dict(step=key, E=em, E_se=ee, E_per_site=em / N,
                   dE_full=mu, dE_full_se=se, dE_per_site_full=mu / N, dE_per_site_full_se=se / N,
                   dE_sampled16=mus, dE_sampled16_se=ses, dE_per_site_sampled16=mus / N, dE_per_site_sampled16_se=ses / N,
                   eval_central_wrong_frac=min(mm, 1 - mm),
                   eval_central_changed_vs_marshall=float(np.mean(sx_by[key] != sx_by['0'])))
        rows_E.append(row); log('ENERGY', json.dumps(row))
    mv, sv = chain_stats(e_vit)
    log('ENERGY_VIT', mv, sv, mv / N)
    np.savez_compressed(f'{args.out}_eval.npz', XE=XE, XT=XT, e_vit=e_vit, T=np.asarray(T),
                        **{f'e_{k}': eloc(v) for k, v in ss_by.items()},
                        **{f'dsampled_{k}': sampled_delta(v) for k, v in ss_by.items()})
    del E, Eo

# ---------------------------------------------------------------- wrong-sign mass on samples A/B
def central_signs(states, kmax, Tlist):
    states = np.asarray(states, np.uint64)
    D = max(kmax, 1)
    s_all = {k: np.zeros(len(states)) for k in range(kmax + 1)}; h = np.zeros(len(states))
    for sl in chunks_for(D, len(states)):
        Xc = states[sl]
        res = run_levels(Xc, D, Tlist[:kmax], need_imag=True)
        lv = res['levels']; p0 = np.searchsorted(lv[0], Xc); p1 = np.searchsorted(lv[1], Xc)
        h[sl] = np.where(np.cos(res['z1'].imag[p1] - PHI) >= 0, 1., -1.)
        for k in range(kmax + 1): s_all[k][sl] = get_level(res, 's', k, 0)[p0]
        if (sl.start // (sl.stop - sl.start)) % 64 == 0:
            log(f'  AB k<={kmax} {sl.stop}/{len(states)} neval={NEVAL[0]}')
    return s_all, h

def wmass(s, h, iw):
    m = float(np.sum(iw * (s != h))); return min(m, 1 - m)

rows_W = {}
for lab, st, w in (('A_train', ab['train_states'], ab['iwtrain']), ('B_val', ab['val_states'], ab['iwval'])):
    if 'ab' not in PARTS or lab not in args.ab_sets.split(','): continue
    st = st.astype(np.uint64); w = np.asarray(w, float)
    w = w / w.sum()            # global normalisation; per-range sums kept for recombination
    if args.ab_range:
        a0, a1 = (int(t) for t in args.ab_range.split(':')); st = st[a0:a1]; w = w[a0:a1]
    n2 = len(st) if args.nab < 0 else min(args.nab, len(st))
    iw2 = w[:n2] / w[:n2].sum()
    t1 = time.time()
    s2, h2 = central_signs(st[:n2], min(K, 2), T)
    row = {str(k): dict(wrong=wmass(s2[k], h2, iw2)) for k in s2}
    row['wsum'] = float(w[:n2].sum())
    np.savez_compressed(f'{args.out}_ab_{lab}{args.ab_range.replace(":", "-")}.npz', states=st[:n2], w=w[:n2], h=h2,
                        **{f's{k}': s2[k] for k in s2})
    for k in s2: row[str(k)]['raw_wrong_wsum'] = float(np.sum(w[:n2] * (s2[k] != h2)))
    row['n_k<=2'] = int(n2); row['ESS_k<=2'] = float(1 / np.sum(iw2 * iw2))
    for tname, tval in (('k1_otsu_ref', T_OTSU_REF), ('k1_otsu_audit', T_OTSU_AUDIT)):
        so, ho = central_signs(st[:n2], 1, [tval]); row[tname] = dict(wrong=wmass(so[1], ho, iw2))
        row[tname]['raw_wrong_wsum'] = float(np.sum(w[:n2] * (so[1] != ho)))
    log('WRONG', lab, json.dumps(row), 'sec', time.time() - t1)
    if K >= 3:
        n3 = len(st) if args.nab_k3 < 0 else min(args.nab_k3, len(st))
        iw3 = w[:n3] / w[:n3].sum(); t1 = time.time(); n0 = NEVAL[0]
        s3, h3 = central_signs(st[:n3], 3, T)
        row['3'] = dict(wrong=wmass(s3[3], h3, iw3))
        row['n_k3'] = int(n3); row['ESS_k3'] = float(1 / np.sum(iw3 * iw3))
        row['k<=2_on_k3_subset'] = {str(k): wmass(s3[k], h3, iw3) for k in range(3)}
        row['k3_sec'] = time.time() - t1; row['k3_amp_evals'] = NEVAL[0] - n0
        np.savez_compressed(f'{args.out}_abk3_{lab}{args.ab_range.replace(":", "-")}.npz', states=st[:n3], w=w[:n3], h=h3,
                            **{f's{k}': s3[k] for k in s3})
        log('WRONG_K3', lab, json.dumps(row))
    rows_W[lab] = row

out = dict(eval_curve=(eval_curve if 'eval' in PARTS else {}), args=vars(args), description='repeated current-sign K1 at fixed ViT amplitude, 6x6 J2=0.5; energy-optimal label-free T_k',
           kmax=K, thresholds=T, threshold_info=thr_info, ntrain=int(len(XT)), train_seed=args.train_seed,
           neval=int(len(XE)), fp32=bool(args.fp32), energy_rows=rows_E, check_sampled_vs_published=check_sampled,
           E_vit=dict(mean=mv, se=sv, per_site=mv / N), check_vs_published=check,
           wrong_sign=rows_W, T_otsu_ref=T_OTSU_REF, T_otsu_audit=T_OTSU_AUDIT,
           amp_evals=int(NEVAL[0]), wall_sec=time.time() - T0)
with open(f'{args.out}.json', 'w') as f:
    json.dump(out, f, indent=2)
log('DONE', json.dumps(dict(T=T, amp_evals=NEVAL[0])))
