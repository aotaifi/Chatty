"""True wrong-sign probabilities vs the exact 6x6 J1-J2 (J2/J1=0.5) ground state.

Samples x ~ |psi0|^2 are exact (results/ed_6x6/samples_psi0sq_6x6.npz).  For each x we compute the
sign of: Marshall (s_0), ViT (own sign), K1 (s_1 from Marshall on |psi_ViT|, threshold T_0) and
K2 (s_2, threshold T_1), using exactly the Krylov machinery of k1_repeat_fixed_amp_6x6.py
(a = |psi_ViT| fixed, s_{k+1} = s_k sgn(T_k - r_k), r_k = (H a s_k)/(a s_k)).
Per-sample arrays are stored; wrong-sign probabilities/jackknife are done in
ed_6x6_sign_errors_analyze.py.

Bit convention (shared with the ED evaluator psi0_6x6.py): bit i of the uint64 = spin up on site
i = x + 6 y; ViT input is 2*bit-1 in site order i.  Marshall = (-1)^{popcount(bits & A)},
A = sites with (x+y) even  (identical to Psi0.marshall).

Sharding: shard j of P handles samples [j*B, (j+1)*B) of the first --n samples, processed in
chunks, checkpointing after every chunk; --max-seconds stops cleanly.
"""
import os, sys, time, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--kmax', type=int, default=2)
ap.add_argument('--n', type=int, default=100000)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--chunk', type=int, default=256)
ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--max-seconds', type=float, default=1e9)
ap.add_argument('--start', type=int, default=0)         # sample offset (to avoid overlap with previous stage)
ap.add_argument('--T', default='-15.024319756762353,-1.7500295942988462')
ap.add_argument('--samples', default='samples_psi0sq_6x6.npz')
ap.add_argument('--out', default='ed6x6_sign')
ap.add_argument('--maxlevel-states', type=int, default=3_000_000)
args = ap.parse_args()
T0 = time.time()
def log(*a):
    print(f'[{time.time()-T0:8.1f}s][sh{args.shard}]', *a, flush=True)

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
_params = v0.variables
_logpsi = jax.jit(lambda pars, x: apply(pars, x))
with open('krylov_scaling_6x6_a1.20_tr4096_va2048.json') as f:
    PHI = float(json.load(f)['phi'])
log('LOADED backend', jax.default_backend(), 'PHI', PHI)
NEVAL = [0]

def bits2x(ss):
    a = np.asarray(ss, dtype=np.uint64).reshape(-1, 1)
    return (2 * ((a >> np.arange(N, dtype=np.uint64)) & 1).astype(np.float64) - 1)

def evalz_states(states, batch=None):
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

def bonds():
    nn = []; nnn = []; q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return nn, nnn
NN, NNN = bonds(); ALL = [(i, j, 1., -1.) for i, j in NN] + [(i, j, J2, 1.) for i, j in NNN]
BI = np.array([b[0] for b in ALL], np.uint64); BJ = np.array([b[1] for b in ALL], np.uint64)
JB = np.array([b[2] for b in ALL], float)
MASKS = (np.uint64(1) << BI) | (np.uint64(1) << BJ)
A_MASK = np.uint64(sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0))

def neighbors(S):
    S = np.asarray(S, np.uint64)[:, None]
    valid = (((S >> BI) ^ (S >> BJ)) & np.uint64(1)).astype(bool)
    return S ^ MASKS, valid

def diag_vec(valid):
    return 0.25 * JB.sum() - 0.5 * (valid.astype(float) @ JB)

def marshall_vec(S):
    return np.where(np.bitwise_count(np.asarray(S, np.uint64) & A_MASK) % 2 == 0, 1., -1.)

def run_levels(X0, D, T):
    """Exact recursion on the D-hop closure of X0 (same as k1_repeat_fixed_amp_6x6.run_levels)."""
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
    z1 = z[np.searchsorted(top, levels[1])]
    del z
    s_created = {0: (D, marshall_vec(top))}
    s_cur = s_created[0][1]
    for k in range(D):
        l = D - 1 - k
        idx, Vl, selfidx = nbinfo[l]
        s_self = s_cur[selfidx]
        w = np.exp(las[l + 1][idx] - las[l][:, None])
        r = diag_vec(Vl) + np.sum(np.where(Vl, 0.5 * JB[None, :] * s_cur[idx] * s_self[:, None] * w, 0.), axis=1)
        if k >= len(T): break
        s_cur = s_self * np.where(r <= T[k], 1., -1.)
        s_created[k + 1] = (l, s_cur)
    return dict(levels=levels, las=las, s=s_created, z1=z1, D=D)

def get_level0(res, k):
    l, arr = res['s'][k]
    lv = res['levels']
    return arr if l == 0 else arr[np.searchsorted(lv[l], lv[0])]

Tl = [float(t) for t in args.T.split(',')][:args.kmax]
assert len(Tl) == args.kmax
S = np.load(args.samples)
x_all = S['x'].astype(np.uint64)[args.start:args.start + args.n]
psi_all = S['psi'][args.start:args.start + args.n]
mar_ed = S['marshall'][args.start:args.start + args.n]
# convention check 1: this code's Marshall == ED evaluator's Marshall
mv = marshall_vec(x_all)
log('CHECK marshall(ViT-code) == marshall(ED) on all samples:', bool(np.all(mv == mar_ed)), 'n', len(x_all))
B = int(np.ceil(len(x_all) / args.nshards))
i0, i1 = args.shard * B, min(len(x_all), (args.shard + 1) * B)
log('shard range', i0, i1, 'kmax', args.kmax, 'T', Tl)
res_idx = []; res = {k: [] for k in range(args.kmax + 1)}; res_h = []; res_la = []; res_chunk_done = 0
tstart = time.time(); neval0 = 0
def dump():
    np.savez_compressed(f'{args.out}_k{args.kmax}_sh{args.shard}of{args.nshards}.npz',
                        idx=np.concatenate(res_idx) + args.start if res_idx else np.zeros(0, int),
                        x=x_all[np.concatenate(res_idx)] if res_idx else np.zeros(0, np.uint64),
                        psi0=psi_all[np.concatenate(res_idx)] if res_idx else np.zeros(0),
                        h=np.concatenate(res_h) if res_h else np.zeros(0), la=np.concatenate(res_la) if res_la else np.zeros(0),
                        T=np.asarray(Tl), PHI=PHI, neval=NEVAL[0], wall=time.time() - tstart,
                        **{f's{k}': (np.concatenate(res[k]) if res[k] else np.zeros(0)) for k in res})
for a in range(i0, i1, args.chunk):
    if time.time() - tstart > args.max_seconds:
        log('time limit reached; stopping'); break
    ii = np.arange(a, min(a + args.chunk, i1))
    Xc = x_all[ii]
    r = run_levels(Xc, max(args.kmax, 1), Tl)
    lv = r['levels']; p0 = np.searchsorted(lv[0], Xc); p1 = np.searchsorted(lv[1], Xc)
    res_h.append(np.where(np.cos(r['z1'].imag[p1] - PHI) >= 0, 1., -1.))
    res_la.append(r['las'][0][p0])
    for k in range(args.kmax + 1): res[k].append(get_level0(r, k)[p0])
    res_idx.append(ii)
    dump()
    done = sum(len(t) for t in res_idx)
    log(f'chunk done {done}/{i1-i0} neval={NEVAL[0]} ({NEVAL[0]/(time.time()-tstart):.0f} evals/s)')
dump()
log('DONE', sum(len(t) for t in res_idx), 'samples, neval', NEVAL[0], 'wall', time.time() - tstart)
