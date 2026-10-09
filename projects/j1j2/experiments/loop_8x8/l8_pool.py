"""Stage 1: training pool for the write-back (guide-only quantities; no FN walkers anywhere).
Proposal: defensive mixture of Metropolis chains at a^2 (beta = 1) and a^1 (beta = 1/2), balance-heuristic weights to a^2
with the Meng-Wong bridge estimate of Z_2/Z_1 (bounded weights).  Per configuration x: K uniformly drawn valid bonds
(neighbours y_k); one-hop guide data (log a, s, V, W, D, E_L, semi-implicit target T) at x and at every y_k.
  python l8_pool.py OUT SPEC.json
SPEC: sym, phi, Ea (total energy of the binarised guide), nchains (per beta), burn, nsamp, thin, K, seed, chunk, batch
"""
import sys, os, time, json
import numpy as np
import l8_core as C

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'pool_meta.json'); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=SPEC['sym'], batch=SPEC.get('batch', 8192), phi=SPEC['phi'])
Ea = float(SPEC['Ea']); K = SPEC.get('K', 8); nch = SPEC['nchains']; seed = SPEC.get('seed', 1000)
meta = dict(spec=SPEC, stage={})

# ---------------------------------------------------------------- chains
Xs, betas, chains = [], [], []
for bi, beta in enumerate((1.0, 0.5)):
    t = time.time(); n0 = g.neval
    St, acc = C.sample_chains(g, beta, nch, SPEC['burn'], SPEC['nsamp'], SPEC['thin'], seed + 17 * bi)
    Xs.append(St.reshape(-1)); betas.append(np.full(St.size, beta)); chains.append(np.tile(np.arange(nch), SPEC['nsamp']) + bi * nch)
    meta['stage'][f'sample_beta{beta}'] = dict(sec=time.time() - t, evals=g.neval - n0, accept=acc)
    C.log('chains beta', beta, meta['stage'][f'sample_beta{beta}'])
X = np.concatenate(Xs); BETA = np.concatenate(betas); CH = np.concatenate(chains); P = len(X)

# ---------------------------------------------------------------- K bonds per configuration
rg = np.random.default_rng(seed + 99)
NBs, V = C.neighbors(X)
sc = np.where(V, rg.random(V.shape), -1.0)
bsel = np.argsort(-sc, axis=1)[:, :K]                               # K distinct valid bonds (nval >= K always here)
assert np.all(np.take_along_axis(V, bsel, 1))
Y = np.take_along_axis(NBs, bsel, 1); nval = V.sum(1); del NBs, V, sc

# ---------------------------------------------------------------- one-hop data on x and y_k
t = time.time(); n0 = g.neval
U = np.unique(np.concatenate([X, Y.reshape(-1)]))
keys = ('la', 's', 'V', 'W', 'D', 'ELa', 'T', 'ph')
tab = {k: np.empty(len(U)) for k in keys}
ch = SPEC.get('chunk', 4096)
for i in range(0, len(U), ch):
    d = C.onehop(g, U[i:i + ch], Ea=Ea)
    for k in keys: tab[k][i:i + ch] = d[k]
    if (i // ch) % 20 == 0:
        el = time.time() - t
        C.log(f'onehop {i + ch}/{len(U)}  {el:.0f}s  evals {g.neval - n0:.3e}  rate {(g.neval - n0) / max(el, 1e-9):.3e}/s')
meta['stage']['onehop'] = dict(sec=time.time() - t, evals=g.neval - n0, configs=len(U), configs_per_sample=len(U) / P,
                               evals_per_sample=(g.neval - n0) / P)
C.log('onehop done', meta['stage']['onehop'])
ix = np.searchsorted(U, X); iy = np.searchsorted(U, Y)

# ---------------------------------------------------------------- mixture weights to a^2 (balance heuristic, bridge)
la = tab['la'][ix]
ref = la.max(); l = np.exp(la - ref)
m1, m5 = BETA == 1.0, BETA == 0.5
n1, n5 = m1.sum(), m5.sum(); s1, s5 = n1 / P, n5 / P
r = np.median(l[m5])
for _ in range(500):
    r_new = np.mean(l[m5] / (s5 * r + s1 * l[m5])) / np.mean(1.0 / (s5 * r + s1 * l[m1]))
    if abs(r_new / r - 1) < 1e-12: r = r_new; break
    r = r_new
w = l / (s5 * r + s1 * l)
ess = w.sum() ** 2 / (w * w).sum() / P
meta['weights'] = dict(bridge_r=float(r), ess_frac=float(ess), wmax_over_mean=float(w.max() / w.mean()))
C.log('weights', meta['weights'])

np.savez(os.path.join(OUT, 'pool.npz'), X=X, Y=Y, bond=bsel.astype(np.int16), beta=BETA, chain=CH, nval=nval,
         lw=np.log(w), ix=ix, iy=iy, U=U, **{'U_' + k: tab[k] for k in keys}, Ea=Ea, phi=g.phi)
meta['P'] = P; meta['sec'] = time.time() - T0; meta['evals_total'] = g.neval
C.dump(F, meta); C.log('DONE', meta['sec'])
