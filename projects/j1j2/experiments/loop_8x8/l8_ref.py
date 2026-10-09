"""Stage 3: referee data (independent of training).  Fresh Metropolis chains at a^2 (beta = 1, own seeds); for every
sample x the guide one-hop data at x and at ALL its valid neighbours y (= the guide on the two-hop set), so that the exact
local energy of any b = a exp(f) with f = f(spins, log a, V, W) is available without further ViT evaluations, as are
the local energies of the guide (E_L), of the network with its own complex phase, and (H^2 psi)/psi for one Lanczos step.
  python l8_ref.py OUT SPEC.json
SPEC: sym, phi, Ea, nchains, burn, nsamp, thin, seed, chunk, batch
"""
import sys, os, time, json
import numpy as np
import l8_core as C

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'ref_meta.json'); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=SPEC['sym'], batch=SPEC.get('batch', 8192), phi=SPEC['phi'])
Ea = float(SPEC['Ea']); nch = SPEC['nchains']; meta = dict(spec=SPEC, stage={})

t = time.time(); n0 = g.neval
St, acc = C.sample_chains(g, 1.0, nch, SPEC['burn'], SPEC['nsamp'], SPEC['thin'], SPEC.get('seed', 5000))
X = St.reshape(-1); CH = np.tile(np.arange(nch), SPEC['nsamp']); R = len(X)
meta['stage']['sample'] = dict(sec=time.time() - t, evals=g.neval - n0, accept=acc); C.log('chains', meta['stage']['sample'])

keys = ('la', 's', 'V', 'W', 'D', 'ELa', 'T')
xk = {k: np.empty(R) for k in keys + ('ELc_re', 'ELc_im')}
own, ystate, bond = [], [], []
yk = {k: [] for k in keys}
t = time.time(); n0 = g.neval
ch = SPEC.get('chunk', 256)
for i in range(0, R, ch):
    S = X[i:i + ch]; n = len(S)
    NBs, V = C.neighbors(S)
    o, b = np.nonzero(V); ys = NBs[o, b]
    Yu = np.unique(np.concatenate([S, ys]))
    d = C.onehop(g, Yu, Ea=Ea)                                    # one hop of {x} u N1(x)  = two-hop set of x
    px = np.searchsorted(Yu, S); py = np.searchsorted(Yu, ys)
    for k in keys: xk[k][i:i + n] = d[k][px]; yk[k].append(d[k][py])
    xk['ELc_re'][i:i + n] = d['ELc'][px].real; xk['ELc_im'][i:i + n] = d['ELc'][px].imag
    own.append(o + i); ystate.append(ys); bond.append(b.astype(np.int16))
    if (i // ch) % 10 == 0:
        el = time.time() - t
        C.log(f'ref {i + n}/{R}  {el:.0f}s  evals/sample {(g.neval - n0) / (i + n):.0f}  rate {(g.neval - n0) / max(el, 1e-9):.3e}/s')
meta['stage']['twohop'] = dict(sec=time.time() - t, evals=g.neval - n0, evals_per_sample=(g.neval - n0) / R)
own = np.concatenate(own); ystate = np.concatenate(ystate); bond = np.concatenate(bond)
yk = {k: np.concatenate(v) for k, v in yk.items()}
np.savez(os.path.join(OUT, 'ref.npz'), X=X, chain=CH, own=own, Y=ystate, bond=bond,
         **{'x_' + k: v for k, v in xk.items()}, **{'y_' + k: v for k, v in yk.items()}, Ea=Ea, phi=g.phi)
meta['R'] = R; meta['edges'] = len(own); meta['sec'] = time.time() - T0
e = xk['ELa']; meta['E_bin_site'] = float(e.mean() / C.N); meta['E_bin_se'] = C.chain_se(e, CH) / C.N
meta['E_complex_site'] = float(xk['ELc_re'].mean() / C.N); meta['E_complex_se'] = C.chain_se(xk['ELc_re'], CH) / C.N
C.dump(F, meta); C.log('DONE', json.dumps({k: v for k, v in meta.items() if k != 'spec'}, default=float))
