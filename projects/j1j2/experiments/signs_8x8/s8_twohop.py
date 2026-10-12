"""Two-hop guide data at 8x8 (GPU): fresh Metropolis chains at a^(2 beta) of the D4 x flip projected ViT, and for every
sample x the guide (log a, binarised own sign) on the full two-hop set  U2 = {x} u N1(x) u N2(x).
From U2 every sign function b (ViT sign, Marshall, any stored sign net) gives, without further ViT calls:
  r_b(z) = (H a b)(z) / (a(z) b(z)) on z in {x} u N1(x)   (Krylov step input, labels),
  E_L of (a, b) and of (a, b sgn(T - r_b)) at x            (energies of a sign and of the sign + one exact hop).
Used for the training set (beta = 0.5: labels on x and ALL its valid neighbours) and for the referee (beta = 1).
Chunks are saved as they are done (restartable).
  python s8_twohop.py OUT SPEC.json      SPEC: beta, nchains, burn, nsamp, thin, seed, chunk, batch, phi
"""
import sys, os, time, json, glob
import numpy as np
import l8_core as C

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=True, batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
fs = os.path.join(OUT, 'samples.npz')
meta = dict(spec=SPEC, stage={})
if os.path.exists(fs):
    d = np.load(fs); X = d['X']; CH = d['chain']; C.log('samples reloaded', len(X))
else:
    t = time.time(); n0 = g.neval
    St, acc = C.sample_chains(g, SPEC['beta'], SPEC['nchains'], SPEC['burn'], SPEC['nsamp'], SPEC['thin'], SPEC['seed'])
    X = St.reshape(-1); CH = np.tile(np.arange(SPEC['nchains']), SPEC['nsamp'])
    np.savez(fs, X=X, chain=CH)
    meta['stage']['sample'] = dict(sec=time.time() - t, evals=g.neval - n0, accept=acc); C.log('chains', meta['stage']['sample'])
ch = SPEC.get('chunk', 128); R = len(X); tt = time.time(); n0 = g.neval; nU = 0
for k, i in enumerate(range(0, R, ch)):
    fk = os.path.join(OUT, f'chunk_{k:04d}.npz')
    if os.path.exists(fk): continue
    S = X[i:i + ch]
    NBs, V = C.neighbors(S)
    L1 = np.unique(np.concatenate([S, NBs[V]]))
    NB1, V1 = C.neighbors(L1)
    U2 = np.unique(np.concatenate([L1, NB1[V1]]))
    la, ph = g.eval(U2)
    np.savez(fk, X=S, chain=CH[i:i + ch], U2=U2, la=la, s=g.sign(ph).astype(np.int8))
    nU += len(U2)
    el = time.time() - tt
    C.log(f'chunk {k}  samples {i + len(S)}/{R}  U2 {len(U2)}  {el:.0f}s  rate {(g.neval - n0) / max(el, 1e-9):.3e}/s')
meta['stage']['twohop'] = dict(sec=time.time() - tt, evals=g.neval - n0, evals_per_sample=(g.neval - n0) / R)
meta['R'] = R; meta['sec'] = time.time() - T0
C.dump(os.path.join(OUT, 'twohop_meta.json'), meta); C.log('DONE', json.dumps(meta, default=float))
