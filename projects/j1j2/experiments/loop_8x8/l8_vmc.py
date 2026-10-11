"""FU1a: large-sample one-hop VMC of the projected guide psi_P (absolute <H> with SE from independent chains).
  python l8_vmc.py OUT SPEC.json   SPEC: sym, phi, nchains, burn, nsamp, thin, seed, chunk, batch
"""
import sys, os, time, json
import numpy as np
import l8_core as C

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=SPEC['sym'], batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
t = time.time()
St, acc = C.sample_chains(g, 1.0, SPEC['nchains'], SPEC['burn'], SPEC['nsamp'], SPEC['thin'], SPEC['seed'])
ts = time.time() - t
X = St.reshape(-1); CH = np.tile(np.arange(SPEC['nchains']), SPEC['nsamp'])
EL = np.empty(len(X)); t = time.time(); ch = SPEC.get('chunk', 8192)
for i in range(0, len(X), ch):
    EL[i:i + ch] = C.onehop(g, X[i:i + ch])['ELa']
    C.log(f'onehop {i + ch}/{len(X)} {time.time() - t:.0f}s')
res = dict(spec=SPEC, n=len(X), accept=acc, sample_sec=ts, onehop_sec=time.time() - t, E_site=float(EL.mean() / C.N),
           se_site=C.chain_se(EL, CH) / C.N, sd_site=float(EL.std() / C.N), sec=time.time() - T0)
np.savez(os.path.join(OUT, 'vmc.npz'), X=X, chain=CH, EL=EL)
C.dump(os.path.join(OUT, 'vmc.json'), res); C.log('DONE', json.dumps(res, default=float))
