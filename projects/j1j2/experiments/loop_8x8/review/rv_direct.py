"""Review T5 (REVIEW_lanczos8x8.md): independent direct VMC of psi_1 = psi_P (1 + alpha (E_L - E_ref)) with the SAME fixed
alpha, E_ref as FU1c, sampled by delayed-acceptance Metropolis (rv_da.py) from psi_P starting points; exact two-hop local
energies E_L1 (lines identical to l8_direct.py).  Records an equilibration trace every sweep and saves partial results.
  python rv_direct.py OUT SPEC.json
SPEC: phi, alpha, Eref, init_refs [paths], nchains, burn, nsamp, thin, seed, chunk, batch
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import l8_core as C
import rv_da

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=True, batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
al = float(SPEC['alpha']); Er = float(SPEC['Eref']); nch = SPEC['nchains']; rg = np.random.default_rng(SPEC['seed'])
X0 = np.unique(np.concatenate([np.load(p)['X'] for p in SPEC['init_refs']]))
S0 = X0[rg.choice(len(X0), nch, replace=False)]


def onehop_f(S):
    d = C.onehop(g, S)
    return d['la'], np.log(np.abs(1 + al * (d['ELa'] - Er))), d['ELa']


ch_ = rv_da.DAChain(C, S0, lambda S: g.eval(S)[0], onehop_f, rg)
trace = []; samples = []; stime = []
nb = int(SPEC['burn']); ns = int(SPEC['nsamp']); th = int(SPEC['thin']); t = time.time()
for sw in range(1, nb + ns * th + 1):
    ch_.run(C.N)
    trace.append(dict(sweep=sw, mean_eps=float(np.mean(ch_.el - Er)), median_eps=float(np.median(ch_.el - Er)),
                      mean_logf=float(np.mean(ch_.lf)), acc1=ch_.acc1 / ch_.prop, acc2=ch_.acc2 / max(ch_.acc1, 1)))
    if sw > nb and (sw - nb) % th == 0:
        samples.append(ch_.S.copy()); stime.append(sw)
    C.log(f'sweep {sw}', trace[-1], f'{time.time() - t:.0f}s')
ts = time.time() - t
X = np.stack(samples).reshape(-1); CH = np.tile(np.arange(nch), len(samples)); TT = np.repeat(np.array(stime), nch); R = len(X)
np.savez(os.path.join(OUT, 'samples.npz'), X=X, chain=CH, sweep=TT)
C.dump(os.path.join(OUT, 'trace.json'), dict(trace=trace, sample_sec=ts))
# ---- two-hop local energies: identical to l8_direct.py ----
EL = np.full(R, np.nan); EL2 = np.full(R, np.nan); t = time.time(); ch = SPEC.get('chunk', 64)
for i in range(0, R, ch):
    Sx = X[i:i + ch]; n = len(Sx)
    NBs, V = C.neighbors(Sx); o, bb = np.nonzero(V); ys = NBs[o, bb]
    Yu = np.unique(np.concatenate([Sx, ys])); d = C.onehop(g, Yu)
    px = np.searchsorted(Yu, Sx); py = np.searchsorted(Yu, ys)
    h = 0.5 * C.JB[bb] * d['s'][px][o] * d['s'][py] * np.exp(np.clip(d['la'][py] - d['la'][px][o], -60, 60))
    EL[i:i + n] = d['ELa'][px]
    EL2[i:i + n] = d['D'][px] * d['ELa'][px] + np.bincount(o, weights=h * d['ELa'][py], minlength=n)
    if (i // ch) % 8 == 0:
        np.savez(os.path.join(OUT, 'direct2.npz'), X=X, chain=CH, sweep=TT, EL=EL, EL2=EL2)
        C.log(f'twohop {i + n}/{R} {time.time() - t:.0f}s')
np.savez(os.path.join(OUT, 'direct2.npz'), X=X, chain=CH, sweep=TT, EL=EL, EL2=EL2)
eps = EL - Er
EL1 = ((1 - al * Er) * EL + al * EL2) / (1 + al * eps)
res = dict(spec=SPEC, R=R, accept1=ch_.acc1 / ch_.prop, accept2=ch_.acc2 / max(ch_.acc1, 1), sample_sec=ts, twohop_sec=time.time() - t,
           E1_site=float(EL1.mean() / C.N), E1_se_site=C.chain_se(EL1, CH) / C.N,
           by_sweep={int(s): float(EL1[TT == s].mean() / C.N) for s in np.unique(TT)},
           min_factor=float((1 + al * eps).min()), max_factor=float((1 + al * eps).max()), sec=time.time() - T0)
C.dump(os.path.join(OUT, 'direct2.json'), res); C.log('DONE', json.dumps(res, default=float))
