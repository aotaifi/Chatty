"""FU1c: direct VMC of the one-Lanczos-step state psi_1 = psi_P (1 + alpha (E_L - E_ref)) = (1 - alpha E_ref) psi_P
+ alpha H psi_P (alpha, E_ref fixed numbers), sampled from |psi_1|^2 by bond-exchange Metropolis (each amplitude = one hop
of psi_P).  Chains start from referee samples of psi_P.  Local energy E_L1 = ((1 - a E_ref) E_L + a E_L2) / (1 + a (E_L - E_ref))
from the two-hop set.  Also the mirror paired estimate Delta = <H>_1 - <H>_P with weights 1/(1 + a eps)^2 back to psi_P.
  python l8_direct.py OUT SPEC.json   SPEC: sym, phi, alpha, Eref, init_ref, nchains, burn, nsamp, thin, seed, chunk, batch
"""
import sys, os, time, json
import numpy as np
import l8_core as C

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=SPEC['sym'], batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
al = float(SPEC['alpha']); Er = float(SPEC['Eref']); nch = SPEC['nchains']
rg = np.random.default_rng(SPEC['seed'])
z = np.load(SPEC['init_ref']); X0 = z['X'].reshape(-1, int(z['chain'].max()) + 1)[-1]
S = X0[rg.choice(len(X0), nch, replace=False)].copy()


def lp1(S):
    d = C.onehop(g, S)
    return d['la'] + np.log(np.abs(1 + al * (d['ELa'] - Er)))


cur = lp1(S); samples = []; acc = 0; prop = 0; t = time.time()
nburn = int(C.N * SPEC['burn']); nthin = max(1, int(C.N * SPEC['thin'])); nsteps = nburn + SPEC['nsamp'] * nthin
for it in range(1, nsteps + 1):
    b = rg.integers(C.NB_, size=nch)
    ok = (((S >> C.BI[b]) ^ (S >> C.BJ[b])) & C.ONE).astype(bool)
    cand = S ^ C.MASKS[b]
    new = np.full(nch, -np.inf)
    if ok.any(): new[ok] = lp1(cand[ok])
    a = ok & (np.log(rg.random(nch)) < 2 * (new - cur))
    S = np.where(a, cand, S); cur = np.where(a, new, cur); acc += a.sum(); prop += nch
    if it > nburn and (it - nburn) % nthin == 0:
        samples.append(S.copy())
    if it % 64 == 0: C.log(f'step {it}/{nsteps} accept {acc / prop:.3f} {time.time() - t:.0f}s')
ts = time.time() - t
X = np.stack(samples).reshape(-1); CH = np.tile(np.arange(nch), len(samples)); R = len(X)
EL = np.empty(R); EL2 = np.empty(R); t = time.time(); ch = SPEC.get('chunk', 64)
for i in range(0, R, ch):
    Sx = X[i:i + ch]; n = len(Sx)
    NBs, V = C.neighbors(Sx); o, bb = np.nonzero(V); ys = NBs[o, bb]
    Yu = np.unique(np.concatenate([Sx, ys])); d = C.onehop(g, Yu)
    px = np.searchsorted(Yu, Sx); py = np.searchsorted(Yu, ys)
    h = 0.5 * C.JB[bb] * d['s'][px][o] * d['s'][py] * np.exp(np.clip(d['la'][py] - d['la'][px][o], -60, 60))
    EL[i:i + n] = d['ELa'][px]
    EL2[i:i + n] = d['D'][px] * d['ELa'][px] + np.bincount(o, weights=h * d['ELa'][py], minlength=n)
eps = EL - Er
EL1 = ((1 - al * Er) * EL + al * EL2) / (1 + al * eps)
E1 = EL1.mean(); v = 1.0 / (1 + al * eps) ** 2; vt = v / v.mean(); EP = np.sum(vt * EL) / len(EL)
dinf = (EL1 - E1) - vt * (EL - EP)
res = dict(spec=SPEC, R=R, accept=acc / prop, sample_sec=ts, twohop_sec=time.time() - t,
           E1_direct_site=float(E1 / C.N), E1_direct_se=C.chain_se(EL1, CH) / C.N,
           EP_mirror_site=float(EP / C.N), Delta_mirror_site=float((E1 - EP) / C.N), Delta_mirror_se=C.chain_se(dinf, CH) / C.N,
           mirror_ess=float(v.sum() ** 2 / (v * v).sum() / R), min_factor=float((1 + al * eps).min()), sec=time.time() - T0)
np.savez(os.path.join(OUT, 'direct.npz'), X=X, chain=CH, EL=EL, EL2=EL2)
C.dump(os.path.join(OUT, 'direct.json'), res); C.log('DONE', json.dumps(res, default=float))
