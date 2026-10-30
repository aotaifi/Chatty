"""Item 2 (near-node accuracy), the gpu_profile precision-arm pairs re-evaluated with the fast path.
  python acc_fast.py OUTDIR --acc ACC_F64_DIR --params P --consts C --ratio fast|ref --nrand 5000
ACC_F64_DIR = gpu_profile/acc_f64 (cfg.npz = the configurations, met.npz/uni.npz = the stored f64 reference-path outputs, p = 1 guide).
Pairs evaluated per set: the 1% with smallest |c(x)|, the 1% with smallest |psi_P(x)|, the 1% with smallest |c(y)| (as acc_compare.py,
selected on the stored f64 values) plus nrand random pairs.  Writes OUT/{met,uni}_<ratio>.npz with E_L, log psi_P at x (unique configs) and y."""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice
from gfmc_fn_lib import RBMPPGuideG
from gfmc_run import load_consts
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--acc', required=True)
ap.add_argument('--params', required=True); ap.add_argument('--consts', required=True); ap.add_argument('--ratio', default='fast')
ap.add_argument('--nrand', type=int, default=5000); ap.add_argument('--sets', default='met,uni'); ap.add_argument('--chunk', type=int, default=256)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
z = np.load(a.params); lat = Lattice(8, 0.5, 'translations'); c = load_consts(a.consts, 1)
g = RBMPPGuideG({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, c['alpha'], c['E'], w=c['w'], chunk=a.chunk, use_rbm=True, ratio=a.ratio)
cf = np.load(os.path.join(a.acc, 'cfg.npz')); BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ)
for name in a.sets.split(','):
    d = dict(np.load(os.path.join(a.acc, f'{name}.npz'))); wi, bi = d['wi'], d['bi']; npair = len(wi)
    al, E, w = float(d['alpha']), float(d['E']), complex(d['w'])
    fx = 1 + al * (d['Ex'] - E); fy = 1 + al * (d['Ey'] - E); ph = np.exp(1j * d['lx'].imag)
    cx = (w * ph * fx).real; cy = (w * np.exp(d['ly'] - d['lx'].real[wi]) * fy).real
    k = max(1, int(round(0.01 * npair))); sel = np.zeros(npair, bool)
    for key in (np.abs(cx)[wi], np.exp(d['lx'].real)[wi], np.abs(cy)):
        sel[np.argsort(key)[:k]] = True
    rng = np.random.default_rng(7); sel[rng.choice(npair, a.nrand, replace=False)] = True
    pi = np.nonzero(sel)[0]; ux = np.unique(wi[pi])
    X = cf[name]; s = X.astype(np.float64)
    Y = s[wi[pi]].copy(); r = np.arange(len(pi)); Y[r, BI[bi[pi]]] = s[wi[pi], BJ[bi[pi]]]; Y[r, BJ[bi[pi]]] = s[wi[pi], BI[bi[pi]]]
    t0 = time.time()
    Ex, lx = g.amp(s[ux]); Ey, ly = g.amp(Y)
    np.savez(os.path.join(a.out, f'{name}_{a.ratio}.npz'), pairs=pi, ux=ux, Ex=Ex, lx=lx, Ey=Ey, ly=ly, nfallback=g.n_fallback)
    print(name, a.ratio, 'pairs', len(pi), 'unique x', len(ux), 'sec', round(time.time() - t0), 'fallback', g.n_fallback, flush=True)
