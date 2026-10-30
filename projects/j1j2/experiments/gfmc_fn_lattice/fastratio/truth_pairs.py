"""50-digit (mpmath) truth for selected (configuration, bond) pairs: psi_P(y)/psi_P(x) with exact det of every image (8x8, translations x flip, RBM+PP).
  python truth_pairs.py AGREE.json OUT.json --params P.npz --cfg cfg.npz --set met --npairs 12 [--procs 8]
Compares the f64 ref and fast rP with the truth (relative error), i.e. separates the error of the launched formula from the error of the new one."""
import argparse, json, os, sys
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mpmath as mp
mp.mp.dps = 50
ap = argparse.ArgumentParser(); ap.add_argument('agree'); ap.add_argument('out'); ap.add_argument('--params', required=True); ap.add_argument('--cfg', required=True)
ap.add_argument('--set', default='met'); ap.add_argument('--npairs', type=int, default=12); ap.add_argument('--procs', type=int, default=8); ap.add_argument('--L', type=int, default=8)
a = ap.parse_args()
z = np.load(a.params); L = a.L; N = L * L; n = N // 2
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from rbmpp_lib import Lattice, make_fns
lat = Lattice(L, 0.5, 'translations'); IMG = np.asarray(lat.IMG); FL = np.asarray(lat.FL)
f = z['f']; bb = z['bre'] + 1j * z['bim']; W = z['Wre'] + 1j * z['Wim']
fmp = [[mp.mpf(float(f[i, j])) for j in range(N)] for i in range(N)]
bmp = [mp.mpc(float(x.real), float(x.imag)) for x in bb]; Wmp = [[mp.mpc(float(W[i, k].real), float(W[i, k].imag)) for k in range(W.shape[1])] for i in range(N)]


def psi_img(zz):
    zz = [int(v) for v in zz]
    U = [i for i in range(N) if zz[i] > 0]; D = [i for i in range(N) if zz[i] < 0]
    F = mp.matrix(n, n)
    for r, i in enumerate(U):
        for c, j in enumerate(D): F[r, c] = fmp[i][j]
    det = mp.det(F)
    inv = sum(1 for i in U for j in D if j < i)
    rbm = mp.mpf(1)
    for k in range(W.shape[1]):
        th = bmp[k] + sum(zz[i] * Wmp[i][k] for i in range(N))
        rbm *= mp.cosh(th)
    return (-1) ** inv * det * rbm


def psi_P(sig):
    s = np.asarray(sig, float)
    return sum(psi_img(FL[g] * s[IMG[g]]) for g in range(len(IMG)))


def job(args):
    return psi_P(args)


if __name__ == '__main__':
    ag = json.load(open(a.agree)); pairs = ag['worst_rP_pairs'][:a.npairs]
    X = np.load(a.cfg)[a.set].astype(np.float64)
    fr = make_fns(lat, ratio='ref'); ff = make_fns(lat, ratio='fast'); p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ)
    out = []
    with Pool(a.procs) as pool:
        for pr in pairs:
            k, b = pr['config'], pr['bond']; x = X[k].copy(); y = x.copy(); y[BI[b]], y[BJ[b]] = x[BJ[b]], x[BI[b]]
            # split the 128 images over the pool: x and y
            def parts(sig):
                s = np.asarray(sig, float); return [FL[g] * s[IMG[g]] for g in range(len(IMG))]
            vals = pool.map(psi_img, parts(x) + parts(y), chunksize=4)
            px = sum(vals[:len(IMG)]); py = sum(vals[len(IMG):]); truth = complex(py / px)
            rr = complex(np.asarray(fr['eloc_full'](p, jnp.asarray(x))[2])[b]); rf = complex(np.asarray(ff['eloc_full'](p, jnp.asarray(x))[2])[b])
            d = dict(config=k, bond=b, truth=[truth.real, truth.imag], err_ref_vs_truth=abs(rr - truth) / abs(truth), err_fast_vs_truth=abs(rf - truth) / abs(truth),
                     fast_vs_ref=abs(rf - rr) / abs(rr), config_max_image_kappa=pr.get('config_max_image_kappa'))
            out.append(d); print(json.dumps(d), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
