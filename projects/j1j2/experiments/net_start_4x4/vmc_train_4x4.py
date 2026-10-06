#!/usr/bin/env python3
"""Plain VMC + SR on the 4x4 periodic J1-J2 (J2/J1=0.5, S^z=0, D=12870), exact i.i.d. sampling.

Network: complex-output translational ViT (nqsmagic.models.ViT, log psi = log-cosh head of
amp + i*sign) -> amplitude AND sign are learned; no Marshall prior.
Every step: forward pass on the whole basis (cheap), draw N i.i.d. configurations from the exact |psi|^2
(an ideal, perfectly-mixed VMC sampler), E_loc = (H psi)/psi from the exact sparse H and the full psi,
SR with the unique sampled configurations (weights = counts/N), complex SR written with real parameters:
  g = 2 Re <(E_loc - E)^* (O - <O>)>,  S = Re <(O-<O>)^dagger (O-<O>)>,  O = d log psi / d theta,
solved by CG (S + shift) d = g/2; step = min(lr, trust/RMS_w(delta log psi)) * d.
Scoring only (never used for training): exact E of psi (full basis), eps=(E-E0)/|E0|; guide
(|psi|, sgn Re psi e^{-i th0}) energy/eps/w_s; fidelity |<psi|psi0>|^2.
Continuable: --init ckpt.npz resumes from parameters (optimizer is stateless).
"""
import argparse, json, time, os
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as sla
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--d-model', type=int, default=60)
    ap.add_argument('--heads', type=int, default=10)
    ap.add_argument('--N', type=int, default=4000, help='i.i.d. samples per SR step')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--lr-end', type=float, default=None, help='cosine decay of lr to this value over --steps (default: constant)')
    ap.add_argument('--trust', type=float, default=0.05)
    ap.add_argument('--shift', type=float, default=1e-3)
    ap.add_argument('--cg-tol', type=float, default=1e-4)
    ap.add_argument('--cg-maxiter', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--marshall', type=int, default=0, help='1: psi = Marshall sign x network (sign rule built in)')
    ap.add_argument('--init', default=None)
    ap.add_argument('--cpus', type=int, default=16, help='cpus of the job (for CPU-h accounting)')
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--jac-chunk', type=int, default=128)
    ap.add_argument('--ckpt-every', type=int, default=50)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    t_start = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    H, diag, ei, ej, hij = build_H(0.5)
    E0, psi0 = ground(H, tol=1e-12)
    sM = canonical(marshall())
    if np.dot(psi0, sM) < 0: psi0 = -psi0
    strue = canonical(np.where(psi0 >= 0, 1, -1).astype(np.int8)); ptrue = psi0 ** 2

    X = bits2x(BASIS)
    model = ViT(num_layers=args.layers, d_model=args.d_model, heads=args.heads, L_eff=4, b=B,
                transl_invariant=True, two_dimensional=True)
    p0 = model.init(jax.random.PRNGKey(20261100 + args.seed), jnp.asarray(X[:2]))["params"]
    flat, unravel = ravel_pytree(p0)
    npar = int(flat.size)
    if args.init:
        flat = jnp.asarray(np.load(args.init)['flat'])
        assert flat.size == npar

    MA = jnp.asarray(np.array([1.0 if (xx + yy) % 2 == 0 else 0.0 for yy in range(L) for xx in range(L)]))  # site i = x + L*y
    def lp(z, x):
        o = _logpsi_transl_2d(model.apply, B, {"params": unravel(z)}, x)
        if args.marshall:
            nA = jnp.sum(MA * (x + 1.0) / 2.0, axis=-1)
            o = o + 1j * jnp.pi * jnp.mod(nA, 2.0)
        return o

    f_chunk = jax.jit(lp)

    def f2(z, x):
        o = lp(z, x[None, :])[0]
        return jnp.stack([o.real, o.imag])
    jac_chunk = jax.jit(jax.vmap(jax.jacrev(f2), in_axes=(None, 0)))

    def padded(idx, c):
        idx = np.asarray(idx); n = len(idx); npad = (-n) % c
        return (np.r_[idx, np.zeros(npad, dtype=idx.dtype)] if npad else idx), n

    def logpsi_all(z):
        pidx, n = padded(np.arange(D), args.chunk)
        o = [np.asarray(f_chunk(z, jnp.asarray(X[pidx[lo:lo + args.chunk]]))) for lo in range(0, len(pidx), args.chunk)]
        return np.concatenate(o)[:n]

    def jac(z, idx):
        pidx, n = padded(idx, args.jac_chunk)
        O = np.empty((n, 2, npar), dtype=np.float32)
        for lo in range(0, len(pidx), args.jac_chunk):
            g = np.asarray(jac_chunk(z, jnp.asarray(X[pidx[lo:lo + args.jac_chunk]])))
            hi = min(lo + args.jac_chunk, n)
            O[lo:hi] = g[:hi - lo]
        return O

    def score(lpsi):
        psi = np.exp(lpsi - lpsi.real.max())
        nrm2 = np.sum(np.abs(psi) ** 2)
        Hpsi = H @ psi
        E = float(np.real(np.vdot(psi, Hpsi)) / nrm2)
        amp, sg, imf = guide_from_psi(psi)
        pg = amp * sg
        Eg = float(pg @ (H @ pg))
        O = abs(float(np.sum(ptrue * sg * strue)))
        fid = float(abs(np.vdot(psi0, psi)) ** 2 / nrm2)
        return dict(E=E, eps=(E - E0) / abs(E0), E_guide=Eg, eps_guide=(Eg - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2),
                    fidelity=fid, imag_frac=imf), psi, Hpsi

    rng = np.random.default_rng(args.seed + 12345)
    hist = []; cum_wall = 0.0
    prev = Path(args.init).with_suffix('.json') if args.init else None
    base_wall = 0.0; base_step = 0
    if prev is not None and prev.exists():
        pj = json.loads(prev.read_text()); base_wall = pj.get('cum_wall_sec', 0.0); base_step = pj.get('steps_done', 0)
    print('SETUP', json.dumps(dict(D=D, npar=npar, E0=E0, **vars(args))), flush=True)

    def save(step, flat, final=False):
        np.savez(out / f'{args.tag}_ckpt.npz', flat=np.asarray(flat))
        (out / f'{args.tag}_ckpt.json').write_text(json.dumps(dict(
            args=vars(args), npar=npar, E0=E0, steps_done=base_step + step, cum_wall_sec=base_wall + time.time() - t_start,
            cpus=args.cpus, cpu_hours=(base_wall + time.time() - t_start) * args.cpus / 3600, history=hist)))

    for step in range(0, args.steps + 1):
        t0 = time.time()
        lpsi = logpsi_all(flat)
        sc, psi, Hpsi = score(lpsi)
        rec = dict(step=base_step + step, wall=base_wall + time.time() - t_start, cpu_h=(base_wall + time.time() - t_start) * args.cpus / 3600, **sc)
        if step == args.steps:
            hist.append(rec); break
        p = np.abs(psi) ** 2; p /= p.sum()
        smp = rng.choice(D, size=args.N, p=p)
        u, c = np.unique(smp, return_counts=True)
        w = c / c.sum()
        Eloc = (Hpsi[u] / psi[u])
        Es = np.sum(w * Eloc)
        eps_c = Eloc - Es
        O = jac(flat, u)                                    # (n,2,P)
        Ob = np.einsum('i,ikp->kp', w.astype(np.float32), O)
        O -= Ob[None]
        sw = np.sqrt(w).astype(np.float32)
        A = (O * sw[:, None, None]).reshape(-1, npar)       # rows: (re_i, im_i) interleaved
        del O
        rvec = np.stack([eps_c.real, eps_c.imag], axis=1) * np.sqrt(w)[:, None]   # (n,2)
        gh = (A.T @ rvec.reshape(-1).astype(np.float32)).astype(np.float64)   # = g/2
        gnorm = float(np.linalg.norm(gh))
        def mv(v):
            vv = np.asarray(v).astype(np.float32)
            return (A.T @ (A @ vv)).astype(np.float64) + args.shift * np.asarray(v)
        Aop = sla.LinearOperator((npar, npar), matvec=mv, dtype=np.float64)
        sol, info = sla.cg(Aop, gh, rtol=args.cg_tol, atol=0.0, maxiter=args.cg_maxiter)
        d = -sol
        rms = float(np.linalg.norm(A @ d.astype(np.float32)))
        del A
        lr = args.lr if args.lr_end is None else args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * step / args.steps))
        eta = min(lr, args.trust / max(rms, 1e-300))
        flat = flat + jnp.asarray(eta * d)
        rec.update(Es=float(Es.real), n_unique=int(len(u)), eta=float(eta), rms=rms, gnorm=gnorm, cg_info=int(info), lr=float(lr), t_step=time.time() - t0)
        hist.append(rec)
        if step % 10 == 0:
            print('STEP', json.dumps(rec), flush=True)
        if (step + 1) % args.ckpt_every == 0:
            save(step + 1, flat)
    save(args.steps, flat, True)
    print('DONE', json.dumps(hist[-1]), flush=True)


if __name__ == '__main__':
    main()
