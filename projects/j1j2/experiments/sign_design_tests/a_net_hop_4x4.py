#!/usr/bin/env python3
"""(a) Stored sign net + ONE exact Krylov hop, 4x4 J2/J1=0.5.

Re-runs the stored-sign loop of experiments/stored_signs/stored_sign_loop_4x4.py (imported, not modified) for the
tempered translation-averaged net (--method cnn --beta 0.5 --nsamp 1e4), saving per-iteration diagnostics:
  net      : s_hat_k (net fitted at iteration k to the one-hop target srec_k = K(a_k, s_hat_{k-1}))
  hop_same : K(a_k, s_hat_k)  = ONE exact hop on the stored net with the current amplitude (energy-optimal T)
  hop_exact: K(|psi0|, s_hat_k) hop with the exact amplitude (diagnostic)
  srec_k   : the working sign actually used = K(a_k, s_hat_{k-1}) (one hop on the PREVIOUS net, current amplitude)
--work net : FN of the next iteration uses the stored net (reproduces the stored-sign loop)
--work hop : FN uses the working sign srec_k = net + one hop (the proposed design); net still fitted to srec_k.
Metrics vs ED (ED only for scoring): w_s = |psi0|^2 weight of wrong signs, eps = (E-E0)/|E0| of (a, s).
Decade table of residual errors vs |psi0(x)|^2 at selected iterations.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ.get("SSL_DIR", str(Path(__file__).resolve().parents[1] / "stored_signs")))
import stored_sign_loop_4x4 as S

ap = argparse.ArgumentParser()
ap.add_argument('--work', choices=['net', 'hop'], default='net')
ap.add_argument('--nsamp', type=int, default=10000)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--maxiter', type=int, default=40)
ap.add_argument('--cnn-steps', type=int, default=400)
ap.add_argument('--amp-floor', type=float, default=1e-6)
ap.add_argument('--tabit', default='10,20,30,35,40')
ap.add_argument('--out', required=True)
args = ap.parse_args()

H, diag, ei, ej, hij = S.build_H(0.5)
Hi, *_ = S.build_H(0.0)
M = S.marshall(); D = S.D
E0, psi0 = S.ground(H, tol=1e-12)
strue = np.where(psi0 >= 0, 1, -1).astype(np.int8); ptrue = psi0 ** 2; a0 = np.abs(psi0)
_, pin = S.ground(Hi, tol=1e-12); a = np.abs(pin); a /= np.linalg.norm(a)
rng = np.random.default_rng(args.seed)
cnn = S.CNNSign(args.seed, steps=args.cnn_steps)
tabit = [int(t) for t in args.tabit.split(',')]


def gauge(s):
    return 1 if np.sum(ptrue * s * strue) >= 0 else -1


def score(av, s):
    E = S.physical_energy(H, av, s); O = abs(float(np.sum(ptrue * s * strue)))
    return dict(eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2))


def errflags(s):
    return (s * strue * gauge(s)) < 0


edges = np.arange(-12, 0)  # decades of |psi0|^2
def decade_table(snet, shop):
    lg = np.log10(np.maximum(ptrue, 1e-300)); en = errflags(snet); eh = errflags(shop)
    rows = []
    for lo in range(int(np.floor(lg.min())), int(np.ceil(lg.max()))):
        m = (lg >= lo) & (lg < lo + 1)
        if not m.any(): continue
        rows.append(dict(log10_p_lo=lo, n=int(m.sum()), w=float(ptrue[m].sum()),
                         n_err_net=int((en & m).sum()), w_err_net=float(ptrue[en & m].sum()),
                         n_err_hop=int((eh & m).sum()), w_err_hop=float(ptrue[eh & m].sum()),
                         n_fixed=int((en & ~eh & m).sum()), n_broken=int((~en & eh & m).sum())))
    return rows


s_net = M.copy(); s_work = M.copy()
hist = [dict(it=0, **score(a, M))]
tabs = {}
for it in range(1, args.maxiter + 1):
    tt = time.time()
    s_fn = s_net if args.work == 'net' else s_work
    efn, a = S.fixed_node(H, diag, ei, ej, hij, a, s_fn, floor=args.amp_floor)
    srec, r = S.krylov(H, diag, ei, ej, hij, a, s_net)       # working sign: one hop on stored net, current amplitude
    p = a * a
    q = a ** (2 * args.beta); q /= q.sum()
    idx, cnt = np.unique(rng.choice(D, size=args.nsamp, p=q), return_counts=True)
    lab = (srec[idx] * M[idx]).astype(np.int8)
    loss, acc = cnn.fit(idx, cnt.astype(float), lab)
    snew = (M * cnn.predict_all()).astype(np.int8)
    if np.sum(p * snew * srec) < 0: snew = -snew
    hop_same, _ = S.krylov(H, diag, ei, ej, hij, a, snew)
    hop_ex, _ = S.krylov(H, diag, ei, ej, hij, a0, snew)
    rec = dict(it=it, E_FN=efn, eps_FN=(efn - E0) / abs(E0), flip_weight_hop=float(np.sum(p[hop_same != snew])),
               net=score(a, snew), work_srec=score(a, srec), hop_same=score(a, hop_same),
               net_exactamp=score(a0, snew), hop_exactamp=score(a0, hop_ex), train_loss=loss, train_acc=acc,
               n_unique=int(len(idx)), sec=time.time() - tt)
    hist.append(rec)
    print('ITER', json.dumps(rec), flush=True)
    if it in tabit:
        tabs[str(it)] = decade_table(snew, hop_same)
        np.savez_compressed(args.out.replace('.json', f'_it{it}.npz'), ptrue=ptrue.astype(np.float32),
                            a=a.astype(np.float32), err_net=errflags(snew), err_hop=errflags(hop_same),
                            err_srec=errflags(srec))
    s_net = snew; s_work = srec
out = dict(args=vars(args), E0=E0, history=hist, decade_tables=tabs)
Path(args.out).write_text(json.dumps(out, indent=1)); print('DONE', args.out, flush=True)
