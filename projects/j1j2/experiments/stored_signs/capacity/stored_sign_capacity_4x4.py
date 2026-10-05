#!/usr/bin/env python3
"""Capacity-vs-distribution test for the stored-sign 4x4 loop (net, tempered samples).

Imports (does not modify) ../stored_sign_loop_4x4.py and repeats its `run` loop for method 'cnn' with
  --hid/--nlayers : width / number of hidden layers of the translation-averaged MLP (default 64/2 = 5313 params = baseline)
  --beta          : tempering, samples from a^(2 beta)
  --nbr           : training set augmented with the one-hop neighbours of the tempered samples
                    (labels = exact one-hop Krylov sign there; weight = own count + parent count/deg)
  --device        : cpu / cuda
Extra per-iteration logging (all exact, ED used only for scoring):
  acc_train_w     : count-weighted accuracy on the training samples
  acc_hold_w      : count-weighted accuracy on a fresh i.i.d. held-out sample (size N) from the tempered distribution
  acc_hold_unseen_w : same restricted to held-out configs not in the training set
  acc_q_full / acc_p_full : accuracy over the whole space, weights tempered q(x) ~ a^(2 beta) / a^2
  D_true_stored_vs_rec : psi0^2-weighted mass of configurations where stored sign != exact one-hop Krylov sign
  wrong_mass_by_dist : psi0^2-weighted wrong-sign mass (stored vs ED sign) split by one-hop distance to the training set
At it in --dump-its the top wrong-sign configurations (state, a, psi0^2, local energy r, hop distance) are stored.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stored_sign_loop_4x4 as S
from stored_sign_loop_4x4 import basis, D, N, L, PERMS, build_H, marshall, ground, physical_energy, krylov, fixed_node, canon_rep


class CapNet(S.CNNSign):
    def __init__(self, seed, hid=64, nlayers=2, steps=400, lr=3e-3, wd=1e-5, device='cpu', threads=2):
        import torch
        self.torch = torch; self.dev = torch.device(device)
        torch.manual_seed(seed); torch.set_num_threads(threads)
        nn = torch.nn
        mods = [nn.Linear(N, hid)]
        for _ in range(nlayers - 1): mods += [nn.GELU(), nn.Linear(hid, hid)]
        mods += [nn.GELU(), nn.Linear(hid, 1)]
        self.mlp = nn.Sequential(*mods).to(self.dev)
        self.params = list(self.mlp.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr, weight_decay=wd)
        self.steps = steps; self.hid = hid
        self.nparam = sum(p.numel() for p in self.params)
        bits = ((basis[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float32)
        spins = 2 * bits - 1
        tr = PERMS[:L * L]
        Xt = np.empty((D, len(tr), N), dtype=np.float32)
        for k, p in enumerate(tr): Xt[:, k, p] = spins
        self.Xall = torch.tensor(Xt).to(self.dev)

    def fit(self, idx, wts, labels):
        t = self.torch
        X = self.Xall[idx]; y = t.tensor((labels > 0).astype(np.float32), device=self.dev)
        w = t.tensor(wts / wts.sum(), dtype=t.float32, device=self.dev)
        lossf = t.nn.functional.binary_cross_entropy_with_logits
        for _ in range(self.steps):
            self.opt.zero_grad()
            loss = (lossf(self.logits(X), y, reduction='none') * w).sum()
            loss.backward(); self.opt.step()
        return float(loss.detach())

    def predict_all(self):
        with self.torch.no_grad():
            return np.where(self.logits(self.Xall).cpu().numpy() > 0, 1, -1).astype(np.int8)


def run(args):
    t0 = time.time()
    H, diag, ei, ej, hij = build_H(args.target_j2)
    Hi, _, _, _, _ = build_H(args.init_source_j2)
    M = marshall()
    E0, psi0 = ground(H, tol=1e-12)
    strue = np.where(psi0 >= 0, 1, -1).astype(np.int8); ptrue = psi0 ** 2
    _, pin = ground(Hi, tol=1e-12)
    a = np.abs(pin); a /= np.linalg.norm(a)
    rng = np.random.default_rng(args.seed)           # same stream as the original script
    rng_h = np.random.default_rng(10_000 + args.seed)  # held-out samples (separate stream)
    net = CapNet(args.seed, hid=args.hid, nlayers=args.nlayers, steps=args.cnn_steps, device=args.device, threads=args.threads)
    A = (H - sp.diags(H.diagonal())).tocsr(); A.data[:] = 1.0; A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    s = M.copy()

    def score(a, s):
        E = physical_energy(H, a, s); O = abs(float(np.sum(ptrue * s * strue)))
        return dict(E=E, eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2))

    hist = [dict(it=0, **score(a, s))]
    print('ITER', json.dumps(hist[-1]), flush=True)
    dumps = {}
    for it in range(1, args.maxiter + 1):
        tt = time.time()
        efn, a = fixed_node(H, diag, ei, ej, hij, a, s, floor=args.amp_floor)
        srec, r = krylov(H, diag, ei, ej, hij, a, s)
        c = srec * s; p = a * a
        rec = dict(it=it, E_FN=efn, eps_FN=(efn - E0) / abs(E0), flip_weight=float(np.sum(p[c < 0])), min_amp=float(a.min()),
                   rec_eps=score(a, srec)['eps'], rec_w_s=score(a, srec)['w_s'])
        q = a ** (2 * args.beta); q /= q.sum()
        smp = rng.choice(D, size=args.nsamp, p=q)
        idx, cnt = np.unique(smp, return_counts=True)
        hs = rng_h.choice(D, size=args.nsamp, p=q); hidx, hcnt = np.unique(hs, return_counts=True)
        rec['n_unique'] = int(len(idx)); rec['sample_cover_weight'] = float(np.sum(p[idx]))
        in_train = np.zeros(D, dtype=bool); in_train[idx] = True
        tidx = idx; wts = cnt.astype(float)
        if args.nbr:
            w = np.zeros(D); w[idx] = cnt
            wfull = w + A.T @ (w / np.maximum(deg, 1))
            tidx = np.nonzero(wfull > 0)[0]; wts = wfull[tidx]
            rec['n_train_configs'] = int(len(tidx))
        lab = (srec[tidx] * M[tidx]).astype(np.int8)
        loss = net.fit(tidx, wts, lab)
        sig = net.predict_all(); snew = (M * sig).astype(np.int8)
        if np.sum(p * snew * srec) < 0: snew = -snew
        okM = (snew * srec) > 0          # stored agrees with exact one-hop sign (global sign aligned)
        okt = okM[tidx]; rec['train_loss'] = loss
        rec['acc_train_w'] = float(np.sum(wts * okt) / wts.sum())
        rec['acc_train_samples_w'] = float(np.sum(cnt * okM[idx]) / cnt.sum())
        rec['acc_hold_w'] = float(np.sum(hcnt * okM[hidx]) / hcnt.sum())
        un = ~in_train[hidx]
        rec['hold_unseen_frac'] = float(np.sum(hcnt[un]) / hcnt.sum())
        rec['acc_hold_unseen_w'] = float(np.sum(hcnt[un] * okM[hidx][un]) / max(hcnt[un].sum(), 1))
        rec['acc_q_full'] = float(np.sum(q * okM)); rec['acc_p_full'] = float(np.sum(p * okM))
        rec['D_stored_vs_rec'] = float(np.sum(p[~okM]))
        rec['D_true_stored_vs_rec'] = float(np.sum(ptrue[~okM]))
        # residual wrong-sign configs vs ED sign
        sc = np.sign(np.sum(ptrue * snew * strue)) or 1
        wrong = (snew * strue * sc) < 0
        d = np.full(D, 9, dtype=np.int64); d[idx] = 0
        front = np.zeros(D); front[idx] = 1; seen = front > 0
        for h in (1, 2, 3):
            nf = (A @ front) > 0; new = nf & ~seen; d[new] = h; seen |= nf; front = new.astype(float)
        rec['wrong_mass_by_dist'] = [float(np.sum(ptrue[wrong & (d == h)])) for h in (0, 1, 2, 3)] + [float(np.sum(ptrue[wrong & (d == 9)]))]
        rec['wrong_count_by_dist'] = [int(np.sum(wrong & (d == h))) for h in (0, 1, 2, 3)] + [int(np.sum(wrong & (d == 9)))]
        rec['n_wrong'] = int(wrong.sum())
        rec['wrong_ptrue_mean_a'] = float(np.sum(ptrue[wrong] * a[wrong]) / max(ptrue[wrong].sum(), 1e-300))
        s = snew
        rec.update(score(a, s))
        if it in args.dump_its:
            psi = a * s; rl = np.asarray((H @ psi) / np.where(np.abs(psi) > 1e-300, psi, 1.0), float)
            w_idx = np.nonzero(wrong)[0]; top = w_idx[np.argsort(-ptrue[w_idx])[:40]]
            dumps[str(it)] = dict(state=[int(basis[i]) for i in top], a=a[top].tolist(), ptrue=ptrue[top].tolist(), r=rl[top].tolist(),
                                  dist=d[top].tolist(), a_rank_pct=[float(np.mean(a < a[i])) for i in top],
                                  srec_ok=okM[top].tolist(), rec_ED_sign_ok=((srec[top] * strue[top] * np.sign(np.sum(ptrue * srec * strue))) > 0).tolist())
            rec['wrong_ptrue_weighted_r_mean'] = float(np.sum(ptrue[wrong] * rl[wrong]) / max(ptrue[wrong].sum(), 1e-300))
            rec['r_all_ptrue_mean'] = float(np.sum(ptrue * rl)); rec['r_all_ptrue_std'] = float(np.sqrt(np.sum(ptrue * (rl - np.sum(ptrue * rl)) ** 2)))
        rec['sec'] = time.time() - tt
        hist.append(rec)
        print('ITER', json.dumps({k: v for k, v in rec.items() if k not in ('wrong_count_by_dist',)}), flush=True)
    out = dict(method=dict(meth='cnn', nsamp=args.nsamp, seed=args.seed, beta=args.beta, nbr=args.nbr, hid=args.hid, nlayers=args.nlayers,
                           params=net.nparam, cnn_steps=args.cnn_steps, device=args.device, maxiter=args.maxiter, amp_floor=args.amp_floor,
                           T='exact full-vector energy-optimal (idealised)', FN='exact Perron GS of H_FN[a,s_hat]'),
               E0=E0, elapsed_sec=time.time() - t0, history=hist, dumps=dumps)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('DONE', args.out, time.time() - t0, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsamp', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxiter', type=int, default=40)
    ap.add_argument('--cnn-steps', type=int, default=400)
    ap.add_argument('--target-j2', type=float, default=0.5)
    ap.add_argument('--init-source-j2', type=float, default=0.0)
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--hid', type=int, default=64)
    ap.add_argument('--nlayers', type=int, default=2)
    ap.add_argument('--nbr', action='store_true')
    ap.add_argument('--amp-floor', type=float, default=1e-6)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--dump-its', type=int, nargs='*', default=[10, 20, 30, 40])
    ap.add_argument('--out', required=True)
    run(ap.parse_args())
