#!/usr/bin/env python3
"""Exact-FN / Krylov loop on 4x4 (J2/J1=0.5) with a STORED sign instead of the recursive one.

Reference (recursive) loop: krylov_sign_structure/experiments/closed_fn_krylov_anderson.py --method plain
  a_{k+1} = phi_FN[a_k, s_k];  s_{k+1} = Krylov(a_{k+1}, s_k) = s_k * c_k,  c_k = sgn[T_k - r_k] (aligned)
Here s_k is replaced after every step by an O(1)-evaluable stored sign s_hat_{k+1}, fitted on N
i.i.d. samples x ~ a_{k+1}^2 of the one-hop target s_rec_{k+1} = Krylov(a_{k+1}, s_hat_k):
  exact   : s_hat = s_rec (no storage; must reproduce full_4x4_plain.json)
  symtab  : cumulative parity table over symmetry orbits (translations x D4 x spin flip, |G|=256);
            orbits of sampled x with c=-1 are flipped; unseen orbits unchanged.
  tab     : same without symmetry (per-configuration table).
  options : --beta b samples from a^(2b) (b<1 tempers toward low-amplitude configs; table needs no reweighting,
            the net then uses a tempered loss); --nbr also stores c on the one-hop neighbours of each sample.
  cnn     : translation-invariant net (mean over translations of an MLP = full-kernel periodic CNN) for the Marshall-gauge sign sigma=s*M, BCE on samples
            (|a|^2-weighted by sampling), warm-started from the previous iteration.
Idealisations: FN amplitude exact; threshold T exact (full-vector energy optimum, as in the reference).
Target ED is used only for scoring (w_s, eps).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

L = 4; N = 16


def red(x, y):
    return (x % L) + L * (y % L)


NN = []; NNN = []
for y in range(L):
    for x in range(L):
        i = red(x, y)
        NN += [(i, red(x + 1, y)), (i, red(x, y + 1))]
        NNN += [(i, red(x + 1, y + 1)), (i, red(x + 1, y - 1))]
MMASK = 0
for y in range(L):
    for x in range(L):
        if (x + y) % 2 == 0:
            MMASK |= 1 << red(x, y)

basis = np.array([s for s in range(1 << N) if s.bit_count() == N // 2], dtype=np.int64)
D = len(basis)
s2i = np.full(1 << N, -1, dtype=np.int64); s2i[basis] = np.arange(D)


def build_H(J2):
    diag = np.zeros(D); rr = []; cc = []; vv = []; rows = np.arange(D)
    for bonds, J in ((NN, 1.0), (NNN, J2)):
        for u, v in bonds:
            same = ((basis >> u) & 1) == ((basis >> v) & 1)
            diag += J * np.where(same, .25, -.25)
            sel = ~same; r = rows[sel]
            t = basis[sel] ^ ((1 << u) | (1 << v))
            rr.append(r); cc.append(s2i[t]); vv.append(np.full(r.size, .5 * J))
    rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
    H = sp.coo_matrix((np.r_[vv, diag], (np.r_[rr, rows], np.r_[cc, rows])), shape=(D, D)).tocsr()
    up = sp.triu(H - sp.diags(diag), k=1).tocoo()
    return H, diag, up.row.astype(np.int64), up.col.astype(np.int64), up.data.astype(float)


def marshall():
    return np.array([1 if (int(s) & MMASK).bit_count() % 2 == 0 else -1 for s in basis], dtype=np.int8)


def ground(A, v0=None, tol=2e-11, ncv=64):
    ew, ev = sla.eigsh(A, k=1, which='SA', v0=v0, tol=tol, maxiter=300000, ncv=ncv)
    v = np.asarray(ev[:, 0], float); v /= np.linalg.norm(v)
    return float(ew[0]), v


def physical_energy(H, a, s):
    psi = a * s
    return float(psi @ (H @ psi) / (psi @ psi))


def group_ids(r, atol=1e-11, rtol=1e-11):
    order = np.argsort(r, kind='mergesort'); rs = r[order]
    scale = np.maximum(np.maximum(np.abs(rs[1:]), np.abs(rs[:-1])), 1.0)
    br = np.empty(len(r), dtype=bool); br[0] = True; br[1:] = np.abs(rs[1:] - rs[:-1]) > (atol + rtol * scale)
    gs = np.cumsum(br) - 1; gid = np.empty(len(r), dtype=np.int64); gid[order] = gs
    return gid, int(gs[-1]) + 1


def krylov(H, diag, ei, ej, hij, a, s):
    # identical to projected_krylov_update of closed_fn_krylov_anderson.py; returns sign aligned with s
    psi = a * s; r = np.asarray((H @ psi) / np.where(np.abs(psi) > 1e-300, psi, 1.0), float)
    gid, G = group_ids(r)
    wij = hij * a[ei] * a[ej]
    c0 = -s.astype(np.int8); val = 2.0 * wij * c0[ei] * c0[ej]
    e0 = float(np.sum(diag * a * a)) + float(np.sum(val))
    gi = gid[ei]; gj = gid[ej]; lo = np.minimum(gi, gj); hi = np.maximum(gi, gj); m = lo < hi
    delta = (np.bincount(lo[m], weights=-2.0 * val[m], minlength=G) +
             np.bincount(hi[m], weights=+2.0 * val[m], minlength=G))
    cand = np.r_[e0, e0 + np.cumsum(delta)]; kb = int(np.argmin(cand)) - 1
    sn = c0.copy()
    if kb >= 0: sn[gid <= kb] *= -1
    if np.sum(a * a * sn * s) < 0: sn = -sn
    return sn.astype(np.int8), r


def fixed_node(H, diag, ei, ej, hij, a, s, tol=2e-11, floor=1e-10):
    # floor (default 1e-6 via --amp-floor; reference code: 1e-15); reference loop has min a ~2e-5, so identical there. Stored-sign
    # runs drive a(x)->0 on unsampled wrong-sign states, and a 1e-15 floor makes Lanczos stall (ratio 1e10).
    af = np.maximum(a, floor)
    rr = np.r_[ei, ej]; cc = np.r_[ej, ei]; hh = np.r_[hij, hij]
    kij = s[rr].astype(float) * hh * s[cc].astype(float)
    keep = kij < 0; bad = ~keep
    d = diag + np.bincount(rr[bad], weights=kij[bad] * af[cc[bad]] / af[rr[bad]], minlength=D)
    rows = np.r_[rr[keep], np.arange(D)]; cols = np.r_[cc[keep], np.arange(D)]
    F = sp.coo_matrix((np.r_[kij[keep], d], (rows, cols)), shape=(D, D)).tocsr()
    e, v = ground(F, v0=af, tol=tol)
    v = np.abs(v); v /= np.linalg.norm(v)
    return e, v


# ---------------- symmetry group (translations x D4 x spin flip) ----------------
def site_perms():
    pg = [lambda x, y: (x, y), lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x),
          lambda x, y: (-x, y), lambda x, y: (x, -y), lambda x, y: (y, x), lambda x, y: (-y, -x)]
    perms = []
    for g in pg:
        for dx in range(L):
            for dy in range(L):
                p = np.empty(N, dtype=np.int64)
                for y in range(L):
                    for x in range(L):
                        X, Y = g(x, y); p[red(x, y)] = red(X + dx, Y + dy)
                perms.append(p)
    return np.array(perms)


PERMS = site_perms()
FULL = (1 << N) - 1


def canon_rep(states, spinflip=True):
    """Orbit representative (min integer image) of a batch of configurations: |G| N bit ops each."""
    states = np.asarray(states, dtype=np.int64)
    best = np.full(states.shape, np.iinfo(np.int64).max, dtype=np.int64)
    for p in PERMS:
        img = np.zeros_like(states)
        for i in range(N):
            img |= ((states >> i) & 1) << p[i]
        best = np.minimum(best, img)
        if spinflip:
            best = np.minimum(best, img ^ FULL)
    return best


# ---------------- CNN sign model ----------------
class CNNSign:
    """Translation-invariant sign net: logit(x) = mean_t MLP(T_t x) over the 16 lattice translations
    (= periodic CNN with a full 4x4 kernel followed by mean pooling)."""
    def __init__(self, seed, hid=64, steps=400, lr=3e-3, wd=1e-5):
        import torch
        self.torch = torch
        torch.manual_seed(seed); torch.set_num_threads(2)
        nn = torch.nn
        self.mlp = nn.Sequential(nn.Linear(N, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.params = list(self.mlp.parameters())
        self.opt = torch.optim.Adam(self.params, lr=lr, weight_decay=wd)
        self.steps = steps; self.hid = hid
        self.nparam = sum(p.numel() for p in self.params)
        bits = ((basis[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float32)
        spins = 2 * bits - 1
        tr = PERMS[:L * L]  # identity point-group element, all 16 translations
        Xt = np.empty((D, len(tr), N), dtype=np.float32)
        for k, p in enumerate(tr):
            Xt[:, k, p] = spins  # spin at site i moves to p[i]
        self.Xall = torch.tensor(Xt)

    def logits(self, X):
        return self.mlp(X).squeeze(-1).mean(dim=1)

    def fit(self, idx, cnt, labels):
        t = self.torch
        X = self.Xall[idx]; y = t.tensor((labels > 0).astype(np.float32)); w = t.tensor(cnt / cnt.sum(), dtype=t.float32)
        lossf = t.nn.functional.binary_cross_entropy_with_logits
        for _ in range(self.steps):
            self.opt.zero_grad()
            loss = (lossf(self.logits(X), y, reduction='none') * w).sum()
            loss.backward(); self.opt.step()
        with t.no_grad():
            acc = float((((self.logits(X) > 0).float() == y).float() * w).sum())
        return float(loss.detach()), acc

    def predict_all(self):
        with self.torch.no_grad():
            return np.where(self.logits(self.Xall).numpy() > 0, 1, -1).astype(np.int8)


def bench(out):
    rng = np.random.default_rng(0)
    B = 10000
    xs = basis[rng.integers(0, D, B)]
    reps_tab = np.unique(canon_rep(basis[rng.integers(0, D, 3000)]))
    t0 = time.perf_counter(); r = canon_rep(xs); pos = np.searchsorted(reps_tab, r)
    hit = (pos < len(reps_tab)) & (reps_tab[np.minimum(pos, len(reps_tab) - 1)] == r)
    t_sym = (time.perf_counter() - t0) / B
    m = CNNSign(0)
    import torch
    X = m.Xall[rng.integers(0, D, B)]
    with torch.no_grad():
        m.logits(X); t0 = time.perf_counter(); m.logits(X); t_cnn = (time.perf_counter() - t0) / B
    flops = 2 * L * L * (N * m.hid + m.hid * m.hid + m.hid)
    res = dict(symtab_sec_per_config=t_sym, symtab_bitops_per_config=len(PERMS) * 2 * N,
               cnn_sec_per_config=t_cnn, cnn_flops_per_config=flops, cnn_params=m.nparam, batch=B,
               note='numpy/torch CPU, 2 threads, batched; 6x6: |G|=576, 36 sites')
    print('BENCH', json.dumps(res), flush=True)
    Path(out).write_text(json.dumps(res, indent=2))


def run(args):
    t0 = time.time()
    H, diag, ei, ej, hij = build_H(args.target_j2)
    Hi, _, _, _, _ = build_H(args.init_source_j2)
    M = marshall()
    E0, psi0 = ground(H, tol=1e-12)
    strue = np.where(psi0 >= 0, 1, -1).astype(np.int8); ptrue = psi0 ** 2
    _, pin = ground(Hi, tol=1e-12)
    a = np.abs(pin); a /= np.linalg.norm(a)
    rep_all = canon_rep(basis)
    _, orb = np.unique(rep_all, return_inverse=True); norb = int(orb.max()) + 1
    rng = np.random.default_rng(args.seed)
    meth = args.method
    P = np.zeros(norb if meth == 'symtab' else D, dtype=np.int8)  # cumulative flip parity
    key = orb if meth == 'symtab' else np.arange(D)
    cnn = CNNSign(args.seed, steps=args.cnn_steps) if meth == 'cnn' else None
    s = M.copy()

    def score(a, s):
        E = physical_energy(H, a, s); O = abs(float(np.sum(ptrue * s * strue)))
        return dict(E=E, eps=(E - E0) / abs(E0), w_s=max(0.0, (1 - O) / 2))

    hist = [dict(it=0, **score(a, s))]
    print('ITER', json.dumps(hist[-1]), flush=True)
    seen = np.zeros(len(P), dtype=bool)
    for it in range(1, args.maxiter + 1):
        tt = time.time()
        efn, a = fixed_node(H, diag, ei, ej, hij, a, s, floor=args.amp_floor)
        srec, r = krylov(H, diag, ei, ej, hij, a, s)
        c = srec * s; p = a * a
        flipw = float(np.sum(p[c < 0]))
        # symmetry check of c (should be orbit-constant)
        cmin = np.full(norb, 2); cmax = np.full(norb, -2)
        np.minimum.at(cmin, orb, c); np.maximum.at(cmax, orb, c)
        c_asym_w = float(np.sum(p[(cmin != cmax)[orb]]))
        rec = dict(it=it, E_FN=efn, eps_FN=(efn - E0) / abs(E0), flip_weight=flipw, c_nonsym_weight=c_asym_w,
                   rec_eps=score(a, srec)['eps'], rec_w_s=score(a, srec)['w_s'], min_amp=float(a.min()),
                   n_amp_lt_1e8=int(np.sum(a < 1e-8)), w_true_on_amp_lt_1e8=float(np.sum(ptrue[a < 1e-8])),
                   wrong_true_w_on_amp_lt_1e8=float(np.sum(ptrue[(a < 1e-8) & (s * strue * np.sign(np.sum(ptrue * s * strue)) < 0)])))
        if meth == 'exact':
            snew = srec
        else:
            q = a ** (2 * args.beta); q /= q.sum()
            smp = rng.choice(D, size=args.nsamp, p=q)
            idx, cnt = np.unique(smp, return_counts=True)
            rec['n_unique'] = int(len(idx)); rec['sample_cover_weight'] = float(np.sum(p[idx]))
            if meth in ('symtab', 'tab'):
                tidx = idx
                if args.nbr:  # also store c on one-hop neighbours of samples (2-hop cost at build time only)
                    Hs = H[idx]; tidx = np.unique(np.r_[idx, Hs.indices])
                    rec['n_table_configs'] = int(len(tidx))
                k = np.unique(key[tidx[c[tidx] < 0]])
                P[k] ^= 1; seen[np.unique(key[tidx])] = True
                snew = (M * np.where(P[key] == 1, -1, 1)).astype(np.int8)
                rec['n_flipped_keys'] = int(len(k)); rec['table_cover_weight'] = float(np.sum(p[seen[key]]))
                rec['table_nonzero'] = int(np.sum(P)); rec['n_seen_keys'] = int(seen.sum()); rec['n_keys'] = int(len(seen))
                rec['flip_weight_unseen_this_step'] = float(np.sum(p[(c < 0) & ~np.isin(key, key[tidx])]))
            else:
                lab = (srec[idx] * M[idx]).astype(np.int8)
                loss, acc = cnn.fit(idx, cnt.astype(float), lab)
                sig = cnn.predict_all(); snew = (M * sig).astype(np.int8)
                if np.sum(p * snew * srec) < 0: snew = -snew
                rec['train_loss'] = loss; rec['train_acc_w'] = acc
        rec['D_stored_vs_rec'] = float(np.sum(p[snew != srec]))
        if np.sum(p * snew * srec) < 0: rec['D_stored_vs_rec'] = float(np.sum(p[snew == srec]))
        s = snew
        rec.update(score(a, s)); rec['sec'] = time.time() - tt
        hist.append(rec)
        print('ITER', json.dumps(rec), flush=True)
    out = dict(method=dict(meth=meth, nsamp=args.nsamp, seed=args.seed, target_j2=args.target_j2,
                           init_source_j2=args.init_source_j2, maxiter=args.maxiter, norb=norb, amp_floor=args.amp_floor, beta=args.beta, nbr=args.nbr,
                           cnn_params=(cnn.nparam if cnn else None), cnn_steps=args.cnn_steps,
                           T='exact full-vector energy-optimal (idealised)', FN='exact Perron GS of H_FN[a,s_hat]',
                           oracle_policy='ED only for scoring'),
               E0=E0, elapsed_sec=time.time() - t0, history=hist)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print('DONE', args.out, time.time() - t0, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['exact', 'symtab', 'tab', 'cnn'], default='exact')
    ap.add_argument('--nsamp', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxiter', type=int, default=40)
    ap.add_argument('--cnn-steps', type=int, default=400)
    ap.add_argument('--target-j2', type=float, default=0.5)
    ap.add_argument('--init-source-j2', type=float, default=0.0)
    ap.add_argument('--beta', type=float, default=1.0, help='sample from a^(2 beta)')
    ap.add_argument('--nbr', action='store_true', help='table also stores c on one-hop neighbours of samples')
    ap.add_argument('--amp-floor', type=float, default=1e-6)
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bench(a.out) if a.bench else run(a)
