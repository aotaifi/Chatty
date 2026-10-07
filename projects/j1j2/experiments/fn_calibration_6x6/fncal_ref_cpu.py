"""Independent CPU/numpy reference for fncal_dmc.py: same guide tables, canonical representative by brute-force min over
the 288 images and spin flip (no xor trick), and the propagation loop copied from learned_loop_6x6/ll6_fn.py (python loop
over walkers, np.searchsorted selection).  Used by fncal_test.py (local-data equality, population statistics)."""
import sys, json, time
import numpy as np
import fncal_dmc as F


class RefGuide:
    def __init__(self, name):
        self.reps, self.la, self.s = F.load_guide(name)
        self.T = F.byte_tables(F.PERMS)

    def canon_idx(self, x):
        x = np.atleast_1d(np.asarray(x, np.uint64))
        out = np.empty(len(x), np.int64)
        for a in range(0, len(x), 4096):
            img = F.apply_np(self.T, x[a:a + 4096])
            m = np.minimum(img.min(1), (img ^ F.FULL).min(1))
            out[a:a + 4096] = np.searchsorted(self.reps, m)
            assert np.all(self.reps[out[a:a + 4096]] == m)
        return out

    def local(self, x):
        """(dfn, eh, children (allowed), rates (allowed)) for one full-basis config x."""
        x = np.uint64(x)
        valid = (((x >> F.BI) ^ (x >> F.BJ)) & np.uint64(1)).astype(bool)
        ch = (x ^ F.MASKS)
        ix = self.canon_idx(x)[0]
        ic = self.canon_idx(ch[valid])
        w = np.zeros(F.NB); al = np.zeros(F.NB, bool)
        w[valid] = 0.5 * F.JB[valid] * np.exp(self.la[ic] - self.la[ix])
        al[valid] = self.s[ix] * self.s[ic] < 0
        diag = 0.25 * F.JB.sum() - 0.5 * float(valid @ F.JB)
        dfn = diag + w[valid & ~al].sum()
        eh = dfn - w[al].sum()
        return float(dfn), float(eh), ch[al], w[al]


def systematic(w, rng, M):
    c = np.cumsum(w); c[-1] = 1
    return np.searchsorted(c, rng.random() / M + np.arange(M) / M, 'right')


def run(G, localc, pool, M, beta_target, burn_beta, tau_max, seed):
    """ll6_fn.run, verbatim except for the guide object."""
    rg = np.random.default_rng(seed)
    walkers = pool[rg.choice(len(pool), M, replace=M > len(pool))].copy()
    def ensure_local(ss):
        for x in np.unique(ss):
            if int(x) not in localc:
                localc[int(x)] = G.local(x)
    beta = 0.; Es = []; curve = []
    while beta < beta_target:
        ensure_local(walkers); dat = [localc[int(x)] for x in walkers]
        diag = np.array([z[0] for z in dat]); el = np.array([z[1] for z in dat]); Eref = float(el.mean())
        mx = max(0., float(np.max(diag - Eref))); tau = min(tau_max, 0.8 / mx if mx > 0 else tau_max)
        curve.append((beta, tau, Eref))
        nxt = np.empty(M, np.uint64); bw = np.empty(M)
        for k, (x, (d, e, ys, rate)) in enumerate(zip(walkers, dat)):
            stay = 1 - tau * (d - Eref); ws = tau * rate; tot = stay + ws.sum()
            if stay < 0 or tot <= 0: raise RuntimeError(("bad", stay, tot, tau, d, Eref))
            u = rg.random() * tot
            if u < stay: y = x
            else:
                j = np.searchsorted(np.cumsum(ws), u - stay, 'right'); y = int(ys[min(j, len(ys) - 1)])
            nxt[k] = y; bw[k] = tot
        bw /= bw.sum(); walkers = nxt[systematic(bw, rg, M)]
        beta += tau
        if beta >= burn_beta:
            ensure_local(walkers); Es.append(float(np.mean([localc[int(x)][1] for x in walkers])))
    return np.asarray(Es), np.asarray(curve)


if __name__ == '__main__':
    guide, M, npop, beta, burn, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    out = sys.argv[7]
    G = RefGuide(guide); pool = np.load(F.POOL)['states'].astype(np.uint64).reshape(-1)
    localc = {}; res = []
    for k in range(npop):
        t = time.time()
        Es, curve = run(G, localc, pool, M, beta, burn, 0.025, seed0 + k)
        res.append(Es.mean() / F.N)
        print(f'pop {k} E={res[-1]:.7f} steps={len(curve)} cache={len(localc)} sec={time.time()-t:.0f}', flush=True)
        json.dump(dict(guide=guide, M=M, beta=beta, burn=burn, E_site=res), open(out, 'w'))
