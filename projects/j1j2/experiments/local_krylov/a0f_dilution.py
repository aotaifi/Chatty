"""Follow-up F1 of check (a0): dilution with N of one GLOBAL step on M decoupled 4x4 copies, for
  K  = Krylov sign step s' = s sgn(T - E_L), best T;
  L  = Lanczos step psi' = psi f, f = 1 + alpha (E_L - E), best alpha (closed form from per-copy moments);
  Lc = clipped Lanczos psi' = psi sgn(f) max(|f|, 0.3), best alpha (per-copy structure + grid);
for several guides per copy (Marshall, perturbed-J2 exact signs, perturbed exact state, random sign flips).
Pre-registration: results/local_krylov/PREREG.md, section F1.
  python a0f_dilution.py OUT.json [guide ...]      (default: all guides; NPROC env for parallel guides)
"""
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numba as nb
import a0_copies as A

CLIP = 0.3; H0 = 2e-3; NMAX_GRID = 1.6e7
MGRID = sorted(set(int(round(2 ** (k / 2))) for k in range(25)))


def Hj2(st, idx, j2):
    B = A.bonds(); Jx = np.array([b[2] for b in B]); Jx[Jx == 0.5] = j2
    d = np.zeros(len(st)); rows = [np.arange(len(st))]; cols = [np.arange(len(st))]
    for k, (i, j, _) in enumerate(B):
        anti = ((st >> i) ^ (st >> j)) & 1; d += np.where(anti == 1, -Jx[k] / 4, Jx[k] / 4)
    vals = [d]
    for k in range(len(B)):
        m = idx[k] >= 0; rows.append(np.nonzero(m)[0]); cols.append(idx[k][m]); vals.append(np.full(m.sum(), Jx[k] / 2))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(st),) * 2)


def gs(Hm):
    v, V = eigsh(Hm, k=1, which='SA', tol=1e-14, ncv=60); p = V[:, 0]; return float(v[0]), p / np.linalg.norm(p)


def make_guides():
    st, H, idx, J, diag, mar = A.build()
    E0, p0 = gs(H); a = np.abs(p0); s0 = np.sign(p0)
    out = dict(MAR=a * mar)
    for j2, nm in ((0.45, 'P45s'), (0.48, 'P48s')):
        out[nm] = a * np.sign(gs(Hj2(st, idx, j2))[1])
    p48 = gs(Hj2(st, idx, 0.48))[1]; out['P48f'] = p48
    rng = np.random.default_rng(0); order = rng.permutation(len(a))
    for target, nm in ((1e-3, 'F3'), (1e-4, 'F4')):
        s = s0.copy(); tot = 0.0
        for i in order:
            if tot + a[i] ** 2 <= target: s[i] = -s[i]; tot += a[i] ** 2
        out[nm] = a * s
    for k in out:
        out[k] = out[k] / np.linalg.norm(out[k]); out[k] *= np.sign(out[k] @ p0)
    return dict(st=st, H=H, idx=idx, J=J, diag=diag, E0=E0, p0=p0), out


@nb.njit(cache=True)
def _band_terms(ux, uy, w, Rp, p, alpha, c, d_x, a2x, ue, diag_only):
    """Clipped-remainder sums.  Edge part: sum_e w [E r_x g_y + E f_x r_y];  node part: sum_x a2 (1 or d) E[g^2 - f^2].
    R' grid: Rp[j] = Rp0 + j h with probabilities p[j]; band where |1 + alpha z| < c."""
    h = Rp[1] - Rp[0]; R0 = Rp[0]; n = len(p)
    zlo = (1.0 - c) / (-alpha); zhi = (1.0 + c) / (-alpha)      # alpha < 0
    off = 0.0
    if not diag_only:
        for k in range(len(w)):
            for pass_ in range(2):
                u1 = ux[k] if pass_ == 0 else uy[k]; u2 = uy[k] if pass_ == 0 else ux[k]
                jlo = int(np.ceil((zlo - u1 - R0) / h)); jhi = int(np.floor((zhi - u1 - R0) / h))
                if jlo < 0: jlo = 0
                if jhi > n - 1: jhi = n - 1
                acc = 0.0
                for j in range(jlo, jhi + 1):
                    z1 = u1 + Rp[j]; f1 = 1.0 + alpha * z1
                    if abs(f1) >= c: continue
                    r1 = (c if f1 >= 0 else -c) - f1
                    z2 = u2 + Rp[j]; f2 = 1.0 + alpha * z2
                    if pass_ == 0:      # E[r_x g_y]
                        g2 = f2 if abs(f2) >= c else (c if f2 >= 0 else -c)
                        acc += p[j] * r1 * g2
                    else:               # E[f_x r_y]  (here u1 = u_y, u2 = u_x)
                        acc += p[j] * r1 * f2
                off += w[k] * acc
    nod = 0.0; nrm = 0.0
    for k in range(len(ue)):
        jlo = int(np.ceil((zlo - ue[k] - R0) / h)); jhi = int(np.floor((zhi - ue[k] - R0) / h))
        if jlo < 0: jlo = 0
        if jhi > n - 1: jhi = n - 1
        acc = 0.0
        for j in range(jlo, jhi + 1):
            f1 = 1.0 + alpha * (ue[k] + Rp[j])
            if abs(f1) >= c: continue
            acc += p[j] * (c * c - f1 * f1)
        nod += a2x[k] * d_x[k] * acc; nrm += a2x[k] * acc
    return off, nod, nrm


class Guide:
    def __init__(self, S, psi):
        H, idx, J, diag = S['H'], S['idx'], S['J'], S['diag']
        self.psi = psi; a = np.abs(psi); good = a > 1e-14
        Hp = H @ psi; self.E1 = float(psi @ Hp); a2 = psi ** 2
        e = np.zeros(len(a)); e[good] = Hp[good] / psi[good]
        self.good = good; self.e = e; self.a2 = a2; self.diag = diag
        hp = Hp - self.E1 * psi; self.c2 = float(hp @ hp); self.c3 = float(hp @ (H @ hp) - self.E1 * (hp @ hp))
        ex, ey, w = [], [], []
        for k in range(idx.shape[0]):
            m = (idx[k] >= 0) & good; x = np.nonzero(m)[0]; y = idx[k][m]; kk = good[y]; x = x[kk]; y = y[kk]
            ex.append(e[x]); ey.append(e[y]); w.append(psi[x] * 0.5 * J[k] * psi[y])
        self.ex = np.concatenate(ex); self.ey = np.concatenate(ey); self.w = np.concatenate(w)
        assert abs(float(a2 @ diag) + self.w.sum() - self.E1) < 1e-12
        self.eg = e[good]; self.ag = a2[good]; self.dg = diag[good]
        self.emin, self.emax = float(self.eg.min()), float(self.eg.max())
        lo = np.minimum(self.ex, self.ey); hi = np.maximum(self.ex, self.ey)
        self.cand = np.unique(np.r_[lo, hi]); dif = np.zeros(len(self.cand) + 1)
        np.add.at(dif, np.searchsorted(self.cand, lo), -2 * self.w); np.add.at(dif, np.searchsorted(self.cand, hi), 2 * self.w)
        self.g1s = np.cumsum(dif)[:len(self.cand)] / 16
        self.I = np.r_[0.0, np.cumsum(self.g1s[:-1] * np.diff(self.cand))]      # integral of g1 up to cand_k
        self._pm = {}; self._shift = {}

    def stats(self, E0):
        flip = self.ag[self.eg > 0]  # placeholder, not used
        return dict(E_err_site=(self.E1 - E0) / 16, c2=self.c2, c3=self.c3, sd_e=float(np.sqrt(self.c2)),
                    e_min=self.emin, e_max=self.emax)

    def hgrid(self, M, hscale=1.0):
        return hscale * max(H0, M * (self.emax - self.emin) / NMAX_GRID)

    def pmf(self, n, h):
        """grid pmf of the sum of n copies (offset n*emin, spacing h)."""
        key = (n, h)
        if key not in self._pm:
            K = int(np.ceil((self.emax - self.emin) / h)) + 1
            kb = np.rint((self.eg - self.emin) / h).astype(np.int64)
            p1 = np.bincount(kb, weights=self.ag, minlength=K)
            # F1-A3: rounding to the grid shifts the mean of every copy coherently; record the exact shift per copy
            self._shift[h] = float(self.ag @ (self.eg - (self.emin + h * kb)))
            if n == 0: pm = np.array([1.0])
            elif n == 1: pm = p1
            else:
                L = n * (K - 1) + 1; nf = 1 << int(np.ceil(np.log2(L)))
                pm = np.clip(np.fft.irfft(np.fft.rfft(p1, nf) ** n, nf)[:L], 0, None)
            if any(k[0] != n for k in self._pm): self._pm = {}
            self._pm[key] = pm; return pm
        return self._pm[key]

    def g1int(self, t):
        k = np.searchsorted(self.cand, t, side='right') - 1
        v = np.where(k >= 0, self.I[np.maximum(k, 0)] + self.g1s[np.maximum(k, 0)] * (t - self.cand[np.maximum(k, 0)]), 0.0)
        return v

    # ---- K: Krylov sign step
    def krylov(self, M, hscale=1.0):
        if M == 1:
            j = int(np.argmin(self.g1s)); return float(self.g1s[j]), float(self.cand[j] - self.E1)
        h = self.hgrid(M, hscale); K = int(np.ceil((self.emax - self.emin) / h)) + 1
        tk = self.emin + h * np.arange(K)
        g1a = (self.g1int(tk + h / 2) - self.g1int(tk - h / 2)) / h
        pm = self.pmf(M - 1, h); L = len(pm) + K - 1; nf = 1 << int(np.ceil(np.log2(L)))
        gM = np.fft.irfft(np.fft.rfft(pm, nf) * np.fft.rfft(g1a, nf), nf)[:L]
        j = int(np.argmin(gM)); T = M * self.emin + h * j + (M - 1) * self._shift[h]
        return float(gM[j]), float(T - M * self.E1)

    # ---- L: Lanczos step (closed form)
    def lanczos(self, M):
        m2, m3 = M * self.c2, M * self.c3
        al = (m3 - np.sqrt(m3 ** 2 + 4 * m2 ** 3)) / (2 * m2 ** 2)
        return float((2 * al * m2 + al ** 2 * m3) / (1 + al ** 2 * m2) / (16 * M)), float(al)

    # ---- Lc: clipped Lanczos (per-copy structure)
    def lc_energy(self, M, alpha, c=CLIP, hscale=1.0):
        """per-site energy change of psi sgn(f) max(|f|, c)."""
        hb = (2 * c / abs(alpha)) / 400                      # >= 400 grid points across the clip band
        hb = min(hb, np.sqrt(self.c2) / 64)                  # F1-A1: resolve the single-copy energy distribution
        h = hscale * H0 * 2.0 ** np.ceil(np.log2(max(self.hgrid(M), hb) / H0))
        pm = self.pmf(M - 1, h)
        V = (M - 1) * self.c2
        R0 = (M - 1) * self.emin - (M - 1) * self.E1 + (M - 1) * self._shift.get(h, 0.0)
        Rp = R0 + h * np.arange(max(len(pm), 2))
        pp = np.r_[pm, np.zeros(len(Rp) - len(pm))]
        ux = self.ex - self.E1; uy = self.ey - self.E1; ue = self.eg - self.E1
        if M == 1: Rp = np.array([0.0, h]); pp = np.array([1.0, 0.0])
        # F1-A2: aggregate edges / nodes whose energies fall in the same h-bin (bin = mean of its members); the
        # band sums then run over bins.  Same resolution as the grid of R.
        kk = np.stack([np.rint(ux / h), np.rint(uy / h)], 1).astype(np.int64)
        _, ie, cnt = np.unique(kk, axis=0, return_inverse=True, return_counts=True); ie = ie.ravel()
        wa = np.bincount(ie, weights=self.w); uxa = np.bincount(ie, weights=ux) / cnt; uya = np.bincount(ie, weights=uy) / cnt
        _, inn, cn = np.unique(np.rint(ue / h).astype(np.int64), return_inverse=True, return_counts=True); inn = inn.ravel()
        uea = np.bincount(inn, weights=ue) / cn; a2a = np.bincount(inn, weights=self.ag)
        a2d = np.bincount(inn, weights=self.ag * self.dg); da = np.where(a2a > 0, a2d / np.where(a2a > 0, a2a, 1), 0.0)
        off, nod, nrm = _band_terms(uxa, uya, wa, Rp, pp, alpha, c, da, a2a, uea, False)
        base = (1 + alpha * ue) ** 2 + alpha ** 2 * V
        Nn = float(self.ag @ base) + nrm
        Dg = float((self.ag * self.dg) @ base) + nod
        Of = float(self.w @ ((1 + alpha * ux) * (1 + alpha * uy) + alpha ** 2 * V)) + off
        return (Dg + Of) / (16 * Nn) - self.E1 / 16

    def lc_opt(self, M, hscale=1.0):
        _, aL = self.lanczos(M)
        sc = np.geomspace(0.2, 5, 13); vals = np.array([self.lc_energy(M, aL * s, hscale=hscale) for s in sc])
        j = int(np.argmin(vals)); lo, hi = sc[max(j - 1, 0)], sc[min(j + 1, len(sc) - 1)]
        f = lambda s: self.lc_energy(M, aL * s, hscale=hscale)
        gr = (np.sqrt(5) - 1) / 2; a_, b_ = np.log(lo), np.log(hi)
        c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_); fc, fd = f(np.exp(c_)), f(np.exp(d_))
        for _ in range(16):
            if fc < fd: b_, d_, fd = d_, c_, fc; c_ = b_ - gr * (b_ - a_); fc = f(np.exp(c_))
            else: a_, c_, fc = c_, d_, fd; d_ = a_ + gr * (b_ - a_); fd = f(np.exp(d_))
        s = np.exp(c_) if fc < fd else np.exp(d_)
        best = min(fc, fd, vals[j])
        return float(best), float(aL * s), bool(j in (0, len(sc) - 1))


def half_point(Ms, ratios):
    for i in range(1, len(Ms)):
        if ratios[i] <= 0.5:
            x0, x1 = np.log(Ms[i - 1]), np.log(Ms[i]); y0, y1 = ratios[i - 1], ratios[i]
            return float(16 * np.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))), int(Ms[i])
    return None, None


def run_guide(name, steps=('K', 'L', 'Lc')):
    if isinstance(name, tuple): name, steps = name
    S, G = make_guides(); g = Guide(S, G[name]); t0 = time.time()
    R = dict(name=name, **g.stats(S['E0']))
    R['w_s'] = float(g.a2[np.sign(g.psi) != np.sign(S['p0'])].sum())
    for step in steps:
        rows = []
        for M in MGRID:
            if step == 'K': gain, T = g.krylov(M); extra = dict(T_minus_E=T)
            elif step == 'L': gain, al = g.lanczos(M); extra = dict(alpha=al)
            else: gain, al, edge = g.lc_opt(M); extra = dict(alpha=al, alpha_at_scan_edge=edge)
            rows.append(dict(M=M, N=16 * M, gain_site=gain, **extra))
            rows[-1]['ratio'] = gain / rows[0]['gain_site']
            print(name, step, M, f"{gain:.4e} {rows[-1]['ratio']:.4f}", f'{time.time() - t0:.0f}s', flush=True)
            if rows[-1]['ratio'] < 0.25 and step != 'L': break
        Ms = [r['M'] for r in rows]; rat = [r['ratio'] for r in rows]
        Nh, Mn = half_point(Ms, rat)
        rr = {r['N']: r['ratio'] for r in rows}
        R[step] = dict(curve=rows, N_half=Nh, ratio_N64=rr.get(64), ratio_N192=float(np.interp(np.log(192), np.log([r['N'] for r in rows]), rat)))
        # grid check: h halved at the grid point nearest N_1/2
        if Mn is not None and step != 'L':
            if step == 'K': a_, b_ = g.krylov(Mn)[0], g.krylov(Mn, hscale=0.5)[0]
            else: a_, b_ = g.lc_opt(Mn)[0], g.lc_opt(Mn, hscale=0.5)[0]
            R[step]['h_check'] = dict(M=Mn, ratio=a_ / rows[0]['gain_site'], ratio_half_h=b_ / rows[0]['gain_site'])
    # c -> 0 limit of the Lc machinery vs the L closed form
    R['check_Lc_c0'] = []
    for M in (1, 16):
        gL, aL = g.lanczos(M); gc = g.lc_energy(M, aL, c=1e-12)
        R['check_Lc_c0'].append(dict(M=M, closed=gL, machinery=gc, rel=abs(gc - gL) / abs(gL)))
    R['sec'] = time.time() - t0
    return R


if __name__ == '__main__':
    out = sys.argv[1]; args = sys.argv[2:]
    steps = tuple(args[args.index('--steps') + 1].split(',')) if '--steps' in args else ('K', 'L', 'Lc')
    names = [a for a in args if a not in ('--steps',) and a != ','.join(steps)] or ['MAR', 'P45s', 'P48s', 'P48f', 'F3', 'F4']
    from multiprocessing import Pool
    import a0f_dilution as F_mod          # workers must see the module by its real name (numba cache)
    with Pool(int(os.environ.get('NPROC', 3))) as P: res = P.map(F_mod.run_guide, [(n, steps) for n in names])
    prev = json.load(open(out)) if os.path.exists(out) else {}
    for r in res:
        if r['name'] in prev: prev[r['name']].update({k: v for k, v in r.items()})
        else: prev[r['name']] = r
    json.dump(prev, open(out, 'w'), indent=1)
