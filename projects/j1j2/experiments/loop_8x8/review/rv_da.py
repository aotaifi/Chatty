"""Delayed-acceptance Metropolis (Christen & Fox 2005) for |psi_1|^2 = |psi_P|^2 f^2, f = 1 + alpha (E_L - E_ref).
Proposal: exchange the spins of a random bond (symmetric).  Stage 1 accepts with min(1, |psi_P(y)/psi_P(x)|^2) using one
amplitude; only stage-1 survivors get the one-hop evaluation for f(y), accepted with min(1, f(y)^2 / f(x)^2).
Detailed balance: pi(x) q a1 a2 = q min(psi_x^2, psi_y^2) min(f_x^2, f_y^2), symmetric in x, y.
"""
import numpy as np


class DAChain:
    def __init__(self, C, S, lp_psi, onehop_f, rg):
        self.C = C; self.S = np.asarray(S, np.uint64).copy(); self.lp_psi = lp_psi; self.onehop_f = onehop_f; self.rg = rg
        self.la, self.lf, self.el = onehop_f(self.S)
        self.acc1 = 0; self.acc2 = 0; self.prop = 0

    def run(self, nsteps):
        C = self.C; n = len(self.S)
        for _ in range(nsteps):
            b = self.rg.integers(C.NB_, size=n)
            ok = (((self.S >> C.BI[b]) ^ (self.S >> C.BJ[b])) & C.ONE).astype(bool)
            cand = self.S ^ C.MASKS[b]
            lac = np.full(n, -np.inf)
            if ok.any(): lac[ok] = self.lp_psi(cand[ok])
            a1 = ok & (np.log(self.rg.random(n)) < 2 * (lac - self.la))
            u2 = np.log(self.rg.random(n))
            if a1.any():
                la2, lf2, el2 = self.onehop_f(cand[a1])
                a2 = u2[a1] < 2 * (lf2 - self.lf[a1])
                ii = np.nonzero(a1)[0][a2]
                self.S[ii] = cand[ii]; self.la[ii] = la2[a2]; self.lf[ii] = lf2[a2]; self.el[ii] = el2[a2]
                self.acc2 += int(a2.sum())
            self.acc1 += int(a1.sum()); self.prop += n
