"""Brute-force checks for a0f_dilution.py (PREREG F1): Lanczos (L) and clipped Lanczos (Lc) energies on the explicit
M = 1 and M = 2 states (E_L obtained by applying H = H1 x 1 + 1 x H2 to the explicit 2-copy state), against the
per-copy machinery, for the guides MAR and P48s.
  python a0f_checks.py OUT.json
"""
import sys, json, time
import numpy as np
import numba as nb
import a0f_dilution as F


@nb.njit(cache=True)
def brute2_EL(psi, diag, idx, J):
    n = len(psi); nb_ = idx.shape[0]; EL = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            p = psi[a] * psi[b]
            h = (diag[a] + diag[b]) * p
            for k in range(nb_):
                ia = idx[k, a]
                if ia >= 0: h += 0.5 * J[k] * psi[ia] * psi[b]
                ib = idx[k, b]
                if ib >= 0: h += 0.5 * J[k] * psi[a] * psi[ib]
            EL[a, b] = h / p
    return EL


@nb.njit(cache=True)
def brute2_energy(psi, diag, idx, J, Fm):
    """<Psi'|H|Psi'> and <Psi'|Psi'> for Psi'(a, b) = psi[a] psi[b] Fm[a, b]."""
    n = len(psi); nb_ = idx.shape[0]; E = 0.0; Nn = 0.0
    for a in range(n):
        for b in range(n):
            p = psi[a] * psi[b] * Fm[a, b]
            h = (diag[a] + diag[b]) * p
            for k in range(nb_):
                ia = idx[k, a]
                if ia >= 0: h += 0.5 * J[k] * psi[ia] * psi[b] * Fm[ia, b]
                ib = idx[k, b]
                if ib >= 0: h += 0.5 * J[k] * psi[a] * psi[ib] * Fm[a, ib]
            E += p * h; Nn += p * p
    return E, Nn


def gfun(f, clip):
    return f if clip is None else np.sign(f) * np.maximum(np.abs(f), clip)


if __name__ == '__main__':
    S, G = F.make_guides(); out = {}
    idx = S['idx'].astype(np.int64); J = S['J']; diag = S['diag']; H = S['H']
    for name in ('MAR', 'P48s'):
        g = F.Guide(S, G[name]); psi = G[name]; E1 = g.E1; res = []
        assert np.all(np.abs(psi) > 1e-14)
        e = (H @ psi) / psi
        for M in (1, 2):
            gL, aL = g.lanczos(M)
            _, aC, _ = g.lc_opt(M)
            for step, al, clip in (('L', aL, None), ('Lc', aC, F.CLIP)):
                t = time.time()
                if M == 1:
                    ps = psi * gfun(1 + al * (e - E1), clip)
                    brute = (float(ps @ (H @ ps)) / float(ps @ ps) - E1) / 16
                else:
                    EL = brute2_EL(psi, diag, idx, J)
                    Fm = gfun(1 + al * (EL - 2 * E1), clip)
                    del EL
                    Eb, Nb = brute2_energy(psi, diag, idx, J, Fm)
                    del Fm
                    brute = (Eb / Nb - 2 * E1) / 32
                mach = gL if (step == 'L') else g.lc_energy(M, al)
                if step == 'L' and al != aL: mach = None
                res.append(dict(M=M, step=step, alpha=al, brute=brute, machinery=mach,
                                abs_diff=abs(brute - mach), rel_diff=abs(brute - mach) / abs(brute), sec=time.time() - t))
                print(name, res[-1], flush=True)
        out[name] = res
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
