"""Exact p-step Lanczos (Krylov-space) improvement of a trial vector, matvec-only (numpy or jax arrays).

lanczos_ritz(Hfun, v, pmax) -> list over p = 0..pmax of dict(psi, E, var, alpha):
  psi_p = lowest Ritz vector in span{v, Hv, ..., H^p v}  (orthonormal Lanczos basis with full reorthogonalisation),
  E = <psi_p|H|psi_p>, var = <H^2> - <H>^2 (exact; H q_k are all available, so no extra matvec), costs p+1 matvecs.
  p = 1: psi_1 = (1 + alpha H) v up to normalisation; alpha from the 2x2 generalised eigenproblem in span{v, Hv}.
extrapolate(Es, vars): linear E(var) -> var = 0 (Hu, Becca, Parola, Sorella, PRB 88, 060402 (2013)).
"""
import numpy as np
import scipy.linalg


def _dot(a, b):
    return float(a @ b)


def lanczos_ritz(Hfun, v, pmax=2):
    q = [v / _dot(v, v) ** 0.5]
    Hq = []
    for k in range(pmax + 1):
        h = Hfun(q[k]); Hq.append(h)
        if k < pmax:
            w = h
            for _ in range(2):
                for j in range(k + 1):
                    w = w - _dot(q[j], w) * q[j]
            q.append(w / _dot(w, w) ** 0.5)
    m = pmax + 1
    T = np.array([[_dot(q[i], Hq[j]) for j in range(m)] for i in range(m)])
    T = 0.5 * (T + T.T)
    out = []
    for p in range(pmax + 1):
        ev, ec = scipy.linalg.eigh(T[:p + 1, :p + 1])
        c = ec[:, 0]
        psi = sum(c[k] * q[k] for k in range(p + 1))
        Hpsi = sum(c[k] * Hq[k] for k in range(p + 1))
        E = _dot(psi, Hpsi)
        d = dict(p=p, psi=psi, E=E, var=_dot(Hpsi, Hpsi) - E * E, theta=float(ev[0]))
        if p == 1:   # psi ~ (1 + alpha H) v,  q1 = (Hv - E0 v)/beta, beta = ||Hv - E0 v||
            E0 = T[0, 0]
            beta = T[1, 0]                       # = q1 . H q0 = beta
            d['alpha'] = float((c[1] / beta) / (c[0] - c[1] * E0 / beta))
        out.append(d)
    return out


def extrapolate(Es, vs):
    """linear E = Eext + k*var through the given points (2 points: exact line; >2: least squares)."""
    Es = np.asarray(Es, float); vs = np.asarray(vs, float)
    A = np.stack([np.ones_like(vs), vs], 1)
    sol = np.linalg.lstsq(A, Es, rcond=None)[0]
    return float(sol[0])
