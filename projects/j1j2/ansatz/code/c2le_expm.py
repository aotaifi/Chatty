"""exp(tau H) v by Lanczos-Krylov with full reorthogonalisation, adaptive Krylov dimension (C2-L exp form, c2le_gtax.py).

expm_krylov(op, v, tau): op(x) = H x with H Hermitian (real symmetric here); v real; tau real (either sign).
Lanczos from q_0 = v/|v| builds T_m (m x m tridiagonal); y_m = exp(tau T_m) e_1 (eigh); result |v| Q_m y_m.
Dimension: m = m0, m0 + dm, ...; converged when |y_{m+dm} - y_m| / |y_{m+dm}| < tol (coefficient-space norm = vector
norm, Q orthonormal); the m + dm vector is returned. Lanczos breakdown (beta_j < 1e-14 |T|) = invariant subspace: exact,
converged. Not converged at m > mmax -> info['converged'] False (the caller makes that fatal).
Works with numpy or jax.numpy arrays (only @, *, -, + and float() on vectors are used).
"""
import numpy as np


def expm_krylov(op, v, tau, tol=1e-12, m0=10, dm=10, mmax=100):
    beta0 = float(np.sqrt(float(v @ v)))
    info = dict(tau=float(tau), tol=tol, m0=m0, dm=dm, mmax=mmax, matvecs=0, m=0, change=None, changes=[], converged=False, breakdown=False)
    if beta0 == 0.0:
        info.update(converged=True); return v, info
    Q = [v / beta0]; al = []; be = []

    def extend(m):
        """grow the Lanczos basis to m steps (len(al) == m); return False on breakdown."""
        while len(al) < m:
            j = len(al); z = op(Q[j]); info['matvecs'] += 1
            a = float(Q[j] @ z); z = z - a * Q[j]
            if j > 0: z = z - be[j - 1] * Q[j - 1]
            for _ in range(2):                                   # full reorthogonalisation, twice
                for q in Q: z = z - float(q @ z) * q
            b = float(np.sqrt(float(z @ z))); al.append(a)
            scale = max(abs(x) for x in al) + (max(be) if be else 0.0)
            if b < 1e-14 * max(scale, 1e-300):
                info['breakdown'] = True; return False
            be.append(b); Q.append(z / b)
        return True

    def coeffs(m):
        Tm = np.diag(al[:m]) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
        w, U = np.linalg.eigh(Tm)
        return U @ (np.exp(tau * (w - 0.0)) * U[0, :])

    m = m0; y_prev = None
    while True:
        ok = extend(m); mm = len(al); y = coeffs(mm)
        if not ok:                                               # invariant subspace: exact
            info.update(converged=True, m=mm, change=0.0 if y_prev is None else float(np.linalg.norm(y - np.pad(y_prev, (0, mm - y_prev.size))) / np.linalg.norm(y)))
            break
        if y_prev is not None:
            ch = float(np.linalg.norm(y - np.pad(y_prev, (0, mm - y_prev.size))) / np.linalg.norm(y)); info['changes'].append([mm, ch])
            if ch < tol:
                info.update(converged=True, m=mm, change=ch); break
        if m + dm > mmax:
            info.update(converged=False, m=mm, change=info['changes'][-1][1] if info['changes'] else None); break
        y_prev = y; m += dm
    out = 0.0 * Q[0]
    for i in range(y.size): out = out + float(y[i]) * Q[i]
    return beta0 * out, info
