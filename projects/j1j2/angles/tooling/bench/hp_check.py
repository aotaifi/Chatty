"""exact-rational truth (Fractions, f converted exactly) for the worst-element configurations: is the float64 brute force or the arm off?"""
import os, sys, json
import numpy as np, jax, jax.numpy as jnp
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_lib import *

def exact_det(M):
    n = len(M); M = [row[:] for row in M]; det = Fr(1)
    for k in range(n):
        piv = next(i for i in range(k, n) if M[i][k] != 0)
        if piv != k: M[k], M[piv] = M[piv], M[k]; det = -det
        det *= M[k][k]
        for i in range(k + 1, n):
            if M[i][k] != 0:
                fct = M[i][k] / M[k][k]
                M[i] = [M[i][j] - fct * M[k][j] if j > k else Fr(0) for j in range(n)]
    return det
def truth(fn, z, L, bond, A):
    z = np.asarray(z); n = len(z)//2
    def det_par(zz):
        order = np.argsort(-zz, kind='stable'); U = order[:n]; D = order[n:]
        M = [[Fr(float(fn[U[i], D[j]])) for j in range(n)] for i in range(n)]
        up = (zz > 0).astype(int); dn = 1 - up
        inv = int(np.sum(up * (np.cumsum(dn) - dn)))
        return exact_det(M) * (-1) ** inv
    bi, bj = int(A.BI[bond]), int(A.BJ[bond])
    y = z.copy(); y[bi], y[bj] = z[bj], z[bi]
    return det_par(y) / det_par(z)
rep = {}
for tag, L, path, seedn, B, arm_keys in [('n32', 8, '/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params_final.npz', 11, 200, None),
                                         ('n50', 10, '/scratch/a/A.Otaifi/ansatz_runs/fastratio/data/params10_random.npz', 12, 200, None)]:
    f = load_f(path, jnp.complex128); A = Arms(L)
    Z = jnp.asarray(random_configs(L*L, B, seedn))
    valid = Z[:, A.BI] != Z[:, A.BJ]
    vm = lambda fn: jax.jit(jax.vmap(fn, (None, 0)))
    bf = vm(A.brute)(f, Z); R = vm(A.R_ratios)(f, Z)[1]; Q = vm(A.Q_fresh)(f, Z)[1]
    d = jnp.where(valid, jnp.abs(R - bf) / jnp.abs(bf), 0.0)
    # take the 3 worst (config,bond) by |R-bf|/|bf| and by |Q-bf|/|bf|
    rows = []
    for nm, X in [('R', R), ('Q', Q)]:
        dd = np.asarray(jnp.where(valid, jnp.abs(X - bf) / jnp.abs(bf), 0.0)).ravel()
        for fl in np.argsort(-dd)[:3]:
            c, b = divmod(int(fl), A.NB)
            t = truth(np.asarray(f.real), Z[c], L, b, A)
            tr = float(t)   # exact rational ratio -> float
            rows.append(dict(arm=nm, config=c, bond=b, truth=tr, bruteforce=float(bf[c, b].real), R=float(R[c, b].real), Q=float(Q[c, b].real),
                             rel_err_brute=abs(float(bf[c, b].real) - tr) / abs(tr), rel_err_R=abs(float(R[c, b].real) - tr) / abs(tr), rel_err_Q=abs(float(Q[c, b].real) - tr) / abs(tr)))
    rep[tag] = rows
    for r in rows: print(tag, {k: (('%.3e' % v) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rep, open(sys.argv[1], 'w'), indent=1)
