#!/usr/bin/env python3
"""Compare eigensolvers for the FN Hamiltonian on a checkpointed (v,s) state of a torus run."""
import os, sys, time, json
import numpy as np
import scipy.sparse.linalg as sla
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import closed_fn_krylov_sym6x6 as sym
import torus_cluster

torus = [int(z) for z in sys.argv[1].split(",")]
ck = sys.argv[2]
C = torus_cluster.Cluster((torus[0], torus[1]), (torus[2], torus[3]))
S = sym.Sym(0, 0.5, cluster=C)
if ck.startswith("run:"):
    nit = int(ck[4:])
    v = sym.np.abs(np.load(os.path.join(os.path.dirname(HERE), "results/closed_fn_krylov_6x6/validation/none.npy"))) if False else None
    vinit, _ = sym.init_amplitude(S, 0.0, 1e-12)
    v = vinit / np.linalg.norm(vinit); s = sym.canonical(S.marshall.copy())
    for k in range(nit):
        e_, af, nm_ = S.fn_solve(v, s, 1e-8)
        s, _, _, _ = S.krylov(af, s); v = af
        print("prep it", k + 1, "fn matvecs", nm_, flush=True)
    np.savez(f"/tmp/fnstate_{os.getpid()}.npz", v=v, s=s, it=nit)
else:
    z = np.load(ck)
    v = z["v"]; s = z["s"]
print("min amp", (v / S.sqrt_n).min())
a = np.maximum(v / S.sqrt_n, 1e-15); w = a * S.sqrt_n
Dg = np.empty(S.D); sym.k_fn_diag(S.indptr, S.indices, S.data, s, w, Dg)
print("Dg range", Dg.min(), Dg.max(), "n(Dg>1e3)", int((Dg > 1e3).sum()), "n(Dg>1e6)", int((Dg > 1e6).sum()))
cnt = [0]
def mv(x):
    cnt[0] += 1
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    y = np.empty(S.D); sym.k_fn_matvec(S.indptr, S.indices, S.data, s, Dg, x, y); return y
def mvm(X):
    X = np.asarray(X)
    if X.ndim == 1: return mv(X)
    return np.stack([mv(X[:, i]) for i in range(X.shape[1])], axis=1)
op = sla.LinearOperator((S.D, S.D), matvec=mv, matmat=mvm, dtype=np.float64)

t0 = time.time(); cnt[0] = 0
e, u = sla.eigsh(op, k=1, which="SA", v0=w, tol=1e-10)
print(f"ARPACK ncv=20: E={e[0]:.14f} matvecs={cnt[0]} {time.time()-t0:.1f}s")
E_ref = e[0]; u_ref = np.abs(u[:, 0])
tol = 1e-10
for floor in (1.0,):
    theta0 = E_ref
    den = np.maximum(Dg - theta0, floor)
    for bs in (1, 2):
        X = np.stack([w] + [np.random.default_rng(i).random(S.D) * w for i in range(bs - 1)], axis=1) if bs > 1 else w[:, None].copy()
        Minv = sla.LinearOperator((S.D, S.D), matvec=lambda x: x.reshape(-1) / den, matmat=lambda X: X / den[:, None], dtype=np.float64)
        t0 = time.time(); cnt[0] = 0
        try:
            lam, V, hist = sla.lobpcg(op, X, M=Minv, largest=False, tol=tol * 20, maxiter=2000, retLambdaHistory=True)
            vv = np.abs(V[:, 0]); vv /= np.linalg.norm(vv)
            res = np.linalg.norm(mv(vv) - lam[0] * vv)
            print(f"LOBPCG bs={bs} floor={floor}: E={lam[0]:.14f} dE={lam[0]-E_ref:.2e} matvecs={cnt[0]} iters={len(hist)} resid={res:.2e} vec_err={np.linalg.norm(vv-u_ref):.2e} {time.time()-t0:.1f}s")
        except Exception as ex:
            print("LOBPCG failed", ex)
