#!/usr/bin/env python3
"""Check that the k=0, trivial-point-irrep, spin-flip-even sector contains the lowest state of H(J2) for a
tilted torus, by comparing with ED in coarser symmetry sectors (or full S^z=0 ED when small enough).
usage: check_gs_sector_torus.py t1x,t1y,t2x,t2y [J2] [mode]   mode: full | k0 | scan"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lattice_symmetries as ls
import closed_fn_krylov_sym6x6 as sym
from torus_cluster import Cluster

tv = [int(z) for z in sys.argv[1].split(",")]
J2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
mode = sys.argv[3] if len(sys.argv) > 3 else "k0"
C = Cluster((tv[0], tv[1]), (tv[2], tv[3]))

def lowest(basis, nev=1, tol=1e-10):
    basis.build()
    op = sym.make_op(basis, 0, J2, C)
    ip, ix, da = sym.build_csr(basis, op)
    lop, cnt = sym.csr_op(ip, ix, da)
    D = basis.number_states
    if D < 300:
        # dense for tiny sectors
        A = np.stack([lop.matvec(np.eye(D)[:, i]) for i in range(D)], axis=1)
        return D, float(np.linalg.eigvalsh(0.5 * (A + A.T))[0])
    ew = sym.sla.eigsh(lop, k=nev, which="SA", tol=tol, return_eigenvectors=False)
    return D, float(np.min(ew))

res = {}
b, G = C.make_basis(k=(0, 0), flip=1)
res["k0 point-trivial flip+"] = lowest(b)
if mode == "full":
    b = ls.SpinBasis(ls.Group([]), number_spins=C.N, hamming_weight=C.N // 2)
    res["FULL Sz=0 (no symmetry)"] = lowest(b)
if mode in ("k0", "scan"):
    # k=0: translations only, flip +/-, no point group
    for flip in (1, -1):
        syms = [ls.Symmetry(list(C.gens["Tx"]), sector=0), ls.Symmetry(list(C.gens["Ty"]), sector=0)]
        b = ls.SpinBasis(ls.Group(syms), number_spins=C.N, hamming_weight=C.N // 2, spin_inversion=flip)
        res[f"k0 translations only flip{flip:+d}"] = lowest(b)
if mode == "scan":
    px, py = C.order(C.gens["Tx"]), C.order(C.gens["Ty"])
    for kx in range(px):
        for ky in range(py):
            if (kx, ky) == (0, 0): continue
            # momentum characters must be real for flip symmetry use; only scan real ones (k=0 or pi)
            if (2 * kx) % px or (2 * ky) % py: continue
            syms = [ls.Symmetry(list(C.gens["Tx"]), sector=kx), ls.Symmetry(list(C.gens["Ty"]), sector=ky)]
            for flip in (1, -1):
                try:
                    b = ls.SpinBasis(ls.Group(syms), number_spins=C.N, hamming_weight=C.N // 2, spin_inversion=flip)
                    res[f"k=({kx}/{px},{ky}/{py}) transl only flip{flip:+d}"] = lowest(b)
                except Exception as e:
                    res[f"k=({kx},{ky}) flip{flip}"] = ("err", str(e)[:60])
for k, v in res.items():
    print(f"{k:45s} dim={v[0]} E0={v[1]}")
