"""Shared definitions for the sign-learnability scaling study (pure numpy; importable without lattice_symmetries).

Clusters (same as results/sign_design_tests/b_depth_N*.json): J1-J2, J2/J1 = 0.5, symmetric sector
(k=0, all point characters trivial, spin flip +).  Site index / bit conventions follow
krylov_sign_structure/experiments/{ed_common,torus_cluster}.py: bit i = spin up on site i.
"""
import os
import numpy as np

CLUSTERS = {
    16: dict(L=4, torus=None),
    20: dict(L=0, torus=(4, 0, 1, 5)),
    24: dict(L=0, torus=(4, 0, 0, 6)),
    28: dict(L=0, torus=(4, 0, 1, 7)),
    32: dict(L=0, torus=(4, 4, -4, 4)),
    36: dict(L=6, torus=None),
}
ED36 = '/home/a/A.Otaifi/chatty_ed6x6/run_gs'


def tvec(N):
    c = CLUSTERS[N]
    if c['torus'] is not None:
        return c['torus']
    L = c['L']
    return (L, 0, 0, L)     # square torus: Cluster((L,0),(0,L)) has site index x + L*y (= ed_common.site)


def geometry(N, k=3):
    """Returns (nbr, point_perms): nbr[i, o] = site at offset o of a k x k window around site i (periodic);
    point_perms[g, i] = image of site i under point operation g (operations fixing site 0 that map the torus
    to itself).  Configuration transform: x'[perm[i]] = x[i]  ->  x' = x[:, argsort(perm)]."""
    import torus_cluster as tc
    t = tvec(N)
    cl = tc.Cluster((t[0], t[1]), (t[2], t[3]))
    assert cl.N == N
    h = k // 2
    offs = [(dx, dy) for dy in range(-h, h + 1) for dx in range(-h, h + 1)]
    nbr = np.array([[cl.red(x + dx, y + dy) for (dx, dy) in offs] for (x, y) in cl.coords], np.int32)
    perms = [cl.perm(f) for f in tc.POINT_OPS if cl._preserves(f)]
    return nbr, np.array(perms, np.int64), cl


def data_dir(base, N):
    return os.path.join(base, f'N{N}')
