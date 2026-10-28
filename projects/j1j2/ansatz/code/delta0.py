"""delta0.py: exact one-round opportunity Delta_0 = E_FN[A_0] - E_FN[chi_0] for the complex copy of seed 1, and the same for the target phi_1.
Solver = bond_split.build + the body of bond_split.efn (copied only to also return the eigenvector; same operator, start, preconditioner, tol).
chi_0 = ground state of H_FN[guide s2|A_0|] in the rep basis: vec = chi*sqrt(n)  ->  amplitude per orbit |vec|/sqrt(n).
usage: python3 delta0.py OUTDIR
"""
import os, sys, json, time
import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS

A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
TARGET = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/o2b_round_1/target_la.npy'
SPLIT = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/split/split_s1.json'
T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


def efn_vec(K, Vm, D0, n, la, tol=1e-6, maxit=60):
    sn = np.sqrt(n); a = BS.guide(la, n); w = a * sn
    Dg = D0 + (Vm @ a) / a; N_ = len(a)
    A = LinearOperator((N_, N_), matvec=lambda x: Dg * np.ravel(x) - K @ np.ravel(x), matmat=lambda X: Dg[:, None] * X - K @ X, dtype=np.float64)
    x = (w / np.linalg.norm(w))[:, None]
    th = float(x[:, 0] @ (Dg * x[:, 0] - K @ x[:, 0]))
    den = 1.0 / np.maximum(Dg - th, 1.0)
    M = LinearOperator((N_, N_), matvec=lambda r: den * np.ravel(r), matmat=lambda Rr: den[:, None] * Rr, dtype=np.float64)
    val, vec, hist = lobpcg(A, x, M=M, largest=False, tol=tol, maxiter=maxit, retResidualNormsHistory=True)
    e = float(val[0])
    return dict(E=e, dE_site=(e - BS.E0) / BS.NS, E_guide_var=th, iters=len(hist), res=float(np.ravel(hist[-1])[0])), vec[:, 0]


def main():
    OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64)
    assert n.sum() == 9075135300 and set(np.unique(s)) <= {-1.0, 1.0}
    res = dict(A0=A0, target=TARGET, E0=BS.E0, NS=BS.NS)
    def dump(): res['sec'] = time.time() - T0; json.dump(res, open(f'{OUT}/delta0.json', 'w'), indent=1, default=float)
    K, Vm, D0 = BS.build(n, s); log(f'built K nnz {K.nnz}')
    ref = json.load(open(SPLIT))
    for tag, path, refd in (('copy_s1', A0, ref['E_FN_copy']), ('target', TARGET, ref['E_FN_target'])):
        la = np.load(path).astype(np.float64)
        t = time.time(); r1, v = efn_vec(K, Vm, D0, n, la); r1['sec'] = time.time() - t
        rel = (r1['E'] - refd['E']) / abs(refd['E'])
        res[tag] = dict(first=r1, ref_E=refd['E'], ref_dE_site=refd['dE_site'], rel_E=rel, rel_dE_site=(r1['dE_site'] - refd['dE_site']) / refd['dE_site'], control_pass=bool(abs(rel) <= 1e-9))
        log(f'{tag} first: dE_site {r1["dE_site"]:.12e} ref {refd["dE_site"]:.12e} rel {rel:.2e} res {r1["res"]:.1e} it {r1["iters"]} {r1["sec"]:.0f}s'); dump()
        if abs(rel) > 1e-9: log('CONTROL FAIL: STOP'); res['status'] = 'CONTROL FAIL ' + tag; dump(); sys.exit(2)
        chi = np.abs(v) / np.sqrt(n); res[tag]['chi_min_over_max'] = float(chi.min() / chi.max()); res[tag]['vec_sign_frac_neg'] = float((v < 0).sum() / v.size) if v.sum() > 0 else float((v > 0).sum() / v.size)
        lc = np.log(np.maximum(chi, 1e-300))
        t = time.time(); r2, _ = efn_vec(K, Vm, D0, n, lc); r2['sec'] = time.time() - t
        res[tag]['second'] = r2; res[tag]['Delta0_site'] = (r1['E'] - r2['E']) / BS.NS; res[tag]['Delta0_total'] = r1['E'] - r2['E']
        log(f'{tag} second: dE_site {r2["dE_site"]:.12e} res {r2["res"]:.1e} it {r2["iters"]} {r2["sec"]:.0f}s  Delta0/site {res[tag]["Delta0_site"]:.6e}'); dump()
    res['status'] = 'DONE'; dump(); log('DONE')

if __name__ == '__main__': main()
