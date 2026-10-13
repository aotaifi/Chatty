"""Gate V1 (PREREG E2): dc_fn.py with a 4x4 lattice and table guides must reproduce the exact lattice FN energy.
Guides (Sz = 0 sector, 12870 states): G1 = |psi0| exp(0.3 xi) with the exact sign (FN energy above E0: the violating-edge terms use the noisy amplitude);
G2 = same amplitude, sign of 10% of the configurations flipped (FN energy > E0).  Exact FN: lowest eigenvalue of H_FN.
  python dc_fn_gate4x4.py OUT.json [npop] [beta_target]
Amendment 1 (2026-10-13, after the first gate run, see PREREG): these toy guides are far from the ground state, so the
mixed estimator still carries a projection transient in the [0.8, 2.4] window (exact: +1.2e-4 / +3.2e-4 per site). The
gate therefore compares (a) the late window [4, 8] with E_FN + the exact transient over that window and (b) the [0.8, 2.4]
window with the exact transient average (continuous-time e^{-beta H_FN} from the psi_T^2 start).
"""
from scipy.sparse.linalg import expm_multiply
import sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import dc_fn as F

lat = F.Lattice(4); N = 16
states = np.array([s for s in range(1 << N) if bin(s).count('1') == N // 2], np.uint64)
D = len(states)
NBs, V = lat.neighbors(states)
rows = np.repeat(np.arange(D), lat.NB)[V.reshape(-1)]
cols = np.searchsorted(states, NBs[V])
vals = 0.5 * np.broadcast_to(lat.JB, (D, lat.NB))[V]
H = sp.csr_matrix((vals, (rows, cols)), shape=(D, D)) + sp.diags(lat.diag(V))
w, v = sla.eigsh(H, k=1, which='SA'); E0 = w[0]; psi0 = v[:, 0]
print('E0/N', E0 / N)
rg = np.random.default_rng(3)
la = np.log(np.abs(psi0)) + 0.3 * rg.standard_normal(D)
s1 = np.sign(psi0); s2 = s1.copy(); fl = rg.random(D) < 0.10; s2[fl] *= -1


def hfn(la, s):
    Hc = H.tocoo(); off = Hc.row != Hc.col
    r, c, h = Hc.row[off], Hc.col[off], Hc.data[off]
    kept = s[r] * s[c] < 0
    vd = np.bincount(r[~kept], weights=h[~kept] * np.exp(la[c[~kept]] - la[r[~kept]]), minlength=D)
    return sp.csr_matrix((h[kept], (r[kept], c[kept])), shape=(D, D)) + sp.diags(lat.diag(V) + vd)


def transient(la, s, w0, w1, n=161):
    Hf = hfn(la, s); pT = s * np.exp(la)
    ph = expm_multiply(-Hf, pT, start=w0, stop=w1, num=n, endpoint=True)
    return float(np.mean([(pT @ (Hf @ p)) / (pT @ p) for p in ph]) / N)


def exact_fn(la, s):
    Hc = H.tocoo(); off = Hc.row != Hc.col
    r, c, h = Hc.row[off], Hc.col[off], Hc.data[off]
    kept = s[r] * s[c] < 0
    viol_diag = np.bincount(r[~kept], weights=h[~kept] * np.exp(la[c[~kept]] - la[r[~kept]]), minlength=D)
    Hf = sp.csr_matrix((h[kept], (r[kept], c[kept])), shape=(D, D)) + sp.diags(lat.diag(V) + viol_diag)
    return sla.eigsh(Hf, k=1, which='SA')[0][0]


npop = int(sys.argv[2]) if len(sys.argv) > 2 else 128
BT = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
res = dict(E0_site=E0 / N)
for name, s in (('G1_exact_sign', s1), ('G2_10pct_flips', s2)):
    Efn = exact_fn(la, s)
    guide = lambda S, s=s: (la[np.searchsorted(states, S)], s[np.searchsorted(states, S)])
    pool = states[rg.choice(D, 4096, p=np.exp(2 * la) / np.exp(2 * la).sum())]
    t = time.time(); Es, Ts = [], []
    for b in range(npop // 32):
        E, T, Nd, nev = F.run_batch(lat, guide, pool, 32, 512, np.random.default_rng([11, b]), beta_target=BT)
        Es.append(E); Ts.append(T)
    S = max(e.shape[1] for e in Es)
    E = np.concatenate([np.pad(e, ((0, 0), (0, S - e.shape[1]))) for e in Es]); T = np.concatenate([np.pad(e, ((0, 0), (0, S - e.shape[1]))) for e in Ts])
    out = dict(E_FN_exact_site=Efn / N, npop=len(E), sec=time.time() - t)
    for (w0, w1) in ((0.8, 2.4), (4.0, 8.0)):
        if w1 > BT: continue
        pm = F.pop_means(E, T, w0, w1, N); se = pm.std(ddof=1) / np.sqrt(len(pm)); ref = transient(la, s, w0, w1)
        out[f'w{w0}-{w1}'] = dict(E_dmc_site=float(pm.mean()), se=float(se), exact_transient_avg=ref, diff=float(pm.mean() - ref),
                                  z=float((pm.mean() - ref) / se), sd_pop=float(pm.std(ddof=1)), PASS=bool(abs(pm.mean() - ref) < 3 * se and se <= 2e-5))
    res[name] = out
    print(name, json.dumps(res[name]))
json.dump(res, open(sys.argv[1], 'w'), indent=1)
