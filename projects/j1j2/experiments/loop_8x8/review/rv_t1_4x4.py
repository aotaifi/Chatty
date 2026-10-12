"""Review T1 (REVIEW_lanczos8x8.md): exact test of the Lanczos-step formulas on the 4x4 PBC J1-J2 lattice (J2 = 0.5).
The l8_core functions (bonds, neighbors, diag_vec, onehop) are rebuilt with L = 4 from the unmodified l8_core.py source;
the E_L2 / E_L1 lines are copied verbatim from l8_direct.py, the Ritz moments from l8_eval.py.  The Hamiltonian used as
the exact reference is built independently here (bit operations, coordinates).  Toy state: exact ground state times
exp(0.4 * gaussian) (its own sign kept), so that one Lanczos step is a non-trivial change.
Also validates the delayed-acceptance Metropolis sampler of rv_direct.py against exact |psi_1|^2.
  python rv_t1_4x4.py OUT.json        (CPU is enough)
"""
import os, sys, json, types, itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = next(p for p in (os.path.join(HERE, 'l8_core.py'), os.path.join(HERE, '..', 'l8_core.py')) if os.path.exists(p))
src = open(CORE).read()
assert 'L = 8; N = 64; J2 = 0.5' in src
C = types.ModuleType('l4_core'); C.__file__ = CORE
exec(compile(src.replace('L = 8; N = 64; J2 = 0.5', 'L = 4; N = 16; J2 = 0.5'), CORE, 'exec'), C.__dict__)
L = 4; N = 16; OUT = sys.argv[1]; res = {}

# ---------------------------------------------------------------- independent exact H (Sz = 0 sector)
basis = np.array([sum(1 << i for i in c) for c in itertools.combinations(range(N), N // 2)], np.uint64)
idx = {int(s): k for k, s in enumerate(basis)}
bonds = set()
for y in range(L):
    for x in range(L):
        for dx, dy, J in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 0.5), (1, -1, 0.5)):
            i = x + L * y; j = (x + dx) % L + L * ((y + dy) % L); bonds.add((min(i, j), max(i, j), J))
bonds = sorted(bonds); assert len(bonds) == 4 * N
rows, cols, vals = [], [], []
for k, s in enumerate(basis):
    s = int(s); d = 0.0
    for i, j, J in bonds:
        if ((s >> i) & 1) == ((s >> j) & 1): d += 0.25 * J
        else:
            d -= 0.25 * J; rows.append(k); cols.append(idx[s ^ (1 << i) ^ (1 << j)]); vals.append(0.5 * J)
    rows.append(k); cols.append(k); vals.append(d)
H = sp.csr_matrix((vals, (rows, cols)), shape=(len(basis), len(basis)))
e0, v0 = sla.eigsh(H, k=1, which='SA'); e0 = float(e0[0]); v0 = v0[:, 0]
res['E0_site'] = e0 / N
rg = np.random.default_rng(7)
psi = v0 * np.exp(0.4 * rg.standard_normal(len(v0)))
psi /= np.linalg.norm(psi)
Hpsi = H @ psi
h1 = float(psi @ Hpsi)
res['toy_E_site'] = h1 / N


class Toy:
    """guide interface of l8_core: eval(S) -> (log|psi|, phase), sign(phase)."""
    phi = 0.0; neval = 0

    def eval(self, S):
        k = np.array([idx[int(s)] for s in S]); a = psi[k]
        return np.log(np.abs(a)), np.where(a >= 0, 0.0, np.pi)

    def sign(self, ph):
        return np.where(np.cos(ph - self.phi) >= 0, 1.0, -1.0)


g = Toy()
X = basis.copy(); R = len(X)
# ---- verbatim from l8_direct.py (two-hop local energies) ----
EL = np.empty(R); EL2 = np.empty(R); ch = 64
for i in range(0, R, ch):
    Sx = X[i:i + ch]; n = len(Sx)
    NBs, V = C.neighbors(Sx); o, bb = np.nonzero(V); ys = NBs[o, bb]
    Yu = np.unique(np.concatenate([Sx, ys])); d = C.onehop(g, Yu)
    px = np.searchsorted(Yu, Sx); py = np.searchsorted(Yu, ys)
    h = 0.5 * C.JB[bb] * d['s'][px][o] * d['s'][py] * np.exp(np.clip(d['la'][py] - d['la'][px][o], -60, 60))
    EL[i:i + n] = d['ELa'][px]
    EL2[i:i + n] = d['D'][px] * d['ELa'][px] + np.bincount(o, weights=h * d['ELa'][py], minlength=n)
# ------------------------------------------------------------
H2psi = H @ Hpsi
res['EL_maxdiff'] = float(np.max(np.abs(EL - Hpsi / psi)))
res['EL2_maxdiff'] = float(np.max(np.abs(EL2 - H2psi / psi)))
p2 = psi ** 2
res['E_from_EL_minus_exact'] = float(np.sum(p2 * EL) - h1)

# Lanczos state with fixed alpha, E_ref (as FU1c) -> exact vs estimator formula of l8_direct (weighted by |psi_1|^2)
m2_ex = float(Hpsi @ Hpsi - h1 ** 2)
for al, Er in ((-0.3, h1), (-1.0, h1 + 0.1), (-2.5, h1 - 0.3)):
    psi1 = (1 - al * Er) * psi + al * Hpsi
    E1_ex = float(psi1 @ (H @ psi1) / (psi1 @ psi1))
    eps = EL - Er
    EL1 = ((1 - al * Er) * EL + al * EL2) / (1 + al * eps)            # verbatim l8_direct
    w1 = (psi * (1 + al * eps)) ** 2
    E1_est = float(np.sum(w1 * EL1) / np.sum(w1))
    # reweighting from psi^2 as in l8_eval alpha scan
    w_ = (1 + al * eps) ** 2
    E1_rw = float(np.sum(p2 * w_ * EL1) / np.sum(p2 * w_))
    res.setdefault('fixed_alpha', []).append(dict(alpha=al, Eref=Er, E1_exact=E1_ex, E1_direct_formula=E1_est, E1_reweight=E1_rw,
                                                  rel_err=abs(E1_est - E1_ex) / abs(E1_ex), min_factor=float((1 + al * eps).min())))

# Ritz: exact 2x2 generalized eigenproblem in span{psi, H psi} vs the centred-moment formula of l8_eval (weighted means)
B = np.stack([psi, Hpsi], 1); S_ = B.T @ B; M_ = B.T @ (H @ B)
import scipy.linalg as la_
ritz_ex = float(np.min(la_.eigh(M_, S_, eigvals_only=True)))
mean = lambda v: float(np.sum(p2 * v))
hh = mean(EL); m2 = mean((EL - hh) ** 2); m3 = mean((EL - hh) * (EL2 - 2 * hh * EL + hh * hh))
c = m3 / (2 * m2); ritz_mom = hh + c - np.sqrt(c * c + m2)
# symmetrised 2-state version of l8_eval.ritz
Ea = hh; G = [np.ones(R), EL - Ea]; HG = [EL - Ea, (EL2 - Ea * EL) - Ea * (EL - Ea)]
S2 = np.array([[mean(G[i] * G[j]) for j in range(2)] for i in range(2)]); M2 = np.array([[mean(G[i] * HG[j]) for j in range(2)] for i in range(2)])
M2 = 0.5 * (M2 + M2.T); ritz_sym = Ea + float(np.min(np.linalg.eigvals(np.linalg.solve(S2, M2)).real))
beta = (ritz_mom - hh) / m2
psib = psi + beta * (Hpsi - hh * psi); E_beta = float(psib @ (H @ psib) / (psib @ psib))
res['ritz'] = dict(exact=ritz_ex, moments=float(ritz_mom), symmetrised=ritz_sym, m2_exact=m2_ex, m2_formula=m2,
                   rel_err_moments=abs(ritz_mom - ritz_ex) / abs(ritz_ex), rel_err_sym=abs(ritz_sym - ritz_ex) / abs(ritz_ex),
                   E_at_beta=E_beta, beta=beta)

# ---------------------------------------------------------------- delayed-acceptance sampler of rv_direct.py vs exact |psi_1|^2
sys.path.insert(0, HERE)
import rv_da


class PlainMH(rv_da.DAChain):
    """reference: ordinary Metropolis on |psi_1|^2 with the full ratio (one-hop for every candidate)."""
    def run(self, nsteps):
        C_ = self.C; n = len(self.S)
        for _ in range(nsteps):
            b = self.rg.integers(C_.NB_, size=n)
            ok = (((self.S >> C_.BI[b]) ^ (self.S >> C_.BJ[b])) & C_.ONE).astype(bool)
            cand = self.S ^ C_.MASKS[b]; ii = np.nonzero(ok)[0]
            la2, lf2, el2 = self.onehop_f(cand[ii])
            acc = np.log(self.rg.random(len(ii))) < 2 * (la2 + lf2 - self.la[ii] - self.lf[ii])
            jj = ii[acc]; self.S[jj] = cand[jj]; self.la[jj] = la2[acc]; self.lf[jj] = lf2[acc]; self.el[jj] = el2[acc]
            self.acc1 += int(acc.sum()); self.acc2 += int(acc.sum()); self.prop += n


for tag, al, Er in (('realistic', res['ritz']['beta'], h1), ('nodal', -1.0, h1 + 0.1)):
    psi1 = (1 - al * Er) * psi + al * Hpsi; P1 = psi1 ** 2 / np.sum(psi1 ** 2)
    E1_ex = float(psi1 @ (H @ psi1) / (psi1 @ psi1))
    eps = EL - Er; EL1 = ((1 - al * Er) * EL + al * EL2) / (1 + al * eps)

    def onehop_f(S, al=al, Er=Er):
        d = C.onehop(g, S); return d['la'], np.log(np.abs(1 + al * (d['ELa'] - Er))), d['ELa']

    out = {}
    for name, cls in (('DA', rv_da.DAChain), ('plainMH', PlainMH)):
        rgc = np.random.default_rng(11 if name == 'DA' else 12)
        S0 = basis[rgc.choice(R, 4096, p=p2 / p2.sum())]
        st = cls(C, S0, lambda S: g.eval(S)[0], onehop_f, rgc)
        st.run(N * 10)
        cnt = np.zeros(R); ests = []
        for t in range(100):
            st.run(N * 3)
            k = np.array([idx[int(s)] for s in st.S]); np.add.at(cnt, k, 1); ests.append(EL1[k].mean())
        emp = cnt / cnt.sum(); ests = np.array(ests)
        out[name] = dict(TV=float(0.5 * np.abs(emp - P1).sum()), E1_mc=float(ests.mean()),
                         E1_se=float(ests.std(ddof=1) / np.sqrt(len(ests))), accept=st.acc2 / st.prop)
    rgn = np.random.default_rng(3)
    out['TV_iid_reference'] = float(np.mean([0.5 * np.abs(rgn.multinomial(int(cnt.sum()), P1) / cnt.sum() - P1).sum() for _ in range(5)]))
    out.update(alpha=al, E1_exact=E1_ex, min_factor=float((1 + al * eps).min()))
    res['sampler_' + tag] = out
json.dump(res, open(OUT, 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k.startswith('sampler')}, indent=1, default=float))
