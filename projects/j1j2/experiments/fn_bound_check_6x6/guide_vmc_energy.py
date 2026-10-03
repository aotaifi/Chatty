"""Variational energy <H> of the one-step-Krylov guide (|psi_ViT| x Marshall*sgn(T-r_M)) on 6x6 J1-J2.

Uses the machinery of fn_guides_compare.py (imported, not modified): same checkpoint, same
bond-exchange Metropolis sampler from |psi_ViT|^2 (identical to the guide's |psi_G|^2), same r_M
and sign rule as the FN runs.  Local energies on the SAME samples are computed for
  ViT (own complex phase), Marshall, and the Krylov guide for several thresholds T.
Usage: python guide_vmc_energy.py nchains nround between burn seed tag
"""
import sys, os, json, time, math
import numpy as np
BASE = os.environ.get("FNGC_DIR", ".")
sys.path.insert(0, BASE)
nchains, nround, between, burn, seed, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.argv = ['fn_guides_compare.py', 'fn', '6', 'krylov', '1']
import fn_guides_compare as fgc
fgc.GUIDE = 'krylov'
N = fgc.N; ALL = fgc.ALL; BI = fgc.BI; BJ = fgc.BJ
T_FN = -14.985799779143964       # frozen threshold used by all FN runs (gfmc_6x6_gr_population_size.py, fn_guides_compare.py)
T_OLD = -14.75594475197843       # threshold hard-coded in energy_krylov_vs_vit_6x6_indep.py
TS = {"T_FN": T_FN, "T_old": T_OLD}
for k, dt in enumerate([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]): TS[f"T_FN{dt:+.2f}"] = T_FN + dt

# ---- sampling from |psi_ViT|^2 (copied from fgc.vmc)
rg = np.random.default_rng(seed)
ss = np.zeros(nchains, np.uint64)
for k in range(nchains):
    q = 0
    for i in rg.choice(N, N // 2, replace=False): q |= 1 << int(i)
    ss[k] = q
la = fgc.evalz(fgc.bits2x(ss)).real
one = np.uint64(1); out = []; acc = 0; props = 0; t0 = time.time()
for it in range((burn + between * nround) * N):
    b = rg.integers(len(ALL), size=nchains); bi = BI[b]; bj = BJ[b]
    ok = (((ss >> bi) ^ (ss >> bj)) & one).astype(bool)
    cand = np.where(ok, ss ^ (one << bi) ^ (one << bj), ss)
    lb = fgc.evalz(fgc.bits2x(cand)).real
    ac = (np.log(rg.random(nchains)) < np.minimum(0, 2. * (lb - la))) & ok
    acc += int(ac.sum()); props += nchains
    ss = np.where(ac, cand, ss); la = np.where(ac, lb, la)
    sweep = (it + 1) / N
    if (it + 1) % N == 0 and sweep > burn and (sweep - burn) % between == 0: out.append(ss.copy())
    if (it + 1) % (20 * N) == 0: print("VMC_PROG", sweep, acc / props, time.time() - t0, flush=True)
S = np.stack(out)   # (nround, nchains)
flat = S.reshape(-1)
print("SAMPLED", S.shape, "accept", acc / props, "sec", time.time() - t0, flush=True)

# ---- r_M and logs for samples and their first shell
xs = [int(x) for x in np.unique(flat)]
shell = {}
allY = set()
for x in xs:
    ls = fgc.neigh(x); shell[x] = ls
    for y, J, mr in ls: allY.add(y)
fgc.ensure_r(xs + sorted(allY))
print("R_DONE samples", len(xs), "shell", len(allY), "neval", fgc.neval, "sec", time.time() - t0, flush=True)

def sgn(x, T): return fgc.marshall(x) * (1 if fgc.rc[x] <= T else -1)
names = ["ViT", "Marshall"] + list(TS)
EL = {n: np.zeros(len(flat)) for n in names}
for k, x in enumerate(flat):
    x = int(x); lx = fgc.logc[x]; ix = fgc.imc[x]; d = fgc.diag_energy(x)
    acc_ = {n: d for n in names}
    sx = {n: sgn(x, TS[n]) for n in TS}
    for y, J, mr in shell[x]:
        rat = .5 * J * math.exp(fgc.logc[y] - lx)
        acc_["ViT"] += rat * math.cos(fgc.imc[y] - ix)
        acc_["Marshall"] += rat * mr
        for n in TS: acc_[n] += rat * sx[n] * sgn(y, TS[n])
    for n in names: EL[n][k] = acc_[n]
res = {"seed": seed, "nchains": nchains, "nround": nround, "between": between, "burn": burn, "accept": acc / props, "N": N, "T": TS}
for n in names:
    V = EL[n].reshape(S.shape); cm = V.mean(0)
    res[n] = dict(E=float(V.mean()), SE=float(cm.std(ddof=1) / np.sqrt(len(cm))))
    if n != "ViT":
        dcm = (V - EL["ViT"].reshape(S.shape)).mean(0)
        res[n].update(dE_minus_ViT=float(dcm.mean()), dSE=float(dcm.std(ddof=1) / np.sqrt(len(dcm))))
    print("ENERGY", n, res[n], "per_site", res[n]["E"] / N, flush=True)
json.dump(res, open(f"guide_vmc_{tag}.json", "w"), indent=1)
np.savez_compressed(f"guide_vmc_{tag}.npz", states=S, **{n.replace('+', 'p'): EL[n].reshape(S.shape) for n in names})
