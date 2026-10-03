"""Low-variance exact variational energy of the one-step-Krylov guide on 6x6 J1-J2.

<H>_G - <H>_ViT is a sum over ordered edges (x,y) with s_x s_y != h_x h_y, i.e. d_x != d_y where
d = s*h (guide sign relative to the binarised ViT sign; ViT phase leakage is 2e-7 so ViT == binary ViT).
By the detailed-balance symmetry p_x*rat_xy = p_y*rat_yx every such unordered edge can be attributed to
its LARGER-amplitude endpoint, which removes the |psi_y|/|psi_x| >> 1 heavy tail of the naive local energy:
  Delta E = E_{x~|psi|^2}[ -2 * sum_{y~x, |psi_y|<|psi_x|, d_y != d_x} J_xy (|psi_y|/|psi_x|) h_x h_y ]
Exactly unbiased, same samples as guide_vmc_energy.py machinery (fn_guides_compare.py imported unchanged).
Usage: python guide_vmc_sym.py nchains nround between burn seed tag [statesfile.npz]
"""
import sys, os, json, time, math
import numpy as np
BASE = os.environ.get("FNGC_DIR", ".")
sys.path.insert(0, BASE)
nchains, nround, between, burn, seed, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
statesfile = sys.argv[7] if len(sys.argv) > 7 else None
sys.argv = ['fn_guides_compare.py', 'fn', '6', 'krylov', '1']
import fn_guides_compare as fgc
fgc.GUIDE = 'krylov'
N = fgc.N; ALL = fgc.ALL; BI = fgc.BI; BJ = fgc.BJ
T_FN = -14.985799779143964; T_OLD = -14.75594475197843
TS = {"T_FN": T_FN, "T_old": T_OLD}
for dt in [-0.5, -0.25, 0.25, 0.5]: TS[f"T_FN{dt:+.2f}"] = T_FN + dt
t0 = time.time()
if statesfile:
    S = np.load(statesfile)['states'].astype(np.uint64); print("LOADED_STATES", S.shape, flush=True)
else:
    rg = np.random.default_rng(seed)
    ss = np.zeros(nchains, np.uint64)
    for k in range(nchains):
        q = 0
        for i in rg.choice(N, N // 2, replace=False): q |= 1 << int(i)
        ss[k] = q
    la = fgc.evalz(fgc.bits2x(ss)).real
    one = np.uint64(1); out = []; acc = 0; props = 0
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
    S = np.stack(out); print("SAMPLED", S.shape, "accept", acc / props, "sec", time.time() - t0, flush=True)
flat = S.reshape(-1)
xs = [int(x) for x in np.unique(flat)]
shell = {}; allY = set()
for x in xs:
    ls = fgc.neigh(x); shell[x] = ls
    for y, J, mr in ls: allY.add(y)
fgc.ensure_log(xs + sorted(allY))
smaller = set()
for x in xs:
    lx = fgc.logc[x]
    for y, J, mr in shell[x]:
        if fgc.logc[y] < lx: smaller.add(y)
fgc.ensure_r(xs + sorted(smaller))
print("R_DONE samples", len(xs), "smaller-shell", len(smaller), "neval", fgc.neval, "sec", time.time() - t0, flush=True)

# global ViT phase (as fgc.vmc), binary sign h, global orientation g so that guide agrees with ViT on the majority
zi = np.array([fgc.imc[int(x)] for x in flat])
phi = .5 * float(np.angle(np.mean(np.exp(2j * zi))))
def hsgn(x): return 1 if math.cos(fgc.imc[x] - phi) >= 0 else -1
def dsgn(x, T, g): return g * fgc.marshall(x) * (1 if fgc.rc[x] <= T else -1) * hsgn(x)
G = {n: (1 if np.mean([fgc.marshall(int(x)) * (1 if fgc.rc[int(x)] <= T else -1) * hsgn(int(x)) for x in flat]) >= 0 else -1) for n, T in TS.items()}
print("PHASE phi", phi, "orient g", G, flush=True)
EL_vit = np.zeros(len(flat)); dE = {n: np.zeros(len(flat)) for n in TS}; Dfrac = {n: 0. for n in TS}
for k, x in enumerate(flat):
    x = int(x); lx = fgc.logc[x]; ix = fgc.imc[x]; e = fgc.diag_energy(x); hx = hsgn(x)
    dx = {n: dsgn(x, TS[n], G[n]) for n in TS}
    for n in TS:
        if dx[n] < 0: Dfrac[n] += 1. / len(flat)
    for y, J, mr in shell[x]:
        rat = math.exp(fgc.logc[y] - lx)
        e += .5 * J * rat * math.cos(fgc.imc[y] - ix)
        if fgc.logc[y] < lx:
            hy = hsgn(y)
            for n in TS:
                if dsgn(y, TS[n], G[n]) != dx[n]: dE[n][k] += -2. * J * rat * hx * hy
    EL_vit[k] = e
def stat(v):
    V = v.reshape(S.shape); cm = V.mean(0); return float(V.mean()), float(cm.std(ddof=1) / np.sqrt(len(cm)))
res = {"seed": seed, "shape": list(S.shape), "N": N, "T": TS, "statesfile": statesfile, "phi": phi, "g": G, "Dfrac": Dfrac}
mu, se = stat(EL_vit); res["ViT"] = dict(E=mu, SE=se)
print("ENERGY ViT", mu, se, "per_site", mu / N, se / N, flush=True)
for n in TS:
    mu, se = stat(dE[n]); res[n] = dict(dE=mu, dSE=se, dE_site=mu / N, dSE_site=se / N, Dfrac=Dfrac[n],
                                          maxabs=float(np.abs(dE[n]).max()), E=res["ViT"]["E"] + mu)
    print("DELTA", n, res[n], flush=True)
json.dump(res, open(f"guide_sym_{tag}.json", "w"), indent=1)
np.savez_compressed(f"guide_sym_{tag}.npz", states=S, ViT=EL_vit.reshape(S.shape), **{"dE_" + n.replace('+', 'p'): dE[n].reshape(S.shape) for n in TS})
