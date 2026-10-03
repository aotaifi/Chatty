"""Matched comparison of guides on 6x6 / 8x8 J1-J2 (J2/J1=0.5), 2026-10-03.

Modes
  vmc : variational energy of the optimized ViT alone (complex amplitude, its own
        phases), sampled from |psi|^2 by a bond-exchange Metropolis chain.
        Also reports the binarized-ViT energy (|psi| x sign cos(Im logpsi - phi))
        and the global phase phi / phase leakage used to binarize the ViT sign.
  fn  : lattice fixed-node GFMC, algorithm copied verbatim from
        results/8x8_krylov_3471544/gfmc_8x8_gr_population_size.py and
        results/8x8_marshall_3471545/gfmc_8x8_marshall_control.py
        (discrete-time 1 - tau(H_FN - Eref), tau_max=.025, systematic
        resampling every step, mixed estimator averaged over beta>=burn_beta),
        with guide amplitude |psi_ViT| and sign chosen by --guide:
          marshall : Marshall sign
          krylov   : Marshall * sgn(T - r_M), r_M = (H |psi| M)/(|psi| M), label-free T
          vit      : ViT's own sign, h(x)=sgn cos(Im logpsi(x) - phi)
Usage
  python fn_guides_compare.py vmc L [nchains nround between burn seed]
  python fn_guides_compare.py fn  L guide M seed1 [seed2 ...] [--phi PHI]
"""
import math, time, json, sys, os
import numpy as np
import jax, jax.numpy as jnp
import netket as nk, flax
import netket.jax as nkjax
from nqsmagic.models import ViT
from nqsmagic.utils import _logpsi_transl_2d

MODE = sys.argv[1]
L = int(sys.argv[2]); N = L * L; J2 = .5
if L == 6:
    T = -14.985799779143964  # alpha=1.2 label-free Otsu 5-95 (as gfmc_6x6_gr_population_size.py)
    model = ViT(num_layers=4, d_model=60, heads=10, L_eff=9, b=2, transl_invariant=True, two_dimensional=True)
    CKPT = 'vit_J2=0.50_N=6x6_k=0.mpack'; POOL = 'energy_krylov_vs_vit_6x6_indep.npz'
elif L == 8:
    T = -28.37107876288694  # alpha=1.2 label-free robust kmeans 10-90 (as 3471544)
    model = ViT(num_layers=8, d_model=60, heads=10, L_eff=16, b=2, transl_invariant=True, two_dimensional=True)
    CKPT = 'vit_J2=0.50_N=8x8_k=0.mpack'; POOL = 'krylov_phys8_a2_fixedT.npz'
else:
    raise SystemExit("L must be 6 or 8")

hi = nk.hilbert.Spin(s=.5, N=N)
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True, max_neighbor_order=2)
apply = nkjax.HashablePartial(_logpsi_transl_2d, model.apply, 2)
sam0 = nk.sampler.MetropolisExchange(hi, graph=g, d_max=2, n_chains=16, sweep_size=N)
v0 = nk.vqs.MCState(sampler=sam0, apply_fun=apply, n_samples=16,
                    variables=model.init(jax.random.PRNGKey(1234), jnp.zeros((1, N))), n_discard_per_chain=10)
with open(CKPT, 'rb') as f:
    obj = flax.serialization.msgpack_restore(f.read())
if 'variables' in obj:
    obj = obj['variables']
v0.variables = flax.serialization.from_state_dict(v0.variables, obj)
print("LOADED", L, v0.n_parameters, "T", T, "backend", jax.default_backend(), flush=True)


def evalz(X, batch=4096):
    out = []; X = np.asarray(X, float)
    for i in range(0, len(X), batch):
        out.append(np.asarray(v0.log_value(jnp.asarray(X[i:i + batch]))))
    return np.concatenate(out) if out else np.zeros(0, complex)


def bits2x(ss):
    a = np.asarray(ss, np.uint64).reshape(-1, 1)
    return 2 * ((a >> np.arange(N, dtype=np.uint64)) & 1).astype(float) - 1


def bonds():
    nn = []; nnn = []; q = lambda x, y: (x % L) + L * (y % L)
    for y in range(L):
        for x in range(L):
            i = q(x, y); nn += [(i, q(x + 1, y)), (i, q(x, y + 1))]
            nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
    return nn, nnn


NN, NNN = bonds(); ALL = [(i, j, 1., -1.) for i, j in NN] + [(i, j, J2, 1.) for i, j in NNN]
A_MASK = sum(1 << (x + L * y) for y in range(L) for x in range(L) if (x + y) % 2 == 0)


def marshall(s): return 1 if ((int(s) & A_MASK).bit_count() % 2) == 0 else -1


def diag_energy(s):
    e = 0.
    for i, j, J, mr in ALL: e += J * (.25 if (((s >> i) & 1) == ((s >> j) & 1)) else -.25)
    return e


def neigh(s):
    out = []
    for i, j, J, mr in ALL:
        if ((s >> i) ^ (s >> j)) & 1: out.append((s ^ (1 << i) ^ (1 << j), J, mr))
    return out


# ---------------------------------------------------------------- VMC (E_ViT)
BI = np.array([b[0] for b in ALL], np.uint64); BJ = np.array([b[1] for b in ALL], np.uint64)
BJV = np.array([b[2] for b in ALL], float)


def vmc(nchains=1024, nround=16, between=4, burn=100, seed=20261003):
    rg = np.random.default_rng(seed)
    ss = np.zeros(nchains, np.uint64)
    for k in range(nchains):
        q = 0
        for i in rg.choice(N, N // 2, replace=False): q |= 1 << int(i)
        ss[k] = q
    la = evalz(bits2x(ss)).real
    one = np.uint64(1)
    out = []; acc = 0; props = 0; t0 = time.time()
    nsteps = (burn + between * nround) * N
    for it in range(nsteps):
        b = rg.integers(len(ALL), size=nchains)
        bi = BI[b]; bj = BJ[b]
        ok = (((ss >> bi) ^ (ss >> bj)) & one).astype(bool)
        cand = np.where(ok, ss ^ (one << bi) ^ (one << bj), ss)
        lb = evalz(bits2x(cand)).real
        ac = (np.log(rg.random(nchains)) < np.minimum(0, 2. * (lb - la))) & ok
        acc += int(ac.sum()); props += nchains
        ss = np.where(ac, cand, ss); la = np.where(ac, lb, la)
        sweep = (it + 1) / N
        if (it + 1) % N == 0 and sweep > burn and (sweep - burn) % between == 0:
            out.append(ss.copy())
        if (it + 1) % (10 * N) == 0:
            print("VMC_PROG sweep", sweep, "acc", acc / props, "sec", time.time() - t0, flush=True)
    S = np.stack(out)  # (nround, nchains)
    print("VMC_SAMPLED", S.shape, "accept", acc / props, "sec", time.time() - t0, flush=True)
    flat = S.reshape(-1)
    zx = evalz(bits2x(flat))
    eA = np.array([diag_energy(int(s)) for s in flat], complex)
    eB = eA.real.copy()
    # neighbours, vectorised over bonds
    ys = []; owner = []; jv = []
    for b in range(len(ALL)):
        ok = (((flat >> BI[b]) ^ (flat >> BJ[b])) & one).astype(bool)
        idx = np.nonzero(ok)[0]
        ys.append(flat[idx] ^ (one << BI[b]) ^ (one << BJ[b])); owner.append(idx); jv.append(np.full(len(idx), BJV[b]))
    ys = np.concatenate(ys); owner = np.concatenate(owner); jv = np.concatenate(jv)
    uy, inv = np.unique(ys, return_inverse=True)
    print("VMC_NEIGH occ", len(ys), "unique", len(uy), flush=True)
    zy = np.concatenate([evalz(bits2x(uy[i:i + 65536]), batch=8192) for i in range(0, len(uy), 65536)])
    # global phase for binarisation, |psi|^2-weighted on the samples themselves
    c = np.mean(np.exp(2j * zx.imag)); phi = .5 * float(np.angle(c))
    leak = float(np.mean(np.sin(zx.imag - phi) ** 2))
    hx = np.where(np.cos(zx.imag - phi) >= 0, 1., -1.); hy = np.where(np.cos(zy.imag - phi) >= 0, 1., -1.)
    zyo = zy[inv]
    np.add.at(eA, owner, .5 * jv * np.exp(zyo - zx[owner]))
    np.add.at(eB, owner, .5 * jv * np.exp(zyo.real - zx[owner].real) * hy[inv] * hx[owner])
    res = {}
    for name, v in (("ViT", eA.real), ("ViTbinary", eB)):
        V = v.reshape(S.shape)
        cm = V.mean(axis=0)  # per-chain mean over rounds
        mu = float(V.mean()); se = float(cm.std(ddof=1) / np.sqrt(len(cm)))
        h1 = float(V[:nround // 2].mean()); h2 = float(V[nround // 2:].mean())
        res[name] = dict(E=mu, SE=se, E_site=mu / N, SE_site=se / N, first_half=h1, second_half=h2)
        print("VMC_ENERGY", name, mu, "SE", se, "per_site", mu / N, se / N, "halves", h1, h2, flush=True)
    res["imag_EL_mean"] = float(eA.imag.mean())
    res.update(dict(L=L, N=N, phi=phi, leak=leak, nchains=nchains, nround=nround, between_sweeps=between,
                    burn_sweeps=burn, seed=seed, accept=acc / props, nsamples=int(len(flat))))
    print("VMC_PHASE phi", phi, "leak", leak, flush=True)
    with open(f"vmc_vit_{L}x{L}.json", "w") as f: json.dump(res, f, indent=1)
    np.savez_compressed(f"vmc_vit_{L}x{L}_samples.npz", states=S, eA=eA.real.reshape(S.shape), eB=eB.reshape(S.shape))
    return res


# ---------------------------------------------------------------- GFMC
logc = {}; imc = {}; rc = {}; signc = {}; localc = {}; neval = 0
GUIDE = None; PHI = 0.


def ensure_log(ss):
    global neval
    miss = [int(x) for x in np.unique(np.asarray(ss, np.uint64)) if int(x) not in logc]
    if miss:
        z = evalz(bits2x(np.asarray(miss, np.uint64))); neval += len(miss)
        for x, v in zip(miss, z): logc[x] = float(v.real); imc[x] = float(v.imag)


def ensure_r(ss):  # Krylov sign (identical to 3471544)
    miss = [int(x) for x in np.unique(np.asarray(ss, np.uint64)) if int(x) not in rc]
    if not miss: return
    alln = []; meta = []
    for x in miss:
        ls = neigh(x); meta.append(ls); alln.extend(y for y, J, mr in ls)
    ensure_log(miss + alln)
    for x, ls in zip(miss, meta):
        lx = logc[x]; r = diag_energy(x)
        for y, J, mr in ls: r += .5 * J * mr * math.exp(logc[y] - lx)
        rc[x] = float(r); signc[x] = marshall(x) * (1 if r <= T else -1)


def ensure_sign(ss):
    if GUIDE == 'krylov':
        ensure_r(ss); return
    miss = [int(x) for x in np.unique(np.asarray(ss, np.uint64)) if int(x) not in signc]
    if not miss: return
    ensure_log(miss)
    for x in miss:
        if GUIDE == 'marshall': signc[x] = marshall(x)
        else: signc[x] = 1 if math.cos(imc[x] - PHI) >= 0 else -1


def ensure_local(ss):
    miss = [int(x) for x in np.unique(np.asarray(ss, np.uint64)) if int(x) not in localc]
    if not miss: return
    alln = []; met = []
    for x in miss:
        ls = neigh(x); met.append(ls); alln.extend(y for y, J, mr in ls)
    ensure_sign(miss + alln)
    for x, ls in zip(miss, met):
        lx = logc[x]; sx = signc[x]; d = diag_energy(x); ys = []; rates = []
        for y, J, mr in ls:
            rat = math.exp(logc[y] - lx)
            if sx * signc[y] < 0:
                ys.append(y); rates.append(.5 * J * rat)
            else: d += .5 * J * rat
        rates = np.asarray(rates, float); ys = np.asarray(ys, np.uint64)
        localc[x] = (float(d), float(d - rates.sum()), ys, rates)


def systematic(w, rng, M):
    c = np.cumsum(w); c[-1] = 1
    return np.searchsorted(c, rng.random() / M + np.arange(M) / M, 'right')


def run(M=128, beta_target=1.4, burn_beta=.5, tau_max=.025, seed=777, outfile=None):
    rg = np.random.default_rng(seed)
    pool = np.load(POOL)['states'].astype(np.uint64)
    walkers = pool[rg.choice(len(pool), M, replace=False)].copy()
    beta = 0.; it = 0; Es = []; snaps = []; t0 = time.time()
    while beta < beta_target:
        ensure_local(walkers); dat = [localc[int(x)] for x in walkers]
        diag = np.array([z[0] for z in dat]); el = np.array([z[1] for z in dat]); Eref = float(el.mean())
        mx = max(0., float(np.max(diag - Eref))); tau = min(tau_max, 0.8 / mx if mx > 0 else tau_max)
        nxt = np.empty(M, np.uint64); bw = np.empty(M)
        for k, (x, (d, e, ys, rate)) in enumerate(zip(walkers, dat)):
            stay = 1 - tau * (d - Eref); ws = tau * rate; tot = stay + ws.sum()
            if stay < 0 or tot <= 0: raise RuntimeError(("bad", stay, tot, tau, d, Eref))
            u = rg.random() * tot
            if u < stay: y = x
            else:
                j = np.searchsorted(np.cumsum(ws), u - stay, 'right'); y = int(ys[min(j, len(ys) - 1)])
            nxt[k] = y; bw[k] = tot
        bw /= bw.sum(); walkers = nxt[systematic(bw, rg, M)]
        beta += tau; it += 1
        if beta >= burn_beta:
            ensure_local(walkers); ee = float(np.mean([localc[int(x)][1] for x in walkers]))
            Es.append(ee); snaps.append(walkers.copy())
        if it == 1 or it % 5 == 0:
            print("PROG", it, "beta", beta, "tau", tau, "E", Eref, "uniq", len(np.unique(walkers)),
                  "rc", len(rc), "local", len(localc), "neval", neval, "sec", time.time() - t0, flush=True)
    mixed = np.concatenate(snaps)
    print("RESULT Emean", np.mean(Es), "tail", np.mean(Es[-min(8, len(Es)):]),
          "naiveSE", np.std(Es, ddof=1) / np.sqrt(len(Es)) if len(Es) > 1 else np.nan,
          "nE", len(Es), "mixed", len(mixed), "unique", len(np.unique(mixed)),
          "neval", neval, "sec", time.time() - t0, flush=True)
    np.savez_compressed(outfile, mixed=mixed, Es=np.asarray(Es), threshold=T, guide=GUIDE, phi=PHI)
    return mixed, Es


if __name__ == "__main__":
    if MODE == "vmc":
        a = [int(v) for v in sys.argv[3:]]
        vmc(*a)
    elif MODE == "fn":
        args = sys.argv[3:]
        if "--phi" in args:
            k = args.index("--phi"); PHI = float(args[k + 1]); args = args[:k] + args[k + 2:]
        GUIDE = args[0]; M = int(args[1]); seeds = [int(s) for s in args[2:]]
        BETA = float(os.environ.get("BETA_TARGET", "1.2")); BURN = float(os.environ.get("BURN_BETA", "0.4"))
        assert GUIDE in ("marshall", "krylov", "vit")
        print("FN_SETUP L", L, "guide", GUIDE, "M", M, "seeds", seeds, "beta", BETA, "burn", BURN, "phi", PHI, flush=True)
        rows = []
        for seed in seeds:
            t0 = time.time()
            mix, Es = run(M=M, beta_target=BETA, burn_beta=BURN, tau_max=.025, seed=seed,
                          outfile=f"fn_{GUIDE}_{L}x{L}_M{M}_seed{seed}.npz")
            E = np.asarray(Es, float)
            row = dict(seed=seed, Emean=float(E.mean()), tail8=float(E[-min(8, len(E)):].mean()),
                       naiveSE=float(E.std(ddof=1) / np.sqrt(len(E))), nE=len(E), mixed=len(mix),
                       unique=int(len(np.unique(mix))), sec=time.time() - t0)
            rows.append(row); print("FN_REP", L, GUIDE, M, json.dumps(row), flush=True)
        em = np.array([r["Emean"] for r in rows])
        summ = dict(L=L, N=N, guide=GUIDE, M=M, beta_target=BETA, burn_beta=BURN, tau_max=.025, phi=PHI,
                    T=T if GUIDE == 'krylov' else None, reps=rows, mean=float(em.mean()),
                    between_rep_SE=float(em.std(ddof=1) / np.sqrt(len(em))) if len(em) > 1 else None)
        summ["mean_site"] = summ["mean"] / N
        summ["SE_site"] = summ["between_rep_SE"] / N if summ["between_rep_SE"] is not None else None
        print("FN_SUMMARY", json.dumps(summ), flush=True)
        with open(f"fn_{GUIDE}_{L}x{L}_M{M}_summary.json", "w") as f: json.dump(summ, f, indent=1)
