"""Validation of vit10_jax: energy on shipped spins, broken variants, symmetries, throughput, neighbour count.
usage: python -I validate_vit10.py {energy|symm|bench} [nsamples]   (spins read via msgpack ext hook, no pickle)"""
import sys, time, json, numpy as np, msgpack, jax, jax.numpy as jnp
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vit10_jax as V
L = 10
def spins():
    hook = lambda c, d: (lambda s, dt, b: np.frombuffer(b, dtype=dt).reshape(s))(*msgpack.unpackb(d)) if c == 1 else msgpack.ExtType(c, d)
    return msgpack.unpackb(open(V.HF + "symm_trxy_ising/spins", "rb").read(), ext_hook=hook, raw=False)
def bonds(pbc=True):
    s = lambda x, y: (x % L) * L + (y % L)
    b1 = [(s(x, y), s(x + dx, y + dy)) for x in range(L) for y in range(L) for dx, dy in ((1, 0), (0, 1))]
    b2 = [(s(x, y), s(x + dx, y + dy)) for x in range(L) for y in range(L) for dx, dy in ((1, 1), (1, -1))]
    return np.array(b1, int), np.array(b2, int)
B1, B2 = bonds(); J1, J2 = 1.0, 0.5
BI = np.concatenate([B1, B2]); BJ = np.array([J1] * len(B1) + [J2] * len(B2))
def eloc(f, S, chunk=64):
    """E_loc/N, spin-1/2 units: H=sum J S.S, S=sigma/2 -> diag J*s_i s_j/4, offdiag (antiparallel swap) J/2 * psi(s')/psi(s)
    (= Pauli-convention E/4). No Marshall sign: the state carries its own sign."""
    out = []; nswap = []
    fj = jax.jit(f)
    for a in range(0, len(S), chunk):
        s = S[a:a + chunk]; n = len(s)
        diag = (BJ * (s[:, BI[:, 0]] * s[:, BI[:, 1]])).sum(1) / 4
        anti = s[:, BI[:, 0]] != s[:, BI[:, 1]]                       # (n,nb)
        nb = anti.sum(1); nswap.append(nb)
        ii, bb = np.nonzero(anti)                                      # all swaps in chunk
        sw = s[ii].copy(); r = np.arange(len(ii))
        sw[r, BI[bb, 0]], sw[r, BI[bb, 1]] = s[ii, BI[bb, 1]], s[ii, BI[bb, 0]]
        lp0 = np.asarray(fj(jnp.asarray(s)))
        lp1 = np.concatenate([np.asarray(fj(jnp.asarray(sw[k:k + 4096]))) for k in range(0, len(sw), 4096)])
        ratio = np.exp(lp1 - lp0[ii]) * BJ[bb] / 2
        off = np.zeros(n, complex); np.add.at(off, ii, ratio)
        out.append((diag + off) / (L * L))
    return np.concatenate(out), np.concatenate(nswap)
def stat(e):
    m = e.real.mean(); se = e.real.std(ddof=1) / np.sqrt(len(e)); return m, se, len(e), abs(e.imag.mean())
if __name__ == "__main__":
    mode = sys.argv[1]; S = spins().astype(np.float64)
    print("jax", jax.__version__, jax.devices(), "spins", S.shape, "mean sz", S.sum(1).mean(), "sz!=0 frac", (S.sum(1) != 0).mean(), flush=True)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(S)
    if mode == "energy":
        Ps = V.load(V.SYMM); Pp = V.load(V.PLAIN)
        cases = {"symm": V.make_symm(Ps), "symm_ROLL_TRANSPOSED": V.make_symm(Ps, transposed=True),
                 "symm_GELU_EXACT": V.make_symm(Ps, gelu_approx=False)}
        for k, f in cases.items():
            nn_ = n if k == "symm" else max(n // 8, 64)
            t = time.time(); e, nb = eloc(f, S[:nn_]); print(k, "E/N= %.6f +- %.6f (n=%d, |Im mean|=%.1e) mean_antiparallel_bonds=%.2f  %.0fs" % (*stat(e), nb.mean(), time.time() - t), flush=True)
        e, nb = eloc(V.make_plain(Pp), S[:n]); print("plain(main weights, symm-branch spins) E/N= %.6f +- %.6f (n=%d, |Im|=%.1e)" % stat(e), flush=True)
    if mode == "symm":
        f = jax.jit(V.make_symm(V.load(V.SYMM))); s = jnp.asarray(S[:32]); g = s.reshape(-1, L, L)
        wr = lambda d: jnp.abs(d.real) + jnp.abs(jnp.angle(jnp.exp(1j * d.imag)))   # log branch: Im defined mod 2pi
        l0 = f(s); tr = f(jnp.roll(jnp.roll(g, 3, 1), 4, 2).reshape(-1, 100)); rot = f(jnp.rot90(g, 1, (1, 2)).reshape(-1, 100)); fl = f(-s)
        print("max|dlogpsi| translation", float(wr(tr - l0).max()), "rot90", float(wr(rot - l0).max()), "flip: Re(dlogpsi)", float(abs(fl.real - l0.real).max()), " complex dlogpsi mod 2pi i (info)", float(wr(fl - l0).max()))
        p = jax.jit(V.make_plain(V.load(V.SYMM))); print("control, plain net under rot90 (should differ):", float(wr(p(jnp.rot90(g, 1, (1, 2)).reshape(-1, 100)) - p(s)).max()))
    if mode == "bench":
        res = {}
        for name, mk, P in (("plain", V.make_plain, V.load(V.PLAIN)), ("symm", V.make_symm, V.load(V.SYMM))):
            f = jax.jit(mk(P))
            for b in (256, 1024, 4096):
                s = jnp.asarray(S[:b]); f(s).block_until_ready(); t = time.time(); r = 5
                for _ in range(r): o = f(s)
                o.block_until_ready(); dt = (time.time() - t) / r; res[f"{name}_{b}"] = b / dt; print(name, b, "%.0f evals/s" % (b / dt), flush=True)
        _, nb = eloc(V.make_plain(V.load(V.PLAIN)), S[:512]); res["n_antiparallel_bonds_mean"] = float(nb.mean()); res["n_antiparallel_bonds_std"] = float(nb.std())
        print(json.dumps(res))
