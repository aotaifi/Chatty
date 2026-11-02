"""Step A0.3: precision + throughput of vit10_jax in f32 vs f64 (symm and plain guide).
usage: python -I vit10_prec.py prec OUTDIR [nconf]   |   python -I vit10_prec.py bench OUTDIR
prec: nconf shipped spins (fixed seed) + ALL their distinct one-swap neighbours (J1+J2 bonds, antiparallel), logpsi in
f64 (reference), f32 default matmul precision, f32 matmul precision 'highest'. Saves npz, prints the table."""
import sys, time, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vit10_jax as V, validate_vit10 as VV
f32, f64 = jnp.float32, jnp.float64
def build(kind, dt, prec=None):
    P = V.load(V.SYMM if kind == "symm" else V.PLAIN, dt)
    mk = V.make_symm if kind == "symm" else V.make_plain
    return jax.jit(mk(P, dt=dt, prec=prec))
def run(f, S, dt, bs):
    o = []
    for i, k in enumerate(range(0, len(S), bs)):
        o.append(np.asarray(f(jnp.asarray(S[k:k + bs], dt))))
        if i % 20 == 0: print("PROGRESS", k, len(S), flush=True)
    return np.concatenate(o)
def neighbours(S):
    BI = VV.BI; ii, bb, out = [], [], []   # bb: bond index of the swap
    for a, s in enumerate(S):
        for b in np.nonzero(s[BI[:, 0]] != s[BI[:, 1]])[0]:
            y = s.copy(); y[BI[b, 0]], y[BI[b, 1]] = s[BI[b, 1]], s[BI[b, 0]]; ii.append(a); bb.append(b); out.append(y)
    return np.array(ii), np.array(bb), np.array(out)
def rel(lx, ly, lx0, ly0):
    """relative error of ratio r=psi(y)/psi(x): |r/r0-1| = |exp(d)-1|, d=(ly-lx)-(ly0-lx0) (log branch safe)."""
    d = (ly - lx) - (ly0 - lx0); return np.abs(np.expm1(d))
if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    S = VV.spins().astype(np.float64)
    if mode == "prec":
        n = int(sys.argv[3]); rng = np.random.default_rng(12345)
        held = len(sys.argv) > 4 and sys.argv[4] == "heldout"      # fresh disjoint set: complement of the seed-12345 draw, drawn with seed 777; symm only, f64 + f32h only
        first = np.sort(rng.choice(len(S), n, replace=False))
        if held:
            rest = np.setdiff1d(np.arange(len(S)), first); X = S[np.sort(np.random.default_rng(777).choice(rest, n, replace=False))]
        else: X = S[first]
        ii, bb, Y = neighbours(X)
        print("configs", n, "neighbours", len(Y), "mean/config", len(Y) / n, jax.devices(), flush=True)
        res = {"ii": ii}
        for kind, bs in ((("symm", 512),) if held else (("plain", 2048), ("symm", 512))):
            lp = {}
            for tag, dt, prec in ((("f64", f64, None), ("f32h", f32, "highest")) if held else (("f64", f64, None), ("f32d", f32, None), ("f32h", f32, "highest"))):
                f = build(kind, dt, prec); t = time.time()
                lp[tag] = (run(f, X, dt, bs).astype(np.complex128), run(f, Y, dt, bs).astype(np.complex128))
                print(kind, tag, "%.0fs" % (time.time() - t), flush=True)
                np.savez(f"{out}/prec_{kind}.npz", ii=ii, bb=bb, **{k + "_x": v[0] for k, v in lp.items()}, **{k + "_y": v[1] for k, v in lp.items()})
            ax = np.exp(lp["f64"][0].real); ay = np.exp(lp["f64"][1].real)     # |psi| (f64 reference)
            ox = np.argsort(ax); nx = max(1, n // 100); smallx = np.zeros(n, bool); smallx[ox[:nx]] = True
            smally = ay <= np.quantile(ay, 0.01)
            r0 = np.exp(lp["f64"][1] - lp["f64"][0][ii])
            diag = (VV.BJ * (X[:, VV.BI[:, 0]] * X[:, VV.BI[:, 1]])).sum(1) / 4
            def EL(tag):
                r = np.exp(lp[tag][1] - lp[tag][0][ii]) * VV.BJ[bb] / 2; off = np.zeros(n, complex); np.add.at(off, ii, r); return (diag + off) / 100
            E64 = EL("f64"); print(f"E_L {kind} f64: mean {E64.real.mean():.6f} +- {E64.real.std(ddof=1)/np.sqrt(n):.6f} std/N {E64.real.std():.3e} (n={n})", flush=True)
            for tag in (("f32h",) if held else ("f32d", "f32h")):
                e = rel(lp[tag][0][ii], lp[tag][1], lp["f64"][0][ii], lp["f64"][1])
                r32 = np.exp(lp[tag][1] - lp[tag][0][ii])
                flips = int(np.sum(np.sign(r32.real) != np.sign(r0.real)))
                dE = EL(tag) - E64; print(f"ELDIFF {kind} {tag}: max|dE_L/N| {np.abs(dE).max():.2e} mean dE_L/N (bias) {dE.real.mean():.2e} +- {dE.real.std(ddof=1)/np.sqrt(n):.1e}", flush=True)
                m = lambda a: (a.max(), np.median(a))
                print(f"RESULT {kind} {tag}: all max {m(e)[0]:.3e} med {m(e)[1]:.3e} | small|psi(x)| 1% (n={int(smallx[ii].sum())}) max {e[smallx[ii]].max():.3e} med {np.median(e[smallx[ii]]):.3e} | "
                      f"small|psi(y)| 1% (n={int(smally.sum())}) max {e[smally].max():.3e} med {np.median(e[smally]):.3e} | flips {flips} of {len(e)} | "
                      f"min|Re r0| {np.abs(r0.real).min():.2e} | max|dlogpsi(x)| {np.abs(lp[tag][0]-lp['f64'][0]).max():.2e}", flush=True)
    if mode == "bench":
        res = {}
        for kind in ("plain", "symm"):
            for tag, dt, prec in (("f64", f64, None), ("f32d", f32, None), ("f32h", f32, "highest")):
                f = build(kind, dt, prec)
                for b in (1024, 4096):
                    s = jnp.asarray(S[:b], dt); f(s).block_until_ready(); t = time.time(); r = 5 if kind == "plain" else 3
                    for _ in range(r): o = f(s)
                    o.block_until_ready(); eps = b / ((time.time() - t) / r); res[f"{kind}_{tag}_{b}"] = eps
                    print(f"BENCH {kind} {tag} B={b} {eps:.0f} evals/s", flush=True)
        json.dump(res, open(f"{out}/bench.json", "w")); print(json.dumps(res))
