"""VMC of OUR symm 10x10 ViT guide (vit10_jax): Metropolis-exchange chains (swap of a uniformly chosen antiparallel J1/J2 bond; acceptance
min(1, |psi(y)/psi(x)|^2 n_anti(x)/n_anti(y)), exact detailed balance) and the exact local energy E_L(x)/N (spin-1/2 units, all one-swap
neighbours, as validate_vit10.eloc) every K steps. W independent walkers per chain set.
usage: python vit10_vmc.py OUTDIR --start {shipped|random} --seed S --dtype {f64,f32} [--W 32 --K 100 --burn 1500 --maxsec 10800]
writes OUTDIR/E.npy (round-mean Re E_L/N, time order; quotable.py input), EL.npy (rounds, W), acc.npy, meta.json; progress.txt. Ends 'A03 VMC DONE'."""
import sys, os, time, json, argparse, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vit10_jax as V, validate_vit10 as VV
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--start", default="shipped"); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--dtype", default="f64"); ap.add_argument("--W", type=int, default=32); ap.add_argument("--K", type=int, default=100)
ap.add_argument("--burn", type=int, default=1500); ap.add_argument("--maxsec", type=float, default=10800); ap.add_argument("--maxrounds", type=int, default=10**9)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dt = jnp.float64 if a.dtype == "f64" else jnp.float32; npdt = np.float64 if a.dtype == "f64" else np.float32
f = jax.jit(V.make_symm(V.load(V.SYMM, dt), dt=dt, prec=None if a.dtype == "f64" else "highest"))
BI, BJ = VV.BI, VV.BJ; NB = len(BI); W = a.W; EB = 16 * W; rng = np.random.default_rng(a.seed)
def ev(X, bs):
    n = len(X); out = np.empty(n, np.complex128)
    for k in range(0, n, bs):
        c = X[k:k + bs]; m = len(c)
        if m < bs: c = np.concatenate([c, np.repeat(c[:1], bs - m, 0)])
        out[k:k + m] = np.asarray(f(jnp.asarray(c, dt)))[:m]
    return out
def P(msg):
    l = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'; print(l, flush=True); open(a.out + "/progress.txt", "a").write(l + "\n")
if a.start == "shipped":
    S = VV.spins(); X = S[rng.choice(len(S), W, replace=False)].astype(npdt)
else:
    X = np.array([rng.permutation(np.repeat([1.0, -1.0], 50)) for _ in range(W)], npdt)
lp = ev(X, W); acc_tot = 0; prop_tot = 0; nev = W
def step():
    global X, lp, acc_tot, prop_tot, nev
    anti = X[:, BI[:, 0]] != X[:, BI[:, 1]]; na = anti.sum(1)
    b = np.argmax(rng.random(anti.shape) * anti, 1); r = np.arange(W)
    Y = X.copy(); Y[r, BI[b, 0]], Y[r, BI[b, 1]] = X[r, BI[b, 1]], X[r, BI[b, 0]]
    lq = ev(Y, W); nev += W
    nay = (Y[:, BI[:, 0]] != Y[:, BI[:, 1]]).sum(1)
    ok = np.log(rng.random(W)) < 2 * (lq.real - lp.real) + np.log(na / nay)
    X[ok] = Y[ok]; lp = np.where(ok, lq, lp); acc_tot += ok.sum(); prop_tot += W
def eloc():
    global nev
    anti = X[:, BI[:, 0]] != X[:, BI[:, 1]]; ii, bb = np.nonzero(anti)
    Y = X[ii].copy(); r = np.arange(len(ii)); Y[r, BI[bb, 0]], Y[r, BI[bb, 1]] = X[ii, BI[bb, 1]], X[ii, BI[bb, 0]]
    ly = ev(Y, EB); nev += len(Y)
    off = np.zeros(W, complex); np.add.at(off, ii, np.exp(ly - lp[ii]) * BJ[bb] / 2)
    return ((BJ * (X[:, BI[:, 0]] * X[:, BI[:, 1]])).sum(1) / 4 + off) / 100
t0 = time.time(); P(f"start {a.start} seed {a.seed} {a.dtype} W {W} K {a.K} burn {a.burn} devices {jax.devices()}")
for i in range(a.burn):
    step()
    if i % 100 == 0: P(f"burn {i}/{a.burn} acc {acc_tot / max(prop_tot, 1):.3f} <Re lp> {lp.real.mean():.3f} t {time.time() - t0:.0f}s evals {nev}")
acc_b = acc_tot / max(prop_tot, 1); acc_tot = prop_tot = 0
EL, ACC = [], []; t1 = time.time()
def save():
    el = np.array(EL); np.save(a.out + "/EL.npy.tmp.npy", el); os.replace(a.out + "/EL.npy.tmp.npy", a.out + "/EL.npy")
    np.save(a.out + "/E.npy.tmp.npy", el.real.mean(1)); os.replace(a.out + "/E.npy.tmp.npy", a.out + "/E.npy")
    json.dump(dict(args=vars(a), rounds=len(EL), evals=nev, sec=time.time() - t0, sec_measure=time.time() - t1, acc_burn=acc_b, acc=acc_tot / max(prop_tot, 1), dtype=a.dtype), open(a.out + "/meta.json", "w"))
while time.time() - t0 < a.maxsec and len(EL) < a.maxrounds:
    for _ in range(a.K): step()
    EL.append(eloc())
    if len(EL) % 5 == 0:
        el = np.array(EL).real; P(f"round {len(EL)} E/N {el.mean():.6f} +- {el.mean(1).std(ddof=1) / np.sqrt(len(EL)) if len(EL) > 2 else 0:.1e} acc {acc_tot / prop_tot:.3f} evals {nev} t {time.time() - t0:.0f}s |ImE| {abs(np.array(EL).imag.mean()):.1e}")
    if len(EL) % 10 == 0: save()
save(); P(f"rounds {len(EL)} evals {nev} sec {time.time() - t0:.0f}"); print("A03 VMC DONE", flush=True)
