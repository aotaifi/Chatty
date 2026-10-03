import argparse
import importlib.util
import math
import time
import numpy as np


class CachedAmplitude:
    """Cache a batched callable returning log positive amplitudes for uint64 states."""
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        self.neval = 0

    def ensure(self, states):
        uu = np.unique(np.asarray(states, dtype=np.uint64))
        miss = [int(x) for x in uu if int(x) not in self.cache]
        if not miss:
            return
        vals = np.asarray(self.fn(np.asarray(miss, dtype=np.uint64)), dtype=float).reshape(-1)
        if len(vals) != len(miss) or not np.all(np.isfinite(vals)):
            raise ValueError("amplitude backend returned wrong shape or non-finite log amplitudes")
        for x, v in zip(miss, vals):
            self.cache[x] = float(v)
        self.neval += len(miss)

    def loga(self, x):
        x = int(x)
        self.ensure([x])
        return self.cache[x]


class SquareJ1J2:
    def __init__(self, L, J2=0.5):
        self.L = int(L)
        self.N = self.L * self.L
        self.J2 = float(J2)
        q = lambda x, y: (x % self.L) + self.L * (y % self.L)
        nn, nnn = [], []
        for y in range(self.L):
            for x in range(self.L):
                i = q(x, y)
                nn += [(i, q(x + 1, y)), (i, q(x, y + 1))]
                nnn += [(i, q(x + 1, y + 1)), (i, q(x + 1, y - 1))]
        self.bonds = [(i, j, 1.0, -1.0) for i, j in nn]
        self.bonds += [(i, j, self.J2, 1.0) for i, j in nnn]
        self.A_mask = sum(
            1 << (x + self.L * y)
            for y in range(self.L) for x in range(self.L)
            if (x + y) % 2 == 0
        )

    def marshall(self, s):
        return 1 if ((int(s) & self.A_mask).bit_count() % 2) == 0 else -1

    def diag_energy(self, s):
        s = int(s)
        e = 0.0
        for i, j, J, _ in self.bonds:
            e += J * (0.25 if (((s >> i) & 1) == ((s >> j) & 1)) else -0.25)
        return e


    def neigh(self, s):
        s = int(s)
        out = []
        for i, j, J, mr in self.bonds:
            if ((s >> i) ^ (s >> j)) & 1:
                out.append((s ^ (1 << i) ^ (1 << j), J, mr))
        return out


def weighted_quantile(x, w, p):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    o = np.argsort(x)
    xx, ww = x[o], w[o]
    c = np.cumsum(ww)
    if c[-1] <= 0:
        raise ValueError("non-positive total threshold weight")
    c /= c[-1]
    return float(np.interp(p, c, xx))


def robust_two_means_threshold(r, w=None, qlo=0.10, qhi=0.90):
    r = np.asarray(r, float)
    w = np.ones(len(r), float) if w is None else np.asarray(w, float)
    w = w / w.sum()
    lo = weighted_quantile(r, w, qlo)
    hi = weighted_quantile(r, w, qhi)
    a = np.clip(r, lo, hi)
    m1 = weighted_quantile(a, w, 0.30)
    m2 = weighted_quantile(a, w, 0.80)


    for _ in range(100):
        cut = 0.5 * (m1 + m2)
        lab = a > cut
        if lab.all() or (~lab).all():
            break
        n1 = float(np.sum(w[~lab] * a[~lab]) / np.sum(w[~lab]))
        n2 = float(np.sum(w[lab] * a[lab]) / np.sum(w[lab]))
        if abs(n1 - m1) + abs(n2 - m2) < 1e-10:
            m1, m2 = n1, n2
            break
        m1, m2 = n1, n2
    return 0.5 * (m1 + m2), (m1, m2, lo, hi)


class K1FNEngine:
    def __init__(self, lattice, amplitude):
        self.lat = lattice
        self.amp = amplitude
        self.threshold = None
        self.r_cache = {}
        self.sign_cache = {}
        self.local_cache = {}

    def clear_derived(self):
        self.r_cache.clear()
        self.sign_cache.clear()
        self.local_cache.clear()

    def ensure_r(self, states):
        miss = [int(x) for x in np.unique(np.asarray(states, np.uint64))
                if int(x) not in self.r_cache]
        if not miss:
            return
        meta, alln = [], []


        for x in miss:
            ls = self.lat.neigh(x)
            meta.append(ls)
            alln.extend(y for y, _, _ in ls)
        self.amp.ensure(miss + alln)
        for x, ls in zip(miss, meta):
            lx = self.amp.loga(x)
            r = self.lat.diag_energy(x)
            for y, J, mr in ls:
                r += 0.5 * J * mr * math.exp(self.amp.loga(y) - lx)
            self.r_cache[x] = float(r)

    def fit_threshold(self, states, weights=None, qlo=0.10, qhi=0.90):
        states = np.asarray(states, np.uint64)
        self.ensure_r(states)
        rr = np.asarray([self.r_cache[int(x)] for x in states], float)
        T, info = robust_two_means_threshold(rr, weights, qlo=qlo, qhi=qhi)
        self.threshold = float(T)
        self.sign_cache.clear()
        self.local_cache.clear()
        return self.threshold, rr, info

    def ensure_sign(self, states):
        if self.threshold is None:
            raise RuntimeError("fit/set threshold before constructing K1 signs")
        states = np.unique(np.asarray(states, np.uint64))
        self.ensure_r(states)
        for x in states:
            xx = int(x)
            if xx not in self.sign_cache:
                self.sign_cache[xx] = self.lat.marshall(xx) * (
                    1 if self.r_cache[xx] <= self.threshold else -1
                )


    def ensure_local(self, states):
        miss = [int(x) for x in np.unique(np.asarray(states, np.uint64))
                if int(x) not in self.local_cache]
        if not miss:
            return
        meta, alln = [], []
        for x in miss:
            ls = self.lat.neigh(x)
            meta.append(ls)
            alln.extend(y for y, _, _ in ls)
        self.ensure_sign(miss + alln)
        for x, ls in zip(miss, meta):
            lx = self.amp.loga(x)
            sx = self.sign_cache[x]
            d = self.lat.diag_energy(x)
            ys, rates = [], []
            for y, J, _ in ls:
                rat = math.exp(self.amp.loga(y) - lx)
                if sx * self.sign_cache[y] < 0:
                    ys.append(y)
                    rates.append(0.5 * J * rat)
                else:
                    d += 0.5 * J * rat
            rates = np.asarray(rates, float)
            self.local_cache[x] = (
                float(d), float(d - rates.sum()),
                np.asarray(ys, np.uint64), rates,
            )

    @staticmethod
    def _systematic(w, rng, M):
        c = np.cumsum(w)
        c[-1] = 1.0
        return np.searchsorted(c, rng.random() / M + np.arange(M) / M, "right")


    def run_fn(self, pool, M=128, beta_target=1.2, burn_beta=0.4,
               tau_max=0.025, seed=1, pool_weights=None, progress_every=5):
        rng = np.random.default_rng(seed)
        pool = np.asarray(pool, np.uint64)
        p = None
        if pool_weights is not None:
            p = np.asarray(pool_weights, float)
            p = p / p.sum()
        replace = M > len(pool)
        walkers = pool[rng.choice(len(pool), M, replace=replace, p=p)].copy()
        beta, it, Es, snaps = 0.0, 0, [], []
        t0 = time.time()
        while beta < beta_target:
            self.ensure_local(walkers)
            dat = [self.local_cache[int(x)] for x in walkers]
            diag = np.asarray([z[0] for z in dat])
            eloc = np.asarray([z[1] for z in dat])
            Eref = float(eloc.mean())
            mx = max(0.0, float(np.max(diag - Eref)))
            tau = min(tau_max, 0.8 / mx if mx > 0 else tau_max)
            nxt, bw = np.empty(M, np.uint64), np.empty(M, float)
            for k, (x, (d, _, ys, rates)) in enumerate(zip(walkers, dat)):
                stay = 1.0 - tau * (d - Eref)
                ws = tau * rates
                tot = stay + ws.sum()
                if stay < 0 or tot <= 0:
                    raise RuntimeError(("bad propagator weight", stay, tot, tau, d, Eref))
                u = rng.random() * tot
                if u < stay:
                    y = x
                else:
                    j = np.searchsorted(np.cumsum(ws), u - stay, "right")
                    y = int(ys[min(j, len(ys) - 1)])
                nxt[k], bw[k] = y, tot


            bw /= bw.sum()
            walkers = nxt[self._systematic(bw, rng, M)]
            beta += tau
            it += 1
            if beta >= burn_beta:
                self.ensure_local(walkers)
                Es.append(float(np.mean([self.local_cache[int(x)][1] for x in walkers])))
                snaps.append(walkers.copy())
            if progress_every and (it == 1 or it % progress_every == 0):
                print("CALLABLE_FN_PROG", seed, it, "beta", beta, "tau", tau,
                      "E", Eref, "uniq", len(np.unique(walkers)),
                      "r", len(self.r_cache), "local", len(self.local_cache),
                      "amp_eval", self.amp.neval, "sec", time.time() - t0, flush=True)
        mixed = np.concatenate(snaps) if snaps else np.empty(0, np.uint64)
        E = np.asarray(Es, float)
        return mixed, E


def load_backend(module_path, checkpoint=None):
    spec = importlib.util.spec_from_file_location("fn_amplitude_backend", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load amplitude backend {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    obj = mod.load(checkpoint) if hasattr(mod, "load") else mod
    if hasattr(obj, "log_amplitude_bits"):
        return obj.log_amplitude_bits
    if hasattr(obj, "log_amplitude"):
        return obj.log_amplitude
    if callable(obj):
        return obj
    raise TypeError("backend must expose load(...)->object with log_amplitude_bits(states)")


def _npz_array(path, key):
    z = np.load(path)
    if key not in z.files:
        raise KeyError(f"{path} has no key {key}; keys={z.files}")
    return z[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--J2", type=float, default=0.5)
    ap.add_argument("--threshold-npz", required=True)
    ap.add_argument("--threshold-states-key", default="train_states")
    ap.add_argument("--threshold-weights-key", default="iwtrain")
    ap.add_argument("--pool-npz", required=True)
    ap.add_argument("--pool-states-key", default="states")
    ap.add_argument("--pool-weights-key", default="")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--beta-target", type=float, default=1.2)
    ap.add_argument("--burn-beta", type=float, default=0.4)
    ap.add_argument("--tau-max", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=12001)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fn = load_backend(args.backend, args.checkpoint)
    amp = CachedAmplitude(fn)
    eng = K1FNEngine(SquareJ1J2(args.L, args.J2), amp)
    tz = np.load(args.threshold_npz)


    ts = tz[args.threshold_states_key].astype(np.uint64)
    tw = None
    if args.threshold_weights_key:
        tw = tz[args.threshold_weights_key].astype(float)
    T, rr, info = eng.fit_threshold(ts, tw)
    print("CALLABLE_THRESHOLD", T, "centers_trim", info,
          "r_quantiles", np.quantile(rr, [0, .1, .5, .9, 1]).tolist(),
          "amp_eval", amp.neval, flush=True)

    pz = np.load(args.pool_npz)
    pool = pz[args.pool_states_key].astype(np.uint64)
    pw = None
    if args.pool_weights_key:
        pw = pz[args.pool_weights_key].astype(float)
    mixed, E = eng.run_fn(
        pool, M=args.M, beta_target=args.beta_target, burn_beta=args.burn_beta,
        tau_max=args.tau_max, seed=args.seed, pool_weights=pw
    )
    tail = float(E[-min(8, len(E)):].mean()) if len(E) else np.nan
    print("CALLABLE_FN_RESULT", "T", T, "Emean",
          float(E.mean()) if len(E) else np.nan, "tail8", tail,
          "nE", len(E), "mixed", len(mixed), "unique", len(np.unique(mixed)),
          "amp_eval", amp.neval, flush=True)
    np.savez_compressed(
        args.out, threshold=T, threshold_r=rr, threshold_info=np.asarray(info),
        Es=E, mixed=mixed, amp_eval=amp.neval, M=args.M,
        beta_target=args.beta_target, burn_beta=args.burn_beta,
        tau_max=args.tau_max, seed=args.seed, L=args.L, J2=args.J2,
    )


if __name__ == "__main__":
    main()
