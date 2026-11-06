"""pool8.py: tempered MCMC sampler of |B(x)|^(2 beta) on the Sz = 0 sector (PLAN_pfaffian.md "8x8 BUILD, route 2", piece (1a)).
Plugs into pool_fit.py's 8x8 hook (build_pool: c_, l_, o_ = draw_fn(beta, S, rng)):  Pool8Sampler.draw_fn -> (words (S,) uint64, lnB (S,) float64 = ln|B|, orb (S,) int64),
words = bit s set iff sigma_s = +1 (site s = x + L y; N <= 64; the 6x6 reps use the same bit order), orb = canonical orbit id = the minimum image word over the group
of the guide's projection (lat.IMG with flip), reinterpreted as int64 (6x6: equals reps[orbit index]; there is no table at 8x8).  Pool8Sampler.draw_cfg returns the (S, N) int8 +-1 configs instead.
Draws thinned to >= 2 tau_int of the slowest of ln|B|^2 and the decade index (measured; recorded in .tau).  New file; pp2_lemma.py is only imported
(API: logB, one_swap_ratios, PP2.from_cx_theta, PP2.from_rbmpp_npz).

TARGET  pi_beta(x) ~ |B(x)|^(2 beta) per CONFIGURATION on {sigma in {+-1}^N, sum sigma = 0}   (so orbit r has probability n_r |B_r|^(2 beta) / Z, n_r = orbit size).

MOVES (mode)
  'single'   : pick one ANTIPARALLEL bond (J1 or J2) uniformly, exchange its two spins (Sz conserved).  q(x -> y) = 1 / n_anti(x), n_anti = number of
               antiparallel bonds, so the Hastings factor is n_anti(x) / n_anti(y):
                   log acc = 2 beta (lnB(y) - lnB(x)) + ln n_anti(x) - ln n_anti(y).            (lnB(y) by pp2_lemma.logB; one config per proposal)
  'informed' : locally balanced proposal (Zanella): bond b is picked with prob |r_b(x)|^beta / W(x), r_b = B(y_b)/B(x) from ONE call of
               pp2_lemma.one_swap_ratios, W(x) = sum_b |r_b(x)|^beta.  Reverse ratio is 1/r_b, so  acc = min(1, W(x) / W(y)).  The call at y is
               reused as the cache of x' = y when accepted, so there is one call per proposal.
PARALLEL TEMPERING across rungs beta_0 = 1 > beta_1 > ...;  after every sweep, adjacent pairs (even / odd alternating) swap states with probability
  min(1, exp(2 (beta_i - beta_j)(lnB_j - lnB_i))).   One SWEEP = N proposals per walker (N = number of sites; PT(sweep_len=...) overrides, recorded in the output).

CLI (one process = one core = E independent PT ensembles of R rungs each):
  python pool8.py run  --problem 6|8 --out F.npz --betas 1,0.5,0.25 --E 4 --sweeps S --burn B --thin T --seed s [--mode single|informed] [--eval pp2|table6] [--broken none|beta|hastings|pt]
"""
import os, sys, json, time, math, argparse, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
LN10 = math.log(10.0)
THETA4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/theta_4.npy'
LA4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy'
REPS6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz'
NORB6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy'
PROD8 = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'     # = rb8_prod/params_final.npz of gfmc_m13b/prod/rb8/p0_s1 (sha256 0520c5be...cbe9)


# ============================================================================================ evaluators
class PP2Eval:
    """lnB(sig) = Re pp2_lemma.logB ;  ratios(sig) = (lnB_complex (n,), r (n, NB) = B(y_b)/B(x), 0 where aligned)."""

    def __init__(self, params, dtype='f64'):
        import pp2_lemma as PL
        self.PL, self.p, self.dtype = PL, params, dtype; self.lat = params.lat; self.N = self.lat.N
        self.BI, self.BJ, self.NB = self.lat.BI, self.lat.BJ, self.lat.NB

    def lnB(self, sig):
        return np.real(np.asarray(self.PL.logB(self.p, np.asarray(sig, np.float64), dtype=self.dtype)))

    def ratios(self, sig):
        lb, r = self.PL.one_swap_ratios(self.p, np.asarray(sig, np.float64), dtype=self.dtype, return_logB=True)
        return np.asarray(lb), np.asarray(r)


class TableEval6:
    """6x6 B_4 from the exact table ln|B| per orbit (la_table) by canonicalisation; test-only fast path of 'single' mode (no ratios)."""

    def __init__(self, la_path=LA4, reps_path=REPS6):
        import pp2_lemma as PL
        self.lat = PL.lattice(6, 'full'); self.N = 36; self.BI, self.BJ, self.NB = self.lat.BI, self.lat.BJ, self.lat.NB
        self.reps = np.load(reps_path)['reps']; self.la = np.load(la_path).astype(np.float64)
        self.pw = (np.uint64(1) << np.arange(36, dtype=np.uint64))

    def code_min(self, sig):
        out = np.empty(len(sig), np.uint64)
        for i in range(0, len(sig), 64):
            s = np.asarray(sig[i:i + 64]); im = self.lat.FL[None, :, None] * s[:, self.lat.IMG]
            out[i:i + 64] = ((im > 0).astype(np.uint64) * self.pw).sum(-1).min(1)
        return out

    def orbit(self, sig):
        c = self.code_min(sig); ix = np.searchsorted(self.reps, c); assert np.all(self.reps[np.minimum(ix, len(self.reps) - 1)] == c)
        return ix

    def lnB(self, sig): return self.la[self.orbit(sig)]


# ============================================================================================ the sampler
class PT:
    """R rungs x E ensembles of walkers; state sig (R, E, N) int8, lnb (R, E)."""

    def __init__(self, ev, betas, E, seed, mode='single', broken='none', init=None, sweep_len=None):
        self.ev, self.betas = ev, np.asarray(betas, np.float64); self.R = len(betas); self.E = E; self.N = ev.N; self.mode = mode; self.broken = broken; self.sweep_len = sweep_len or ev.N
        self.BI, self.BJ, self.NB = ev.BI, ev.BJ, ev.NB; self.rng = np.random.default_rng(seed)
        n = self.R * E
        if init is None:
            s = np.ones((n, self.N), np.int8)
            for i in range(n): s[i, self.rng.permutation(self.N)[:self.N // 2]] = -1
        else:
            s = np.asarray(init, np.int8).reshape(n, self.N).copy()
        self.sig = s.reshape(self.R, E, self.N); assert np.all(self.sig.sum(-1) == 0)
        self.bw = np.repeat(self.betas, E)                                      # beta per walker (rung-major, matches sig.reshape(n, N))
        self.nacc = np.zeros(self.R); self.nprop = np.zeros(self.R)
        self.sw_acc = np.zeros(max(self.R - 1, 1)); self.sw_try = np.zeros(max(self.R - 1, 1)); self.sweep_no = 0
        self.t_eval = 0.0
        flat = self.sig.reshape(n, self.N)
        if mode == 'single':
            self.lnb = np.ascontiguousarray(self._lnB(flat).reshape(self.R, E))
        else:
            lb, r = self._ratios(flat); self.lnb = np.ascontiguousarray(np.real(lb).reshape(self.R, E)); self.absr = np.abs(r)

    def _lnB(self, s):
        t = time.time(); v = self.ev.lnB(s); self.t_eval += time.time() - t; return v

    def _ratios(self, s):
        t = time.time(); v = self.ev.ratios(s); self.t_eval += time.time() - t; return v

    def n_anti(self, s):
        return ((s[:, self.BI] * s[:, self.BJ]) < 0).sum(1)

    # ---- one proposal for every walker
    def _step_single(self):
        flat = self.sig.reshape(-1, self.N); n = len(flat); lnb = self.lnb.reshape(-1)
        anti = (flat[:, self.BI] * flat[:, self.BJ]) < 0; na = anti.sum(1)
        k = np.minimum((self.rng.random(n) * na).astype(np.int64), na - 1)
        b = (np.cumsum(anti, 1) <= k[:, None]).sum(1); i, j = self.BI[b], self.BJ[b]; ar = np.arange(n)
        y = flat.copy(); y[ar, i] = flat[ar, j]; y[ar, j] = flat[ar, i]
        lby = self._lnB(y); nay = self.n_anti(y)
        if self.broken == 'beta': la = self.bw * (lby - lnb)                          # BROKEN: |B|^beta instead of |B|^(2 beta)
        else: la = 2.0 * self.bw * (lby - lnb)
        if self.broken != 'hastings': la = la + np.log(na) - np.log(nay)                 # (BROKEN 'hastings' drops this factor)
        acc = np.log(self.rng.random(n)) < la
        flat[acc] = y[acc]; lnb[acc] = lby[acc]
        self.nacc += acc.reshape(self.R, self.E).sum(1); self.nprop += self.E

    def _step_informed(self):
        flat = self.sig.reshape(-1, self.N); n = len(flat); lnb = self.lnb.reshape(-1); ar = np.arange(n)
        absr = self.absr; bw = self.bw[:, None]
        w = absr ** bw; Wx = w.sum(1)
        cdf = np.cumsum(w, 1); u = self.rng.random(n) * Wx
        b = np.minimum((cdf <= u[:, None]).sum(1), self.NB - 1); i, j = self.BI[b], self.BJ[b]
        y = flat.copy(); y[ar, i] = flat[ar, j]; y[ar, j] = flat[ar, i]
        lby, ry = self._ratios(y); lby = np.real(lby); absry = np.abs(ry); Wy = (absry ** bw).sum(1)
        la = np.log(Wx) - np.log(Wy)
        if self.broken == 'hastings': la = np.zeros(n)
        acc = np.log(self.rng.random(n)) < la
        flat[acc] = y[acc]; lnb[acc] = lby[acc]; absr[acc] = absry[acc]
        self.nacc += acc.reshape(self.R, self.E).sum(1); self.nprop += self.E

    def _swap(self):
        if self.R < 2: return
        for q in range(self.sweep_no % 2, self.R - 1, 2):
            la = 2.0 * (self.betas[q] - self.betas[q + 1]) * (self.lnb[q + 1] - self.lnb[q])
            if self.broken == 'pt': la = np.zeros(self.E)                                  # BROKEN: always swap
            acc = np.log(self.rng.random(self.E)) < la
            for e in np.nonzero(acc)[0]:
                self.sig[q, e], self.sig[q + 1, e] = self.sig[q + 1, e].copy(), self.sig[q, e].copy()
                self.lnb[q, e], self.lnb[q + 1, e] = self.lnb[q + 1, e], self.lnb[q, e]
                if self.mode == 'informed': self.absr[q * self.E + e], self.absr[(q + 1) * self.E + e] = self.absr[(q + 1) * self.E + e].copy(), self.absr[q * self.E + e].copy()
            self.sw_acc[q] += acc.sum(); self.sw_try[q] += self.E
        self.sweep_no += 1

    def sweep(self):
        for _ in range(self.sweep_len): (self._step_single if self.mode == 'single' else self._step_informed)()
        self._swap()

    def lnb_view(self): return self.lnb.reshape(self.R, self.E)


def hot_start(ev, n, seed, beta=6.0, sweeps=60):
    """n configs quenched at beta (a deliberately opposite start to the random Sz = 0 one: near the top of |B|)."""
    q = PT(ev, [beta], n, seed, 'single'); [q.sweep() for _ in range(sweeps)]; return q.sig[0].copy()


def _bind(pt):
    # lnb / absr are views into flat arrays when reshaped; make the (R, E) arrays contiguous so reshape(-1) is a view
    return pt


# ============================================================================================ statistics
def tau_int(x, c=5.0, maxlag=None):
    """Sokal automatic-windowing integrated autocorrelation time of the series x (T,) or of E independent series x (E, T) (autocovariances averaged
    with a common mean).  Returns (tau, window M, n_eff-ish)."""
    x = np.atleast_2d(np.asarray(x, np.float64)); E, T = x.shape; x = x - x.mean()
    L = maxlag or T // 2; f = np.fft.rfft(x, n=2 * T, axis=1); ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :L].sum(0)
    ac = ac / (E * (T - np.arange(L)))
    if ac[0] <= 0: return 0.5, 0
    rho = ac / ac[0]; tau = 0.5; M = 0
    for m in range(1, L):
        tau += rho[m]
        if m >= c * tau: M = m; break
    else: M = L - 1
    return float(max(tau, 0.5)), int(M)


def decade_index(lnb, med2, lo=-8, hi=4):
    return np.clip(np.floor((2.0 * np.asarray(lnb) - med2) / LN10), lo, hi)


# ============================================================================================ words / orbit ids for the pool_fit hook
def pack_words(cfg):
    c = np.asarray(cfg); w = np.zeros(len(c), np.uint64)
    for s in range(c.shape[1]): w |= (c[:, s] > 0).astype(np.uint64) << np.uint64(s)
    return w


def unpack_words(w, N):
    w = np.asarray(w, np.uint64); return (((w[:, None] >> np.arange(N, dtype=np.uint64)[None]) & np.uint64(1)).astype(np.int8) * 2 - 1)


def orbit_ids(cfg, lat, chunk=2048):
    """int64 view of the minimum image word over lat.IMG x flip (6x6 'full': the psi0 table's reps)."""
    c = np.asarray(cfg, np.int8); out = np.empty(len(c), np.uint64); pw = np.uint64(1) << np.arange(lat.N, dtype=np.uint64)
    for i in range(0, len(c), chunk):
        im = lat.FL[None, :, None].astype(np.int8) * c[i:i + chunk][:, lat.IMG]
        out[i:i + chunk] = ((im > 0).astype(np.uint64) * pw).sum(-1).min(1)
    return out.view(np.int64)


# ============================================================================================ production sampler with the pool_fit hook
class Pool8Sampler:
    """draw_fn(beta, S, rng) -> (configs int8 (S, N), lnB (S,)) from PT of |B|^(2 beta); thinning by 2 tau of the slowest of ln|B|^2 and the decade index
    (tau measured in a pilot at the first draw_fn call unless thin is given).  Output betas are the rungs requested; other rungs only help mixing."""

    def __init__(self, ev, betas=(1.0, 0.5, 0.25), E=8, seed=1, mode='single', thin=None, burn=200, pilot=400, log=print, start='random', keep_pilot=0, med2=None):
        self.ev, self.betas, self.E, self.mode, self.log = ev, list(betas), E, mode, log
        self.pt = PT(ev, betas, E, seed, mode, init=hot_start(ev, len(betas) * E, seed + 7777) if start == 'hot' else None); self.thin = thin; self.burn = burn; self.pilot = pilot; self.buf = {b: [] for b in betas}
        self.keep_pilot = keep_pilot                                                         # TEST HOOK (broken version): push the first keep_pilot unequilibrated pilot sweeps into the output buffers, skip the burn
        self.tau = {}; self.did_burn = False; self.sweeps_done = 0; self.med2 = None
        self.med2_in = med2                                                                  # opt-in: given decade reference (median 2lnB of the beta = 1 rung), needed when the ladder has no beta = 1 rung; None = measured (default path)

    def _advance(self, n, record=None):
        for t in range(n):
            self.pt.sweep(); self.sweeps_done += 1
            if record is not None: record.append(self.pt.lnb.copy())

    def _measure(self):
        rec = []
        for t in range(self.pilot):
            self.pt.sweep(); self.sweeps_done += 1; rec.append(self.pt.lnb.copy())
            if t < self.keep_pilot:                                                          # BROKEN (test hook only): unequilibrated pilot states leak into the output
                for b in self.buf: self.buf[b].append((self.pt.sig[self.betas.index(b)].copy(), self.pt.lnb[self.betas.index(b)].copy()))
        lb = np.array(rec)                                                                   # (T, R, E)
        self.med2 = float(self.med2_in) if self.med2_in is not None else float(np.median(2.0 * lb[:, self.betas.index(1.0), :]))   # the beta = 1 rung (or the given med2)
        dec = decade_index(lb, self.med2); tau = []
        for q in range(self.pt.R):
            t1 = tau_int(2.0 * lb[:, q, :].T)[0]; t2 = tau_int(dec[:, q, :].T)[0]; tau.append(dict(beta=self.betas[q], tau_lnB2=t1, tau_dec=t2))
        self.tau = tau; self.log(f'[pool8] pilot {self.pilot} sweeps: tau per rung {tau}')
        return max(max(t['tau_lnB2'], t['tau_dec']) for t in tau)

    def draw_fn(self, beta, S, rng=None):
        c, l = self.draw_cfg(beta, S, rng)
        return pack_words(c), l, orbit_ids(c, self.ev.lat)

    def draw_cfg(self, beta, S, rng=None):
        if not self.did_burn:
            self._advance(0 if self.keep_pilot else self.burn); self.did_burn = True                  # (broken hook: no burn either)
            tm = self._measure()
            if self.thin is None: self.thin = int(math.ceil(2.0 * tm))
            self.log(f'[pool8] burn {self.burn}, thin = {self.thin} sweeps')
        q = [i for i, b in enumerate(self.betas) if abs(b - beta) < 1e-12]; assert q, f'beta {beta} not a rung of {self.betas}'; q = q[0]
        while sum(len(c) for c, _ in self.buf[beta]) < S:
            self._advance(self.thin)
            for b in self.buf:
                i = self.betas.index(b); self.buf[b].append((self.pt.sig[i].copy(), self.pt.lnb[i].copy()))
        cfg = np.concatenate([c for c, _ in self.buf[beta]])[:, :] ; lb = np.concatenate([l for _, l in self.buf[beta]])
        # entries are (time, ensemble)-ordered blocks of E; take the first S and keep the remainder
        c_out, l_out = cfg[:S], lb[:S]; rest_c, rest_l = cfg[S:], lb[S:]
        self.buf[beta] = [(rest_c, rest_l)] if len(rest_c) else []
        return c_out.astype(np.int8), l_out.astype(np.float64)


# ============================================================================================ CLI: run a set of ensembles on one core, save everything
def load_eval(problem, which, params=None, dtype='f64'):
    import pp2_lemma as PL
    if problem == 6:
        if which == 'table6': return TableEval6()
        th = np.load(THETA4); N = 36
        p = PL.PP2.from_cx_theta(np.concatenate([th, np.zeros(2 * N * N + 2)]), L=6, NA=4)           # B_4 = theta_4: single PP + RBM, P_2 = 0
        p = PL.PP2(p.f1, None, 0.0, p.b, p.W, L=6, proj='full', rbm='outer')                          # drop the zero P_2 (identical lnB, half the cost)
        return PP2Eval(p, dtype)
    p = PL.PP2.from_rbmpp_npz(params or PROD8)
    return PP2Eval(p, dtype)


def run(a):
    ev = load_eval(a.problem, a.eval, a.params, a.dtype); betas = [float(b) for b in a.betas.split(',')]
    init = hot_start(ev, len(betas) * a.E, a.seed + 7777) if a.start == 'hot' else None
    pt = _bind(PT(ev, betas, a.E, a.seed, a.mode, a.broken, init=init, sweep_len=a.sweep_len)); R, E, N = pt.R, pt.E, pt.N; t0 = time.time(); logf = open(a.out + '.log', 'a')
    log = lambda m: (logf.write(m + '\n'), logf.flush(), print(m, flush=True))
    log(f'[pool8] start {time.ctime()} host {os.uname().nodename} problem {a.problem} eval {a.eval} mode {a.mode} broken {a.broken} betas {betas} E {E} seed {a.seed} '
        f'sweeps {a.sweeps} burn {a.burn} thin {a.thin} start {a.start}')
    for t in range(a.burn): pt.sweep()
    pt.nacc[:] = 0; pt.nprop[:] = 0; pt.sw_acc[:] = 0; pt.sw_try[:] = 0; pt.t_eval = 0.0; t_burn = time.time() - t0
    ser = np.zeros((a.sweeps, R, E), np.float32); cfg = []; lb = []; t1 = time.time()
    for t in range(a.sweeps):
        pt.sweep(); ser[t] = pt.lnb
        if (t + 1) % a.thin == 0: cfg.append(pt.sig.copy()); lb.append(pt.lnb.copy())
        if (t + 1) % max(1, a.sweeps // 20) == 0: log(f'[pool8] sweep {t + 1}/{a.sweeps} {time.time() - t1:.0f}s  acc {np.round(pt.nacc / np.maximum(pt.nprop, 1), 3).tolist()}  swap {np.round(pt.sw_acc / np.maximum(pt.sw_try, 1), 3).tolist()}')
    sec = time.time() - t1
    np.savez(a.out, lnb_series=ser, cfg=np.array(cfg, np.int8) if cfg else np.zeros(0), lnb_thin=np.array(lb), betas=np.array(betas), thin=a.thin, burn=a.burn,
             acc=pt.nacc / np.maximum(pt.nprop, 1), swap_acc=pt.sw_acc / np.maximum(pt.sw_try, 1), sec_run=sec, sec_burn=t_burn, sec_eval=pt.t_eval,
             s_per_sweep_per_walker=sec / (a.sweeps * R * E), seed=a.seed, sweep_len=pt.sweep_len, mode=a.mode, broken=a.broken, start=a.start, evalname=a.eval, problem=a.problem)
    log(f'[pool8] done {sec:.0f}s run + {t_burn:.0f}s burn; {sec / (a.sweeps * R * E):.4f} s per sweep per walker (core); eval share {pt.t_eval / max(sec + t_burn, 1e-9):.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run'])
    ap.add_argument('--problem', type=int, default=6); ap.add_argument('--eval', default='pp2'); ap.add_argument('--params', default=None); ap.add_argument('--dtype', default='f64')
    ap.add_argument('--out', required=True); ap.add_argument('--betas', default='1,0.5,0.25'); ap.add_argument('--E', type=int, default=4)
    ap.add_argument('--sweeps', type=int, default=1000); ap.add_argument('--burn', type=int, default=200); ap.add_argument('--thin', type=int, default=10)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--sweep_len', type=int, default=None); ap.add_argument('--start', default='random', choices=['random', 'hot']); ap.add_argument('--mode', default='single', choices=['single', 'informed'])
    ap.add_argument('--broken', default='none', choices=['none', 'beta', 'hastings', 'pt'])
    run(ap.parse_args())
