"""gfmc_aug8.py: ADAPTER that drives gfmc_aug.AugGuide (067c473, unedited) at 8x8 on the launched M1 walker code, with the pp2_lemma
evaluator (ebeb550) on eval_path 'dedup' and the lemma-path PhaseSign(guard 1e-8) as the stored sign.  New file; gfmc_aug.py, pp2_lemma.py and
the M1 code are only imported (the monkeypatches below wrap functions for timing, they do not change what they compute).

What gfmc_aug.AugGuide already does at 8x8 (checked by reading it): rows() = aug_ratios per power-of-2 bucket; any L (bonds order asserted).
What it does NOT do and this file adds:
  * eval_path: AugGuide._aug calls aug_ratios with the module default 'old' and walker_chunk = bucket.  AugGuide8 passes eval_path='dedup', a
    walker_chunk (GPU), buckets that are multiples of walker_chunk above it (few compiled shapes, little padding), a dmax floor (no early recompiles).
  * c = 0: AugGuide with mode 'fix' puts a node add (-69) on every x without kept rows even at c = 0 (G != s|B| there), and computes f_J and the two-swap
    lemma although c = 0 does not need them.  'c0' provider (AugGuide8 with fast0=True): G = s |B| exactly, from the ONE-swap ratios and the sign at x and at the
    one-swap endpoints only (stored-sign arm of c1).  Same ratio as aug_ratios(mode 'old', c = 0) (checked).
  * direct s_G arm (DirectSG): G = s_G |B| with s_G(z) = sign Re[w e^{i Im ln B(z)} (1 + alpha (E_L(z) - E))] at x and at EVERY one-swap endpoint (E_L from the
    one-swap ratios at that endpoint: one fast E_L per endpoint).  c1's second arm.
  * timers (Prof): plain / lemma / sign / f_J split, compile events (jax.monitoring), GPU peak memory, host overhead; per-block records.
  * run_chain(): the gfmc_aug main loop (same GFMC calls, xhash, E_mix) with these records, a maxsec stop and a .jsonl heartbeat.
CLI:  python gfmc_aug8.py OUT --sys {6,8} --arm {aug,c0,sg} --c C --M M --blocks B --burn b --seed s --X0 file.npy [options]
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gfmc_aug as GA                                  # also puts the M1 walker code on sys.path (GFMC_M1_CODE)
from gfmc_fn_lib import GFMC, GuideBase, blocked_error, iat
import jax, jax.numpy as jnp
import pp2_lemma as PL

PROD8 = os.environ.get('G8_PROD8', '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz')        # env overrides: paths only (Paderborn copies)
PROD8_SHA = '0520c5be4a7770d84e1968f3669fbbf8063415388c1b50a47ac043317754cbe9'
M1CONSTS = os.environ.get('G8_CONSTS', '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json')
EP8 = -0.4987565 * 64
F0_8 = 1.22
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ------------------------------------------------------------------------------------------ profiling
class Prof:
    """cumulative seconds per label.  wrap(obj, name, label) replaces obj.name by a timing proxy that blocks on the result (async dispatch).
    'sign' is inclusive of its own lemma (PhaseSign.endpoints); two_swap_ratios / one_swap_ratios called while 'sign' is on the stack are counted
    in 'sign_lemma' / 'sign_plain' (informational, inside 'sign')."""
    def __init__(self):
        self.t = defaultdict(float); self.n = defaultdict(int); self.stack = []; self.ev = defaultdict(lambda: [0, 0.0])
        try:
            import jax.monitoring as jm
            jm.register_event_duration_secs_listener(self._on_event)
        except Exception as e:
            self.ev['_listener_error'] = [str(e), 0.0]

    def _on_event(self, event, duration, **kw):
        if 'compile' in event or 'cache' in event:
            self.ev[event][0] += 1; self.ev[event][1] += float(duration)

    def n_compiles(self):
        return sum(v[0] for k, v in self.ev.items() if k.endswith('backend_compile_duration'))

    def sec_compile(self):
        return sum(v[1] for k, v in self.ev.items() if k.endswith('backend_compile_duration'))

    def wrap(self, obj, name, label, inside_sign=None):
        f = getattr(obj, name); prof = self
        def g(*a, **k):
            lab = inside_sign if (inside_sign and 'sign' in prof.stack) else label
            t0 = time.perf_counter(); prof.stack.append(label)
            try:
                out = f(*a, **k); jax.block_until_ready(out)
                return out
            finally:
                prof.stack.pop(); prof.t[lab] += time.perf_counter() - t0; prof.n[lab] += 1
        g.__wrapped__ = f; setattr(obj, name, g)

    def snap(self):
        return dict(self.t), dict(self.n), self.n_compiles(), self.sec_compile()


PROF = Prof()
_PATCHED = [False]


def patch_timers():
    if _PATCHED[0]: return
    PROF.wrap(PL, 'two_swap_ratios', 'lemma', inside_sign='sign_lemma')
    PROF.wrap(PL, 'one_swap_ratios', 'plain', inside_sign='sign_plain')
    PROF.wrap(PL.PhaseSign, 'endpoints', 'sign')
    PROF.wrap(PL.PhaseSign, '_guard', 'sign_guard')
    _PATCHED[0] = True


def peak_gib():
    try:
        st = jax.devices()[0].memory_stats() or {}
        return st['peak_bytes_in_use'] / 2 ** 30 if 'peak_bytes_in_use' in st else -1.0
    except Exception:
        return -1.0


# ------------------------------------------------------------------------------------------ guides
class AugGuide8(GA.AugGuide):
    """gfmc_aug.AugGuide with eval_path / walker_chunk / bucket / dmax floor, and fast0 (c = 0 provider, see header)."""
    def __init__(self, params, sign_fn, EP, c, f0, mode='fix', eval_path='dedup', wchunk=128, chunk=1024, min_bucket=16, dmax0=0, sign_chunk=16384,
                 node=None, dtype='f64', fast0=False):
        super().__init__(params, sign_fn, EP, c, f0, mode=mode, node=node, dtype=dtype, chunk=chunk)
        self.eval_path = eval_path; self.wchunk = int(wchunk); self.min_bucket = int(min_bucket); self.sign_chunk = int(sign_chunk)
        self.dmax = max(self.dmax, int(dmax0)); self.fast0 = bool(fast0); self.n_calls = 0; self.shapes = set()
        assert not fast0 or self.c == 0.0
        self.chunk = int(np.ceil(chunk / self.wchunk) * self.wchunk) if chunk > self.wchunk else int(chunk)

    def _bucket(self, n):
        if n > self.wchunk: return int(min(np.ceil(n / self.wchunk) * self.wchunk, self.chunk))
        b = self.min_bucket
        while b < n: b *= 2
        return int(min(b, self.wchunk))

    def rows(self, X):
        self.n_calls += 1
        return super().rows(X)

    def _aug(self, Xc):
        wc = min(self.wchunk, Xc.shape[0]); self.shapes.add((Xc.shape[0], self.dmax))
        if self.fast0: return self._aug0(Xc, wc)
        return self.PL.aug_ratios(self.p, Xc, self.sign_fn, self.EP, self.c, self.f0, self.mode, dtype=self.dtype, dmax=self.dmax, node=self.node,
                                  walker_chunk=wc, eval_path=self.eval_path, sign_chunk=self.sign_chunk)

    def _aug0(self, Xc, wc):
        """G(y_b)/G(x) = s(x) s(y_b) |B(y_b)/B(x)| for the antiparallel bonds (0 otherwise); one-swap ratios + sign at x and y only."""
        PL_ = self.PL; sf = self.sign_fn
        d = PL_._one_as_two(self.p, Xc, self.dtype, wc)
        if getattr(sf, 'lemma_phase', False):
            sx, sy, _ = sf.endpoints(Xc, d, False, self.dtype, self.eval_path, wc)
        else:
            sx, sy, _ = PL_._signs_dedup(sf, np.asarray(Xc), d, False, self.sign_chunk, None)
        ga = jnp.where(d['am'], sx[:, None] * sy * jnp.abs(d['r1']), 0.0)
        return dict(ratio=ga, fx=jnp.zeros(Xc.shape[0]), logB=d['logB'])


class DirectSG(GuideBase):
    """c = 0 guide G = s_G |B| with s_G(z) = sign Re[w e^{i Im lnB(z)} (1 + alpha (E_L(z) - E))] evaluated at x and at every antiparallel one-swap endpoint
    (E_L(z) = diag(z) + sum_b HW_b r_b(z), r_b(z) = B(swap_b z)/B(z) from pp2_lemma._one_j: the fast one-swap ratios, one evaluation per endpoint).
    The constants (alpha, E, w) are M1's p = 1 constants (consts.json); same formula as decades8.make_sG8 (checked against it)."""
    def __init__(self, params, alpha, E, w, chunk=256, dtype='f64'):
        self.PL = PL; self.p = params.cast(dtype); lat = params.lat; self.lat = lat; self.dtype = dtype
        self.N = lat.N; self.NB = lat.NB; self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float)
        self.chunk = int(chunk); self.alpha = float(alpha); self.E = float(E); self.w = complex(w)
        self.rd = jnp.float64 if dtype == 'f64' else jnp.float32
        JBj = jnp.asarray(lat.JB); HWj = jnp.asarray(lat.HW); al = self.alpha; E_ = self.E; w_ = self.w; BI = lat.BI; BJ = lat.BJ

        @jax.jit
        def sg(p, z):
            lB, r = PL._one_j(p, z)
            EL = 0.25 * jnp.sum(JBj * z[:, BI] * z[:, BJ], 1) + jnp.sum(HWj * r, 1)
            v = jnp.real(w_ * jnp.exp(1j * jnp.imag(lB)) * (1.0 + al * (EL - E_)))
            return jnp.where(v >= 0, 1.0, -1.0), lB, r
        self._sg = sg; self.n_calls = 0; self.n_eval = 0; self.n_rows = 0; self.n_x = 0; self.n_pairs = 0

    def _run(self, Z):
        """Z (K, N) float64 numpy -> (sign (K,), lnB (K,), r (K, NB)) in padded chunks of self.chunk (one compile)."""
        K = len(Z); outs = []
        for i in range(0, K, self.chunk):
            blk = Z[i:i + self.chunk]; m = len(blk)
            if m < self.chunk and K > self.chunk: blk = np.concatenate([blk, np.repeat(blk[:1], self.chunk - m, 0)])
            o = self._sg(self.p, jnp.asarray(blk, self.rd)); outs.append(tuple(np.asarray(a)[:m] for a in o))
        return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))

    def sign_at(self, Z):
        return self._run(np.asarray(Z, np.float64))[0]

    def rows(self, X):
        self.n_calls += 1
        B = X.shape[0]; s = X.astype(np.float64); anti = X[:, self.BI] != X[:, self.BJ]; wi, bi = np.nonzero(anti)
        sx, lB, r = self._run(s)                                             # sign at x, and the one-swap ratios r_b(x) (|r| = |B(y_b)/B(x)|)
        Y = s[wi].copy(); q = np.arange(len(wi)); Y[q, self.BI[bi]] = s[wi, self.BJ[bi]]; Y[q, self.BJ[bi]] = s[wi, self.BI[bi]]
        sy = self._run(Y)[0]                                                 # sign at every endpoint (one E_L each)
        rr = sx[wi] * sy * np.abs(r[wi, bi])
        self.n_rows += B; self.n_x += B; self.n_pairs += len(wi); self.n_eval += B + len(wi)
        if not (np.all(np.isfinite(rr)) and np.all(rr != 0)): raise FloatingPointError('DirectSG: non-finite or zero ratio')
        return self.finish(X, anti, wi, bi, rr)

    def stats(self):
        return dict(n_rows=self.n_rows, n_eval=self.n_eval, n_pairs=self.n_pairs, n_x=self.n_x, n_pairs_stat=self.n_pairs, n_rechk=0, n_flip_caught=0)


# ------------------------------------------------------------------------------------------ builders
def m1_consts():
    c = json.load(open(M1CONSTS)); return float(c['alpha']), float(c['E']), complex(*c['w'])


def build(a, prof=PROF):
    """returns (guide, X0, info).  sys 8: amplitude rb8_prod, sign = PhaseSign(a.sign theta.npy), E' = EP8, f0 = F0_8.  sys 6: theta_4 (gfmc_aug defaults),
    sign orbit6:<table.npy> (OrbitSign6) or phase6:<theta.npy> (PhaseSign on 6x6)."""
    info = dict(sys=a.sys, arm=a.arm, c=a.c, eval_path=a.eval_path, wchunk=a.wchunk, M=a.M, dtau=a.dtau, seed=a.seed, mode=a.mode)
    if a.sys == 8:
        assert sha(PROD8) == PROD8_SHA, 'rb8_prod params sha mismatch'
        p = PL.PP2.from_rbmpp_npz(PROD8); info.update(params=PROD8, params_sha256=PROD8_SHA)
        if a.two_pp:                                                       # timing arm only: P_2 = P_1 + noise (pp2_lemma_bench.py construction)
            r = np.random.default_rng(1); f1 = np.asarray(p.f1)
            f2 = f1 + 0.3 * np.std(np.real(f1)) * (r.standard_normal((64, 64)) + 1j * r.standard_normal((64, 64)))
            pP2 = PL.PP2(f2, L=8, proj=p.proj, rbm='inner', b=p.b, W=p.W)
            x0 = np.ones((1, 64)); x0[0, 32:] = -1; np.random.default_rng(5).shuffle(x0[0])
            lam = 0.5 * np.exp(np.real(complex(PL.logB(p, x0)[0]) - complex(PL.logB(pP2, x0)[0])))
            p = p.with_p2(f2, lam); info['two_pp'] = True
        EP = a.EP if a.EP is not None else EP8; f0 = a.f0 if a.f0 is not None else F0_8
    else:
        p = GA.theta4_params(a.theta6); EP = a.EP if a.EP is not None else -18.13670889863218
        f0 = a.f0 if a.f0 is not None else 2.4942595701520887 / 4; info.update(theta=a.theta6, theta_sha256=sha(a.theta6))
    info.update(EP=EP, f0=f0)
    X0 = np.load(a.X0).astype(np.int8)[:a.M]; assert X0.shape == (a.M, p.lat.N), X0.shape
    if a.arm == 'sg':
        assert a.sys == 8 and a.c == 0.0
        al, E, w = m1_consts(); info.update(consts=M1CONSTS, alpha=al, E=E, w=[w.real, w.imag], consts_sha256=sha(M1CONSTS))
        g = DirectSG(p, al, E, w, chunk=a.sgchunk)
        return g, X0, info
    if a.sign.startswith('orbit6:'):
        f = a.sign.split(':', 1)[1]; sfn = GA.OrbitSign6(np.load(f), name=os.path.basename(f)); info.update(sign=f, sign_sha256=sha(f))
    else:
        kind, f = a.sign.split(':', 1); th = np.load(f); info.update(sign=f, sign_sha256=sha(f), sign_size=int(th.size))
        sfn = PL.PhaseSign.from_theta(th, L=a.sys, guard=a.guard); info['guard'] = a.guard
    dmax0 = int(min(p.lat.NB, 8 * int(np.ceil((int(np.max(np.sum(X0[:, p.lat.BI] != X0[:, p.lat.BJ], 1))) + a.dmargin) / 8)))) if a.dmargin >= 0 else 0
    fast0 = (a.arm == 'c0'); mode = 'old' if (a.arm == 'aug' and a.c == 0.0 and a.mode == 'old') else a.mode
    g = AugGuide8(p, sfn, EP, a.c, f0, mode=mode, eval_path=a.eval_path, wchunk=a.wchunk, chunk=a.M, dmax0=dmax0, fast0=fast0, dtype='f64')
    info['dmax0'] = dmax0
    return g, X0, info


# ------------------------------------------------------------------------------------------ chain driver
def run_chain(g, X0, out, a, info, P):
    """gfmc_aug main loop + records.  Writes OUT/blocks.jsonl (one line per block, flushed), blocks.npz, E_mix.npy (after burn, /N), result.json."""
    N = g.N; M = X0.shape[0]; os.makedirs(out, exist_ok=True); jl = open(os.path.join(out, 'blocks.jsonl'), 'a')
    patch_timers(); json.dump(info, open(os.path.join(out, 'run_info.json'), 'w'), indent=1, default=float)
    if not hasattr(g.rows, '__wrapped__'): PROF.wrap(g, 'rows', 'rows')
    t0 = time.perf_counter(); s0 = PROF.snap(); gf = GFMC(g, X0, a.dtau, a.seed); t_init = time.perf_counter() - t0; s1 = PROF.snap()
    init = dict(sec=t_init, rows=g.n_rows if hasattr(g, 'n_rows') else None, t={k: s1[0].get(k, 0) - s0[0].get(k, 0) for k in s1[0]}, compiles=s1[2] - s0[2], sec_compile=s1[3] - s0[3],
                peak_GiB=peak_gib())
    P(f'init rows {M} walkers {t_init:.1f}s compiles {init["compiles"]} ({init["sec_compile"]:.1f}s) peak {init["peak_GiB"]:.2f} GiB')
    np.save(os.path.join(out, 'X0.npy'), X0)
    tot = a.burn + a.blocks; rec = []; tstart = time.perf_counter(); status = 'OK'; stopped = False
    for b in range(tot):
        sn0 = PROF.snap(); st0 = dict(PL.STATS); n_rows0 = getattr(g, 'n_rows', 0); calls0 = getattr(g, 'n_calls', 0)
        chk0 = getattr(getattr(g, 'sign_fn', None), 'n_checked', 0); fb0 = getattr(getattr(g, 'sign_fn', None), 'n_fallback', 0)
        tb = time.perf_counter()
        try:
            r = gf.block()
        except FloatingPointError as e:
            status = f'NAN {e}'; P(f'ABORT block {b + 1}: {e}'); break
        sec = time.perf_counter() - tb; sn1 = PROF.snap(); st1 = dict(PL.STATS)
        dt = {k: sn1[0].get(k, 0) - sn0[0].get(k, 0) for k in sn1[0]}
        r.update(block=b, sec=sec, rows=getattr(g, 'n_rows', 0) - n_rows0, calls=getattr(g, 'n_calls', 0) - calls0, t=dt, compiles=sn1[2] - sn0[2],
                 sec_compile=sn1[3] - sn0[3], peak_GiB=peak_gib(), M=M,
                 stats={k: st1.get(k, 0) - st0.get(k, 0) for k in st1 if isinstance(st1[k], (int, float))},
                 phase_checked=getattr(getattr(g, 'sign_fn', None), 'n_checked', 0) - chk0, phase_fallback=getattr(getattr(g, 'sign_fn', None), 'n_fallback', 0) - fb0)
        r['xhash'] = int(hashlib.sha256(np.ascontiguousarray(gf.X).tobytes()).hexdigest()[:15], 16)
        r['logw_ok'] = bool(np.isfinite(r['emix']) and np.isfinite(r['egrow'])); rec.append(r)
        jl.write(json.dumps(r, default=float) + '\n'); jl.flush()
        P(f'block {b + 1}/{tot} E_mix/N {r["emix"] / N:.8f} E_grow/N {r["egrow"] / N:.8f} ess {r["ess"]:.1f} elvar {r["elvar"]:.4f} hops {r["hops"]} iters {r["iters"]} '
          f'rows {r["rows"]} calls {r["calls"]} sec {sec:.2f} rows_sec {dt.get("rows", sec):.2f} compiles {r["compiles"]} ({r["sec_compile"]:.1f}s) peak {r["peak_GiB"]:.2f}GiB '
          f'fallback {r["phase_fallback"]}/{r["phase_checked"]}')
        if not r['logw_ok']: status = 'NAN energy'; P(f'ABORT block {b + 1}: non-finite energy'); break
        if time.perf_counter() - tstart > a.maxsec: stopped = True; P(f'MAXSEC reached at block {b + 1}'); break
    nb = len(rec); em = np.array([r['emix'] for r in rec[a.burn:]]); eg = np.array([r['egrow'] for r in rec[a.burn:]])
    np.save(os.path.join(out, 'E_mix.npy'), em / N); np.save(os.path.join(out, 'E_mix_all.npy'), np.array([r['emix'] for r in rec]) / N)
    keys = ('emix', 'egrow', 'ess', 'elvar', 'hops', 'iters', 'xhash', 'sec', 'rows', 'calls', 'compiles', 'sec_compile', 'peak_GiB', 'phase_checked', 'phase_fallback')
    np.savez(os.path.join(out, 'blocks.npz'), burn=a.burn, dtau=a.dtau, M=M, N=N, **{k: np.array([r[k] for r in rec]) for k in keys})
    res = dict(args=vars(a), info=info, nblocks=nb, completed=bool(not stopped and status == 'OK' and nb == tot), status=status, N=N, init=init,
               prof_cum={k: v for k, v in PROF.t.items()}, prof_n=dict(PROF.n), events={k: v for k, v in PROF.ev.items()}, guide_stats=g.stats())
    if len(em) >= 32:
        m, errs, tau = blocked_error(em); res['mix'] = dict(E_site=m / N, err_site=max(errs[-4:]) / N, err_level0_site=errs[0] / N, tau_blocking=tau, tau_int_blocks=iat(em))
    json.dump(res, open(os.path.join(out, 'result.json'), 'w'), indent=1, default=float)
    P(f'GFMC_AUG8_DONE status {status} blocks {nb}/{tot} completed {res["completed"]}')
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--sys', type=int, default=8, choices=[6, 8]); ap.add_argument('--arm', default='aug', choices=['aug', 'c0', 'sg'])
    ap.add_argument('--c', type=float, default=4.0); ap.add_argument('--mode', default='fix', choices=['fix', 'a1', 'old'])
    ap.add_argument('--M', type=int, default=1024); ap.add_argument('--dtau', type=float, default=0.1); ap.add_argument('--blocks', type=int, default=100)
    ap.add_argument('--burn', type=int, default=0); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--X0', required=True)
    ap.add_argument('--maxsec', type=float, default=1e18)
    ap.add_argument('--eval_path', default='dedup', choices=['old', 'pack', 'dedup']); ap.add_argument('--wchunk', type=int, default=128)
    ap.add_argument('--sgchunk', type=int, default=256); ap.add_argument('--dmargin', type=int, default=16)
    ap.add_argument('--sign', default='phase:/scratch/a/A.Otaifi/ansatz_runs/sign_scale/r8/ppw_s1/theta.npy'); ap.add_argument('--guard', type=float, default=1e-8)
    ap.add_argument('--EP', type=float, default=None); ap.add_argument('--f0', type=float, default=None); ap.add_argument('--two_pp', action='store_true')
    ap.add_argument('--theta6', default=f'{GA.R6}/pfaffian_r4/pdb_tabs/tab_t4/round_1/theta.npy')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv); os.makedirs(a.out, exist_ok=True); prog = os.path.join(a.out, 'progress.txt')

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'; print(line, flush=True); open(prog, 'a').write(line + '\n')
    if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit('gfmc_aug8: no CUDA device (%s); set G8_ALLOW_CPU=1 to run on CPU' % jax.devices()[0])
    patch_timers(); g, X0, info = build(a)
    info['code_sha256'] = {f: sha(os.path.join(HERE, f)) for f in ('gfmc_aug8.py', 'gfmc_aug.py', 'pp2_lemma.py')}; info['device'] = str(jax.devices()[0]); info['jax'] = jax.__version__
    P(f'guide sys {a.sys} arm {a.arm} c {a.c} mode {a.mode} M {a.M} dtau {a.dtau} seed {a.seed} eval_path {a.eval_path} wchunk {a.wchunk} device {info["device"]} '
      f'info {dict((k, v) for k, v in info.items() if k in ("params_sha256", "sign_sha256", "theta_sha256", "EP", "f0", "dmax0", "guard"))}')
    return run_chain(g, X0, a.out, a, info, P)


if __name__ == '__main__':
    main()
