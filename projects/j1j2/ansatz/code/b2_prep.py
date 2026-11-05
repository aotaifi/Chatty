"""b2_prep.py: branch 2 (PLAN_pfaffian.md "BRANCH 2 = OPTION (ii)", "Branch 2 GO") stages (2) labels and (3) the R1 fit, prepared so that each can launch when its gate
opens.  New file; pool_fit.py and pp2_lemma.py are only imported (pool_fit's fit is wrapped, not edited).  CPU or GPU.

SIGN sgn(M1) = sgn Re[w psi_rb8_prod] (frozen across rounds; w from M1's consts.json, w = -1; v >= 0 -> +1, the decades8 / eps_m1 / pool_x2_fj convention).
  M1Sign: at x from the complex ln B of the SIGN network (rb8_prod), at every one-swap endpoint y_b from the same evaluation's ratio r_b = B_s(y_b)/B_s(x):
  s(y_b) = sgn Re[w e^{i Im ln B_s(x)} r_b]  (one lemma evaluation, no logB of y).  When the guide B_k is NOT the sign network (rounds k >= 1) the sign network gets its own
  one_swap_ratios call (a second lemma evaluator of P_1).  Guard: |margin| < 1e-8 -> that endpoint is recomputed from the direct ln B_s of the configuration.
  Implements pp2_lemma's lemma_phase protocol (endpoints(...)) for fJ_fixed(at_y = False) and a direct __call__ on full configurations.  broken = 'marshall' (TEST ONLY).
LABELS (stage 2)   ln T(x) = ln|B_k(x)| + add_x,  add_x = c sign(f_J) min(|f_J|, f0) (mode 'fix'; node -69 where no kept bond),  f_J = f_J[B_k; sgn(M1)], c = 4, f0 = 1.22,
  E' = -0.4987565 x 64 (labels_m1(): pool_fit.Labels8 with M1Sign).  PARTNERS: NeighboursM1 = pool_fit.Neighbours8's tuple, blocked = anti-aligned & s(x) s(y) > 0 under
  sgn(M1), from the guide's one_swap_ratios (and the sign network's when different), no sign call on full configurations.
POOL (stage 3 input) from x-draw files in the pool8 / pool_x / pool_x2 format (words, lnB, orb, beta, chain, tblock): label_task (one draws file -> partners + labels of its
  draws in decs_rel) and merge (-> pool.npz in pool_fit's format).  chain = rung index in BETAS = [1, 0.5, 0.25] (pool_x's three rungs; pool_x2's beta 0.25 draws join the
  0.25 cell).  Decades d = floor((2 lnB - med) / ln 10) with med FIXED in the spec (the pool's own beta-1 subset is filtered, so its median is not the median);
  weights = pool_fit.pool_weights' rule (band-uniform: per (beta, decade) cell exp(-2 beta lnB) self-normalised, cells mixed by ESS) with that med (pool_weights_med;
  equal to pool_fit.pool_weights when med is the pool's beta-1 median, tested).  Held-out = pool_fit.ho_hash of the orbit id (the draws' 'orb').
FIT (stage 3)  fit(OUT, SPEC): pool_fit.fit (arm 'pp8') with pool_fit.pool_weights replaced, for the held-out sampler only, by pool_weights_med at the pool's stored med2lnB
  (unpatched pool_fit would recompute the median from the pool's filtered beta-1 draws).  Spec 'b2_code_sha256' pins this file.
CLI
  b2_prep.py label_task OUT.npz SPEC.json DRAWS.npz FILE_INDEX [--limit N --select_json F]     partners + labels of one draws file
  b2_prep.py merge OUTDIR SPEC.json PART1.npz PART2.npz ...                                     -> OUTDIR/pool.npz, pool_summary.json, pool.npz.sha256
  b2_prep.py fit OUT SPEC.json                                                                 pool_fit.fit with the med patch
"""
import os, sys, json, time, math, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pp2_lemma as PL
import pool_fit as PF

PROD8 = os.environ.get('G8_PROD8', '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz')
PROD8_SHA = '0520c5be4a7770d84e1968f3669fbbf8063415388c1b50a47ac043317754cbe9'
M1CONSTS = os.environ.get('G8_CONSTS', '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json')
EP8 = -0.4987565 * 64
C_AUG = 4.0
F0 = 1.22
BETAS = [1.0, 0.5, 0.25]
DECS_R1 = list(range(-2, -10, -1))                       # the round's decades -2..-9 (gate (e): range -2..-9 for every f0 in 1.05-1.61)
MED_POOLX = -66.90268441278339                           # median of 2 lnB over pool_x/prod's 450,080 beta-1 draws (pool_x/prod/gate.json "median")
MED_SIZING = -67.03209121446969                          # decades8's beta-1 median (stage-1 PREREG and pool_x2_fjtask default)
GUARD = 1e-8
LN10 = math.log(10.0)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_w():
    c = json.load(open(M1CONSTS)); return complex(*c['w'])


def prod_params(check=True):
    if check: assert sha(PROD8) == PROD8_SHA, 'rb8_prod params sha mismatch'
    return PL.PP2.from_rbmpp_npz(PROD8)


def marshall_np(X):
    """Marshall sign (-1)^{# up on sublattice A (x + y even), site = x + L y}; the eps_m1 broken provider."""
    X = np.asarray(X); L = int(round(np.sqrt(X.shape[-1]))); A = np.array([(s % L + s // L) % 2 == 0 for s in range(X.shape[-1])])
    return np.where((((X[..., A] > 0).sum(-1)) % 2) == 0, 1, -1).astype(np.int8)


# ============================================================================================ sign sgn(M1)
class M1Sign:
    """sgn Re[w B_s] (B_s = sign network, rb8_prod) from the lemma ratios.  guide: the amplitude network whose evaluation d is handed to endpoints(); if it IS the sign
    network (same object) its ratios are used, else the sign network is evaluated separately on x.  broken (TEST ONLY): 'marshall' returns the Marshall sign everywhere."""
    lemma_phase = True

    def __init__(self, sign_params=None, guide=None, w=None, guard=GUARD, broken=None, walker_chunk=256):
        self.sp = sign_params if sign_params is not None else prod_params(); self.same = guide is self.sp      # guide = the network whose evaluation endpoints() receives
        self.w = load_w() if w is None else complex(w); self.guard = float(guard); self.broken = broken; self.wc = int(walker_chunk)
        self.n_checked = 0; self.n_fallback = 0; self.min_margin = np.inf
        lat = self.sp.lat; self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ)

    def _direct(self, X):
        lB = np.asarray(PL.logB(self.sp, np.asarray(X, np.float64))); m = np.real(self.w * np.exp(1j * np.imag(lB))); return m

    def from_ratios(self, X, lB, r):
        """X (M, N) +-1, lB (M,) complex ln B_s(x), r (M, NB) complex B_s(y_b)/B_s(x) -> sx (M,), sy (M, NB) int8 (sy = sx on aligned bonds: unused)."""
        X = np.asarray(X); valid = X[:, self.BI] != X[:, self.BJ]
        if self.broken == 'marshall':
            sx = marshall_np(X); Y = np.repeat(X[:, None, :], len(self.BI), 1).copy(); ar = np.arange(len(self.BI))
            Y[:, ar, self.BI], Y[:, ar, self.BJ] = X[:, self.BJ], X[:, self.BI]; return sx, np.where(valid, marshall_np(Y), sx[:, None]).astype(np.int8)
        ph = self.w * np.exp(1j * np.imag(lB)); mx = np.real(ph); my = np.real(ph[:, None] * r)
        self.n_checked += int(len(X) + valid.sum())
        sm = np.abs(mx) < self.guard
        if sm.any(): mx[sm] = self._direct(X[sm]); self.n_fallback += int(sm.sum())
        smy = valid & (np.abs(my) < self.guard)
        if smy.any():
            mi, bi = np.nonzero(smy); Y = X[mi].astype(np.float64).copy(); a = np.arange(len(mi))
            Y[a, self.BI[bi]], Y[a, self.BJ[bi]] = X[mi, self.BJ[bi]], X[mi, self.BI[bi]]; my[mi, bi] = self._direct(Y); self.n_fallback += len(mi)
        if valid.any(): self.min_margin = min(self.min_margin, float(np.min(np.abs(mx))), float(np.min(np.abs(my[valid]))))
        sx = np.where(mx >= 0, 1, -1).astype(np.int8); sy = np.where(valid, np.where(my >= 0, 1, -1), sx[:, None]).astype(np.int8)
        return sx, sy

    def ratios_s(self, X, d=None):
        """(lB_s, r_s) of the sign network at X: from d (the guide's _one_as_two output) when the guide is the sign network, else one own call."""
        if self.same and d is not None: return np.asarray(d['logB']), np.asarray(d['r1'])
        lB, r = PL.one_swap_ratios(self.sp, np.asarray(X, np.float64), return_logB=True, walker_chunk=self.wc); return np.asarray(lB), np.asarray(r)

    def endpoints(self, x, d, at_y=True, dtype='f64', eval_path=None, walker_chunk=256, margins=False):
        assert not at_y, 'M1Sign: labels need the sign at x and y only (fJ_fixed at_y=False)'
        X = np.asarray(x); lB, r = self.ratios_s(X, d); sx, sy = self.from_ratios(X, lB, r)
        return jnp.asarray(sx.astype(np.float64)), jnp.asarray(sy.astype(np.float64)), None

    def __call__(self, cfg):
        X = np.asarray(cfg)
        if self.broken == 'marshall': return marshall_np(X).astype(np.float64)
        return np.where(self._direct(X) >= 0, 1.0, -1.0)


def labels_m1(guide, sign=None, c=C_AUG, f0=F0, EP=EP8, chunk=256):
    """the stage-2 labels_fn: pool_fit.Labels8 (mode 'fix') with the sign sgn(M1) from the lemma ratios."""
    sign = sign if sign is not None else M1Sign(guide=guide)
    return PF.Labels8(guide, sign, EP, c, f0, 'fix', chunk=chunk)


class NeighboursM1:
    """neighbours_fn (pool_fit.build_pool tuple): y words (S, NB), hw, valid, blocked (anti-aligned & s(x) s(y) > 0 under sgn(M1)), dlnB = ln|B(y)/B(x)| of the GUIDE, orbit ids 0."""

    def __init__(self, guide, sign, walker_chunk=256):
        self.g, self.s, self.wc = guide, sign, int(walker_chunk); lt = PF.lat8(); self.BI, self.BJ, self.hw = lt['BI'], lt['BJ'], lt['HW']
        self.MASK = PF._BIT64[self.BI] | PF._BIT64[self.BJ]

    def __call__(self, cfg, orb=None, sub=4096):
        w = np.asarray(cfg, np.uint64)
        if len(w) > sub:
            parts = [self(w[i:i + sub]) for i in range(0, len(w), sub)]
            return (np.concatenate([q[0] for q in parts]), self.hw, np.concatenate([q[2] for q in parts]), np.concatenate([q[3] for q in parts]),
                    np.concatenate([q[4] for q in parts]), np.zeros((len(w), len(self.BI)), np.int64))
        X = PF.unpack64(w); Xf = X.astype(np.float64)
        lB, r = PL.one_swap_ratios(self.g, Xf, return_logB=True, walker_chunk=self.wc); lB = np.asarray(lB); r = np.asarray(r)
        lBs, rs = (lB, r) if self.s.sp is self.g else self.s.ratios_s(Xf, None)       # sign network = guide: its own ratios; else a second evaluation (P_1)
        sx, sy = self.s.from_ratios(X, lBs, rs)
        valid = X[:, self.BI] != X[:, self.BJ]; ar = np.abs(r)
        with np.errstate(divide='ignore'): dl = np.where(valid & (ar > 0), np.log(np.where(ar > 0, ar, 1.0)), -700.0)
        y = np.where(valid, w[:, None] ^ self.MASK[None, :], w[:, None])
        blocked = valid & (sx[:, None].astype(np.int64) * sy > 0)
        return y, self.hw, valid, blocked, dl, np.zeros(y.shape, np.int64)


# ============================================================================================ pool from x-draws
def pool_weights_med(lBx, chain, betas, decs_rel, rho, xorb, ho_fn=None, med=None):
    """pool_fit.pool_weights with the decade origin med given (None -> pool_fit.pool_weights itself).  Body copied from pool_fit.pool_weights (sha in the pool summary);
    only the median line differs."""
    if med is None: return PF.pool_weights(lBx, chain, betas, decs_rel, rho, xorb, ho_fn)
    med = float(med)
    dec = np.floor((2.0 * lBx - med) / LN10).astype(np.int64)
    ho = ho_fn(xorb) if ho_fn is not None else PF.ho_hash(np.arange(lBx.size))
    w_in = np.zeros(lBx.size); ess = {}; Wbar = np.zeros(len(decs_rel)); W_est = rho.mean(1)
    for g, d in enumerate(decs_rel):
        md = (dec == d) & ~ho; parts = []; e_ = []
        for q, b in enumerate(betas):
            m = md & (chain == q)
            if not m.any(): e_.append(0.0); parts.append(None); continue
            lw = -2.0 * b * lBx[m]; w = np.exp(lw - lw.max()); w /= w.sum(); parts.append((m, w)); e_.append(1.0 / np.sum(w * w))
        E = np.asarray(e_); assert E.sum() > 0, f'decade {d}: no training entry'
        for q, p in enumerate(parts):
            if p is not None: w_in[p[0]] += (E[q] / E.sum()) * p[1]
        ess[str(d)] = dict(per_beta=E.tolist(), total_mixed=float(1.0 / np.sum(w_in[md] ** 2)))
        Wbar[g] = float(np.sum(w_in[md] * W_est[md]))
    return dict(dec=dec, med2lnB=np.float64(med), ho=ho, w_in=w_in, decs_rel=np.asarray(decs_rel, np.int64), Wbar=Wbar, ess_json=np.asarray(json.dumps(ess)))


def partners_labels(x, lBx, neighbours_fn, labels_fn, K, rng, log=print, block=65536):
    """the partner + label part of pool_fit.build_pool (lines after the draw), unchanged: same rng calls in the same order."""
    S = x.size; xorb = np.zeros(S, np.int64)
    yK = np.empty((S, K), np.uint64); yorb = np.empty((S, K), np.int64); dlB = np.zeros((S, K)); WB = np.zeros(S); nblk = np.zeros(S, np.int32)
    for i0 in range(0, S, block):
        i1 = min(S, i0 + block)
        ycfg, hw, valid, blocked, dlnB, yo = neighbours_fn(x[i0:i1], xorb[i0:i1])
        lw = np.where(blocked, np.log(hw)[None, :] + np.clip(dlnB, -700, 700), -np.inf)
        mx = np.max(lw, 1, keepdims=True); has = np.isfinite(mx[:, 0]); mx = np.where(has[:, None], mx, 0.0)
        w = np.exp(lw - mx); W = w.sum(1); WB[i0:i1] = np.where(has, W * np.exp(mx[:, 0]), 0.0); nblk[i0:i1] = blocked.sum(1)
        cdf = np.cumsum(w, 1) / np.where(has, W, 1.0)[:, None]
        sel = np.empty((i1 - i0, K), np.int64)
        for k in range(K): sel[:, k] = np.minimum((cdf < rng.random(i1 - i0)[:, None]).sum(1), cdf.shape[1] - 1)
        ar = np.arange(i1 - i0)[:, None]
        yK[i0:i1] = np.where(has[:, None], ycfg[ar, sel], x[i0:i1, None]); yorb[i0:i1] = np.where(has[:, None], yo[ar, sel], xorb[i0:i1, None])
        dlB[i0:i1] = np.where(has[:, None], dlnB[ar, sel], 0.0)
    log(f'[b2] partners drawn; entries without a blocked bond: {int((nblk == 0).sum())}')
    lTx = labels_fn(x, xorb); lTy = labels_fn(yK.reshape(-1), yorb.reshape(-1)).reshape(S, K)
    lBy = lBx[:, None] + dlB
    rho = np.where(nblk[:, None] > 0, WB[:, None] * np.exp(np.clip((lTy - lBy) - (lTx - lBx)[:, None], -700, 700)), 0.0)
    return dict(y=yK, lnB_y=lBy, WB=WB, nblk=nblk, lnT_x=lTx, lnT_y=lTy, rho=rho)


def build_pool_given(x, lBx, xorb, chain, betas, neighbours_fn, labels_fn, K, seed, decs_rel, ho_fn=None, med=None, log=print):
    """pool_fit.build_pool for GIVEN draws (unequal counts per rung allowed), decade origin med (None -> pool_fit's beta-1 median rule).  Equal to pool_fit.build_pool
    (all keys) for a draw_fn that does not consume the rng (tested)."""
    rng = np.random.default_rng(seed)
    pl = partners_labels(np.asarray(x, np.uint64), np.asarray(lBx, np.float64), neighbours_fn, labels_fn, K, rng, log)
    S = len(x)
    return dict(x=np.asarray(x, np.uint64), x_orb=np.asarray(xorb), chain=np.asarray(chain, np.int8), betas=np.asarray(betas, np.float64), lnB_x=np.asarray(lBx, np.float64),
                y=pl['y'], y_orb=np.zeros((S, K), np.int64), lnB_y=pl['lnB_y'], WB=pl['WB'], nblk=pl['nblk'], lnT_x=pl['lnT_x'], lnT_y=pl['lnT_y'], rho=pl['rho'],
                K=np.int64(K), seed=np.int64(seed), **pool_weights_med(np.asarray(lBx, np.float64), np.asarray(chain, np.int8), betas, decs_rel, pl['rho'], np.asarray(xorb), ho_fn, med))


def rung_of(beta):
    b = np.asarray(beta, np.float64); q = np.full(b.shape, -1, np.int8)
    for k, v in enumerate(BETAS): q[np.abs(b - v) < 1e-12] = k
    assert (q >= 0).all(), f'beta outside {BETAS}: {np.unique(b)}'; return q


def guide_of(SP):
    """the round's guide B_k: rb8_prod (round 0) or a pool_fit theta8 (SP['guide_theta8'])."""
    if SP.get('guide_theta8'): return PF.theta8_to_pp2(np.load(SP['guide_theta8']), int(SP.get('rbm_M', 16)))
    return prod_params()


def label_task(out, SP, draws, fidx, limit=0, select=None, log=print):
    """partners + labels for the draws of one file whose decade (relative to SP['med2lnB']) is in SP['decs_rel'] (or the given row selection)."""
    T0 = time.time(); z = np.load(draws); words = z['words']; lnB = z['lnB']; beta = z['beta']; orb = z['orb']
    if 'final' in z.files: assert int(z['final']) == 1, f'{draws} not final'
    assert str(z['params_sha']) == PROD8_SHA, 'draws are not of rb8_prod'
    med = float(SP['med2lnB']); decs = [int(d) for d in SP.get('decs_rel', DECS_R1)]
    dec = np.floor((2.0 * lnB - med) / LN10).astype(np.int64)
    sel = np.asarray(select, np.int64) if select is not None else np.nonzero(np.isin(dec, decs))[0]
    if limit: sel = sel[:limit]
    guide = guide_of(SP); sgn = M1Sign(guide=guide) if SP.get('guide_theta8') else M1Sign(sign_params=guide, guide=guide)
    lab = labels_m1(guide, sgn, float(SP.get('c', C_AUG)), float(SP.get('f0', F0)), float(SP.get('EP', EP8)))
    nb = NeighboursM1(guide, sgn); K = int(SP.get('K', 2)); seed = int(SP.get('seed', 1)) * 100000 + int(fidx)
    log(f'[b2 label_task] {draws} rows {len(words)} selected {len(sel)} decades {decs} med {med} K {K} seed {seed}')
    pl = partners_labels(words[sel], lnB[sel], nb, lab, K, np.random.default_rng(seed), log)
    # check: the guide's ln|B| at x equals the stored draw lnB (the draws' own network) on the first rows
    nchk = min(64, len(sel)); lchk = np.real(np.asarray(PL.logB(guide, PF.unpack64(words[sel[:nchk]]).astype(np.float64)))) if nchk else np.zeros(0)
    dev = float(np.max(np.abs(lchk - lnB[sel[:nchk]]))) if nchk else 0.0
    np.savez(out, idx=sel, x=words[sel], lnB_x=lnB[sel], orb=orb[sel], beta=beta[sel], src=np.asarray(draws), fidx=np.int64(fidx), seed=np.int64(seed), **pl,
             lnB_guide_vs_stored_maxdev=np.float64(dev), guard_checked=np.int64(sgn.n_checked), guard_fallbacks=np.int64(sgn.n_fallback), min_margin=np.float64(sgn.min_margin),
             sec=np.float64(time.time() - T0), code_sha256=np.asarray(sha(os.path.abspath(__file__))), final=np.int64(1))
    log(f'[b2 label_task] done {len(sel)} rows {time.time() - T0:.0f}s; lnB guide vs stored {dev:.2e}; guard {sgn.n_fallback}/{sgn.n_checked}; min margin {sgn.min_margin:.3e}')


def merge(OUT, SP, parts, log=print):
    os.makedirs(OUT, exist_ok=True); Z = [np.load(p) for p in parts]
    for p, z in zip(parts, Z): assert int(z['final']) == 1, p
    cat = lambda k: np.concatenate([z[k] for z in Z])
    x = cat('x'); lBx = cat('lnB_x'); orb = cat('orb'); chain = rung_of(cat('beta')); K = int(Z[0]['y'].shape[1])
    P = dict(x=x, x_orb=orb, chain=chain, betas=np.asarray(BETAS), lnB_x=lBx, y=cat('y'), y_orb=np.zeros((len(x), K), np.int64), lnB_y=cat('lnB_y'), WB=cat('WB'), nblk=cat('nblk'),
             lnT_x=cat('lnT_x'), lnT_y=cat('lnT_y'), rho=cat('rho'), K=np.int64(K), seed=np.int64(int(SP.get('seed', 1))))
    P.update(pool_weights_med(lBx, chain, BETAS, [int(d) for d in SP.get('decs_rel', DECS_R1)], P['rho'], orb, PF.ho_hash, float(SP['med2lnB'])))
    path = os.path.join(OUT, 'pool.npz'); np.savez(path, **P); summ = PF.pool_summary(P)
    summ.update(spec=SP, parts=list(parts), part_sha256={p: sha(p) for p in parts}, pool_sha256=sha(path), b2_code_sha256=sha(os.path.abspath(__file__)),
                pool_fit_sha256=sha(os.path.join(HERE, 'pool_fit.py')), lnB_guide_vs_stored_maxdev=max(float(z['lnB_guide_vs_stored_maxdev']) for z in Z),
                guard=[int(sum(int(z['guard_fallbacks']) for z in Z)), int(sum(int(z['guard_checked']) for z in Z))])
    PF.jdump(os.path.join(OUT, 'pool_summary.json'), summ); open(path + '.sha256', 'w').write(summ['pool_sha256'] + '  pool.npz\n')
    log(f'[b2 merge] {path} S {len(x)} sha {summ["pool_sha256"][:12]} counts ' + ' '.join(f'{k}:{v["train"]}/{v["heldout"]}' for k, v in summ['counts'].items() if int(k) in SP.get('decs_rel', DECS_R1)))
    return P, summ


# ============================================================================================ fit wrapper
def fit(OUT, SP):
    """pool_fit.fit with pool_fit.pool_weights -> pool_weights_med at the pool's stored med2lnB (used only by the held-out PoolSampler; the training weights are read from the pool)."""
    if SP.get('b2_code_sha256') and SP['b2_code_sha256'] != sha(os.path.abspath(__file__)):
        os.makedirs(OUT, exist_ok=True); PF.jdump(os.path.join(OUT, 'pool_fit.json'), dict(stop='CODE DRIFT b2_prep.py')); print('CODE DRIFT b2_prep.py'); sys.exit(3)
    med = float(np.load(SP['pool'])['med2lnB']); orig = PF.pool_weights
    PF.pool_weights = lambda lBx, chain, betas, decs_rel, rho, xorb, ho_fn=None: pool_weights_med(lBx, chain, betas, decs_rel, rho, xorb, ho_fn, med) if med is not None else orig(lBx, chain, betas, decs_rel, rho, xorb, ho_fn)
    try:
        return PF.fit(OUT, SP)
    finally:
        PF.pool_weights = orig


def r1_spec(pool, pool_sha, warm_npz=PROD8, code_dir=HERE, **over):
    """the R1 fit spec (arm 'pp8', two-PP warm from rb8_prod: P_2 = P_1 + 0.3 rms(f) noise, lam 0.1; knobs = round 5 / V's (o2b round_1 spec_fit.json values));
    every key can be overridden."""
    sp = dict(arm='pp8', rbm_M=16, pool=pool, pool_sha256=pool_sha, warm_npz=warm_npz, p2_sigma_rel=0.3, p2_seed=1, f1_im_sigma=0.0, lambda_init_re=0.1, lambda_init_im=0.0,
              B=8208, B_acc=8208, B_ho=4096, K=2, lam_b=1.0, lam0=1e-2, lam_cap=1.0, lam_floor=1e-6, tau=0.05, tau_min=0.01, anneal_steps=20, z_acc=2.0, ho_every=10,
              flat_tol=0.02, max_steps=1500, save_theta_at=[600, 750, 900, 1050, 1200, 1350], seed=1, fit_sec=27000.0,
              code_sha256={f: sha(os.path.join(code_dir, f)) for f in ('pool_fit.py', 'cx_lib.py', 'rbmpp_lib.py')}, b2_code_sha256=sha(os.path.join(code_dir, 'b2_prep.py')))
    sp.update(over); return sp


def p2_logratio(pool, SP, n_max=None):
    """ln|P_2(x) / P_1(x)| on the pool's x entries for the warm start of SP (warm_theta8's own P_2 construction: same seed, sep, RBM)."""
    w = np.load(pool)['x']; w = w[:n_max] if n_max else w
    th = PF.warm_theta8(dict(SP, lambda_init_re=0.0, lambda_init_im=0.0)); f1, b, W, f2, _ = PF.unpack8(th)
    P1 = PL.PP2(f1, L=8, proj='translations', rbm='inner', b=b, W=W); P2 = PL.PP2(f2, L=8, proj='translations', rbm='inner', b=b, W=W)
    X = PF.unpack64(w).astype(np.float64)
    return np.real(np.asarray(PL.logB(P2, X))) - np.real(np.asarray(PL.logB(P1, X)))


def resolve_lam(sp, pool_local):
    """spec key 'p2_lam_median_target' t (lead 2026-11-06): lambda_init_re = t exp(-median ln|P_2/P_1|) over the pool's x, lambda_init_im = 0, so that the
    pool-median |lam P_2 / P_1| = t at the start.  Records the median and quantiles in the spec."""
    t = sp.get('p2_lam_median_target')
    if t is None: return sp
    lr = p2_logratio(pool_local, sp); med = float(np.median(lr))
    sp.update(lambda_init_re=float(t) * math.exp(-med), lambda_init_im=0.0, p2_logratio_median=med,
              p2_logratio_q=[float(np.quantile(lr, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)], p2_logratio_n=int(lr.size))
    return sp


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); s = ap.add_subparsers(dest='cmd')
    t = s.add_parser('label_task'); t.add_argument('out'); t.add_argument('spec'); t.add_argument('draws'); t.add_argument('fidx', type=int); t.add_argument('--limit', type=int, default=0)
    t.add_argument('--select_json', default='')
    m = s.add_parser('merge'); m.add_argument('outdir'); m.add_argument('spec'); m.add_argument('parts', nargs='+')
    f = s.add_parser('fit'); f.add_argument('out'); f.add_argument('spec')
    a = ap.parse_args()
    if a.cmd == 'label_task':
        sel = json.load(open(a.select_json))[str(a.fidx)] if a.select_json else None
        label_task(a.out, json.load(open(a.spec)), a.draws, a.fidx, a.limit, sel)
    elif a.cmd == 'merge': merge(a.outdir, json.load(open(a.spec)), a.parts)
    elif a.cmd == 'fit': fit(a.out, json.load(open(a.spec)))
    else: ap.print_help()
