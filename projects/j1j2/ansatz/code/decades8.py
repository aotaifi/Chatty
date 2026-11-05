"""decades8.py: where the round-1 target correction's blocked-bond weight sits per relative |B|^2 decade (PLAN_pfaffian.md "8x8 decade range";
Test W (iii) at 8x8).  CPU only.  New file; pool8.py / pp2_lemma.py / M1 code are only imported.

Quantity (per configuration x, under |B|^2):
  g(x) = sum_{y blocked} |H_xy| |B(y)/B(x)| (D_x - D_y)^2,  D = c f_J[B] (add of pp2_lemma._apply_mode, mode 'fix' with f0, or 'old'),
  blocked = valid antiparallel bond with s_G(x) s_G(y) > 0.  sum_x |B_x|^2 g(x) = sum over directed pairs |H| |B_x B_y| dD^2.
  Decade assignment (Test W convention, primary): each unordered pair goes to the decade of its SMALLER-|B| side, i.e. from the x end only the y with
  |B_y| > |B_x| count (ties 1/2).  Secondary ('sym'): every directed entry counts 1/2 in its own x decade.
  Decades d = floor((2 lnB_x - med) / ln 10), med = median of 2 lnB over the beta = 1 draws (8x8); at 6x6 the control also uses absolute decades
  (2 lnB normalised with the exact table norm) to compare with pfaffian_w/w_result.json part1 'share'.
Sign s_G (M1 p = 1, experiments/addon_10x10/m1_code/gfmc_fn_lib.py RBMPPGuideG.rows):  s_G(x) = sign Re[w e^{i Im lnpsi(x)} (1 + alpha (E_L(x) - E))],
  E_L from ELnp2 (numpy rank-2 update, production) or pp2_lemma.one_swap_ratios (H = sum J S.S; diag 1/4 J s_i s_j, off-diagonal J/2 r_b).  sigcheck compares it with M1's own code path.
Raw f_J at x and at every one-swap y are stored (mode independent), so c, f0 and mode are applied in the analysis.

CLI
  decades8.py sigcheck OUT.json [--n 240]                       8x8: own sign_fn vs M1 RBMPPGuideG on pool8 meas8 configs (+ broken alpha = 0)
  decades8.py task  --problem 8|6 --out F.npz --betas 1,.5,.25,.125 --E 2 --nper 40 --seed s [--maxsec T] [--thin t]
        PT sampler (pool8.Pool8Sampler), every thin sweeps one draw per rung and ensemble; f_J raw at x and y for each draw; checkpoint F.npz
  decades8.py analyze OUT.json F1.npz F2.npz ... [--problem 8|6] [--c 4] [--f0 auto|<val>] [--mode fix|old] [--plain]
"""
import os, sys, json, time, math, argparse, hashlib, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
LN10 = math.log(10.0)
PROD8 = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'
PROD8_SHA = '0520c5be4a7770d84e1968f3669fbbf8063415388c1b50a47ac043317754cbe9'
M1CODE = '/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/addon_10x10/m1_code'
M1CONSTS = '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json'
EP8 = -0.4987565 * 64                     # E' = p=0 FN energy x N (brief)
SG6 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
FJ6 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/fJ_theta4.npy'
EP6 = -18.136708898632186                 # pfaffian_r5/prep_result.json E_bar_fresh.E (T_5's E')
NORB6 = '/project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def m1_consts():
    c = json.load(open(M1CONSTS)); return float(c['alpha']), float(c['E']), complex(*c['w'])


# ============================================================================================ sign functions
def make_sG8(p, alpha, E, w, chunk=256):
    """s_G from pp2_lemma (own path).  numpy (M, N) +-1 -> (M,) +-1."""
    import pp2_lemma as PL
    lat = p.lat; HW = np.asarray(lat.HW); BI, BJ = lat.BI, lat.BJ; JB = np.asarray(lat.JB)
    def sign_fn(x):
        x = np.asarray(x, np.float64)
        lB, r = PL.one_swap_ratios(p, x, walker_chunk=chunk, return_logB=True); lB = np.asarray(lB); r = np.asarray(r)
        EL = 0.25 * (JB[None] * x[:, BI] * x[:, BJ]).sum(1) + (HW[None] * r).sum(1)
        v = (w * np.exp(1j * lB.imag) * (1.0 + alpha * (EL - E))).real
        sign_fn.n += len(x)
        return np.where(v >= 0, 1, -1).astype(np.int8)
    sign_fn.n = 0
    return sign_fn


class ELnp:
    """numpy port of M1 rbmpp_lib.make_fns eloc_full (img_quant: rank-2 determinant update per image, 'inner' RBM per image), batched over
    configurations.  E_L (complex), ln psi_P (complex).  Validated against pp2_lemma.one_swap_ratios and M1 (sigcheck / elcheck)."""
    def __init__(self, path=PROD8, kchunk=8):
        sys.path.insert(0, M1CODE)
        from rbmpp_lib import Lattice
        z = np.load(path); self.lat = lat = Lattice(int(round(np.sqrt(z['f'].shape[0]))), 0.5, str(z['proj']))
        self.f = z['f'].astype(np.float64); self.b = z['bre'] + 1j * z['bim']; self.W = z['Wre'] + 1j * z['Wim']; self.rbm = bool(z['rbm'])
        self.IMG = np.asarray(lat.IMG); self.FL = np.asarray(lat.FL, float); self.INV = np.asarray(lat.INV); self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ)
        self.JB = np.asarray(lat.JB, float); self.N = lat.N; self.n = lat.N // 2; self.NB = lat.NB; self.k = kchunk
        self.SI = self.INV[:, self.BI]; self.SJ = self.INV[:, self.BJ]                       # (nI, NB) image sites of each bond

    @staticmethod
    def _lc(x):
        s = np.where(x.real >= 0, 1.0, -1.0); return s * x + np.log1p(np.exp(-2.0 * s * x)) - np.log(2.0)

    def __call__(self, sig):
        sig = np.asarray(sig, np.float64); out_E = np.empty(len(sig), complex); out_l = np.empty(len(sig), complex)
        for i in range(0, len(sig), self.k):
            out_E[i:i + self.k], out_l[i:i + self.k] = self._chunk(sig[i:i + self.k])
        return out_E, out_l

    def _chunk(self, sig):
        K = len(sig); n = self.n; N = self.N; f = self.f; nI = self.IMG.shape[0]; NB = self.NB
        Z = self.FL[None, :, None] * sig[:, self.IMG]                                          # (K, nI, N)
        order = np.argsort(-Z, axis=-1, kind='stable'); U = order[..., :n]; D = order[..., n:]
        F = f[U[..., :, None], D[..., None, :]]                                                # (K, nI, n, n)
        sg, ld = np.linalg.slogdet(F); G = np.linalg.inv(F)
        up = (Z > 0).astype(np.int64); dn = 1 - up
        inv = np.sum(up * (np.cumsum(dn, -1) - dn), -1)
        par = ((sg < 0).astype(np.int64) + inv) % 2
        lg = ld + 1j * np.pi * par
        pos = np.where(up > 0, np.cumsum(up, -1) - 1, np.cumsum(dn, -1) - 1)                  # (K, nI, N)
        SI = np.broadcast_to(self.SI, (K, nI, NB)); SJ = np.broadcast_to(self.SJ, (K, nI, NB))
        zi = np.take_along_axis(Z, SI, -1); zj = np.take_along_axis(Z, SJ, -1); valid = zi != zj
        p_ = np.where(zi > 0, SI, SJ); q_ = np.where(zi > 0, SJ, SI)
        a_ = np.take_along_axis(pos, p_, -1); bc = np.take_along_axis(pos, q_, -1)
        fD = f[:, D].transpose(1, 2, 0, 3)                                                     # (K, nI, N, n)
        R = fD @ G                                                                             # (K, nI, N, n)
        fU = f[U]                                                                              # (K, nI, n, N)
        V = np.take_along_axis(fU, np.broadcast_to(p_[..., None, :], (K, nI, n, NB)), -1) - np.take_along_axis(fU, np.broadcast_to(q_[..., None, :], (K, nI, n, NB)), -1)
        kk, gg, bb = np.meshgrid(np.arange(K), np.arange(nI), np.arange(NB), indexing='ij')
        V[kk, gg, a_, bb] = f[q_, p_] - f[q_, q_]
        Gb = np.take_along_axis(G, np.broadcast_to(bc[..., None], (K, nI, NB, n)), -2)         # (K, nI, NB, n)
        Rq = np.take_along_axis(R, np.broadcast_to(q_[..., None], (K, nI, NB, n)), -2)
        VT = V.transpose(0, 1, 3, 2)                                                           # (K, nI, NB, n)
        Gv = np.sum(Gb * VT, -1); rGv = np.sum(Rq * VT, -1)
        Rqa = np.take_along_axis(Rq, a_[..., None], -1)[..., 0]; Gba = np.take_along_axis(Gb, a_[..., None], -1)[..., 0]
        Va = np.take_along_axis(VT, a_[..., None], -1)[..., 0]
        ratio = -(Rqa * (1.0 + Gv) - Gba * (rGv - Va)).astype(complex)
        if self.rbm:
            th = self.b + Z.astype(complex) @ self.W                                           # (K, nI, M)
            lct = self._lc(th); lg = lg + lct.sum(-1)
            dth = -2.0 * self.W[p_] + 2.0 * self.W[q_]                                        # (K, nI, NB, M)
            ratio = ratio * np.exp(np.sum(self._lc(th[:, :, None, :] + dth) - lct[:, :, None, :], -1))
        ratio = np.where(valid, ratio, 0.0)
        m = lg.real.max(-1, keepdims=True); s = np.sum(np.exp(lg - m), -1); lP = m[:, 0] + np.log(s)
        wts = np.exp(lg - lP[:, None]); rP = np.einsum('kg,kgb->kb', wts, ratio)
        vb = sig[:, self.BI] != sig[:, self.BJ]
        EL = 0.25 * np.sum(self.JB * sig[:, self.BI] * sig[:, self.BJ], -1) + np.sum(np.where(vb, 0.5 * self.JB * rP, 0.0), -1)
        return EL, lP


class ELnp2(ELnp):
    """faster numpy E_L: per image G = F^-1, R = f[:, D] G (N x n), C = G f[U, :] (n x N), S = R f[U, :] (N x N); a swap of up site p (row slot a) with
    down site q (column slot b) has det F'/det F = R[q,a] C[b,p] + G[b,a] (f[q,p] - S[q,p]) (derived by Sherman-Morrison; = M1 img_quant rdet), psi ratio = -that.
    RBM per image: cosh(th + D_pq)/cosh(th) = cosh D_pq + tanh(th) sinh D_pq, D_pq = 2 (W[q] - W[p]) (precomputed per site pair)."""
    def __init__(self, path=PROD8, kchunk=8):
        super().__init__(path, kchunk)
        Dl = 2.0 * (self.W[None, :, :] - self.W[:, None, :])                                   # [p, q, k] = 2 (W[q] - W[p])
        self.chD = np.cosh(Dl); self.shD = np.sinh(Dl)

    def _chunk(self, sig):
        K = len(sig); n = self.n; N = self.N; f = self.f; nI = self.IMG.shape[0]; NB = self.NB
        Z = self.FL[None, :, None] * sig[:, self.IMG]
        order = np.argsort(-Z, axis=-1, kind='stable'); U = order[..., :n]; D = order[..., n:]
        F = f[U[..., :, None], D[..., None, :]]
        sg, ld = np.linalg.slogdet(F); G = np.linalg.inv(F)
        up = (Z > 0).astype(np.int64); dn = 1 - up
        inv = np.sum(up * (np.cumsum(dn, -1) - dn), -1)
        par = ((sg < 0).astype(np.int64) + inv) % 2
        lg = ld + 1j * np.pi * par
        pos = np.where(up > 0, np.cumsum(up, -1) - 1, np.cumsum(dn, -1) - 1)
        fD = f[:, D].transpose(1, 2, 0, 3)                                                     # (K, nI, N, n)
        fU = f[U]                                                                              # (K, nI, n, N)
        R = fD @ G; C = G @ fU; S = R @ fU                                                     # (K,nI,N,n) (K,nI,n,N) (K,nI,N,N)
        SI = np.broadcast_to(self.SI, (K, nI, NB)); SJ = np.broadcast_to(self.SJ, (K, nI, NB))
        zi = np.take_along_axis(Z, SI, -1); zj = np.take_along_axis(Z, SJ, -1); valid = zi != zj
        p_ = np.where(zi > 0, SI, SJ); q_ = np.where(zi > 0, SJ, SI)
        a_ = np.take_along_axis(pos, p_, -1); b_ = np.take_along_axis(pos, q_, -1)
        kk = np.arange(K)[:, None, None]; gg = np.arange(nI)[None, :, None]
        rdet = R[kk, gg, q_, a_] * C[kk, gg, b_, p_] + G[kk, gg, b_, a_] * (f[q_, p_] - S[kk, gg, q_, p_])
        ratio = np.where(valid, -rdet, 0.0).astype(complex)
        if self.rbm:
            th = self.b + Z.astype(complex) @ self.W
            lg = lg + self._lc(th).sum(-1)
            tt = np.tanh(th)                                                                   # (K, nI, M)
            ki, gi, bi = np.nonzero(valid); pv = p_[ki, gi, bi]; qv = q_[ki, gi, bi]          # valid bonds only
            ratio[ki, gi, bi] *= np.prod(self.chD[pv, qv] + tt[ki, gi] * self.shD[pv, qv], -1)
        m = lg.real.max(-1, keepdims=True); s = np.sum(np.exp(lg - m), -1); lP = m[:, 0] + np.log(s)
        wts = np.exp(lg - lP[:, None]); rP = np.einsum('kg,kgb->kb', wts, ratio)
        vb = sig[:, self.BI] != sig[:, self.BJ]
        EL = 0.25 * np.sum(self.JB * sig[:, self.BI] * sig[:, self.BJ], -1) + np.sum(np.where(vb, 0.5 * self.JB * rP, 0.0), -1)
        return EL, lP


def make_sG8_np(alpha, E, w, path=PROD8, kchunk=8):
    el = ELnp2(path, kchunk)
    def sign_fn(x):
        EL, lP = el(np.asarray(x, np.float64)); sign_fn.n += len(x)
        v = (w * np.exp(1j * lP.imag) * (1.0 + alpha * (EL - E))).real
        return np.where(v >= 0, 1, -1).astype(np.int8)
    sign_fn.n = 0; sign_fn.el = el
    return sign_fn


def m1_sign(cfg, alpha, E, w):
    """M1's own s_G: RBMPPGuideG (gfmc_fn_lib.py) amp + its cx formula, production lattice 'translations'."""
    sys.path.insert(0, M1CODE)
    import gfmc_fn_lib as G
    from rbmpp_lib import Lattice
    z = np.load(PROD8); lat = Lattice(8, 0.5, str(z['proj']))
    g = G.RBMPPGuideG({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, alpha, E, w=w, chunk=64, use_rbm=bool(z['rbm']))
    Ex, lx = g.amp(np.asarray(cfg, np.float64))
    cx = (g.w * np.exp(1j * lx.imag) * (1.0 + g.alpha * (Ex - g.E))).real
    return np.where(cx >= 0, 1, -1).astype(np.int8), cx, Ex, lx


class TableSign6:
    """exact s_G of the 6x6 problem by orbit lookup (pfaffian_lswo2/s_G.npy)."""
    def __init__(self, ev):
        self.ev = ev; self.s = np.load(SG6).astype(np.int8); self.n = 0
    def __call__(self, x):
        self.n += len(x); return self.s[self.ev.orbit(np.asarray(x))]


# ============================================================================================ per-draw f_J raw
def _signs_keys(sign_fn, keys, N, chunk=4096):
    U, inv = np.unique(keys, return_inverse=True)
    out = np.empty(len(U), np.int8)
    sh = np.arange(N, dtype=np.uint64)[None]
    for i in range(0, len(U), chunk):
        kb = U[i:i + chunk]; cfg = (((kb[:, None] >> sh) & np.uint64(1)).astype(np.int8) * 2 - 1)
        out[i:i + chunk] = np.asarray(sign_fn(cfg)).astype(np.int8)
    assert np.all(np.abs(out) == 1)
    return out[inv].reshape(keys.shape), len(U)


def fj_raw(p, x, sign_fn, EP, eval_path='dedup', full=False, side='small'):
    """raw f_J at x (all y) and at the BLOCKED y only (s_G(x) s_G(y) > 0); s_G at z two swaps away is only evaluated behind blocked y.
    full=True: f_J at every y (= pp2_lemma.fJ_fixed, for the check).  Returns per-bond (NB) layout: fy (nan unless blocked / full), dl = ln|B(y)/B(x)|
    (nan if aligned), sy."""
    import pp2_lemma as PL
    import jax.numpy as jnp
    x = np.asarray(x); M, N = x.shape; lat = p.lat
    d = PL.two_swap_ratios(p, x, eval_path=eval_path)
    am = np.asarray(d['am']); a = np.asarray(d['a']); dA = am.shape[1]
    mask = np.asarray(PL._flip_masks(d['upa'], d['dna'], d['am'], d['ru'], d['rd'], d['k'], d['bm']))
    xkey = PL._cfg_keys(x)
    sxy, nu1 = _signs_keys(sign_fn, np.concatenate([xkey[:, None], xkey[:, None] ^ mask[:, :dA]], 1), N)
    sx = sxy[:, 0].astype(np.float64); sy = sxy[:, 1:].astype(np.float64)
    blk = am & (sx[:, None] * sy > 0)
    if side == 'small':                                  # only y with |B_y| >= |B_x| enter the smaller-side estimator from x's end
        blk = blk & (np.abs(np.asarray(d['r1'])) >= 1.0)
    elif side == 'large':                                # only y with |B_y| <= |B_x|: the pair is seen from its LARGER end (weight ratio <= 1)
        blk = blk & (np.abs(np.asarray(d['r1'])) <= 1.0)
    bm = np.asarray(d['bm']); dB = bm.shape[2]
    need = (blk[:, :, None] | full) & bm & am[:, :, None]
    zk = (xkey[:, None] ^ mask[:, dA:]).reshape(M, dA, dB)
    sz = np.repeat(sy[:, :, None], dB, 2).copy()                   # dummy (not used) where not needed
    if need.any():
        v, nu2 = _signs_keys(sign_fn, zk[need], N); sz[need] = v
    else: nu2 = 0
    fx, py = PL._fj_parts(lat, jnp.asarray(x, jnp.float64), d, jnp.asarray(sx), jnp.asarray(sy), jnp.asarray(sz), EP, True)
    SKy, VAy = py
    Y = np.asarray(PL._configs(jnp.asarray(x, jnp.float64), {k: d[k] for k in ('upa', 'dna', 'am', 'ru', 'rd', 'k')}, with_z=False)[0])
    Hy = 0.25 * (np.asarray(lat.JB)[None, None] * Y[..., lat.BI] * Y[..., lat.BJ]).sum(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        fy = np.log(np.asarray(SKy)) - np.log(Hy + np.asarray(VAy) - EP)
    fy = np.where((blk | full) & am, fy, np.nan)
    r1 = np.asarray(d['r1']); lB = np.asarray(d['logB'])
    NB = lat.NB
    FY = np.full((M, NB), np.nan); DL = np.full((M, NB), np.nan); SY = np.zeros((M, NB), np.int8)
    rr, kk = np.nonzero(am); b = a[rr, kk]
    FY[rr, b] = fy[rr, kk]; DL[rr, b] = np.log(np.abs(r1[rr, kk])); SY[rr, b] = sy[rr, kk].astype(np.int8)
    return dict(lnB=lB.real, fx=np.asarray(fx), sx=sx.astype(np.int8), fy=FY, dl=DL, sy=SY, n_sign=nu1 + nu2)


# ============================================================================================ task: sample + measure
def task(a):
    import pp2_lemma as PL
    import pool8 as P8
    t0 = time.time(); betas = [float(b) for b in a.betas.split(',')]
    logf = open(a.out + '.log', 'a')
    def log(m): s = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} [{time.time() - t0:8.0f}s] {m}'; logf.write(s + '\n'); logf.flush(); print(s, flush=True)
    if a.problem == 8:
        assert sha(PROD8) == PROD8_SHA
        p = PL.PP2.from_rbmpp_npz(PROD8); ev = P8.PP2Eval(p)
        al, E, w = m1_consts(); al = al if not a.alpha0 else 0.0
        sfn = make_sG8_np(al, E, w); EP = EP8
    else:
        ev = P8.TableEval6(); p = P8.load_eval(6, 'pp2').p; sfn = TableSign6(ev); EP = EP6
    log(f'START task problem {a.problem} betas {betas} E {a.E} nper {a.nper} seed {a.seed} thin {a.thin} host {os.uname().nodename} path {a.eval_path}')
    S = P8.Pool8Sampler(ev, betas=betas, E=a.E, seed=a.seed, thin=a.thin, burn=a.burn, pilot=a.pilot, log=log, start='random')
    rec = {k: [] for k in ('beta', 'lnB', 'fx', 'sx', 'fy', 'dl', 'sy', 'x', 'sec')}
    ck = a.out
    nd = 0; tmeas = 0.0
    mlist = [b for b in betas for _ in range(1)] if not a.mbetas else [float(t.split(':')[0]) for t in a.mbetas.split(',') for _ in range(int(t.split(':')[1]))]
    log(f'measured rungs per round {mlist}')
    while nd < a.nper:
        for b in mlist:
            c, l = S.draw_cfg(b, a.E)
            tt = time.time(); n0 = getattr(sfn, 'n', 0)
            r = fj_raw(p, c, sfn, EP, a.eval_path, full=False, side=a.side); dt = time.time() - tt; tmeas += dt
            if a.problem == 6: assert np.allclose(r['lnB'] - r['lnB'][0], l - l[0], atol=1e-8), 'pp2 lnB != table lnB (up to a constant)'
            for k in ('lnB', 'fx', 'sx', 'fy', 'dl', 'sy'): rec[k].append(r[k])
            rec['beta'].append(np.full(len(c), b)); rec['x'].append(c.astype(np.int8)); rec['sec'].append(np.full(len(c), dt / len(c)))
            log(f'PROGRESS draw {nd + 1}/{a.nper} beta {b} lnB {np.round(l, 2).tolist()} fx {np.round(r["fx"], 3).tolist()} '
                f'{dt / len(c):.1f} s/config sign evals {r["n_sign"]}')
        nd += 1
        out = {k: np.concatenate(v) for k, v in rec.items()}
        np.savez(ck + '.tmp.npz', **out, betas=np.array(betas), seed=a.seed, problem=a.problem, med_pilot=S.med2, thin=S.thin, tau=json.dumps(S.tau),
                 sweeps=S.sweeps_done, side=a.side, sec_meas=tmeas, sec_total=time.time() - t0, done=nd >= a.nper)
        os.replace(ck + '.tmp.npz', ck)
        if time.time() - t0 > a.maxsec: log(f'MAXSEC after {nd} draws'); break
    log(f'TASK_DONE draws {nd} configs {len(out["lnB"])} sec_meas {tmeas:.0f} sec_total {time.time() - t0:.0f} sweeps {S.sweeps_done} thin {S.thin}')


# ============================================================================================ sign check
def sigcheck(a):
    import pp2_lemma as PL
    al, E, w = m1_consts(); assert sha(PROD8) == PROD8_SHA
    fs = sorted(glob.glob('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/pool8/meas8/out_*.npz'))
    cfg = []; bet = []
    for f in fs:
        z = np.load(f); c = z['cfg']; cfg.append(c.reshape(-1, 64)); bet.append(np.broadcast_to(z['betas'][None, :, None], c.shape[:3]).reshape(-1))
    cfg = np.concatenate(cfg); bet = np.concatenate(bet)
    rng = np.random.default_rng(5); idx = np.concatenate([rng.choice(np.nonzero(bet == b)[0], a.n // 3, replace=False) for b in (1.0, 0.5, 0.25)])
    X = cfg[idx].astype(np.float64)
    p = PL.PP2.from_rbmpp_npz(PROD8)
    t = time.time(); s_own = make_sG8(p, al, E, w)(X); t_own = time.time() - t
    t = time.time(); s_m1, cx, Ex, lx = m1_sign(X, al, E, w); t_m1 = time.time() - t
    s_brk = make_sG8(p, 0.0, E, w)(X)
    t = time.time(); s_np = make_sG8_np(al, E, w)(X); t_np = time.time() - t
    s_npbrk = make_sG8_np(0.0, E, w)(X)
    lB, r = PL.one_swap_ratios(p, X, return_logB=True)
    lat = p.lat; EL = 0.25 * (np.asarray(lat.JB)[None] * X[:, lat.BI] * X[:, lat.BJ]).sum(1) + (np.asarray(lat.HW)[None] * np.asarray(r)).sum(1)
    res = dict(n=int(len(X)), per_beta={str(b): int((bet[idx] == b).sum()) for b in (1.0, 0.5, 0.25)}, alpha=al, E=E, w=[w.real, w.imag],
               agree=int((s_own == s_m1).sum()), disagree_frac=float(np.mean(s_own != s_m1)),
               broken_alpha0_disagree_frac=float(np.mean(s_brk != s_m1)), broken_alpha0_disagree=int((s_brk != s_m1).sum()),
               EL_maxabs_dev=float(np.max(np.abs(EL - Ex))), lnpsi_re_dev_spread=float(np.ptp(np.asarray(lB).real - lx.real)),
               min_abs_cx_over_med=float(np.min(np.abs(cx)) / np.median(np.abs(cx))), frac_minus_sG=float(np.mean(s_m1 < 0)),
               sec_own=t_own, sec_m1=t_m1, configs_from=fs[0].rsplit('/', 1)[0], code_sha=dict(decades8=sha(__file__), pp2_lemma=sha(f'{HERE}/pp2_lemma.py'),
               gfmc_fn_lib=sha(f'{M1CODE}/gfmc_fn_lib.py'), rbmpp_lib=sha(f'{M1CODE}/rbmpp_lib.py')))
    res.update(np_disagree=int((s_np != s_m1).sum()), np_broken_alpha0_disagree=int((s_npbrk != s_m1).sum()), sec_np=t_np)
    res['PASS'] = bool(res['disagree_frac'] == 0.0 and res['broken_alpha0_disagree'] > 0 and res['np_disagree'] == 0 and res['np_broken_alpha0_disagree'] > 0)
    json.dump(res, open(a.outjson, 'w'), indent=1); print(json.dumps(res, indent=1))


# ============================================================================================ beta = 0 draws of f_J at x (f0)
def beta0(a):
    """i.i.d. uniform Sz = 0 configurations (beta = 0: per-configuration uniform = n-weighted over orbits); raw f_J at x only
    (pp2_lemma.fJ_fixed at_y = False, mode 'old' arithmetic; s_G at x and its one-swap neighbours).  Checkpoint every chunk."""
    import pp2_lemma as PL
    import pool8 as P8
    t0 = time.time(); rng = np.random.default_rng(a.seed)
    if a.problem == 8:
        assert sha(PROD8) == PROD8_SHA
        p = PL.PP2.from_rbmpp_npz(PROD8); al, E, w = m1_consts(); sfn = make_sG8_np(al, E, w); EP = EP8; N = 64
    else:
        ev = P8.TableEval6(); p = P8.load_eval(6, 'pp2').p; sfn = TableSign6(ev); EP = EP6; N = 36
    logf = open(a.out + '.log', 'a')
    def log(m): s = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} [{time.time() - t0:8.0f}s] {m}'; logf.write(s + '\n'); logf.flush()
    log(f'START beta0 problem {a.problem} n {a.n} chunk {a.chunk} seed {a.seed} host {os.uname().nodename}')
    fx = []; lnB = []
    while sum(len(v) for v in fx) < a.n:
        X = np.ones((a.chunk, N), np.int8)
        for i in range(a.chunk): X[i, rng.permutation(N)[:N // 2]] = -1
        o = PL.fJ_fixed(p, X, sfn, EP, 1.0, mode='old', at_y=False)
        fx.append(np.asarray(o['fx'])); lnB.append(np.asarray(o['logB']).real)
        f = np.concatenate(fx); np.savez(a.out + '.tmp.npz', fx=f, lnB=np.concatenate(lnB), seed=a.seed, problem=a.problem); os.replace(a.out + '.tmp.npz', a.out)
        log(f'PROGRESS {len(f)}/{a.n} |f| q.999 {np.quantile(np.abs(f[np.isfinite(f)]), 0.999):.4f} max {np.nanmax(np.abs(f[np.isfinite(f)])):.4f} nonfinite {int((~np.isfinite(f)).sum())}')
    log(f'TASK_DONE n {len(f)} sec {time.time() - t0:.0f}')


# ============================================================================================ f_J check
def fjcheck(a):
    """fj_raw (blocked-only signs) vs pp2_lemma.fJ_fixed (all y) on the same configs; 6x6: also vs the exact table fJ_theta4 (orbit lookup of x, y).
    Broken version: E' shifted by +1 must fail."""
    import pp2_lemma as PL
    import pool8 as P8
    if a.problem == 8:
        p = PL.PP2.from_rbmpp_npz(PROD8); al, E, w = m1_consts(); sfn = make_sG8_np(al, E, w); EP = EP8
        z = np.load('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/pool8/meas8/out_10.npz'); X = z['cfg'][-1].reshape(-1, 64)[:a.n]
    else:
        ev = P8.TableEval6(); p = P8.load_eval(6, 'pp2').p; sfn = TableSign6(ev); EP = EP6
        S = P8.Pool8Sampler(ev, betas=[1.0, 0.5, 0.25], E=4, seed=11, burn=100, pilot=100, log=lambda m: None)
        X = np.concatenate([S.draw_cfg(b, 4)[0] for b in (1.0, 0.5, 0.25)])[:a.n]
    X = X.astype(np.int8); res = dict(problem=a.problem, n=int(len(X)))
    t = time.time(); r = fj_raw(p, X, sfn, EP); res['sec_blocked'] = time.time() - t; res['n_sign_blocked'] = int(r['n_sign'])
    t = time.time(); rf = fj_raw(p, X, sfn, EP, full=True); res['sec_full'] = time.time() - t; res['n_sign_full'] = int(rf['n_sign'])
    o = PL.fJ_fixed(p, X, sfn, EP, 1.0, mode='old', at_y=True, eval_path='dedup')
    am = np.asarray(o['am']); aa = np.asarray(o['a']); rr, kk = np.nonzero(am); fyo = np.full_like(rf['fy'], np.nan); fyo[rr, aa[rr, kk]] = np.asarray(o['fy'])[rr, kk]
    fin = np.isfinite(fyo)
    res['full_vs_fJfixed_fx'] = float(np.nanmax(np.abs(rf['fx'] - np.asarray(o['fx']))))
    res['full_vs_fJfixed_fy'] = float(np.max(np.abs(rf['fy'][fin] - fyo[fin]))); res['n_fy_nonfinite_ref'] = int((np.isfinite(rf['dl']) & ~fin).sum())
    b = np.isfinite(r['fy']); res['n_blocked'] = int(b.sum()); res['n_valid'] = int(np.isfinite(r['dl']).sum())
    res['blocked_vs_full_fy'] = float(np.max(np.abs(r['fy'][b] - rf['fy'][b]))); res['blocked_vs_full_fx'] = float(np.max(np.abs(r['fx'] - rf['fx'])))
    rb = fj_raw(p, X, sfn, EP + 1.0); res['broken_EPplus1_fy_dev'] = float(np.nanmax(np.abs(rb['fy'][b] - r['fy'][b])))
    if a.problem == 6:
        T = np.load(FJ6); ox = ev.orbit(X); res['table_fx_dev'] = float(np.max(np.abs(r['fx'] - T[ox])))
        lat = p.lat; dev = []
        for i, j in zip(*np.nonzero(b)):
            y = X[i].copy(); y[lat.BI[j]], y[lat.BJ[j]] = X[i, lat.BJ[j]], X[i, lat.BI[j]]; dev.append(abs(r['fy'][i, j] - T[ev.orbit(y[None])[0]]))
        res['table_fy_dev'] = float(max(dev)); res['table_n_fy'] = len(dev)
        res['table_sx_dev'] = int((r['sx'] != sfn.s[ox]).sum())
    tol = 1e-9
    res['PASS'] = bool(res['full_vs_fJfixed_fx'] < tol and res['full_vs_fJfixed_fy'] < tol and res['blocked_vs_full_fy'] < tol and res['broken_EPplus1_fy_dev'] > 1e-3
                       and (a.problem == 8 or (res['table_fx_dev'] < 1e-9 and res['table_fy_dev'] < 1e-9)))
    json.dump(res, open(a.outjson, 'w'), indent=1); print(json.dumps(res, indent=1))


# ============================================================================================ analysis
def mbar_logZ(lnb_list, betas, iters=5000, tol=1e-12):
    """binless multi-histogram (MBAR) for pi_k ~ exp(2 beta_k lnB): returns f_k = ln Z_k (up to a common constant, f_0 = 0) and per-sample log
    weights to the target beta = 1: ln w_i = 2 lnB_i - ln sum_k N_k exp(2 beta_k lnB_i - f_k)."""
    lnb = np.concatenate(lnb_list); Nk = np.array([len(l) for l in lnb_list], float); bk = np.asarray(betas)
    u = 2.0 * bk[:, None] * lnb[None, :]                                    # (K, n)
    f = np.zeros(len(bk))
    for it in range(iters):
        den = np.logaddexp.reduce(np.log(Nk)[:, None] + u - f[:, None], axis=0)
        fn = np.logaddexp.reduce(u - den[None, :], axis=1); fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol: f = fn; break
        f = fn
    den = np.logaddexp.reduce(np.log(Nk)[:, None] + u - f[:, None], axis=0)
    lw = 2.0 * lnb - den
    return f, lw


def per_draw_entries(Z, c, f0, mode, plain=False, node=-69.0, conv='small'):
    """per draw: lists of (dl = ln|B_y/B_x|, hw, dd2) over blocked bonds.  returns arrays (M, NB) masks and values."""
    N = Z['x'].shape[1]; HW = np.concatenate([np.full(2 * N, 0.5), np.full(2 * N, 0.25)])    # pp2_lemma.Lat: J1 bonds then J2 bonds, |H| = J/2
    fx = Z['fx']; fy = Z['fy']; sx = Z['sx'].astype(float); sy = Z['sy'].astype(float); dl = Z['dl']
    valid = np.isfinite(dl)
    blk = valid & (sx[:, None] * sy > 0) & ((dl >= 0) if conv == 'small' else (dl <= 0))   # fy is computed only on this side
    assert not np.isnan(fy[blk]).any(), 'f_J(y) missing on a needed blocked bond'
    def add(f):
        fin = np.isfinite(f)
        if mode == 'old': return c * np.where(fin, f, 0.0)
        if mode == 'fix': return np.where(fin, c * np.sign(f) * np.minimum(np.abs(np.where(fin, f, 0.0)), f0), node)
        raise ValueError(mode)
    ax = add(fx); ay = add(np.where(valid, fy, 0.0))
    dd2 = (ax[:, None] - ay) ** 2 if not plain else np.ones_like(ay)
    return blk, np.where(valid, dl, 0.0), HW[None, :] * np.ones_like(dl), dd2


def shares_from(lnB, beta, betas, blk, dl, hw, dd2, med, dec_off=None, conv='small', est='cell', d_lo=-30, d_hi=6):
    """decade shares of the |B|^2-weighted total.  est 'mbar': all draws reweighted by MBAR; 'cell': per (beta, decade) self-normalised IS
    (weight |B|^(2 - 2 beta)) for the conditional mean, mixed over betas by ESS (pool_fit.pool_weights), times the MBAR decade mass."""
    lst = [lnB[beta == b] for b in betas]
    f, lw_k = mbar_logZ(lst, betas)
    lw = np.empty_like(lnB); off = 0
    order = np.concatenate([np.nonzero(beta == b)[0] for b in betas]); lw[order] = lw_k
    ref = med if dec_off is None else dec_off
    decx = np.floor((2.0 * lnB - ref) / LN10).astype(np.int64)
    term = blk * hw * np.exp(dl) * dd2
    D = np.arange(d_lo, d_hi + 1); wt = np.exp(lw - lw.max())
    # per-draw contribution vector over target decades (the pair goes to the decade of its SMALLER-|B| side)
    if conv == 'small':                     # seen from the smaller end x: target decade = decade(x)
        tw = np.where(dl > 0, term, np.where(dl == 0, 0.5 * term, 0.0)); tdec = np.broadcast_to(decx[:, None], tw.shape)
    else:                                   # seen from the larger end x: target decade = decade(y), lnB_y = lnB_x + dl
        tw = np.where(dl < 0, term, np.where(dl == 0, 0.5 * term, 0.0)); tdec = np.floor((2.0 * (lnB[:, None] + dl) - ref) / LN10).astype(np.int64)
    tdec = np.clip(tdec, d_lo, d_hi) - d_lo
    H = np.zeros((len(lnB), len(D)))
    np.add.at(H, (np.broadcast_to(np.arange(len(lnB))[:, None], tw.shape), tdec), tw)
    mass = np.array([wt[decx == d].sum() for d in D]); mass /= mass.sum()
    if est == 'mbar':
        tot = (wt[:, None] * H).sum(0) / wt.sum()
    else:
        tot = np.zeros(len(D))
        for j, d in enumerate(D):          # cells = (beta, decade of x): conditional mean of the contribution vector, times the x-decade mass
            md = decx == d; parts = []; E = []
            for b in betas:
                m = md & (beta == b)
                if not m.any(): continue
                l = (2.0 - 2.0 * b) * lnB[m]; ww = np.exp(l - l.max()); ww /= ww.sum(); parts.append(ww @ H[m]); E.append(1.0 / np.sum(ww * ww))
            if E: tot += mass[j] * (np.array(E) @ np.array(parts)) / np.sum(E)
    sh = tot / tot.sum()
    return D, sh, mass, dict(f=f.tolist(), n_per_dec=[int((decx == d).sum()) for d in D])


def load_runs(files):
    Zs = []
    for fn in files:
        z = np.load(fn, allow_pickle=True); Zs.append({k: z[k] for k in ('beta', 'lnB', 'fx', 'sx', 'fy', 'dl', 'sy', 'x', 'sec')} | dict(betas=z['betas'], file=fn))
    return Zs


def cat(Zs, keys=('beta', 'lnB', 'fx', 'sx', 'fy', 'dl', 'sy', 'x', 'sec')):
    return {k: np.concatenate([Z[k] for Z in Zs]) for k in keys}


def contiguous95(D, sh, thr=0.95):
    best = None
    for i in range(len(D)):
        for j in range(i, len(D)):
            s = sh[i:j + 1].sum()
            if s >= thr:
                if best is None or (j - i) < (best[1] - best[0]) or ((j - i) == (best[1] - best[0]) and s > best[2]): best = (i, j, s)
                break
    return int(D[best[0]]), int(D[best[1]]), float(best[2])


def analyze(a):
    files = sorted(sum([glob.glob(f) for f in a.files], []))
    Zs = load_runs(files); Z = cat(Zs); betas = sorted({float(b) for b in np.unique(Z['beta'])}, reverse=True)      # union over files
    res = dict(files=files, n_files=len(files), n_draws=int(len(Z['lnB'])), betas=betas, c=a.c, mode=a.mode, plain=a.plain, problem=a.problem,
               sec_meas_sum=float(Z['sec'].sum()), sec_per_config_mean=float(Z['sec'].mean()))
    med = float(np.median(2.0 * Z['lnB'][Z['beta'] == 1.0])); res['med2lnB_beta1'] = med
    if a.f0 == 'auto':
        fx1 = np.abs(Z['fx'][(Z['beta'] == 1.0) & np.isfinite(Z['fx'])]); f0 = float(np.quantile(fx1, 0.9999)); res['f0_n_beta1'] = int(fx1.size)
        res['fx_beta1_quantiles'] = {q: float(np.quantile(fx1, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999)}; res['fx_beta1_max'] = float(fx1.max())
    else: f0 = float(a.f0)
    res['f0'] = f0
    res['n_nonfinite_fx'] = int((~np.isfinite(Z['fx'])).sum()); res['n_nonfinite_fy_valid'] = int((np.isfinite(Z['dl']) & ~np.isfinite(Z['fy'])).sum())
    dec_off = None
    if a.problem == 6:
        n6 = np.load(NORB6).astype(np.float64); la4 = np.load('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/tab_t4/round_1/la_table.npy').astype(np.float64)
        lz = np.logaddexp.reduce(np.log(n6) + 2.0 * la4); dec_off = float(lz)   # 2 lnB_table - lz = log |B|^2 normalised (table lnB = sampler lnB)
        res['abs_offset_lnZ'] = lz
        # exact |B|^2 decade masses
        dex = np.floor((2 * la4 - lz) / LN10).astype(np.int64); wex = np.exp(np.log(n6) + 2 * la4 - lz)
        res['exact_mass'] = {int(d): float(wex[dex == d].sum()) for d in np.unique(dex)}
    def one(Zsub, est, conv, plain):
        blk, dl, hw, dd2 = per_draw_entries(Zsub, a.c, f0, a.mode, plain, conv=conv)
        md = float(np.median(2.0 * Zsub['lnB'][Zsub['beta'] == 1.0]))
        return shares_from(Zsub['lnB'], Zsub['beta'], betas, blk, dl, hw, dd2, md, dec_off, conv, est)
    out = {}
    for est in ('cell', 'mbar'):
        for conv in (a.conv,):
            D, sh, mass, info = one(Z, est, conv, a.plain)
            # jackknife over files (independent PT ensembles / processes)
            J = []
            for i in range(len(Zs)):
                Zj = cat([Zs[k] for k in range(len(Zs)) if k != i]); J.append(one(Zj, est, conv, a.plain)[1])
            J = np.array(J); n = len(Zs); err = np.sqrt((n - 1) / n * ((J - J.mean(0)) ** 2).sum(0))
            lo, hi, s95 = contiguous95(D, sh)
            out[f'{est}_{conv}'] = dict(dec=D.tolist(), share=sh.tolist(), err=err.tolist(), mass=mass.tolist(), info=info, range95=[hi, lo], range95_share=s95)
    res['est'] = out
    if a.problem == 6:                                                       # control: compare with Test W (iii) (exact, all orbits)
        W = json.load(open('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_w/w_result.json'))['part1']['rows']; ws = {r['decade']: r['share'] for r in W}
        r = out[f'cell_{a.conv}']; cmp = []
        for d, s_, e_ in zip(r['dec'], r['share'], r['err']):
            ref = ws.get(d, 0.0)
            if ref > 1e-3 or s_ > 1e-3: cmp.append(dict(decade=d, share=s_, err=e_, testW=ref, z=(s_ - ref) / e_ if e_ > 0 else None))
        zz = [abs(c_['z']) for c_ in cmp if c_['z'] is not None and c_['testW'] >= 0.02]
        res['vs_testW'] = dict(rows=cmp, max_abs_z_ge2pct=max(zz) if zz else None, chi2=float(sum(z_ * z_ for z_ in zz)), ndec=len(zz),
                               testW_range95=list(contiguous95(np.array(sorted(ws)), np.array([ws[k] for k in sorted(ws)]))))
        rm = out[f'mbar_{a.conv}']; res['mass_vs_exact'] = [dict(decade=d, mbar=m, exact=res['exact_mass'].get(d, 0.0)) for d, m in zip(rm['dec'], rm['mass'])
                                                        if m > 1e-4 or res['exact_mass'].get(d, 0.0) > 1e-4]
        print('vs Test W:', json.dumps(res['vs_testW'], default=float)[:3000])
    # half split (files odd / even) for the quotable gate
    for h, sel in (('A', slice(0, None, 2)), ('B', slice(1, None, 2))):
        Zh = Zs[sel]; Zc = cat(Zh); D, sh, mass, info = one(Zc, 'cell', a.conv, a.plain)
        J = np.array([one(cat([Zh[k] for k in range(len(Zh)) if k != i]), 'cell', a.conv, a.plain)[1] for i in range(len(Zh))]); n = len(Zh)
        err = np.sqrt((n - 1) / n * ((J - J.mean(0)) ** 2).sum(0)); res[f'half_{h}'] = dict(dec=D.tolist(), share=sh.tolist(), err=err.tolist())
        if a.halves_prefix:
            keep = np.array([out[f'cell_{a.conv}']['share'][out[f'cell_{a.conv}']['dec'].index(int(d))] >= 0.005 for d in D])   # decades with >= 0.5% share (full data)
            np.savetxt(f'{a.halves_prefix}_{h}.txt', np.c_[D, sh, err][keep], header='decade share err (decades with >= 0.5% share in the full data)')
    json.dump(res, open(a.outjson, 'w'), indent=1, default=float)
    r = out[f'cell_{a.conv}']
    print(f'f0 {f0:.4f}  med {med:.3f}  draws {res["n_draws"]}  range95 {r["range95"]} ({r["range95_share"]:.3f})')
    for d, s, e, m, nn in zip(r['dec'], r['share'], r['err'], r['mass'], r['info']['n_per_dec']):
        if s > 1e-4 or m > 1e-4: print(f'{d:4d} share {s:.4f} +- {e:.4f}  mass {m:.3e}  n {nn}  mbar {out[f"mbar_{a.conv}"]["share"][r["dec"].index(d)]:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd')
    s = sp.add_parser('sigcheck'); s.add_argument('outjson'); s.add_argument('--n', type=int, default=240)
    t = sp.add_parser('task'); t.add_argument('--problem', type=int, default=8); t.add_argument('--out', required=True); t.add_argument('--betas', default='1,0.5,0.25,0.125')
    t.add_argument('--E', type=int, default=2); t.add_argument('--nper', type=int, default=10); t.add_argument('--seed', type=int, default=1); t.add_argument('--thin', type=int, default=None)
    t.add_argument('--burn', type=int, default=200); t.add_argument('--pilot', type=int, default=200); t.add_argument('--maxsec', type=float, default=1e18)
    t.add_argument('--alpha0', action='store_true'); t.add_argument('--side', default='large', choices=['small', 'large', 'all']); t.add_argument('--mbetas', default=None); t.add_argument('--eval_path', default='dedup')
    b0 = sp.add_parser('beta0'); b0.add_argument('--problem', type=int, default=8); b0.add_argument('--out', required=True); b0.add_argument('--n', type=int, default=1000); b0.add_argument('--chunk', type=int, default=64); b0.add_argument('--seed', type=int, default=1)
    k = sp.add_parser('fjcheck'); k.add_argument('outjson'); k.add_argument('--problem', type=int, default=8); k.add_argument('--n', type=int, default=4)
    z = sp.add_parser('analyze'); z.add_argument('outjson'); z.add_argument('files', nargs='+'); z.add_argument('--problem', type=int, default=8); z.add_argument('--c', type=float, default=4.0)
    z.add_argument('--f0', default='auto'); z.add_argument('--mode', default='fix'); z.add_argument('--plain', action='store_true'); z.add_argument('--halves_prefix', default=None); z.add_argument('--conv', default='small', choices=['small', 'large'])
    a = ap.parse_args()
    {'sigcheck': sigcheck, 'task': task, 'analyze': analyze, 'fjcheck': fjcheck, 'beta0': beta0}[a.cmd](a)
