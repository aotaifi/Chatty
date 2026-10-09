"""Shared machinery for the compression pre-tests (experiments/compress_6x6, README.md).

Exact 6x6 J1-J2 (J2/J1 = 0.5) symmetric sector (st6_sector.py, D = 15.8M orbit representatives).
The wtLOOP8 guide stack (writeback_tail, 8 stored FEAT nets) is replayed read-only; the FEAT net, the guide-only
quantities (V, W, semi-implicit target T) and the guide-metric proposal are copied from writeback_tail/wt_loop.py
(that file runs its loop at import, so it is not imported).

Conventions (as wt_loop.py):
  guide (la, s): per-rep log-amplitude and sign; V(x) = sum_{violating y} |H_xy| a_y/a_x (FN wall term),
  W(x) = sum_{kept y} |H_xy| a_y/a_x;  T = log(1 + W) - log(1 + (H_xx + V - E_g)),  E_g = <H>(la, s).
  frac of a write-back f from guide g = (E_g - <a e^f|H_FN[g]|a e^f>) / (E_g - E_FN[g])  (exact frozen-FN identity).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for p_ in (HERE, os.path.join(HERE, '..', 'stall_6x6'), os.path.join(HERE, '..', 'learned_loop_6x6'),
           os.path.join(HERE, '..', 'lanczos_baseline_6x6')):
    if os.path.isdir(p_) and p_ not in sys.path: sys.path.insert(0, p_)
import st6_sector as SS
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.flatten_util import ravel_pytree
jax.config.update('jax_default_matmul_precision', 'highest')          # ViT and CNNs at full fp32

log = SS.log
N = 36
f64 = jnp.float64
ROOT = '/project/theorie/a/A.Otaifi/chatty_stall6'
CSR = os.environ.get('ST6_CSR', ROOT + '/csr')
TABLE = os.environ.get('ST6_TABLE', ROOT + '/data/psi0_6x6_table.npz')
SYM = os.environ.get('ST6_SYM', ROOT + '/data/sym_tables.npz')
LOOP8 = '/project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/wtLOOP8_17090119'     # read-only
TABDIR = os.environ.get('CX_TABDIR', '/project/theorie/a/A.Otaifi/chatty_compress6/tables')
VIT_CKPT = '/project/theorie/a/A.Otaifi/chatty_stall6/data/vit_J2=0.50_N=6x6_k=0.mpack'
H_START = 1.3193875041276706e-4                                        # <H>(psi_P, s_P) per site, exact
LANCZOS_P1_PSIP = 2.55e-5                                              # lanczos_baseline_6x6
RBM_PP = 4.47e-5

AR = jnp.arange(N, dtype=jnp.uint64)
BI = jnp.asarray(SS.BI); BJ = jnp.asarray(SS.BJ); JB = jnp.asarray(SS.JB); MASKS = jnp.asarray(SS.MASKS)
BI_I = jnp.asarray(SS.BI.astype(np.int32)); BJ_I = jnp.asarray(SS.BJ.astype(np.int32))
NB = int(SS.NBOND)
BTYPE = jnp.asarray(np.array([(b % 2) + 2 * (b >= len(SS.NN)) for b in range(NB)], np.int32))  # NNx, NNy, d1, d2
NBR = jnp.asarray(np.array([[((x + dx) % 6) + 6 * ((y + dy) % 6) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                            for y in range(6) for x in range(6)], np.int32))
INV8 = jnp.asarray(np.argsort(SS.space_group()[:8], axis=1))
SG2 = jnp.asarray([1.0, -1.0], jnp.float32)


def dump(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, default=float)


def bits(Sx):
    return (((Sx[:, None] >> AR[None, :]) & jnp.uint64(1)).astype(jnp.float32) * 2 - 1)


def valid_bonds(x):
    return (((x[:, None] >> BI[None, :]) ^ (x[:, None] >> BJ[None, :])) & jnp.uint64(1)).astype(bool)


# ============================================================================================ exact lab
class Lab:
    def __init__(self):
        t0 = time.time()
        self.sec = sec = SS.Sector(CSR, TABLE)
        sec.v0 = sec.la0 = sec.s0 = sec.p0 = None
        z = np.load(SYM); self.lP0 = jnp.asarray(z['lP']); self.sP0 = jnp.asarray(z['sP'].astype(np.float32)); del z
        self.D = sec.D
        self.TIDX = jnp.arange(sec.D, dtype=jnp.float32)
        self.D0 = self._diag()
        # 20% of the reps held out (same hash as writeback_tail/wt_run.py)
        self.HO = (((jnp.arange(sec.D, dtype=jnp.uint32) * jnp.uint32(2654435761)) >> 7) % 5) == 0
        log(f'[lab] ready {time.time() - t0:.0f}s')

    def _diag(self, batch=1 << 20):
        sec = self.sec

        @jax.jit
        def f(x):
            valid = valid_bonds(x).astype(f64)
            return 0.25 * JB.sum() - 0.5 * (valid * JB[None, :]).sum(1)
        return jnp.concatenate([f(sec.reps[i:i + batch]) for i in range(0, sec.D, batch)])

    def Kop(self, x, s):
        h1, h2 = self.sec.H2(x, s * x)
        return 0.5 * (h1 - s * h2)

    def H_site(self, la, s):
        sec = self.sec
        return (sec.energy(sec.vec(la, s)) - sec.E0) / N

    def VW(self, la, s):
        """guide-only one-hop quantities (exact, via two sector matvecs)."""
        sec = self.sec
        lan = la - jnp.max(la)
        w = jnp.maximum(jnp.exp(lan), 1e-300) * sec.sqrt_n
        h1, h2 = sec.H2(w, s * w)
        V = 0.5 * (h1 + s * h2) / w - self.D0; W = 0.5 * (h1 - s * h2) / w
        return lan, V, W

    def T_of(self, V, W, Eg):
        return jnp.log1p(jnp.maximum(W, 0.0)) - jnp.log1p(jnp.maximum(self.D0 + V - Eg, 1e-12))

    def UA(self, la):
        ua = jnp.exp(la - jnp.max(la)) * self.sec.sqrt_n
        return ua / jnp.linalg.norm(ua)

    def proposal_q(self, target, la, s):
        """guide-metric tail proposal (wt_loop.proposal): Q = 1/2 n a + 1/2 node Dirichlet weight of `target` in the
        metric of guide (la, s).  Returns (Q, PA, meas)."""
        sec = self.sec
        UA = self.UA(la)
        KUA = self.Kop(UA, s); PA = UA * UA
        mu = float(jnp.sum(PA * target)); e = target - mu
        K1 = self.Kop(UA * e, s); CT = e * e * UA * KUA - 2 * e * UA * K1; del K1
        CT = jnp.maximum(0.5 * (CT + UA * self.Kop(UA * e * e, s)), 0.0)
        qb = jnp.exp(la - jnp.max(la)) * sec.n; qb = qb / jnp.sum(qb)
        Q = 0.5 * qb + 0.5 * CT / jnp.sum(CT)
        meas = 0.5 * PA + 0.5 * CT / jnp.sum(CT)
        return Q, PA, meas

    @staticmethod
    def feat_table(lan, V, W, meas):
        cols = []
        for v_ in (lan, jnp.log(jnp.maximum(V, 0) + 1e-6), jnp.log(jnp.maximum(W, 0) + 1e-6)):
            m_ = float(jnp.sum(meas * v_)); s_ = float(jnp.sqrt(jnp.sum(meas * (v_ - m_) ** 2))) + 1e-30
            cols.append(((v_ - m_) / s_).astype(jnp.float32))
        return jnp.stack(cols, 1)

    def frozen(self, la_g, s, la_b):
        return self.sec.fn_rayleigh(la_g, s, la_b)

    def decade_of(self, la):
        """per-configuration probability decade of a^2 (normalised: sum_reps n a^2 = 1)."""
        lan = la - jnp.max(la)
        lz = jnp.log(jnp.sum(self.sec.n * jnp.exp(2 * lan)))
        lp = 2 * lan - lz
        return jnp.floor(lp / np.log(10.0)).astype(jnp.int32), lp


# ============================================================================================ FEAT net (wt_loop copy)
class ResCNN(nn.Module):
    C: int = 32
    layers: int = 4
    use_a: bool = False
    hid: int = 64
    nfeat: int = 1

    @nn.compact
    def __call__(self, x, t):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(g.shape[0], N, -1))
        h = nn.gelu(conv(x[:, :, None]))
        for _ in range(1, self.layers):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        h = h.sum(axis=1) / 6.0
        if self.use_a:
            om = jnp.asarray([0.25, 0.5, 1.0, 2.0, 4.0], jnp.float32)
            tt = t[:, None] if t.ndim == 1 else t
            B_ = tt.shape[0]
            emb = jnp.concatenate([tt, 0.1 * tt * tt, jnp.sin(tt[:, :, None] * om).reshape(B_, -1),
                                   jnp.cos(tt[:, :, None] * om).reshape(B_, -1)], 1)
            h = jnp.concatenate([h, emb], 1)
            h = nn.gelu(nn.Dense(self.hid, **kw)(h))
            h = nn.gelu(nn.Dense(self.hid, **kw)(h))
        return nn.Dense(1, kernel_init=nn.initializers.zeros, **kw)(h)[:, 0]


class FeatNet:
    """the wt_loop FEAT net (C=32, 4 layers, inputs log a, V, W looked up by rep index), exactly D4 x flip symmetric."""

    def __init__(self, feat, seed=0):
        nf = int(feat.shape[1])
        self.net = ResCNN(32, 4, True, nfeat=nf)
        params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros((1, N), jnp.float32), jnp.zeros((1, nf), jnp.float32))['params']
        self.flat0, self.unravel = ravel_pytree(params)
        self.npar = int(self.flat0.size)
        net0, unravel = self.net, self.unravel

        def f(flat, X, t):
            p = unravel(flat); ft = feat[t.astype(jnp.int32)]
            out = jax.lax.map(lambda k: net0.apply({'params': p}, X[:, INV8[k // 2]] * SG2[k % 2], ft), jnp.arange(16))
            return out.mean(0)

        def f_aug(flat, X, t, G):
            p = unravel(flat); ft = feat[t.astype(jnp.int32)]
            Xg = jnp.take_along_axis(X, INV8[G // 2], axis=1) * SG2[G % 2][:, None]
            return net0.apply({'params': p}, Xg, ft)
        self.f, self.f_aug = f, f_aug
        self._tab = jax.jit(lambda flat, Sx, t: f(flat, bits(Sx), t))

    def table(self, flat, reps, T, batch=16384):
        D = reps.shape[0]; out = []
        for i in range(0, D, batch):
            s = reps[i:i + batch]; t = T[i:i + batch]; m = s.shape[0]
            if m < batch:
                s = jnp.concatenate([s, jnp.repeat(s[:1], batch - m)]); t = jnp.concatenate([t, jnp.repeat(t[:1], batch - m)])
            out.append(self._tab(flat, s, t)[:m].astype(f64))
        return jnp.concatenate(out)


def load_loop_params(k):
    return jnp.asarray(np.load(os.path.join(LOOP8, f'params_it{k}.npy')), jnp.float32)


def save_guide(k, la, s):
    os.makedirs(TABDIR, exist_ok=True)
    np.save(os.path.join(TABDIR, f'la_{k}.npy'), np.asarray(la, np.float64))
    np.save(os.path.join(TABDIR, f's_{k}.npy'), np.asarray(s, np.int8))


def load_guide(k):
    la = jnp.asarray(np.load(os.path.join(TABDIR, f'la_{k}.npy')))
    s = jnp.asarray(np.load(os.path.join(TABDIR, f's_{k}.npy')).astype(np.float32))
    return la, s


# ============================================================================================ one loop step (replay)
def loop_step(lab, la, s, params, Efn=None, seed=1, feats=None, T_override=None, krylov=True):
    """one wt_loop iteration from guide (la, s) with STORED FEAT params (no training).
    feats: optional (lan, V, W) replacing the exact guide quantities in the net input and in T (sensitivity / student
    features); the write-back is always applied to la.  Returns (la_new, s_new or None, rec)."""
    sec = lab.sec
    lan, V, W = lab.VW(la, s)
    Eg = lab.frozen(la, s, la)
    T = lab.T_of(V, W, Eg)
    _, _, meas = lab.proposal_q(T, la, s)
    if feats is not None:
        lan_f, V_f, W_f = feats
    else:
        lan_f, V_f, W_f = lan, V, W
    T_f = lab.T_of(V_f, W_f, Eg)
    FEAT = lab.feat_table(lan_f, V_f, W_f, meas)
    rec = {}
    if Efn is not None:
        rec['SI_target_frac'] = (Eg - lab.frozen(la, s, la + T_f)) / (Eg - Efn)
    del V, W, T, meas, V_f, W_f, T_f
    model = FeatNet(FEAT, seed=seed)
    fb = model.table(params, sec.reps, lab.TIDX)
    la_new = la + fb
    rec['rms_f_pa'] = float(jnp.sqrt(jnp.sum(lab.UA(la) ** 2 * (fb - jnp.sum(lab.UA(la) ** 2 * fb)) ** 2)))
    del fb, FEAT, model
    if Efn is not None:
        rec['frac'] = (Eg - lab.frozen(la, s, la_new)) / (Eg - Efn)
        rec['ideal_gain_site'] = (Eg - Efn) / N
    s_new = None
    if krylov:
        s_new, _, _ = sec.krylov(sec.vec(la_new, jnp.ones_like(la_new)), s)
        rec['H_kry'] = lab.H_site(la_new, s_new)
    return la_new, s_new, rec


# ============================================================================================ fidelity (sampled)
def sample_by_decade(lab, dec, M, seed, decs=range(-7, -16, -1)):
    """up to M reps per decade from the training and the held-out reps; decade -15 collects everything <= 1e-15."""
    rng = np.random.default_rng(seed)
    dec_h = np.asarray(dec); ho = np.asarray(lab.HO)
    out = {}
    for d in decs:
        m = (dec_h <= d) if d == -15 else (dec_h == d)
        for split, msk in (('train', m & ~ho), ('test', m & ho)):
            idx = np.nonzero(msk)[0]
            if idx.size == 0: continue
            out[(d, split)] = np.sort(rng.choice(idx, size=min(M, idx.size), replace=False))
    return out


def neighbours(lab, idx):
    """all bonds of reps idx: valid mask (B, NB), neighbour rep index (B, NB)."""
    sec = lab.sec
    x = sec.reps[idx]
    v = valid_bonds(x)
    y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
    iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
    return v, iy


def fidelity(lab, la_t, s_t, samples, ratio_fn, la_s=None, chunk=2048):
    """per decade and split: rms log-ratio error over valid bonds (weighted by |H_xy| a_y/a_x of the target, and
    unweighted), rms log(V_s/V), log(W_s/W), and the pointwise rms of la_s - la_t (if la_s given; mean removed in the
    a_t^2 metric).  ratio_fn(idx, v, iy) -> (B, NB) student log-ratios."""
    out = []
    if la_s is not None:
        PA = lab.UA(la_t) ** 2
        c0 = float(jnp.sum(PA * (la_s - la_t)))
    for (d, split), idx_all in samples.items():
        acc = dict(w=0.0, we=0.0, n=0, e2=0.0, lv=[], lw=[], pe=[])
        for i in range(0, idx_all.size, chunk):
            idx = jnp.asarray(idx_all[i:i + chunk])
            v, iy = neighbours(lab, idx)
            r = jnp.clip(la_t[iy] - la_t[idx][:, None], -60, 60)
            rs = ratio_fn(idx, v, iy)
            e = jnp.where(v, rs - r, 0.0)
            w = jnp.where(v, 0.5 * JB[None, :] * jnp.exp(r), 0.0)
            acc['w'] += float(jnp.sum(w)); acc['we'] += float(jnp.sum(w * e * e))
            acc['n'] += int(jnp.sum(v)); acc['e2'] += float(jnp.sum(e * e))
            kept = v & (s_t[idx][:, None] * s_t[iy] < 0); viol = v & ~kept
            ws = jnp.where(v, 0.5 * JB[None, :] * jnp.exp(jnp.clip(rs, -60, 60)), 0.0)
            Vt = jnp.sum(jnp.where(viol, w, 0), 1); Wt = jnp.sum(jnp.where(kept, w, 0), 1)
            Vs = jnp.sum(jnp.where(viol, ws, 0), 1); Ws = jnp.sum(jnp.where(kept, ws, 0), 1)
            okv = Vt > 1e-12; okw = Wt > 1e-12
            acc['lv'].append(np.asarray(jnp.where(okv, jnp.log(jnp.maximum(Vs, 1e-300) / jnp.maximum(Vt, 1e-300)), np.nan)))
            acc['lw'].append(np.asarray(jnp.where(okw, jnp.log(jnp.maximum(Ws, 1e-300) / jnp.maximum(Wt, 1e-300)), np.nan)))
            if la_s is not None:
                acc['pe'].append(np.asarray(la_s[idx] - la_t[idx] - c0))
        lv = np.concatenate(acc['lv']); lw = np.concatenate(acc['lw'])
        rec = dict(decade=d, split=split, n_reps=int(idx_all.size), n_bonds=acc['n'],
                   ratio_rms_w=float(np.sqrt(acc['we'] / max(acc['w'], 1e-300))),
                   ratio_rms=float(np.sqrt(acc['e2'] / max(acc['n'], 1))),
                   logV_rms=float(np.sqrt(np.nanmean(lv ** 2))) if np.isfinite(lv).any() else float('nan'),
                   logW_rms=float(np.sqrt(np.nanmean(lw ** 2))) if np.isfinite(lw).any() else float('nan'))
        if la_s is not None:
            pe = np.concatenate(acc['pe']); rec['point_rms'] = float(np.sqrt(np.mean(pe ** 2)))
        out.append(rec)
    return out
