"""Test 2: STRUCTURED TOLERANCE. Replay one write-back step (stored wtLOOP8 net k+1 from the exact table guide la_k,
k = 3 -> iteration 4, k = 6 -> iteration 7) with the ACTUAL error fields of candidate predictors, scaled by c, and
compare with white noise at equal size.
  log-amplitude fields e (guide la_k + c e enters log a, V, W and the semi-implicit target):
    Aw_k3     : A' warm ViT (Adam) student of la_3 minus la_3        (P2 run, la_student.npy)
    As_k6     : A ViT-from-scratch student of la_6 minus la_6
    fb_k3/k6  : base-feature net distilled from la_k (the re-base of test 1; fit here, 20k Adam, edge loss) minus la_k
    white     : iid N(0, 1) per orbit (bond rms ~ sqrt 2 x the pointwise rms)
  V/W fields (log V, log W replaced by (1-c) log V + c log V_B):
    B_k3      : the bond-ratio net B (k = 3, seed 0)
Size measures (exact): |H_xy| a_x a_y-weighted rms bond difference of c e; a^2-weighted rms of log V, log W errors.
  python cx_struct.py OUT SPEC.json
"""
import os, sys, time
import numpy as np
from cx_lib import *

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_struct.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
REFS = json.load(open(os.path.join(TABDIR, 'cx_replay.json')))['refs']
res = dict(spec=SPEC, base={}, points=[])


class BondCNN(nn.Module):                                    # copy of cx_distill.BondCNN (that file runs at import)
    C: int = 64
    layers: int = 6
    hid: int = 64

    @nn.compact
    def __call__(self, x):
        kw = dict(dtype=jnp.float32, param_dtype=jnp.float32)

        def conv(h):
            g = h[:, NBR, :]
            return nn.Dense(self.C, **kw)(g.reshape(g.shape[0], N, -1))
        h = nn.gelu(conv(x[:, :, None]))
        for _ in range(1, self.layers):
            h = h + nn.gelu(conv(nn.LayerNorm(**kw)(h)))
        h = nn.LayerNorm(**kw)(h)
        hi, hj = h[:, BI_I, :], h[:, BJ_I, :]
        si = x[:, BI_I][:, :, None]
        z = jnp.concatenate([hi + hj, hi * hj, si * (hi - hj)], -1)
        Fz = z.shape[-1]
        W1 = self.param('W1', nn.initializers.lecun_normal(), (4, Fz, self.hid), jnp.float32)
        b1 = self.param('b1', nn.initializers.zeros, (4, self.hid), jnp.float32)
        W2 = self.param('W2', nn.initializers.lecun_normal(), (4, self.hid, self.hid), jnp.float32)
        b2 = self.param('b2', nn.initializers.zeros, (4, self.hid), jnp.float32)
        w3 = self.param('w3', nn.initializers.lecun_normal(), (4, self.hid, 1), jnp.float32)
        b3 = self.param('b3', nn.initializers.zeros, (4,), jnp.float32)
        u = nn.gelu(jnp.einsum('bnf,nfh->bnh', z, W1[BTYPE]) + b1[BTYPE][None])
        u = nn.gelu(jnp.einsum('bnf,nfh->bnh', u, W2[BTYPE]) + b2[BTYPE][None])
        return jnp.einsum('bnf,nf->bn', u, w3[BTYPE][:, :, 0]) + b3[BTYPE][None]


ctx = {}
for k in (3, 6):
    la, s = load_guide(k)
    ctx[k] = GuideCtx(lab, la, s, load_loop_params(k + 1), sec.E0 + N * REFS[str(k)]['E_FN_stack'])
    f0, si0 = ctx[k].step()
    res['base'][str(k)] = dict(frac=f0, SI=si0, table_ref=REFS[str(k)]['continuation_frac_table'])
    log(f'[k={k}] unperturbed frac {f0:.6f} (table {REFS[str(k)]["continuation_frac_table"]:.6f}) SI {si0:.4f}')
dump(F, res)


def run_la(name, k, e, scales):
    c_ = ctx[k]
    for c in scales:
        t0 = time.time()
        la_p = c_.la + c * e
        feats = lab.VW(la_p, c_.s)
        rv, rw = c_.vw_err(feats[1], feats[2])
        br = bond_rms(lab, c_.la, c * e)
        fr, si = c_.step(feats); del feats, la_p
        b = res['base'][str(k)]
        p = dict(field=name, kind='logamp', k=k, c=c, bond_rms=br, logV_rms=rv, logW_rms=rw, frac=fr, SI=si,
                 kept=fr / b['frac'], SI_kept=si / b['SI'], sec=time.time() - t0)
        res['points'].append(p); dump(F, res)
        log(f'[{name} k={k} c={c}] bond rms {br:.4f}  logV {rv:.4f} logW {rw:.4f}  frac {fr:.4f} (kept {p["kept"]:.3f})  SI {si:.4f}')


# ---------------- white noise reference (pointwise sigma = bond rms / sqrt 2)
for k in (3, 6):
    for sb in SPEC.get('white', [0.001, 0.003, 0.01, 0.03]):
        xi = jax.random.normal(jax.random.PRNGKey(4242 + k + int(sb * 1e4)), (sec.D,), f64)
        run_la('white', k, xi, [sb / np.sqrt(2.0)]); del xi
# ---------------- students of P2
for name, k, path, scales in SPEC.get('students', []):
    e = jnp.asarray(np.load(path)) - ctx[k].la
    e = e - jnp.sum(ctx[k].PA * e)
    run_la(name, k, e, scales); del e
# ---------------- base-feature distillation of la_k (the re-base of test 1)
if SPEC.get('featbase', True):
    FEAT_BASE = base_features(lab)
    for k in (3, 6):
        c_ = ctx[k]
        D = c_.la - lab.lP0
        CDF, LIW, DLR, _ = tail_proposal(lab, D, c_.la, c_.s); del D
        mF = FeatNet(FEAT_BASE, seed=50 + k)
        sec.offload()
        flF, curve = fit_edge(lab, mF, c_.la, c_.s, CDF, LIW, DLR, SPEC.get('steps_r', 20000), 50 + k, tag=f'fb{k}')
        sec.reload()
        g = mF.table(flF, sec.reps, lab.TIDX); del mF, CDF, LIW, DLR
        e = lab.lP0 + g - c_.la; e = e - jnp.sum(c_.PA * e); del g
        H_fb = lab.H_site(c_.la + e, c_.s)
        res['base'][f'fb{k}'] = dict(H=H_fb, kept_acc=(H_START - H_fb) / (H_START - REFS[str(k)]['H_stack']), curve=curve)
        log(f'[fb k={k}] <H> {H_fb:.4e} accumulated kept {res["base"][f"fb{k}"]["kept_acc"]:.3f}')
        np.save(os.path.join(OUT, f'fb_err_k{k}.npy'), np.asarray(e))
        run_la(f'fb_k{k}', k, e, SPEC.get('fb_scales', [0.3, 1.0, 2.0])); del e
# ---------------- B: actual V/W error fields
if SPEC.get('B_params'):
    net = BondCNN()
    p0 = net.init(jax.random.PRNGKey(0), jnp.zeros((1, N), jnp.float32))['params']
    _, unravel = ravel_pytree(p0)
    flB = jnp.asarray(np.load(SPEC['B_params']), jnp.float32)
    k = 3; c_ = ctx[k]

    @jax.jit
    def vw_chunk(ii):
        x = sec.reps[ii]
        v = valid_bonds(x)
        y = jnp.where(v, x[:, None] ^ MASKS[None, :], x[:, None])
        iy = SS.canon(sec.T, sec.reps, y.reshape(-1)).reshape(x.shape[0], NB)
        kept = v & (c_.s[ii][:, None] * c_.s[iy] < 0)
        R = net.apply({'params': unravel(flB)}, bits(x)).astype(f64)
        wr = jnp.where(v, 0.5 * JB[None, :] * jnp.exp(jnp.clip(R, -60, 60)), 0.0)
        return jnp.sum(jnp.where(v & ~kept, wr, 0), 1), jnp.sum(jnp.where(kept, wr, 0), 1)
    Vb, Wb = [], []
    for i in range(0, sec.D, 8192):
        a_, b_ = vw_chunk(jnp.minimum(jnp.arange(i, i + 8192), sec.D - 1)); m_ = min(8192, sec.D - i)
        Vb.append(a_[:m_]); Wb.append(b_[:m_])
    Vb = jnp.concatenate(Vb); Wb = jnp.concatenate(Wb)
    lV, lW = jnp.log(c_.V + 1e-12), jnp.log(c_.W + 1e-12)
    dV, dW = jnp.log(Vb + 1e-12) - lV, jnp.log(Wb + 1e-12) - lW; del Vb, Wb
    for c in SPEC.get('B_scales', [0.03, 0.1, 0.3, 1.0]):
        t0 = time.time()
        Vp = jnp.exp(lV + c * dV) - 1e-12; Wp = jnp.exp(lW + c * dW) - 1e-12
        rv, rw = c_.vw_err(Vp, Wp)
        fr, si = c_.step((c_.lan, jnp.maximum(Vp, 0), jnp.maximum(Wp, 0))); del Vp, Wp
        b = res['base'][str(k)]
        p = dict(field='B_k3', kind='VW', k=k, c=c, bond_rms=float('nan'), logV_rms=rv, logW_rms=rw, frac=fr, SI=si,
                 kept=fr / b['frac'], SI_kept=si / b['SI'], sec=time.time() - t0)
        res['points'].append(p); dump(F, res)
        log(f'[B_k3 c={c}] logV {rv:.4f} logW {rw:.4f}  frac {fr:.4f} (kept {p["kept"]:.3f})  SI {si:.4f}')
    # white V/W reference at the same log V size
    for sv in SPEC.get('white_VW', [0.001, 0.003, 0.01]):
        xi = jax.random.normal(jax.random.PRNGKey(777 + int(sv * 1e4)), (2, sec.D), f64)
        Vp, Wp = c_.V * jnp.exp(sv * xi[0]), c_.W * jnp.exp(sv * xi[1]); del xi
        rv, rw = c_.vw_err(Vp, Wp)
        fr, si = c_.step((c_.lan, Vp, Wp)); del Vp, Wp
        b = res['base'][str(k)]
        res['points'].append(dict(field='white_VW', kind='VW', k=k, c=sv, bond_rms=float('nan'), logV_rms=rv, logW_rms=rw,
                                  frac=fr, SI=si, kept=fr / b['frac'], SI_kept=si / b['SI'])); dump(F, res)
        log(f'[white_VW k=3 {sv}] logV {rv:.4f}  frac {fr:.4f} (kept {fr / b["frac"]:.3f})')
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
