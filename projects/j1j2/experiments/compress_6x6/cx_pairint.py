"""D2a' (amendment to the fold-back diagnostics): ratio / sign interaction vs bond distance for a guide given as
per-orbit tables, x ~ |psi|^2.  Guides: 'psi0' (exact ground state), 'psiP' (symmetrised ViT; repeats D2a).
I(b,b') = la(x^{bb'}) + la(x) - la(x^b) - la(x^{b'}),  sigma = s(x^{bb'}) s(x) s(x^b) s(x^{b'}),  disjoint valid bonds.
  python cx_pairint.py OUT SPEC.json    SPEC: {"guides": ["psi0", "psiP"], "n": 1024}
"""
import os, sys, time
import numpy as np
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_pairint.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
z = np.load(TABLE); amp = np.asarray(z['amp'], np.float64); del z
G = {'psi0': (jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)), jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0)),
     'psiP': (lab.lP0, lab.sP0)}
del amp
sec.offload()
canon_j = jax.jit(SS.canon)
BIn = np.asarray(SS.BI).astype(int); BJn = np.asarray(SS.BJ).astype(int)


def sdist(p, q):
    dx = abs(p % 6 - q % 6); dy = abs(p // 6 - q // 6); dx = min(dx, 6 - dx); dy = min(dy, 6 - dy)
    return np.sqrt(dx * dx + dy * dy)
DB = np.array([[min(sdist(p, q) for p in (BIn[b], BJn[b]) for q in (BIn[c], BJn[c])) for c in range(NB)] for b in range(NB)])
DBj = jnp.asarray(DB); dkeys = sorted(set(np.round(DB.ravel(), 4).tolist()))


def lookup(conf):
    M = conf.shape[0]; B = 32768; out = []
    for i in range(0, M, B):
        c = conf[i:i + B]; m = c.shape[0]
        if m < B: c = jnp.concatenate([c, jnp.repeat(c[:1], B - m)])
        out.append(canon_j(sec.T, sec.reps, c)[:m])
    return jnp.concatenate(out)


res = dict(spec=SPEC, guides={})
n = SPEC.get('n', 1024); CH = 64
for gname in SPEC.get('guides', ['psi0', 'psiP']):
    la, s = G[gname]
    p = np.array(sec.n * jnp.exp(2 * (la - jnp.max(la))), np.float64); p = p / p.sum()
    idx = np.random.default_rng(11).choice(sec.D, size=n, p=p)
    acc = {k_: np.zeros(7) for k_ in dkeys}
    for c0 in range(0, n, CH):
        ix = jnp.asarray(idx[c0:c0 + CH]); x = sec.reps[ix]; C = x.shape[0]
        vx = valid_bonds(x); xb = x[:, None] ^ MASKS[None, :]
        ib = lookup(xb.reshape(-1)).reshape(C, NB)
        ibb = lookup((xb[:, :, None] ^ MASKS[None, None, :]).reshape(-1)).reshape(C, NB, NB)
        la_x = la[ix]; la_b = la[ib]; la_bb = la[ibb]
        I = la_bb + la_x[:, None, None] - la_b[:, :, None] - la_b[:, None, :]
        sig = s[ibb] * s[ix][:, None, None] * s[ib][:, :, None] * s[ib][:, None, :]
        ok = vx[:, :, None] & vx[:, None, :] & (DBj[None] > 0)
        wgt = jnp.exp(jnp.clip(la_b[:, :, None] + la_b[:, None, :] - 2 * la_x[:, None, None], -60, 60))
        # contribution error of the additive approximation to the b'-term of E_L(x^b): r_b' |sigma e^I - 1|
        rb2 = jnp.exp(jnp.clip(la_b - la_x[:, None], -60, 60))[:, None, :]
        cerr = rb2 * jnp.abs(sig * jnp.exp(jnp.clip(I, -60, 60)) - 1.0)
        I = np.asarray(I); sig = np.asarray(sig); ok = np.asarray(ok); wgt = np.asarray(wgt); cerr = np.asarray(cerr)
        for k_ in dkeys:
            if k_ == 0: continue
            m = ok & (np.abs(DB - k_) < 1e-3)[None]
            if not m.any(): continue
            Ii = I[m]; wi = wgt[m]
            acc[k_] += np.array([m.sum(), np.sum(Ii * Ii), np.sum(np.abs(Ii)), np.sum(sig[m] < 0), np.sum(wi),
                                 np.sum(wi * Ii * Ii), np.sum(cerr[m])])
    res['guides'][gname] = [dict(dist=k_, n=int(a[0]), rms_I=float(np.sqrt(a[1] / a[0])), mean_abs_I=float(a[2] / a[0]),
                                 sign_defect_frac=float(a[3] / a[0]), wrms_I=float(np.sqrt(a[5] / a[4])),
                                 mean_contrib_err=float(a[6] / a[0])) for k_, a in acc.items() if a[0] > 0]
    for r_ in res['guides'][gname]: log(gname, r_)
    dump(F, res)
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
