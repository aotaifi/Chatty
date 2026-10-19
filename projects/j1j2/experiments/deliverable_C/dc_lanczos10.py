"""Deliverable C, E4 (10x10): one Lanczos step on the public 10x10 ViT (nqs-models/j1j2_square_10x10_05), PREREG E4/E6.

Guide psi = ViT summed over the checkpoint's permutations (+ spin flip), with its sign binarised: s(x) = sign cos(arg psi - phi),
phi = half the angle of the mean of exp(2 i arg psi) over psi^2 samples (the state is real up to a global phase; the
sign leak is measured as the complex-vs-binarised energy difference).  A = H - E_ref, a_k = (A^k psi)/psi.
Modes:
  moments: sample |psi|^2 (Metropolis, bond exchange), exact two-hop a1, a2 on every sample ->
           E (complex own phase and binarised), variance, phi, alpha from the 2x2 Ritz problem in span{psi, A psi}
           (m2 = <a1^2>, m3 = <a1 a2>, E_ref = <E_L>), Ritz energy (moment estimate, cross-check only).
  direct:  delayed-acceptance Metropolis on |psi_1|^2 = |psi|^2 f^2, f = 1 + alpha a1 (fixed alpha, E_ref, phi from the
           moments run), exact two-hop E_L1 = E_ref + (a1 + alpha a2)/f on every sample (the number to quote).
  python dc_lanczos10.py OUT SPEC.json
SPEC: mode, ckpt, symm (or null), flip, batch, nchains, burn, nsamp, thin, seed; direct also: alpha, Eref, phi, init (npz)
"""
import os, sys, json, time, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loop_8x8', 'review'))
import dc_fn as F
import dc_vit10 as V
import rv_da

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
g = V.Guide10(SPEC['ckpt'], SPEC.get('symm'), SPEC.get('flip', False), batch=SPEC.get('batch', 512))
lat = F.Lattice(10); N = 100; rg = np.random.default_rng(SPEC['seed']); log = V.log
CC = types.SimpleNamespace(NB_=lat.NB, BI=lat.BI, BJ=lat.BJ, ONE=np.uint64(1), MASKS=lat.MASKS, N=N)
phi = float(SPEC.get('phi', 0.0))


def sgn(ph):
    return np.where(np.cos(ph - phi) >= 0, 1.0, -1.0)


def onehop(S, Er):
    """per configuration: la, phase, sign, binarised E_L, complex E_L, D, and per-bond h, valid, neighbour states."""
    S = np.asarray(S, np.uint64); n = len(S)
    NBs, Vd = lat.neighbors(S)
    st = np.concatenate([S, NBs[Vd]]); u, inv = np.unique(st, return_inverse=True)
    la_u, ph_u = g.eval(u); la = la_u[inv]; ph = ph_u[inv]
    lan = np.zeros((n, lat.NB)); phn = np.zeros((n, lat.NB)); lan[Vd] = la[n:]; phn[Vd] = ph[n:]
    r = np.where(Vd, 0.5 * lat.JB[None, :] * np.exp(np.clip(lan - la[:n, None], -60, 60)), 0.0)
    s = sgn(ph[:n]); sn = np.where(Vd, sgn(phn), 0.0)
    D = lat.diag(Vd); EL = D + np.sum(r * s[:, None] * sn, 1)
    ELc = D + np.sum(r * np.exp(1j * (phn - ph[:n, None])), 1)
    return dict(la=la[:n], ph=ph[:n], EL=EL, ELc=ELc, D=D)


def twohop(S, Er):
    """a1, a2 at S (psi on the two-hop balls), plus complex E_L and phases at S."""
    S = np.asarray(S, np.uint64); n = len(S)
    NBs, Vd = lat.neighbors(S); o, bb = np.nonzero(Vd); ys = NBs[o, bb]
    Yu = np.unique(np.concatenate([S, ys])); d = onehop(Yu, Er)
    px = np.searchsorted(Yu, S); py = np.searchsorted(Yu, ys)
    s = sgn(d['ph'])
    h = 0.5 * lat.JB[bb] * s[px][o] * s[py] * np.exp(np.clip(d['la'][py] - d['la'][px][o], -60, 60))
    a1 = d['EL'] - Er
    a2 = (d['D'][px] - Er) * a1[px] + np.bincount(o, weights=h * a1[py], minlength=n)
    return dict(a1=a1[px], a2=a2, la=d['la'][px], ph=d['ph'][px], ELc=d['ELc'][px])


def chain_se(v, ch):
    ids, inv = np.unique(ch, return_inverse=True); m = np.bincount(inv, weights=v) / np.bincount(inv)
    return float(m.std(ddof=1) / np.sqrt(len(m)))


nch = SPEC['nchains']; nb, ns, th = SPEC['burn'], SPEC['nsamp'], SPEC['thin']
if SPEC['mode'] == 'moments':
    S = np.array([np.sum(np.uint64(1) << rg.choice(N, N // 2, replace=False).astype(np.uint64)) for _ in range(nch)], np.uint64)
    la, _ = g.eval(S); out = []; acc = prop = 0
    for sw in range(nb + ns * th):
        for _ in range(N):
            b = rg.integers(lat.NB, size=nch)
            ok = (((S >> lat.BI[b]) ^ (S >> lat.BJ[b])) & np.uint64(1)).astype(bool)
            cand = S ^ lat.MASKS[b]; lc = np.full(nch, -np.inf)
            if ok.any(): lc[ok] = g.eval(cand[ok])[0]
            A = ok & (np.log(rg.random(nch)) < 2 * (lc - la)); S = np.where(A, cand, S); la = np.where(A, lc, la)
            acc += A.sum(); prop += nch
        if sw >= nb and (sw - nb + 1) % th == 0: out.append(S.copy())
        log(f'sweep {sw + 1} accept {acc / prop:.3f} {time.time() - T0:.0f}s')
    X = np.stack(out).reshape(-1); CH = np.tile(np.arange(nch), len(out)); R = len(X)
    np.savez(os.path.join(OUT, 'samples.npz'), X=X, chain=CH)
    # phase of the state (one-hop pass only for the phases at x)
    _, phx = g.eval(X); z = np.mean(np.exp(2j * phx)); phi = 0.5 * float(np.angle(z))
    log('phi', phi, '|<e^{2i ph}>|', abs(z))
    A1 = np.empty(R); A2 = np.empty(R); ELc = np.empty(R, complex); Er0 = 0.0; t = time.time(); ch = SPEC.get('chunk', 16)
    for i in range(0, R, ch):
        d = twohop(X[i:i + ch], Er0); A1[i:i + ch] = d['a1']; A2[i:i + ch] = d['a2']; ELc[i:i + ch] = d['ELc']
        if (i // ch) % 16 == 0:
            np.savez(os.path.join(OUT, 'moments.npz'), X=X, chain=CH, A1=A1, A2=A2, ELc=ELc, phi=phi, upto=i + ch)
            log(f'twohop {i + ch}/{R} {time.time() - t:.0f}s')
    np.savez(os.path.join(OUT, 'moments.npz'), X=X, chain=CH, A1=A1, A2=A2, ELc=ELc, phi=phi, upto=R)
    EL = A1                                              # E_ref = 0 here: a1 = E_L, a2 = (H^2 psi)/psi
    h1 = EL.mean(); m2 = np.mean((EL - h1) ** 2); m2b = np.mean(A2) - h1 ** 2
    c2 = A2 - 2 * h1 * EL + h1 ** 2                      # ((H - h1)^2 psi)/psi
    m3 = np.mean((EL - h1) * c2)
    Hm = np.array([[h1, m2], [m2, h1 * m2 + m3]]); Sm = np.diag([1.0, m2])
    from scipy.linalg import eigh
    ev, vec = eigh(Hm, Sm); c = vec[:, 0] / vec[0, 0]
    alpha = float(c[1])                                  # psi_1 = psi + alpha (H - h1) psi = psi (1 + alpha (E_L - h1))
    res = dict(spec=SPEC, R=R, accept=acc / prop, phi=phi, phase_coherence=float(abs(z)),
               E_bin_site=float(h1 / N), E_bin_se=chain_se(EL, CH) / N, E_complex_site=float(ELc.real.mean() / N),
               E_complex_se=chain_se(ELc.real, CH) / N, var_site=float(m2 / N), var_alt_site=float(m2b / N),
               alpha=alpha, Eref=float(h1), ritz_site=float(ev[0] / N), ritz_gain_site=float((ev[0] - h1) / N), sec=time.time() - T0)
else:
    al = float(SPEC['alpha']); Er = float(SPEC['Eref'])
    X0 = np.unique(np.load(SPEC['init'])['X']); S0 = X0[rg.choice(len(X0), nch, replace=False)]

    def onehop_f(S):
        d = onehop(S, Er)
        return d['la'], np.log(np.abs(1 + al * (d['EL'] - Er))), d['EL']
    dac = rv_da.DAChain(CC, S0, lambda S: g.eval(S)[0], onehop_f, rg); out = []
    for sw in range(1, nb + ns * th + 1):
        dac.run(N)
        if sw > nb and (sw - nb) % th == 0: out.append(dac.S.copy())
        log(f'DA sweep {sw} acc1 {dac.acc1 / dac.prop:.3f} acc2 {dac.acc2 / max(dac.acc1, 1):.3f} {time.time() - T0:.0f}s')
    X = np.stack(out).reshape(-1); CH = np.tile(np.arange(nch), len(out)); R = len(X)
    A1 = np.full(R, np.nan); A2 = np.full(R, np.nan); t = time.time(); ch = SPEC.get('chunk', 16)
    for i in range(0, R, ch):
        d = twohop(X[i:i + ch], Er); A1[i:i + ch] = d['a1']; A2[i:i + ch] = d['a2']
        if (i // ch) % 16 == 0:
            np.savez(os.path.join(OUT, 'direct.npz'), X=X, chain=CH, A1=A1, A2=A2); log(f'twohop {i + ch}/{R} {time.time() - t:.0f}s')
    np.savez(os.path.join(OUT, 'direct.npz'), X=X, chain=CH, A1=A1, A2=A2)
    f = 1 + al * A1; EL1 = Er + (A1 + al * A2) / f
    res = dict(spec=SPEC, R=R, accept1=dac.acc1 / dac.prop, accept2=dac.acc2 / max(dac.acc1, 1), E1_site=float(EL1.mean() / N),
               E1_se_site=chain_se(EL1, CH) / N, var1_site=float(EL1.var() / N), f_min=float(f.min()), f_max=float(f.max()),
               sec=time.time() - T0)
json.dump(res, open(os.path.join(OUT, 'lanczos10.json'), 'w'), indent=1, default=float); log('DONE', json.dumps(res, default=float))
