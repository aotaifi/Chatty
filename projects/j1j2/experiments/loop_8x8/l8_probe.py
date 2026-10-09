"""Stage 0 (measurements only, no decisions on the write-back): ViT throughput (translation-only and D4 x flip
projected), pilot VMC (check that the loaded ViT reproduces -0.498823(35)), global phase phi, energies of the binarised
guide, tempered-chain ESS, one- and two-hop cost per configuration.
  python l8_probe.py OUT [CKPT]
"""
import sys, time, json, os
import numpy as np
import l8_core as C
import jax, jax.numpy as jnp

OUT = sys.argv[1]; CKPT = sys.argv[2] if len(sys.argv) > 2 else 'vit_J2=0.50_N=8x8_k=0.mpack'
os.makedirs(OUT, exist_ok=True); F = os.path.join(OUT, 'probe.json'); res = dict(ckpt=CKPT)
rg = np.random.default_rng(7)
Srand = C.spins_to_bits_np(C.random_sz0(rg, 65536))

# ---------------------------------------------------------------- throughput
thr = []
for sym, bss in ((False, (4096, 8192, 16384)), (True, (1024, 2048, 4096))):
    for bs in bss:
        g = C.Guide(CKPT, sym=sym, batch=bs)
        g.eval(Srand[:bs])                                        # compile
        n = min(len(Srand), bs * (4 if not sym else 2))
        t = time.time(); g.eval(Srand[:n]); dt = time.time() - t
        thr.append(dict(sym=sym, batch=bs, n=n, sec=dt, evals_per_s=n / dt)); C.log('THR', thr[-1])
res['throughput'] = thr; C.dump(F, res)
best = {s: max([t for t in thr if t['sym'] == s], key=lambda t: t['evals_per_s']) for s in (False, True)}
res['best_batch'] = {str(k): v['batch'] for k, v in best.items()}

# ---------------------------------------------------------------- consistency: sym of a symmetric image
gT = C.Guide(CKPT, sym=False, batch=best[False]['batch'])
gS = C.Guide(CKPT, sym=True, batch=best[True]['batch'])
la1, ph1 = gT.eval(Srand[:4096])
Simg = C.spins_to_bits_np(C.bits_to_spins_np(Srand[:4096])[:, C.PERM[1]])
la2, _ = gT.eval(Simg)
res['transl_net_D4_asym_rms'] = float(np.std(la2 - la1)); C.log('D4 asymmetry of the network (rms log a)', res['transl_net_D4_asym_rms'])
lsa, _ = gS.eval(Srand[:4096]); lsb, _ = gS.eval(Simg)
res['sym_guide_D4_check_maxdiff'] = float(np.max(np.abs(lsa - lsb))); C.log('sym guide invariance max diff', res['sym_guide_D4_check_maxdiff'])
C.dump(F, res)


# ---------------------------------------------------------------- pilot VMC
def pilot(g, tag, nch, burn, ns, beta=1.0, seed=11):
    t = time.time(); n0 = g.neval
    St, acc = C.sample_chains(g, beta, nch, burn, ns, 1, seed)
    tsamp = time.time() - t
    S = St.reshape(-1); ch = np.tile(np.arange(nch), ns)
    t = time.time(); n1 = g.neval
    d = C.onehop(g, S)
    t1 = time.time() - t
    out = dict(tag=tag, beta=beta, nchains=nch, burn=burn, nsamp=ns, accept=acc, sample_sec=tsamp,
               onehop_sec=t1, onehop_evals_per_conf=(g.neval - n1) / len(S), onehop_unique_per_conf=d['nunique'] / len(S))
    return S, ch, d, out


rows = {}
for g, tag, nch, burn in ((gT, 'transl', 1024, 30), (gS, 'sym', 512, 20)):
    S, ch, d, out = pilot(g, tag, nch, burn, 8)
    phi = 0.5 * float(np.angle(np.sum(np.exp(2j * d['ph']))))
    g.phi = phi
    d = C.onehop(g, S)                                           # signs with this phi
    rel = np.angle(np.exp(1j * (d['ph'] - phi)))
    elc = d['ELc'].real
    out.update(phi=phi, leak=float(np.mean(np.abs(np.sin(rel)))),
               E_complex=float(elc.mean() / C.N), E_complex_se=C.chain_se(elc, ch) / C.N,
               E_bin=float(d['ELa'].mean() / C.N), E_bin_se=C.chain_se(d['ELa'], ch) / C.N,
               sd_EL_bin_site=float(d['ELa'].std() / C.N), sd_EL_complex_site=float(elc.std() / C.N),
               mean_nval=float(d['nval'].mean()), mean_W=float(d['W'].mean()), mean_V=float(d['V'].mean()),
               frac_viol_weight=float(np.mean(d['V'] / (d['V'] + d['W']))))
    ELr = d['ELa'].reshape(8, nch); m = ELr - ELr.mean()
    out['lag1_corr_EL'] = float(np.sum(m[1:] * m[:-1]) / np.sum(m[:-1] * m[:-1]))
    rows[tag] = out; C.log('PILOT', out)
    np.savez(os.path.join(OUT, f'pilot_{tag}.npz'), S=S, ch=ch, ELa=d['ELa'], ELc=d['ELc'], la=d['la'])
    res['pilot'] = rows; C.dump(F, res)

# ---------------------------------------------------------------- tempered chains: ESS of weights to a^2, bridge ratio
for g, tag, nch in ((gT, 'transl', 1024), (gS, 'sym', 256)):
    Sq, chq = None, None
    S5, acc5 = C.sample_chains(g, 0.5, nch, 30 if tag == 'transl' else 20, 8, 1, 21)
    S5 = S5.reshape(-1); la5, _ = g.eval(S5)
    P = np.load(os.path.join(OUT, f'pilot_{tag}.npz')); la1_ = P['la']
    # ESS of the pure tempered proposal (weights a^2 / a^1 = a)
    w = np.exp(la5 - la5.max()); ess = w.sum() ** 2 / (w * w).sum() / len(w)
    # bridge (Meng-Wong) estimate of r = Z2/Z1 (Z_beta = sum a^(2 beta)) from the two sample sets; mixture ESS
    ref = max(la5.max(), la1_.max()); l1 = np.exp(la1_ - ref); l5 = np.exp(la5 - ref)
    n1, n5 = len(l1), len(l5); s1, s5 = n1 / (n1 + n5), n5 / (n1 + n5)
    r = np.median(l5)
    for _ in range(200):
        num = np.mean(l5 / (s5 * r + s1 * l5)); den = np.mean(1.0 / (s5 * r + s1 * l1))
        r = num / den
    allla = np.concatenate([l1, l5])
    wm = allla / (s5 * r + s1 * allla)
    essm = wm.sum() ** 2 / (wm * wm).sum() / len(wm)
    q = dict(accept_beta05=acc5, ess_frac_tempered_only=float(ess), ess_frac_mixture=float(essm), bridge_r=float(r),
             la_sd_beta1=float(la1_.std()), la_sd_beta05=float(la5.std()), la_mean_shift=float(la1_.mean() - la5.mean()))
    rows[tag]['tempered'] = q; C.log('TEMPERED', tag, q)
    res['pilot'] = rows; C.dump(F, res)

# ---------------------------------------------------------------- two-hop cost per configuration (translation-only)
S = np.load(os.path.join(OUT, 'pilot_transl.npz'))['S'][:128]
t = time.time(); n0 = gT.neval
NBs, V = C.neighbors(S)
Y = np.unique(np.concatenate([S, NBs[V]]))
d2 = C.onehop(gT, Y)
res['twohop_transl'] = dict(conf=len(S), onehop_set=len(Y), evals=gT.neval - n0, evals_per_conf=(gT.neval - n0) / len(S),
                            sec=time.time() - t, sec_per_conf=(time.time() - t) / len(S))
C.log('TWOHOP', res['twohop_transl'])
C.dump(F, res); C.log('DONE')
