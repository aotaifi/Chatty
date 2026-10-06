"""Prep for stored signs 6x6: tempered samples from |psi_ViT|^(2 beta) + symmetry/timing pilot.

Writes samples_b{beta}.npz with arrays train (nr, nch) and held (nr, nch) [beta=1 also: thr, energy].
Pilot: is log a and c1 = sgn(T_FN - r0) invariant under D4 / spin flip? CNN eval timing.
Usage: python ss6_prep.py [--pilot-only]
"""
import sys, json, time, os
import numpy as np
import jax, jax.numpy as jnp
import ss6_core as S
import ll6_core as C

pilot_only = '--pilot-only' in sys.argv
net = C.Net(S.CKPT, dtype='float32', batch=16384)
amp = S.AmpCache(net)
out = {}
t0 = time.time()
specs = {  # beta: {name: (nchains, nrounds, between, burn, seed)}
    1.0: dict(thr=(1024, 16, 2, 200, 11), energy=(2048, 25, 2, 200, 12), held=(512, 20, 2, 200, 13)),
    0.5: dict(train=(1024, 40, 2, 200, 21), held=(512, 20, 2, 200, 23)),
    0.3: dict(train=(1024, 40, 2, 200, 31), held=(512, 20, 2, 200, 33)),
}
if pilot_only:
    specs = {0.5: dict(held=(256, 4, 2, 50, 99))}
allarrs = {}
for beta, d in specs.items():
    arrs = {}; allarrs[beta] = arrs
    for name, (nch, nr, btw, burn, seed) in d.items():
        X, acc = S.sample_tempered(net, beta, nch, nr, btw, burn, seed)
        arrs[name] = X
        out[f'b{beta}_{name}'] = dict(shape=list(X.shape), accept=acc, n_unique=int(len(np.unique(X))),
                                      sec=time.time() - t0)
        S.log('SAMPLE', beta, name, out[f'b{beta}_{name}'])
    if not pilot_only:
        np.savez_compressed(f'samples_b{beta}.npz', **arrs)
        import shutil; shutil.copy(f'samples_b{beta}.npz', os.environ.get('SS6_DATA', '/home/a/A.Otaifi/chatty_ss6/data') + f'/samples_b{beta}.npz')
    # amplitude spread diagnostic
    la = amp(arrs['held'].reshape(-1))
    out[f'b{beta}_held_loga'] = dict(mean=float(la.mean()), std=float(la.std()), min=float(la.min()), max=float(la.max()))
    S.log('LOGA', beta, out[f'b{beta}_held_loga'])

# ---- symmetry pilot on beta=0.5 held states
Xh = allarrs[0.5]['held'].reshape(-1)[:2000]
la0 = amp(Xh)
r0, _ = S.r_onehop_chunked(Xh, amp, C.marshall_vec)
c0 = np.where(r0 <= C.T_FN, 1, -1)
sym = {}
for g in range(1, 8):
    Xg = S.permute_bits(Xh, S.D4[g])
    lg = amp(Xg); rg, _ = S.r_onehop_chunked(Xg, amp, C.marshall_vec)
    cg = np.where(rg <= C.T_FN, 1, -1)
    sym[g] = dict(dloga_rms=float(np.sqrt(np.mean((lg - la0) ** 2))), dr_rms=float(np.sqrt(np.mean((rg - r0) ** 2))),
                  c_mismatch=float(np.mean(cg != c0)), marshall_mismatch=float(np.mean(C.marshall_vec(Xg) != C.marshall_vec(Xh))))
Xf = Xh ^ np.uint64((1 << 36) - 1)
lf = amp(Xf); rf, _ = S.r_onehop_chunked(Xf, amp, C.marshall_vec)
sym['flip'] = dict(dloga_rms=float(np.sqrt(np.mean((lf - la0) ** 2))), c_mismatch=float(np.mean(np.where(rf <= C.T_FN, 1, -1) != c0)))
out['sym'] = sym; out['c1_flip_frac_b0.5_held'] = float(np.mean(c0 < 0))
S.log('SYM', json.dumps(sym))
# ---- CNN timing
for ch, dp in ((32, 4), (48, 6)):
    for d4 in (False, True):
        sn = S.SignNet(ch, dp, d4=d4)
        Z = np.repeat(Xh, 100)[:200000]
        sn.logit(Z[:70000]); jax.block_until_ready(sn.params)
        t1 = time.time(); sn.logit(Z); dt = time.time() - t1
        out[f'cnn_{ch}x{dp}_d4{int(d4)}'] = dict(npar=sn.npar, sec_per_state=dt / len(Z))
        S.log('CNN', ch, dp, d4, sn.npar, dt / len(Z))
Zv = np.repeat(Xh, 50); t1 = time.time(); net.logabs(net.flat0, Zv)
out['vit_sec_per_state'] = (time.time() - t1) / len(Zv)
out['amp_new'] = amp.nnew; out['sec'] = time.time() - t0
json.dump(out, open('prep_pilot.json' if pilot_only else 'prep.json', 'w'), indent=1)
S.log('DONE', json.dumps({k: v for k, v in out.items() if k != 'sym'}))
