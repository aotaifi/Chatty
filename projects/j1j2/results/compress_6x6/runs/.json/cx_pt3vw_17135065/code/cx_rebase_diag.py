"""Diagnosis of the test-1 plateau (README round 2, amendment 4): is the re-base retention limited by the optimizer
(own loss still falling, Adam spikes at lr 3e-3), by capacity, or by the base features?  Re-base the SAME write-back
table la_w of chop seed 0, iteration 6 (saved), with variants; retention = share of iteration 6's <H> gain kept,
(H(G_5, s_6) - H(G_new, s_6)) / (H(G_5, s_6) - H(la_w, s_6)), and accumulated share since psi_P.
  python cx_rebase_diag.py OUT SPEC.json     SPEC: run_dir, it, variants [[tag, C, steps, lr, warm_params_or_null]]
"""
import os, sys, time
import numpy as np
from cx_lib import *

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_rebase_diag.json'); T00 = time.time()
lab = Lab(); sec = lab.sec
FEAT_BASE = base_features(lab)
RD, IT = SPEC['run_dir'], SPEC['it']
chop = json.load(open(os.path.join(RD, 'cx_chop.json')))['iters'][IT - 1]
la_w = jnp.asarray(np.load(os.path.join(RD, f'la_w_{IT}.npy'))); s = jnp.asarray(np.load(os.path.join(RD, f's_{IT}.npy')).astype(np.float32))
H_w = lab.H_site(la_w, s); H_prev = chop['H_G_newsign']
res = dict(spec=SPEC, H_w=H_w, H_w_chop=chop['H_w'], H_prev=H_prev, H_chop=chop['H'], retention_chop=chop['retention_step'], variants=[])
log(f'[diag] it {IT}: H(la_w) {H_w:.5e} (chop {chop["H_w"]:.5e}) H(G_prev) {H_prev:.5e} chop re-base {chop["H"]:.5e}')
D = la_w - lab.lP0
CDF, LIW, DLR, _ = tail_proposal(lab, D, la_w, s); del D
for tag, C, steps, lr, warm in SPEC['variants']:
    t0 = time.time()
    mF = FeatNet(FEAT_BASE, seed=50 + IT, C=C)
    if warm: mF.flat0 = jnp.asarray(np.load(warm), jnp.float32)
    sec.offload()
    fl, curve = fit_edge(lab, mF, la_w, s, CDF, LIW, DLR, steps, 50 + IT, lr=lr, tag=tag)
    sec.reload()
    G = lab.lP0 + mF.table(fl, sec.reps, lab.TIDX)
    H = lab.H_site(G, s)
    v = dict(tag=tag, C=C, steps=steps, lr=lr, warm=warm, npar=mF.npar, H=H, curve=curve,
             retention_step=(H_prev - H) / (H_prev - H_w), retention_acc=(H_START - H) / (H_START - H_w),
             bond_rms=bond_rms(lab, la_w, G - la_w), sec=time.time() - t0)
    res['variants'].append(v); dump(F, res)
    log(f'== {tag}: <H> {H:.5e}  retention step {v["retention_step"]:.3f} acc {v["retention_acc"]:.3f}  bond rms {v["bond_rms"]:.4f}  ({v["sec"]:.0f}s)')
    del mF, fl, G
res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
