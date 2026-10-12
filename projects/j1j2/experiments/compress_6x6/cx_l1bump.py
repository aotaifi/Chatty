"""Why does guide + 1 Lanczos step get WORSE after loop iteration 1 (both loops), then improve?
Exact 6x6 decomposition of iteration 1 of the table loop (wtLOOP8, tables la_1/s_1):
  A  ViT amplitude, ViT sign                 (start)
  B  ViT amplitude, Krylov sign of the ViT   (sign step only, no write-back)
  C  ViT amplitude, loop sign s_1
  D  written-back amplitude la_1, ViT sign   (amplitude step only)
  E  la_1, s_1                               (loop iteration 1)
For each: <H>, energy variance, and <H> after 1 and 2 exact Lanczos steps (per site, minus E0).
  python cx_l1bump.py OUT
"""
import os, sys, time
from cx_lib import *
from lanczos_lib import lanczos_ritz

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); F = os.path.join(OUT, 'cx_l1bump.json')
lab = Lab(); sec = lab.sec
la1, s1 = load_guide(1)
sK, _, _ = sec.krylov(sec.vec(lab.lP0, jnp.ones_like(lab.lP0)), lab.sP0)
cases = dict(A=(lab.lP0, lab.sP0), B=(lab.lP0, sK), C=(lab.lP0, s1), D=(la1, lab.sP0), E=(la1, s1))
res = {}
for k, (la, s) in cases.items():
    v = sec.vec(la, s); v = v / jnp.linalg.norm(v)
    Hv = sec.Hm(v); E = float(v @ Hv); var = float(Hv @ Hv) - E * E
    rl = lanczos_ritz(lambda x: sec.Hm(x), v, 2)
    res[k] = dict(H=(E - sec.E0) / N, var_site=var / N, L1=(rl[1]['E'] - sec.E0) / N, L2=(rl[2]['E'] - sec.E0) / N,
                  n_flip_vs_ViT=int(jnp.sum(s != lab.sP0)))
    log(k, res[k]); dump(F, res)
log('DONE')
