"""Equal-hop reference: psi_P (+ its own sign s_P) + p exact Lanczos steps, p = 0..4 (lanczos_lib), per site.
  python cx_lanczos_ref.py OUT
"""
import os, sys
from cx_lib import *
from lanczos_lib import lanczos_ritz
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
lab = Lab(); sec = lab.sec
rl = lanczos_ritz(lambda x: sec.Hm(x), sec.vec(lab.lP0, lab.sP0), 4)
res = [dict(p=d['p'], H=(d['E'] - sec.E0) / N, var=d['var']) for d in rl]
dump(os.path.join(OUT, 'cx_lanczos_ref.json'), res)
for d in res: log(f'psi_P + {d["p"]} Lanczos: <H> {d["H"]:.4e}')
