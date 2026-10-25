"""Control (i) at 8x8, stage 3 half: the stage-3 VMC (vmc_moments.py, alpha by Ritz) on the zero-step refit's params_final.npz must reproduce
g_0's guide constants (REF/consts.json: alpha, E, w) to <= 1e-8 relative, without paying for the full 1-H100-h VMC.
vmc_moments.py is deterministic given its seeds (PRNGKey(7000 + seed), random_configs(9500 + seed)); so the run is reproduced as a PREFIX:
stage A (phase w, Ec), B0 (E_ref) and the first K sweeps of seed 1 with the exact production arguments of REF (chains 1024, stageA 512 x 20,
therm 100, groups 16, chunk 256), compared with REF/stage_A.json and REF/vmc_sums_s1.npz[:K]; then consts are recomputed from REF's stored
sums of both seeds with vmc_moments' own pooled analysis and compared with REF/consts.json.  If the prefix sums agree bitwise (or to 1e-12)
and the analysis of the stored sums reproduces consts.json, the full stage-3 run on the same params reproduces consts.json.
Broken version (--perturb 1e-7): params f[0,1] *= 1 + perturb; stage A's Ec must differ beyond 1e-8.
  python l8_vmc_prefix.py OUT --params P.npz --ref /scratch/hpc-prf-ppdmrg/hpcalot/gfmc8/runs/rb8/V [--sweeps 2] [--perturb 0]
Last line: PREFIX_DONE verdict PASS|FAIL ...
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'rbmpp_6x6')):
    sys.path.insert(0, d)
import vmc_moments as VM
from rbmpp_lib import Lattice

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--params', required=True); ap.add_argument('--ref', required=True)
ap.add_argument('--sweeps', type=int, default=2); ap.add_argument('--perturb', type=float, default=0.0)
ap.add_argument('--stageA_only', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
prog = os.path.join(a.out, 'progress.txt')


def log(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'; print(line, flush=True); open(prog, 'a').write(line + '\n')


rel = lambda x, y: abs(x - y) / max(abs(y), 1e-300)
z = np.load(a.params); prm = {k: np.array(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
if a.perturb:
    prm['f'][0, 1] *= 1 + a.perturb
lat = Lattice(8, 0.5, 'translations'); N = lat.N
refA = json.load(open(os.path.join(a.ref, 'stage_A.json'))); refC = json.load(open(os.path.join(a.ref, 'consts.json')))
sm = VM.Sampler(prm, lat, True, 256)
t0 = time.time()
w, Ec, infoA = VM.stage_phase(sm, 512, 20, 0, 100, lambda m: log('PROGRESS ' + m))
out = dict(Ec=Ec, Ec_ref=refA['Ec'], Ec_rel=rel(Ec, refA['Ec']), w=[w.real, w.imag], w_ref=refA['w'])
log(f'PROGRESS stage A {time.time() - t0:.0f}s: Ec {Ec!r} ref {refA["Ec"]!r} rel {out["Ec_rel"]:.2e}; w {w} ref {refA["w"]}')
ok = out['Ec_rel'] <= 1e-8 and abs(w - complex(*refA['w'])) <= 1e-8
if not a.stageA_only:
    s0, _ = VM.stage_moments(sm, w, Ec, 512, 10, 0, 100, 4, lambda m: log(m), tag=' B0')
    E_ref = Ec + s0[:, :, 1].sum() / s0[:, :, 0].sum()
    out.update(E_ref=E_ref, E_ref_ref=refA['E_ref'], E_ref_rel=rel(E_ref, refA['E_ref']))
    log(f'PROGRESS B0 {time.time() - t0:.0f}s: E_ref {E_ref!r} ref {refA["E_ref"]!r} rel {out["E_ref_rel"]:.2e}')
    sums, info = VM.stage_moments(sm, w, E_ref, 1024, a.sweeps, 1, 100, 16, lambda m: log(m), tag=' s1')
    rs = np.load(os.path.join(a.ref, 'vmc_sums_s1.npz'))['sums'][:a.sweeps]
    srel = float(np.max(np.abs(sums - rs)) / np.max(np.abs(rs)))
    out.update(sums_maxrel=srel, sums_bitwise=bool(np.array_equal(sums, rs)))
    log(f'PROGRESS s1 prefix {a.sweeps} sweeps {time.time() - t0:.0f}s: max rel diff of the per-group sums vs REF {srel:.2e} (bitwise {out["sums_bitwise"]})')
    # consts from REF's stored sums with vmc_moments' pooled analysis (what vmc_moments main writes to consts.json)
    S1 = np.load(os.path.join(a.ref, 'vmc_sums_s1.npz')); S2 = np.load(os.path.join(a.ref, 'vmc_sums_s2.npz'))
    pooled = np.concatenate([S1['sums'], S2['sums']], 0)
    rP = VM.analyse(pooled, float(S1['E_ref']), N, 1024, pooled.shape[0])
    out.update(alpha_from_sums=rP['alpha']['value'], E_from_sums=rP['E']['value'], alpha_ref=refC['alpha'], E_ref_consts=refC['E'],
               alpha_rel=rel(rP['alpha']['value'], refC['alpha']), E_rel=rel(rP['E']['value'], refC['E']))
    log(f'PROGRESS consts from stored sums: alpha {rP["alpha"]["value"]!r} vs {refC["alpha"]!r} (rel {out["alpha_rel"]:.1e}); E {rP["E"]["value"]!r} vs {refC["E"]!r} (rel {out["E_rel"]:.1e})')
    ok = ok and out['E_ref_rel'] <= 1e-8 and srel <= 1e-8 and out['alpha_rel'] <= 1e-8 and out['E_rel'] <= 1e-8
json.dump(out, open(os.path.join(a.out, 'prefix.json'), 'w'), indent=1, default=float)
log(f'PREFIX_DONE verdict {"PASS" if ok else "FAIL"} perturb {a.perturb} {time.time() - t0:.0f}s')
