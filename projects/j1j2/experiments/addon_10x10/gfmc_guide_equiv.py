"""A0.4 item 1: the new pluggable-guide driver (gfmc_guide.py, --guide rbmpp) against the LAUNCHED M1 driver (m1_code/gfmc_run.py,
md5 0f858d33) at the 8x8 production p = 0 settings (translations x flip RBM+PP, --p 0 --M 256 --dtau 0.1 --chunk 256), same seed.
Both drivers are run end to end as subprocesses; the per-block series are read from their blocks.npz.

  python gfmc_guide_equiv.py OUT [--blocks 3] [--seed 1]
Controls (each must DIFFER from the launched series, so the comparison can fail): new driver at seed+1; new driver with --broken drop_vsf.
Also: guide rows (rate, d_FN) of both providers on the same walkers.  Last line 'EQUIV_RESULT ...'.
"""
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); M1 = os.path.join(HERE, 'm1_code')
PARAMS = '/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz'          # sha256 0520c5be... = production params (rb8/p0_s1 run_metadata)
CONSTS = '/scratch/a/A.Otaifi/gfmc_m13b/prod/rb8/V/consts.json'          # production walker constants (w = -1 - 1.5e-13 i)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--blocks', type=int, default=3); ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--M', type=int, default=256); ap.add_argument('--dtau', type=float, default=0.1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
common = ['--params', PARAMS, '--consts', CONSTS, '--L', '8', '--proj', 'translations', '--M', str(a.M), '--dtau', str(a.dtau),
          '--blocks', str(a.blocks), '--burn', '0', '--chunk', '256']


def run(tag, cmd):
    d = os.path.join(a.out, tag)
    with open(d + '.log', 'w') as fo:
        rc = subprocess.call([sys.executable, '-u'] + cmd[:1] + [d] + cmd[1:], stdout=fo, stderr=subprocess.STDOUT)
    assert rc == 0, f'{tag} exit {rc}, see {d}.log'
    return np.load(os.path.join(d, 'blocks.npz'))


old = run('launched', [os.path.join(M1, 'gfmc_run.py')] + common + ['--p', '0', '--seed', str(a.seed)])
new = run('new', [os.path.join(HERE, 'gfmc_guide.py'), '--guide', 'rbmpp', '--init', 'rbmpp'] + common + ['--seed', str(a.seed)])
ctl_seed = run('new_seed2', [os.path.join(HERE, 'gfmc_guide.py'), '--guide', 'rbmpp', '--init', 'rbmpp'] + common + ['--seed', str(a.seed + 1)])
ctl_vsf = run('new_dropvsf', [os.path.join(HERE, 'gfmc_guide.py'), '--guide', 'rbmpp', '--init', 'rbmpp', '--broken', 'drop_vsf'] + common + ['--seed', str(a.seed)])
res = {}
for k in ('emix', 'egrow', 'ess', 'elvar'):
    res[k] = dict(launched=old[k].tolist(), new=new[k].tolist(), max_abs_diff=float(np.max(np.abs(old[k] - new[k]))),
                  max_rel_diff=float(np.max(np.abs(old[k] - new[k]) / np.abs(old[k]))), bitwise=bool(np.array_equal(old[k], new[k])))
res['hops_equal'] = bool(np.array_equal(old['hops'], new['hops'])); res['hops'] = [old['hops'].tolist(), new['hops'].tolist()]
res['ctl_seed_emix'] = ctl_seed['emix'].tolist(); res['ctl_seed_maxdiff'] = float(np.max(np.abs(ctl_seed['emix'] - old['emix'])))
res['ctl_dropvsf_emix'] = ctl_vsf['emix'].tolist(); res['ctl_dropvsf_maxdiff'] = float(np.max(np.abs(ctl_vsf['emix'] - old['emix'])))

# rows of the two guide providers on the same walkers (the launched provider's fast-update ratios vs logpsi differences)
sys.path.insert(0, M1); sys.path.insert(0, HERE)
import types, jax
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice
from gfmc_fn_lib import RBMPPGuideG
from gfmc_run import load_consts, init_walkers
import gfmc_guide as GG
z = np.load(PARAMS); lat = Lattice(8, 0.5, 'translations'); c = load_consts(CONSTS, 0)
g_old = RBMPPGuideG({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, c['alpha'], c['E'], w=c["w"], chunk=32, use_rbm=bool(z['rbm']) if 'rbm' in z.files else True)
fn, _, _, _ = GG.guide_rbmpp(types.SimpleNamespace(params=PARAMS, consts=CONSTS, L=8, proj='translations'))
g_new = GG.LogpsiGuide(fn, 8, chunk=256)
assert np.array_equal(g_new.BI, lat.BI) and np.array_equal(g_new.BJ, lat.BJ) and np.array_equal(g_new.JB, lat.JB)
X, _ = init_walkers(lat, z, 64, 5, True, nsweep=500)
ro, do = g_old.rows(X); rn, dn = g_new.rows(X)
res['rows_rate_max_rel'] = float(np.max(np.abs(ro - rn)) / np.max(np.abs(ro))); res['rows_dFN_max_abs'] = float(np.max(np.abs(do - dn)))
res['rows_kept_pattern_equal'] = bool(np.array_equal(ro > 0, rn > 0))
ok = res['emix']['max_abs_diff'] < 1e-12 and res['hops_equal'] and res['ctl_seed_maxdiff'] > 1e-6 and res['ctl_dropvsf_maxdiff'] > 1e-6
res['PASS'] = bool(ok)
json.dump(res, open(os.path.join(a.out, 'equiv.json'), 'w'), indent=1)
print('launched emix', np.array2string(old['emix'], precision=12)); print('new      emix', np.array2string(new['emix'], precision=12))
print(f"EQUIV_RESULT {'PASS' if ok else 'FAIL'} | emix max|diff| {res['emix']['max_abs_diff']:.2e} (rel {res['emix']['max_rel_diff']:.1e}, bitwise {res['emix']['bitwise']}) "
      f"egrow max|diff| {res['egrow']['max_abs_diff']:.2e} hops equal {res['hops_equal']} | rows rate rel {res['rows_rate_max_rel']:.1e} dFN {res['rows_dFN_max_abs']:.1e} "
      f"| controls differ: seed+1 {res['ctl_seed_maxdiff']:.2e}, drop_vsf {res['ctl_dropvsf_maxdiff']:.2e}")
