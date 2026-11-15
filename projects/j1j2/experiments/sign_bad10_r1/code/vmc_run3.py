"""(vmc_run3 = EXACT two-hop estimator, step 0 method) T-BAD10 round 1, stage VMC: alpha_0, E_0 = one-step Lanczos moments of the (Marshall x |ViT|) guide, >= 4000 samples in C independent Metropolis chains, jackknife over chains.
h1 and h2 are EXACT (two-hop set of every sample, 8.3 s/sample; the subsampled estimator failed its gate in job 17997722).
Writes OUT/samples.npz, OUT/consts_alpha0_E0.json and OUT/consts_alpha0_E0.json.sha256 (written ONCE; later stages read and never recompute).
usage: python vmc_run3.py OUT --C 8 --samples 400 --maxsec S --seed 5"""
import sys, os, time, json, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--C', type=int, default=8); ap.add_argument('--K', type=int, default=100); ap.add_argument('--Kb', type=int, default=16)
ap.add_argument('--samples', type=int, default=400); ap.add_argument('--maxsec', type=float, default=6000); ap.add_argument('--seed', type=int, default=5); ap.add_argument('--burn', type=int, default=300); ap.add_argument('--nexact_rounds', type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import vitlib as VL, vmc10 as VM
P = lambda *s: print(time.strftime('%H:%M:%S'), 'PROGRESS', *s, flush=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
ev = VL.make_vit('f32h'); S = VL.shipped(); rng = np.random.default_rng(a.seed); t0 = time.time()
ia = rng.choice(len(S), 96, replace=False); h1, vt, _, ne = VM.h1_one_hop(ev, S[ia], 1024); Eref = float(np.mean(vt * h1)); P('stage A E_ref', Eref, 'sd', float(np.std(vt * h1)), 'sec', time.time() - t0)
rest = np.setdiff1d(np.arange(len(S)), ia); X = S[rng.choice(rest, a.C, replace=False)].copy()
acc = VM.metropolis(ev, X, rng, a.burn, a.C); P('burn acc', acc)
R = {k: [] for k in ('chain', 'rnd', 'vt', 'h1', 'h2')}; rnd = 0; PAIR = []
while len(R['vt']) < a.samples and time.time() - t0 < a.maxsec:
    acc = VM.metropolis(ev, X, rng, a.K, a.C); ex = [VM.two_hop(ev, X[w], 1024) for w in range(a.C)]; vt = np.array([e['vt'] for e in ex]); h1 = np.array([e['h1'] for e in ex]); h2 = np.array([e['h2'] for e in ex]); ne = sum(e['n_unique'] for e in ex)
    R['chain'] += list(range(a.C)); R['rnd'] += [rnd] * a.C; R['vt'] += list(vt); R['h1'] += list(h1); R['h2'] += list(h2); rnd += 1
    if rnd % 5 == 0:
        np.savez(a.out + '/samples.npz.tmp.npz', **{k: np.array(v) for k, v in R.items()}, E_ref=Eref); os.replace(a.out + '/samples.npz.tmp.npz', a.out + '/samples.npz')
        P('round', rnd, 'samples', len(R['vt']), 'acc', f'{acc:.3f}', 'evals/round', ne, 'sec', f'{time.time() - t0:.0f}')
np.savez(a.out + '/samples.npz.tmp.npz', **{k: np.array(v) for k, v in R.items()}, E_ref=Eref); os.replace(a.out + '/samples.npz.tmp.npz', a.out + '/samples.npz')
n = len(R['vt']); d = {k: np.array(v) for k, v in R.items()}; vt, h1, h2, ch = d['vt'], d['h1'], d['h2'], d['chain']
H1 = h1 - Eref * vt; H2 = h2 - 2 * Eref * h1 + Eref ** 2 * vt
def est(m):
    s = (1.0, np.mean(vt[m] * H1[m]), np.mean(H1[m] ** 2), np.mean(H1[m] * H2[m]), 0.0)
    m1, m2, m3 = s[1], s[2], s[3]; mu2 = m2 - m1 ** 2; mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    al = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2); return dict(E=Eref + m1, var=mu2, mu3=mu3, alpha=al, E1=Eref + m1 + al * mu2, chk_vtH2=float(np.mean(vt[m] * H2[m])), chk_H1sq=float(m2))
r0 = est(np.ones(n, bool)); J = [est(ch != c) for c in range(a.C)]; out = dict(n_samples=int(n), chains=a.C, estimator='exact two-hop (vmc10.two_hop)', E_ref=Eref)
for k in ('E', 'var', 'mu3', 'alpha', 'E1'):
    v = np.array([j[k] for j in J]); out[k] = float(r0[k]); out[k + '_err'] = float(np.sqrt((a.C - 1) / a.C * np.sum((v - v.mean()) ** 2)))
out['chk_vtH2'] = r0['chk_vtH2']; out['chk_H1sq'] = r0['chk_H1sq']
cm = np.array([np.mean(vt[ch == c] * H1[ch == c]) for c in range(a.C)]); out['chain_E_means'] = (Eref + cm).tolist()
out['chain_E_chi2_per_dof'] = float(np.sum((cm - cm.mean()) ** 2 / (np.array([np.std(vt[ch == c] * H1[ch == c], ddof=1) / np.sqrt(np.sum(ch == c)) for c in range(a.C)]) ** 2)) / (a.C - 1))
hi = d['rnd'] >= np.median(d['rnd']); out['alpha_firsthalf'] = est(~hi)['alpha']; out['alpha_secondhalf'] = est(hi)['alpha']
out['samples_sha256'] = sha(a.out + '/samples.npz'); out['code_sha256'] = {f: sha(os.path.join(HERE, f)) for f in ('vmc_run3.py', 'vmc10.py', 'vitlib.py', 'vit10_jax.py')}
if PAIR:
    pr = np.array(PAIR); dd = pr[:, 0] - pr[:, 1]; out['paired_exact_h2'] = dict(n=int(len(dd)), mean_sub_minus_exact=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))), z=float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))), sd_sub_minus_exact=float(dd.std(ddof=1)), sd_exact_h2=float(pr[:, 1].std(ddof=1)))
out['units'] = 'totals, spin-1/2 units, N=100; alpha = Ritz one-step coefficient of the Marshall x |ViT| guide'
import vmc_gate; ok, why, rel = vmc_gate.gate(out, need_paired=False); out['passed'] = bool(ok); out['gate_reasons'] = why; out['alpha_rel_err'] = float(rel)
cj = a.out + '/consts_alpha0_E0.json'; json.dump(out, open(cj, 'w'), indent=1); open(cj + '.sha256', 'w').write(sha(cj) + '\n')
P('CONSTS', json.dumps({k: out[k] for k in ('alpha', 'alpha_err', 'E', 'E_err', 'n_samples')}), 'sha256', sha(cj), 'passed', out['passed'], why)
if not ok: P('VMC_R1_GATE_FAILED'); sys.exit(3)
P('VMC_R1_DONE')
