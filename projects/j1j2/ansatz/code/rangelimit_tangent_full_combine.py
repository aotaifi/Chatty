"""rangelimit_tangent_full_combine.py OUTDIR M SCALE SEED NBLOCK: assemble the blocks of rangelimit_tangent_full.py and print the exact shares.
share(lam) = b^T (G + lam tr(G)/P I)^-1 b / yy, both norms, targets u, xi, rtheta_other; E[share_xi] (L2) = tr((G + lam..)^-1 H) / sum w."""
import sys, json, glob, numpy as np
OUT, M, SC, SD, NB = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
base = f'M{M}_s{SC}_seed{SD}'; Z = [np.load(f'{OUT}/blk_{base}_b{k}of{NB}.npz') for k in range(NB)]; JS = [json.load(open(f'{OUT}/blk_{base}_b{k}of{NB}.json')) for k in range(NB)]
cat = lambda key: np.concatenate([z[key] for z in Z], 1)
GL, HL, GQ = cat('GL'), cat('HL'), cat('GD') - cat('GA'); P = GL.shape[0]
TG = ('u', 'xi', 'rtheta_other'); LAMS = [10.0 ** k for k in range(-10, 1)]
out = dict(M=M, scale=SC, seed=SD, P=P, rows=JS[0]['rows'], lams=LAMS, sec_blocks=[j['sec'] for j in JS], hosts=[j['host'] for j in JS])
for nm, G, bk, yk in (('L2', GL, 'bL', 'yL'), ('Q', GQ, 'bQ', 'yQ')):
    asym = float(np.abs(G - G.T).max() / np.abs(G).max()); G = 0.5 * (G + G.T); ev, V = np.linalg.eigh(G); evc = np.maximum(ev, 0.0); tr = np.trace(G) / P
    o = dict(asym_rel=asym, eig_max=float(ev[-1]), eig_min=float(ev[0]), n_neg=int((ev < 0).sum()), rank_1e8=int((ev > 1e-8 * ev[-1]).sum()), rank_1e12=int((ev > 1e-12 * ev[-1]).sum()))
    for t in TG:
        bb = Z[0][f'{bk}_{t}']; assert all(np.allclose(z[f'{bk}_{t}'], bb, rtol=1e-10, atol=1e-14) for z in Z[1:]); yy = JS[0][yk][t]
        c = V.T @ bb; o[t] = dict(yy=yy, share=[float((c * c / (evc + lam * tr)).sum() / yy) for lam in LAMS])
    if nm == 'L2':
        Hr = V.T @ (0.5 * (HL + HL.T)) @ V; o['xi_expected'] = [float((np.diag(Hr) / (evc + lam * tr)).sum() / JS[0]['sum_w']) for lam in LAMS]; o['sum_w'] = JS[0]['sum_w']
    else: o['sum_d'] = JS[0]['sum_d']
    out[nm] = o
json.dump(out, open(f'{OUT}/full_{base}.json', 'w'), indent=1)
for nm in ('L2', 'Q'):
    o = out[nm]; print(f"{base} {nm}: asym {o['asym_rel']:.1e} rank(1e-8) {o['rank_1e8']} eig_min/max {o['eig_min'] / o['eig_max']:.1e} yy_u {o['u']['yy']:.6e}")
    for i, lam in enumerate(LAMS):
        print(f"   lam {lam:.0e}: u {o['u']['share'][i]:.4f} xi {o['xi']['share'][i]:.4f} r' {o['rtheta_other']['share'][i]:.4f}" + (f" E[xi] {o['xi_expected'][i]:.4f}" if nm == 'L2' else ''))
