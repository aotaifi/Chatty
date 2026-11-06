"""pool_x2_convert.py: stage-1 f_J rows (pool_x2/fjrun/fj_T.npz: raw f_J(x), K = 2 partner picks, raw f_J(y), W^B, ln|B_y/B_x|; sign sgn(M1), decades -7..-9) -> b2_prep.label_task-format parts
(mode 'fix': add = c sign(f) min(|f|, f0), node -69 where f is not finite).  Same quantities as pool_x2_label / b2_prep.label_task, no new lemma evaluation.
 python pool_x2_convert.py OUTDIR [--check TASK N]    (--check: compare against pool_x2_label's path on the first N rows of the task: lnT_x, WB, nblk exactly, lnT_y at the SAME partner configurations)"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool_fit as PF, b2_prep as B2
RUND = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8_r1/pool_x2'; C = 4.0; F0 = 1.22; NODE = -69.0


def fix(f, c=C, f0=F0, node=NODE):
    f = np.asarray(f, np.float64); fin = np.isfinite(f)
    return np.where(fin, c * np.sign(np.where(fin, f, 0.0)) * np.minimum(np.abs(np.where(fin, f, 0.0)), f0), node)


def convert(t, out, sub='prod', fjsub='fjrun', fidx=None):
    z = np.load(f'{RUND}/{sub}/out_{t}.npz'); f = np.load(f'{RUND}/{fjsub}/fj_{t}.npz'); assert int(f['final']) == 1 and int(z['final']) == 1
    idx = f['idx']; K = int(f['K']); lt = PF.lat8(); MASK = PF._BIT64[lt['BI']] | PF._BIT64[lt['BJ']]
    words = z['words'][idx]; lnBx = z['lnB'][idx]; pk = f['picks']; has = pk >= 0 if pk.ndim == 2 else None
    y = np.where(has, words[:, None] ^ MASK[np.maximum(pk, 0)], words[:, None]); dl = np.where(has, f['dlnB'], 0.0); lBy = lnBx[:, None] + dl
    addx = fix(f['fx']); addy = fix(np.where(has, f['fy'], np.nan)); lTx = lnBx + addx; lTy = np.where(has, lBy + addy, lTx[:, None])
    nblk = f['nblk'].astype(np.int32); WB = f['WB']
    rho = np.where(nblk[:, None] > 0, WB[:, None] * np.exp(np.clip((lTy - lBy) - (lTx - lnBx)[:, None], -700, 700)), 0.0)
    np.savez(out, idx=idx, x=words, lnB_x=lnBx, orb=z['orb'][idx], beta=z['beta'][idx], src=np.asarray(f'{RUND}/{sub}/out_{t}.npz'), fidx=np.int64(fidx if fidx is not None else 1000 + t),
             seed=np.int64(f['seed']), y=y, lnB_y=lBy, WB=WB, nblk=nblk, lnT_x=lTx, lnT_y=lTy, rho=rho, lnB_guide_vs_stored_maxdev=np.float64(0.0), guard_checked=np.int64(f['guard_checked']),
             guard_fallbacks=np.int64(f['guard_fallbacks']), min_margin=np.float64(np.nan), sec=np.float64(f['sec']), code_sha256=np.asarray(B2.sha(os.path.abspath(__file__))), final=np.int64(1),
             converted_from=np.asarray(f'{RUND}/{fjsub}/fj_{t}.npz'))
    return len(idx)


def check(t, n, spec):
    import pool_x2_label as L
    SP = json.load(open(spec)); SP = dict(SP, decs_rel=[-7, -8, -9]); z = np.load(f'{RUND}/prod/out_{t}.npz'); f = np.load(f'{RUND}/fjrun/fj_{t}.npz')
    convert(t, '/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/f76c4540-52b2-43a7-b12a-5e1e5b7f3b53/scratchpad/conv_chk.npz'); c = np.load('/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/f76c4540-52b2-43a7-b12a-5e1e5b7f3b53/scratchpad/conv_chk.npz')
    sel = f['idx'][:n]; L.label_task_fast('/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/f76c4540-52b2-43a7-b12a-5e1e5b7f3b53/scratchpad/lab_chk.npz', SP, f'{RUND}/prod/out_{t}.npz', 7, select=sel, log=lambda m: None)
    l = np.load('/scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/f76c4540-52b2-43a7-b12a-5e1e5b7f3b53/scratchpad/lab_chk.npz')
    guide = B2.guide_of(SP); sgn = B2.M1Sign(sign_params=guide, guide=guide); lab = B2.labels_m1(guide, sgn)
    res = dict(n=int(n), lnT_x_maxdiff=float(np.max(np.abs(c['lnT_x'][:n] - l['lnT_x']))), WB_maxreldiff=float(np.max(np.abs(c['WB'][:n] - l['WB']) / l['WB'])), nblk_equal=bool(np.array_equal(c['nblk'][:n], l['nblk'])))
    yy = c['y'][:n].reshape(-1); lTy_b2 = lab(yy, np.zeros(len(yy), np.int64)).reshape(n, -1)
    res['lnT_y_maxdiff_same_partner'] = float(np.max(np.abs(c['lnT_y'][:n] - lTy_b2))); res['rho_maxreldiff_same_partner'] = float(np.max(np.abs(c['rho'][:n] - np.where(c['nblk'][:n, None] > 0, c['WB'][:n, None] * np.exp(np.clip((lTy_b2 - c['lnB_y'][:n]) - (l['lnT_x'] - c['lnB_x'][:n])[:, None], -700, 700)), 0)) / np.maximum(c['rho'][:n], 1e-300)))
    print(json.dumps(res)); return res


if __name__ == '__main__':
    if sys.argv[1] == '--check': check(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        out = sys.argv[1]; os.makedirs(out, exist_ok=True); tot = 0
        for t in range(1, 41): tot += convert(t, f'{out}/part_{t}.npz', fidx=1000 + t)
        print('converted rows', tot)
