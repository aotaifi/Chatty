"""consol_verify_target.py: the target ln A_1 (rangelimit_6x6/rounds23/lA_1.npy) is on the same 15804956-orbit ordering as A_0's la_table.npy.
Recomputes l = ln max|1 + alpha_0 (E_L - E)|, f_J[G_0] (E' = E_FN[s2 G_0]) and 1.307 f_J on selected orbits from A_0's table and the neighbour tables
(rangelimit_rounds.py formulas, numpy, rows only) and compares lA_1 - la_A0.  usage: python consol_verify_target.py OUT.json [n_random]"""
import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import bond_split as BS
A0 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy'
A1 = '/scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6/rounds23/lA_1.npy'
RLA = '/scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6/rangelimit_A.json'
OUT = sys.argv[1]; NR = int(sys.argv[2]) if len(sys.argv) > 2 else 400
la0 = np.load(A0).astype(np.float64); lA1 = np.load(A1).astype(np.float64); D = la0.size; assert lA1.size == D == 15804956
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); assert n.size == D
j = json.load(open(RLA)); E = BS.E0 + BS.NS * j['Evar_A0_site']; EPG0 = j['step1']['E_FN']; AL = -0.1; CJ = 1.307
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
a = np.exp(la0 - la0.max())
def rows(idx):
    idx = np.asarray(idx); t = np.concatenate([t1[idx], t2[idx]], 1); va = t >= 0; tt = np.where(va, t, 0)
    ss = s[idx, None] * s[tt]; kept = va & (ss < 0); viol = va & ~kept
    return idx, tt, va, kept, viol
def EL_of(idx):
    idx, tt, va, kept, viol = rows(idx)
    D0 = 27.0 - 0.5 * (va * BS.JW[None, :]).sum(1)
    H = D0 * a[idx] + (viol * BS.HW * a[tt]).sum(1) - (kept * BS.HW * a[tt]).sum(1)
    return H / a[idx]
def l_of(idx):
    return np.log(np.maximum(np.abs(1.0 + AL * (EL_of(idx) - E)), np.exp(-60.0)))
def fJ_of(idx):
    idx, tt, va, kept, viol = rows(idx); k = idx.size
    lg_x = la0[idx] + l_of(idx)
    lg_y = (la0[tt] + l_of(tt.reshape(-1)).reshape(tt.shape))
    ratio = np.exp(np.clip(lg_y - lg_x[:, None], -700, 700))
    Hxx = 27.0 - 0.5 * (va * BS.JW[None, :]).sum(1)
    VA = (viol * BS.HW * ratio).sum(1); SK = (kept * BS.HW * ratio).sum(1)
    return np.log(SK) - np.log(Hxx + VA - EPG0)
diff = lA1 - la0
rng = np.random.default_rng(7); sub = rng.choice(D, 2_000_000, replace=False)
pick = np.unique(np.concatenate([rng.choice(D, NR, replace=False), sub[np.argsort(diff[sub])[:100]], sub[np.argsort(diff[sub])[-100:]]]))
l = l_of(pick); fj = fJ_of(pick); rec = l + CJ * fj; d = diff[pick] - rec
shifted = diff[np.minimum(pick + 1, D - 1)] - rec
out = dict(n_checked=int(pick.size), max_abs_dev=float(np.abs(d).max()), rms_dev=float(np.sqrt((d ** 2).mean())), diff_range=[float(diff[pick].min()), float(diff[pick].max())],
           rms_of_diff=float(np.sqrt((diff[pick] ** 2).mean())), control_shift_by_one_orbit_rms_dev=float(np.sqrt((shifted ** 2).mean())), E=E, EPG0=EPG0,
           diff_full_range=[float(diff.min()), float(diff.max())])
json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out))
