"""m64_gate.py E0EXT_C1 E0EXT_C2 M64_DIR OUT.json : population-bias gate.  e_0(M16) = the two extended chains (quotable.series_chain per chain, mean of chain means weighted 1/err^2, error = max(combined, scatter) as quotable.py series);
e_0(M64) = the single M=64 chain (series_chain mean and binning error).  PASS iff |e_0(M64) - e_0(M16)| < 2e-4.  Exit 3 on failure (afterok stops vmc).  Prints z = diff / hypot(errs) as information.
Options for the broken check: --shift X adds X to the M64 series."""
import sys, json, numpy as np
sys.path.insert(0, '/project/theorie/a/A.Otaifi/tools'); import quotable as Q
a = [x for x in sys.argv[1:] if not x.startswith('--')]; shift = float(sys.argv[sys.argv.index('--shift') + 1]) if '--shift' in sys.argv else 0.0
if '--shift' in sys.argv: a = [x for x in a if x != sys.argv[sys.argv.index('--shift') + 1]]
c1, c2, d64, out = a
x1 = np.load(c1 + '/E_mix.npy'); x2 = np.load(c2 + '/E_mix.npy'); x64 = np.load(d64 + '/E_mix.npy') + shift
r = [Q.series_chain(x) for x in (x1, x2)]; m = np.array([q['mean'] for q in r]); e = np.array([q['err'] for q in r]); w = 1 / e ** 2
e16 = float(np.sum(w * m) / np.sum(w)); s16 = float(max(1 / np.sqrt(np.sum(w)), np.std(m, ddof=1) / np.sqrt(2)))
q = Q.series_chain(x64); e64 = q['mean']; s64 = q['err']; d = e64 - e16; z = d / np.hypot(s16, s64); ok = bool(abs(d) < 2e-4 and len(x64) >= 256)
res = dict(e0_M16=e16, err_M16=s16, n16=[len(x1), len(x2)], e0_M64=float(e64), err_M64=float(s64), n64=int(len(x64)), diff=float(d), z=float(z), threshold=2e-4, passed=ok, shift_applied=shift)
json.dump(res, open(out, 'w'), indent=1); print('M64_GATE', 'PASS' if ok else 'FAIL', json.dumps(res)); sys.exit(0 if ok else 3)
