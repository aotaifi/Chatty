"""Pre-registered FN recheck test (A0.3): for pairs with |Re r_f32h| < thr |r_f32h| take the f64 value of r (=recheck), else f32h. Same 2000 configs.
Reports fraction rechecked, flips vs f64 before/after, E_L and V_sf bias/rms after. usage: python vit10_recheck.py DIR [thr=1e-3] [mode=rel|absr|absre]   rel = registered trigger |Re r|<thr|r|; absr (post hoc) |r|<thr; absre (post hoc) |Re r|<thr"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import validate_vit10 as VV
VERD = "--verdict" in sys.argv; sys.argv = [a for a in sys.argv if a != "--verdict"]
D = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
z = np.load(f"{D}/prec_symm.npz"); ii, bb = z["ii"], z["bb"]; n = ii.max() + 1; K = VV.BJ[bb] / 2
r64 = np.exp(z["f64_y"] - z["f64_x"][ii]); r32 = np.exp(z["f32h_y"] - z["f32h_x"][ii])
def q(r):
    off = np.zeros(n, complex); np.add.at(off, ii, K * r); v = np.zeros(n); m = r.real > 0; np.add.at(v, ii[m], (K * np.abs(r))[m]); return off.real / 100, v / 100
mode = sys.argv[3] if len(sys.argv) > 3 else "rel"
sel = {"rel": np.abs(r32.real) < thr * np.abs(r32), "absr": np.abs(r32) < thr, "absre": np.abs(r32.real) < thr}[mode]; rc = np.where(sel, r64, r32)
f0 = int(np.sum(np.sign(r32.real) != np.sign(r64.real))); f1 = int(np.sum(np.sign(rc.real) != np.sign(r64.real)))
flipsel = np.sign(r32.real) != np.sign(r64.real)
e0, v0 = q(r64)
for nm, r in (("f32h", r32), ("recheck", rc)):
    e, v = q(r); de, dv = e - e0, v - v0
    print(f"RECHECK {nm}: E_L bias {de.mean():.2e}+-{de.std(ddof=1)/np.sqrt(n):.1e} rms {np.sqrt((de**2).mean()):.2e} | V_sf bias {dv.mean():.2e}+-{dv.std(ddof=1)/np.sqrt(n):.1e} rms {np.sqrt((dv**2).mean()):.2e}")
fr = sel.mean(); print(f"RECHECK mode {mode} thr {thr}: pairs rechecked {sel.sum()} of {len(sel)} = {fr:.3e}; flips before {f0} after {f1}; flipped pairs inside recheck set: {int((flipsel & sel).sum())}/{f0}")
sf, s64 = 3000., 200.   # symm evals/s, A40 (bench.json); a recheck costs one f64 eval of y (x in f64 is one per config)
print(f"RECHECK cost per pair: f32h {1/sf*1e3:.3f} ms + {fr:.2e}*{1/s64*1e3:.1f} ms (f64 y) -> x{1 + fr*sf/s64:.3f} throughput cost; +1 f64 eval of x only if any pair of x is rechecked (<= {1/s64*1e3:.1f} ms per config vs {226/sf*1e3:.0f} ms)")

if VERD:   # pre-registered held-out criteria: 0 flips after recheck, E_L and V_sf |bias| < 2e-7/site, recheck fraction < 1%
    e, v = q(rc); ok = f1 == 0 and abs((e - e0).mean()) < 2e-7 and abs((v - v0).mean()) < 2e-7 and fr < 0.01
    print("RECHECK VERDICT held-out:", "PASS" if ok else "FAIL", f"(flips after {f1}, |E_L bias| {abs((e-e0).mean()):.1e}, |V_sf bias| {abs((v-v0).mean()):.1e}, fraction {fr:.2e})")
