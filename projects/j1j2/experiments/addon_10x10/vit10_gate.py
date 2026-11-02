"""Revised f32 gate (A0.3): reads prec_{kind}.npz (vit10_prec.py). Per site (/100): bias (mean), rms of f32-f64 for Re E_L and V_sf, and sign flips of Re r.
usage: python vit10_gate.py DIR"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import validate_vit10 as VV
D = sys.argv[1]; ok = True
for kind in ("symm", "plain"):
    z = np.load(f"{D}/prec_{kind}.npz"); ii, bb = z["ii"], z["bb"]; n = ii.max() + 1; K = VV.BJ[bb] / 2
    X = VV.spins().astype(float)  # configs not stored; recover diag-free quantities only (differences cancel diag)
    def quant(tag):
        r = np.exp(z[tag + "_y"] - z[tag + "_x"][ii]); off = np.zeros(n, complex); np.add.at(off, ii, K * r)
        v = np.zeros(n); viol = r.real > 0; np.add.at(v, ii[viol], (K * np.abs(r))[viol]); return off.real / 100, v / 100, r
    e0, v0, r0 = quant("f64")
    for tag in ("f32h", "f32d"):
        e, v, r = quant(tag); de, dv = e - e0, v - v0; flips = int(np.sum(np.sign(r.real) != np.sign(r0.real)))
        bias = lambda a: (a.mean(), a.std(ddof=1) / np.sqrt(n)); rms = lambda a: np.sqrt((a ** 2).mean())
        g = abs(de.mean()) < 2e-7 and abs(dv.mean()) < 2e-7 and rms(de) < 1e-6 and rms(dv) < 1e-6 and flips == 0
        print(f"GATE {kind} {tag}: E_L bias {bias(de)[0]:.2e}+-{bias(de)[1]:.1e} rms {rms(de):.2e} | V_sf bias {bias(dv)[0]:.2e}+-{bias(dv)[1]:.1e} rms {rms(dv):.2e} | flips {flips} of {len(r)} | {'PASS' if g else 'FAIL'}")
        if tag == "f32h" and kind == "symm": ok = g
print("GATE VERDICT: f32 (highest) usable for symm guide" if ok else "GATE VERDICT: f32 (highest) NOT usable")
