#!/usr/bin/env python3
"""Tables for README from results/bad_start/data/*.json."""
import json, sys
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[2] / "results" / "bad_start" / "data"

def load(lat, meth):
    p = D / f"bad_start_{lat}_{meth}.json"
    runs = json.loads(p.read_text())["runs"] if p.exists() else {}
    for r in runs.values():
        if "it" in r: r["hist"]["_it"] = r["it"]
    return runs

def growth(h):
    """per-iteration growth factor of the GS weight (ov_gs) in the linear regime 1e-14<ov<1e-3"""
    ov = np.array(h["ov_gs"], float); it = np.array(h.get("_it", range(len(ov))))
    m = (ov > 1e-14) & (ov < 1e-3)
    # restrict to monotone-growth window after the plateau
    if m.sum() < 5: return None
    idx = np.where(m)[0]
    # last contiguous block
    blocks = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    b = max(blocks, key=len)
    if len(b) < 5: return None
    sl = np.polyfit(it[b], np.log(ov[b]), 1)[0]
    return float(sl)

def esc(h, thr=1e-2):
    e = np.array(h["eps_FN"], float)
    i = np.where(e < thr)[0]
    return int(i[0]) if len(i) else None

def row(name, r):
    s = r["summary"]; h = r["hist"]
    g = growth(h)
    f = lambda x: "-" if x is None else (f"{x:.2e}" if isinstance(x, float) else str(x))
    return (f"| `{name}` | {h['eps_FN'][0]:.2e} | {h['w_s'][0]:.3f} | {h['ov_gs'][0]:.2e} | {s['final_eps_FN']:.2e} | "
            f"{s['final_w_s']:.1e} | {s['final_ov_gs']:.4f} | {s['n_iter']} | {f(s['it_epsFN_1e2'])} | {f(s['it_epsFN_1e6'])} | "
            f"{f(s['it_epsFN_1e9'])} | {s['terminal'][:9]} |")

if __name__ == "__main__":
    for lat in ("4x4", "20"):
        for meth in ("plain", "anderson"):
            runs = load(lat, meth)
            if not runs: continue
            print(f"\n### {lat} {meth}\n")
            print("| start | eps_FN(0) | w_s(0) | overlap GS (0) | eps_FN final | w_s final | overlap GS final | n_iter | it eps<1e-2 | it eps<1e-6 | it eps<1e-9 | end |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for n, r in runs.items(): print(row(n, r))
