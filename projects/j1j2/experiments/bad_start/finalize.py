#!/usr/bin/env python3
"""Merge per-start part files (scratch cache) into compact result files under results/bad_start/data/.
Columnar format: one dict of arrays per run (keys once) -> small files."""
import json, sys
from pathlib import Path
parts = Path(sys.argv[1]); out = Path(sys.argv[2]); lattice = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
KEYS = ["E_FN", "eps_FN", "eps_guide", "w_s", "ov_gs", "ov_ex1", "ov_ex2", "S2", "n_sign_flips"]
def r4(x): return float(f"{x:.5g}") if isinstance(x, float) else x
for method in ("plain", "anderson"):
    runs = {}
    for f in sorted(parts.glob(f"{lattice}_{method}_*.json")):
        d = json.loads(f.read_text())
        h = d["history"]
        n_full = len(h)
        if n_full > 1000: h = [r for r in h if r["it"] <= 800 or r["it"] % 10 == 0 or r["it"] == h[-1]["it"]]
        cols = {k: [r4(r.get(k)) if k in r else None for r in h] for k in KEYS if k in h[0] or k == "n_sign_flips"}
        symk = [k for k in h[0] if k.startswith("sym_") or k.startswith("ov_sec_")]
        for k in symk: cols[k] = [r4(r[k]) for r in h]
        runs[d["name"]] = dict(desc=d["desc"], E0=d["E0"], summary=d["summary"], hist=cols, it=[r["it"] for r in h])
    if runs:
        p = out / f"bad_start_{lattice}_{method}.json"
        p.write_text(json.dumps(dict(lattice=lattice, method=method, runs=runs), separators=(",", ":")))
        print(p, p.stat().st_size // 1024, "kB", len(runs), "runs")
