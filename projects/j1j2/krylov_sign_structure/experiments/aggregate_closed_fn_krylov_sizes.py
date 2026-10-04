#!/usr/bin/env python3
"""Table of loop iterations vs system size from the symmetric-sector histories (json or jsonl)."""
import json, sys, glob, os
rows = []
def load(path):
    if path.endswith(".json"):
        d = json.load(open(path)); return d["history"]
    return [json.loads(l) for l in open(path)]
def first(h, pred):
    for r in h:
        if pred(r): return r["it"]
    return None
def settled(h, pred):
    k = None
    for r in reversed(h):
        if pred(r): k = r["it"]
        else: break
    return k
specs = [a.split("=", 1) for a in sys.argv[1:]]   # label=path
print(f"{'run':28s} {'nit':>5s} {'eps<1e-5':>8s} {'1e-6':>6s} {'1e-7':>6s} {'ws<1e-8':>8s} {'1e-10':>6s} {'1e-12':>6s} {'ws=0(<1e-14)':>12s} {'sec/it':>7s} {'fnmv/it':>8s}")
for lab, path in specs:
    h = load(path)
    hh = h[1:]
    sec = sum(r.get("sec", 0) for r in hh) / max(1, len(hh))
    mv = sum(r.get("fn_matvecs", 0) for r in hh) / max(1, len(hh))
    f = lambda p: first(h, p)
    print(f"{lab:28s} {h[-1]['it']:5d} {str(f(lambda r: r['eps'] < 1e-5)):>8s} {str(f(lambda r: r['eps'] < 1e-6)):>6s} {str(f(lambda r: r['eps'] < 1e-7)):>6s} "
          f"{str(settled(h, lambda r: r['w_s'] < 1e-8)):>8s} {str(settled(h, lambda r: r['w_s'] < 1e-10)):>6s} {str(settled(h, lambda r: r['w_s'] < 1e-12)):>6s} "
          f"{str(settled(h, lambda r: r['w_s'] < 1e-14)):>12s} {sec:7.1f} {mv:8.1f}")
