#!/usr/bin/env python3
"""Rewrite old absolute paths (Chatty consolidation 2026-10-24) using PATHS_MOVED.tsv.
Longest old prefix first; an old prefix matches only if not followed by [A-Za-z0-9_-].
Scope: git-tracked md/py/sh/sbatch/json under projects/j1j2 and tools (minus this directory).
Usage: rewrite_paths.py [--apply]   (run from the worktree root)"""
import re, subprocess, sys, collections, os
TSV = "/project/theorie/a/A.Otaifi/Chatty/runs/PATHS_MOVED.tsv"
EXT = (".md", ".py", ".sh", ".sbatch", ".json")
SKIP = "projects/j1j2/tools/path_rewrite_20261024/"
apply = "--apply" in sys.argv
rows = [l.rstrip("\n").split("\t") for l in open(TSV)][1:]
rows = sorted([r for r in rows if len(r) == 2], key=lambda r: -len(r[0]))
pat = re.compile("(" + "|".join(re.escape(o) for o, _ in rows) + r")(?![A-Za-z0-9_-])")
m = dict(rows)
files = subprocess.check_output(["git", "ls-files", "projects/j1j2", "tools"], text=True).split("\n")
per_file = collections.Counter(); per_row = collections.Counter(); changed = []
for f in files:
    if not f.endswith(EXT) or f.startswith(SKIP) or not os.path.isfile(f) or os.path.islink(f):
        continue
    try: s = open(f, encoding="utf-8").read()
    except Exception as e:
        print("SKIP unreadable", f, e); continue
    def sub(mo):
        per_row[mo.group(1)] += 1; return m[mo.group(1)]
    t, n = pat.subn(sub, s)
    if n:
        per_file[f] = n; changed.append(f)
        if apply:
            open(f, "w", encoding="utf-8").write(t)
print("MODE", "APPLY" if apply else "DRY")
print("FILES", len(per_file), "REPLACEMENTS", sum(per_file.values()))
print("--- per mapping row")
for o, c in sorted(per_row.items(), key=lambda x: -x[1]): print(c, o, "->", m[o])
print("--- per file")
for f, c in sorted(per_file.items()): print(c, f)
