#!/usr/bin/env python3
"""Reproduce the exact-20-site FN/Krylov sign-learning diagnostics in an isolated scratch directory.

All archived research scripts originally used /mnt/data paths. Stage patched *copies*
of the scripts, never edit the tracked originals. Requires numpy/scipy; torch for CNN.
Example:
  python reproduce.py --workdir ~/scratch/j1j2_20 --stage data
  python reproduce.py --workdir ~/scratch/j1j2_20 --stage diagnostics
  python reproduce.py --workdir ~/scratch/j1j2_20 --stage cnn

Tested analytical checks: j1j2_real_fn_drift_validation20.py, j1j2_margin_precision20.py.
This remains an exact-amplitude oracle diagnostic, NOT a scalable FN solver.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

FILES = [
    "fn_collapse_exact20_krylov_v2.py",
    "prepare_sign20.py", "audit_sign20_targets.py",
    "measure_suppression20_fast.py",
    "j1j2_local_query_diagnostic.py", "j1j2_budgeted_orbit20.py",
    "j1j2_noisy_oracle_diagnostic.py", "j1j2_margin_precision20.py",
    "j1j2_robust_gate20.py",
    "j1j2_real_fn_drift20.py", "j1j2_real_fn_drift_validation20.py",
    "j1j2_orbit_priority20.py", "train_sign20_corrected.py",
    "j1j2_neural_guide20.py", "j1j2_walker_only_amplitude20.py",
    "j1j2_walker_only_eval20.py", "j1j2_walker_focus20.py",
    "j1j2_walker_focus20_eval.py",
]
STAGES = {
    "data": ["prepare_sign20.py", "audit_sign20_targets.py", "measure_suppression20_fast.py"],
    "diagnostics": [
        "j1j2_budgeted_orbit20.py",
        "j1j2_noisy_oracle_diagnostic.py",
        "j1j2_margin_precision20.py",
        "j1j2_real_fn_drift20.py",
        "j1j2_real_fn_drift_validation20.py",
    ],
    "cnn": ["train_sign20_corrected.py"],
    "walker": ["j1j2_walker_only_amplitude20.py", "j1j2_walker_only_eval20.py", "j1j2_walker_focus20.py", "j1j2_walker_focus20_eval.py"],
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,required=True)
    ap.add_argument("--stage",choices=("data","diagnostics","cnn","walker","all","stage-only"),default="stage-only")
    ap.add_argument("--overwrite-copied-code",action="store_true")
    args=ap.parse_args()
    src=Path(__file__).resolve().parent
    wd=args.workdir.expanduser().resolve()
    wd.mkdir(parents=True,exist_ok=True)
    # Clone scripts with paths rewritten only in scratch, retaining source as archived.
    for name in FILES:
        sf=src/name
        if not sf.exists():
            raise FileNotFoundError(f"Missing archived dependency {sf}")
        dest=wd/name
        if dest.exists() and not args.overwrite_copied_code:
            continue
        text=sf.read_text().replace("/mnt/data", str(wd))
        dest.write_text(text)
    if args.stage=="stage-only":
        print("Staged scripts at",wd)
        print("Now run --stage data, then --stage diagnostics, and optionally --stage cnn.")
        return
    steps=sum((STAGES[k] for k in ("data","diagnostics","cnn","walker")),[]) if args.stage=="all" else STAGES[args.stage]
    if args.stage in ("diagnostics","cnn","walker") and not (wd/"sign20_training_data.npz").exists():
        raise FileNotFoundError(f"Run --stage data first to create {wd/'sign20_training_data.npz'}")
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    if args.stage in ("cnn","all"):
        env.update(NSTEPS="3000",NSEEDS="5")
    for name in steps:
        print("RUN",name,flush=True)
        subprocess.run([sys.executable,name], cwd=str(wd),env=env,check=True)
    print("All requested stages finished. Results saved at",wd,flush=True)

if __name__=="__main__":
    main()
