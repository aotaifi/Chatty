#!/bin/zsh
set -euo pipefail
PY=/Users/aliotaifi/j1j2_vit_bench/.venv311/bin/python
SCRIPT=/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/experiments/closed_fn_krylov_exact4x4.py
OUT=/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure/results

echo "=== CONTROL: exact target modulus initialization, Marshall signs ==="
$PY $SCRIPT \
  --target-j2 0.5 \
  --init-source-j2 0.5 \
  --maxiter 30 \
  --out $OUT/closed_fn_krylov_4x4_J2p5_exactinit.json

echo "=== NO-ORACLE BOOTSTRAP: J2=0 modulus initialization, Marshall signs ==="
$PY $SCRIPT \
  --target-j2 0.5 \
  --init-source-j2 0.0 \
  --maxiter 30 \
  --out $OUT/closed_fn_krylov_4x4_J2p5_J2zero_init.json

echo "=== DONE ==="
