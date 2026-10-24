#!/bin/bash
#SBATCH --job-name=g8reg6
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --qos=express_hpcalot
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8/runs/%x-%j.out
# 6x6 regression of the generalised guide (verify_general_6x6.py, needs the 4.2 GB CSR in chatty_compress6/data).
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8; G=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_gfmc; C=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
RUN=$B/runs/reg6_${SLURM_JOB_ID}; mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/
export ST6_CSR=$C/data/csr ST6_TABLE=$C/data/psi0_6x6_table.npz GFMC_PARAMS=$G/data/params_final.npz GFMC_TABLE_A=$G/data/table_A.json GFMC_SYMA=$C/data/rpTabA_17168764/sym_tables_A.npz
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $RUN/code/*.py; } > $RUN/run_metadata.txt
cd $RUN/code
$B/../venv_jax_0.8.2/bin/python verify_general_6x6.py $RUN > $RUN/progress.txt 2>&1; rc=$?
echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; echo "GFMC_DONE exit=$rc" >> $RUN/progress.txt; exit $rc
