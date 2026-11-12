#!/bin/bash
#SBATCH --job-name=pfn-build6
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# build + check the 11 exact-6x6 reference guides (build_check6.py): ~1 h expected (5 exact solves ~5 min each at 8 threads as q0r.py; builds; E_var passes)
# against 4 h.  Memory: q0r.py MaxRSS 18.1 GB for build + solve (job 17662380) -> 64G.  usage: sbatch --output RUN/build6.log build6.sbatch RUN
set -u
RUN=$1; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
echo "host $(hostname) job $SLURM_JOB_ID $(date -Is)"; sha256sum $C/*.py
cat /scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables/nb_J1.npy /scratch/a/A.Otaifi/ansatz_runs/c2_t0b_6x6/tables/nb_J2.npy > /dev/null
/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python $C/build_check6.py $RUN/build_check6.json; rc=$?
echo "build_check6 exit $rc $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P; exit $rc
