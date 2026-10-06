#!/bin/bash
#SBATCH --job-name=badstart20
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
# usage: sbatch --export=ALL,METHOD=anderson,MAXITER=250,STARTS=a,b,c bad_start_20site.sbatch
cd $HOME/bad_start_tmp/j1j2/experiments/bad_start
python3 run_bad_start.py --lattice 20 --method $METHOD --starts "$STARTS" --maxiter $MAXITER --workers 7 \
  --cache $HOME/bad_start_tmp/cache20 --out $HOME/bad_start_tmp/out/bad_start_20_${METHOD}.json
