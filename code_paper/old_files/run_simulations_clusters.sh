#!/bin/bash

#SBATCH --constraint=[wsm|ivb|hsw]
#SBATCH --time=0-24:00:00 --mem-per-cpu=5000M
#SBATCH --output=./job_out/simulation_%A_%a.out
#SBATCH --error=./job_err/simulation_%A_%a.err
#SBATCH --array=1-1000

ml anaconda3

cd /scratch/work/sommars1/sara_flame_pythontools/code_analysis

srun python analysis_simulations_runs.py $SLURM_ARRAY_TASK_ID



