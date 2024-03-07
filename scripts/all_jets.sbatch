#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=05:55:00
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2_short
#SBATCH --mem-per-cpu=15G
#SBATCH --cpus-per-task=50

source ~/setup_conda.sh
srun python all_jets.py 
