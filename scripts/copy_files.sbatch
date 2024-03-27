#!/bin/bash
#SBATCH --job-name="copy_files"
#SBATCH --time=02:00:00
#SBATCH --partition=epyc2
#SBATCH --qos=job_epyc2_short
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1

source ~/setup_conda.sh
srun python copy_files.py 
