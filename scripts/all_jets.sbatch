#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=64

# Your code below this line
source ~/setup_conda.sh
srun python all_jets.py 
