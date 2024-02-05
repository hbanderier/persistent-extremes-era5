#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=64

# Your code below this line
source ~/setup_conda.sh
srun python all_jets.py 
