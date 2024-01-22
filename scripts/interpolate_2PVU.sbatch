#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Your code below this line
source ~/setup_conda.sh
srun python interpolate_2PVU.py 
