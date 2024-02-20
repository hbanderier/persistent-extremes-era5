#!/bin/bash
#SBATCH --job-name="all_jets"
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=32

source ~/setup_conda.sh
srun python all_jets.py 
