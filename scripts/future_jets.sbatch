#!/bin/bash
#SBATCH --job-name="future_jets"
#SBATCH --time=20:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=49G
#SBATCH --cpus-per-task=20

source ~/.bashrc
conda activate env12
srun python future_jets.py 
