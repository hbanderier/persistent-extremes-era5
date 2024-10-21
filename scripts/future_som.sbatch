#!/bin/bash
#SBATCH --job-name="future_som"
#SBATCH --time=4:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=20

mamba activate env11_2
srun python future_som.py 
