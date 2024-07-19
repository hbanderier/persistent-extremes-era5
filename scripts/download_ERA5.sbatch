#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=10

# Your code below this line
mamba activate env11_2
srun python download_ERA5.py
