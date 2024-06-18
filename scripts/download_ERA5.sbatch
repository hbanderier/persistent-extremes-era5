#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=10

# Your code below this line
mamba activate env11
srun python download_ERA5_mars.py
