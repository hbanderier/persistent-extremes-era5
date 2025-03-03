#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=10

# Your code below this line
source ~/.bashrc
conda activate env12
srun python download_ERA5_dailymeans.py
