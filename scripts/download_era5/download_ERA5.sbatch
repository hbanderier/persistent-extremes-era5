#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4

# Your code below this line
source ~/.bashrc
conda activate env12
srun python download_ERA5_mars.py
