#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=1G

# Your code below this line
srun python ../download_ERA5.py
