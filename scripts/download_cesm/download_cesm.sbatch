#!/bin/bash
#SBATCH --job-name="download_cesm"
#SBATCH --time=90:00:00
#SBATCH --partition="epyc2"
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=20

source ~/.bashrc
conda activate env12
srun python download_cesm.py $1 $2 20
