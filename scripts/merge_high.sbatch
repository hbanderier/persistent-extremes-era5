#!/bin/bash
#SBATCH --job-name="Postprocess"
#SBATCH --time=00:50:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8

srun bash merge_high.sh
