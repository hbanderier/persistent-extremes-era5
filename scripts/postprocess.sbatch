#!/bin/bash
#SBATCH --job-name="Postprocess"
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=8

# Your code below this line
srun bash postprocess.sh $1 $2
