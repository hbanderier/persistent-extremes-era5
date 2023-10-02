#!/bin/bash
#SBATCH --job-name="Download ERA5"
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=1G

# Your code below this line
source /storage/homefs/hb22g102/setup_conda.sh
srun python ../download_ERA5_.py
