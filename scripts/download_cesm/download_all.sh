#!/bin/bash

sbatch download_cesm_past.sbatch high_wind
sbatch download_cesm_past.sbatch mid_wind
sbatch download_cesm_future.sbatch high_wind
sbatch download_cesm_future.sbatch mid_wind
