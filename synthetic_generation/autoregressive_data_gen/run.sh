#!/bin/bash

#SBATCH --cpus-per-task=20
#SBATCH -o data_gen.txt
#SBATCH --job-name=data_gen

pwd

module load anaconda/2023a

python autoreg_svg.py

