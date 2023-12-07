#!/bin/bash

#SBATCH --cpus-per-task=20
#SBATCH -o classify_job.txt
#SBATCH --job-name=classify_job

pwd

module load anaconda/2023a

python classify_trainer.py