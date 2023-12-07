#!/bin/bash

#SBATCH --cpus-per-task=20
#SBATCH -o pdf_compile.txt
#SBATCH --job-name=compile_latex2pdf

pwd

module load anaconda/2023a

python synth.py

python tex2pdf.py

python pdf2img.py


