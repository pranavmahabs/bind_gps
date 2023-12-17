#!/bin/bash
#SBATCH -J seq_transformer
#SBATCH -n 64
#SBATCH -t 48:00:00
#SBATCH --mem=128G
#SBATCH -o seq_transformer_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joseph_aguilera@brown.edu
#SBATCH --account=ccmb-condo

python main_transformer.py
