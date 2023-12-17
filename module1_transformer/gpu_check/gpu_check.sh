#!/bin/bash
#SBATCH -J gpu_check
#SBATCH -p gpu --gres=gpu:2
#SBATCH -t 00:05:00
#SBATCH --mem=16G
#SBATCH -e gpu_check_%J.err
#SBATCH -o gpu_check_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joseph_aguilera@brown.edu

python gpu_check.py
