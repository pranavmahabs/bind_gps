#!/bin/bash
#SBATCH -p gpu --gres=gpu:2
#SBATCH -J module_3_generate_data
#SBATCH -e module_3_generate_data.err
#SBATCH -o module_3_generate_data.out
#SBATCH --mem=50GB
#SBATCH -t 24:00:00

# Make sure this path ends with a /
DATA_PATH="mre_data_tsv_combined/"
TOKENIZER="zhihan1996/DNABERT-2-117M"
CACHE="mre_data_tsv_combined/cache/"

## 1. Activate a conda/mamba environment
module load anaconda/2022.05
source activate
conda activate /oscar/data/larschan/tpham43/bind_gps/module3_dnabert2/conda 
# mamba activate <>

# 2. Generate the dataset.
echo $CACHE
python3 data.py \
    --pickle_dataset True \
    --file_base $DATA_PATH \
    --model_name_or_path $TOKENIZER \
    --cache_dir $CACHE \
    --model_max_length 512 \