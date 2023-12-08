#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err
#SBATCH --mail-user=pranav_mahableshwarkar@brown.edu

#SBATCH --mem=80G
#SBATCH -t 12:00:00

# Author: Pranav Mahableshwarkar
# Last Modified: 08-02-2021
# Description: Script to fine-tune the model on the balanced dataset.

LABELJSON="labels.json"
MODEL_PATH="zhihan1996/DNABERT-2-117M"

# Edit these file-paths to point to the correct data and output directories.
DATA_PATH="zhihan1996/DNABERT-2-117M"
OUTPATH="output/three-class/"
PICKLE="mre_data_tsv_combined/supervised_dataset.p"
NUM_GPUS=1

# Code to activate conda environment - this can either be done through conda or mamba.
# source myconda
# mamba activate learning

# Code to fine-tune the model.
values=(42 2023 786 555 9000)
for SEED in "${values[@]}"; do
    python3 train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --label_json "$LABELJSON" \
        --data_pickle "$PICKLE" \
        --run_name clamp_xvsa \
        --model_max_length 512 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-4 \
        --num_train_epochs 10 \
        --save_steps 1000 \
        --output_dir "$OUTPATH/clamp_xva_${SEED}" \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --warmup_steps 200 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --seed "$SEED" \
        --use_lora \
        --lora_target_modules 'query,value,key,dense' \
        --fp16 True
done