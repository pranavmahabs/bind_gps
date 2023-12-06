#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err

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
#source myconda
#mamba activate learning

# Code to fine-tune the model.
LOCAL_RANK=$(seq 0 $((NUM_GPUS - 1))) CUDA_VISIBLE_DEVICE=$(seq 0 $((NUM_GPUS - 1))) 
#torchrun --nproc_per_node $NUM_GPUS train.py \
python3 train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --label_json $LABELJSON \
        --data_pickle $PICKLE \
        --run_name dnabert-enhancer \
        --model_max_length 512 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-4 \
        --num_train_epochs 10 \
        --save_steps 1000 \
        --output_dir $OUTPATH \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --warmup_steps 200 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --fp16 True 
