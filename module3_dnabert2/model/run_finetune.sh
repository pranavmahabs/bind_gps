#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err

#SBATCH --mem=100G
#SBATCH -t 12:00:00

# Author: Pranav Mahableshwarkar
# Last Modified: 08-02-2021
# Description: Script to fine-tune the model on the balanced dataset.

LABELJSON="labels.json"
MODEL_PATH="pretrained_6mer/"

# Edit these file-paths to point to the correct data and output directories.
DATA_PATH="../data/three-class/"
OUTPATH="../output/three-class/"
PICKLE="../data/three-class/supervised_dataset.p"
NUM_GPUS=2

# Code to activate conda environment - this can either be done through conda or mamba. 
# source myconda
# mamba activate learning

module load python/3.11.0
module load openssl/3.0.0

# Code to fine-tune the model.
#LOCAL_RANK=$(seq 0 $((NUM_GPUS - 1))) CUDA_VISIBLE_DEVICE=$(seq 0 $((NUM_GPUS - 1))) \
#torchrun --nproc_per_node $NUM_GPUS transformer_src/train.py \
python3 transformer_src/train.py \
        --model_config "dna6" \
        --model_name_or_path $MODEL_PATH \
        --data_path  $DATA_PATH \
        --kmer 6 \
        --data_pickle $PICKLE \
        --label_json $LABELJSON \
        --run_name dnabert-enhancer \
        --model_max_length 512 \
        --use_lora \
        --lora_target_modules 'query,value,key,dense' \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-4 \
        --num_train_epochs 6 \
        --fp16 True \
        --save_steps 500 \
        --output_dir $OUTPATH \
        --evaluation_strategy steps \
        --eval_steps 500 \
        --warmup_steps 200 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \

