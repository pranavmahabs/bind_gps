LABELJSON="labels.json"
PEFT_PATH="output/three-class/saves/initial_best_1206"
PICKLE="mre_data_tsv_combined/evaluate.p"
OUTPATH="output/three_class/val_results"

# Command to be executed with the --normal flag
    # Add your normal command here
# source myconda
# mamba activate learning
python3 eval.py \
        --dnabert_path zhihan1996/DNABERT-2-117M \
        --peft_path $PEFT_PATH \
        --label_json $LABELJSON \
        --data_pickle $PICKLE \
        --run_name dnabert-enhancer \
        --model_max_length 512 \
        --per_device_eval_batch_size 16 \
        --evaluation_strategy steps \
        --output_dir $OUTPATH \
        --overwrite_output_dir True \
        --log_level info \
        --re_eval True \

# If you are using a pickle file that contains a test dataset,
# then make sure to include the --re_eval True setting.
