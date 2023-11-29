# Make sure this path ends with a /
DATA_PATH="mre_data_tsv_combined/"
TOKENIZER="zhihan1996/DNABERT-2-117M"
CACHE="mre_data_tsv_combined/cache/"

## 1. Activate a conda/mamba environment
# source myconda
# mamba activate <>

# 2. Generate the dataset.
echo $CACHE
python3 data.py \
    --pickle_dataset True \
    --file_base $DATA_PATH \
    --model_name_or_path $TOKENIZER \
    --cache_dir $CACHE \
    --model_max_length 512 \