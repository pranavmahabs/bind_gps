# Author: Pranav Mahableshwarkar
# Last Modified: 08-07-2021
# Description: This script is used to generate smaller datasets from single BED files.

#SBATCH
#SBATCH
#SBATCH
#SBATCH

# Make sure this path ends with a /
DATA_PATH="../data/"
KMER=6
CONFIG="dna6" # This must MATCH the KMER. 
HG_FASTA="../../../common/genomes/hg38/hg38.fa"
NAME="neg_sample"

# 1. Activate a conda/mamba environment
#source myconda
#mamba activate learning

# 2. First Generate the TSV files using custom_process.py
echo "Preprocessing and Creating TSV File"
python3 utils_dir/custom_preprocess.py \
    --generate-single-tsv True \
    --single-bed-file "../data/negative_fullsample.bed" \
    --single-name $NAME \
    --fast-file $HG_FASTA \
    --k $KMER \
    --results-folder $DATA_PATH \

# 3. Generate the Pickle Files that Contain the Dataset
echo "Generating pickle file"
python3 transformer_src/data_dnabert.py \
    --single_file "${DATA_PATH}${NAME}.tsv" \
    --single_name $NAME \
    --config $CONFIG \
    --file_base $DATA_PATH \
