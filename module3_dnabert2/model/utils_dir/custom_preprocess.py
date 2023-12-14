# Author: Pranav Mahableshwarkar
# Last Modified: 08-02-2021
# Description: Generates the TSV files that are inputs for the DNABERT model.
# Citation: DNABERT

import pandas as pd
import numpy as np
import sys
import os
from Bio import SeqIO
from pybedtools import BedTool
from module3_dnabert2.model.transformer_src.motif_utils import seq2kmer

import argparse
from dataclasses import dataclass


@dataclass
class ProcessInput:
    pos_bed: str
    neg_bed: str
    fasta: str
    k: int
    res_dir: str


@dataclass
class SingleInput:
    bed: str
    fasta: str
    k: int
    res_dir: str


train_chromosomes = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
]
valid_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]


def get_chrom2seq(FASTA_FILE, capitalize=True):
    """
    Load in the genome fasta file to extract sequences from the BED Files.
    """
    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = (
            seq.seq.upper() if capitalize else seq.seq
        )
    return chrom2seq


def generate_tsv(sequences, labels, tsv_filename):
    if len(sequences) != len(labels):
        print(len(sequences), len(labels))
        print("Error. Sequences and labels are not of the same length.")

    tsv_content = "sequence\tlabel\n"
    for sequence, label in zip(sequences, labels):
        tsv_content += f"{sequence}\t{label}\n"

    with open(tsv_filename, "w") as tsv_file:
        tsv_file.write(tsv_content)


def get_positive_labels(bed_interval):
    """
    Return the positive labels for the binding sites.
    """
    # Default Label for Noise
    if len(bed_interval.fields) == 3:
        return 0
    # Determine Positive Label Based on the BED File
    # 1 for AE and -1 for PE
    _k = int(bed_interval[3]) - int(bed_interval[4])
    if _k == 0:
        _k = 1
    return _k


def create_dataset(param: ProcessInput, custom_label_function=get_positive_labels):
    """
    Generate the kmer-ized dataset in preparation of model training.
    Input: BED files of positive and negative binding sites, FASTA file of the genome, and kmer length.
    Output: TSV file of sequences and labels.
    """
    pos_bed_file, neg_bed_file, fasta, K, results_dir = (
        param.pos_bed,
        param.neg_bed,
        param.fasta,
        param.k,
        param.res_dir,
    )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load in the genome prior to building the dataset.
    chrom2seq = get_chrom2seq(fasta)

    print("Generating the CSV Dataset File")
    pos_beds = list(BedTool(pos_bed_file))
    neg_beds = list(BedTool(neg_bed_file))

    # Add all binding sites in BED files to train/test/val based on chromosome.
    pos_train_bed = [r for r in pos_beds if r.chrom in train_chromosomes]
    pos_val_bed = [r for r in pos_beds if r.chrom in valid_chromosomes]
    pos_test_bed = [r for r in pos_beds if r.chrom in test_chromosomes]

    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    pos_train_label = []
    pos_val_label = []
    pos_test_label = []

    for bed_list, data_list, label_list in zip(
        [pos_train_bed, pos_val_bed, pos_test_bed],
        [pos_train_data, pos_val_data, pos_test_data],
        [pos_train_label, pos_val_label, pos_test_label],
    ):
        for r in bed_list:
            # Maximum sequence length of 512
            _seq = chrom2seq[r.chrom][r.start : r.stop]
            # if (r.stop - r.start) > 512:
            #     sys.exit("Sequence length greater than maximum of 512 detected.")
            # _seq = chrom2seq[r.chrom][r.start : r.stop]
            kmerized = seq2kmer(str(_seq), K)
            data_list.append(kmerized)
            # Insert the Label into the Approriate List
            _k = custom_label_function(r)
            label_list.append(_k)

    print(len(pos_train_data))
    print(len(pos_val_data))
    print(len(pos_test_data))

    print("Generating the negative dataset...")

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in valid_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []

    for bed_list, data_list in zip(
        [neg_train_bed, neg_val_bed, neg_test_bed],
        [neg_train_data, neg_val_data, neg_test_data],
    ):
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start : r.stop]
            kmerized = seq2kmer(str(_seq), 6)
            data_list.append(kmerized)

    neg_train_label = [0 for i in range(len(neg_train_data))]
    neg_val_label = [0 for i in range(len(neg_val_data))]
    neg_test_label = [0 for i in range(len(neg_test_data))]

    print(len(neg_train_data))
    print(len(neg_val_data))
    print(len(neg_test_data))

    # Concatenate the positive and negative datasets in preparation for final return
    train_data = pos_train_data + neg_train_data
    # print(pos_train_data.shape, neg_train_data.shape, train_data.shape)
    train_label = pos_train_label + neg_train_label
    val_data = pos_val_data + neg_val_data
    val_label = pos_val_label + neg_val_label
    test_data = pos_test_data + neg_test_data
    test_label = pos_test_label + neg_test_label

    for data, label, filename in zip(
        [train_data, val_data, test_data],
        [train_label, val_label, test_label],
        ["train.tsv", "val.tsv", "test.tsv"],
    ):
        generate_tsv(data, label, results_dir + filename)


def create_single_tsv(param: SingleInput, name, label_function=get_positive_labels):
    bed_file, fasta, K, results_dir = (
        param.bed,
        param.fasta,
        param.k,
        param.res_dir,
    )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load in the genome prior to building the dataset.
    chrom2seq = get_chrom2seq(fasta)

    beds = list(BedTool(bed_file))
    data_list = []
    label_list = []

    for r in beds:
        # Maximum sequence length of 512
        _seq = chrom2seq[r.chrom][r.start : r.stop]
        kmerized = seq2kmer(str(_seq), K)
        data_list.append(kmerized)
        # Insert the Label into the Approriate List
        # 1:AE - Enhancers with H3K27AC; -1:PE, Enhancers only with H3K4me1
        _k = label_function(r)
        label_list.append(_k)

    generate_tsv(data_list, label_list, results_dir + name + ".tsv")


def custom_label_function(r):
    ##
    # Edit this function to change the label function. Otherwise it will default to
    # the get_positive_labels function which is the default for the other functions.
    ##
    # Default Label for Noise is 0 and Enhancer is 1.
    if len(r.fields) == 3:
        return 0
    else:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input files.")
    parser.add_argument(
        "--negative-file", required=False, help="Path to the negative file"
    )
    parser.add_argument(
        "--positive-file", required=False, help="Path to the positive file"
    )
    parser.add_argument("--fast-file", required=True, help="Path to the fast file")
    parser.add_argument("--k", type=int, required=True, help="Value for K")
    parser.add_argument(
        "--results-folder", required=True, help="Path to the results folder"
    )
    parser.add_argument(
        "--generate-single-tsv",
        required=False,
        help="If you only want to convert ONE BED file to a TSV.",
    )
    parser.add_argument("--single-bed-file", required=False)
    parser.add_argument("--single-name", required=False, default="positive")

    args = parser.parse_args()

    if args.single_bed_file:
        print("Generating a single TSV file for {}...".format(args.single_bed_file))
        param = SingleInput(
            args.single_bed_file, args.fast_file, args.k, args.results_folder
        )
        create_single_tsv(param, args.single_name, label_function=custom_label_function)

    if args.negative_file and args.positive_file:
        print("Generating the dataset...")
        param = ProcessInput(
            args.positive_file,
            args.negative_file,
            args.fast_file,
            args.k,
            args.results_folder,
        )
        create_dataset(param, custom_label_function=custom_label_function)
