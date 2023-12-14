# Author: Pranav Mahableshwarkar
# Last Modified: 08-02-2021
# Description: This file is a modified version of the original data file from the DNABERT repository.

import csv
import sys
import pickle
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from motif_utils import seq2kmer, kmer2seq

from dna_tokenizer import (
    DNATokenizer,
    PRETRAINED_INIT_CONFIGURATION,
    PRETRAINED_VOCAB_FILES_MAP,
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    VOCAB_KMER,
)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = 6,
    ):
        """
        args:
        - data_path: path to the data file.
        - tokenizer: tokenizer for the model.
        - kmer: k-mer for input sequence. Must be 3, 4, 5, or 6.
        """
        assert kmer in [3, 4, 5, 6], "kmer must be 3, 4, 5, or 6"
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f, delimiter="\t"))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            print("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(float(d[1])) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            print("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            print(f"Tokenizing input with {kmer}-mer as input...")
            with open(data_path, "r", newline="\n") as file:
                reader = csv.reader(file, delimiter="\t")
                texts = list(reader)

            # Drop the header - note that this will drop the first sample
            #                   if the header is not included.
            
            texts = np.asarray(texts)
            labels = texts[:, 1]
            texts = list(texts[:, 0])
            texts = [seq2kmer(seq, kmer) for seq in texts]

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        # EDIT HERE for LABEL TRANSFORMATION
        if -1 in self.labels:
            self.labels += 1
        # EDIT ABOVE for LABEL TRANSFORMATION
        self.num_labels = len(set(self.labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def getweights(self):
        """
        Returns the weights for each class in the dataset.
        """
        unique_labels = list(set(self.labels))
        counts = []
        for label in unique_labels:
            counts.append(np.count_nonzero(self.labels == label))
        top = max(counts)
        weights = [top / count for count in counts]
        return weights


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = np.asarray([int(float(i)) for i in labels])
        encoded = np.zeros((labels.size, 2 + 1))
        encoded[np.arange(labels.size), labels] = 1
        encoded = torch.Tensor(encoded)
        print(encoded.shape)
        return dict(
            input_ids=input_ids,
            labels=encoded,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def pickle_dataset(config, file_base):
    """
    Pickles the dataset for faster loading.
    """
    tokenizer = DNATokenizer(
        vocab_file=PRETRAINED_VOCAB_FILES_MAP["vocab_file"][config],
        do_lower_case=PRETRAINED_INIT_CONFIGURATION[config]["do_lower_case"],
        max_len=PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[config],
    )
    supervised_outfile = file_base + "supervised_dataset.p"
    tsv_file = file_base + "dataset.tsv"
    to_dump = {}
    dataset = SupervisedDataset(tsv_file, tokenizer)

    weights = dataset.getweights()
    train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2])
    to_dump["train"] = train
    to_dump["val"] = val
    to_dump["test"] = test

    positive_outfile = file_base + "evaluate.p"
    file = file_base + "positive.tsv"
    positive_dataset = SupervisedDataset(file, tokenizer)
    positive_dump = {"positive": positive_dataset, "test": to_dump["test"]}

    with open(supervised_outfile, "wb") as handle:
        pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(positive_outfile, "wb") as handle2:
        pickle.dump(positive_dump, handle2, protocol=pickle.HIGHEST_PROTOCOL)



def pickle_single(config, file_base, file, name):
    """
    Pickles a single evaluation dataset for faster loading.
    """
    tokenizer = DNATokenizer(
        vocab_file=PRETRAINED_VOCAB_FILES_MAP["vocab_file"][config],
        do_lower_case=PRETRAINED_INIT_CONFIGURATION[config]["do_lower_case"],
        max_len=PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[config],
    )
    positive_outfile = file_base + name + ".p"
    positive_dataset = SupervisedDataset(file, tokenizer)
    positive_dump = {"positive": positive_dataset}

    with open(positive_outfile, "wb") as handle2:
        pickle.dump(positive_dump, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="dna5", help="config")
    args.add_argument(
        "--file_base", type=str, default="data/processed/", help="file base"
    )
    args.add_argument(
        "--pickle_dataset", type=bool, default=False, help="pickle dataset"
    )
    args.add_argument(
        "--single_file", type=str, default="data/processed/train.tsv", help="data path"
    )
    args.add_argument("--single_name", type=str, default=None, help="data name")
    args = args.parse_args()

    print(args.single_file)

    if args.pickle_dataset is True:
        print("Pickling dataset...")
        pickle_dataset(args.config, args.file_base)
    elif args.single_file is not None:
        print("Pickling single file for {}...".format(args.single_name))
        pickle_single(args.config, args.file_base, args.single_file, args.single_name)
