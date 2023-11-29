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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
    ):
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        logging.warning("Creating single sequence classification dataset...")
        texts = [(d[0].split())[1] for d in data]
        labels = [int((d[0].split())[2]) for d in data]

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
        self.num_labels = len(set(labels))

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
        labels = np.asarray(self.labels)
        for label in unique_labels:
            counts.append(np.count_nonzero(labels == label))
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
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def pickle_dataset(file_base, tokenizer):
    """
    Pickles the dataset for faster loading.
    """
    supervised_outfile = file_base + "supervised_dataset.p"
    tsv_file = file_base + "dataset.tsv"
    to_dump = {}
    dataset = SupervisedDataset(tsv_file, tokenizer)

    weights = dataset.getweights()
    train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.1, 0.2])
    to_dump["train"] = train
    to_dump["val"] = val
    to_dump["test"] = test
    to_dump["weights"] = weights

    positive_outfile = file_base + "evaluate.p"
    file = file_base + "positive.tsv"
    positive_dataset = SupervisedDataset(file, tokenizer)
    positive_dump = {"positive": positive_dataset, "test": to_dump["test"]}

    with open(supervised_outfile, "wb") as handle:
        pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(positive_outfile, "wb") as handle2:
        pickle.dump(positive_dump, handle2, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_single(file_base, file, name, tokenizer):
    """
    Pickles a single evaluation dataset for faster loading.
    """
    positive_outfile = file_base + name + ".p"
    positive_dataset = SupervisedDataset(file, tokenizer)
    positive_dump = {"positive": positive_dataset}

    with open(positive_outfile, "wb") as handle2:
        pickle.dump(positive_dump, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
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
    args.add_argument("--model_name_or_path", type=str, help="data path")
    args.add_argument("--cache_dir", type=str, default="cache/", help="data path")
    args.add_argument("--model_max_length", type=int, default=None, help="max seq len")
    args = args.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if args.pickle_dataset is True:
        print("Pickling dataset...")
        pickle_dataset(args.file_base, tokenizer)
    elif args.single_file is not None:
        print("Pickling single file for {}...".format(args.single_name))
        pickle_single(args.file_base, args.single_file, args.single_name, tokenizer)
