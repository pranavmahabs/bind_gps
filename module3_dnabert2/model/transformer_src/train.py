# Author: Pranav Mahableshwarkar
# Last Modified: 08-02-2021
# Description: This file contains the code for the DNABERT-Enhancer fine-tuning.

import os
import csv
import copy
import json
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import pickle
from pynvml import *
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from data_dnabert import SupervisedDataset, DataCollatorForSupervisedDataset
from dna_tokenizer import (
    DNATokenizer,
    PRETRAINED_INIT_CONFIGURATION,
    PRETRAINED_VOCAB_FILES_MAP,
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    VOCAB_KMER,
)


@dataclass
class ModelArguments:
    model_config: str = field(
        default="dna6", metadata={"help": "Choose dna3, dna4, dna5, or dna6"}
    )
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    label_json: str = field(
        default=None, metadata={"help": "Json with Label2Id config."}
    )
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "dropout rate for LoRA"}
    )
    lora_target_modules: str = field(
        default="query,value", metadata={"help": "where to perform LoRA"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    kmer: int = field(
        default=-1,
        metadata={"help": "k-mer for input sequence. Must be 3, 4, 5, or 6."},
    )
    data_pickle: str = field(
        default=None, metadata={"help": "Pickle file that contains the T/T/V split."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    tf32: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)


"""
GPU/Training Utils
"""


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupies: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/Second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


"""
Functionality to SAVE the model/trainer. 
"""


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Custom Train Function to Support Class-Weighted Training. 
"""


class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (using the global variable defined above)
        weights = self.train_dataset.getweights()
        rank = os.environ["LOCAL_RANK"]
        this_device = torch.device(int(rank))
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, device=this_device)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.

FEEL FREE TO EDIT THE METRICS HERE TO SUIT YOUR NEEDS.
"""


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    """Metrics used during validation."""
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "AUC_score_0":
        ## Expects that labels are provided in a one-hot encoded format.
        sklearn.metrics.roc_auc_score((labels == 0), logits[:, 0]),
        "AUC_score_2":
        ## Expects that labels are provided in a one-hot encoded format.
        sklearn.metrics.roc_auc_score((labels == 2), logits[:, 2]),
    }


def compute_auc_fpr_thresholds(logits, labels):
    """Metrics used during FINAL model evaluation."""
    ## class 0
    [fprs0, tprs0, thrs0] = sklearn.metrics.roc_curve((labels == 0), logits[:, 0])
    sort_ix = np.argsort(np.abs(fprs0 - 0.1))
    fpr10_0 = thrs0[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs0 - 0.05))
    fpr05_0 = thrs0[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs0 - 0.03))
    fpr03_0 = thrs0[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs0 - 0.01))
    fpr01_0 = thrs0[sort_ix[0]]

    ## class 2
    [fprs2, tprs2, thrs2] = sklearn.metrics.roc_curve((labels == 2), logits[:, 2])
    sort_ix = np.argsort(np.abs(fprs2 - 0.1))
    fpr10_2 = thrs2[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs2 - 0.05))
    fpr05_2 = thrs2[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs2 - 0.03))
    fpr03_2 = thrs2[sort_ix[0]]
    sort_ix = np.argsort(np.abs(fprs2 - 0.01))
    fpr01_2 = thrs2[sort_ix[0]]

    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "AUC_score_0":
        ## Expects that labels are provided in a one-hot encoded format.
        sklearn.metrics.roc_auc_score((labels == 0), logits[:, 0]),
        "AUC_score_2":
        ## Expects that labels are provided in a one-hot encoded format.
        sklearn.metrics.roc_auc_score((labels == 2), logits[:, 2]),
        "FPR_Thresholds_0": [fpr10_0, fpr05_0, fpr03_0, fpr01_0],
        "FPR_Thresholds_2": [fpr10_2, fpr05_2, fpr03_2, fpr01_2],
    }


"""
Compute metrics used for huggingface trainer.
"""


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)


def compute_final_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_auc_fpr_thresholds(logits, labels)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = DNATokenizer(
        vocab_file=PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_args.model_config],
        do_lower_case=PRETRAINED_INIT_CONFIGURATION[model_args.model_config][
            "do_lower_case"
        ],
        max_len=PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[model_args.model_config],
    )

    print(f"Provided Data Path: {data_args.data_path}")
    print(f"Provided Pickle File: {data_args.data_pickle}")

    # define datasets and data collator
    if data_args.data_pickle is None:
        print("Dataset not provided - manually generated from TSV files")
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=os.path.join(data_args.data_path, "train.tsv"),
            kmer=data_args.kmer,
        )
        val_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=os.path.join(data_args.data_path, "val.tsv"),
            kmer=data_args.kmer,
        )
        test_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=os.path.join(data_args.data_path, "test.tsv"),
            kmer=data_args.kmer,
        )
        to_dump = {}
        for name, dset in zip(
            ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
        ):
            to_dump[name] = dset
        with open(
            os.path.join(data_args.data_path, "supervised_dataset.p"), "wb"
        ) as handle:
            pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading the Pickled Dataset")
        with open(data_args.data_pickle, "rb") as handle:
            dataset = pickle.load(handle)
        train_dataset = dataset["train"]
        val_dataset = dataset["val"]
        test_dataset = dataset["test"]

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    with open(model_args.label_json, "r") as jfile:
        data = json.load(jfile)

    label2id = data.get("label2id", {})
    id2label = {v: k for k, v in label2id.items()}

    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
        id2label=id2label,
        label2id=label2id,
    )

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Configure Parallel Training and Trainer.
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"There are {torch.cuda.device_count()} {torch.cuda.get_device_name(0)}'s available."
    )
    print(f"Using {device_name} for training...")

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    result = trainer.train()
    print_summary(result)

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        trainer.compute_metrics = compute_final_metrics
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(
            os.path.join(training_args.output_dir, "eval_results.json"), "w"
        ) as f:
            json.dump(results, f)

    cleanup()


if __name__ == "__main__":
    train()
