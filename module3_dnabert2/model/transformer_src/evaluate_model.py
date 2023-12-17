import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import pickle
from pynvml import *
import transformers
import sklearn
import numpy as np
from torch.utils.data import SequentialSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange

from peft import PeftConfig, PeftModel

from train import compute_final_metrics, CustomTrainer
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
    dnabert_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "Dir that has Pretrained DNABERT."},
    )
    peft_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={"help": "Dir that has Finetuned PEFT DNABERT."},
    )
    label_json: str = field(
        default=None, metadata={"help": "Json with Label2Id config."}
    )
    out_dir: Optional[str] = field(
        default="evaluation_output",
        metadata={"help": "Where you want results to be saved."},
    )


@dataclass
class DataArguments:
    kmer: int = field(
        default=-1,
        metadata={"help": "k-mer for input sequence. Must be 3, 4, 5, or 6."},
    )
    data_pickle: str = field(
        default=None, metadata={"help": "ONLY accepts the pickle file from training."}
    )


@dataclass
class TestingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    per_device_eval_batch_size: int = field(default=1)
    seed: int = field(default=42)
    re_eval: bool = field(default=False)
    evaluation_strategy: str = field(default="steps")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def process_scores(attention_scores, kmer):
    scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])
    unnorm = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

    # attention_scores: (batch_size, num_heads, seq_len, seq_len)
    for index, attention_score in enumerate(attention_scores):
        # (1, num_heads, seq_len, seq_len)
        attn_score = []
        for i in range(1, attention_score.shape[-1] - kmer + 2):
            # sum (heads, 0, all_scores) -> 0: Beginning of Sentence Token
            attn_score.append(float(attention_score[:, 0, i].sum()))

        for i in range(len(attn_score) - 1):
            if attn_score[i + 1] == 0:
                attn_score[i] = 0
                break

        # attn_score[0] = 0
        counts = np.zeros([len(attn_score) + kmer - 1])
        real_scores = np.zeros([len(attn_score) + kmer - 1])
        for i, score in enumerate(attn_score):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score
        real_scores = real_scores / counts
        unnorm[index] = real_scores
        real_scores = real_scores / np.linalg.norm(real_scores)

        scores[index] = real_scores
    return scores, unnorm


def process_multi_score(attention_scores, kmer):
    scores = np.zeros(
        [
            attention_scores.shape[0],
            attention_scores.shape[1],
            attention_scores.shape[-1],
        ]
    )

    # attention_scores: (batch_size, num_heads, seq_len, seq_len)
    for index, attention_score in enumerate(attention_scores):
        head_scores = np.zeros([attention_scores.shape[1], attention_scores.shape[-1]])
        for head in range(0, len(attention_score)):
            attn_score = []

            for i in range(1, attention_score.shape[-1] - kmer + 2):
                attn_score.append(float(attention_score[head, 0, i]))

            for i in range(len(attn_score) - 1):
                if attn_score[i + 1] == 0:
                    attn_score[i] = 0
                    break

            counts = np.zeros([len(attn_score) + kmer - 1])
            real_scores = np.zeros([len(attn_score) + kmer - 1])

            for i, score in enumerate(attn_score):
                for j in range(kmer):
                    counts[i + j] += 1.0
                    real_scores[i + j] += score
            real_scores = real_scores / counts
            real_scores = real_scores / np.linalg.norm(real_scores)

            head_scores[head] = real_scores

        scores[index] = head_scores
    return scores


def evaluate():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TestingArguments)
    )
    model_args, data_args, test_args = parser.parse_args_into_dataclasses()

    tokenizer = DNATokenizer(
        vocab_file=PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_args.model_config],
        do_lower_case=PRETRAINED_INIT_CONFIGURATION[model_args.model_config][
            "do_lower_case"
        ],
        max_len=PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[model_args.model_config],
    )

    print(f"Provided Pickle File: {data_args.data_pickle}")

    print("Loading the Pickled Dataset")
    with open(data_args.data_pickle, "rb") as handle:
        dataset = pickle.load(handle)

    complete_dataset = dataset.get("positive", None)
    if complete_dataset is None:
        raise ValueError("No dataset found in the pickle file.")
    if test_args.re_eval:
        test_dataset = dataset.get("test", None)
        if test_dataset is None:
            raise ValueError("No test dataset found in the pickle file.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    with open(model_args.label_json, "r") as jfile:
        data = json.load(jfile)

    label2id = data.get("label2id", {})
    metadata = data.get("metadata", {})

    num_labels = metadata["num_labels"]
    id2label = {v: k for k, v in label2id.items()}

    model2 = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.dnabert_path,
        cache_dir=None,
        num_labels=num_labels,
        trust_remote_code=True,
        id2label=id2label,
        label2id=label2id,
        output_attentions=True,
    )
    inference_model = PeftModel.from_pretrained(model2, model_args.peft_path)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    inference_model = inference_model.to(device)

    if torch.cuda.device_count() > 1:
        sys.exit("Too many GPUs in use. Please configure for only 1.")

    batch_size = test_args.per_device_eval_batch_size
    pred_loader = DataLoader(
        dataset=complete_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    score_len = len(complete_dataset.input_ids[0])
    #if score_len != 496:
    #    raise ValueError("Score Length is not 496")
    single_attentions = np.zeros((len(complete_dataset), score_len))
    unnorm_attentions = np.zeros((len(complete_dataset), score_len))
    pred_results = np.zeros((len(complete_dataset), num_labels))
    multi_attentions = np.zeros((len(complete_dataset), 12, score_len))
    true_labels = np.zeros(len(complete_dataset))

    for index, batch in enumerate(tqdm(pred_loader, desc="Predicting")):
        inference_model.eval()

        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = inference_model(input_ids, masks)

            # Save Attention Scores #
            out_attns = (outputs.attentions[-1]).cpu().numpy()
            single_attn, unnormed_attn = process_scores(out_attns, data_args.kmer)
            multi_attn = process_multi_score(out_attns, data_args.kmer)
            single_attentions[
                index * batch_size : index * batch_size + len(input_ids), :
            ] = single_attn
            unnorm_attentions[
                index * batch_size : index * batch_size + len(input_ids), :
            ] = unnormed_attn
            multi_attentions[
                index * batch_size : index * batch_size + len(input_ids), :, :
            ] = multi_attn
            # Save Logits #
            out_logits = outputs.logits.cpu().detach().numpy()
            pred_results[
                index * batch_size : index * batch_size + len(input_ids), :
            ] = out_logits

            true_labels[
                index * batch_size : index * batch_size + len(input_ids)
            ] = labels

    if not os.path.exists(test_args.output_dir):
        os.makedirs(test_args.output_dir)
        print(f"Directory '{test_args.output_dir}' created. Saving results now.")
    else:
        print(f"Directory '{test_args.output_dir}' already exists. Saving results now.")

    np.save(os.path.join(test_args.output_dir, "atten.npy"), single_attentions)
    np.save(os.path.join(test_args.output_dir, "unnorm_atten.npy"), unnorm_attentions)
    np.save(os.path.join(test_args.output_dir, "pred_results.npy"), pred_results)
    np.save(os.path.join(test_args.output_dir, "heads_atten.npy"), multi_attentions)
    np.save(os.path.join(test_args.output_dir, "labels.npy"), true_labels)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if test_args.re_eval:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.dnabert_path,
            cache_dir=None,
            num_labels=num_labels,
            trust_remote_code=True,
            id2label=id2label,
            label2id=label2id
        )
        inference_model2 = PeftModel.from_pretrained(model, model_args.peft_path)
        trainer = transformers.Trainer(
            model=inference_model2,
            args=test_args,
            tokenizer=tokenizer,
            compute_metrics=compute_final_metrics,
        )
        results = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(test_args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    evaluate()
