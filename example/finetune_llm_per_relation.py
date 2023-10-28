# coding:utf-8
import torch
import numpy as np
import argparse
import logging
import random
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from ast import literal_eval
from pathlib import Path
from datetime import datetime
import opennre
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

# LLM
parser.add_argument('--llm', default="igorvln/dare_gpt2_ddi", type=str,
        help='LLM')

# Dataset
parser.add_argument('--dataset', default="ddi", type=str,
        help='Dataset')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

root_path = '.'
if args.dataset != 'none':
    try:
        opennre.download(args.dataset, root_path=root_path)
    except:
        raise Exception('--train_file are not specified or files do not exist. Or specify --dataset')
else:
    raise Exception('--train_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

dataset = load_dataset('text', data_files={"train": [
    f"benchmark/{args.dataset}/{args.dataset}_train_gpt.txt"]})
dataset = dataset.map(lambda x: {
    "text": 
            literal_eval(x["text"]), 
    "label": 
        literal_eval(x["text"])["relation"]
    }
)
## Preprocess dataset to mask entities with special tokens
dataset = dataset.map(lambda x: {
    "text": 
        " ".join([f"[{x['label']}]"] +
        x["text"]["token"][:x["text"]["h"]["position"][0]] + ["drug_a"] + \
        x["text"]["token"][x["text"]["h"]["position"][1]:x["text"]["t"]["position"][0]] + \
        ["drug_b"] + x["text"]["token"][x["text"]["t"]["position"][1]:]),
    "label": x["label"]
    }
)

labels = list(set(dataset['train']['label']))

LLM_MODEL = Path('ckpt') / args.dataset / args.llm

for label in labels:
    print(f"Finetuning GPT2 for label: {label}")

    dataset_label = dataset.filter(lambda x: x["label"] == label)
    model = AutoModelForCausalLM.from_pretrained(args.llm)

    MODELS_DIR = Path('ckpt')
    MODELS_PATH = MODELS_DIR / args.dataset / label
    MODEL_NAME = f"dare_{args.llm}_{args.dataset}_{label}_finetuning"
    MODELS_PATH_NAME = MODELS_PATH / MODEL_NAME
    MODELS_PATH_NAME.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=MODELS_PATH,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset_label["train"],
        dataset_text_field="text",
        max_seq_length=128,
        args=train_args,
    )

    trainer.train()

    trainer.save_model(MODELS_PATH_NAME)