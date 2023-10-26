# coding:utf-8
import torch
import numpy as np
import argparse
import logging
import random
from trl import SFTTrainer
from datasets import load_dataset
from ast import literal_eval


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
parser.add_argument('--llm', default="gpt2", type=str,
        help='LLM')

# Dataset
parser.add_argument('--dataset', default="ddi", type=str,
        help='Dataset')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

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
        " ".join(
        x["text"]["token"][:x["text"]["h"]["position"][0]] + ["druga"] + \
        x["text"]["token"][x["text"]["h"]["position"][1]:x["text"]["t"]["position"][0]] + \
        ["drugb"] + x["text"]["token"][x["text"]["t"]["position"][1]:]),
    "label": x["label"]
    }
)

trainer = SFTTrainer(
    args.llm,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=128,
    num_train_epochs=5
)

trainer.train()