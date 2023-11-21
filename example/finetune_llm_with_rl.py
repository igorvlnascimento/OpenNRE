import time
import random
import torch
import opennre
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from random import choices
from datasets import Dataset
from ast import literal_eval
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

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

# LLM
parser.add_argument('--classifier', default="none", type=str,
        help='RE classifier')

# Dataset
parser.add_argument('--dataset', default="none", type=str,
        help='Dataset')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

np.random.seed(args.seed)

root_path = '.'
opennre.download(args.dataset, root_path=root_path)
dataset = load_dataset('text', data_files={"train": [
    f"benchmark/{args.dataset}/{args.dataset}_train_gpt.txt"]})
dataset = dataset['train']
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
        " ".join(["<s>"] + 
        x["text"]["token"][:x["text"]["h"]["pos"][0]] + \
            ['<SUB>'] + x["text"]["token"][x["text"]["h"]["pos"][0]:x["text"]["h"]["pos"][1]] + ['</SUB>'] + \
        x["text"]["token"][x["text"]["h"]["pos"][1]:x["text"]["t"]["pos"][0]] + \
        ['<OBJ>'] + x["text"]["token"][x["text"]["t"]["pos"][0]:x["text"]["t"]["pos"][1]] + ['</OBJ>'] + \
        x["text"]["token"][x["text"]["t"]["pos"][1]:]),
    "label": x["label"]
    }
)
labels = list(set(dataset['label']))

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def format_sentences(texts):
    sentences_formatted = []
    for text in texts:
        dict_format = {}
        tokenized_sentence = text.split()[1:]
        if len(tokenized_sentence) >= 9 and \
            '<SUB>' in tokenized_sentence and \
            '</SUB>' in tokenized_sentence and \
            '<OBJ>' in tokenized_sentence and \
            '</OBJ>' in tokenized_sentence:
            head_entity_start_index = tokenized_sentence.index('<SUB>')
            head_entity_end_index = tokenized_sentence.index('</SUB>') - 1
            tokenized_sentence.remove('<SUB>')
            tokenized_sentence.remove('</SUB>')
            tail_entity_start_index = tokenized_sentence.index('<OBJ>')
            tail_entity_end_index = tokenized_sentence.index('</OBJ>') - 1
            tokenized_sentence.remove('<OBJ>')
            tokenized_sentence.remove('</OBJ>')
            dict_format['text'] = " ".join(tokenized_sentence)
            dict_format['h'] = {'pos': [head_entity_start_index, head_entity_end_index]}
            dict_format['t'] = {'pos': [tail_entity_start_index, tail_entity_end_index]}
            sentences_formatted.append(dict_format.copy())
        else:
            sentences_formatted.append([])
    return sentences_formatted

def extract_output(model, texts):
    text_sentences_formatted = format_sentences(texts)
    logits = []
    labels = []
    for text_formatted in text_sentences_formatted:
        if text_formatted == []:
            logits.append(torch.tensor(-1.0))
            labels.append("none")
        else:
            result = model.infer(text_formatted)
            logits.append(torch.tensor(result[1]).squeeze())
            labels.append(result[0])
    return logits, labels

ctrl_str = [f"[{label}]" for label in labels]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this should be handled by accelerate

relation_classifier = opennre.get_model(args.classifier, '.')

def label_logit_to_reward(logit, task, labels):
    """
    Take the positive sentiment logit and scale it for the task.
    """
    for i in range(len(logit)):
        if labels[i] == "none":
            pass
        elif task[i] != f"[{labels[i]}]":
            logit[i] = -logit[i]
        elif task[i] == f"[{labels[i]}]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2, 3]!")
    return logits

MODELS_DIR = Path('ckpt')
MODELS_PATH = MODELS_DIR / args.dataset / args.llm
model_name = f"igorvln/dare_{args.llm}_{args.dataset}_byrelation_finetuning"
config = PPOConfig(
    model_name=model_name, steps=51200, learning_rate=1.41e-5, remove_unused_columns=False#, log_with="wandb"
)

txt_in_len = 1
txt_out_len = 101

llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
llm_model_ref = create_reference_model(llm_model)
llm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

llm_tokenizer.pad_token = llm_tokenizer.eos_token

dataset = dataset.map(lambda x: {"query": f"[{x['label']}] ", "input_ids": llm_tokenizer.encode(f"[{x['label']}] ", return_tensors="pt")}, batched=False)

dataset = Dataset.from_dict(dataset[:])
dataset.set_format("pytorch")

ppo_trainer = PPOTrainer(config, llm_model, llm_model_ref, llm_tokenizer, dataset, data_collator=collator)

ctrl_tokens = dict((s, llm_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
else:
    device = ppo_trainer.accelerator.device

generation_kwargs = {
    "min_length": 9,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": llm_tokenizer.eos_token_id,
    "max_new_tokens": txt_out_len,
    "eos_token_id": 2,
}

for epoch in range(2):
    for batch in tqdm(ppo_trainer.dataloader):
        (logs, game_data,) = (
            dict(),
            dict(),
        )

        #### prepend a random control token
        task_list = choices(ctrl_str, k=config.batch_size)
        query_tensors = [input_ids for input_ids in batch["input_ids"]]

        #### get response from LLM
        response_tensors = []
        for query in tqdm(query_tensors):
            response = llm_model.generate(query, **generation_kwargs)
            response_str = llm_tokenizer.decode(response[0])
            first_sentence = sent_tokenize(response_str)[0]
            tokenized_sentence = first_sentence.split()    
            response = llm_tokenizer.encode(" ".join(tokenized_sentence), return_tensors="pt")
            response_tensors.append(response.squeeze())
        game_data["response"] = [llm_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### sentiment analysis
        texts = [r for r in game_data["response"]]
        logits, labels = extract_output(relation_classifier, texts)
        rewards = label_logit_to_reward(logits, task_list, labels)
        query_tensors = [query.squeeze() for query in query_tensors]

        #### Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        for cs in ctrl_str:
            key = "env/reward_" + cs.strip("[]")
            stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
        ppo_trainer.log_stats(stats, game_data, rewards)

for ctrl_s in ctrl_str:
    plt.hist(
        [r for r, t in zip(logs["env/reward_dist"], task_list) if t == ctrl_s], density=True, alpha=0.5, label=ctrl_s
    )
    plt.legend(loc="best")
    plt.title("reward distribution")
    plt.grid(True)
    plt.show()

MODEL_NAME = f"dare_{args.llm}_{args.dataset}_byrelation_finetuning_with_rl"
MODELS_PATH_NAME = MODELS_PATH / MODEL_NAME
MODELS_PATH_NAME.mkdir(parents=True, exist_ok=True)

llm_model.save_pretrained(MODELS_PATH_NAME)
llm_tokenizer.save_pretrained(MODELS_PATH_NAME)
