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
from nltk.tokenize import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

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
parser.add_argument('--classifier', default="ddi_bert-base-uncased_entity", type=str,
        help='RE classifier')

# Dataset
parser.add_argument('--dataset', default="ddi", type=str,
        help='Dataset')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

np.random.seed(args.seed)

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
        " ".join(
        x["text"]["token"][:x["text"]["h"]["pos"][0]] + ["drug_a"] + \
        x["text"]["token"][x["text"]["h"]["pos"][1]:x["text"]["t"]["pos"][0]] + \
        ["drug_b"] + x["text"]["token"][x["text"]["t"]["pos"][1]:]),
    "label": x["label"]
    }
)
labels = list(set(dataset['label']))

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def extract_output(model, texts, label):
    text_sentences_formatted = []
    for text in texts:
        dict_format = {}
        tokenized_sentence = word_tokenize(text)
        head_entity_index = tokenized_sentence.index('drug_a')
        tail_entity_index = tokenized_sentence.index('drug_b')
        dict_format['text'] = text
        dict_format['h'] = {'pos': [head_entity_index, head_entity_index+1]}
        dict_format['t'] = {'pos': [tail_entity_index, tail_entity_index+1]}
        text_sentences_formatted.append(dict_format.copy())
    logits = []
    for text_formatted in text_sentences_formatted:
        result = model.infer(text_formatted)
        if result[0] == label:
            logits.append(torch.tensor(result[1]))
    return logits

ctrl_str = [f"[{label}]" for label in labels]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this should be handled by accelerate

relation_classifier = opennre.get_model(args.classifier, '.')

def label_logit_to_reward(logit, task, label):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] != label:
            logit[i] = -logit[i]
        elif task[i] == label:
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2, 3]!")
    return logit

for label in labels:
    print(f"Training for label: {label}")
    config = PPOConfig(
        model_name=f"igorvln/dare_{args.llm}_{args.dataset}_train_{label}_finetuning", steps=51200, learning_rate=1.41e-5, remove_unused_columns=False#, log_with="wandb"
    )

    txt_in_len = 1
    txt_out_len = 101

    llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    llm_model_ref = create_reference_model(llm_model)
    llm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    dataset = dataset.map(lambda x: {"query": " <s>", "input_ids": llm_tokenizer.encode(" <s>", return_tensors="pt")}, batched=False)

    dataset = Dataset.from_dict(dataset[:])
    dataset.set_format("pytorch")

    ppo_trainer = PPOTrainer(config, llm_model, llm_model_ref, llm_tokenizer, dataset, data_collator=collator)

    ctrl_tokens = dict((s, llm_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)
    print(ctrl_tokens)

    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = ppo_trainer.accelerator.device

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": llm_tokenizer.eos_token_id,
        "max_new_tokens": txt_out_len,
        "eos_token_id": -1,
    }

    for epoch in range(2):
        for batch in tqdm(ppo_trainer.dataloader):
            (logs, game_data,) = (
                dict(),
                dict(),
            )

            #### prepend a random control token
            task_list = choices(ctrl_str, k=config.batch_size)
            game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
            query_tensors = [torch.cat((ctrl_tokens[t], input_ids.squeeze())) for t, input_ids in zip(task_list, batch["input_ids"])]

            #### get response from LLM
            response_tensors = []
            for query in query_tensors:
                while True:
                    response = ppo_trainer.generate(query, **generation_kwargs)
                    response_str = llm_tokenizer.decode(response[0])
                    first_sentence = sent_tokenize(response_str)[0]
                    tokenized_sentence = word_tokenize(first_sentence)[6:]
                    if len(tokenized_sentence) >= 8 and 'drug_a' in tokenized_sentence and 'drug_b' in tokenized_sentence:
                        response = llm_tokenizer.encode(" ".join(tokenized_sentence), return_tensors="pt")
                        break
                response_tensors.append(response.squeeze())
            game_data["response"] = [llm_tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### sentiment analysis
            texts = [q + r for q, r in zip(batch["query"], game_data["response"])]
            logits = extract_output(relation_classifier, texts, label)
            rewards = label_logit_to_reward(logits, task_list, label)

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

    llm_model.save_pretrained(f"dare_{args.llm}_{args.dataset}_train_{label}_finetuning_rl")
    llm_tokenizer.save_pretrained(f"dare_{args.llm}_{args.dataset}_train_{label}_finetuning_rl")
