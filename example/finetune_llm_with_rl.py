import time
import random
import torch
import opennre
import argparse
import numpy as np

from tqdm import tqdm
from random import choices
from datasets import Dataset
from ast import literal_eval
from pathlib import Path
#from nltk.tokenize import sent_tokenize, word_tokenize

#import matplotlib.pyplot as plts

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer, util

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
        " ".join(
        x["text"]["token"][:x["text"]["h"]["pos"][0]] + \
            ['<SUB>'] + x["text"]["token"][x["text"]["h"]["pos"][0]:x["text"]["h"]["pos"][1]] + ['</SUB>'] + \
        x["text"]["token"][x["text"]["h"]["pos"][1]:x["text"]["t"]["pos"][0]] + \
        ['<OBJ>'] + x["text"]["token"][x["text"]["t"]["pos"][0]:x["text"]["t"]["pos"][1]] + ['</OBJ>'] + \
        x["text"]["token"][x["text"]["t"]["pos"][1]:]),
    "label": x["label"]
    }
)
labels = list(set(dataset['label']))

model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings_by_relation = {f'{relation}': 
                          model_sim.encode(
                              [dict_["text"] for dict_ in dataset
                              .filter(
                                  lambda x : x['label'] == relation,
                                  batched=False
                                )
                              .map(
                                  lambda x : {"text": x["text"]},
                                  batched=False
                                )]
                            ) for relation in labels
                        }

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def validate_sentence(tokenized_sentence):
    indexes = [-1, -1, -1, -1]
    for i, token in enumerate(tokenized_sentence):
        if '<SUB>' in token and indexes[0] == -1:
            indexes[0] = i
        if '</SUB>' in token and indexes[1] == -1:
            indexes[1] = i
        if '<OBJ>' in token and indexes[2] == -1:
            indexes[2] = i
        if '</OBJ>' in token and indexes[3] == -1:
            indexes[3] = i
    sorted_indexes = indexes[:]
    sorted_indexes.sort()
    return indexes.count(-1) <= 1 and sorted_indexes == indexes, indexes

def format_sentences(texts, relations):
    sentences_formatted = []
    for i, text in enumerate(texts):
        dict_format = {}
        tokenized_sentence = text.split()
        valid, indexes = validate_sentence(tokenized_sentence)
        if valid:
            if indexes[0] > -1:
                head_entity_start_index = indexes[0]
            else:
                head_entity_start_index = indexes[1] - 1

            if indexes[1] > -1 and indexes[0] > -1:
                head_entity_end_index = indexes[1] - 1
            elif indexes[0] == -1:
                head_entity_end_index = indexes[1]
            else:
                head_entity_end_index = indexes[0] + 1

            if indexes[2] > -1 and indexes[1] > -1 and indexes[0] > -1:
                tail_entity_start_index = indexes[2] - 2
            elif indexes[2] == -1:
                tail_entity_start_index = indexes[3] - 3
            elif indexes[-1] == -1 or indexes[0] == -1:
                tail_entity_start_index = indexes[2] - 1

            if indexes[3] > -1 and indexes[2] > -1 and indexes[1] > -1 and indexes[0] > -1:
                tail_entity_end_index = indexes[3] - 3
            elif indexes[3] == -1:
                tail_entity_end_index = indexes[2] + 1
            elif indexes[0] == -1 or indexes[1] == -1 or indexes[2] == -1:
                tail_entity_end_index = indexes[3] - 2

            for i in range(len(indexes) - 1, 0, -1):
                del tokenized_sentence[indexes[i]]
            dict_format['text'] = " ".join(tokenized_sentence)
            dict_format['h'] = {'pos': [head_entity_start_index, head_entity_end_index]}
            dict_format['t'] = {'pos': [tail_entity_start_index, tail_entity_end_index]}
            sentences_formatted.append(dict_format.copy())
        else:
            print(text)
            #text_wo_relation = text[text.index(']')+2:]
            #relation = text[:text.index(']')+1].strip('[]')
            embedding = model_sim.encode(text)
            mean_sim = torch.mean(util.pytorch_cos_sim(embedding, embeddings_by_relation[relations[i].strip("[]")]))
            no_special_tokens_count = indexes.count(-1)
            sorted_indexes = indexes[:].sort()
            is_not_sorted_indexes = not sorted_indexes == indexes
            reward = no_special_tokens_count * .2 + is_not_sorted_indexes * .2 + mean_sim
            sentences_formatted.append(reward)
    return sentences_formatted

def extract_output(model, texts, relations):
    text_sentences_formatted = format_sentences(texts, relations)
    logits = []
    labels = []
    for i, text_formatted in enumerate(text_sentences_formatted):
        if type(text_formatted) == torch.Tensor:
            logits.append(torch.tensor(text_formatted))
            labels.append("none")
        else:
            print(text_formatted)
            text = text_formatted["text"]
            relation = relations[i].strip("[]")#text[:text.index(']')+1].strip('[]')

            embedding = model_sim.encode(text)
            mean_sim = torch.mean(util.pytorch_cos_sim(embedding, embeddings_by_relation[relation]))
            result = model.infer(text_formatted)
            logits.append(torch.tensor(result[1]).squeeze() + mean_sim)
            labels.append(result[0])
    return logits, labels

ctrl_str = [f"[{label}]" for label in labels]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this should be handled by accelerate

relation_classifier = opennre.get_model(args.classifier, '.')

def label_logit_to_reward(logits, task, labels):
    """
    Take the logit and scale it for the task.
    """
    for i in range(len(logits)):
        if labels[i] == "none":
            logits[i] = -logits[i]
        elif task[i] != f"[{labels[i]}]":
            logits[i] = 1 - logits[i]
        elif task[i] == f"[{labels[i]}]":
            logits[i] = logits[i] + 1
        else:
            raise ValueError("task has to be in [0, 1, 2, 3]!")
    return logits

MODELS_DIR = Path('ckpt')
MODELS_PATH = MODELS_DIR / args.dataset / args.llm
model_name = f"igorvln/dare_{args.llm}_{args.dataset}_byrelation_finetuning"
config = PPOConfig(
    model_name=model_name, steps=1000000, learning_rate=1.41e-6, remove_unused_columns=False, log_with="wandb"
)

txt_in_len = 1
txt_out_len = 101

llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
llm_model_ref = create_reference_model(llm_model)
llm_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

llm_tokenizer.pad_token = llm_tokenizer.eos_token

dataset = dataset.map(lambda x: {"query": " ", "input_ids": llm_tokenizer.encode(" ", return_tensors="pt")}, batched=False)

dataset = Dataset.from_dict(dataset[:])
dataset.set_format("pytorch")

ppo_trainer = PPOTrainer(config, llm_model, llm_model_ref, llm_tokenizer, dataset, data_collator=collator)

ctrl_tokens = dict((s, llm_tokenizer.encode(s, return_tensors="pt").to(device)) for s in ctrl_str)

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
        game_data["query"] = [t for t in task_list]
        query_tensors = [ctrl_tokens[query] for query in game_data["query"]]

        #### get response from LLM
        response_tensors = []
        for query in tqdm(query_tensors):
            #while True:
            response = llm_model.generate(query.to(device), **generation_kwargs)
            response_str = llm_tokenizer.decode(response[0])
            print(response_str)
                # tokenized_sentence = response_str.split()
                # valid, _ = validate_sentence(tokenized_sentence)
                # print(valid)
                # if valid:
                #     response_tensors.append(response.squeeze()[-txt_out_len:])
                #     break
            # tokenized_sentence = first_sentence.split()    
            # response = llm_tokenizer.encode(" ".join(tokenized_sentence), return_tensors="pt")
            response_tensors.append(response.squeeze()[-txt_out_len:])
        game_data["response"] = [llm_tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### relation extraction
        texts = [r for r in game_data["response"]]
        print("texts:", texts[0])
        print(len(texts))
        logits, labels = extract_output(relation_classifier, texts, task_list)
        print(len(labels), len(logits))
        rewards = label_logit_to_reward(logits, task_list, labels)
        print(sum(rewards)/len(rewards))
        query_tensors = [query.squeeze() for query in query_tensors]

        #### Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        for cs in ctrl_str:
            key = "env/reward_" + cs.strip("[]")
            stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
        ppo_trainer.log_stats(stats, game_data, rewards)

# for ctrl_s in ctrl_str:
#     plt.hist(
#         [r for r, t in zip(logs["env/reward_dist"], task_list) if t == ctrl_s], density=True, alpha=0.5, label=ctrl_s
#     )
#     plt.legend(loc="best")
#     plt.title("reward distribution")
#     plt.grid(True)
#     plt.show()

MODEL_NAME = f"dare_{args.llm}_{args.dataset}_byrelation_finetuning_with_rl"
MODELS_PATH_NAME = MODELS_PATH / MODEL_NAME
MODELS_PATH_NAME.mkdir(parents=True, exist_ok=True)

llm_model.save_pretrained(MODELS_PATH_NAME)
llm_tokenizer.save_pretrained(MODELS_PATH_NAME)
