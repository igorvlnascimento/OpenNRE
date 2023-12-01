import opennre
import logging
import argparse
import nltk
import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from ast import literal_eval
from datasets import load_dataset
from transformers import pipeline, set_seed
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead

nltk.download('punkt')

parser = argparse.ArgumentParser()

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

# Dataset
parser.add_argument('--dataset', default="none", type=str,
        help='Dataset')

# LLM
parser.add_argument('--llm', default="gpt2", type=str,
        help='LLM')

parser.add_argument('--rl', action='store_true', 
        help='Use RL model')

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
ddataset = dataset.map(lambda x: {
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = f'igorvln/dare_{args.llm}_{args.dataset}_byrelation_finetuning'
if args.rl:
    model_name = f'igorvln/dare_{args.llm}_{args.dataset}_byrelation_finetuning_with_rl'
llm_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)

generation_kwargs = {
    "min_length": 11,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": llm_tokenizer.eos_token_id,
    "max_new_tokens": 103,
    "eos_token_id": -1,
}

labels = list(set(dataset['train']['label']))
synthetic_texts = []
relations = []
for relation in labels:
    print(relation)
    dataset_label = dataset.filter(lambda x: x["label"] == relation)
    #generator = pipeline('text-generation', model=f'igorvln/dare_{args.llm}_{args.dataset}_byrelation_finetuning')

    for _ in tqdm(range(len(dataset_label["train"]))):
        while True:
            generated_text = llm_model.generate(llm_tokenizer.encode(f"[{relation}] ", return_tensors="pt").to(device), **generation_kwargs)#generator(f"[{relation}] ", max_length=103, min_length=11, num_return_sequences=1, pad_token_id=50256)[0]["generated_text"]
            response_str = llm_tokenizer.decode(generated_text[0])
            first_sentence = sent_tokenize(response_str)[0][response_str.index(']')+2:]
            tokenized_sentence = first_sentence.split()
            if len(tokenized_sentence) >= 14 and \
                '<SUB>' in tokenized_sentence and \
                '</SUB>' in tokenized_sentence and \
                '<OBJ>' in tokenized_sentence and \
                '</OBJ>' in tokenized_sentence:
                synthetic_texts.append(tokenized_sentence)
                relations.append(relation)
                break
            else:
                continue

filename_path = f"benchmark/{args.dataset}/synt_{args.dataset}_train.txt"
if args.rl:
    filename_path = f"benchmark/{args.dataset}/rl_synt_{args.dataset}_train.txt"
with open(filename_path, "w") as f:
    for i, text in enumerate(synthetic_texts):
        entity_head_start_idx = text.index('<SUB>')
        entity_head_end_idx = text.index('</SUB>') - 1
        text.remove('<SUB>')
        text.remove('</SUB>')
        entity_tail_start_idx = text.index('<OBJ>')
        entity_tail_end_idx = text.index('</OBJ>') - 1
        text.remove('<OBJ>')
        text.remove('</OBJ>')
        obj = {'token': text, 
                'h': {'name': text[entity_head_start_idx:entity_head_end_idx], 'pos': [entity_head_start_idx, entity_head_end_idx]}, 
                't': {'name': text[entity_tail_start_idx:entity_tail_end_idx], 'pos': [entity_tail_start_idx, entity_tail_end_idx]}, 
                'relation': relations[i]}
        f.write(str(obj) + "\n")
