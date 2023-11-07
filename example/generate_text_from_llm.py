import opennre
import logging
import argparse
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from ast import literal_eval
from datasets import load_dataset
from transformers import pipeline, set_seed

nltk.download('punkt')

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
        " ".join(
        x["text"]["token"][:x["text"]["h"]["pos"][0]] + ["drug_a"] + \
        x["text"]["token"][x["text"]["h"]["pos"][1]:x["text"]["t"]["pos"][0]] + \
        ["drug_b"] + x["text"]["token"][x["text"]["t"]["pos"][1]:]),
    "label": x["label"]
    }
)

labels = list(set(dataset['train']['label']))

for relation in labels:
    print(relation)
    dataset_label = dataset.filter(lambda x: x["label"] == relation)
    generator = pipeline('text-generation', model=f'igorvln/dare_gpt2_ddi_train_{relation}_finetuning')
    synthetic_texts = []

    for _ in tqdm(range(len(dataset_label["train"]))):
        while True:
            generated_text = generator("<s>", max_length=101, min_length=9, num_return_sequences=1, pad_token_id=50256)[0]       
            first_sentence = sent_tokenize(generated_text["generated_text"])[0][4:]
            tokenized_sentence = word_tokenize(first_sentence)
            if len(tokenized_sentence) >= 8 and 'drug_a' in tokenized_sentence and 'drug_b' in tokenized_sentence:
                synthetic_texts.append(tokenized_sentence)
                break
            else:
                continue

    with open(f"benchmark/{args.dataset}/synt_{args.dataset}_{relation}_train.txt", "w") as f:
        for text in synthetic_texts:
            drug_a_idx = text.index('drug_a')
            drug_b_idx = text.index('drug_b')
            obj = {'token': text, 'h': {'name': 'drug_a', 'pos': [drug_a_idx, drug_a_idx+1]}, 't': {'name': 'drug_b', 'pos': [drug_b_idx, drug_b_idx+1]}, 'relation': relation}
            f.write(str(obj) + "\n")