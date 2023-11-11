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

# Dataset
parser.add_argument('--dataset', default="none", type=str,
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
ddataset = dataset.map(lambda x: {
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

labels = list(set(dataset['train']['label']))

for relation in labels:
    print(relation)
    dataset_label = dataset.filter(lambda x: x["label"] == relation)
    generator = pipeline('text-generation', model=f'igorvln/dare_gpt2_{args.dataset}_{relation}_finetuning')
    synthetic_texts = []

    for _ in tqdm(range(len(dataset_label["train"]))):
        while True:
            generated_text = generator("<s>", max_length=105, min_length=11, num_return_sequences=1, pad_token_id=50256)[0]       
            first_sentence = sent_tokenize(generated_text["generated_text"])[0][4:]
            tokenized_sentence = word_tokenize(first_sentence)
            if len(tokenized_sentence) >= 10 and \
                '<SUB>' in tokenized_sentence and \
                '</SUB>' in tokenized_sentence and \
                '<OBJ>' in tokenized_sentence and \
                '</OBJ>' in tokenized_sentence:
                synthetic_texts.append(tokenized_sentence)
                break
            else:
                continue

    with open(f"benchmark/{args.dataset}/synt_{args.dataset}_{relation}_train.txt", "w") as f:
        for text in synthetic_texts:
            entity_head_start_idx = text.index('<SUB>')
            entity_head_end_idx = text.index('</SUB>') - 1
            text.remove('<SUB>')
            text.remove('</SUB>')
            entity_tail_start_idx = text.index('<OBJ>')
            entity_tail_end_idx = text.index('</OBJ>')
            text.remove('<OBJ>')
            text.remove('</OBJ>')
            obj = {'token': text, 
                   'h': {'name': text[entity_head_start_idx:entity_head_end_idx], 'pos': [entity_head_start_idx, entity_head_end_idx]}, 
                   't': {'name': text[entity_tail_start_idx:entity_tail_end_idx], 'pos': [entity_tail_start_idx, entity_tail_end_idx]}, 
                   'relation': relation}
            f.write(str(obj) + "\n")